"""T7h.5g op-level microbench: LoRA base + rank-r term, main's order vs #16071's fused order and alternatives.

argv: OUT_JSON [--quick]
Torch only (no unsloth import). Per model shape set, M in {1024, 4096}, every projection (q/k/v/o/gate/up/down):

forward (Y [M, N] = X [M, K] @ W[N, K].t() + s * (X @ A.t()) @ B.t()):
  a_main      out = X @ W.t(); out.addmm_(XA, B.t(), alpha = s)                       (two roundings)
  b_pr        out = addmm(XA.new_zeros(1), XA, B.t(), beta 0, alpha s); out.addmm_(X, W.t())   (PR eager path)
  b2_pr_empty out = empty; out.addmm_(XA, B.t(), beta 0, alpha s); out.addmm_(X, W.t())
  c_pr_func   lora = addmm(new_zeros(1), ...); out = torch.addmm(lora, X, W.t())       (PR compiled path)
  d_fp32out   torch.mm(X, W.t(), out_dtype = fp32) + s * (XA @ B.t()) in fp32, one cast   (if supported)
  e_kcat      one GEMM: cat([X, s*XA], 1) @ cat([W, B], 1).t()                          (fp32 accumulate of both)
  f_main_bias out = torch.addmm(lora_out, X, W.t()) with lora computed in fp32 then cast  (informational)
backward base dX ([M, K] = dY [M, N] @ W [N, K] + s * dB [M, r] @ A [r, K]):
  a_main  dX = dY @ W; dX.addmm_(dB, A, alpha = s)
  b_pr    dX = addmm(new_zeros(1), dB, A, beta 0, alpha s); dX.addmm_(dY, W)
  e_kcat  cat([dY, s*dB], 1) @ cat([W, A], 0)
isolated GEMMs: big beta 0 (mm), big beta 1 in place (addmm_), big beta 1 functional (addmm), small rank-r beta 0 / 1.

Timing: CUDA events, warmup then ITERS calls per sample, arms interleaved round-robin per repeat, median over REPS.
Accuracy (one shape per projection, M = 1024): mean |Y - Y64| / mean |Y64 - base64| (error relative to the LoRA term's
size) and max abs error, vs an fp64 oracle, realistic magnitudes (X ~ N(0, 1), W ~ N(0, 0.02), A kaiming-uniform,
B ~ N(0, 2e-3) so |LoRA| ~ 1-2% of |base|).
"""
import json
import math
import os
import platform
import statistics
import sys
import time
import traceback

import torch

OUT = sys.argv[1]
QUICK = "--quick" in sys.argv
ITERS = 5 if QUICK else 20
REPS = 3 if QUICK else 7
R = 16
S = 1.0
dev = os.environ.get("T7H5G_DEVICE", "cuda")
dt = torch.bfloat16

MODELS = {
    # name: (hidden, intermediate, q_out, kv_out)
    "Llama-3.2-1B": (2048, 8192, 2048, 512),
    "Llama-3.2-3B": (3072, 8192, 3072, 1024),
    "Qwen3-4B": (2560, 9728, 4096, 1024),
}
if QUICK:
    MODELS = {"Llama-3.2-1B": MODELS["Llama-3.2-1B"]}
MS = [int(os.environ.get("T7H5G_M", "1024"))] if QUICK else [1024, 4096]

rec = {"argv": sys.argv, "host": platform.platform(), "python": sys.version.split()[0], "torch": torch.__version__,
       "hip": getattr(torch.version, "hip", None), "cuda": torch.version.cuda,
       "device": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu", "iters": ITERS, "reps": REPS, "rows": [], "isolated": [],
       "accuracy": [], "errors": []}
try:
    rec["arch"] = getattr(torch.cuda.get_device_properties(0), "gcnArchName", None) if dev == "cuda" else None
except Exception:  # noqa: BLE001
    rec["arch"] = None
for k in ("TORCH_BLAS_PREFER_HIPBLASLT", "HIPBLASLT_TENSILE_LIBPATH", "ROCBLAS_LAYER"):
    rec[k] = os.environ.get(k)
try:
    rec["blas_backend"] = str(torch.backends.cuda.preferred_blas_library())
except Exception as e:  # noqa: BLE001
    rec["blas_backend"] = f"n/a {type(e).__name__}"


def save():
    with open(OUT, "w", encoding = "utf-8") as fh:
        json.dump(rec, fh, indent = 1, default = str)


def log(*a):
    print(*a, flush = True)


def projections(h, inter, q, kv):
    return {"q": (h, q), "k": (h, kv), "v": (h, kv), "o": (q, h), "gate": (h, inter), "up": (h, inter),
            "down": (inter, h)}


HAS_OUT_DTYPE = False
try:
    _a = torch.randn(64, 64, device = dev, dtype = dt)
    torch.mm(_a, _a, out_dtype = torch.float32)
    HAS_OUT_DTYPE = True
except Exception as e:  # noqa: BLE001
    rec["out_dtype_error"] = f"{type(e).__name__}: {e}"[:300]
rec["has_out_dtype"] = HAS_OUT_DTYPE


def fwd_arms(X, W, A, B):
    Bt = B.t()

    def a_main():
        out = X @ W.t()
        out.addmm_(X @ A.t(), Bt, alpha = S)
        return out

    def b_pr():
        XA = X @ A.t()
        out = torch.addmm(XA.new_zeros(1), XA, Bt, beta = 0, alpha = S)
        out.addmm_(X, W.t())
        return out

    def b2_pr_empty():
        XA = X @ A.t()
        out = torch.empty((X.shape[0], W.shape[0]), dtype = X.dtype, device = X.device)
        out.addmm_(XA, Bt, beta = 0, alpha = S)
        out.addmm_(X, W.t())
        return out

    def c_pr_func():
        XA = X @ A.t()
        lo = torch.addmm(XA.new_zeros(1), XA, Bt, beta = 0, alpha = S)
        return torch.addmm(lo, X, W.t())

    def d_fp32out():
        base = torch.mm(X, W.t(), out_dtype = torch.float32)
        XA = X @ A.t()
        base.addmm_(XA.float(), Bt.float(), alpha = S)
        return base.to(X.dtype)

    def e_kcat():
        XA = X @ A.t()
        Xc = torch.cat([X, XA * S], 1)
        Wc = torch.cat([W, B], 1)
        return Xc @ Wc.t()

    def f_lora32_bias():
        XA = X @ A.t()
        lo = torch.mm(XA.float(), Bt.float()).mul_(S).to(X.dtype)
        return torch.addmm(lo, X, W.t())

    arms = {"a_main": a_main, "b_pr": b_pr, "b2_pr_empty": b2_pr_empty, "c_pr_func": c_pr_func, "e_kcat": e_kcat,
            "f_lora32_bias": f_lora32_bias}
    if HAS_OUT_DTYPE:
        arms["d_fp32out"] = d_fp32out
    return arms


def bwd_arms(dY, W, dB, A):
    def a_main():
        dX = dY @ W
        dX.addmm_(dB, A, alpha = S)
        return dX

    def b_pr():
        dX = torch.addmm(dB.new_zeros(1), dB, A, beta = 0, alpha = S)
        dX.addmm_(dY, W)
        return dX

    def e_kcat():
        return torch.cat([dY, dB * S], 1) @ torch.cat([W, A], 0)

    arms = {"a_main": a_main, "b_pr": b_pr, "e_kcat": e_kcat}
    if HAS_OUT_DTYPE:
        def d_fp32out():
            base = torch.mm(dY, W, out_dtype = torch.float32)
            base.addmm_(dB.float(), A.float(), alpha = S)
            return base.to(dY.dtype)

        arms["d_fp32out"] = d_fp32out
    return arms


def time_arms(arms):
    ev = {k: [] for k in arms}
    for f in arms.values():
        for _ in range(3):
            f()
    if dev == "cuda":
        torch.cuda.synchronize()
    names = list(arms)
    for rep in range(REPS):
        order = names if rep % 2 == 0 else names[::-1]
        for k in order:
            f = arms[k]
            if dev == "cpu":
                t0 = time.perf_counter()
                for _ in range(ITERS):
                    f()
                ev[k].append((time.perf_counter() - t0) * 1000 / ITERS)
                continue
            s, e = torch.cuda.Event(enable_timing = True), torch.cuda.Event(enable_timing = True)
            s.record()
            for _ in range(ITERS):
                f()
            e.record()
            e.synchronize()
            ev[k].append(s.elapsed_time(e) / ITERS)
    return {k: {"median_ms": statistics.median(v), "min_ms": min(v), "max_ms": max(v)} for k, v in ev.items()}


def mk(M, K, N, seed = 0):
    g = torch.Generator(device = dev).manual_seed(seed)
    X = torch.randn(M, K, device = dev, generator = g).to(dt)
    W = (torch.randn(N, K, device = dev, generator = g) * 0.02).to(dt)
    A = ((torch.rand(R, K, device = dev, generator = g) * 2 - 1) / math.sqrt(K)).to(dt)
    B = (torch.randn(N, R, device = dev, generator = g) * 2e-3).to(dt)
    return X, W, A, B


t_start = time.time()
save()
try:
    for model, (h, inter, q, kv) in MODELS.items():
        for name, (K, N) in projections(h, inter, q, kv).items():
            for M in MS:
                try:
                    X, W, A, B = mk(M, K, N)
                    fa = time_arms(fwd_arms(X, W, A, B))
                    dY = torch.randn(M, N, device = dev).to(dt) * 1e-2
                    dB = torch.randn(M, R, device = dev).to(dt) * 1e-2
                    ba = time_arms(bwd_arms(dY, W, dB, A))
                    XA = X @ A.t()
                    out = torch.empty(M, N, device = dev, dtype = dt)
                    iso = time_arms({
                        "big_mm_beta0": lambda: X @ W.t(),
                        "big_addmm_inplace_beta1": lambda: out.addmm_(X, W.t()),
                        "big_addmm_func_beta1": lambda: torch.addmm(out, X, W.t()),
                        "big_mm_out_beta0": lambda: torch.mm(X, W.t(), out = out),
                        "small_addmm_inplace_beta1": lambda: out.addmm_(XA, B.t(), alpha = S),
                        "small_addmm_bcast_beta0": lambda: torch.addmm(XA.new_zeros(1), XA, B.t(), beta = 0, alpha = S),
                        "small_addmm_empty_beta0": lambda: torch.empty_like(out).addmm_(XA, B.t(), beta = 0, alpha = S),
                    })
                    row = {"model": model, "proj": name, "M": M, "K": K, "N": N, "fwd": fa, "bwd": ba, "iso": iso}
                    rec["rows"].append(row)
                    fm, bm = fa["a_main"]["median_ms"], ba["a_main"]["median_ms"]
                    log(f"{model} {name} M={M} K={K} N={N} fwd main {fm:.3f} "
                        + " ".join(f"{k}={v['median_ms'] / fm:.3f}" for k, v in fa.items() if k != "a_main")
                        + f" | bwd main {bm:.3f} "
                        + " ".join(f"{k}={v['median_ms'] / bm:.3f}" for k, v in ba.items() if k != "a_main")
                        + " | iso " + " ".join(f"{k}={v['median_ms']:.3f}" for k, v in iso.items()))
                    del X, W, A, B, dY, dB, XA, out
                except Exception as e:  # noqa: BLE001
                    rec["errors"].append(f"{model} {name} {M}: {type(e).__name__}: {e}"[:500])
                    log("ERROR", rec["errors"][-1])
                save()
    # Accuracy vs an fp64 oracle, M = 1024, Llama-1B projections.
    h, inter, q, kv = MODELS["Llama-3.2-1B"]
    for name, (K, N) in projections(h, inter, q, kv).items():
        X, W, A, B = mk(1024, K, N, seed = 7)
        X64, W64, A64, B64 = X.double(), W.double(), A.double(), B.double()
        base64 = X64 @ W64.t()
        lora64 = S * (X64 @ A64.t()) @ B64.t()
        Y64 = base64 + lora64
        lora_mag = lora64.abs().mean().item()
        acc = {"proj": name, "lora_over_base": lora_mag / base64.abs().mean().item(),
               # The unavoidable error: Y64 rounded once to bf16.
               "floor_err_over_lora": (Y64.to(dt).double() - Y64).abs().mean().item() / lora_mag}
        for k, f in fwd_arms(X, W, A, B).items():
            Y = f().double()
            err = (Y - Y64).abs()
            # How much of the LoRA term survives: projection of (Y - base_bf16_rounded) on lora64 is noisy;
            # report the plain error relative to |lora| and relative to the bf16 half-ulp of Y.
            acc[k] = {"mean_abs_err": err.mean().item(), "max_abs_err": err.max().item(),
                      "err_over_lora": err.mean().item() / lora_mag}
        dY = (torch.randn(1024, N, device = dev) * 1e-2).to(dt)
        dB = dY @ B  # consistent rank-r gradient
        dX64 = dY.double() @ W64 + S * dB.double() @ A64
        for k, f in bwd_arms(dY, W, dB, A).items():
            e = (f().double() - dX64).abs()
            acc["bwd_" + k] = {"mean_abs_err": e.mean().item(), "max_abs_err": e.max().item()}
        rec["accuracy"].append(acc)
        log("acc", name, json.dumps({k: (v["err_over_lora"] if isinstance(v, dict) and "err_over_lora" in v else v)
                                     for k, v in acc.items() if not k.startswith("bwd_")}))
        save()
except Exception as e:  # noqa: BLE001
    rec["errors"].append(f"{type(e).__name__}: {e} | {traceback.format_exc()[-1500:]}")
rec["elapsed_s"] = time.time() - t_start
rec["done"] = True
save()
log("DONE", rec["elapsed_s"])
