"""NF4 (bitsandbytes 4-bit) kernels: unsloth/kernels/nf4.py (_nf4_dequant_kernel) and nf4_gemv.py
(_gemv_nf4_kernel / _gemv_nf4_words_kernel), reached through kernels/utils.py:
  fast_dequantize      -> dequantize_nf4 (Triton) when _USE_NF4_KERNELS else _fast_dequantize_ctypes (bnb CUDA)
  fast_gemv (M=1)      -> gemv_nf4 / planned rerun (Triton) when _TRITON_GEMV_EAGER else _fast_gemv_ctypes (bnb)
  fast_linear_forward  -> gemv_nf4_rows for 2 <= bsz <= _gemv_rows_cap(arch), else dequant + cuBLAS
  matmul_lora (train / prefill) -> fast_dequantize(use_global_buffer) + torch.matmul
Every bnb-4bit model (Llama / Qwen / Mistral / Gemma ...), QLoRA training (dequant) and 4-bit inference (gemv)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import *

# (N, K) = (out_features, in_features) of Llama-3-8B / Qwen3-8B linears
SHAPES = dict(q=(4096, 4096), kv=(1024, 4096), up=(14336, 4096), down=(4096, 14336), qwen_up=(12288, 4096))

CASES = (
    [dict(name="dequant_" + k, kind="dequant", N=n, K=kk) for k, (n, kk) in SHAPES.items()]
    + [dict(name="gemv_M1_" + k, kind="gemv", N=n, K=kk) for k, (n, kk) in SHAPES.items()]
    + [dict(name="rows_M%d_%s" % (m, k), kind="rows", M=m, N=n, K=kk)
       for m in (2, 4, 8) for k, (n, kk) in (("q", SHAPES["q"]), ("up", SHAPES["up"]), ("down", SHAPES["down"]))]
    + [dict(name="prefill_M4096_" + k, kind="prefill", M=4096, N=n, K=kk)
       for k, (n, kk) in (("q", SHAPES["q"]), ("up", SHAPES["up"]), ("down", SHAPES["down"]))]
    + [dict(name="tiny_gemv_M1_256x256", kind="gemv", N=256, K=256, host=True),
       dict(name="tiny_dequant_256x256", kind="dequant", N=256, K=256, host=True)]
    + [dict(name="sweep_dequant_up", kind="sweep_dequant", N=14336, K=4096),
       dict(name="sweep_gemv_up", kind="sweep_gemv", N=14336, K=4096),
       dict(name="sweep_gemv_q", kind="sweep_gemv", N=4096, K=4096)]
)

_MODS = {}


def mods():
    if not _MODS:
        import_zoo("unsloth_zoo.utils", stub=("unsloth_zoo", "unsloth_zoo.temporary_patches"))
        _MODS["u"] = import_unsloth("unsloth.kernels.utils")
        _MODS["nf4"] = import_unsloth("unsloth.kernels.nf4")
        _MODS["gemv"] = import_unsloth("unsloth.kernels.nf4_gemv")
        import bitsandbytes as bnb
        _MODS["bnb"] = bnb
    return _MODS


_WCACHE = {}


def make_weight(N, K, dtype):
    key = (N, K, dtype)
    if key not in _WCACHE:
        import torch
        bnb = mods()["bnb"]
        Wd = torch.randn(N, K, device="cuda", dtype=dtype) * 0.02
        Wq, qs = bnb.functional.quantize_4bit(Wd, blocksize=64, compress_statistics=True, quant_type="nf4",
                                              quant_storage=torch.uint8)
        W_deq = bnb.functional.dequantize_4bit(Wq, qs)  # golden dense weight
        _WCACHE.clear()
        _WCACHE[key] = (Wq, qs, W_deq)
    return _WCACHE[key]


def torch_nf4_dequant(Wq, absmax_u8, code2, absmax2, offset, code, bs, bs2, N, K, dtype):
    """Plain-torch NF4 dequant (nested absmax), the math bnb's kernel computes."""
    import torch
    a = (code2[absmax_u8.int()].view(-1, bs2) * absmax2[:, None]).reshape(-1)[: N * K // bs] + offset
    q = Wq.view(-1)
    idx = torch.stack(((q >> 4), (q & 15)), -1).view(-1).int()
    vals = code[idx].view(-1, bs) * a[:, None]
    return vals.to(dtype).view(N, K)


def run(case, dtype, H):
    import torch
    m = mods()
    u, nf4, gm, bnb = m["u"], m["nf4"], m["gemv"], m["bnb"]
    dt = H.dtype
    N, K = case["N"], case["K"]
    Wq, qs, W_deq = make_weight(N, K, dt)
    absmax, shape, qdt, bs, code, code2, absmax2, offset, bs2 = u._unpack_quant_state(qs)
    H.emit(kind="info", use_nf4_kernels=u._USE_NF4_KERNELS, triton_gemv_eager=u._TRITON_GEMV_EAGER,
           gemv_rows_cap=u._gemv_rows_cap(torch.device("cuda", torch.cuda.current_device())),
           nf4_cfg=str(nf4._config_for((N * K + 1) // 2, bs // 2, Wq.device)),
           gemv_cfg=str(gm._gemv_config(N, K, bs, torch.cuda.get_device_capability(0), True, None)))
    shp = [N, K]
    kind = case["kind"]
    host = case.get("host", False)

    if kind == "dequant":
        out = torch.empty(N, K, device="cuda", dtype=dt)
        tdq = lambda: torch_nf4_dequant(Wq, absmax, code2, absmax2, offset, code, bs, bs2, N, K, dt)
        cdq = torch.compile(tdq, dynamic=False)
        H.interleave(dict(
            ours=lambda: u.fast_dequantize(Wq, qs),                         # Triton dequantize_nf4 (fresh out)
            ours_gbuf=lambda: u.fast_dequantize(Wq, qs, use_global_buffer=True),  # planned rerun into scratch
            ours_outarg=lambda: u.fast_dequantize(Wq, qs, out=out),              # caller-provided out
            ctypes_bnb=lambda: u._fast_dequantize_ctypes(Wq, qs, out, False),     # unsloth's bnb ctypes path
            bnb=lambda: bnb.functional.dequantize_4bit(Wq, qs),
            compiled=cdq,
            eager=tdq,
        ), "fwd", shape=shp, host=host, budget_ms=150)
        o = u.fast_dequantize(Wq, qs)
        r32 = torch_nf4_dequant(Wq, absmax, code2, absmax2, offset, code, bs, bs2, N, K, torch.float32)
        H.parity("dequant_vs_fp32", o, r32, W_deq, shape=shp)
        H.emit(kind="exact", what="dequant_vs_bnb", mismatches=int((o != W_deq).sum().item()), numel=o.numel())
        if not host:
            H.compile_check("fast_dequantize_matmul", lambda x: torch.matmul(x, u.fast_dequantize(Wq, qs).t()),
                            [(torch.randn(64, K, device="cuda", dtype=dt),), (torch.randn(128, K, device="cuda", dtype=dt),)])

    elif kind == "gemv":
        X = torch.randn(1, 1, K, device="cuda", dtype=dt)
        Wt = W_deq.t()
        H.interleave(dict(
            ours=lambda: u.fast_gemv(X, Wq, qs),                 # path the product takes (planned rerun)
            triton_direct=lambda: gm.gemv_nf4(X, Wq, absmax, code2, absmax2, offset, code, bs, bs2, shape, dt),
            ctypes_bnb=lambda: u._fast_gemv_ctypes(X, Wq, qs),  # bnb cgemm_4bit via ctypes
            bnb=lambda: bnb.matmul_4bit(X, Wq.t(), quant_state=qs),
            cublas_dense=lambda: torch.matmul(X, Wt),           # speed-of-light: 16-bit weight already dense
            dequant_matmul=lambda: torch.matmul(X, u.fast_dequantize(Wq.t(), qs, use_global_buffer=True)),
        ), "fwd", shape=[1] + shp, host=host, budget_ms=150)
        o = u.fast_gemv(X, Wq, qs)
        r64 = (X.double() @ W_deq.double().t())
        H.parity("gemv_vs_fp64", o, r64, bnb.matmul_4bit(X, Wq.t(), quant_state=qs), shape=[1] + shp)
        if not host:
            H.compile_check("fast_gemv", lambda x: u.fast_gemv(x, Wq, qs),
                            [(X,), (torch.randn(1, 1, K, device="cuda", dtype=dt),)])

    elif kind == "rows":
        M = case["M"]
        X = torch.randn(M, 1, K, device="cuda", dtype=dt)
        out = torch.empty(M, 1, N, device="cuda", dtype=dt)
        Wt = W_deq.t()
        X2 = X.view(M, K)
        H.interleave(dict(
            ours_rows=lambda: gm.gemv_nf4_rows(X, Wq, absmax, code2, absmax2, offset, code, bs, bs2, shape, out),
            dequant_matmul=lambda: torch.matmul(X, u.fast_dequantize(Wq.t(), qs, use_global_buffer=True)),
            bnb=lambda: bnb.matmul_4bit(X2, Wq.t(), quant_state=qs),
            cublas_dense=lambda: torch.matmul(X2, Wt),
        ), "fwd", shape=[M] + shp, budget_ms=150)
        o = gm.gemv_nf4_rows(X, Wq, absmax, code2, absmax2, offset, code, bs, bs2, shape, out).clone()
        H.parity("rows_vs_fp64", o.view(M, N), (X2.double() @ W_deq.double().t()), torch.matmul(X2, Wt), shape=[M] + shp)

    elif kind == "prefill":
        M = case["M"]
        X = torch.randn(1, M, K, device="cuda", dtype=dt)
        dY = torch.randn(1, M, N, device="cuda", dtype=dt)
        H.interleave(dict(
            ours=lambda: u.matmul_lora(X, Wq, qs, None, None, None),
            bnb=lambda: torch.matmul(X, bnb.functional.dequantize_4bit(Wq, qs).t()),
            cublas_dense=lambda: torch.matmul(X, W_deq.t()),
        ), "fwd", shape=[M] + shp, budget_ms=150)
        # QLoRA backward dX = dY @ dequant(W): fast_lora calls fast_dequantize(W.t()) then matmul.
        H.interleave(dict(
            ours=lambda: torch.matmul(dY, u.fast_dequantize(Wq.t(), qs, use_global_buffer=True).t()),
            bnb=lambda: torch.matmul(dY, bnb.functional.dequantize_4bit(Wq, qs)),
            cublas_dense=lambda: torch.matmul(dY, W_deq),
        ), "bwd", shape=[M] + shp, budget_ms=150)
        o = u.matmul_lora(X, Wq, qs, None, None, None)
        H.parity("prefill_vs_fp32", o, X.float() @ W_deq.float().t(), torch.matmul(X, W_deq.t()), shape=[M] + shp)

    elif kind == "sweep_dequant":
        f = lambda: nf4.dequantize_nf4(Wq, absmax, code2, absmax2, offset, code, bs, bs2, shape, dt)
        cfgs = [(t, w, wo, True, l) for t in (1024, 2048, 4096) for w in (2, 4) for wo in (False, True)
                for l in ((0, 2) if nf4._HAS_TL_GATHER else (0,))]
        sweep(H, "nf4_dequant", shp, f, cfgs, lambda c: setattr(nf4, "_CONFIG_OVERRIDE", c),
              str(nf4._config_for((N * K + 1) // 2, bs // 2, Wq.device)))
        nf4._CONFIG_OVERRIDE = None

    elif kind == "sweep_gemv":
        X = torch.randn(1, 1, K, device="cuda", dtype=dt)
        f = lambda: gm.gemv_nf4(X, Wq, absmax, code2, absmax2, offset, code, bs, bs2, shape, dt)
        sweep(H, "nf4_gemv_kernel_choice", shp, f, ["bytes", "words"], lambda c: setattr(gm, "_FORCE_KERNEL", c),
              str(gm._gemv_config(N, K, bs, torch.cuda.get_device_capability(0), True, None)))
        gm._FORCE_KERNEL = None


def sweep(H, what, shp, f, cfgs, apply, default_cfg):
    """Each candidate is timed between two default measurements (ABA); ratio = cand / mean(adjacent defaults)."""
    apply(None)
    d_prev = H.measure(f, budget_ms=30)[0]
    res = []
    for c in cfgs:
        apply(c)
        try:
            t = H.measure(f, budget_ms=30)[0]
        except Exception as ex:
            res.append(dict(cfg=str(c), err=repr(ex)[:80]))
            apply(None)
            continue
        apply(None)
        d = H.measure(f, budget_ms=30)[0]
        res.append(dict(cfg=str(c), ms=t, ratio=t / ((d + d_prev) / 2)))
        d_prev = d
    ok = sorted((r for r in res if "ratio" in r), key=lambda r: r["ratio"])
    H.emit(kind="config_sweep", what=what, shape=shp, default_cfg=default_cfg, default_ms=d_prev,
           best=ok[:5], worst=ok[-1:] if ok else None, n=len(res),
           best_over_default=ok[0]["ratio"] if ok else None, errors=[r for r in res if "err" in r][:3])


if __name__ == "__main__":
    main(CASES, run)
