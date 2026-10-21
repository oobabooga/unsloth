#!/usr/bin/env python3
"""One arm of the NF4 HIP GEMV probe, in its own process. Observes only, writes JSON to --out.

Imports the state's unsloth (PYTHONPATH set by the probe) and records:
  * route: which kernel unsloth.kernels.utils.fast_gemv takes for batch-1 decode
    (calls into _fast_gemv_ctypes = bitsandbytes' GEMV, gemv_nf4_planned / gemv_nf4 = Triton GEMV)
  * parity: fast_gemv output vs an fp64 dequantize + matmul reference, and the same for
    bitsandbytes' ctypes GEMV (the old route) as the baseline error
  * per-call kernel time (CUDA events, as the Triton audit's b_nf4.py measured it) of the
    product route, forced bitsandbytes ctypes, forced Triton, bnb.matmul_4bit
  * decode-loop wall time per token (host launch overhead included) over Llama 3 8B and
    Llama 3.2 1B linear shapes, arms interleaved: product route, forced ctypes, forced Triton
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import traceback

# (N, K) = (out_features, in_features), the audit's b_nf4.py shapes plus a tiny one.
SHAPES = dict(q = (4096, 4096), kv = (1024, 4096), up = (14336, 4096), down = (4096, 14336),
              qwen_up = (12288, 4096), tiny = (256, 256))
# One decoder layer's batch-1 linears: q, k, v, o, gate, up, down.
MODELS = {
    "llama3_8b_x8": (8, 4096, 1024, 14336),
    "llama32_1b_x16": (16, 2048, 512, 8192),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required = True)
    ap.add_argument("--quick", action = "store_true")
    args = ap.parse_args()
    rec: dict = {"stage": "import", "env_UNSLOTH_TRITON_GEMV_EAGER": os.environ.get("UNSLOTH_TRITON_GEMV_EAGER")}

    def dump():
        with open(args.out, "w", encoding = "utf-8") as fh:
            json.dump(rec, fh, indent = 2, default = str)

    try:
        import torch
        import triton
        import bitsandbytes as bnb
        import unsloth.kernels.utils as U
        from unsloth.kernels import nf4_gemv

        props = torch.cuda.get_device_properties(0)
        rec.update(
            unsloth_file = U.__file__,
            torch = torch.__version__, hip = torch.version.hip, triton = triton.__version__,
            bitsandbytes = bnb.__version__, gpu = torch.cuda.get_device_name(0),
            arch = getattr(props, "gcnArchName", None),
            use_nf4_kernels = U._USE_NF4_KERNELS, triton_gemv_eager = U._TRITON_GEMV_EAGER,
            gate_fn = nf4_gemv.triton_gemv_eager(),
        )
        rec["stage"] = "route"
        dump()

        counts = {"ctypes": 0, "triton_planned": 0, "triton_gemv": 0}
        real = {n: getattr(U, n) for n in ("_fast_gemv_ctypes", "gemv_nf4_planned", "gemv_nf4")}
        key = {"_fast_gemv_ctypes": "ctypes", "gemv_nf4_planned": "triton_planned", "gemv_nf4": "triton_gemv"}

        def counting(on):
            for n, f in real.items():
                if on:
                    setattr(U, n, (lambda *a, _f = f, _k = key[n], **k: counts.__setitem__(_k, counts[_k] + 1) or _f(*a, **k)))
                else:
                    setattr(U, n, f)

        dt = torch.bfloat16
        weights = {}

        def weight(N, K, seed = 0):
            k = (N, K, seed)
            if k not in weights:
                g = torch.Generator(device = "cuda").manual_seed(seed)
                Wd = torch.randn(N, K, device = "cuda", dtype = dt, generator = g) * 0.02
                Wq, qs = bnb.functional.quantize_4bit(Wd, blocksize = 64, compress_statistics = True,
                                                      quant_type = "nf4", quant_storage = torch.uint8)
                weights[k] = (Wq, qs)
            return weights[k]

        # ---- route + parity, per shape
        route, parity = {}, {}
        for name, (N, K) in SHAPES.items():
            Wq, qs = weight(N, K)
            X = torch.randn(1, 1, K, device = "cuda", dtype = dt)
            counts.update(ctypes = 0, triton_planned = 0, triton_gemv = 0)
            counting(True)
            o = U.fast_gemv(X, Wq, qs)
            o2 = U.fast_gemv(X, Wq, qs)
            counting(False)
            torch.cuda.synchronize()
            route[name] = dict(counts)
            W_deq = bnb.functional.dequantize_4bit(Wq, qs)
            ref = (X.double().view(1, K) @ W_deq.double().t()).view(1, 1, N)
            rel = lambda a: ((a.double() - ref).abs().max() / ref.abs().max()).item()
            ct = real["_fast_gemv_ctypes"](X, Wq, qs)
            parity[name] = dict(product_rel = rel(o), ctypes_rel = rel(ct), repeat_equal = bool(torch.equal(o, o2)),
                                finite = bool(torch.isfinite(o).all().item()))
        rec.update(route = route, parity = parity, stage = "kernel_time")
        dump()

        # ---- per-call time, CUDA events around each call (the audit's method)
        def measure(fn, n = 200):
            for _ in range(10):
                fn()
            torch.cuda.synchronize()
            s = [torch.cuda.Event(enable_timing = True) for _ in range(n)]
            e = [torch.cuda.Event(enable_timing = True) for _ in range(n)]
            for i in range(n):
                s[i].record(); fn(); e[i].record()
            torch.cuda.synchronize()
            return statistics.median(a.elapsed_time(b) for a, b in zip(s, e))

        def flag(v):
            def f():
                U._TRITON_GEMV_EAGER = v
            return f

        product_flag = U._TRITON_GEMV_EAGER
        kt = {}
        for name, (N, K) in SHAPES.items():
            Wq, qs = weight(N, K)
            absmax, shape, qdt, bs, code, code2, absmax2, offset, bs2 = U._unpack_quant_state(qs)
            X = torch.randn(1, 1, K, device = "cuda", dtype = dt)
            arms = {
                "product": (flag(product_flag), lambda: U.fast_gemv(X, Wq, qs)),
                "ctypes": (flag(product_flag), lambda: real["_fast_gemv_ctypes"](X, Wq, qs)),
                "triton": (flag(product_flag), lambda: nf4_gemv.gemv_nf4(X, Wq, absmax, code2, absmax2, offset, code, bs, bs2, shape, dt)),
                "bnb_matmul_4bit": (flag(product_flag), lambda: bnb.matmul_4bit(X, Wq.t(), quant_state = qs)),
            }
            res = {a: [] for a in arms}
            for r in range(2 if args.quick else 3):
                order = list(arms) if r % 2 == 0 else list(arms)[::-1]
                for a in order:
                    arms[a][0]()
                    res[a].append(measure(arms[a][1]))
            U._TRITON_GEMV_EAGER = product_flag
            kt[name] = {a: dict(median_ms = statistics.median(v), runs = v) for a, v in res.items()}
        rec.update(kernel_ms = kt, stage = "decode")
        dump()

        # ---- decode loop: every linear of L layers once per token, sync at the end
        dec = {}
        for mname, (L, H, KV, I) in MODELS.items():
            shapes = [(H, H), (KV, H), (KV, H), (H, H), (I, H), (I, H), (H, I)]
            layers = [[weight(N, K, seed = 100 + li) for (N, K) in shapes] for li in range(L)]
            xs = {K: torch.randn(1, 1, K, device = "cuda", dtype = dt) for K in (H, I)}
            plan = [(Wq, qs, xs[qs.shape[1]]) for layer in layers for (Wq, qs) in layer]

            def token():
                for Wq, qs, X in plan:
                    U.fast_gemv(X, Wq, qs)

            def run_tokens(T):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(T):
                    token()
                torch.cuda.synchronize()
                return (time.perf_counter() - t0) / T * 1e3

            arms = {"product": product_flag, "ctypes": False, "triton": True}
            res = {a: [] for a in arms}
            T = 10 if args.quick else 30
            for a, v in arms.items():  # warm up every route (Triton JIT) before timing
                U._TRITON_GEMV_EAGER = v
                run_tokens(3)
            for r in range(3 if args.quick else 5):
                order = list(arms) if r % 2 == 0 else list(arms)[::-1]
                for a in order:
                    U._TRITON_GEMV_EAGER = arms[a]
                    res[a].append(run_tokens(T))
            U._TRITON_GEMV_EAGER = product_flag
            dec[mname] = {a: dict(median_ms_per_token = statistics.median(v), runs = v) for a, v in res.items()}
            dec[mname]["linears_per_token"] = len(plan)
            del layers, plan
            weights.clear()
            torch.cuda.empty_cache()
        rec.update(decode = dec, stage = "done", ok = True)
    except Exception as e:  # noqa: BLE001
        rec.update(ok = False, error = f"{type(e).__name__}: {e}", tb = traceback.format_exc()[-3000:])
    dump()
    return 0


if __name__ == "__main__":
    sys.exit(main())
