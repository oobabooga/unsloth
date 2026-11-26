#!/usr/bin/env python3
"""R3-06: Studio's RDNA Triton W8A8 GEMM kernel (``diffusion_native_w8a8_rdna``) at DiT linear shapes, tile configs.

  gemm.py sweep   --backend B --out F                      one process: every candidate per (M, K, N), round-robin
  gemm.py confirm --backend B --out F --table T.json --arm aa|proposed    fresh process; times Studio's tile and the arm interleaved

Every candidate's output is compared to the Studio-table output with torch.equal (int32 accumulate: must be exact).
Writes one JSON file."""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
import time

# (name, K, N): Qwen-Image-2.1 (hidden 4096, mlp_ratio 3), Z-Image (dim 3840, SwiGLU 10240), FLUX.3 klein (3072,
# MLP 9216, fused single-block in 3072 -> 18432)
SHAPES = [("q21_qkv", 4096, 12288), ("q21_out", 4096, 4096), ("q21_down", 12288, 4096),
          ("zi_qkv", 3840, 3840), ("zi_up", 3840, 10240), ("zi_down", 10240, 3840),
          ("kl_o", 3072, 3072), ("kl_fused", 3072, 18432), ("kl_down", 9216, 3072)]
SWEEP_M = [64, 128, 256, 512, 1024, 2048, 4096]
CONFIRM_M = [64, 128, 200, 256, 512, 1024, 2048, 4096, 4176]
# (BM, BN, BK, GROUP_M, warps, stages, waves_per_eu)
CANDS = {
    "st_short": (64, 128, 128, 8, 4, 2, 0), "st_large": (128, 128, 128, 8, 8, 2, 0), "st_small": (64, 64, 64, 8, 4, 2, 0),
    "ai_le64": (16, 32, 256, 1, 2, 2, 1), "ai_le128": (32, 64, 256, 1, 4, 2, 1), "ai_any": (64, 128, 128, 1, 4, 2, 0),
    "ai_le16": (16, 64, 256, 1, 4, 1, 2),
    "bk256_64x128": (64, 128, 256, 8, 4, 1, 0), "bk256_64x64": (64, 64, 256, 8, 4, 2, 0),
    "bk256_32x64": (32, 64, 256, 8, 4, 2, 0), "bk256_32x128": (32, 128, 256, 8, 4, 1, 0),
    "t32x128x128": (32, 128, 128, 8, 4, 2, 0), "t32x64x128": (32, 64, 128, 8, 4, 2, 0),
    "t64x64x128": (64, 64, 128, 8, 4, 2, 0), "st_short_w2": (64, 128, 128, 8, 4, 2, 2),
    "st_large_w2": (128, 128, 128, 8, 8, 2, 2), "t128x64x128": (128, 64, 128, 8, 4, 2, 0),
}


def studio_cfg(R, m, k):
    c = R._mm_cfg(m, k)
    return tuple(c) + (0,)


def launch(R, kern, cfg, xq, wq, xs, ws, bias, out):
    import torch
    bm, bn, bk, gm, warps, stages, wpe = cfg
    m, k = xq.shape
    n = wq.shape[0]
    extra = {"waves_per_eu": wpe} if wpe else {}
    with torch.cuda.device(xq.device):
        kern.w8a8_mm[(kern.triton.cdiv(m, bm) * kern.triton.cdiv(n, bn),)](
            xq, wq, xs, ws, bias, out, m, n, k, HAS_BIAS = True, EVEN_K = k % bk == 0, BLOCK_M = bm, BLOCK_N = bn,
            BLOCK_K = bk, GROUP_M = gm, num_warps = warps, num_stages = stages, enable_fp_fusion = False, **extra)


def inputs(m, k, n, seed):
    import torch
    g = torch.Generator(device = "cuda").manual_seed(seed)
    xq = torch.randint(-127, 128, (m, k), generator = g, device = "cuda", dtype = torch.int8)
    wq = torch.randint(-127, 128, (n, k), generator = g, device = "cuda", dtype = torch.int8)
    xs = torch.rand(m, generator = g, device = "cuda") * 1e-3 + 1e-4
    ws = torch.rand(n, generator = g, device = "cuda") * 1e-3 + 1e-4
    bias = torch.randn(n, generator = g, device = "cuda").to(torch.bfloat16)
    return xq, wq, xs, ws, bias


def time_round(fn, n_calls):
    import torch
    a, b = torch.cuda.Event(enable_timing = True), torch.cuda.Event(enable_timing = True)
    a.record()
    for _ in range(n_calls):
        fn()
    b.record()
    b.synchronize()
    return a.elapsed_time(b) / n_calls


def bench(R, kern, cfgs: dict, m, k, n, rounds, seed = 0):
    """cfgs: name -> cfg tuple. Returns name -> {ms, exact} (or error); 'studio' must be in cfgs (exactness ref)."""
    import torch
    xq, wq, xs, ws, bias = inputs(m, k, n, seed)
    outs, fns, res = {}, {}, {}
    for name, cfg in cfgs.items():
        out = torch.empty((m, n), device = "cuda", dtype = torch.bfloat16)
        try:
            launch(R, kern, cfg, xq, wq, xs, ws, bias, out)
            torch.cuda.synchronize()
            outs[name] = out
            fns[name] = (lambda cfg = cfg, out = out: launch(R, kern, cfg, xq, wq, xs, ws, bias, out))
        except Exception as exc:  # noqa: BLE001
            res[name] = {"error": repr(exc)[:200]}
    ref = outs.get("studio")
    for name, out in outs.items():
        res[name] = {"exact": bool(ref is not None and torch.equal(out, ref)), "ms": []}
    if not fns:
        return res
    est = time_round(fns.get("studio") or next(iter(fns.values())), 3)
    n_calls = int(max(3, min(50, 30.0 / max(est, 1e-3))))
    names = list(fns)
    for r in range(rounds):
        order = names if r % 2 == 0 else names[::-1]
        for name in order:
            res[name]["ms"].append(time_round(fns[name], n_calls))
    for name in fns:
        res[name]["med"] = st.median(res[name]["ms"])
    del xq, wq, outs
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices = ("sweep", "confirm"))
    ap.add_argument("--backend", required = True)
    ap.add_argument("--out", required = True)
    ap.add_argument("--table")
    ap.add_argument("--arm")
    ap.add_argument("--rounds", type = int, default = 9)
    a = ap.parse_args()
    sys.path.insert(0, a.backend)
    import torch
    import triton

    from core.inference import diffusion_native_w8a8_rdna as R
    t0 = time.time()
    dev = torch.device("cuda", 0)
    obs = {"mode": a.mode, "arm": a.arm, "torch": torch.__version__, "triton": triton.__version__,
           "arch": getattr(torch.cuda.get_device_properties(0), "gcnArchName", None), "prepare": R.prepare(dev),
           "studio_cfg": {k: list(v) for k, v in R._CFG.items()}, "rows": []}
    kern = R._kernels()
    if a.mode == "sweep":
        for name, k, n in SHAPES:
            for m in SWEEP_M:
                cfgs = {"studio": studio_cfg(R, m, k), **CANDS}
                obs["rows"].append({"shape": name, "m": m, "k": k, "n": n, "studio": list(cfgs["studio"]),
                                    "res": bench(R, kern, cfgs, m, k, n, a.rounds)})
    else:
        table = json.loads(open(a.table, encoding = "utf-8").read())

        def pick(m, k):
            if a.arm == "aa":
                return studio_cfg(R, m, k)
            cls = "short" if k <= 4096 else "long"
            for ent in table[cls]:  # ascending m_le buckets
                if m <= ent["m_le"]:
                    return tuple(ent["cfg"])
            return tuple(table[cls][-1]["cfg"])
        for name, k, n in SHAPES:
            for m in CONFIRM_M:
                cfg = pick(m, k)
                res = bench(R, kern, {"studio": studio_cfg(R, m, k), "arm": cfg}, m, k, n, a.rounds, seed = 1)
                st_, arm = res.get("studio") or {}, res.get("arm") or {}
                obs["rows"].append({"shape": name, "m": m, "k": k, "n": n, "cfg": list(cfg), "studio_ms": st_.get("med"),
                                    "arm_ms": arm.get("med"), "exact": arm.get("exact"),
                                    "error": arm.get("error") or st_.get("error")})
    obs["seconds"] = round(time.time() - t0, 1)
    open(a.out, "w", encoding = "utf-8").write(json.dumps(obs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
