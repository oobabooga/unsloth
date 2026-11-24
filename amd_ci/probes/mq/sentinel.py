#!/usr/bin/env python3
"""Warm-up sentinel: a fixed GPU workload timed in a fresh process (bf16 GEMM 4096^3 and a Wan-VAE-like bf16 Conv3d),
median of 5 x 20 calls each. The Windows boxes have run 2-4x slow for the first 20-25 minutes of some jobs; timed
arms start only once this reads steady, and every timed arm is bracketed by a reading. Writes JSON to --out."""
import json
import statistics
import sys
import time


def ms(fn, torch, iters = 20, reps = 5):
    fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(reps):
        t = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        out.append((time.perf_counter() - t) * 1000 / iters)
    return round(statistics.median(out), 3), [round(x, 3) for x in out]


def main():
    out = sys.argv[sys.argv.index("--out") + 1]
    import torch

    t0 = time.time()
    a = torch.randn(4096, 4096, device = "cuda", dtype = torch.bfloat16)
    b = torch.randn(4096, 4096, device = "cuda", dtype = torch.bfloat16)
    conv = torch.nn.Conv3d(128, 128, 3, padding = 1).to("cuda", torch.bfloat16)
    x = torch.randn(1, 128, 5, 88, 160, device = "cuda", dtype = torch.bfloat16)
    with torch.no_grad():
        g, gr = ms(lambda: a @ b, torch)
        c, cr = ms(lambda: conv(x), torch, iters = 10)
    rec = {"t": time.time(), "gemm_ms": g, "gemm_reps": gr, "conv_ms": c, "conv_reps": cr,
           "wall_s": round(time.time() - t0, 2)}
    with open(out, "w", encoding = "utf-8") as fh:
        json.dump(rec, fh)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
