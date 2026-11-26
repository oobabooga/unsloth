#!/usr/bin/env python3
"""C6-1 knee point with a reserved caching-allocator pool (round-8 ``win_knee.py`` / knee2, extended). Observes only.

Order, one fresh process per arm:
  1. --reserve-gib N > 0: allocate one N GiB uint8 tensor and fill it (the pool reserved before weights would be).
  2. ballast: --gib GiB in --chunk byte tensors, each filled (the 42 GiB of resident weights), while the reserve is
     still HELD, so the ballast cannot be carved out of it.
  3. free the reserve WITHOUT torch.cuda.empty_cache(): the segment stays in the caching allocator.
  4. operands: the bf16 4096^3 GEMM operand, then a 512 MiB D2D copy pair, allocated now (from the cache when the
     reserve exists). Engagement proof: each operand's data_ptr lies inside the freed reserve's [ptr, ptr + bytes).
  5. GEMM TFLOP/s (median of 20) and D2D GB/s (median of 10).
--opfirst (the round-8 33.3 TFLOP/s control): operands allocated before the ballast instead, no reserve.
N = 0 without --opfirst is the base (operands allocated after the ballast). Writes one JSON.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required = True)
    ap.add_argument("--gib", type = int, default = 42)
    ap.add_argument("--chunk", type = int, default = 2 ** 30)
    ap.add_argument("--reserve-gib", type = int, default = 0)
    ap.add_argument("--opfirst", action = "store_true")
    a = ap.parse_args()
    out: dict = {"ballast_gib_req": a.gib, "chunk_bytes": a.chunk, "reserve_gib": a.reserve_gib, "opfirst": a.opfirst,
                 "platform": sys.platform, "env": {k: os.environ.get(k) for k in (
                     "PAL_DISABLE_SDMA", "PYTORCH_HIP_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF")}}
    calls = {"empty_cache": 0}
    try:
        import torch
        real_empty = torch.cuda.empty_cache

        def counted_empty():
            calls["empty_cache"] += 1
            real_empty()
        torch.cuda.empty_cache = counted_empty  # nothing in this script calls it; a count above 0 means something did
        p = torch.cuda.get_device_properties(0)
        out.update(torch = torch.__version__, hip = getattr(torch.version, "hip", None),
                   arch = getattr(p, "gcnArchName", None), total_memory_gib = round(p.total_memory / 2 ** 30, 3))
        f0, _ = torch.cuda.mem_get_info()
        out["free0_gib"] = round(f0 / 2 ** 30, 3)
        GEMM = 4096
        CP = 512 * 2 ** 20
        x = cp_a = cp_c = None
        res_lo = res_hi = None
        if a.opfirst:
            x = torch.randn(GEMM, GEMM, device = "cuda", dtype = torch.bfloat16)
            cp_a = torch.empty(CP, dtype = torch.uint8, device = "cuda")
            cp_a.fill_(2)
            cp_c = torch.empty_like(cp_a)
        reserve = None
        if a.reserve_gib > 0 and not a.opfirst:
            reserve = torch.empty(a.reserve_gib * 2 ** 30, dtype = torch.uint8, device = "cuda")
            reserve.fill_(0)
            res_lo = reserve.data_ptr()
            res_hi = res_lo + reserve.numel()
        ballast = []
        try:
            for _ in range(max(0, a.gib * 2 ** 30 // a.chunk)):
                b = torch.empty(a.chunk, dtype = torch.uint8, device = "cuda")
                b.fill_(1)
                ballast.append(b)
        except Exception as exc:  # noqa: BLE001
            out["alloc_error"] = repr(exc)[:300]
        torch.cuda.synchronize()
        out["ballast_gib"] = round(sum(b.numel() for b in ballast) / 2 ** 30, 3)
        out["ballast_in_reserve"] = sum(1 for b in ballast if res_lo is not None and res_lo <= b.data_ptr() < res_hi)
        if reserve is not None:
            del reserve  # back to the caching allocator, NOT to the driver (no empty_cache)
        out["reserved_after_free_gib"] = round(torch.cuda.memory_reserved() / 2 ** 30, 3)
        out["allocated_after_free_gib"] = round(torch.cuda.memory_allocated() / 2 ** 30, 3)
        if x is None:
            x = torch.randn(GEMM, GEMM, device = "cuda", dtype = torch.bfloat16)
        y = torch.empty_like(x)  # the GEMM output, also from the pool
        if cp_a is None:
            cp_a = torch.empty(CP, dtype = torch.uint8, device = "cuda")
            cp_a.fill_(2)
            cp_c = torch.empty_like(cp_a)

        def inside(t):
            return None if res_lo is None else bool(res_lo <= t.data_ptr() < res_hi)
        out["in_reserve"] = {"gemm_x": inside(x), "gemm_y": inside(y), "copy_a": inside(cp_a), "copy_c": inside(cp_c)}
        f1, _ = torch.cuda.mem_get_info()
        out["free_after_gib"] = round(f1 / 2 ** 30, 3)
        out["reserved_gib"] = round(torch.cuda.memory_reserved() / 2 ** 30, 3)
        for _ in range(5):
            torch.matmul(x, x, out = y)
        ts = []
        for _ in range(20):
            torch.cuda.synchronize()
            t = time.perf_counter()
            torch.matmul(x, x, out = y)
            torch.cuda.synchronize()
            ts.append(time.perf_counter() - t)
        ts.sort()
        out["gemm_tflops"] = round(2 * GEMM ** 3 / ts[10] / 1e12, 2)
        out["gemm_tflops_best"] = round(2 * GEMM ** 3 / ts[0] / 1e12, 2)
        for _ in range(3):
            cp_c.copy_(cp_a)
        ts = []
        for _ in range(10):
            torch.cuda.synchronize()
            t = time.perf_counter()
            cp_c.copy_(cp_a)
            torch.cuda.synchronize()
            ts.append(time.perf_counter() - t)
        ts.sort()
        out["d2d_gbps"] = round(2 * CP / ts[5] / 1e9, 1)
        out["empty_cache_calls"] = calls["empty_cache"]
        out["verdict"] = "ok"
    except Exception:  # noqa: BLE001
        import traceback
        out["verdict"] = "error"
        out["error"] = traceback.format_exc()[-2000:]
    with open(a.out, "w", encoding = "utf-8") as fh:
        json.dump(out, fh, indent = 1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
