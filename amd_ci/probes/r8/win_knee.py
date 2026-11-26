#!/usr/bin/env python3
"""C6-W1 knee point (r3 ``knee2`` rewritten as a script): hold a ballast of GIB in CHUNK-byte tensors, then time a bf16
4096^3 GEMM (median of 20) and a 1 GiB D2D copy, recording mem_get_info before / after and (Windows) the WDDM GPU
adapter / process memory counters. --opfirst allocates the GEMM operands BEFORE the ballast (C6-W1 "operands before
ballast"). PAL / HIP env arms (GPU_MAX_SUBALLOC_SIZE, PAL_HIP_IPC_FLAG) are set by the caller. Writes one JSON."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time


def counters() -> tuple:
    if sys.platform != "win32":
        return [], ""
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-Command",
                            "(Get-Counter '\\GPU Adapter Memory(*)\\Dedicated Usage','\\GPU Adapter Memory(*)\\Shared Usage',"
                            "'\\GPU Process Memory(pid_" + str(os.getpid()) + "*)\\Dedicated Usage',"
                            "'\\GPU Process Memory(pid_" + str(os.getpid()) + "*)\\Shared Usage' -ErrorAction SilentlyContinue)"
                            ".CounterSamples | Where-Object { $_.CookedValue -gt 0 } | "
                            "ForEach-Object { $_.Path + '=' + [math]::Round($_.CookedValue / 1GB, 3) }"],
                           capture_output = True, text = True, timeout = 120)
        return [ln.split("\\")[-2] + "\\" + ln.split("\\")[-1] for ln in r.stdout.splitlines() if ln.strip()][:20], \
            r.stderr[-300:]
    except Exception as exc:  # noqa: BLE001
        return [], repr(exc)[:300]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required = True)
    ap.add_argument("--gib", type = int, required = True)
    ap.add_argument("--chunk", type = int, default = 2 ** 30)
    ap.add_argument("--opfirst", action = "store_true")
    a = ap.parse_args()
    out: dict = {"ballast_gib_req": a.gib, "chunk_bytes": a.chunk, "opfirst": a.opfirst,
                 "env": {k: os.environ.get(k) for k in ("GPU_MAX_SUBALLOC_SIZE", "PAL_HIP_IPC_FLAG", "PAL_DISABLE_SDMA",
                                                        "PYTORCH_HIP_ALLOC_CONF")}}
    try:
        import torch
        out["total_memory_gib"] = round(torch.cuda.get_device_properties(0).total_memory / 2 ** 30, 3)
        f0, t0 = torch.cuda.mem_get_info()
        out["mem_get_info_before_gib"] = [round(f0 / 2 ** 30, 3), round(t0 / 2 ** 30, 3)]
        x = a_ = c = None
        if a.opfirst:
            x = torch.randn(4096, 4096, device = "cuda", dtype = torch.bfloat16)
            a_ = torch.empty(2 ** 30, dtype = torch.uint8, device = "cuda")
            a_.fill_(2)
            c = torch.empty_like(a_)
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
        out["n_tensors"] = len(ballast)
        f1, t1 = torch.cuda.mem_get_info()
        out["mem_get_info_gib"] = [round(f1 / 2 ** 30, 3), round(t1 / 2 ** 30, 3)]
        out["reserved_gib"] = round(torch.cuda.memory_reserved() / 2 ** 30, 3)
        out["free_drop_gib"] = round((f0 - f1) / 2 ** 30, 3)
        if x is None:
            x = torch.randn(4096, 4096, device = "cuda", dtype = torch.bfloat16)
        for _ in range(5):
            x @ x
        ts = []
        for _ in range(20):
            torch.cuda.synchronize()
            t = time.perf_counter()
            x @ x
            torch.cuda.synchronize()
            ts.append(time.perf_counter() - t)
        ts.sort()
        out["gemm_tflops"] = round(2 * 4096 ** 3 / ts[10] / 1e12, 2)
        if a_ is None:
            a_ = torch.empty(2 ** 30, dtype = torch.uint8, device = "cuda")
            a_.fill_(2)
            c = torch.empty_like(a_)
        for _ in range(3):
            c.copy_(a_)
        ts = []
        for _ in range(10):
            torch.cuda.synchronize()
            t = time.perf_counter()
            c.copy_(a_)
            torch.cuda.synchronize()
            ts.append(time.perf_counter() - t)
        ts.sort()
        out["d2d_gbps"] = round(2 * 2 ** 30 / ts[5] / 1e9, 1)
        out["counters_gib"], out["counters_err"] = counters()
        out["verdict"] = "ok"
    except Exception as exc:  # noqa: BLE001
        import traceback
        out["verdict"] = "error"
        out["error"] = traceback.format_exc()[-2000:]
    with open(a.out, "w", encoding = "utf-8") as fh:
        json.dump(out, fh, indent = 1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
