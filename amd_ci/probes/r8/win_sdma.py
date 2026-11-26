#!/usr/bin/env python3
"""C6-W2: Windows PAL SDMA copy-after-Event.record race (ROCm/rocm-systems #12571, fix PR #12952), observed only.

One process, one stream (the default), one host thread. Per iteration (the #12571 sequence, rewritten here):
  1. X (int64, 10,240 words) filled with OLD, synchronize; queue ~behind_ms of matmuls so the device lags the host.
  2. WRITER: torch.argmax writes W into X[:HALF] (its pointer reaches the kernel inside a by-value struct, like every
     PyTorch elementwise / reduction kernel), ~writer_ms on the device.
  3. READER: flat = X + 1.  4. torch.cuda.Event().record() (cells without "-norec"; "-timing" uses enable_timing=True)
  5. read cells:   h = X.cpu()  (blocking D2H into pageable memory, the .cpu() path)
     upload cells: del X; Y = pinned(C).to(cuda, non_blocking=True) (lands on X's freed block)
  6. synchronize, classify: read -> clean / STALE (all OLD) / MIXED / other; upload -> clean / OVERWRITTEN / other.
The environment (PAL_DISABLE_SDMA) is set by the caller per process. Writes one JSON (--out).
"""
from __future__ import annotations

import argparse
import json
import os
import time

NW = 10240
HALF = NW // 2
OLD = 0x5EED5EED5EED5EED
W = 777


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required = True)
    ap.add_argument("--iters", type = int, default = 50)
    ap.add_argument("--cells", default = "read,read-norec,read-timing,upload,upload-norec")
    ap.add_argument("--writer-ms", type = float, default = 10.0)
    ap.add_argument("--behind-ms", type = float, default = 20.0)
    a = ap.parse_args()
    rec: dict = {"env": {k: os.environ.get(k) for k in ("PAL_DISABLE_SDMA", "AMD_SERIALIZE_COPY", "AMD_SERIALIZE_KERNEL",
                                                        "AMD_DIRECT_DISPATCH", "HIP_LAUNCH_BLOCKING")},
                 "cells": {}, "verdict": "running"}

    def save():
        with open(a.out, "w", encoding = "utf-8") as fh:
            json.dump(rec, fh, indent = 1)
    try:
        import numpy as np
        import torch
        dev = torch.device("cuda")
        rec.update(torch = torch.__version__, hip = getattr(torch.version, "hip", None),
                   device = torch.cuda.get_device_name(0),
                   arch = getattr(torch.cuda.get_device_properties(0), "gcnArchName", None))

        def sync():
            torch.cuda.synchronize()

        def host_ms(fn, reps = 5):
            best = float("inf")
            for _ in range(reps):
                sync()
                t0 = time.perf_counter()
                fn()
                sync()
                best = min(best, (time.perf_counter() - t0) * 1e3)
            return best

        kmax = 1 << 25
        row = torch.rand(kmax, device = dev) * 0.5
        row[W] = 1.0

        def writer(xh, k):
            torch.argmax(row[:k].expand(HALF, k), dim = 1, out = xh)

        xh = torch.empty(HALF, dtype = torch.int64, device = dev)
        k1, k2 = 1 << 18, 1 << 20
        writer(xh, k2)
        m1, m2 = host_ms(lambda: writer(xh, k1)), host_ms(lambda: writer(xh, k2))
        if m2 <= m1:
            raise RuntimeError(f"writer calibration failed {m1} {m2}")
        k = int(min(kmax, max(W + 1, a.writer_ms / ((m2 - m1) / (k2 - k1)))))
        wms = host_ms(lambda: writer(xh, k))
        sync()
        if not bool((xh == W).all()):
            raise RuntimeError("writer does not write W")
        mm = torch.randn(2048, 2048, device = dev)

        def chain(x, n):
            for _ in range(n):
                x = torch.tanh(x @ x * 1e-4)
            return x
        chain(mm, 3)
        t_mm = host_ms(lambda: chain(mm, 10), reps = 3) / 10
        n_behind = max(1, round(a.behind_ms / max(t_mm, 1e-3)))
        rec["writer"] = {"k": k, "ms": round(wms, 3)}
        rec["behind"] = {"matmuls": n_behind, "ms_each": round(t_mm, 4)}
        rng = np.random.default_rng(0)
        for cell in a.cells.split(","):
            st = {"iters": 0, "outcomes": {}, "on_block": 0}
            keep = []
            t0 = time.time()
            for _ in range(a.iters):
                x = torch.empty(NW, dtype = torch.int64, device = dev)
                x.fill_(OLD)
                sync()
                chain(mm, n_behind)
                writer(x[:HALF], k)
                flat = x + 1
                if not cell.endswith("-norec"):
                    ev = torch.cuda.Event(enable_timing = cell.endswith("-timing"))
                    ev.record()
                if cell.startswith("upload"):
                    xptr = x.data_ptr()
                    del x
                    c = rng.integers(0, 2 ** 62, NW, dtype = np.int64)
                    y = torch.from_numpy(c).pin_memory().to(dev, non_blocking = True)
                    del flat
                    sync()
                    keep.append(y)
                    st["on_block"] += int(y.data_ptr() == xptr)
                    yy = y.cpu().numpy()
                    h1, h2 = bool((yy[:HALF] == c[:HALF]).all()), bool((yy[HALF:] == c[HALF:]).all())
                    out = "clean" if h1 and h2 else ("OVERWRITTEN" if bool((yy[:HALF] == W).all()) and h2 else "other")
                else:
                    h = x.cpu().numpy()
                    del flat
                    sync()
                    if not bool((h[HALF:] == np.int64(OLD)).all()):
                        out = "other"
                    else:
                        n_w, n_old = int((h[:HALF] == W).sum()), int((h[:HALF] == np.int64(OLD)).sum())
                        out = ("clean" if n_w == HALF else "STALE" if n_old == HALF else
                               "MIXED" if n_w + n_old == HALF else "other")
                st["outcomes"][out] = st["outcomes"].get(out, 0) + 1
                st["iters"] += 1
            sync()
            keep.clear()
            st["wrong"] = st["iters"] - st["outcomes"].get("clean", 0)
            st["seconds"] = round(time.time() - t0, 1)
            rec["cells"][cell] = st
            save()
        rec["verdict"] = "ok"
    except Exception as exc:  # noqa: BLE001
        import traceback
        rec["verdict"] = "error"
        rec["error"] = traceback.format_exc()[-3000:]
    save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
