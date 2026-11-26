#!/usr/bin/env python3
"""C15-2 (C6-W2 extension): with a ballast held, time a 256 MiB torch zero_() (a fill kernel on torch < 2.15) against
the HIP runtime's own hipMemsetAsync / hipMemsetD32Async on the same buffer (ctypes into the HIP DLL torch loaded).
rocm-systems #11692: PAL hipMemsetAsync past the carve-out ran at 0.53 GB/s (fixed by #13002). Writes one JSON."""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import statistics as st
import sys
import time


def hip_lib():
    names = (["amdhip64_7.dll", "amdhip64_6.dll", "amdhip64.dll"] if sys.platform == "win32"
             else ["libamdhip64.so", "libamdhip64.so.7", "libamdhip64.so.6"])
    errs = []
    for n in names:
        try:
            return ctypes.WinDLL(n) if sys.platform == "win32" else ctypes.CDLL(n), n, errs
        except OSError as exc:
            errs.append(f"{n}: {exc}"[:160])
    import torch
    d = os.path.join(os.path.dirname(torch.__file__), "lib")
    for root, _, files in os.walk(os.path.dirname(os.path.dirname(torch.__file__))):
        for f in files:
            if f.lower().startswith(("amdhip64", "libamdhip64")) and (f.endswith(".dll") or ".so" in f):
                p = os.path.join(root, f)
                try:
                    return (ctypes.WinDLL(p) if sys.platform == "win32" else ctypes.CDLL(p)), p, errs
                except OSError as exc:
                    errs.append(f"{p}: {exc}"[:160])
    errs.append(f"not found under {d}")
    return None, None, errs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required = True)
    ap.add_argument("--ballast-gib", type = int, default = 2)
    ap.add_argument("--mib", type = int, default = 256)
    ap.add_argument("--reps", type = int, default = 7)
    a = ap.parse_args()
    rec: dict = {"ballast_gib": a.ballast_gib, "mib": a.mib, "verdict": "running"}
    try:
        import torch
        rec.update(torch = torch.__version__, hip = getattr(torch.version, "hip", None),
                   arch = getattr(torch.cuda.get_device_properties(0), "gcnArchName", None))
        ballast = [torch.ones(2 ** 30, dtype = torch.uint8, device = "cuda") for _ in range(a.ballast_gib)]
        nbytes = a.mib * 2 ** 20
        buf = torch.empty(nbytes, dtype = torch.uint8, device = "cuda")
        buf.fill_(3)
        torch.cuda.synchronize()
        rec["mem_get_info_gib"] = [round(x / 2 ** 30, 3) for x in torch.cuda.mem_get_info()]

        def timed(fn):
            ts = []
            fn()
            torch.cuda.synchronize()
            for _ in range(a.reps):
                torch.cuda.synchronize()
                t = time.perf_counter()
                fn()
                torch.cuda.synchronize()
                ts.append(time.perf_counter() - t)
            med = st.median(ts)
            return {"ms": [round(x * 1e3, 3) for x in ts], "med_ms": round(med * 1e3, 3),
                    "gbps": round(nbytes / med / 1e9, 2)}
        rec["zero_"] = timed(lambda: buf.zero_())
        rec["zero_ok"] = bool((buf == 0).all())
        lib, name, errs = hip_lib()
        rec["hip_lib"], rec["hip_lib_errors"] = name, errs
        if lib is not None:
            stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
            ptr = ctypes.c_void_p(buf.data_ptr())
            f8 = lib.hipMemsetAsync
            f8.restype, f8.argtypes = ctypes.c_int, [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t, ctypes.c_void_p]
            codes = []

            def m8():
                codes.append(f8(ptr, 0, ctypes.c_size_t(nbytes), stream))
            buf.fill_(3)
            rec["hipMemsetAsync"] = timed(m8)
            rec["hipMemsetAsync_rc"] = sorted(set(codes))
            rec["hipMemsetAsync_ok"] = bool((buf == 0).all())
            try:
                f32 = lib.hipMemsetD32Async
                f32.restype = ctypes.c_int
                f32.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t, ctypes.c_void_p]
                codes32 = []

                def m32():
                    codes32.append(f32(ptr, 0, ctypes.c_size_t(nbytes // 4), stream))
                buf.fill_(3)
                rec["hipMemsetD32Async"] = timed(m32)
                rec["hipMemsetD32Async_rc"] = sorted(set(codes32))
                rec["hipMemsetD32Async_ok"] = bool((buf == 0).all())
            except AttributeError as exc:
                rec["hipMemsetD32Async"] = {"error": repr(exc)[:200]}
        del ballast
        rec["verdict"] = "ok"
    except Exception as exc:  # noqa: BLE001
        import traceback
        rec["verdict"] = "error"
        rec["error"] = traceback.format_exc()[-3000:]
    with open(a.out, "w", encoding = "utf-8") as fh:
        json.dump(rec, fh, indent = 1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
