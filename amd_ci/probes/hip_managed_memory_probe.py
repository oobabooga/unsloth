#!/usr/bin/env python3
"""Observe what hipMallocManaged and hipMemAdvise actually do on this host.

ggml swaps `cudaMalloc` for `cudaMallocManaged` when GGML_CUDA_ENABLE_UNIFIED_MEMORY
is present, then asks for `hipMemAdviseSetCoarseGrain` and DISCARDS the return
value (`ggml_cuda_device_malloc`, ggml/src/ggml-cuda/ggml-cuda.cu). Managed memory
is fine-grain by default and the coarse-grain hint is what buys back device
caching, so whether that call succeeds decides which of two quite different
allocations llama.cpp ends up running on. Nobody has recorded the answer on any
AMD part.

Measured on a Windows gfx1201 R9700 (ROCm 7.1) on 2026-08-28: hipMallocManaged
succeeds, and hipMemAdvise returns hipErrorInvalidValue for EVERY advice value,
on managed and plain pointers alike. So llama.cpp#20536's premise, that the hint
fails because it "is not applicable to UMA systems", does not hold there. This
probe asks the same questions on Linux gfx1151, which is the host that matters.

ctypes against libamdhip64 on purpose: the runner has no compiler, and adding one
to ask a five-line question is the wrong trade.

This probe OBSERVES. It renders no verdict, and it is not wired to a criteria
module, because there is no base and head here: it is one reading of one host.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
from pathlib import Path

# hipMemoryAdvise, hip_runtime_api.h. The HIP-specific advices start at 100.
ADVICE = {
    1: "SetReadMostly", 2: "UnsetReadMostly",
    3: "SetPreferredLocation", 4: "UnsetPreferredLocation",
    5: "SetAccessedBy", 6: "UnsetAccessedBy",
    100: "SetCoarseGrain", 101: "UnsetCoarseGrain",
}
ATTR_COHERENCY_MODE = 100
COHERENCY = {0: "fine-grain", 1: "coarse-grain", 2: "indeterminate"}
HIP_MEM_ATTACH_GLOBAL = 1
HIP_MEMCPY_DEVICE_TO_HOST = 2


def load_hip(lib_dir: Path | None):
    names = ["libamdhip64.so", "libamdhip64.so.7", "libamdhip64.so.6"]
    if sys.platform == "win32":
        names = ["amdhip64_7.dll", "amdhip64_6.dll", "amdhip64.dll"]
    tried = []
    roots = [lib_dir] if lib_dir else []
    roots += [None, Path("/opt/rocm/lib")]
    for root in roots:
        for n in names:
            cand = str(root / n) if root else n
            tried.append(cand)
            try:
                return ctypes.CDLL(cand), cand
            except OSError:
                continue
    return None, tried


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib-dir", type = Path, default = None,
                    help = "where libamdhip64 lives; the llama.cpp ROCm bundle ships one")
    ap.add_argument("--mib", type = int, default = 512)
    ap.add_argument("--out", type = Path, required = True)
    args = ap.parse_args()

    obs: dict = {"platform": sys.platform, "mib": args.mib}
    hip, name = load_hip(args.lib_dir)
    if hip is None:
        obs["error"] = "no HIP runtime found"
        obs["tried"] = name
        args.out.write_text(json.dumps(obs, indent = 2))
        print(json.dumps(obs, indent = 2))
        # Not a failure of the host: report it and let the reader see it.
        return 0
    obs["lib"] = name

    errstr = hip.hipGetErrorString
    errstr.restype = ctypes.c_char_p

    def rc(value):
        return {"rc": int(value), "name": errstr(value).decode("utf-8", "replace")}

    n = ctypes.c_int(0)
    obs["hipGetDeviceCount"] = rc(hip.hipGetDeviceCount(ctypes.byref(n)))
    obs["device_count"] = n.value
    if n.value < 1:
        args.out.write_text(json.dumps(obs, indent = 2))
        print(json.dumps(obs, indent = 2))
        return 0

    buf = ctypes.create_string_buffer(256)
    hip.hipDeviceGetName(buf, 256, 0)
    obs["device_name"] = buf.value.decode("utf-8", "replace")

    free_b, total_b = ctypes.c_size_t(0), ctypes.c_size_t(0)
    obs["hipMemGetInfo"] = rc(hip.hipMemGetInfo(ctypes.byref(free_b), ctypes.byref(total_b)))
    obs["mem_free_gib"] = round(free_b.value / 2 ** 30, 2)
    obs["mem_total_gib"] = round(total_b.value / 2 ** 30, 2)

    size = ctypes.c_size_t(args.mib * 1024 * 1024)
    ptr = ctypes.c_void_p()
    obs["hipMallocManaged"] = rc(
        hip.hipMallocManaged(ctypes.byref(ptr), size, ctypes.c_uint(HIP_MEM_ATTACH_GLOBAL))
    )
    obs["managed_ptr_nonnull"] = bool(ptr.value)
    if not ptr.value:
        args.out.write_text(json.dumps(obs, indent = 2))
        print(json.dumps(obs, indent = 2))
        return 0

    def coherency():
        got = ctypes.c_int(-1)
        r = hip.hipMemRangeGetAttribute(
            ctypes.byref(got), ctypes.c_size_t(ctypes.sizeof(got)),
            ctypes.c_int(ATTR_COHERENCY_MODE), ptr, size,
        )
        return {"query": rc(r), "mode": COHERENCY.get(got.value, f"unreadable({got.value})")}

    obs["coherency_as_allocated"] = coherency()
    # The exact call ggml makes and throws away.
    obs["hipMemAdvise_SetCoarseGrain"] = rc(
        hip.hipMemAdvise(ptr, size, ctypes.c_int(100), ctypes.c_int(0))
    )
    obs["coherency_after_advise"] = coherency()
    # Every other advice, to tell "this hint is unsupported" apart from "the API
    # is unimplemented here". On Windows it was the latter.
    obs["hipMemAdvise_all"] = {
        label: rc(hip.hipMemAdvise(ptr, size, ctypes.c_int(code), ctypes.c_int(0)))
        for code, label in sorted(ADVICE.items())
    }

    # Does a plain device allocation answer differently? If not, the advise
    # failure says nothing about managed memory specifically.
    plain = ctypes.c_void_p()
    obs["hipMalloc"] = rc(hip.hipMalloc(ctypes.byref(plain), size))
    if plain.value:
        obs["hipMemAdvise_SetCoarseGrain_on_hipMalloc"] = rc(
            hip.hipMemAdvise(plain, size, ctypes.c_int(100), ctypes.c_int(0))
        )
        hip.hipFree(plain)

    # Host write, device-side read back. A managed pointer the device cannot see
    # fails here instead of quietly returning stale bytes.
    view = (ctypes.c_ubyte * 16).from_address(ptr.value)
    expect = [(i * 7 + 3) & 0xFF for i in range(16)]
    for i, v in enumerate(expect):
        view[i] = v
    back = (ctypes.c_ubyte * 16)()
    obs["hipMemcpy_D2H"] = rc(
        hip.hipMemcpy(back, ptr, ctypes.c_size_t(16), ctypes.c_int(HIP_MEMCPY_DEVICE_TO_HOST))
    )
    obs["roundtrip_ok"] = list(back) == expect

    hip.hipFree(ptr)

    args.out.parent.mkdir(parents = True, exist_ok = True)
    args.out.write_text(json.dumps(obs, indent = 2))
    print(json.dumps(obs, indent = 2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
