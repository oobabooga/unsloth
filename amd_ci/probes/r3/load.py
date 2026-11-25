#!/usr/bin/env python3
"""R3-08: cold load of a safetensors checkpoint onto the GPU, default mmap vs backend="read_file" (safetensors #760).

  load.py --dir D --arm mmap|read_file --out F

One fresh process per arm. Before loading, every shard is fsync'd and dropped from the page cache with
posix_fadvise(DONTNEED) (no root needed for clean pages); the resident fraction is then measured with mincore, so
"cold" is observed, not assumed. Each tensor: safe_open(device="cuda:0").get_tensor -> bf16 on the GPU, kept (as a
casting load keeps it). Wall time, peak RSS (ru_maxrss, VmHWM) and a checksum of every loaded tensor."""
from __future__ import annotations

import argparse
import ctypes
import glob
import json
import mmap
import os
import resource
import sys
import time


def drop_and_residency(files):
    libc = ctypes.CDLL("libc.so.6", use_errno = True)
    libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
    page = os.sysconf("SC_PAGE_SIZE")
    total = res = 0
    for f in files:
        fd = os.open(f, os.O_RDONLY)
        try:
            os.fsync(fd)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            size = os.fstat(fd).st_size
            mm = mmap.mmap(fd, size, access = mmap.ACCESS_COPY)  # writable view for ctypes; pages untouched
            n = (size + page - 1) // page
            vec = (ctypes.c_ubyte * n)()
            view = (ctypes.c_char * 1).from_buffer(mm)
            addr = ctypes.addressof(view)
            if libc.mincore(ctypes.c_void_p(addr), ctypes.c_size_t(size), vec) == 0:
                res += sum(1 for b in vec if b & 1)
            total += n
            del view
            mm.close()
        finally:
            os.close(fd)
    return res / max(total, 1)


def vmhwm_gib():
    for line in open("/proc/self/status", encoding = "utf-8"):
        if line.startswith("VmHWM:"):
            return int(line.split()[1]) / 2 ** 20
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required = True)
    ap.add_argument("--arm", required = True)
    ap.add_argument("--out", required = True)
    a = ap.parse_args()
    import safetensors
    import torch
    from safetensors import safe_open

    files = sorted(glob.glob(os.path.join(a.dir, "*.safetensors")))
    obs = {"arm": a.arm, "safetensors": safetensors.__version__, "torch": torch.__version__, "files": len(files),
           "bytes": sum(os.path.getsize(f) for f in files)}
    try:
        torch.cuda.init()
        torch.empty(1, device = "cuda")
        obs["resident_before"] = drop_and_residency(files)
        obs["rss0_gib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20
        kw = {} if a.arm == "mmap" else {"backend": a.arm}
        keep = []
        t = time.perf_counter()
        for f in files:
            with safe_open(f, framework = "pt", device = "cuda:0", **kw) as fh:
                for k in fh.keys():
                    x = fh.get_tensor(k)
                    keep.append(x.to(torch.bfloat16) if x.is_floating_point() else x)
                    del x
        torch.cuda.synchronize()
        obs["load_s"] = time.perf_counter() - t
        obs["peak_rss_gib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20
        obs["vmhwm_gib"] = vmhwm_gib()
        obs["tensors"] = len(keep)
        obs["gpu_gib"] = torch.cuda.memory_allocated() / 2 ** 30
        acc = torch.zeros((), dtype = torch.float64, device = "cuda")
        for i, x in enumerate(keep):
            acc += x.float().abs().sum().double() * (1 + (i % 97) / 97.0)
        obs["checksum"] = float(acc)
    except Exception as exc:  # noqa: BLE001
        import traceback
        obs["error"] = repr(exc)[:400]
        obs["tb"] = traceback.format_exc()[-1500:]
    open(a.out, "w", encoding = "utf-8").write(json.dumps(obs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
