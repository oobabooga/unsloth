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


def meminfo_gib(key):
    for line in open("/proc/meminfo", encoding = "utf-8"):
        if line.startswith(key + ":"):
            return int(line.split()[1]) / 2 ** 20
    return None


def status_gib(key):
    for line in open("/proc/self/status", encoding = "utf-8"):
        if line.startswith(key + ":"):
            return int(line.split()[1]) / 2 ** 20
    return None


def tensor_ranges(path):
    """name -> (absolute byte offset, length) from the safetensors header."""
    import struct
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(n))
    base = 8 + n
    return {k: (base + v["data_offsets"][0], v["data_offsets"][1] - v["data_offsets"][0])
            for k, v in hdr.items() if k != "__metadata__"}


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
        fadv = a.arm.endswith("_fadv")
        backend = a.arm[: -len("_fadv")] if fadv else a.arm
        kw = {} if backend == "mmap" else {"backend": backend}
        import threading
        peak = {"anon": 0.0, "avail_min": None, "stop": False}

        def sample():
            while not peak["stop"]:
                peak["anon"] = max(peak["anon"], status_gib("RssAnon") or 0.0)
                av = meminfo_gib("MemAvailable")
                peak["avail_min"] = av if peak["avail_min"] is None else min(peak["avail_min"], av)
                time.sleep(0.2)
        th = threading.Thread(target = sample, daemon = True)
        obs["cached_before_gib"] = meminfo_gib("Cached")
        obs["avail_before_gib"] = meminfo_gib("MemAvailable")
        th.start()
        keep = []
        n_fadv = 0
        t = time.perf_counter()
        for f in files:
            ranges = tensor_ranges(f) if fadv else {}
            fd = os.open(f, os.O_RDONLY) if fadv else None
            try:
                with safe_open(f, framework = "pt", device = "cuda:0", **kw) as fh:
                    for k in fh.keys():
                        x = fh.get_tensor(k)
                        keep.append(x.to(torch.bfloat16) if x.is_floating_point() else x)
                        del x
                        if fadv:
                            # the bytes are on the device now: drop this tensor's pages from the page cache
                            off, ln = ranges[k]
                            os.posix_fadvise(fd, off, ln, os.POSIX_FADV_DONTNEED)
                            n_fadv += 1
            finally:
                if fd is not None:
                    os.close(fd)
        torch.cuda.synchronize()
        obs["load_s"] = time.perf_counter() - t
        peak["stop"] = True
        th.join(timeout = 2)
        obs["cached_after_gib"] = meminfo_gib("Cached")
        obs["cached_delta_gib"] = obs["cached_after_gib"] - obs["cached_before_gib"]
        obs["rss_anon_peak_gib"] = peak["anon"]
        obs["avail_min_gib"] = peak["avail_min"]
        obs["fadvise_calls"] = n_fadv
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
