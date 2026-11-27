#!/usr/bin/env python3
"""Host-memory ballast: commits anonymous memory (every page written) until /proc/meminfo MemAvailable is at most
--target-gib, prints one READY JSON line, then holds it until stdin closes or the process is killed.

Real pressure, not a faked reading: the bytes are resident anonymous pages, so every reader of MemAvailable (the
kernel, Studio's gate) sees the same number. --floor-gib stops it short of driving the host into the OOM killer.

  ballast.py --target-gib 48 [--floor-gib 40]
"""
from __future__ import annotations

import argparse
import ctypes
import json
import sys
import time

GiB = 1 << 30
CHUNK = 256 << 20


def meminfo() -> dict:
    out = {}
    with open("/proc/meminfo", encoding = "ascii") as fh:
        for line in fh:
            name, _, rest = line.partition(":")
            out[name] = int(rest.split()[0]) * 1024
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-gib", type = float, required = True)
    ap.add_argument("--floor-gib", type = float, default = 0.0, help = "never take MemAvailable below this")
    a = ap.parse_args()
    t = time.time()
    m0 = meminfo()
    hold = []
    while True:
        avail = meminfo()["MemAvailable"]
        if avail <= a.target_gib * GiB:
            break
        step = min(CHUNK, avail - int(a.target_gib * GiB))
        if avail - step < a.floor_gib * GiB or step < (16 << 20):
            break
        hold.append(b"\x01" * step)  # bytes(n) filled with a non-zero value: every page is written, so committed
    locked = None
    try:  # best effort: keeps the ballast from being swapped out mid-load (needs RLIMIT_MEMLOCK)
        libc = ctypes.CDLL("libc.so.6", use_errno = True)
        locked = libc.mlockall(1) == 0  # MCL_CURRENT
    except Exception:  # noqa: BLE001
        locked = False
    m1 = meminfo()
    print("READY " + json.dumps({"status": "READY", "held_gib": round(sum(map(len, hold)) / GiB, 2),
                                 "mem_available_before_gib": round(m0["MemAvailable"] / GiB, 2),
                                 "mem_available_gib": round(m1["MemAvailable"] / GiB, 2),
                                 "mem_total_gib": round(m1["MemTotal"] / GiB, 2),
                                 "swap_total_gib": round(m1.get("SwapTotal", 0) / GiB, 2),
                                 "mlocked": locked, "fill_s": round(time.time() - t, 1)}), flush = True)
    try:
        sys.stdin.read()
    except Exception:  # noqa: BLE001
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
