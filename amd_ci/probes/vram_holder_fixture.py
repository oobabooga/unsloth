#!/usr/bin/env python3
"""Fixture: hold VRAM in a separate process for the life of a differential.

Used with `differential.py --fixture`. The condition under measurement is
"another process is holding memory", so it has to be true for every state's
observation equally, which a per-state probe cannot arrange.

Prints one JSON line with "status": "READY" once the memory is committed, then
waits to be terminated.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time

GIB = 1024 ** 3


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gib", type = float, default = 4.0)
    ap.add_argument("--max-seconds", type = int, default = 900)
    args = ap.parse_args()

    import torch
    if not torch.cuda.is_available():
        print(json.dumps({"status": "ERROR", "error": "no cuda/hip device"}), flush = True)
        return 2

    chunks = []
    want = int(args.gib * GIB)
    held = 0
    # TOUCHED, not merely allocated: an uncommitted allocation is not what a
    # resident model is, and may not move the driver's free figure at all.
    step = 256 * 1024 * 1024
    while held < want:
        t = torch.empty(step, dtype = torch.uint8, device = "cuda")
        t.fill_(1)
        chunks.append(t)
        held += step
    torch.cuda.synchronize()
    free_b, total_b = torch.cuda.mem_get_info()
    print(json.dumps({
        "status": "READY", "pid": os.getpid(),
        "held_gib": held / GIB,
        "allocated_gib": torch.cuda.memory_allocated() / GIB,
        "reserved_gib": torch.cuda.memory_reserved() / GIB,
        "driver_free_gib": free_b / GIB, "driver_total_gib": total_b / GIB,
    }), flush = True)

    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    deadline = time.time() + args.max_seconds
    while time.time() < deadline:
        time.sleep(1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
