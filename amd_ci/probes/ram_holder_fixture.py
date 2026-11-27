#!/usr/bin/env python3
"""Fixture (#19768): hold host RAM so Windows-visible available RAM drops to --leave-gib,
making the unified-memory RAM cap bind as on the reporter's 64 GB host. Prints READY, then sleeps."""

from __future__ import annotations

import argparse
import json
import sys
import time

GIB = 1024 ** 3


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--leave-gib", type = float, default = 40.0)
    args = ap.parse_args()
    import subprocess

    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "psutil"], capture_output = True)
    import psutil

    before = psutil.virtual_memory().available
    want = max(0, int(before - args.leave_gib * GIB))
    blocks = []
    held = 0
    while held + GIB <= want:
        b = bytearray(GIB)
        b[::4096] = b"\x01" * (GIB // 4096)
        blocks.append(b)
        held += GIB
    after = psutil.virtual_memory().available
    print(json.dumps({"status": "READY", "held_gib": held / GIB, "avail_before_gib": before / GIB,
                      "avail_after_gib": after / GIB}), flush = True)
    while True:
        time.sleep(30)
        for b in blocks:
            b[0] = 1


if __name__ == "__main__":
    sys.exit(main())
