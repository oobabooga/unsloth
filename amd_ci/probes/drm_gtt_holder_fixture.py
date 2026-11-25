#!/usr/bin/env python3
"""Fixture: hold GTT through amdgpu GEM buffers (what a Vulkan app allocates), invisible to HIP's free figure."""

from __future__ import annotations

import argparse
import fcntl
import glob
import json
import os
import signal
import struct
import sys
import time

GIB = 1024**3
DRM_IOCTL_AMDGPU_GEM_CREATE = 0xC0206440  # _IOWR('d', 0x40, union drm_amdgpu_gem_create /* 32 B */)
AMDGPU_GEM_DOMAIN_GTT = 0x2


def gtt_used_all() -> int:
    total = 0
    for p in glob.glob("/sys/class/drm/card*/device/mem_info_gtt_used"):
        try:
            with open(os.path.join(os.path.dirname(p), "vendor")) as f:
                if int(f.read().strip(), 16) != 0x1002:
                    continue
            with open(p) as f:
                total += int(f.read().strip())
        except (OSError, ValueError):
            pass
    return total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gib", type = float, default = 6.0)
    ap.add_argument("--max-seconds", type = int, default = 1800)
    args = ap.parse_args()
    nodes = sorted(glob.glob("/dev/dri/renderD*"))
    if not nodes:
        print(json.dumps({"status": "ERROR", "error": "no render node"}), flush = True)
        return 2
    before = gtt_used_all()
    fd = os.open(nodes[0], os.O_RDWR)
    step, held = 256 * 1024 * 1024, 0
    try:
        while held < int(args.gib * GIB):
            buf = bytearray(struct.pack("QQQQ", step, 4096, AMDGPU_GEM_DOMAIN_GTT, 0))
            fcntl.ioctl(fd, DRM_IOCTL_AMDGPU_GEM_CREATE, buf, True)
            held += step
    except OSError as e:
        print(json.dumps({"status": "ERROR", "error": f"GEM_CREATE: {e}", "held_gib": held / GIB}), flush = True)
        return 2
    time.sleep(1)
    after = gtt_used_all()
    print(json.dumps({
        "status": "READY", "pid": os.getpid(), "node": nodes[0],
        "requested_gib": held / GIB,
        "allocated_gib": (after - before) / GIB,
        "gtt_used_before_gib": before / GIB, "gtt_used_after_gib": after / GIB,
    }), flush = True)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    deadline = time.time() + args.max_seconds
    while time.time() < deadline:
        time.sleep(1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
