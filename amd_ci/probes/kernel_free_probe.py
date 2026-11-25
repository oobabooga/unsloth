#!/usr/bin/env python3
"""Probe: HIP free vs this checkout's trusted_mem_get_info vs the kernel GTT/VRAM counters. Observes only."""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

GIB = 1024**3


def _int(path):
    try:
        with open(path) as f:
            return int(f.read().strip(), 0)
    except (OSError, ValueError):
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True, type = Path)
    ap.add_argument("--out", required = True, type = Path)
    args = ap.parse_args()
    obs: dict = {"state": args.state}
    try:
        backend = args.checkout / "studio" / "backend"
        sys.path.insert(0, str(backend))
        import torch
        import utils.hardware.hardware as hw

        obs["hardware_file"] = hw.__file__
        free_b, total_b = torch.cuda.mem_get_info()
        props = torch.cuda.get_device_properties(0)
        obs["hip_free_gib"] = free_b / GIB
        obs["hip_total_gib"] = total_b / GIB
        obs["props_total_gib"] = props.total_memory / GIB
        obs["arch"] = getattr(props, "gcnArchName", None)
        t_free, t_total = hw.trusted_mem_get_info()
        obs["trusted_free_gib"] = t_free / GIB
        obs["trusted_total_gib"] = t_total / GIB
        k = hw.rocm_linux_kernel_free_mib()
        obs["kernel_free_gib"] = None if k is None else k / 1024
        obs["observer_allocated_gib"] = torch.cuda.memory_allocated() / GIB
    except Exception as e:  # noqa: BLE001
        obs["error"] = f"{type(e).__name__}: {e}"
    cards = []
    for d in sorted(glob.glob("/sys/class/drm/card*/device")):
        if _int(os.path.join(d, "vendor")) != 0x1002:
            continue
        cards.append({k: _int(os.path.join(d, f"mem_info_{k}")) for k in
                      ("vram_total", "vram_used", "gtt_total", "gtt_used")})
    obs["sysfs"] = cards
    gtt_used = sum(c["gtt_used"] or 0 for c in cards)
    gtt_total = min((c["gtt_total"] for c in cards if c["gtt_total"]), default = None)
    obs["sysfs_gtt_free_gib"] = None if gtt_total is None else (gtt_total - gtt_used) / GIB
    args.out.write_text(json.dumps(obs, indent = 1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
