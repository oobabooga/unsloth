#!/usr/bin/env python3
"""Probe: what does this checkout's GPU summary report, and what does the driver say?

Observes only. Reports both figures side by side and leaves the comparison to a
criteria module, because "is the gap acceptable" depends on what the fixture is
doing and is not the probe's business.

Pairs with probes/vram_holder_fixture.py and criteria/gpu_summary_sees_others.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

GIB = 1024 ** 3


def import_hardware(checkout: Path):
    """Import THIS checkout's hardware module, the way the app does."""
    backend = checkout / "studio" / "backend"
    if not backend.is_dir():
        raise SystemExit(f"no backend at {backend}")
    sys.path.insert(0, str(backend))
    for stale in [m for m in sys.modules if m.startswith("utils.")]:
        del sys.modules[stale]
    import utils.hardware.hardware as hw  # noqa: PLC0415
    return hw


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True, type = Path)
    ap.add_argument("--out", required = True, type = Path)
    args = ap.parse_args()

    obs: dict = {"state": args.state}
    try:
        import torch
        free_b, total_b = torch.cuda.mem_get_info()
        props = torch.cuda.get_device_properties(0)
        obs["raw_driver_free_gib"] = free_b / GIB
        obs["raw_driver_total_gib"] = total_b / GIB
        obs["props_total_gib"] = props.total_memory / GIB
        obs["is_integrated"] = getattr(props, "is_integrated", None)
        obs["arch"] = getattr(props, "gcnArchName", None)
        # The observer must be a bystander or the comparison is self-observation.
        obs["observer_allocated_gib"] = torch.cuda.memory_allocated() / GIB
        obs["observer_reserved_gib"] = torch.cuda.memory_reserved() / GIB
    except Exception as e:  # noqa: BLE001
        obs["torch_error"] = f"{type(e).__name__}: {e}"

    try:
        hw = import_hardware(args.checkout)
        obs["hardware_file"] = hw.__file__
        obs["summary"] = hw.get_gpu_summary()
        obs["memory_info"] = hw.get_gpu_memory_info()
    except Exception as e:  # noqa: BLE001
        obs["summary_error"] = f"{type(e).__name__}: {e}"

    args.out.write_text(json.dumps(obs, indent = 2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
