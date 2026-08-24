#!/usr/bin/env python3
"""Probe: the ROCm total-correction path, as this checkout sees it.

Written for PR 9314, which widens an APU's total from props.total_memory (the
dedicated carve-out) to the driver pool. Most of that PR is Windows-only and
this runner is Linux, so this probe deliberately does NOT try to observe the
carve-out correction. It observes the two things a Linux gfx1151 host can
actually answer:

  1. Is the Windows path inert here? _rocm_windows_per_device_vram must return
     ([], None) off Windows. That is the whole cross-platform safety argument and
     it has only ever been checked against a mocked platform.system().

  2. Does the total any caller sees change? _torch_get_device_inventory is ROCm
     gated but NOT platform gated, so its correction is live on this host, and
     PR 9314 changed it to take the wider of props.total_memory and the driver
     total rather than adopting the driver's answer outright.

Observes only. Whether a difference is acceptable is the criteria module's call.
"""

from __future__ import annotations

import argparse
import json
import platform
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


CONTEXT_CHILD = r'''
import json, sys
sys.path.insert(0, sys.argv[1] + "/studio/backend")
out = {}
import utils.hardware.hardware as hw
import torch
# Before anything touches the GPU. A primary HIP context is what costs the
# ~612 MiB, and torch reports whether one exists without creating one.
out["initialized_before"] = bool(torch.cuda.is_initialized())
try:
    devices = hw.get_visible_gpu_utilization().get("devices")
    out["poll_devices"] = devices
except Exception as e:
    out["poll_error"] = f"{type(e).__name__}: {e}"
out["initialized_after"] = bool(torch.cuda.is_initialized())
try:
    out["ordinal_active_after"] = bool(hw._rocm_device_ordinal_active())
except Exception as e:
    out["ordinal_active_error"] = f"{type(e).__name__}: {e}"
print(json.dumps(out))
'''


def context_cost(checkout: Path) -> dict:
    """Does one telemetry poll attach a primary HIP context on this host?

    Run in a FRESH interpreter: this parent has already called mem_get_info to
    read the totals above, so asking the question here would always answer yes.
    Observation only; whether attaching one is acceptable is the criteria's call.
    """
    import subprocess
    try:
        r = subprocess.run(
            [sys.executable, "-c", CONTEXT_CHILD, str(checkout)],
            capture_output = True, text = True, timeout = 600,
        )
        if r.returncode != 0:
            return {"error": f"exit {r.returncode}: {r.stderr[-400:]}"}
        return json.loads(r.stdout.strip().splitlines()[-1])
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True, type = Path)
    ap.add_argument("--out", required = True, type = Path)
    args = ap.parse_args()

    obs: dict = {"state": args.state, "platform": platform.system()}

    n_gpus = 0
    try:
        import torch
        n_gpus = torch.cuda.device_count()
        props = torch.cuda.get_device_properties(0)
        _, driver_total = torch.cuda.mem_get_info(0)
        obs["gpu_count"] = n_gpus
        obs["arch"] = getattr(props, "gcnArchName", None)
        obs["is_integrated"] = getattr(props, "is_integrated", None)
        obs["props_total_gib"] = props.total_memory / GIB
        obs["driver_total_gib"] = driver_total / GIB
        # The premise the whole PR rests on, measured rather than assumed: on
        # Windows these two disagree on an APU. Recording it here is what lets a
        # reader see that this host cannot reproduce the defect.
        obs["driver_exceeds_props"] = bool(driver_total > props.total_memory)
    except Exception as e:  # noqa: BLE001
        obs["torch_error"] = f"{type(e).__name__}: {e}"

    indices = list(range(max(n_gpus, 1)))
    try:
        hw = import_hardware(args.checkout)
        obs["hardware_file"] = hw.__file__
        obs["is_rocm"] = bool(getattr(hw, "IS_ROCM", False))

        # 1. The Windows path, on a real non-Windows host.
        win_devices, win_aggregate = hw._rocm_windows_per_device_vram(indices)
        obs["windows_path_devices"] = win_devices
        obs["windows_path_aggregate"] = win_aggregate
        obs["windows_path_inert"] = (win_devices == [] and win_aggregate is None)

        # 2. The correction that IS live on Linux.
        obs["inventory"] = hw._torch_get_device_inventory(indices)

        try:
            import torch
            props = torch.cuda.get_device_properties(0)
            obs["classifier_says_carve_out"] = bool(hw._rocm_props_total_is_carve_out(props))
        except Exception as e:  # noqa: BLE001
            obs["classifier_error"] = f"{type(e).__name__}: {e}"

        # What the System tab would show.
        vis = hw.get_visible_gpu_utilization()
        obs["visible_devices"] = vis.get("devices")

        # The cost of that poll, measured in a fresh interpreter: a primary HIP
        # context is permanent and ~612 MiB, and the poll path is meant to avoid
        # attaching one on a device it has not classified as unified.
        obs["context_cost"] = context_cost(args.checkout)
    except Exception as e:  # noqa: BLE001
        obs["hardware_error"] = f"{type(e).__name__}: {e}"

    args.out.write_text(json.dumps(obs, indent = 2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
