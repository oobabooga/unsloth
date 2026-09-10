#!/usr/bin/env python3
"""Probe: what does this checkout's GPU summary report, and what does the driver say?

Observes only. Reports both figures side by side and leaves the comparison to a
criteria module, because "is the gap acceptable" depends on what the fixture is
doing and is not the probe's business.

Pairs with probes/vram_holder_fixture.py and criteria/gpu_summary_sees_others.py.

On an amdgpu host it also records the kernel's own view of the pool: the TTM page
ceiling and the driver's GTT / VRAM totals and used counters. Those are the
quantities the reported "total" could actually be (a BIOS carve-out, a TTM
ceiling, or the figure ROCm/ROCm#5595 reports out of neither), and nothing in
`mem_get_info` says which. Observation only: the probe records them and the
criteria module decides what, if anything, they mean.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

GIB = 1024 ** 3
PAGE = 4096

# The kernel counters an amdgpu card publishes, in bytes, plus the module-wide
# TTM ceiling in PAGES. Read verbatim; no arithmetic beyond a GiB conversion,
# because deciding which of these the driver's "total" is IS the question.
SYSFS_CARD_FILES = (
    "mem_info_gtt_total",
    "mem_info_gtt_used",
    "mem_info_vram_total",
    "mem_info_vram_used",
)
TTM_PAGES_LIMIT = "/sys/module/ttm/parameters/pages_limit"


def _read_int(path: str) -> int | None:
    """One unsigned integer out of a sysfs file, or None if it cannot be read."""
    try:
        return int(Path(path).read_text(encoding = "utf-8").strip())
    except Exception:  # noqa: BLE001
        return None


def kernel_memory_view() -> dict:
    """amdgpu's own accounting, per card, plus the TTM page ceiling.

    Absent everywhere that is not an amdgpu Linux host, which is a fact worth
    recording rather than an error: the Windows half of the pool has no sysfs at
    all, and a report that silently omitted these would look identical to one
    where they read zero.
    """
    out: dict = {}
    pages = _read_int(TTM_PAGES_LIMIT)
    out["ttm_pages_limit"] = pages
    out["ttm_pages_limit_gib"] = (pages * PAGE / GIB) if pages else None

    cards: dict = {}
    for device in sorted(glob.glob("/sys/class/drm/card*/device")):
        card = device.rsplit("/", 2)[-2]
        row: dict = {}
        for name in SYSFS_CARD_FILES:
            value = _read_int(f"{device}/{name}")
            if value is None:
                continue
            row[name] = value
            row[f"{name}_gib"] = value / GIB
        if row:
            cards[card] = row
    out["cards"] = cards
    return out


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
        obs["torch_version"] = torch.__version__
        obs["torch_hip"] = getattr(torch.version, "hip", None)
        # total - free is what the driver says is RESIDENT right now, which is the
        # figure the whole "is free a budget" question turns on.
        obs["driver_used_gib"] = (total_b - free_b) / GIB
    except Exception as e:  # noqa: BLE001
        obs["torch_error"] = f"{type(e).__name__}: {e}"

    # Independent of torch on purpose: if the torch block above threw, the kernel's
    # own numbers are still worth having, and they are the only ones that are not
    # mediated by HIP.
    obs["kernel"] = kernel_memory_view()

    try:
        hw = import_hardware(args.checkout)
        obs["hardware_file"] = hw.__file__
        obs["summary"] = hw.get_gpu_summary()
        obs["memory_info"] = hw.get_gpu_memory_info()
    except Exception as e:  # noqa: BLE001
        obs["summary_error"] = f"{type(e).__name__}: {e}"

    args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
