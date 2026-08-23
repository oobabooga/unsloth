#!/usr/bin/env python3
"""Measure whether the GPU summary can see VRAM held by a DIFFERENT process.

That is the whole substance of #9362. The pre-fix figure is
``total - torch.memory_allocated``, which is process-local by construction, so a
model resident in another process is invisible to it. Mocks cannot show this:
they can only assert that a stubbed number is passed through. Two real
processes on a real GPU can.

Sub-commands:
  holder    allocate and hold N GiB, print READY, wait to be killed
  observe   read one checkout's get_gpu_summary() plus the raw driver figures
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

GIB = 1024 ** 3


def _import_hardware(repo_root: Path):
    """Import a specific checkout's hardware module, the way the app does."""
    backend = repo_root / "studio" / "backend"
    if not backend.is_dir():
        raise SystemExit(f"no backend at {backend}")
    # Front of the path so this checkout wins over any other copy.
    sys.path.insert(0, str(backend))
    for stale in [m for m in sys.modules if m.startswith("utils.")]:
        del sys.modules[stale]
    import utils.hardware.hardware as hw  # noqa: PLC0415
    return hw


def cmd_holder(args: argparse.Namespace) -> int:
    import torch
    if not torch.cuda.is_available():
        print(json.dumps({"error": "no cuda/hip device"}), flush = True)
        return 2
    chunks = []
    want = int(args.gib * GIB)
    held = 0
    # 256 MiB at a time, and TOUCHED: an untouched allocation may not be
    # committed, and an uncommitted allocation is not what a resident model is.
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


def cmd_observe(args: argparse.Namespace) -> int:
    out: dict = {"label": args.label, "repo_root": str(args.repo_root)}
    try:
        import torch
        free_b, total_b = torch.cuda.mem_get_info()
        props = torch.cuda.get_device_properties(0)
        out["raw_driver_free_gib"] = free_b / GIB
        out["raw_driver_total_gib"] = total_b / GIB
        out["props_total_gib"] = props.total_memory / GIB
        out["is_integrated"] = getattr(props, "is_integrated", None)
        out["arch"] = getattr(props, "gcnArchName", None)
        # This process must be a bystander, or the comparison means nothing.
        out["observer_allocated_gib"] = torch.cuda.memory_allocated() / GIB
        out["observer_reserved_gib"] = torch.cuda.memory_reserved() / GIB
    except Exception as e:  # noqa: BLE001
        out["torch_error"] = f"{type(e).__name__}: {e}"

    try:
        hw = _import_hardware(args.repo_root)
        out["hardware_file"] = hw.__file__
        out["summary"] = hw.get_gpu_summary()
        out["memory_info"] = hw.get_gpu_memory_info()
    except Exception as e:  # noqa: BLE001
        out["summary_error"] = f"{type(e).__name__}: {e}"

    if args.out:
        Path(args.out).write_text(json.dumps(out, indent = 2))
    print(json.dumps(out, indent = 2), flush = True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest = "cmd", required = True)

    h = sub.add_parser("holder")
    h.add_argument("--gib", type = float, default = 4.0)
    h.add_argument("--max-seconds", type = int, default = 600)
    h.set_defaults(fn = cmd_holder)

    o = sub.add_parser("observe")
    o.add_argument("--repo-root", required = True, type = Path)
    o.add_argument("--label", required = True)
    o.add_argument("--out")
    o.set_defaults(fn = cmd_observe)

    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
