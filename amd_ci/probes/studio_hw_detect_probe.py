#!/usr/bin/env python3
"""Probe: what does this checkout's Studio hardware detection conclude on THIS host?

Observes only; criteria/studio_hw_detect_same.py (Linux ROCm) and
criteria/studio_hw_detect_same_windows.py judge.

For one checkout, every CELL runs in its own fresh interpreter (module globals,
the inventory TTL cache and torch's lazy CUDA init must not leak between cells):

  unmasked                              the host as the runner presents it
  hip0                                  HIP_VISIBLE_DEVICES=0
  inject_xpu_is_available_raises        torch.xpu.is_available raises
  inject_cuda_get_device_properties_raises
                                        torch.cuda.get_device_properties raises
  inject_cuda_is_available_raises_after_first
                                        the first torch.cuda.is_available call
                                        answers normally, every later call raises

Injections are applied inside the child, after importing the checkout's module and
before detect_hardware(), and are RESTORED once detect_hardware() returns, so the
rest of the observation (inventory, counts, utilization) is the real host.

Per cell it records: detect_hardware() return, IS_ROCM, CHAT_ONLY,
CHAT_ONLY_REASON, CHAT_ONLY_DETAIL, CHAT_ONLY_MISMATCH_VENDORS, the
"Hardware detected" banner the module printed (it carries the device name the
module resolved), raw torch facts taken BEFORE any injection, the physical GPU
inventory, `_devices_that_can_establish_a_mismatch(inventory devices)`, the
physical / visible GPU counts, backend-visible GPU info, visible GPU utilization
and the GPU summary. Every getter is looked up by name and absent ones are
recorded as absent, so the same probe serves base and head.

Writes JSON to --out, never stdout (import banners corrupt stdout). Text I/O
names its encoding (Windows defaults to cp1252).
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path

CELLS: dict[str, dict] = {
    "unmasked": {"env": {}, "inject": None},
    "hip0": {"env": {"HIP_VISIBLE_DEVICES": "0"}, "inject": None},
    "inject_xpu_is_available_raises": {"env": {}, "inject": "xpu_is_available"},
    "inject_cuda_get_device_properties_raises": {"env": {}, "inject": "cuda_get_device_properties"},
    "inject_cuda_is_available_raises_after_first": {"env": {}, "inject": "cuda_is_available_after_first"},
}

ENV_KEYS = ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES",
            "ZE_AFFINITY_MASK", "UNSLOTH_FORCE_XPU", "AMD_CI_SPOOFED_DEVICES",
            "HSA_OVERRIDE_GFX_VERSION", "LD_PRELOAD")

GETTERS = ("get_physical_gpu_count", "get_visible_gpu_count", "get_backend_visible_gpu_info",
           "get_visible_gpu_utilization", "get_gpu_summary", "get_parent_visible_gpu_ids")


def _jsonable(v, depth: int = 0):
    if depth > 8:
        return repr(v)[:200]
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, dict):
        return {str(k): _jsonable(x, depth + 1) for k, x in v.items()}
    if isinstance(v, (list, tuple, set, frozenset)):
        seq = sorted(v, key = repr) if isinstance(v, (set, frozenset)) else v
        return [_jsonable(x, depth + 1) for x in seq]
    if hasattr(v, "value") and hasattr(v, "name"):  # Enum
        return _jsonable(v.value, depth + 1)
    return repr(v)[:300]


def _err(e: BaseException) -> str:
    return f"{type(e).__name__}: {e}"[:500]


def _raiser(what: str):
    def _raise(*a, **k):
        raise RuntimeError(f"amd_ci injected failure: {what}")
    return _raise


def raw_torch() -> dict:
    """Facts straight from torch, taken before any injection."""
    out: dict = {}
    try:
        import torch
    except Exception as e:  # noqa: BLE001
        return {"import_error": _err(e)}
    out["version"] = torch.__version__
    out["hip"] = getattr(torch.version, "hip", None)
    out["cuda"] = getattr(torch.version, "cuda", None)
    out["has_xpu_module"] = hasattr(torch, "xpu")
    try:
        out["cuda_is_available"] = bool(torch.cuda.is_available())
        out["device_count"] = torch.cuda.device_count()
        names = []
        for i in range(out["device_count"]):
            p = torch.cuda.get_device_properties(i)
            names.append({"name": p.name, "arch": getattr(p, "gcnArchName", None),
                          "total_gib": round(p.total_memory / 1024 ** 3, 2)})
        out["devices"] = names
    except Exception as e:  # noqa: BLE001
        out["cuda_error"] = _err(e)
    try:
        out["xpu_is_available"] = bool(torch.xpu.is_available()) if hasattr(torch, "xpu") else None
    except Exception as e:  # noqa: BLE001
        out["xpu_error"] = _err(e)
    return out


def apply_injection(kind: str | None) -> tuple[dict, list]:
    """Returns (record, restore list of (obj, attr, original))."""
    rec: dict = {"kind": kind, "applied": False}
    restore: list = []
    if not kind:
        return rec, restore
    import torch
    if kind == "xpu_is_available":
        if not hasattr(torch, "xpu"):
            rec["why_not"] = "torch has no xpu module"
            return rec, restore
        restore.append((torch.xpu, "is_available", torch.xpu.is_available))
        torch.xpu.is_available = _raiser("torch.xpu.is_available")
    elif kind == "cuda_get_device_properties":
        restore.append((torch.cuda, "get_device_properties", torch.cuda.get_device_properties))
        torch.cuda.get_device_properties = _raiser("torch.cuda.get_device_properties")
    elif kind == "cuda_is_available_after_first":
        real = torch.cuda.is_available
        calls = {"n": 0}

        def flaky(*a, **k):
            calls["n"] += 1
            if calls["n"] > 1:
                raise RuntimeError("amd_ci injected failure: torch.cuda.is_available (call > 1)")
            return real(*a, **k)

        rec["_calls"] = calls
        restore.append((torch.cuda, "is_available", real))
        torch.cuda.is_available = flaky
    else:
        rec["why_not"] = f"unknown injection {kind!r}"
        return rec, restore
    rec["applied"] = True
    return rec, restore


def import_hardware(checkout: Path):
    backend = checkout / "studio" / "backend"
    if not backend.is_dir():
        raise SystemExit(f"no backend at {backend}")
    sys.path.insert(0, str(backend))
    import utils.hardware.hardware as hw  # noqa: PLC0415
    return hw


def run_cell(checkout: Path, cell: str, out: Path) -> int:
    spec = CELLS[cell]
    obs: dict = {"cell": cell, "inject": spec["inject"],
                 "env": {k: os.environ.get(k) for k in ENV_KEYS},
                 "platform": {"system": platform.system(), "release": platform.release(),
                              "machine": platform.machine(), "python": sys.version.split()[0]}}
    obs["raw_torch"] = raw_torch()
    try:
        hw = import_hardware(checkout)
        obs["hardware_file"] = hw.__file__
    except BaseException as e:  # noqa: BLE001
        obs["import_error"] = _err(e)
        obs["import_traceback"] = traceback.format_exc()[-3000:]
        out.write_text(json.dumps(_jsonable(obs), indent = 2), encoding = "utf-8")
        return 0

    inj, restore = ({"kind": spec["inject"], "applied": False}, [])
    try:
        inj, restore = apply_injection(spec["inject"])
    except Exception as e:  # noqa: BLE001
        inj["error"] = _err(e)

    buf = io.StringIO()
    det: dict = {}
    t0 = time.time()
    try:
        with contextlib.redirect_stdout(buf):
            dev = hw.detect_hardware()
        det["device"] = _jsonable(dev)
    except BaseException as e:  # noqa: BLE001
        det["raised"] = _err(e)
        det["traceback"] = traceback.format_exc()[-3000:]
    finally:
        for obj, attr, orig in restore:
            setattr(obj, attr, orig)
    det["seconds"] = round(time.time() - t0, 2)
    for g in ("DEVICE", "IS_ROCM", "CHAT_ONLY", "CHAT_ONLY_REASON", "CHAT_ONLY_DETAIL",
              "CHAT_ONLY_MISMATCH_VENDORS", "TORCH_IMPORT_ERROR"):
        det[g] = _jsonable(getattr(hw, g, "<absent>"))
    printed = buf.getvalue()
    det["stdout"] = printed[-2000:]
    det["banner"] = [ln.strip() for ln in printed.splitlines() if "Hardware detected" in ln]
    if "_calls" in inj:
        inj["is_available_calls"] = inj.pop("_calls")["n"]
    obs["injection"] = inj
    obs["detect"] = det

    # Everything below runs with the injection removed: the real host.
    try:
        inv = hw.get_physical_gpu_inventory()
        obs["inventory"] = _jsonable(inv)
    except Exception as e:  # noqa: BLE001
        inv = None
        obs["inventory_error"] = _err(e)
    fn = getattr(hw, "_devices_that_can_establish_a_mismatch", None)
    if fn is None:
        obs["mismatch_devices_error"] = "absent"
    else:
        try:
            obs["mismatch_devices"] = _jsonable(fn(list((inv or {}).get("devices") or [])))
        except Exception as e:  # noqa: BLE001
            obs["mismatch_devices_error"] = _err(e)
    getters: dict = {}
    for name in GETTERS:
        f = getattr(hw, name, None)
        if f is None:
            getters[name] = {"absent": True}
            continue
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                getters[name] = {"value": _jsonable(f())}
        except Exception as e:  # noqa: BLE001
            getters[name] = {"error": _err(e)}
    obs["getters"] = getters
    out.write_text(json.dumps(_jsonable(obs), indent = 2), encoding = "utf-8")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True, type = Path)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--cell", default = None, help = "internal: run ONE cell in this process")
    ap.add_argument("--cells", default = ",".join(CELLS))
    ap.add_argument("--cell-timeout", type = int, default = 600)
    args = ap.parse_args()

    if args.cell:
        return run_cell(args.checkout.resolve(), args.cell, args.out)

    obs: dict = {"state": args.state, "checkout": str(args.checkout), "cells": {}}
    for cell in [c for c in args.cells.split(",") if c]:
        if cell not in CELLS:
            obs["cells"][cell] = {"error": "unknown cell"}
            continue
        cell_out = args.out.with_name(f"{args.out.stem}_{cell}.json")
        cell_log = args.out.with_name(f"{args.out.stem}_{cell}.log")
        env = dict(os.environ)
        env.update(CELLS[cell]["env"])
        cmd = [sys.executable, str(Path(__file__).resolve()), "--state", args.state,
               "--checkout", str(args.checkout), "--out", str(cell_out), "--cell", cell]
        rec: dict = {}
        try:
            with open(cell_log, "wb") as fh:
                p = subprocess.run(cmd, env = env, stdout = fh, stderr = subprocess.STDOUT,
                                   timeout = args.cell_timeout)
            rec["_rc"] = p.returncode
        except subprocess.TimeoutExpired:
            rec["_rc"] = None
            rec["_timeout"] = args.cell_timeout
        if cell_out.is_file():
            try:
                rec.update(json.loads(cell_out.read_text(encoding = "utf-8")))
            except Exception as e:  # noqa: BLE001
                rec["_parse_error"] = _err(e)
        else:
            rec["_missing_output"] = True
            rec["_log_tail"] = cell_log.read_text(encoding = "utf-8", errors = "replace")[-3000:]
        obs["cells"][cell] = rec
    args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
