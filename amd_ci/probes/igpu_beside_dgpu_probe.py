#!/usr/bin/env python3
"""Probe: Windows ROCm per-device VRAM for a real unified iGPU, alone and beside a spoofed discrete card.

Observes only (criteria/igpu_beside_dgpu.py judges). The real part reads this host's WDDM counters,
HIP's LUID for ordinal 0 and the checkout's lone-device result. The spoofed part keeps the REAL iGPU
at ordinal 0 (real props, real LUID, real Dedicated / Shared counters) and adds a phantom discrete
card at ordinal 1 with its own LUID and a fixed counter row, then asks the checkout's
_rocm_windows_per_device_vram([0, 1]) for both. The reference is Dedicated + Shared for the real
LUID, sampled right before and right after, so the reading can be bracketed against counter drift.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
import types
from pathlib import Path

GIB = 1024 ** 3
PHANTOM_LUID = 0xD1E2
PHANTOM_DED = 4.28 * GIB
PHANTOM_SHARED = 0.1 * GIB
PHANTOM = ("AMD Radeon RX 6800", 16 * GIB, "gfx1030")


def import_hardware(checkout: Path):
    backend = checkout / "studio" / "backend"
    sys.path.insert(0, str(backend))
    for stale in [m for m in sys.modules if m.startswith("utils.") or m == "utils" or m.startswith("loggers")]:
        del sys.modules[stale]
    import utils.hardware.hardware as hw  # noqa: PLC0415
    return hw


def _luid_sum(hw, luid, *snapshots):
    total = 0.0
    for snap in snapshots:
        for inst, val in snap or []:
            if hw._parse_adapter_luid(inst) == luid:
                total += val
    return total


def _timed(fn, *a, **k):
    t = time.perf_counter()
    out = fn(*a, **k)
    return out, (time.perf_counter() - t) * 1000.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True, type = Path)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--repeats", type = int, default = 5)
    args = ap.parse_args()
    obs: dict = {"state": args.state, "system": platform.system()}
    try:
        import torch
        obs["torch"] = torch.__version__
        obs["hip"] = getattr(torch.version, "hip", None)
        obs["device_count"] = torch.cuda.device_count()
        p = torch.cuda.get_device_properties(0)
        obs["props"] = {"name": p.name, "total_gib": p.total_memory / GIB,
                        "arch": getattr(p, "gcnArchName", None)}
        hw = import_hardware(args.checkout)
        obs["hardware_file"] = hw.__file__
        # The app runs detect_hardware() at startup (get_device() ensures it); IS_ROCM stays False until then.
        obs["device"] = str(hw.get_device())
        obs["is_rocm"] = bool(hw.IS_ROCM)
        name = p.name

        # ---- real, one device
        ded, obs["ms_dedicated_query"] = _timed(hw._rocm_windows_perf_counter_vram_by_adapter)
        sh, obs["ms_shared_query"] = _timed(hw._rocm_windows_perf_counter_vram_by_adapter, "Shared Usage")
        obs["dedicated"] = ded
        obs["shared"] = sh
        ids = hw._rocm_windows_hip_adapter_ids([0], [name])
        obs["hip_ids"] = ids
        luid = ids[0][0] if ids else None
        obs["luid"] = luid
        obs["positively_unified"] = bool(hw._rocm_props_are_positively_unified(p))
        lone = []
        for _ in range(args.repeats):
            ref0 = _luid_sum(hw, luid, hw._rocm_windows_perf_counter_vram_by_adapter(),
                             hw._rocm_windows_perf_counter_vram_by_adapter("Shared Usage")) if luid else None
            (devs, agg), ms = _timed(hw._rocm_windows_per_device_vram, [0])
            ref1 = _luid_sum(hw, luid, hw._rocm_windows_perf_counter_vram_by_adapter(),
                             hw._rocm_windows_perf_counter_vram_by_adapter("Shared Usage")) if luid else None
            lone.append({"used_gb": devs[0]["used_gb"] if devs else None,
                         "total_gb": devs[0]["total_gb"] if devs else None, "agg": agg, "ms": ms,
                         "ref_gib": [None if r is None else r / GIB for r in (ref0, ref1)]})
        obs["lone"] = lone
        try:
            vis, ms = _timed(hw.get_visible_gpu_utilization)
            obs["visible_poll_ms"] = ms
            obs["visible_devices"] = [{k: d.get(k) for k in ("index", "name", "vram_used_gb", "vram_total_gb")}
                                      for d in vis.get("devices", [])]
        except Exception as e:  # noqa: BLE001
            obs["visible_error"] = f"{type(e).__name__}: {e}"

        # ---- spoofed: the real iGPU at ordinal 0, a phantom discrete card at ordinal 1
        real_cuda = torch.cuda
        real_props, real_mem = real_cuda.get_device_properties, getattr(real_cuda, "mem_get_info", None)

        def props(i):
            if int(i) == 0:
                return real_props(0)
            return types.SimpleNamespace(name = PHANTOM[0], total_memory = PHANTOM[1], gcnArchName = PHANTOM[2])

        def mem(i = 0):
            return real_mem(0) if int(i) == 0 else (PHANTOM[1], PHANTOM[1])

        real_counter = hw._rocm_windows_perf_counter_vram_by_adapter
        real_ids = hw._rocm_windows_hip_adapter_ids

        def counter(counter = "Dedicated Usage"):
            rows = real_counter(counter)
            if rows is None:
                return None
            phantom = PHANTOM_DED if counter == "Dedicated Usage" else PHANTOM_SHARED
            return list(rows) + [(f"luid_0x00000000_0x{PHANTOM_LUID:08x}_phys_0", phantom)]

        def hip_ids(ordinals, names):
            out = []
            for o, n in zip(ordinals, names):
                if int(o) == 0:
                    r = real_ids([0], [n])
                    if not r:
                        return None
                    out.append(r[0])
                else:
                    out.append((PHANTOM_LUID, 0))
            return out

        spoof = []
        try:
            real_cuda.get_device_properties = props
            if real_mem is not None:
                real_cuda.mem_get_info = mem
            hw._rocm_windows_perf_counter_vram_by_adapter = counter
            hw._rocm_windows_hip_adapter_ids = hip_ids
            for _ in range(args.repeats):
                ref0 = _luid_sum(hw, luid, real_counter(), real_counter("Shared Usage")) if luid else None
                (devs, agg), ms = _timed(hw._rocm_windows_per_device_vram, [0, 1])
                ref1 = _luid_sum(hw, luid, real_counter(), real_counter("Shared Usage")) if luid else None
                spoof.append({"devices": [{k: d.get(k) for k in ("index", "name", "used_gb", "total_gb")} for d in devs],
                              "agg": agg, "ms": ms,
                              "ref_gib": [None if r is None else r / GIB for r in (ref0, ref1)]})
        finally:
            real_cuda.get_device_properties = real_props
            if real_mem is not None:
                real_cuda.mem_get_info = real_mem
            hw._rocm_windows_perf_counter_vram_by_adapter = real_counter
            hw._rocm_windows_hip_adapter_ids = real_ids
        obs["spoof"] = spoof
        obs["phantom"] = {"luid": PHANTOM_LUID, "dedicated_gib": PHANTOM_DED / GIB, "total_gib": PHANTOM[1] / GIB}
    except Exception as e:  # noqa: BLE001
        import traceback
        obs["error"] = f"{type(e).__name__}: {e}"
        obs["traceback"] = traceback.format_exc()[-3000:]
    args.out.write_text(json.dumps(obs, indent = 2, default = str), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
