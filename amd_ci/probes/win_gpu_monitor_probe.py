#!/usr/bin/env python3
"""Probe (Windows ROCm): what does this checkout's GPU monitor report idle and under a HIP load?

Observes only. Next to the checkout's readings it records an independent ground truth: one
Get-Counter of every engine of the adapter's LUID, summed per engine type, read by this probe's
own PowerShell (not the checkout's), so a criteria module can tell "the checkout is blind" from
"there was no load". Pairs with criteria/win_gpu_monitor_18005.py (unslothai/unsloth#18005).
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

LOAD = r"""
import sys, time, torch
a = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
b = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
torch.cuda.synchronize()
print("READY", flush=True)
end = time.time() + float(sys.argv[1]); n = 0
while time.time() < end:
    for _ in range(16):
        a = (a @ b).clamp_(-1, 1)
    torch.cuda.synchronize(); n += 16
print("OPS", n, flush=True)
"""


def import_hardware(checkout: Path):
    backend = checkout / "studio" / "backend"
    sys.path.insert(0, str(backend))
    for stale in [m for m in sys.modules if m.startswith("utils.")]:
        del sys.modules[stale]
    import utils.hardware.hardware as hw  # noqa: PLC0415
    return hw


def raw_engine_types(luid: int) -> dict:
    """Ground truth: this adapter's engines, summed per engine type, read independently."""
    path = f"\\GPU Engine(*luid_0x{luid >> 32:08x}_0x{luid & 0xFFFFFFFF:08x}*)\\Utilization Percentage"
    ps = (f"$s=(Get-Counter '{path}' -ErrorAction SilentlyContinue).CounterSamples;"
          "$s|ForEach-Object{'{0}|{1}' -f $_.InstanceName,$_.CookedValue.ToString([cultureinfo]::InvariantCulture)}")
    t0 = time.monotonic()
    r = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
                       capture_output = True, text = True, encoding = "utf-8", errors = "replace", timeout = 60)
    out: dict = {"seconds": round(time.monotonic() - t0, 2), "types": {}, "instances": 0, "sample_names": []}
    for line in r.stdout.splitlines():
        name, sep, val = line.strip().rpartition("|")
        if not sep or "engtype_" not in name.lower():
            continue
        out["instances"] += 1
        if len(out["sample_names"]) < 4:
            out["sample_names"].append(name)
        t = name.lower().rpartition("engtype_")[2]
        try:
            out["types"][t] = round(out["types"].get(t, 0.0) + float(val), 2)
        except ValueError:
            pass
    return out


def all_engines_seconds() -> float:
    ps = "(Get-Counter '\\GPU Engine(*)\\Utilization Percentage' -ErrorAction SilentlyContinue).CounterSamples.Count"
    t0 = time.monotonic()
    subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
                   capture_output = True, text = True, timeout = 120)
    return round(time.monotonic() - t0, 2)


def sample(hw, luid: int, n: int, spacing: float) -> list[dict]:
    rows = []
    for _ in range(n):
        row: dict = {}
        t0 = time.monotonic()
        try:
            row["util_fn"] = hw._rocm_windows_perf_counter_gpu_util_pct(luid)
        except Exception as e:  # noqa: BLE001
            row["util_fn_error"] = repr(e)
        row["util_fn_s"] = round(time.monotonic() - t0, 2)
        temp_fn = getattr(hw, "_rocm_windows_d3dkmt_temperature_c", None)
        row["temp_fn_present"] = temp_fn is not None
        if temp_fn is not None:
            t0 = time.monotonic()
            row["temp_fn"] = temp_fn(luid)
            row["temp_fn_ms"] = round((time.monotonic() - t0) * 1000, 3)
        try:
            payload = hw.get_gpu_utilization()
            dev = (payload.get("devices") or [{}])[0]
            row["payload"] = {k: dev.get(k) for k in ("backend", "name", "gpu_utilization_pct", "temperature_c",
                                                       "power_draw_w", "vram_used_gb", "vram_total_gb")}
        except Exception as e:  # noqa: BLE001
            row["payload_error"] = repr(e)
        row["raw"] = raw_engine_types(luid)
        rows.append(row)
        time.sleep(spacing)
    return rows


def med(rows, key):
    vals = [r.get(key) for r in rows if isinstance(r.get(key), (int, float))]
    return statistics.median(vals) if vals else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True, type = Path)
    ap.add_argument("--out", required = True, type = Path)
    args = ap.parse_args()
    obs: dict = {"state": args.state}
    try:
        import torch
        obs["torch"] = torch.__version__
        obs["hip"] = getattr(torch.version, "hip", None)
        obs["cuda_available"] = torch.cuda.is_available()
        if obs["cuda_available"]:
            p = torch.cuda.get_device_properties(0)
            obs["arch"] = getattr(p, "gcnArchName", None)
            obs["device_name"] = p.name
    except Exception as e:  # noqa: BLE001
        obs["torch_error"] = repr(e)
    try:
        hw = import_hardware(args.checkout)
        obs["hardware_file"] = hw.__file__
        obs["is_rocm"] = bool(getattr(hw, "IS_ROCM", False))
        devices, _ = hw._rocm_windows_per_device_vram([0])
        luid = (devices or [{}])[0].get("luid")
        obs["luid"] = luid
        if luid is None:
            raise RuntimeError(f"no LUID for device 0: {devices!r}")
        obs["all_engines_query_s"] = all_engines_seconds()
        obs["idle"] = sample(hw, luid, 3, 4.0)
        load = subprocess.Popen([sys.executable, "-c", LOAD, "60"], stdout = subprocess.PIPE,
                                stderr = subprocess.STDOUT, text = True)
        first = load.stdout.readline().strip()
        obs["load_first_line"] = first
        if first == "READY":
            time.sleep(8.0)
            obs["load"] = sample(hw, luid, 6, 4.0)
        rest = load.communicate(timeout = 120)[0]
        obs["load_tail"] = rest[-400:]
        obs["load_ops"] = next((int(x.split()[1]) for x in rest.splitlines() if x.startswith("OPS")), 0)
        obs["load_rc"] = load.returncode
        for phase in ("idle", "load"):
            rows = obs.get(phase) or []
            obs[f"{phase}_util_fn_median"] = med(rows, "util_fn")
            obs[f"{phase}_temp_fn_median"] = med(rows, "temp_fn")
            obs[f"{phase}_payload_util_median"] = med([r.get("payload") or {} for r in rows], "gpu_utilization_pct")
            obs[f"{phase}_payload_temp_median"] = med([r.get("payload") or {} for r in rows], "temperature_c")
            obs[f"{phase}_raw_busiest_median"] = med(
                [{"v": max((r["raw"]["types"] or {"": None}).values(), default = None)} for r in rows], "v")
            obs[f"{phase}_raw_3d_median"] = med([{"v": r["raw"]["types"].get("3d")} for r in rows], "v")
            obs[f"{phase}_util_fn_s_max"] = max((r.get("util_fn_s", 0) for r in rows), default = None)
    except Exception as e:  # noqa: BLE001
        import traceback
        obs["error"] = traceback.format_exc()[-2000:]
    args.out.parent.mkdir(parents = True, exist_ok = True)
    args.out.write_text(json.dumps(obs, indent = 1, default = str), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
