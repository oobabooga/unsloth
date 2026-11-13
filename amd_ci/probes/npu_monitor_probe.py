#!/usr/bin/env python3
"""Probe: what does this checkout report about the host's AMD NPU, and what does the OS say?

Observes only. On Windows the NPU's DEVPKEY_GPU_LUID is read independently and checked
against the adapter LUIDs Windows lists under GPU Adapter Memory (every adapter, busy or not).
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

_LUID_PS = (
    "$d=Get-PnpDevice -InstanceId 'PCI\\VEN_1022&DEV_17F0*' -PresentOnly"
    " -ErrorAction SilentlyContinue|Select-Object -First 1;"
    "if($d){$d.InstanceId+'|'+$d.Status+'|'+(Get-PnpDeviceProperty -InstanceId $d.InstanceId"
    " -KeyName '{60b193cb-5276-4d0f-96fc-f173abad3ec6} 2' -ErrorAction SilentlyContinue).Data}"
)
_ADAPTERS_PS = (
    "(Get-Counter '\\GPU Adapter Memory(*)\\Dedicated Usage' -ErrorAction SilentlyContinue)"
    ".CounterSamples|ForEach-Object{$_.InstanceName}"
)
_ENGINES_PS = (
    "(Get-Counter '\\GPU Engine(*)\\Utilization Percentage' -ErrorAction SilentlyContinue)"
    ".CounterSamples|ForEach-Object{$_.InstanceName}"
)


def _ps(script: str) -> dict:
    try:
        r = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output = True, text = True, encoding = "utf-8", errors = "replace", timeout = 60,
        )
        return {"rc": r.returncode, "stdout": r.stdout.strip(), "stderr": r.stderr.strip()[-400:]}
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}


def _luids(names: list[str]) -> list[int]:
    out = set()
    for n in names:
        m = re.search(r"luid_0x([0-9a-f]{8})_0x([0-9a-f]{8})", n.lower())
        if m:
            out.add((int(m.group(1), 16) << 32) | int(m.group(2), 16))
    return sorted(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True, type = Path)
    ap.add_argument("--out", required = True, type = Path)
    args = ap.parse_args()

    obs: dict = {"state": args.state, "os": platform.system()}
    backend = args.checkout / "studio" / "backend"
    sys.path.insert(0, str(backend))
    try:
        from utils.hardware.npu import detect_amd_npu  # noqa: PLC0415

        obs["detect"] = detect_amd_npu()
    except Exception as e:  # noqa: BLE001
        obs["detect_error"] = f"{type(e).__name__}: {e}"
    try:
        import utils.hardware.hardware as hw  # noqa: PLC0415

        obs["hardware_file"] = hw.__file__
        fn = getattr(hw, "npu_load_percent", None)
        obs["has_npu_load_percent"] = fn is not None
        if fn is not None:
            calls = []
            for _ in range(3):
                t0 = time.monotonic()
                value = fn()
                calls.append({"value": value, "seconds": round(time.monotonic() - t0, 3)})
            obs["npu_load_calls"] = calls
            obs["npu_luid_cached"] = getattr(hw, "_npu_luid", None)
    except Exception as e:  # noqa: BLE001
        obs["hardware_error"] = f"{type(e).__name__}: {e}"

    if obs["os"] == "Windows":
        raw = _ps(_LUID_PS)
        obs["pnp"] = raw
        parts = (raw.get("stdout") or "").split("|")
        obs["pnp_instance"] = parts[0] if parts and parts[0] else None
        obs["pnp_status"] = parts[1] if len(parts) > 1 else None
        luid = parts[2].strip() if len(parts) > 2 else ""
        obs["pnp_luid"] = int(luid) if luid.isdigit() else None
        adapters = _ps(_ADAPTERS_PS)
        obs["adapter_luids"] = _luids((adapters.get("stdout") or "").splitlines())
        engines = _ps(_ENGINES_PS)
        names = (engines.get("stdout") or "").splitlines()
        obs["engine_luids"] = _luids(names)
        if obs["pnp_luid"] is not None:
            key = f"luid_0x{obs['pnp_luid'] >> 32:08x}_0x{obs['pnp_luid'] & 0xFFFFFFFF:08x}"
            obs["npu_engine_instances"] = [n for n in names if key in n.lower()][:20]
    else:
        obs["accel_nodes"] = sorted(str(p) for p in Path("/dev/accel").glob("accel*"))
        obs["amdxdna_loaded"] = Path("/sys/module/amdxdna").is_dir()

    args.out.write_text(json.dumps(obs, indent = 2, default = str), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
