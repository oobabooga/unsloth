#!/usr/bin/env python3
"""Probe: what torch route and GPU does THIS checkout pick on this machine? Observes only."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True, type = Path)
    ap.add_argument("--out", required = True, type = Path)
    args = ap.parse_args()
    obs: dict = {"state": args.state, "os": platform.system()}
    sys.path[:0] = [str(args.checkout / "studio"), str(args.checkout / "studio" / "backend")]
    try:
        import install_python_stack as ips
        obs["ips_file"] = ips.__file__
        if platform.system() == "Windows":
            gfx = ips._detect_windows_gfx_arch()
            obs["gfx"] = gfx
            obs["index_url"] = ips._windows_rocm_index_url(gfx)
            spec = getattr(ips, "_windows_rocm_torch_pkg_specs", None)
            obs["specs"] = list(spec(gfx) if spec else ips._WINDOWS_ROCM_TORCH_PKG_SPECS.get(gfx, ("torch", "torchvision", "torchaudio")))
        else:
            obs["has_rocm_gpu"] = bool(ips._has_rocm_gpu()) if hasattr(ips, "_has_rocm_gpu") else None
    except Exception as e:  # noqa: BLE001
        obs["route_error"] = f"{type(e).__name__}: {e}"
    try:
        import utils.hardware.hardware as hw
        t = time.perf_counter()
        obs["device"] = str(hw.detect_hardware())
        s = hw.get_gpu_summary()
        obs["detect_s"] = time.perf_counter() - t
        obs["gpu_name"] = s.get("gpu_name") or s.get("name")
        obs["summary_keys"] = sorted(s)
        mm = getattr(hw, "_amd_device_can_establish_a_mismatch", None)
        obs["mismatch_eligible"] = None
        if mm and obs["gpu_name"]:
            obs["mismatch_eligible"] = bool(mm({"name": obs["gpu_name"], "vendor": "amd"}))
    except Exception as e:  # noqa: BLE001
        obs["hw_error"] = f"{type(e).__name__}: {e}"
    args.out.write_text(json.dumps(obs, indent = 2, default = str), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
