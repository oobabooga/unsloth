#!/usr/bin/env python3
"""Probe: which ROCm gfx target does this checkout's install_llama_prebuilt.detect_host() pick on
THIS Windows host, per visibility-mask cell?

Observes only, on the host's REAL hipinfo.exe; nothing is installed. Each cell runs in a fresh
interpreter so no mask leaks. Pairs with criteria/windows_hipinfo_same.py.

Writes JSON to --out, never stdout.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

MASK_VARS = ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL")
CELLS = {
    "none": {},
    "hip0": {"HIP_VISIBLE_DEVICES": "0"},
    "hip_empty": {"HIP_VISIBLE_DEVICES": ""},
    "hip_minus1": {"HIP_VISIBLE_DEVICES": "-1"},
    "cuda0": {"CUDA_VISIBLE_DEVICES": "0"},
    "rocr_empty": {"ROCR_VISIBLE_DEVICES": ""},
    "rocr_minus1": {"ROCR_VISIBLE_DEVICES": "-1"},
    "hip0_rocr_empty": {"HIP_VISIBLE_DEVICES": "0", "ROCR_VISIBLE_DEVICES": ""},
    "rocr0": {"ROCR_VISIBLE_DEVICES": "0"},
}

CHILD = r"""
import json, sys
sys.path.insert(0, sys.argv[1])
out = {}
try:
    import install_llama_prebuilt as ilp
    h = ilp.detect_host()
    out = {"has_rocm": h.has_rocm, "rocm_gfx_target": h.rocm_gfx_target,
           "rocm_gfx_targets": list(h.rocm_gfx_targets or []),
           "has_amd_gpu_without_rocm": getattr(h, "has_amd_gpu_without_rocm", None),
           "is_windows": h.is_windows}
except Exception as e:
    out = {"error": f"{type(e).__name__}: {e}"}
open(sys.argv[2], "w", encoding="utf-8").write(json.dumps(out))
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True, type = Path)
    ap.add_argument("--out", required = True, type = Path)
    args = ap.parse_args()
    studio = args.checkout / "studio"
    work = args.out.parent / f"hipinfo_{args.state}"
    work.mkdir(parents = True, exist_ok = True)
    host = {"hipinfo_on_path": shutil.which("hipinfo") or shutil.which("hipInfo"),
            "hip_path": os.environ.get("HIP_PATH"), "platform": sys.platform}
    raw = None
    exe = host["hipinfo_on_path"] or (os.path.join(os.environ["HIP_PATH"], "bin", "hipinfo.exe")
                                       if os.environ.get("HIP_PATH") else None)
    if exe and os.path.exists(exe):
        try:
            raw = subprocess.run([exe], capture_output = True, text = True, encoding = "utf-8",
                                 errors = "replace", timeout = 60).stdout
        except Exception as e:  # noqa: BLE001
            raw = f"error: {e}"
    host["hipinfo_gcn_lines"] = [l.strip() for l in (raw or "").splitlines() if "gcnArch" in l or "gfx" in l][:10]
    cells = {}
    for name, env_over in CELLS.items():
        env = {k: v for k, v in os.environ.items() if k not in MASK_VARS}
        env.update(env_over)
        res = work / f"{name}.json"
        p = subprocess.run([sys.executable, "-c", CHILD, str(studio), str(res)], env = env,
                           capture_output = True, text = True, encoding = "utf-8", errors = "replace",
                           timeout = 600)
        try:
            cells[name] = json.loads(res.read_text(encoding = "utf-8"))
        except Exception as e:  # noqa: BLE001
            cells[name] = {"error": f"no result rc={p.returncode}: {e}; {p.stderr[-400:]}"}
        cells[name]["env"] = env_over
    doc = {"state": args.state, "host": host, "cells": cells}
    args.out.parent.mkdir(parents = True, exist_ok = True)
    args.out.write_text(json.dumps(doc, indent = 2), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
