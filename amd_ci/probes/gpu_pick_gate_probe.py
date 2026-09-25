#!/usr/bin/env python3
"""Probe: the real torch-kernel gate and an explicit gpu_ids=[0] pick, unmocked, at one state.

Observes only; criteria/gpu_pick_gate_no_regression.py judges. Runs the state's own
studio/backend in a child interpreter so base and head never share imported modules.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_CHILD = r"""
import json, sys
out = {}
try:
    import torch
    out["torch"] = torch.__version__
    out["hip"] = getattr(torch.version, "hip", None)
    out["cuda_available"] = bool(torch.cuda.is_available())
    out["device_count"] = torch.cuda.device_count() if out["cuda_available"] else 0
    out["arch_list"] = list(torch.cuda.get_arch_list() or []) if out["cuda_available"] else []
    out["device_archs"] = [
        str(getattr(torch.cuda.get_device_properties(i), "gcnArchName", ""))
        for i in range(out["device_count"])
    ]
except Exception as e:
    out["torch_error"] = f"{type(e).__name__}: {e}"
try:
    import utils.hardware.hardware as hw
    out["uncovered"] = sorted(hw.rocm_gpu_ids_without_torch_kernels())
    try:
        sel, meta = hw.prepare_gpu_selection([0], model_name="unsloth/Qwen3-0.6B")
        out["pick0"] = {"ok": True, "selected": sel, "mode": meta.get("selection_mode")}
    except Exception as e:
        out["pick0"] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
except Exception as e:
    out["import_error"] = f"{type(e).__name__}: {e}"
print("PROBE_JSON " + json.dumps(out))
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--subdir", default = "studio/backend")
    ap.add_argument("--python", default = sys.executable)
    ap.add_argument("--timeout", type = int, default = 900)
    args, _ = ap.parse_known_args()
    workdir = Path(args.checkout) / args.subdir
    obs: dict = {"state": args.state, "workdir": str(workdir)}
    try:
        p = subprocess.run(
            [args.python, "-c", _CHILD], cwd = workdir, capture_output = True,
            text = True, encoding = "utf-8", errors = "replace", timeout = args.timeout,
        )
        obs["rc"] = p.returncode
        lines = [l for l in p.stdout.splitlines() if l.startswith("PROBE_JSON ")]
        if lines:
            obs.update(json.loads(lines[-1][len("PROBE_JSON "):]))
        else:
            obs["error"] = "no PROBE_JSON line"
        obs["stderr_tail"] = p.stderr[-1500:]
    except Exception as e:
        obs["error"] = f"{type(e).__name__}: {e}"
    args.out.write_text(json.dumps(obs, indent = 1), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
