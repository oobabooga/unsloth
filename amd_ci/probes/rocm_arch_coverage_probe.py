#!/usr/bin/env python3
"""Probe: what does this checkout's GPU selection do on a real, fully covered ROCm host?

PR 8791 drops GPUs whose arch is outside `torch.cuda.get_arch_list()`. This runner has
ONE gfx1151 that the installed wheel covers, so the defect (an uncovered iGPU beside a
discrete card) is out of reach. Two things it CAN settle, and both matter:

1. The real `get_arch_list()` of the shipped ROCm wheel. The gate rejects the list
   whole if any token is non-concrete, and a generic code object beside concrete
   targets is normal on ROCm 6.4+. If this wheel carries one, the gate never fires and
   the PR is a no-op on exactly the hosts it targets. Nothing but real hardware
   answers this.
2. That a covered single-GPU host selects exactly what it selected before, under every
   visibility mask, including the `apply_gpu_ids` ROCr translation the PR adds.

Observes only. Both readings are recorded for the base as well, where the functions do
not exist yet; "absent" is a reading, not an error.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

# Masks driven against the real device. Each is (label, env-overlay). "0" is the only
# physical id present, so an identity mask must be a no-op and a reordering cannot be
# expressed -- which is itself the point: a single-GPU host must come out unchanged.
MASKS = [
    ("none", {}),
    ("hip_only", {"HIP_VISIBLE_DEVICES": "0"}),
    ("rocr_identity", {"ROCR_VISIBLE_DEVICES": "0"}),
    ("both_stacked", {"ROCR_VISIBLE_DEVICES": "0", "HIP_VISIBLE_DEVICES": "0"}),
    ("cuda_only", {"CUDA_VISIBLE_DEVICES": "0"}),
    ("rocr_uuid", {"ROCR_VISIBLE_DEVICES": "GPU-deadbeefdeadbeef"}),
]

RUNNER = r'''
import json, os, sys

CHECKOUT = sys.argv[1]
out = {}

try:
    import torch
    out["torch_version"] = torch.__version__
    out["torch_hip"] = getattr(torch.version, "hip", None)
    out["available"] = bool(torch.cuda.is_available())
    out["device_count"] = int(torch.cuda.device_count())
    # THE measurement: what the shipped wheel says it was built for.
    try:
        out["arch_list"] = [str(a) for a in (torch.cuda.get_arch_list() or [])]
    except Exception as e:
        out["arch_list_error"] = "%s: %s" % (type(e).__name__, e)
    archs = []
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        archs.append(str(getattr(p, "gcnArchName", "") or ""))
    out["device_archs"] = archs
except Exception as e:
    out["torch_error"] = "%s: %s" % (type(e).__name__, e)

backend = os.path.join(CHECKOUT, "studio", "backend")
sys.path.insert(0, backend)
for stale in [m for m in list(sys.modules) if m.startswith("utils.")]:
    del sys.modules[stale]

try:
    import utils.hardware.hardware as hw
    out["hardware_file"] = hw.__file__

    fn = getattr(hw, "rocm_gpu_ids_without_torch_kernels", None)
    if fn is None:
        out["gate_present"] = False
    else:
        out["gate_present"] = True
        try:
            out["uncovered_ids"] = sorted(int(i) for i in fn())
        except Exception as e:
            out["gate_error"] = "%s: %s" % (type(e).__name__, e)

    try:
        out["parent_visible_ids"] = list(hw.get_parent_visible_gpu_ids())
    except Exception as e:
        out["parent_visible_error"] = "%s: %s" % (type(e).__name__, e)

    # What apply_gpu_ids writes is the user-visible half of the PR's second change.
    before = {k: os.environ.get(k) for k in
              ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")}
    try:
        hw.apply_gpu_ids([0])
        out["applied"] = {k: os.environ.get(k) for k in
                          ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES",
                           "ROCR_VISIBLE_DEVICES")}
        out["applied_env_before"] = before
    except Exception as e:
        out["apply_error"] = "%s: %s" % (type(e).__name__, e)
except Exception as e:
    import traceback
    out["hardware_error"] = "%s: %s" % (type(e).__name__, e)
    out["hardware_traceback"] = traceback.format_exc()[-2000:]

print("@@PROBE_JSON@@" + json.dumps(out))
'''


def _run(python: str, script: Path, checkout: str, overlay: dict, timeout: int) -> dict:
    env = dict(os.environ)
    # Clear every mask first so an overlay is the ONLY difference between readings;
    # the runner's own job env sets some of these and would otherwise leak in.
    for k in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES",
              "GPU_DEVICE_ORDINAL"):
        env.pop(k, None)
    env.update(overlay)
    try:
        r = subprocess.run([python, str(script), checkout], capture_output = True,
                           text = True, timeout = timeout, env = env)
    except subprocess.TimeoutExpired:
        return {"child_error": f"timed out after {timeout}s"}
    except Exception as e:  # noqa: BLE001
        return {"child_error": f"{type(e).__name__}: {e}"}
    for line in reversed((r.stdout or "").splitlines()):
        if line.startswith("@@PROBE_JSON@@"):
            try:
                d = json.loads(line[len("@@PROBE_JSON@@"):])
                d["child_rc"] = r.returncode
                return d
            except Exception as e:  # noqa: BLE001
                return {"child_error": f"bad JSON: {e}", "child_rc": r.returncode}
    return {"child_error": "no JSON line from the child", "child_rc": r.returncode,
            "stderr_tail": (r.stderr or "")[-1500:]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True, type = Path)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--python", default = sys.executable)
    ap.add_argument("--timeout", type = int, default = 600)
    args = ap.parse_args()

    obs: dict = {"state": args.state, "checkout": str(args.checkout)}
    hwf = args.checkout / "studio" / "backend" / "utils" / "hardware" / "hardware.py"
    src = hwf.read_text(encoding = "utf-8", errors = "replace") if hwf.is_file() else ""
    obs["has_gate_source"] = "def rocm_gpu_ids_without_torch_kernels" in src
    obs["has_rocr_translation_source"] = "def _rocr_relative_visibility" in src

    with tempfile.TemporaryDirectory() as td:
        script = Path(td) / "read_coverage.py"
        script.write_text(RUNNER, encoding = "utf-8")
        obs["masks"] = {}
        for label, overlay in MASKS:
            obs["masks"][label] = _run(args.python, script, str(args.checkout),
                                       overlay, args.timeout)

    args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
