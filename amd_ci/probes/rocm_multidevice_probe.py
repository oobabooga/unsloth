#!/usr/bin/env python3
"""Probe: can this one-GPU host be made to present TWO torch devices, and if so,
does PR 8791's gate then drop the uncovered one?

Why this exists. The defect in issue 8792 is an iGPU the torch wheel has no kernels
for sitting beside a covered discrete card. The runner has a single gfx1151, so every
earlier leg could only show non-regression. `GGML_CUDA_DEVICES` gives llama.cpp real
virtual devices, but it is a ggml-backend feature and does not create torch devices,
so it does not reach `auto_select_gpu_ids`. This probe asks whether anything does.

Phase 1 enumerates candidate mechanisms and records `torch.cuda.device_count()` for
each. It is an experiment, not an assertion: HIP's visibility semantics differ from
CUDA's and the answer is not knowable from a CUDA host.

Phase 2 takes the first mechanism that yields two or more devices and runs the real
selection path with device 1 presenting `gfx1036`, the arch from the issue. Only the
ARCH STRING is faked; the devices, the HIP contexts, the enumeration and the selection
are real. If no mechanism works, phase 2 falls back to a fully mocked device list and
says so, so the criteria can refuse to claim more than was achieved.

Observes only.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

# Each is (label, env overlay). Recorded whether it works or not: a mechanism that
# does nothing is as much a result as one that does, and saves the next person trying.
MECHANISMS = [
    ("baseline", {}),
    ("hip_dup", {"HIP_VISIBLE_DEVICES": "0,0"}),
    ("rocr_dup", {"ROCR_VISIBLE_DEVICES": "0,0"}),
    ("cuda_dup", {"CUDA_VISIBLE_DEVICES": "0,0"}),
    ("ordinal_dup", {"GPU_DEVICE_ORDINAL": "0,0"}),
    ("hip_triple", {"HIP_VISIBLE_DEVICES": "0,0,0"}),
    # The mechanism named in issue 9792. Expected to do nothing for torch; recorded so
    # the report can say that from measurement rather than from reasoning.
    ("ggml_virtual", {"GGML_CUDA_DEVICES": "3"}),
    ("ggml_virtual_plus_hip", {"GGML_CUDA_DEVICES": "3", "HIP_VISIBLE_DEVICES": "0,0"}),
]

_COUNT = r'''
import json, sys
out = {}
try:
    import torch
    out["hip"] = getattr(torch.version, "hip", None)
    out["count"] = int(torch.cuda.device_count())
    out["available"] = bool(torch.cuda.is_available())
    archs = []
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        archs.append(str(getattr(p, "gcnArchName", "") or ""))
    out["archs"] = archs
    try:
        out["arch_list"] = [str(a) for a in (torch.cuda.get_arch_list() or [])]
    except Exception as e:
        out["arch_list_error"] = str(e)
except Exception as e:
    out["error"] = "%s: %s" % (type(e).__name__, e)
print("@@J@@" + json.dumps(out))
'''

# Runs the REAL selection path with device 1 presenting an uncovered arch.
_SELECT = r'''
import json, os, sys

CHECKOUT = sys.argv[1]
UNCOVERED = sys.argv[2]          # arch to present on device 1
MOCK_COUNT = int(sys.argv[3])    # >0 means no real multi-device was available

out = {"mocked_device_count": MOCK_COUNT or None}

import torch
real_count = torch.cuda.device_count()
out["real_device_count"] = real_count
try:
    torch.cuda.is_available() and torch.cuda.init()
except Exception:
    pass

_real_props = torch.cuda.get_device_properties

class _P:
    """Presents a different gcnArchName; every other attribute stays the real one."""
    def __init__(self, p, arch):
        self.__dict__["_p"] = p
        self.__dict__["_arch"] = arch
    def __getattr__(self, name):
        if name in ("gcnArchName", "gcn_arch_name"):
            return self.__dict__["_arch"]
        return getattr(self.__dict__["_p"], name)

def _props(i = 0):
    base = _real_props(0)
    return _P(base, UNCOVERED if i == 1 else str(getattr(base, "gcnArchName", "")))

torch.cuda.get_device_properties = _props
if MOCK_COUNT:
    torch.cuda.device_count = lambda: MOCK_COUNT

out["presented_archs"] = [
    str(torch.cuda.get_device_properties(i).gcnArchName)
    for i in range(torch.cuda.device_count())
]

sys.path.insert(0, os.path.join(CHECKOUT, "studio", "backend"))
for stale in [m for m in list(sys.modules) if m.startswith("utils.")]:
    del sys.modules[stale]

try:
    import utils.hardware.hardware as hw
    out["hardware_file"] = hw.__file__
    fn = getattr(hw, "rocm_gpu_ids_without_torch_kernels", None)
    out["gate_present"] = fn is not None
    if fn is not None:
        try:
            out["uncovered_ids"] = sorted(int(i) for i in fn())
        except Exception as e:
            out["gate_error"] = "%s: %s" % (type(e).__name__, e)
    try:
        sel, meta = hw.auto_select_gpu_ids("unsloth/Qwen3-0.6B", required_gb = None)
        out["selected"] = None if sel is None else list(sel)
        out["selection_mode"] = meta.get("selection_mode")
    except Exception as e:
        out["select_error"] = "%s: %s" % (type(e).__name__, e)
except Exception as e:
    import traceback
    out["hardware_error"] = "%s: %s" % (type(e).__name__, e)
    out["tb"] = traceback.format_exc()[-1500:]

print("@@J@@" + json.dumps(out))
'''


def _run(script: Path, args: list, overlay: dict, timeout: int) -> dict:
    env = dict(os.environ)
    for k in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES",
              "GPU_DEVICE_ORDINAL", "GGML_CUDA_DEVICES"):
        env.pop(k, None)
    env.update(overlay)
    try:
        r = subprocess.run([sys.executable, str(script)] + args, capture_output = True,
                           text = True, timeout = timeout, env = env)
    except subprocess.TimeoutExpired:
        return {"child_error": f"timed out after {timeout}s"}
    except Exception as e:  # noqa: BLE001
        return {"child_error": f"{type(e).__name__}: {e}"}
    for line in reversed((r.stdout or "").splitlines()):
        if line.startswith("@@J@@"):
            try:
                d = json.loads(line[5:])
                d["child_rc"] = r.returncode
                return d
            except Exception as e:  # noqa: BLE001
                return {"child_error": f"bad JSON: {e}"}
    return {"child_error": "no JSON line", "child_rc": r.returncode,
            "stderr_tail": (r.stderr or "")[-1200:]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True, type = Path)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--uncovered-arch", default = "gfx1036")
    ap.add_argument("--timeout", type = int, default = 600)
    args = ap.parse_args()

    obs: dict = {"state": args.state, "uncovered_arch": args.uncovered_arch}
    hwf = args.checkout / "studio" / "backend" / "utils" / "hardware" / "hardware.py"
    obs["has_gate_source"] = (
        hwf.is_file()
        and "def rocm_gpu_ids_without_torch_kernels" in hwf.read_text(
            encoding = "utf-8", errors = "replace")
    )

    with tempfile.TemporaryDirectory() as td:
        count_py = Path(td) / "count.py"
        count_py.write_text(_COUNT, encoding = "utf-8")
        sel_py = Path(td) / "select.py"
        sel_py.write_text(_SELECT, encoding = "utf-8")

        obs["mechanisms"] = {}
        winner = None
        for label, overlay in MECHANISMS:
            rec = _run(count_py, [], overlay, args.timeout)
            rec["env"] = overlay
            obs["mechanisms"][label] = rec
            if winner is None and label != "baseline" and (rec.get("count") or 0) >= 2:
                winner = (label, overlay)

        obs["real_multidevice_mechanism"] = winner[0] if winner else None
        overlay = winner[1] if winner else {}
        # 0 means "do not mock the count"; only fall back to mocking when nothing real
        # was available, and record which happened.
        mock = 0 if winner else 2
        obs["phase2_used_mocked_count"] = bool(mock)
        obs["selection"] = _run(
            sel_py, [str(args.checkout), args.uncovered_arch, str(mock)],
            overlay, args.timeout,
        )

    args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
