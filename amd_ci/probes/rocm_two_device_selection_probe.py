#!/usr/bin/env python3
"""Probe: run the WHOLE selector with two devices, faking as little as possible.

An earlier attempt mocked `torch.cuda.device_count()` and got VOID, because
`hardware.py` does not enumerate through torch: it shells out to `amd-smi`, and the
selector bailed before the gate mattered. That failure says where the real boundaries
are, so this fakes exactly those and nothing else:

  1. `amd-smi` -- stubbed as a binary on PATH. The stub SHELLS OUT TO THE REAL amd-smi
     and duplicates its GPU entry, so the JSON schema is correct by construction rather
     than reverse-engineered. Same standard used for the install.sh probes: fake the
     external tool, keep every line of Python real.
  2. `torch.cuda.device_count()` -> 2 and `get_device_properties(1).gcnArchName` ->
     gfx1036. This one is unavoidable and is the honest limit of the exercise.

No visibility env var is set, and that is deliberate. With no mask,
`_get_parent_visible_gpu_spec()` returns `range(get_physical_gpu_count())`, which on a
ROCm host routes to `amd.get_physical_gpu_count()` -> `amd-smi list`. So the stub alone
carries the count all the way through the real spec function; an earlier draft set
`HIP_VISIBLE_DEVICES=0,1` to force it and that turned out to be an unnecessary third
fake that also risked breaking HIP init against a device that is not there.

Everything between those -- `get_visible_gpu_utilization`, the ranking, the gate, all
four selector exits, the metadata -- is the real code on a real ROCm host.

WHAT THIS CANNOT BECOME. torch's C++ runtime is bound to one HIP device. Device 1 is a
selection input, not a compute device: no tensor can live on it, so nothing here says
anything about kernel dispatch or about `device_map="balanced"` really sharding. It
tests the WIRING, which is where the VOID showed the risk actually is.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

# Runs the REAL amd-smi found outside the stub dir, then duplicates GPU 0 as GPU 1.
# Written to disk as an executable named `amd-smi` and put first on PATH.
STUB = r'''#!{python}
import json, os, subprocess, sys

REAL = {real!r}
if not REAL:
    sys.exit(127)
r = subprocess.run([REAL] + sys.argv[1:], capture_output = True, text = True)
out = r.stdout

# Only the JSON metric/list payloads need doubling; anything else passes through.
try:
    data = json.loads(out)
except Exception:
    sys.stdout.write(out); sys.stderr.write(r.stderr); sys.exit(r.returncode)

# _gpu_entries reads the id from "gpu", then "gpu_id", then "id"; hip_id feeds the
# ordinal->physical mapping. Bump whichever are present so the copy is device 1
# consistently at every layer that looks.
def _scale(v, factor):
    """Scale a memory reading in whatever shape amd-smi wrote it."""
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return type(v)(v * factor)
    if isinstance(v, dict) and isinstance(v.get("value"), (int, float)):
        d = dict(v); d["value"] = type(v["value"])(v["value"] * factor); return d
    if isinstance(v, str):
        try:
            head, _, rest = v.partition(" ")
            return ("%g %s" % (float(head) * factor, rest)).strip()
        except Exception:
            return v
    return v

def bump(entry, idx):
    e = json.loads(json.dumps(entry))
    if not isinstance(e, dict):
        return e
    for key in ("gpu", "gpu_id", "id", "hip_id", "index", "device_id"):
        if key in e and isinstance(e[key], (int, float, str)):
            e[key] = idx

    # Make the copy look like the iGPU from issue 8792: more total VRAM (an APU
    # reports a big slice of shared system memory) and almost none of it used, so
    # it OUTRANKS the real card under a free-VRAM-only ranking. Without this the
    # duplicate ties with device 0 and the tie-break hides the defect -- the base
    # picks the good card by luck and there is nothing for the gate to prove.
    vram = e.get("mem_usage", e.get("vram", e.get("fb_memory_usage")))
    if isinstance(vram, dict):
        for k in ("total_vram", "vram_total", "total"):
            if k in vram:
                vram[k] = _scale(vram[k], 2.0)
        for k in ("used_vram", "vram_used", "used"):
            if k in vram:
                vram[k] = _scale(vram[k], 0.05)
    return e

if isinstance(data, list) and data:
    data = list(data) + [bump(data[0], len(data))]
elif isinstance(data, dict):
    # Envelope keys and order taken from amd.py::_gpu_entries; a key only counts
    # when its value is really a list, exactly as the consumer requires.
    for k in ("gpu_data", "gpus", "gpu"):
        v = data.get(k)
        if isinstance(v, list) and v:
            data[k] = list(v) + [bump(v[0], len(v))]
            break
    else:
        # Single-GPU responses are a bare dict carrying their own "gpu" id.
        # amd-smi list is also consumed via data.get("gpu", data.get("gpus")).
        data = [data, bump(data, 1)]

sys.stdout.write(json.dumps(data))
sys.stderr.write(r.stderr)
sys.exit(r.returncode)
'''

_SELECT = r'''
import io, json, logging, os, sys

CHECKOUT, UNCOVERED = sys.argv[1], sys.argv[2]
out = {}

import torch
try:
    torch.cuda.is_available() and torch.cuda.init()
except Exception:
    pass

_real_props = torch.cuda.get_device_properties
_real_count = torch.cuda.device_count()
out["real_device_count"] = _real_count

class _P:
    def __init__(self, p, arch):
        self.__dict__["_p"] = p; self.__dict__["_a"] = arch
    def __getattr__(self, n):
        if n in ("gcnArchName", "gcn_arch_name"):
            return self.__dict__["_a"]
        return getattr(self.__dict__["_p"], n)

_base = _real_props(0)
_real_arch = str(getattr(_base, "gcnArchName", ""))
torch.cuda.get_device_properties = lambda i = 0: _P(_base, UNCOVERED if i == 1 else _real_arch)
torch.cuda.device_count = lambda: 2
out["presented_archs"] = [str(torch.cuda.get_device_properties(i).gcnArchName) for i in (0, 1)]

sys.path.insert(0, os.path.join(CHECKOUT, "studio", "backend"))
for stale in [m for m in list(sys.modules) if m.startswith("utils.")]:
    del sys.modules[stale]
try:
    import utils.hardware.hardware as hw
    out["hardware_file"] = hw.__file__
    try:
        out["physical_gpu_count"] = hw.get_physical_gpu_count()
    except Exception as e:
        out["physical_count_error"] = "%s: %s" % (type(e).__name__, e)
    try:
        spec = hw._get_parent_visible_gpu_spec()
        out["numeric_ids"] = spec.get("numeric_ids")
    except Exception as e:
        out["spec_error"] = "%s: %s" % (type(e).__name__, e)
    try:
        util = hw.get_visible_gpu_utilization()
        devs = util.get("devices") or []
        out["util_devices"] = [d.get("index") for d in devs]
        out["util_available"] = util.get("available")
        # The ranking input. Device 1 must show MORE free VRAM than device 0 or the
        # base has no reason to prefer it and the differential proves nothing.
        # _extract_gpu_metrics publishes vram_used_gb / vram_total_gb; free is
        # derived. An earlier run read vram_free_mb / memory_total, got None for
        # everything, and the ranking gate failed on my key names rather than on
        # anything the PR does.
        def _free(d):
            u, t = d.get("vram_used_gb"), d.get("vram_total_gb")
            return None if u is None or t is None else round(t - u, 2)
        out["util_vram"] = [
            {"index": d.get("index"), "free_gb": _free(d),
             "used_gb": d.get("vram_used_gb"), "total_gb": d.get("vram_total_gb")}
            for d in devs
        ]
    except Exception as e:
        out["util_error"] = "%s: %s" % (type(e).__name__, e)
    try:
        # get_gpu_vram_mib lives on the amd backend, not on hardware.py.
        from utils.hardware import amd as _amd
        out["vram_report"] = {str(k): v for k, v in
                              (_amd.get_gpu_vram_mib() or {}).items()}
    except Exception as e:
        out["vram_report_error"] = "%s: %s" % (type(e).__name__, e)
    fn = getattr(hw, "rocm_gpu_ids_without_torch_kernels", None)
    out["gate_present"] = fn is not None
    if fn is not None:
        try:
            out["uncovered_ids"] = sorted(int(i) for i in fn())
        except Exception as e:
            out["gate_error"] = "%s: %s" % (type(e).__name__, e)
    try:
        sel, meta = hw.auto_select_gpu_ids("unsloth/Qwen3-0.6B")
        out["selected"] = None if sel is None else list(sel)
        out["selection_mode"] = meta.get("selection_mode")
    except Exception as e:
        out["select_error"] = "%s: %s" % (type(e).__name__, e)
    # Read LAST. IS_ROCM is a module global that detection assigns; reading it
    # straight after import gets the False it was initialised to, which says
    # nothing about the host. An earlier run recorded exactly that and it read
    # as a finding when it was my ordering.
    out["is_rocm"] = bool(getattr(hw, "IS_ROCM", False))
    out["device_type"] = str(getattr(hw, "DEVICE", None))
except Exception as e:
    import traceback
    out["hardware_error"] = "%s: %s" % (type(e).__name__, e)
    out["tb"] = traceback.format_exc()[-1500:]

print("@@J@@" + json.dumps(out))
'''


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True, type = Path)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--uncovered-arch", default = "gfx1036")
    ap.add_argument("--timeout", type = int, default = 900)
    args = ap.parse_args()

    obs: dict = {"state": args.state, "uncovered_arch": args.uncovered_arch}
    hwf = args.checkout / "studio" / "backend" / "utils" / "hardware" / "hardware.py"
    obs["has_gate_source"] = (
        hwf.is_file()
        and "def rocm_gpu_ids_without_torch_kernels" in hwf.read_text(
            encoding = "utf-8", errors = "replace")
    )

    real = shutil.which("amd-smi")
    obs["real_amd_smi"] = real

    with tempfile.TemporaryDirectory() as td:
        stub_dir = Path(td) / "bin"
        stub_dir.mkdir()
        stub = stub_dir / "amd-smi"
        stub.write_text(STUB.format(python = sys.executable, real = real),
                        encoding = "utf-8")
        stub.chmod(stub.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

        script = Path(td) / "select.py"
        script.write_text(_SELECT, encoding = "utf-8")

        env = dict(os.environ)
        # No mask at all: the count must arrive through the real spec function via
        # get_physical_gpu_count() -> amd-smi, not through an env var I wrote.
        # Clearing these is also what makes the gate's own preconditions
        # (_rocm_device_ordinal_active, _rocm_visibility_masks_are_stacked) pass.
        for k in ("CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL",
                  "HIP_VISIBLE_DEVICES"):
            env.pop(k, None)
        env["PATH"] = f"{stub_dir}{os.pathsep}{env.get('PATH','')}"

        try:
            r = subprocess.run(
                [sys.executable, str(script), str(args.checkout), args.uncovered_arch],
                capture_output = True, text = True, timeout = args.timeout, env = env,
            )
            # The backend logs through structlog, which writes to the child's real
            # stdout/stderr rather than any handler installed inside the child. A
            # warning searched for only in a stdlib buffer reads as never emitted.
            rec = {"child_rc": r.returncode,
                   "child_stdout": (r.stdout or "")[-4000:],
                   "child_stderr": (r.stderr or "")[-4000:]}
            for line in reversed((r.stdout or "").splitlines()):
                if line.startswith("@@J@@"):
                    rec.update(json.loads(line[5:]))
                    break
            else:
                rec["no_json"] = True
                rec["stderr_tail"] = (r.stderr or "")[-1500:]
            obs["two_device"] = rec
        except Exception as e:  # noqa: BLE001
            obs["two_device"] = {"error": f"{type(e).__name__}: {e}"}

    args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
