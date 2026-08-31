#!/usr/bin/env python3
"""Probe: PR 8791's gate and selector, against two devices torch REALLY has.

The existing evidence on 8791 used three fabrications: the amd-smi binary,
`torch.cuda.device_count()` and `get_device_properties()`. The HIP device multiplier
removes two of them. Under it torch enumerates two devices because the C++ runtime says
so, and `cuda:1` holds real tensors and runs real kernels -- neither is a Python patch
any more.

What is still faked, and why each is unavoidable here:

  1. The arch string of device 1. The gate compares the device's arch against
     `get_arch_list()`, so with both devices reporting the real gfx1151 neither is
     uncovered and the gate has nothing to find. This is the ONE Python patch left.
  2. The amd-smi binary, for the selection leg only. `hardware.py` enumerates through
     amd-smi, not torch, so the shim does not move it; the stub wraps the REAL amd-smi
     and duplicates its entry so the schema stays correct by construction.

Two legs, because they need different amounts of faking and should be read separately:

  gate leg      -- shim + arch patch. No amd-smi stub. Tests
                   `rocm_gpu_ids_without_torch_kernels()` against two REAL torch
                   devices. This is the strongest form of the 8791 evidence.
  selection leg -- the above plus the amd-smi stub, so `auto_select_gpu_ids` can
                   enumerate and the whole path runs end to end.

Still out of reach, and it is the same limit as before: the phantom device IS device 0
wearing another number, so nothing here says a model really shards across two GPUs.
What it now does say is that the gate reaches its decision on devices torch is not
merely pretending to have.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import device_multiplier as _dm  # noqa: E402

# Wraps the REAL amd-smi and duplicates its GPU entry, so the JSON is the real tool's
# schema rather than one reverse-engineered from _gpu_entries. Envelope keys and the
# id fields follow amd.py::_gpu_entries.
AMDSMI_STUB = r'''#!{python}
import json, subprocess, sys
REAL = {real!r}
if not REAL:
    sys.exit(127)
r = subprocess.run([REAL] + sys.argv[1:], capture_output = True, text = True)
try:
    data = json.loads(r.stdout)
except Exception:
    sys.stdout.write(r.stdout); sys.stderr.write(r.stderr); sys.exit(r.returncode)

def _scale(v, f):
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return type(v)(v * f)
    if isinstance(v, dict) and isinstance(v.get("value"), (int, float)):
        d = dict(v); d["value"] = type(v["value"])(v["value"] * f); return d
    if isinstance(v, str):
        try:
            h, _, rest = v.partition(" ")
            return ("%g %s" % (float(h) * f, rest)).strip()
        except Exception:
            return v
    return v

def bump(e, idx):
    e = json.loads(json.dumps(e))
    if not isinstance(e, dict):
        return e
    for k in ("gpu", "gpu_id", "id", "hip_id", "index", "device_id"):
        if k in e and isinstance(e[k], (int, float, str)):
            e[k] = idx
    # Shape the copy like the iGPU in issue 8792: more total VRAM, almost none used,
    # so it WINS a free-VRAM-only ranking. Without this the two devices tie, the base
    # picks the good card by tie-break, and the defect is hidden behind a clean-looking
    # base. This exact omission voided an earlier run; it is not optional.
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
    for k in ("gpu_data", "gpus", "gpu"):
        v = data.get(k)
        if isinstance(v, list) and v:
            data[k] = list(v) + [bump(v[0], len(v))]
            break
    else:
        data = [data, bump(data, 1)]
sys.stdout.write(json.dumps(data))
sys.stderr.write(r.stderr)
sys.exit(r.returncode)
'''

_BODY = r'''
import json, os, sys
CHECKOUT, UNCOVERED, WITH_SMI = sys.argv[1], sys.argv[2], sys.argv[3] == "1"
out = {"with_amdsmi_stub": WITH_SMI}

import torch
out["device_count"] = int(torch.cuda.device_count())
out["cpp_device_count"] = int(torch._C._cuda_getDeviceCount())
out["arch_list"] = list(torch.cuda.get_arch_list())

# Prove the devices are REAL before using them: a tensor and a kernel on each. If this
# fails the shim is not doing what the run assumes and nothing below is meaningful.
real = {}
for i in range(out["device_count"]):
    try:
        a = torch.randn(128, 128, device = "cuda:%d" % i)
        real["dev%d" % i] = "matmul ok" if (a @ a).sum().item() == (a @ a).sum().item() else "nan"
    except Exception as e:
        real["dev%d" % i] = "ERROR %s: %s" % (type(e).__name__, str(e)[:150])
out["devices_are_real"] = real

# The single remaining Python patch: the arch of the phantom device. Everything else
# about it -- existence, allocation, kernels -- is real.
_props = torch.cuda.get_device_properties
_base0 = _props(0)
_arch0 = str(getattr(_base0, "gcnArchName", ""))
class _P:
    def __init__(self, p, a):
        self.__dict__["_p"], self.__dict__["_a"] = p, a
    def __getattr__(self, n):
        if n in ("gcnArchName", "gcn_arch_name"):
            return self.__dict__["_a"]
        return getattr(self.__dict__["_p"], n)
torch.cuda.get_device_properties = lambda i = 0: _P(_props(i), UNCOVERED if i == 1 else _arch0)
out["presented_archs"] = [str(torch.cuda.get_device_properties(i).gcnArchName)
                          for i in range(out["device_count"])]

sys.path.insert(0, os.path.join(CHECKOUT, "studio", "backend"))
for m in [k for k in list(sys.modules) if k.startswith("utils.")]:
    del sys.modules[m]
try:
    import utils.hardware.hardware as hw
    fn = getattr(hw, "rocm_gpu_ids_without_torch_kernels", None)
    out["gate_present"] = fn is not None
    if fn is not None:
        try:
            out["uncovered_ids"] = sorted(int(i) for i in fn())
        except Exception as e:
            out["gate_error"] = "%s: %s" % (type(e).__name__, e)
    if WITH_SMI:
        try:
            out["physical_gpu_count"] = hw.get_physical_gpu_count()
            util = hw.get_visible_gpu_utilization()
            out["util_devices"] = [d.get("index") for d in (util.get("devices") or [])]
        except Exception as e:
            out["util_error"] = "%s: %s" % (type(e).__name__, e)
        try:
            sel, meta = hw.auto_select_gpu_ids("unsloth/Qwen3-0.6B")
            out["selected"] = None if sel is None else list(sel)
            out["selection_mode"] = meta.get("selection_mode")
        except Exception as e:
            out["select_error"] = "%s: %s" % (type(e).__name__, e)
    out["is_rocm"] = bool(getattr(hw, "IS_ROCM", False))
except Exception as e:
    import traceback
    out["hardware_error"] = "%s: %s" % (type(e).__name__, e)
    out["tb"] = traceback.format_exc()[-1200:]

print("@@J@@" + json.dumps(out))
'''


def _run(script_path: Path, env: dict, args_list: list, timeout: int) -> dict:
    try:
        r = subprocess.run([sys.executable, str(script_path)] + args_list,
                           capture_output = True, text = True, timeout = timeout,
                           env = env)
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}
    rec = {"_rc": r.returncode, "child_stdout": (r.stdout or "")[-3000:],
           "child_stderr": (r.stderr or "")[-3000:]}
    for line in reversed((r.stdout or "").splitlines()):
        if line.startswith("@@J@@"):
            rec.update(json.loads(line[5:]))
            return rec
    rec["no_json"] = True
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True, type = Path)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--uncovered-arch", default = "gfx1036")
    ap.add_argument("--timeout", type = int, default = 1200)
    args = ap.parse_args()

    obs: dict = {"state": args.state, "uncovered_arch": args.uncovered_arch}
    hwf = args.checkout / "studio" / "backend" / "utils" / "hardware" / "hardware.py"
    obs["has_gate_source"] = (
        hwf.is_file()
        and "def rocm_gpu_ids_without_torch_kernels" in hwf.read_text(
            encoding = "utf-8", errors = "replace"))

    real_smi = shutil.which("amd-smi")
    obs["real_amd_smi"] = real_smi

    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        script = tdp / "body.py"
        script.write_text(_BODY, encoding = "utf-8")

        base_env = dict(os.environ)
        for k in ("CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL",
                  "HIP_VISIBLE_DEVICES"):
            base_env.pop(k, None)

        ok, why = _dm.available()
        obs["multiplier_available"] = {"ok": ok, "why": why}
        if not ok:
            obs["error"] = f"device multiplier unavailable: {why}"
            args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
            return 0

        so = _dm.build(tdp / "shim")
        shim_env = _dm.env_for(so, extra_devices = 1, base = base_env)

        # Control: no shim at all, so the single-device host is on record.
        obs["control_no_shim"] = _run(
            script, base_env, [str(args.checkout), args.uncovered_arch, "0"], args.timeout)

        # Leg 1: gate only. Two REAL torch devices, one arch patch, no amd-smi stub.
        obs["gate_leg"] = _run(
            script, shim_env, [str(args.checkout), args.uncovered_arch, "0"], args.timeout)

        # Leg 2: the full selector, which additionally needs amd-smi to enumerate.
        stub_dir = tdp / "bin"
        stub_dir.mkdir()
        stub = stub_dir / "amd-smi"
        stub.write_text(AMDSMI_STUB.format(python = sys.executable, real = real_smi),
                        encoding = "utf-8")
        stub.chmod(stub.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        sel_env = dict(shim_env)
        sel_env["PATH"] = f"{stub_dir}{os.pathsep}{sel_env.get('PATH','')}"
        obs["selection_leg"] = _run(
            script, sel_env, [str(args.checkout), args.uncovered_arch, "1"], args.timeout)

    args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
