#!/usr/bin/env python3
"""Probe: make the runner's REAL GPU genuinely uncovered, by changing the wheel.

Every earlier attempt at PR 8791's defect faked the device: spoofed `gcnArchName`, or
mocked `device_count()`. Neither is available honestly, because no ROCm mechanism
duplicates a GPU (CPX partitioning does, and is CDNA-only).

This inverts the problem. The gate compares the device's arch against
`torch.cuda.get_arch_list()`, so instead of faking the device, install a torch wheel
the device is genuinely outside of. AMD ships per-arch indexes, and `gfx110X-all`
covers gfx1100/1101/1102/1103 -- RDNA3 discrete -- and NOT gfx1151, which is RDNA3.5.
On this runner that wheel is a real, shipping, unmodified build for which the real
silicon is real uncovered hardware. Nothing is spoofed.

What that buys, and what it does not:

  - It reproduces the CONDITION behind issue 8792 on real hardware, and lets a real
    GPU op say what actually happens (the reported symptom is hipErrorInvalidKernelFile).
  - It exercises the gate's detection against a real uncovered device.
  - It does NOT reproduce the two-GPU SELECTION defect. With one device, and that
    device uncovered, the gate's fail-open rule deliberately returns the empty set
    rather than stranding the host on CPU. That rule firing correctly on real silicon
    is itself worth measuring, but it is not the iGPU-beside-a-dGPU case.

Observes only. Returns 0 even when the install or the op fails; those are readings.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

# RDNA3 discrete. Deliberately excludes gfx1151, which is what makes this work.
DEFAULT_INDEX = "https://rocm.nightlies.amd.com/v2/gfx110X-all/"

_READ = r'''
import json, os, sys
out = {}
try:
    import torch
    out["torch_version"] = torch.__version__
    out["hip"] = getattr(torch.version, "hip", None)
    out["available"] = bool(torch.cuda.is_available())
    out["device_count"] = int(torch.cuda.device_count())
    try:
        out["arch_list"] = [str(a) for a in (torch.cuda.get_arch_list() or [])]
    except Exception as e:
        out["arch_list_error"] = "%s: %s" % (type(e).__name__, e)
    archs = []
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        archs.append(str(getattr(p, "gcnArchName", "") or ""))
    out["device_archs"] = archs
    # Is the real device inside what this wheel was built for? Computed the same way
    # the gate does, so the reading and the gate cannot disagree by construction.
    tokens = set(str(a).split(":")[0].strip().lower() for a in (out.get("arch_list") or []))
    dev = [a.split(":")[0].strip().lower() for a in archs]
    out["device_is_covered"] = bool(dev) and all(d in tokens for d in dev)

    # The reported symptom. A real op on real silicon with a wheel that has no code
    # object for it: this is the thing the PR exists to keep users away from.
    try:
        a = torch.randn(256, 256, device = "cuda", dtype = torch.float32)
        b = (a @ a).sum().item()
        out["matmul_ok"] = True
        out["matmul_finite"] = bool(b == b)
    except Exception as e:
        out["matmul_ok"] = False
        out["matmul_error"] = "%s: %s" % (type(e).__name__, e)
except Exception as e:
    out["torch_error"] = "%s: %s" % (type(e).__name__, e)
print("@@J@@" + json.dumps(out))
'''

_GATE = r'''
import io, json, logging, os, sys

CHECKOUT = sys.argv[1]
out = {}

# The gate logs rather than returns when it decides the whole host is uncovered, so
# capture the log or the most informative half of the reading is lost.
buf = io.StringIO()
h = logging.StreamHandler(buf)
logging.getLogger().addHandler(h)
logging.getLogger().setLevel(logging.DEBUG)

sys.path.insert(0, os.path.join(CHECKOUT, "studio", "backend"))
for stale in [m for m in list(sys.modules) if m.startswith("utils.")]:
    del sys.modules[stale]
try:
    import utils.hardware.hardware as hw
    out["hardware_file"] = hw.__file__

    # The gate has several early exits that all return the empty set, and an empty
    # result on its own cannot say which one fired. Record each precondition the way
    # the gate reads it, so "bailed before looking" is distinguishable from "looked,
    # found the whole host uncovered, and failed open on purpose".
    import torch as _t
    out["pre_torch_available"] = bool(_t.cuda.is_available())
    for name in ("_rocm_device_ordinal_active", "_rocm_visibility_masks_are_stacked"):
        f = getattr(hw, name, None)
        try:
            out["pre_" + name] = bool(f()) if f else None
        except Exception as e:
            out["pre_" + name] = "err: %s" % e
    try:
        spec = hw._get_parent_visible_gpu_spec()
        out["pre_numeric_ids"] = spec.get("numeric_ids")
    except Exception as e:
        out["pre_numeric_ids_error"] = "%s: %s" % (type(e).__name__, e)

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
except Exception as e:
    import traceback
    out["hardware_error"] = "%s: %s" % (type(e).__name__, e)
    out["tb"] = traceback.format_exc()[-1500:]

out["log"] = buf.getvalue()[-3000:]
print("@@J@@" + json.dumps(out))
'''


def _json_tail(r) -> dict:
    for line in reversed((r.stdout or "").splitlines()):
        if line.startswith("@@J@@"):
            try:
                return json.loads(line[5:])
            except Exception as e:  # noqa: BLE001
                return {"parse_error": str(e)}
    return {"no_json": True, "rc": r.returncode,
            "stdout_tail": (r.stdout or "")[-1200:],
            "stderr_tail": (r.stderr or "")[-1500:]}


def _run(python: str, code: str, args: list, timeout: int) -> dict:
    env = dict(os.environ)
    for k in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
        env.pop(k, None)
    try:
        r = subprocess.run([python, "-c", code] + args, capture_output = True,
                           text = True, timeout = timeout, env = env)
    except subprocess.TimeoutExpired:
        return {"child_error": f"timed out after {timeout}s"}
    except Exception as e:  # noqa: BLE001
        return {"child_error": f"{type(e).__name__}: {e}"}
    d = _json_tail(r)
    d["child_rc"] = r.returncode
    # The backend logs through structlog, which writes to the child's real streams
    # rather than the stdlib handler the gate script installs. Without these the
    # "no kernels for any GPU" warning is invisible and reads as "never emitted".
    d["child_stdout"] = "\n".join(
        l for l in (r.stdout or "").splitlines() if not l.startswith("@@J@@"))[-3000:]
    d["child_stderr"] = (r.stderr or "")[-3000:]
    return d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True, type = Path)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--index", default = DEFAULT_INDEX)
    ap.add_argument("--venv", default = "")
    ap.add_argument("--timeout", type = int, default = 3000)
    args = ap.parse_args()

    obs: dict = {"state": args.state, "index": args.index}
    hwf = args.checkout / "studio" / "backend" / "utils" / "hardware" / "hardware.py"
    obs["has_gate_source"] = (
        hwf.is_file()
        and "def rocm_gpu_ids_without_torch_kernels" in hwf.read_text(
            encoding = "utf-8", errors = "replace")
    )

    # The stock reading: the covered wheel the runner already has. It is the control
    # that says the uncovered result below came from the wheel and not the host.
    obs["stock"] = _run(sys.executable, _READ, [], args.timeout)
    obs["stock_gate"] = _run(sys.executable, _GATE, [str(args.checkout)], args.timeout)

    # A second venv carrying a wheel this GPU is genuinely outside of.
    venv = Path(args.venv) if args.venv else (args.out.parent / "venv_uncovered")
    py = venv / "bin" / "python"
    steps: dict = {}
    try:
        # Rebuilt per state, and deliberately WITHOUT --system-site-packages. With it,
        # the parent's torch counts as satisfying the requirement, pip reports
        # "already satisfied" and the swap silently does not happen -- which then
        # reads as a result rather than as a broken harness. Observed exactly that.
        if venv.is_dir():
            shutil.rmtree(venv, ignore_errors = True)
        steps["create"] = subprocess.run(
            [sys.executable, "-m", "venv", str(venv)],
            capture_output = True, text = True, timeout = 600).returncode
        # Only torch is replaced; everything else comes from system site-packages, so
        # the studio backend imports keep working.
        p = subprocess.run(
            [str(py), "-m", "pip", "install", "--no-cache-dir", "--force-reinstall",
             "--index-url", args.index, "--pre", "torch"],
            capture_output = True, text = True, timeout = args.timeout,
        )
        steps["pip_rc"] = p.returncode
        # `venv --system-site-packages` inherits the BASE interpreter's site-packages,
        # not the studio venv this runs under, so the backend's own deps are missing
        # and the gate child dies on `import structlog` before reaching the gate.
        # PYTHONPATH is not an option: it precedes the venv and would shadow the
        # uncovered torch with the covered one, silently undoing the whole point.
        d = subprocess.run(
            [str(py), "-m", "pip", "install", "--no-cache-dir", "structlog"],
            capture_output = True, text = True, timeout = 900,
        )
        steps["deps_rc"] = d.returncode
        steps["pip_tail"] = (p.stdout or "")[-1500:] + (p.stderr or "")[-1500:]
    except Exception as e:  # noqa: BLE001
        steps["error"] = f"{type(e).__name__}: {e}"
    obs["install"] = steps

    if py.is_file() and steps.get("pip_rc") == 0:
        obs["uncovered"] = _run(str(py), _READ, [], args.timeout)
        obs["uncovered_gate"] = _run(str(py), _GATE, [str(args.checkout)], args.timeout)
    else:
        obs["uncovered"] = {"skipped": "the uncovered wheel did not install"}
        obs["uncovered_gate"] = {"skipped": "the uncovered wheel did not install"}

    args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
