#!/usr/bin/env python3
"""Observe what the test interpreter's torch IS, and what it SEES.

Observes only. It exists because of a confound that is easy to produce and hard
to notice: two differential runs of the same suite, one with
`HIP_VISIBLE_DEVICES=""` and one without, came back byte-identical. Readings
that agree perfectly across a variable are either evidence the variable does not
matter or evidence the variable never changed, and neither run recorded enough
to tell those apart. A CPU-only torch makes a HIP mask a no-op, so the "control"
would have varied nothing.

`has_real_accelerator` is read from the checkout's own `tests/_shared`, not
reimplemented, because that is the function `tests/conftest.py` branches on when
it decides whether to install the CUDA spoof -- which changes module state for
the whole session and therefore changes test outcomes.

Writes JSON via --out; the human-readable copy goes to stdout, so importing a
banner-printing module cannot corrupt the machine-readable one.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

MASK_VARS = ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES",
             "AMD_CI_SPOOFED_DEVICES")


def torch_facts() -> dict:
    out: dict = {}
    try:
        import torch
    except Exception as e:  # noqa: BLE001
        out["torch"] = None
        out["torch_import_error"] = f"{type(e).__name__}: {e}"
        return out
    out["torch"] = torch.__version__
    out["version_hip"] = torch.version.hip
    out["version_cuda"] = torch.version.cuda
    # A ROCm build reports through the cuda namespace; version_hip is what
    # distinguishes it from a real CUDA build.
    out["is_rocm_build"] = bool(torch.version.hip)
    try:
        out["is_available"] = bool(torch.cuda.is_available())
    except Exception as e:  # noqa: BLE001
        out["is_available"] = False
        out["is_available_error"] = f"{type(e).__name__}: {e}"
    out["device_count"] = 0
    out["device_names"] = []
    out["device_archs"] = []
    if out.get("is_available"):
        try:
            out["device_count"] = torch.cuda.device_count()
            for i in range(out["device_count"]):
                props = torch.cuda.get_device_properties(i)
                out["device_names"].append(props.name)
                out["device_archs"].append(getattr(props, "gcnArchName", "") or "")
        except Exception as e:  # noqa: BLE001
            out["device_enumeration_error"] = f"{type(e).__name__}: {e}"
    return out


def real_accelerator(checkout: Path) -> dict:
    """Call the checkout's own has_real_accelerator(), in a SUBPROCESS.

    In-process would prime this process's torch and the answer is order
    sensitive by design -- the function records itself once, before anything can
    spoof. A fresh interpreter is the only way to read it as a test session
    would.
    """
    # Absolute: the subprocess runs with cwd=checkout, so a relative path built
    # from a relative --checkout resolves twice and the import fails with a
    # ModuleNotFoundError that reads like the helper being absent.
    shared = (checkout / "tests" / "_shared").resolve()
    code = ("import sys; sys.path.insert(0, %r)\n"
            "from real_accelerator import has_real_accelerator\n"
            "print(has_real_accelerator())\n" % str(shared))
    if not (shared / "real_accelerator.py").is_file():
        return {"available": False, "reason": f"no {shared}/real_accelerator.py"}
    p = subprocess.run([sys.executable, "-c", code], cwd = str(checkout),
                       capture_output = True, text = True)
    return {"available": p.returncode == 0,
            "value": p.stdout.strip(),
            "stderr_tail": (p.stderr or "")[-1000:]}


def amd_smi() -> dict:
    """amd-smi does not go through HIP, so it is unaffected by a HIP mask."""
    exe = shutil.which("amd-smi")
    if not exe:
        return {"present": False}
    p = subprocess.run([exe, "static", "-g", "0"], capture_output = True,
                       text = True, timeout = 120)
    return {"present": True, "rc": p.returncode, "stdout_head": (p.stdout or "")[:4000]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default = "head")
    ap.add_argument("--checkout", required = True, type = Path)
    ap.add_argument("--out", required = True, type = Path)
    args = ap.parse_args()

    obs = {
        "state": args.state,
        "checkout": str(args.checkout),
        "python": sys.version,
        "executable": sys.executable,
        "mask_env": {k: os.environ.get(k) for k in MASK_VARS},
        "torch": torch_facts(),
        "has_real_accelerator": real_accelerator(args.checkout),
        "amd_smi": amd_smi(),
    }
    args.out.parent.mkdir(parents = True, exist_ok = True)
    args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
    print(json.dumps({k: v for k, v in obs.items() if k != "amd_smi"}, indent = 2))
    smi = obs["amd_smi"]
    print("amd-smi present:", smi.get("present"), "rc:", smi.get("rc"))
    print((smi.get("stdout_head") or "")[:1500])
    return 0


if __name__ == "__main__":
    sys.exit(main())
