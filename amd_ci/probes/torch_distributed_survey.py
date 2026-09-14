#!/usr/bin/env python3
"""Survey: which torch builds actually lack torch._C._distributed_c10d?

PR #10819 caps accelerate on all of Windows, because a PEP 508 marker cannot say "ROCm".
Whether that blast radius is right turns on a question nobody has measured: is the defect
a property of Windows, or only of AMD's Windows ROCm wheels?

The offline hint is suggestive but not decisive. AMD's Windows wheel ships no uv.dll,
gloo or tensorpipe in torch/lib, and PyPI's win_amd64 wheel ships uv.dll. But that marker
only reads on Windows, because on Linux and macOS libuv and gloo are linked into
libtorch_cpu rather than shipped beside it, and those wheels look equally bare while
importing c10d perfectly well. So the libs cannot answer it; an import can.

Installs each build into its own venv and, for each, records whether
`torch._C._distributed_c10d` imports and whether accelerate 1.15.0 can start a trainer
against it. Observes only; the conclusion is for whoever reads the table.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

_UTF8 = {"PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}

# label -> (extra pip args, requirement). One entry per torch flavour reachable here.
BUILDS = {
    "pypi-default": ([], "torch"),
    "amd-rocm-gfx1151": (["--index-url", "https://repo.amd.com/rocm/whl/gfx1151/",
                          "--extra-index-url", "https://pypi.org/simple"],
                         "torch==2.9.1+rocm7.13.0"),
    "pypi-cpu": (["--index-url", "https://download.pytorch.org/whl/cpu",
                  "--extra-index-url", "https://pypi.org/simple"], "torch"),
}

_CHECK = r"""
import json
out = {}
try:
    import torch
    out["torch"] = torch.__version__
    out["hip"] = getattr(getattr(torch, "version", None), "hip", None)
except Exception as e:
    out["torch_error"] = "%s: %s" % (type(e).__name__, e)
try:
    import torch._C._distributed_c10d
    out["has_c10d"] = True
except Exception as e:
    out["has_c10d"] = False
    out["c10d_error"] = "%s: %s" % (type(e).__name__, e)
try:
    import torch.distributed as d
    out["dist_available"] = bool(d.is_available())
except Exception as e:
    out["dist_available"] = False
try:
    import accelerate, torch.nn as nn
    out["accelerate"] = accelerate.__version__
    accelerate.Accelerator(cpu = True).prepare_model(nn.Linear(4, 4))
    out["prepare_ok"] = True
except Exception as e:
    out["prepare_ok"] = False
    out["prepare_error"] = "%s: %s" % (type(e).__name__, e)
print(json.dumps(out))
"""


def _run(cmd, timeout = 3600, **kw):
    env = dict(kw.pop("env", None) or os.environ, **_UTF8)
    return subprocess.run(cmd, capture_output = True, timeout = timeout, env = env,
                          encoding = "utf-8", errors = "replace", **kw)


def survey_one(root: Path, label: str, extra: list[str], spec: str) -> dict:
    venv = root / label
    py = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    rec: dict = {"label": label, "spec": spec}
    r = _run([sys.executable, "-m", "venv", str(venv)], timeout = 600)
    if r.returncode != 0:
        rec["error"] = f"venv: {(r.stderr or r.stdout)[-300:]}"
        return rec
    r = _run([str(py), "-m", "pip", "install", "-q", "--upgrade", "pip"], timeout = 900)
    r = _run([str(py), "-m", "pip", "install", "-q", *extra, spec])
    if r.returncode != 0:
        rec["error"] = f"torch: {(r.stderr or r.stdout)[-400:]}"
        return rec
    # accelerate 1.15.0 exactly: the version whose unguarded model_has_dtensor is the bug.
    r = _run([str(py), "-m", "pip", "install", "-q", "accelerate==1.15.0"])
    if r.returncode != 0:
        rec["error"] = f"accelerate: {(r.stderr or r.stdout)[-400:]}"
        return rec
    r = _run([str(py), "-c", _CHECK], timeout = 900)
    try:
        rec.update(json.loads((r.stdout or "").strip().splitlines()[-1]))
    except Exception:  # noqa: BLE001
        rec["error"] = f"check produced no JSON: {(r.stderr or r.stdout)[-400:]}"
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--only", nargs = "*", default = None)
    args = ap.parse_args()

    root = Path(os.environ.get("RUNNER_TEMP", ".")) / "torch_survey"
    root.mkdir(parents = True, exist_ok = True)
    wanted = args.only or list(BUILDS)

    out = {"platform": sys.platform, "python": sys.version.split()[0], "builds": []}
    for label in wanted:
        if label not in BUILDS:
            continue
        extra, spec = BUILDS[label]
        try:
            out["builds"].append(survey_one(root, label, extra, spec))
        except Exception as e:  # noqa: BLE001
            out["builds"].append({"label": label, "error": f"{type(e).__name__}: {e}"})

    args.out.parent.mkdir(parents = True, exist_ok = True)
    args.out.write_text(json.dumps(out, indent = 2), encoding = "utf-8")
    for b in out["builds"]:
        print(f"{b.get('label'):22s} torch={b.get('torch')} has_c10d={b.get('has_c10d')} "
              f"prepare_ok={b.get('prepare_ok')} {b.get('error', '')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
