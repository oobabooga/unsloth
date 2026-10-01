#!/usr/bin/env python3
"""Probe: does the installer's post-repair step leave a CUDA xFormers beside a ROCm torch?

Observes only. Installs PyPI's xformers (the CUDA 12.8 build, the only Linux x86_64 wheel) into a
private --target dir, records whether its C++ extension loads under this torch, then runs exactly
the `_evict_xformers*` calls this state's install_python_stack() makes in step 13 (read from the
source, so each state runs its own wiring) with the uninstall recorded instead of performed.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

CHILD = r'''
import ast, contextlib, inspect, io, json, os, sys
target, checkout = sys.argv[1], sys.argv[2]
sys.path[:0] = [target, f"{checkout}/studio", f"{checkout}/studio/backend", checkout]
obs = {}
import torch
obs["torch"] = torch.__version__
obs["hip"] = getattr(torch.version, "hip", None) or ""
obs["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
import importlib.metadata as md
obs["xformers_version"] = md.version("xformers")
obs["xformers_file"] = __import__("importlib.util").util.find_spec("xformers").origin
try:
    import json as _j
    obs["xformers_built_for"] = _j.load(open(os.path.join(target, "xformers", "cpp_lib.json")))["version"]["torch"]
except Exception as e:
    obs["xformers_built_for"] = f"unreadable: {e}"
try:
    import xformers
    from xformers import _cpp_lib
    exc = getattr(_cpp_lib, "_cpp_library_load_exception", "attr-missing")
    obs["cpp_load_error"] = None if exc is None else str(exc)[:300]
except Exception as e:
    obs["cpp_load_error"] = f"import failed: {type(e).__name__}: {e}"[:300]
import install_python_stack as m
src = inspect.getsource(m.install_python_stack)
step = src.split('_progress(_torch_step_label("final"))', 1)[1].split("# 13w.", 1)[0]
calls = []
for line in step.splitlines():
    s = line.strip()
    if s.startswith("_evict_xformers"):
        calls.append(s)
obs["step13_calls"] = calls
removed = []
m._uninstall_distribution = lambda name: removed.append(name) or True
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    for c in calls:
        eval(compile(ast.parse(c, mode="eval"), "<step13>", "eval"), vars(m))
obs["uninstall_requested"] = removed
obs["installer_output"] = buf.getvalue()[-800:]
print("PROBE_JSON " + json.dumps(obs))
'''


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--python", default = sys.executable)
    ap.add_argument("--xformers", default = "xformers==0.0.35")
    args = ap.parse_args()
    obs: dict = {"state": args.state}
    with tempfile.TemporaryDirectory() as target:
        pip = subprocess.run(
            [args.python, "-m", "pip", "install", "-q", "--no-deps", "--target", target,
             "--index-url", "https://pypi.org/simple", args.xformers],
            capture_output = True, text = True, timeout = 900,
        )
        obs["pip_rc"] = pip.returncode
        if pip.returncode != 0:
            obs["error"] = f"pip install failed: {pip.stderr[-500:]}"
        else:
            env = dict(__import__("os").environ, UNSLOTH_IS_PRESENT = "1")
            child = subprocess.run(
                [args.python, "-c", CHILD, target, args.checkout],
                capture_output = True, text = True, timeout = 900, env = env,
            )
            obs["child_rc"] = child.returncode
            line = next((l for l in child.stdout.splitlines() if l.startswith("PROBE_JSON ")), None)
            if line:
                obs.update(json.loads(line[len("PROBE_JSON "):]))
            else:
                obs["error"] = f"no probe output; stderr: {child.stderr[-800:]}"
    args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
