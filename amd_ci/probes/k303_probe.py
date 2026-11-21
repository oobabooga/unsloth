#!/usr/bin/env python3
"""Probe for K3-03 (block swap lossless bf16 fetch codec, unsloth-zoo). Per state: the device seen by torch,
whether Triton imports, then pytest per file over tests/test_block_swap_compress.py (absent at base) and
tests/test_block_swap.py. Linux: one shared venv via zoo_venv.build_venv (--venv, --head-checkout). Windows: the
workflow's ROCm torch venv (--python) plus psutil / numpy. Observes only; JSON to --out."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import zoo_venv  # noqa: E402

TESTS = ["tests/test_block_swap_compress.py", "tests/test_block_swap.py"]
ap = argparse.ArgumentParser()
ap.add_argument("--state", required = True)
ap.add_argument("--checkout", required = True)
ap.add_argument("--out", required = True, type = Path)
ap.add_argument("--python", default = sys.executable)
ap.add_argument("--venv", type = Path, default = None)
ap.add_argument("--head-checkout", default = None)
a = ap.parse_args()
a.out = a.out.resolve()
WIN = sys.platform == "win32"
ck = str(Path(a.checkout).resolve())
out_dir = a.out.parent / f"{a.state}_artifacts"
out_dir.mkdir(parents = True, exist_ok = True)
root = Path(os.environ.get("AMD_CI_WORK") or out_dir)
shared = root / "k303_shared"
shared.mkdir(exist_ok = True)
obs: dict = {"state": a.state, "checkout": ck, "platform": sys.platform}
if a.state not in ("base", "head"):
    obs["_skipped"] = ["everything"]
    a.out.write_text(json.dumps(obs), encoding = "utf-8")
    sys.exit(0)
obs["commit"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd = ck, capture_output = True, text = True).stdout.strip()


def save():
    a.out.write_text(json.dumps(obs, indent = 1, default = str), encoding = "utf-8")


if WIN:
    python = a.python
    mark = shared / "stack.json"
    if not mark.exists():
        rc = zoo_venv.sh([python, "-m", "pip", "install", "-q", "--disable-pip-version-check", "psutil", "numpy", "pytest",
                          "pytest-timeout"], out_dir / "stack.log", timeout = 1800)
        mark.write_text(json.dumps({"rc": rc}), encoding = "utf-8")
    obs["venv"] = {"windows_stack": json.loads(mark.read_text(encoding = "utf-8"))}
else:
    obs["venv"] = zoo_venv.build_venv(a.venv.resolve(), a.head_checkout, "torch==2.11.0+rocm7.13.0", "amd", None, out_dir)
    if obs["venv"].get("error"):
        save()
        sys.exit(0)
    python = obs["venv"]["python"]
obs["python"] = python
probe = ("import json, torch\n"
         "d = {'torch': torch.__version__, 'hip': torch.version.hip, 'cuda_available': torch.cuda.is_available()}\n"
         "if d['cuda_available']:\n"
         "    p = torch.cuda.get_device_properties(0)\n"
         "    d['name'] = p.name; d['arch'] = getattr(p, 'gcnArchName', None)\n"
         "try:\n"
         "    import triton; d['triton'] = triton.__version__\n"
         "except Exception as e:\n"
         "    d['triton'] = 'ERR ' + repr(e)[:200]\n"
         "print('AMDCI' + json.dumps(d))\n")
p = subprocess.run([python, "-c", probe], cwd = ck, env = zoo_venv.state_env(ck), capture_output = True, text = True,
                   timeout = 600)
line = next((ln for ln in (p.stdout or "").splitlines() if ln.startswith("AMDCI")), None)
obs["device"] = json.loads(line[5:]) if line else {"error": (p.stdout + p.stderr)[-2000:]}
save()
obs["pytest"] = zoo_venv.run_pytest_per_file(python, ck, TESTS, out_dir, a.state, 2400)
for k in ("passed", "failed", "errors", "skipped", "absent_at_this_state", "rc", "n_passed", "n_failed", "n_skipped"):
    obs[k] = obs["pytest"].get(k)
save()
