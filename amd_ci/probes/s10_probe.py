#!/usr/bin/env python3
"""Probe for the ROCm SDK-wheel detection change, one state (an unsloth-zoo checkout; the matching core files ride in
amd_ci/s10_core/<state>/). Observes only; JSON to --out.

Per state:
  gates            s10_cell.py --cell gates: torch build facts + every changed ROCm gate (zoo and core)
  op_fp32          s10_cell.py --cell op, only when torch.version.hip is unset (the SDK-wheel premise): the gfx11
                   fp32 gated-delta guard is what the change re-enables, so this is where base faults / hangs
  pytest           zoo tests (checkout) and core tests (s10_core/<state>, zoo checkout on PYTHONPATH), per file
Linux: one shared venv via zoo_venv.build_venv (AMD gfx1151 index, transformers 5.24.0 for Inkling 2).
Windows: the workflow's AMD SDK torch venv (--python) plus the head checkout's deps and triton-windows.
"""
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

CELL = HERE / "s10_cell.py"
CORE = HERE.parent / "s10_core"
ZOO_TESTS = ["tests/test_rocm_sdk_wheel_detection.py", "tests/test_vendor_fla_rocm_fp32.py",
             "tests/test_flex_rocm_static_compile.py", "tests/test_fused_preference_loss.py"]
CORE_TESTS = ["tests/test_prefix_grouper_rocm_static_flex.py", "tests/test_inkling2_flex.py"]
TRANSFORMERS_PIN = "5.24.0"

ap = argparse.ArgumentParser()
ap.add_argument("--state", required = True)
ap.add_argument("--checkout", required = True)
ap.add_argument("--out", required = True, type = Path)
ap.add_argument("--python", default = sys.executable)
ap.add_argument("--op-timeout", type = int, default = 420)
a = ap.parse_args()
a.out = a.out.resolve()
WIN = sys.platform == "win32"
ck = str(Path(a.checkout).resolve())
head_ck = str(Path(ck).parent / "head")
out_dir = a.out.parent / f"{a.state}_artifacts"
out_dir.mkdir(parents = True, exist_ok = True)
root = Path(os.environ.get("AMD_CI_WORK") or out_dir)
shared = root / "s10_shared"
shared.mkdir(exist_ok = True)
obs: dict = {"state": a.state, "checkout": ck, "platform": sys.platform}


def save():
    a.out.write_text(json.dumps(obs, indent = 1, default = str), encoding = "utf-8")


if a.state not in ("base", "head"):
    obs["_skipped"] = ["everything"]
    save()
    sys.exit(0)
obs["commit"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd = ck, capture_output = True, text = True).stdout.strip()
core = CORE / a.state
obs["core_tree"] = str(core)
obs["core_shas"] = (CORE / "SHAS.txt").read_text(encoding = "utf-8").strip() if (CORE / "SHAS.txt").is_file() else None

if WIN:
    python = a.python
    mark = shared / "stack.json"
    if not mark.exists():
        log = out_dir / "stack.log"
        rc1 = zoo_venv.sh([python, "-m", "pip", "install", "-q", "--disable-pip-version-check", head_ck, "triton-windows",
                           "pytest", "pytest-timeout"], log, timeout = 2400)
        rc_t = zoo_venv.sh([python, "-m", "pip", "install", "-q", "--disable-pip-version-check",
                            f"transformers=={TRANSFORMERS_PIN}"], log, timeout = 900)
        rc2 = zoo_venv.sh([python, "-m", "pip", "install", "-q", "--disable-pip-version-check", "--no-deps", "unsloth"],
                          log, timeout = 900)
        pl = subprocess.run([python, "-m", "pip", "list"], capture_output = True, text = True)
        mark.write_text(json.dumps({"rc_deps": rc1, "rc_transformers": rc_t, "rc_unsloth": rc2, "pip": [ln for ln in pl.stdout.splitlines() if
                        ln.split(" ")[0].lower() in ("torch", "triton-windows", "triton", "transformers", "unsloth")],
                        "log_tail": log.read_text(encoding = "utf-8", errors = "replace")[-2000:]}), encoding = "utf-8")
    obs["venv"] = {"windows_stack": json.loads(mark.read_text(encoding = "utf-8"))}
else:
    obs["venv"] = zoo_venv.build_venv(root / "s10_venv", head_ck, "torch==2.11.0+rocm7.13.0", "amd", TRANSFORMERS_PIN, out_dir)
    if obs["venv"].get("error"):
        save()
        sys.exit(0)
    python = obs["venv"]["python"]
obs["python"] = python
save()

obs["gates"] = zoo_venv.run_cell(python, CELL, ["--cell", "gates", "--core", str(core)], ck, out_dir, f"{a.state}_gates",
                                 timeout = 900)
save()
facts = obs["gates"].get("facts") or {}
if not facts:
    obs["op_fp32"] = {"not_run": "the gates cell recorded no torch facts"}
elif facts.get("hip"):
    obs["op_fp32"] = {"not_run": "torch.version.hip is set on this wheel: the SDK-wheel premise does not hold here"}
elif str(facts.get("triton") or "ERR").startswith("ERR"):
    obs["op_fp32"] = {"not_run": f"Triton not importable: {facts.get('triton')}"}
else:
    obs["op_fp32"] = zoo_venv.run_cell(python, CELL, ["--cell", "op"], ck, out_dir, f"{a.state}_op", timeout = a.op_timeout)
save()

obs["pytest_zoo"] = zoo_venv.run_pytest_per_file(python, ck, ZOO_TESTS, out_dir, f"{a.state}_zoo", 1800)
save()
prev = os.environ.get("PYTHONPATH")
os.environ["PYTHONPATH"] = ck + (os.pathsep + prev if prev else "")
try:
    obs["pytest_core"] = zoo_venv.run_pytest_per_file(python, str(core), CORE_TESTS, out_dir, f"{a.state}_core", 3000)
finally:
    if prev is None:
        os.environ.pop("PYTHONPATH", None)
    else:
        os.environ["PYTHONPATH"] = prev
save()
