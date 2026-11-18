#!/usr/bin/env python3
"""Probe for unsloth #17780: native FSDP2 resume of expert LoRA saved in vanilla PEFT layout. Observes only.

Per state (same bytes for every state):
  1. which fsdp2_native / unsloth_zoo this state's python imports, and whether each has the helper
  2. the PR's test (shipped as pr17780_test_resume_expert.py, sha256 recorded) dropped into the state's
     tests/ and run under pytest: 2 gloo ranks on CPU, modes noswap / swap / legacy; per-test outcome + message
  3. pr17780_cell.py: single-GPU tiny Qwen3-MoE LoRA SFT, save at step 3, resume from checkpoint-3 (non-FSDP2 path)
JSON to --out, never stdout.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEST_SRC = HERE / "pr17780_test_resume_expert.py"
TEST_NAME = "test_fsdp2_native_resume_expert_factor_swap_gloo.py"

ap = argparse.ArgumentParser()
ap.add_argument("--state", required = True)
ap.add_argument("--checkout", required = True)
ap.add_argument("--out", required = True, type = Path)
ap.add_argument("--python", default = sys.executable)
ap.add_argument("--timeout", type = int, default = 1500)
a = ap.parse_args()
ck = Path(a.checkout).resolve()
env = dict(os.environ, PYTHONPATH = os.pathsep.join([str(ck)] + ([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else [])),
           UNSLOTH_DISABLE_AUTO_UPDATES = "1", PYTHONUNBUFFERED = "1", PYTHONIOENCODING = "utf-8", PYTHONUTF8 = "1")
for k in ("UNSLOTH_COMPILE_DISABLE", "TORCHDYNAMO_DISABLE", "PYTEST_ADDOPTS", "UNSLOTH_MOE_LORA_B_LAYOUT"):
    env.pop(k, None)
obs: dict = {"state": a.state, "checkout": str(ck)}
work = Path(tempfile.mkdtemp(prefix = f"p17780_{a.state}_"))


def save():
    a.out.write_text(json.dumps(obs, indent = 1, default = str), encoding = "utf-8")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


try:
    obs["commit"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd = ck, capture_output = True, text = True).stdout.strip()
except Exception as e:  # noqa: BLE001
    obs["commit"] = f"ERR {e}"

# 1. imports
imp_out = work / "imports.json"
imp = (
    "import json, sys\n"
    "r = {}\n"
    "try:\n"
    "    import unsloth_zoo, importlib.metadata as m\n"
    "    r['zoo_version'] = m.version('unsloth_zoo')\n"
    "    import unsloth_zoo.temporary_patches.moe_lora_factor_swap as s\n"
    "    r['zoo_has_from_vanilla'] = hasattr(s, '_from_vanilla')\n"
    "except Exception as e:\n"
    "    r['zoo_error'] = repr(e)[:400]\n"
    "try:\n"
    "    import unsloth.models.fsdp2_native as f\n"
    "    r['fsdp2_native_file'] = f.__file__\n"
    "    r['has_peft_trained_layout'] = hasattr(f, '_peft_trained_layout')\n"
    "except Exception as e:\n"
    "    r['fsdp2_error'] = repr(e)[:400]\n"
    "open(sys.argv[1], 'w', encoding='utf-8').write(json.dumps(r))\n"
)
try:
    p = subprocess.run([a.python, "-c", imp, str(imp_out)], cwd = ck, env = env, capture_output = True, text = True,
                       timeout = 600)
    obs["imports"] = json.loads(imp_out.read_text(encoding = "utf-8")) if imp_out.exists() else {"error": p.stderr[-1500:]}
except subprocess.TimeoutExpired:
    obs["imports"] = {"error": "timeout"}
save()

# 2. the PR's test, identical bytes in every state
dst = ck / "tests" / TEST_NAME
obs["test"] = {"shipped_sha256": sha(TEST_SRC), "state_had_test": dst.exists(),
               "state_test_sha256": sha(dst) if dst.exists() else None}
shutil.copyfile(TEST_SRC, dst)
junit = work / "junit.xml"
cmd = [a.python, "-m", "pytest", "-q", "-rfEs", "-p", "no:cacheprovider", "-o", "addopts=", "--color=no",
       f"--junitxml={junit}", "-o", "junit_family=xunit1", f"tests/{TEST_NAME}"]
obs["test"]["cmd"] = " ".join(cmd)
try:
    p = subprocess.run(cmd, cwd = ck, env = env, capture_output = True, text = True, timeout = a.timeout)
    obs["test"]["rc"] = p.returncode
    obs["test"]["tail"] = (p.stdout or "")[-4000:]
    obs["test"]["stderr_tail"] = (p.stderr or "")[-1500:]
except subprocess.TimeoutExpired as e:
    obs["test"]["rc"] = "timeout"
    obs["test"]["tail"] = (e.stdout.decode("utf-8", "replace") if isinstance(e.stdout, bytes) else (e.stdout or ""))[-4000:]
cases = {}
try:
    for c in ET.parse(junit).getroot().iter("testcase"):
        tid = f"tests/{TEST_NAME}::{c.get('name')}"
        kid = next((k for k in c if k.tag in ("failure", "error", "skipped")), None)
        outcome = "passed" if kid is None else {"failure": "failed", "error": "error", "skipped": "skipped"}[kid.tag]
        msg = "" if kid is None else ((kid.get("message") or "") + "\n" + (kid.text or ""))
        cases[tid] = {"outcome": outcome, "message": msg[-1500:]}
except (OSError, ET.ParseError) as e:
    obs["test"]["junit_error"] = repr(e)
obs["test"]["cases"] = cases
save()

# 3. non-FSDP2 single-GPU SFT save + resume smoke
cenv = dict(env, UNSLOTH_COMPILE_LOCATION = str(work / "ucache"))
snap = work / "snap.pt"
obs["sft"] = {}
for phase in ("train", "resume"):
    out = work / f"cell_{phase}.json"
    try:
        p = subprocess.run([a.python, "-u", str(HERE / "pr17780_cell.py"), "--phase", phase, "--work", str(work),
                            "--snap", str(snap), "--out", str(out)], cwd = work, env = cenv, capture_output = True,
                           text = True, timeout = a.timeout)
        r = json.loads(out.read_text(encoding = "utf-8")) if out.exists() else {"stage": "no_output"}
        r["rc"] = p.returncode
        r["stderr_tail"] = (p.stderr or "")[-2000:]
    except subprocess.TimeoutExpired:
        r = {"stage": "timeout"}
    obs["sft"][phase] = r
    save()
    if r.get("stage") != "done":
        break
save()
