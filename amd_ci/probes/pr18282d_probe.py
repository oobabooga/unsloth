#!/usr/bin/env python3
"""Probe for unsloth #18282 (+ DDP flip test, head 1bc9d1edda): one-pass Triton AdamW + stochastic-rounding kernel for bf16 full fine-tuning. Observes only.

Per state:
  1. which bf16_stochastic_rounding module the state's python imports, whether it has the one-pass kernel, the
     torch / triton / HIP versions and the GPU arch
  2. the bf16 SR / AdamW tests present in the state (head adds tests/test_bf16_stochastic_rounding_fused_gpu.py),
     under pytest with `-o addopts= -m "gpu or not gpu"` (GPU tests not deselected); per-test outcome + message
  3. pr18282_cell.py: tiny Qwen3 bf16 FULL fine-tune through SFTTrainer, 30 steps, one subprocess per cell:
     SR on with adamw_torch / adamw_torch_fused / adamw_8bit, and SR off with adamw_torch (scale reference);
     losses, SR engagement counters, the per-device Triton probe result and any fallback message
JSON to --out, never stdout.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
TESTS = ["tests/test_bf16_stochastic_rounding.py", "tests/test_bf16_stochastic_rounding_default_gpu.py",
         "tests/test_bf16_adamw.py", "tests/test_bf16_stochastic_rounding_fused_gpu.py",
         "tests/test_bf16_stochastic_rounding_ddp_gloo.py"]
# The DDP flip test is head-only: the base state runs the head's copy against its own module (the test loads
# <its checkout>/unsloth/optimizers/bf16_stochastic_rounding.py by path), so it must FAIL there.
BORROW = ["tests/test_bf16_stochastic_rounding_ddp_gloo.py"]
CELLS = [("adamw_torch", "1"), ("adamw_torch_fused", "1"), ("adamw_8bit", "1"), ("adamw_torch", "0")]

ap = argparse.ArgumentParser()
ap.add_argument("--state", required = True)
ap.add_argument("--checkout", required = True)
ap.add_argument("--out", required = True, type = Path)
ap.add_argument("--python", default = sys.executable)
ap.add_argument("--timeout", type = int, default = 1800)
a = ap.parse_args()
ck = Path(a.checkout).resolve()
env = dict(os.environ, PYTHONPATH = os.pathsep.join([str(ck)] + ([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else [])),
           UNSLOTH_DISABLE_AUTO_UPDATES = "1", PYTHONUNBUFFERED = "1", PYTHONIOENCODING = "utf-8", PYTHONUTF8 = "1")
for k in ("UNSLOTH_COMPILE_DISABLE", "TORCHDYNAMO_DISABLE", "PYTEST_ADDOPTS", "UNSLOTH_BF16_STOCHASTIC_ROUNDING",
          "UNSLOTH_BF16_SR_FUSED", "CI"):
    env.pop(k, None)
obs: dict = {"state": a.state, "checkout": str(ck)}
work = Path(tempfile.mkdtemp(prefix = f"p18282_{a.state}_"))


def save():
    a.out.write_text(json.dumps(obs, indent = 1, default = str), encoding = "utf-8")


try:
    obs["commit"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd = ck, capture_output = True, text = True).stdout.strip()
except Exception as e:  # noqa: BLE001
    obs["commit"] = f"ERR {e}"

# 1. imports (no unsloth import: the module is loaded by path, torch / triton directly)
imp_out = work / "imports.json"
imp = (
    "import json, sys, importlib.util\n"
    "r = {}\n"
    "try:\n"
    "    import torch\n"
    "    r['torch'] = torch.__version__; r['hip'] = torch.version.hip\n"
    "    r['gpu'] = torch.cuda.is_available()\n"
    "    if r['gpu']:\n"
    "        pr = torch.cuda.get_device_properties(0)\n"
    "        r['device'] = torch.cuda.get_device_name(0); r['arch'] = getattr(pr, 'gcnArchName', None)\n"
    "except Exception as e:\n"
    "    r['torch_error'] = repr(e)[:400]\n"
    "try:\n"
    "    import triton\n"
    "    r['triton'] = triton.__version__; r['triton_file'] = triton.__file__\n"
    "except Exception as e:\n"
    "    r['triton_error'] = repr(e)[:400]\n"
    "try:\n"
    "    p = sys.argv[2] + '/unsloth/optimizers/bf16_stochastic_rounding.py'\n"
    "    s = importlib.util.spec_from_file_location('_sr18282', p); m = importlib.util.module_from_spec(s); s.loader.exec_module(m)\n"
    "    r['sr_file'] = m.__file__; r['has_adamw_kernel'] = hasattr(m, '_adamw_kernel')\n"
    "    if hasattr(m, '_triton_ok') and r.get('gpu'):\n"
    "        r['triton_ok_probe'] = bool(m._triton_ok(torch.device('cuda', 0)))\n"
    "except Exception as e:\n"
    "    r['sr_error'] = repr(e)[:600]\n"
    "open(sys.argv[1], 'w', encoding='utf-8').write(json.dumps(r))\n"
)
try:
    p = subprocess.run([a.python, "-c", imp, str(imp_out), str(ck)], cwd = ck, env = env, capture_output = True, text = True,
                       timeout = 900)
    obs["imports"] = json.loads(imp_out.read_text(encoding = "utf-8")) if imp_out.exists() else {"error": p.stderr[-1500:]}
    obs["imports"]["stdout_tail"] = (p.stdout or "")[-800:]
except subprocess.TimeoutExpired:
    obs["imports"] = {"error": "timeout"}
save()

# 2. the SR tests present in this state (+ the DDP flip test borrowed from the head state)
import hashlib  # noqa: E402
import shutil  # noqa: E402
obs["borrowed"] = {}
for t in BORROW:
    src = ck.parent / "head" / t
    if not (ck / t).exists() and src.exists():
        shutil.copyfile(src, ck / t)
        obs["borrowed"][t] = {"from": str(src), "md5": hashlib.md5(src.read_bytes()).hexdigest()}
present = [t for t in TESTS if (ck / t).exists()]
obs["test"] = {"present": present, "absent": [t for t in TESTS if t not in present]}
junit = work / "junit.xml"
cmd = [a.python, "-m", "pytest", "-q", "-rfEs", "-p", "no:cacheprovider", "-o", "addopts=", "-m", "gpu or not gpu",
       "--color=no", f"--junitxml={junit}", "-o", "junit_family=xunit1", *present]
obs["test"]["cmd"] = " ".join(cmd)
try:
    p = subprocess.run(cmd, cwd = ck, env = dict(env, UNSLOTH_COMPILE_LOCATION = str(work / "ucache_t")), capture_output = True,
                       text = True, timeout = a.timeout)
    obs["test"]["rc"] = p.returncode
    obs["test"]["tail"] = (p.stdout or "")[-5000:]
    obs["test"]["stderr_tail"] = (p.stderr or "")[-1500:]
except subprocess.TimeoutExpired as e:
    obs["test"]["rc"] = "timeout"
    obs["test"]["tail"] = (e.stdout.decode("utf-8", "replace") if isinstance(e.stdout, bytes) else (e.stdout or ""))[-4000:]
cases = {}
try:
    for c in ET.parse(junit).getroot().iter("testcase"):
        tid = f"{c.get('file') or c.get('classname')}::{c.get('name')}"
        kid = next((k for k in c if k.tag in ("failure", "error", "skipped")), None)
        outcome = "passed" if kid is None else {"failure": "failed", "error": "error", "skipped": "skipped"}[kid.tag]
        msg = "" if kid is None else ((kid.get("message") or "") + "\n" + (kid.text or ""))
        cases[tid] = {"outcome": outcome, "message": msg[-1500:]}
except (OSError, ET.ParseError) as e:
    obs["test"]["junit_error"] = repr(e)
obs["test"]["cases"] = cases
save()

# 3. tiny Qwen3 bf16 full fine-tune smoke, one subprocess per cell
obs["sft"] = {}
for optim, sr in CELLS:
    key = f"{optim}_sr{sr}"
    out = work / f"cell_{key}.json"
    cw = work / f"w_{key}"
    cw.mkdir(exist_ok = True)
    try:
        p = subprocess.run([a.python, "-u", str(HERE / "pr18282_cell.py"), "--optim", optim, "--sr", sr, "--work", str(cw),
                            "--out", str(out)], cwd = cw, env = dict(env, UNSLOTH_COMPILE_LOCATION = str(cw / "ucache")),
                           capture_output = True, text = True, timeout = a.timeout)
        r = json.loads(out.read_text(encoding = "utf-8")) if out.exists() else {"stage": "no_output"}
        r["rc"] = p.returncode
        r["stderr_tail"] = (p.stderr or "")[-2000:]
        r["stdout_tail"] = (p.stdout or "")[-1500:]
    except subprocess.TimeoutExpired:
        r = {"stage": "timeout"}
    obs["sft"][key] = r
    save()
save()
