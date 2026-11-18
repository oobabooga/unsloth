#!/usr/bin/env python3
"""Probe for unsloth #18285: NF4 dequant cache for QLoRA (unsloth/kernels/dequant_cache.py, kill switch
UNSLOTH_NF4_DEQUANT_CACHE=0). Observes only. Per state:
  0. Windows venv only: when unsloth_zoo is not importable, install the training stack once (the recipe the
     Windows training units use: transformers 5.23.0 / trl 1.18.0 / peft 0.21.3 / bitsandbytes / triton-windows,
     unsloth_zoo main --no-deps). The base runs first, so both states share one environment; versions recorded.
  1. which dequant_cache module (if any) this state's checkout carries
  2. pytest tests/test_nf4_dequant_cache.py when present (head adds it; recorded as absent at the base), with
     `-o addopts= -m "gpu or not gpu"`; per-test outcome + message from junit
  3. pr18285_cell.py, one process per cell: base runs twice (run-to-run spread), head runs with the cache at its
     default (auto) and with UNSLOTH_NF4_DEQUANT_CACHE=0
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
TEST = "tests/test_nf4_dequant_cache.py"
CELLS = {"base": [("run1", None), ("run2", None)], "head": [("auto", None), ("off", "0")]}

ap = argparse.ArgumentParser()
ap.add_argument("--state", required = True)
ap.add_argument("--checkout", required = True)
ap.add_argument("--out", required = True, type = Path)
ap.add_argument("--python", default = sys.executable)
ap.add_argument("--timeout", type = int, default = 1800)
a = ap.parse_args()
ck = Path(a.checkout).resolve()
obs: dict = {"state": a.state, "checkout": str(ck), "platform": sys.platform}
work = Path(tempfile.mkdtemp(prefix = f"p18285_{a.state}_"))


def save():
    a.out.write_text(json.dumps(obs, indent = 1, default = str), encoding = "utf-8")


def env_for(extra = None):
    e = dict(os.environ, PYTHONPATH = os.pathsep.join([str(ck)] + ([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else [])),
             UNSLOTH_DISABLE_AUTO_UPDATES = "1", PYTHONUNBUFFERED = "1", PYTHONIOENCODING = "utf-8", PYTHONUTF8 = "1")
    for k in ("UNSLOTH_NF4_DEQUANT_CACHE", "UNSLOTH_COMPILE_DISABLE", "TORCHDYNAMO_DISABLE", "PYTEST_ADDOPTS", "CI",
              "UNSLOTH_FORCE_FLOAT32", "UNSLOTH_VLLM_STANDBY"):
        e.pop(k, None)
    e.update({k: v for k, v in (extra or {}).items() if v is not None})
    return e


def run(cmd, timeout, **kw):
    try:
        p = subprocess.run(cmd, capture_output = True, text = True, encoding = "utf-8", errors = "replace", timeout = timeout, **kw)
        return p.returncode, p.stdout or "", p.stderr or ""
    except subprocess.TimeoutExpired as e:
        return "timeout", str(e.stdout or "")[-2000:], ""


try:
    obs["commit"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd = ck, capture_output = True, text = True).stdout.strip()
except Exception as e:  # noqa: BLE001
    obs["commit"] = f"ERR {e}"

# 0. environment (Windows venv carries only torch + pytest)
rc, _, _ = run([a.python, "-c", "import unsloth_zoo, trl, peft, bitsandbytes"], 600, env = env_for(), cwd = str(work))
obs["stack_preinstalled"] = rc == 0
if rc != 0 and sys.platform == "win32":
    steps = []
    for cmd in (
        [a.python, "-m", "pip", "install", "-q", "--disable-pip-version-check", "transformers==5.23.0", "trl==1.18.0",
         "peft==0.21.3", "accelerate", "datasets>=3.4.1,!=4.0.*,!=4.1.0,!=4.4.*,!=4.5.0,<5.0.0", "sentencepiece>=0.2.0",
         "protobuf", "huggingface_hub>=0.34.0", "bitsandbytes", "einops", "msgspec", "psutil", "tqdm", "regex", "pillow",
         "packaging", "numpy", "wheel", "requests", "filelock", "typing_extensions", "typer>=0.19.0,<0.29", "rich",
         "pydantic", "pyyaml", "json5>=0.9.25", "nest-asyncio", "structlog>=24.1.0", "click>=8.0", "tyro", "diffusers",
         "certifi", "triton-windows<3.7", "hf_transfer"],
        [a.python, "-m", "pip", "install", "-q", "--disable-pip-version-check", "--no-deps", "unsloth",
         "unsloth_zoo @ git+https://github.com/unslothai/unsloth-zoo@main"],
    ):
        r, so, se = run(cmd, 2400)
        steps.append({"cmd": " ".join(cmd[3:8]) + " ...", "rc": r, "tail": (so + se)[-1500:]})
    obs["stack_install"] = steps
rc, _, _ = run([a.python, "-c", "import pytest"], 300)
if rc != 0:  # the Studio venv (uv-built) may carry neither pytest nor pip
    tries = []
    for cmd in ([a.python, "-m", "pip", "install", "-q", "pytest"], ["uv", "pip", "install", "--python", a.python, "-q", "pytest"]):
        r, so, se = run(cmd, 600)
        tries.append({"cmd": " ".join(cmd[:4]), "rc": r, "tail": (so + se)[-400:]})
        if r == 0:
            break
    obs["pytest_install"] = tries
rc, so, se = run([a.python, "-m", "pip", "list", "--format=json"], 300)
if rc != 0:
    rc, so, se = run(["uv", "pip", "list", "--python", a.python, "--format=json"], 300)
try:
    want = ("torch", "transformers", "trl", "peft", "bitsandbytes", "unsloth", "unsloth_zoo", "unsloth-zoo", "accelerate",
            "datasets", "triton", "triton-windows", "pytorch-triton-rocm")
    obs["pip"] = {d["name"].lower(): d["version"] for d in json.loads(so) if d["name"].lower() in want}
except Exception as e:  # noqa: BLE001
    obs["pip"] = {"error": f"{e!r} {se[-300:]}"}
save()

# 1. this state's dequant_cache module (by path, no unsloth import)
p = ck / "unsloth" / "kernels" / "dequant_cache.py"
obs["dc_file_present"] = p.is_file()

# 2. the PR's tests
obs["test"] = {"present": (ck / TEST).is_file(), "file": TEST}
if obs["test"]["present"]:
    junit = work / "junit.xml"
    cmd = [a.python, "-m", "pytest", "-q", "-rfEs", "-p", "no:cacheprovider", "-o", "addopts=", "-m", "gpu or not gpu",
           "--color=no", f"--junitxml={junit}", "-o", "junit_family=xunit1", TEST]
    obs["test"]["cmd"] = " ".join(cmd[1:])
    r, so, se = run(cmd, a.timeout, cwd = ck, env = env_for({"UNSLOTH_COMPILE_LOCATION": str(work / "ucache_t")}))
    obs["test"].update({"rc": r, "tail": so[-5000:], "stderr_tail": se[-1500:]})
    cases = {}
    try:
        for c in ET.parse(junit).getroot().iter("testcase"):
            tid = f"{c.get('classname')}::{c.get('name')}"
            kid = next((k for k in c if k.tag in ("failure", "error", "skipped")), None)
            outcome = "passed" if kid is None else {"failure": "failed", "error": "error", "skipped": "skipped"}[kid.tag]
            msg = "" if kid is None else ((kid.get("message") or "") + "\n" + (kid.text or ""))
            cases[tid] = {"outcome": outcome, "message": msg[-1500:]}
    except (OSError, ET.ParseError) as e:
        obs["test"]["junit_error"] = repr(e)
    obs["test"]["cases"] = cases
save()

# 3. the 4-bit LoRA SFT smoke
obs["sft"] = {}
for name, cache in CELLS.get(a.state, CELLS["head"]):
    out = work / f"cell_{name}.json"
    cw = work / f"w_{name}"
    cw.mkdir(exist_ok = True)
    r, so, se = run([a.python, "-u", str(HERE / "pr18285_cell.py"), "--work", str(cw), "--out", str(out)], a.timeout,
                    cwd = str(cw), env = env_for({"UNSLOTH_COMPILE_LOCATION": str(cw / "ucache"), "UNSLOTH_NF4_DEQUANT_CACHE": cache}))
    try:
        c = json.loads(out.read_text(encoding = "utf-8")) if out.exists() else {"stage": "no_output"}
    except Exception as e:  # noqa: BLE001
        c = {"stage": "bad_output", "error": repr(e)}
    c.update({"rc": r, "cache_env": cache, "stderr_tail": se[-3000:], "stdout_tail": so[-1500:]})
    obs["sft"][name] = c
    save()
save()
