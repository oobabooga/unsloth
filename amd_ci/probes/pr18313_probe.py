#!/usr/bin/env python3
"""Probe for unsloth #18313 (ROCm: packed rows on torch's flash varlen after a per-device self-check).
Observes only; JSON to --out, never stdout. Per state:
  0. Windows venv without the training stack: install it once (base runs first, so both states share it).
  1. pr18313_kernel.py (one process): route verdict + cost, self-check robustness (head), kernel parity / timing
     (head only, the kernels are the same torch build in both states), the real wrapper on packed rows.
     Full unsloth import when the env has the stack, else stub mode (hf_packed_attention + flash_impl only).
  2. pytest tests/test_hf_packed_attention.py on the real device (`-o addopts= -m "gpu or not gpu"`), per test id.
  3. pr18313_train.py cells (base and head only): qwen35 (3 gated delta + 1 full) and gemma3 (3 sliding window
     512 + 1 full) LoRA SFT, padded once and packed twice (A/A). Weights built once, shared by both states.
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
TEST = "tests/test_hf_packed_attention.py"
ap = argparse.ArgumentParser()
ap.add_argument("--state", required = True)
ap.add_argument("--checkout", required = True)
ap.add_argument("--out", required = True, type = Path)
ap.add_argument("--python", default = sys.executable)
ap.add_argument("--timeout", type = int, default = 1800)
ap.add_argument("--steps", type = int, default = 12)
a = ap.parse_args()
ck = Path(a.checkout).resolve()
obs: dict = {"state": a.state, "checkout": str(ck), "platform": sys.platform}
root = Path(os.environ.get("AMD_CI_WORK") or tempfile.gettempdir())
work = Path(tempfile.mkdtemp(prefix = f"p18313_{a.state}_", dir = str(root)))
models = root / "p18313_models"
models.mkdir(exist_ok = True)
WIN = sys.platform == "win32"


def save():
    a.out.write_text(json.dumps(obs, indent = 1, default = str), encoding = "utf-8")


def env_for(extra = None):
    pp = [str(ck)] + ([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else [])
    e = dict(os.environ, PYTHONPATH = os.pathsep.join(pp), UNSLOTH_DISABLE_AUTO_UPDATES = "1",
             PYTHONUNBUFFERED = "1", PYTHONIOENCODING = "utf-8", PYTHONUTF8 = "1")
    for k in ("UNSLOTH_COMPILE_DISABLE", "TORCHDYNAMO_DISABLE", "PYTEST_ADDOPTS", "CI", "UNSLOTH_FORCE_FLOAT32",
              "UNSLOTH_HF_PACKED_VARLEN", "UNSLOTH_TORCH_VARLEN", "UNSLOTH_HF_PACKED_EFFICIENT"):
        e.pop(k, None)
    e.update({k: v for k, v in (extra or {}).items() if v is not None})
    return e


def run(cmd, timeout, **kw):
    try:
        p = subprocess.run(cmd, capture_output = True, text = True, encoding = "utf-8", errors = "replace",
                           timeout = timeout, **kw)
        return p.returncode, p.stdout or "", p.stderr or ""
    except subprocess.TimeoutExpired as e:
        return "timeout", str(e.stdout or "")[-2000:], str(e.stderr or "")[-2000:]


def load(p: Path, rc, so, se):
    try:
        c = json.loads(p.read_text(encoding = "utf-8")) if p.exists() else {"stage": "no_output"}
    except Exception as e:  # noqa: BLE001
        c = {"stage": "bad_output", "error": repr(e)}
    c.update({"rc": rc, "stderr_tail": se[-3000:], "stdout_tail": so[-1500:]})
    return c


for key, rev in (("commit", "HEAD"), ("tree", "HEAD^{tree}")):
    try:
        obs[key] = subprocess.run(["git", "rev-parse", rev], cwd = ck, capture_output = True, text = True).stdout.strip()
    except Exception as e:  # noqa: BLE001
        obs[key] = f"ERR {e}"

# 0. environment
rc, _, _ = run([a.python, "-c", "import unsloth_zoo, trl, peft"], 600, env = env_for(), cwd = str(work))
obs["stack_preinstalled"] = rc == 0
if rc != 0 and WIN:
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
if rc != 0:
    for cmd in ([a.python, "-m", "pip", "install", "-q", "pytest"], ["uv", "pip", "install", "--python", a.python, "-q", "pytest"]):
        if run(cmd, 600)[0] == 0:
            break
rc, so, se = run([a.python, "-m", "pip", "list", "--format=json"], 300)
if rc != 0:
    rc, so, se = run(["uv", "pip", "list", "--python", a.python, "--format=json"], 300)
try:
    want = ("torch", "transformers", "trl", "peft", "unsloth", "unsloth_zoo", "unsloth-zoo", "accelerate",
            "datasets", "triton", "triton-windows", "pytorch-triton-rocm", "flash-attn", "flash_attn", "xformers",
            "flash-linear-attention", "fla-core", "causal-conv1d")
    obs["pip"] = {d["name"].lower(): d["version"] for d in json.loads(so) if d["name"].lower() in want}
except Exception as e:  # noqa: BLE001
    obs["pip"] = {"error": f"{e!r} {se[-300:]}"}
rc, so, se = run([a.python, "-c", "import unsloth, sys; print('OK', unsloth.__file__)"], 900, env = env_for(), cwd = str(work))
obs["unsloth_importable"] = rc == 0 and "OK" in so
obs["unsloth_import_tail"] = (so + se)[-1500:]
save()

# 1. kernel cell
mode = "full" if obs["unsloth_importable"] else "stub"
out = work / "kernel.json"
r, so, se = run([a.python, "-u", str(HERE / "pr18313_kernel.py"), "--checkout", str(ck), "--out", str(out), "--mode", mode,
                 "--kernels", "1" if a.state == "head" else "0"], a.timeout, cwd = str(work),
                env = env_for({"UNSLOTH_COMPILE_LOCATION": str(work / "ucache_k")}))
obs["kernel"] = load(out, r, so, se)
if obs["kernel"].get("stage") != "done" and mode == "full":
    # full import broke mid-way: the stub view still answers the route question
    out2 = work / "kernel_stub.json"
    r, so, se = run([a.python, "-u", str(HERE / "pr18313_kernel.py"), "--checkout", str(ck), "--out", str(out2),
                     "--mode", "stub", "--kernels", "1" if a.state == "head" else "0"], a.timeout, cwd = str(work),
                    env = env_for())
    obs["kernel_stub"] = load(out2, r, so, se)
save()

# 2. pytest on the real device (head also runs the reviewer's fixed copy of the test file, as "pytest_fixed")
def pytest_cases(key, label):
    junit = work / f"junit_{label}.xml"
    cmd = [a.python, "-m", "pytest", "-q", "-rfEs", "-p", "no:cacheprovider", "-o", "addopts=", "-m", "gpu or not gpu",
           "--color=no", f"--junitxml={junit}", "-o", "junit_family=xunit1", TEST]
    r, so, se = run(cmd, a.timeout, cwd = ck, env = env_for({"UNSLOTH_COMPILE_LOCATION": str(work / f"ucache_{label}")}))
    t = {"rc": r, "tail": so[-4000:], "stderr_tail": se[-1500:]}
    cases = {}
    try:
        for c in ET.parse(junit).getroot().iter("testcase"):
            tid = f"{c.get('classname')}::{c.get('name')}"
            kid = next((k for k in c if k.tag in ("failure", "error", "skipped")), None)
            outcome = "passed" if kid is None else {"failure": "failed", "error": "error", "skipped": "skipped"}[kid.tag]
            msg = "" if kid is None else ((kid.get("message") or "") + "\n" + (kid.text or ""))
            cases[tid] = {"outcome": outcome, "message": msg[-800:]}
    except (OSError, ET.ParseError) as e:
        t["junit_error"] = repr(e)
    t["cases"] = cases
    obs[key] = t
    save()


if obs["unsloth_importable"]:
    pytest_cases("pytest", "own")
    fixed = HERE / "pr18313_tests" / "test_hf_packed_attention.py"
    if a.state == "head" and fixed.is_file():
        import shutil

        own = (ck / TEST).read_bytes()
        shutil.copy(fixed, ck / TEST)
        try:
            pytest_cases("pytest_fixed", "fixed")
        finally:
            (ck / TEST).write_bytes(own)

# 3. training cells
if a.state in ("base", "head") and obs["unsloth_importable"]:
    obs["train"] = {}
    arches = ("qwen35",)  # gemma3 rows do not reach this wrapper under Unsloth (B200 smoke: 0 fast, 0 compiled)
    for arch in arches:
        mdir = models / arch
        if not (mdir / "config.json").exists():
            bo = work / f"build_{arch}.json"
            r, so, se = run([a.python, "-u", str(HERE / "pr18313_train.py"), "--arch", arch, "--model-dir", str(mdir),
                             "--build", "--out", str(bo)], 1800, cwd = str(work), env = env_for())
            obs.setdefault("build", {})[arch] = load(bo, r, so, se)
            save()
        for name, packed in (("padded", 0), ("packed", 1), ("packed2", 1)):
            cw = work / f"w_{arch}_{name}"
            cw.mkdir(exist_ok = True)
            co = work / f"cell_{arch}_{name}.json"
            r, so, se = run([a.python, "-u", str(HERE / "pr18313_train.py"), "--arch", arch, "--model-dir", str(mdir),
                             "--packed", str(packed), "--steps", str(a.steps), "--work", str(cw), "--out", str(co)],
                            a.timeout, cwd = str(cw),
                            env = env_for({"UNSLOTH_COMPILE_LOCATION": str(cw / "ucache"),
                                           "UNSLOTH_EXPERIMENTAL_HYBRID_PACKING": "1" if packed else None}))
            obs["train"][f"{arch}_{name}"] = load(co, r, so, se)
            save()
save()
