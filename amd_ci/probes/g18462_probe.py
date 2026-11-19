#!/usr/bin/env python3
"""Probe for unsloth #18462 (block compile: keep torch's alignment guard for the mem-efficient SDPA backward on
CUDA torch < 2.14). CUDA-only change, so on gfx1151 base and head are expected to behave identically.
Observes only; JSON to --out. Per state:
  A. pytest tests/test_block_sdpa_efficient_backward_guard.py (head only: absent at base) and tests/test_block_compile.py,
     -o addopts= (GPU tests included); per-test outcome from junit.
  B. tiny random Gemma-4 (26B-A4B layout, real head dims 256 / 512) LoRA bf16, 2 steps on a right-padded batch at
     padded lengths 37, 38, 36 (+ 40 control), via train_repro.py; outcome, losses, grad norms, block-compiled classes.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--state", required = True)
ap.add_argument("--checkout", required = True)
ap.add_argument("--out", required = True, type = Path)
ap.add_argument("--python", default = sys.executable)
a = ap.parse_args()
ck = Path(a.checkout).resolve()
WIN = sys.platform == "win32"
obs: dict = {"state": a.state, "checkout": str(ck), "platform": sys.platform}
root = Path(os.environ.get("AMD_CI_WORK") or tempfile.gettempdir())
work = Path(tempfile.mkdtemp(prefix = f"g18462_{a.state}_", dir = str(root)))
shared = root / "g18462_shared"
shared.mkdir(exist_ok = True)


def save():
    a.out.write_text(json.dumps(obs, indent = 1, default = str), encoding = "utf-8")


def env_for(extra = None):
    pp = [str(ck)] + ([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else [])
    e = dict(os.environ, PYTHONPATH = os.pathsep.join(pp), UNSLOTH_DISABLE_AUTO_UPDATES = "1",
             PYTHONUNBUFFERED = "1", PYTHONIOENCODING = "utf-8", PYTHONUTF8 = "1", HIP_VISIBLE_DEVICES = "0")
    for k in ("UNSLOTH_COMPILE_DISABLE", "TORCHDYNAMO_DISABLE", "PYTEST_ADDOPTS", "CI", "UNSLOTH_BLOCK_COMPILE",
              "UNSLOTH_BLOCK_SDPA_ALIGN_GUARD", "AMD_SERIALIZE_KERNEL", "HIP_LAUNCH_BLOCKING"):
        e.pop(k, None)
    e.update({k: v for k, v in (extra or {}).items() if v is not None})
    return e


def run(cmd, timeout, **kw):
    # Popen + whole-tree kill: on Windows a plain timeout leaves grandchildren holding the pipes.
    p = subprocess.Popen(cmd, stdout = subprocess.PIPE, stderr = subprocess.PIPE, stdin = subprocess.DEVNULL,
                         text = True, encoding = "utf-8", errors = "replace", **kw)
    try:
        so, se = p.communicate(timeout = timeout)
        return p.returncode, so or "", se or ""
    except subprocess.TimeoutExpired:
        if WIN:
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(p.pid)], capture_output = True)
        p.kill()
        try:
            so, se = p.communicate(timeout = 60)
        except subprocess.TimeoutExpired:
            so, se = "", "pipes still held after tree kill"
        return "timeout", str(so or "")[-3000:], str(se or "")[-3000:]


for key, rev in (("commit", "HEAD"), ("tree", "HEAD^{tree}")):
    try:
        obs[key] = subprocess.run(["git", "rev-parse", rev], cwd = ck, capture_output = True, text = True).stdout.strip()
    except Exception as e:  # noqa: BLE001
        obs[key] = f"ERR {e}"

# 0. environment (shared by both states: base runs first)
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
         "certifi", "triton-windows<3.7", "hf_transfer", "torchvision"],
        [a.python, "-m", "pip", "install", "-q", "--disable-pip-version-check", "--no-deps", "unsloth",
         "unsloth_zoo @ git+https://github.com/unslothai/unsloth-zoo@main"],
    ):
        r, so, se = run(cmd, 2400)
        steps.append({"cmd": " ".join(cmd[3:8]) + " ...", "rc": r, "tail": (so + se)[-1500:]})
    obs["stack_install"] = steps
# Gemma-4 needs a transformers that has it; install once into the shared venv if missing (same for both arms).
gm = shared / "gemma4_tf.json"
if not gm.exists():
    r, so, se = run([a.python, "-c", "import transformers.models.gemma4, transformers; print(transformers.__version__)"],
                    300, env = env_for(), cwd = str(work))
    info = {"had_gemma4": r == 0, "tail": (so + se)[-500:]}
    if r != 0:
        r2, so2, se2 = run([a.python, "-m", "pip", "install", "-q", "--disable-pip-version-check", "--no-deps",
                            "transformers==5.23.0"], 1200)
        info["install"] = {"rc": r2, "tail": (so2 + se2)[-800:]}
    gm.write_text(json.dumps(info), encoding = "utf-8")
obs["gemma4_transformers"] = json.loads(gm.read_text(encoding = "utf-8"))
if run([a.python, "-c", "import pytest"], 300)[0] != 0:
    for cmd in ([a.python, "-m", "pip", "install", "-q", "pytest"], ["uv", "pip", "install", "--python", a.python, "-q", "pytest"]):
        if run(cmd, 600)[0] == 0:
            break
if a.state not in ("base", "head"):
    obs["_skipped"] = ["everything"]
    save()
    sys.exit(0)
rc, so, se = run([a.python, "-m", "pip", "list", "--format=json"], 300)
try:
    want = ("torch", "transformers", "trl", "peft", "unsloth", "unsloth_zoo", "unsloth-zoo", "accelerate", "triton",
            "pytorch-triton-rocm", "triton-windows", "bitsandbytes")
    obs["pip"] = {d["name"].lower(): d["version"] for d in json.loads(so) if d["name"].lower() in want}
except Exception as e:  # noqa: BLE001
    obs["pip"] = {"error": f"{e!r} {se[-300:]}"}
code = ("import json, torch, unsloth, unsloth.models._block_compile as bc\n"
        "p = torch.cuda.get_device_properties(0)\n"
        "print('INFO ' + json.dumps({'hip': torch.version.hip, 'arch': str(getattr(p, 'gcnArchName', '')),"
        " 'bc_file': bc.__file__, 'has_gate': hasattr(bc, '_efficient_backward_keeps_guard'),"
        " 'gate': bc._efficient_backward_keeps_guard() if hasattr(bc, '_efficient_backward_keeps_guard') else None,"
        " 'cache_key_tag': getattr(getattr(torch.compiler, 'config', None), 'cache_key_tag', None)}))")
rc, so, se = run([a.python, "-c", code], 900, env = env_for(), cwd = str(work))
info = next((json.loads(l[5:]) for l in so.splitlines() if l.startswith("INFO ")), None)
obs["info"] = info or {"error": (so + se)[-1500:]}
save()

# A. tests
obs["tests"] = {}
for tf in ("tests/test_block_sdpa_efficient_backward_guard.py", "tests/test_block_compile.py"):
    if not (ck / tf).exists():
        obs["tests"][tf] = {"absent": True}
        continue
    junit = work / (Path(tf).stem + ".junit.xml")
    r, so, se = run([a.python, "-m", "pytest", "-q", "-rfEs", "--color=no", "-p", "no:cacheprovider", "-o", "addopts=",
                     f"--junitxml={junit}", tf], 2400, cwd = str(ck),
                    env = env_for({"TORCHINDUCTOR_CACHE_DIR": str(work / "ind_tests")}))
    t = {"rc": r, "tail": (so + se)[-2500:]}
    try:
        cases = {}
        for c in ET.parse(junit).getroot().iter("testcase"):
            kind = "passed"
            for ch in c:
                if ch.tag in ("failure", "error"):
                    kind = "failed"
                elif ch.tag == "skipped" and kind == "passed":
                    kind = "skipped"
            cases[f"{c.get('classname')}::{c.get('name')}"] = kind
        t["cases"] = cases
    except Exception as e:  # noqa: BLE001
        t["junit_error"] = repr(e)
    obs["tests"][tf] = t
    save()

# B. tiny Gemma-4, built once per job from the real 26B-A4B config + tokenizer (no weights)
model_dir = shared / "g4a_tiny26"
if not (model_dir / "config.json").exists() or not list(model_dir.glob("*.safetensors")):
    cfg_dir = shared / "g4a_cfg26"
    code = ("import sys; from huggingface_hub import snapshot_download\n"
            "snapshot_download('unsloth/gemma-4-26B-A4B-it', local_dir=sys.argv[1], "
            "allow_patterns=['*.json', '*.jinja'])")
    r, so, se = run([a.python, "-c", code, str(cfg_dir)], 900, env = env_for(), cwd = str(work))
    r2, so2, se2 = run([a.python, str(HERE / "g18462_make_tiny.py"), str(cfg_dir), str(model_dir)], 1200,
                       env = env_for(), cwd = str(work))
    obs["build"] = {"download_rc": r, "download_tail": (so + se)[-800:], "rc": r2, "tail": (so2 + se2)[-1500:]}
    save()
obs["train"] = {}
for S in (37, 38, 36, 40):
    r, so, se = run([a.python, "-u", str(HERE / "g18462_train_repro.py"), str(model_dir), "--lengths", f"{S},30",
                     "--steps", "2", "--tag", f"{a.state}_S{S}"], 1500, cwd = str(work),
                    env = env_for({"TORCHINDUCTOR_CACHE_DIR": str(work / f"ind_S{S}"),
                                   "UNSLOTH_COMPILE_LOCATION": str(work / f"ucache_S{S}")}))
    blob = so + "\n" + se
    steps = {int(m.group(1)): {"loss": float(m.group(2)), "grad_norm": float(m.group(3))}
             for m in re.finditer(r"step (\d+) T=\d+ loss=([-\d.naif]+) grad_norm=([-\d.naif]+)", blob)}
    res = re.search(r"^RESULT \S+ (.*)$", so, re.M)
    bc = re.search(r"block compiled classes: (.*)$", so, re.M)
    obs["train"][str(S)] = {"rc": r, "result": res.group(1).strip() if res else None, "steps": steps,
                            "block_compiled": bc.group(1) if bc else None,
                            "arm_line": next((l for l in so.splitlines() if l.startswith("ARM")), None),
                            "tail": blob[-2000:]}
    save()
save()
