#!/usr/bin/env python3
"""Probe for unsloth-zoo #2615 (gradient checkpointing runs checkpointed blocks directly when grad is off).
Per state (zoo checkout first on PYTHONPATH, UNSLOTH_IS_PRESENT=1): pytest tests/test_checkpoint_no_grad_direct.py
(head only), then z2615_cell.py: tiny Llama LoRA, unsloth GC, one no-grad forward in train mode + 2 SFT steps.
Linux: one shared venv via zoo_venv.build_venv (--venv, --head-checkout). Windows: the workflow's venv (--python) plus
the training stack installed once. Observes only; JSON to --out."""
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
shared = root / "z2615_shared"
shared.mkdir(exist_ok = True)
obs: dict = {"state": a.state, "checkout": ck, "platform": sys.platform}
if a.state not in ("base", "head"):
    obs["_skipped"] = ["everything"]
    a.out.write_text(json.dumps(obs), encoding = "utf-8")
    sys.exit(0)
obs["commit"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd = ck, capture_output = True, text = True).stdout.strip()


def save():
    a.out.write_text(json.dumps(obs, indent = 1, default = str), encoding = "utf-8")


def sh(cmd, tag, timeout):
    return zoo_venv.sh(cmd, out_dir / f"{tag}.log", timeout = timeout)


if WIN:
    python = a.python
    mark = shared / "stack.json"
    if not mark.exists():
        steps = {}
        steps["deps"] = sh([python, "-m", "pip", "install", "-q", "--disable-pip-version-check", "transformers==5.5.4",
                            "trl==0.23.1", "peft==0.18.1", "accelerate", "datasets>=3.4.1,!=4.0.*,!=4.1.0,!=4.4.*,!=4.5.0,<5.0.0",
                            "sentencepiece>=0.2.0", "protobuf", "huggingface_hub>=0.34.0", "bitsandbytes", "einops", "msgspec",
                            "psutil", "tqdm", "regex", "pillow", "packaging", "numpy", "wheel", "requests", "filelock",
                            "typing_extensions", "typer>=0.19.0,<0.29", "rich", "pydantic", "pyyaml", "json5>=0.9.25",
                            "nest-asyncio", "structlog>=24.1.0", "click>=8.0", "tyro", "diffusers", "certifi",
                            "triton-windows<3.7", "hf_transfer", "pytest", "cut_cross_entropy", "torchao"], "stack", 2400)
        steps["unsloth"] = sh([python, "-m", "pip", "install", "-q", "--disable-pip-version-check", "--no-deps", "unsloth",
                               "unsloth_zoo"], "stack", 1200)
        mark.write_text(json.dumps(steps), encoding = "utf-8")
    obs["venv"] = {"windows_stack": json.loads(mark.read_text(encoding = "utf-8"))}
else:
    obs["venv"] = zoo_venv.build_venv(a.venv.resolve(), a.head_checkout, "torch==2.11.0+rocm7.13.0", "amd", None, out_dir)
    if obs["venv"].get("error"):
        save()
        sys.exit(0)
    python = obs["venv"]["python"]
    # unsloth (installed --no-deps) imports bitsandbytes and friends that unsloth-zoo does not declare.
    mark = shared / "extra.json"
    if not mark.exists():
        import shutil as _sh
        uv = _sh.which("uv")
        base_cmd = [uv, "pip", "install", "--python", python] if uv else [python, "-m", "pip", "install", "-q"]
        rc = sh(base_cmd + ["--extra-index-url", "https://pypi.org/simple", "bitsandbytes", "datasets", "sentencepiece",
                            "protobuf", "hf_transfer", "tyro", "psutil", "wheel", "msgspec", "diffusers"], "extra", 1800)
        mark.write_text(json.dumps({"rc": rc}), encoding = "utf-8")
    obs["venv"]["extra"] = json.loads(mark.read_text(encoding = "utf-8"))
obs["python"] = python
# unsloth main, pinned to one commit, identical for both arms (installed once, --no-deps over the release).
UNSLOTH_SHA = "98f23d884cf9699e87706acfa750a2f37d7420a3"
umark = shared / "unsloth_main.json"
if not umark.exists():
    rc = sh([python, "-m", "pip", "install", "-q", "--disable-pip-version-check", "--no-deps", "--force-reinstall",
             f"unsloth @ git+https://github.com/unslothai/unsloth@{UNSLOTH_SHA}"], "unsloth_main", 1800)
    umark.write_text(json.dumps({"rc": rc, "sha": UNSLOTH_SHA}), encoding = "utf-8")
obs["unsloth_main"] = json.loads(umark.read_text(encoding = "utf-8"))
try:
    du = subprocess.run([python, "-c", "import importlib.metadata as m; print(m.distribution('unsloth').read_text('direct_url.json'))"],
                        capture_output = True, text = True, timeout = 300).stdout.strip()
    obs["unsloth_main"]["direct_url"] = du
except Exception as e:  # noqa: BLE001
    obs["unsloth_main"]["direct_url"] = repr(e)
obs["zoo_import"] = zoo_venv.zoo_import(python, ck)
save()

# pytest: the PR's file (absent at base)
obs["pytest"] = zoo_venv.run_pytest(python, ck, ["tests/test_checkpoint_no_grad_direct.py"], out_dir, a.state, 1800)
save()

# training cell
model_dir = shared / "tiny_llama"
if not (model_dir / "config.json").exists():
    obs["build"] = zoo_venv.run_cell(python, HERE / "z2615_cell.py", ["--mode", "build", "--dir", str(model_dir)], ck,
                                     out_dir, "build", timeout = 900)
work = out_dir / "w_cell"
work.mkdir(exist_ok = True)
obs["cell"] = zoo_venv.run_cell(python, HERE / "z2615_cell.py", ["--mode", "run", "--dir", str(model_dir), "--work", str(work)],
                                ck, out_dir, "run", timeout = 1800,
                                extra_env = {"UNSLOTH_COMPILE_LOCATION": str(work / "ucache"), "PYTHONUTF8": "1",
                                             "UNSLOTH_DISABLE_AUTO_UPDATES": "1", "HIP_VISIBLE_DEVICES": "0"})
save()
