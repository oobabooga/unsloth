#!/usr/bin/env python3
"""Probe for unsloth-zoo #2767 (W8A16 grouped GEMM for block-FP8 routed experts; CUDA sm89+ only, never ROCm).
Per state (zoo checkout first on PYTHONPATH, UNSLOTH_IS_PRESENT=1), Linux ROCm only:
  pytest tests/test_fp8_bank_release_checkpointing.py (both states) and tests/test_moe_fp8_w8a16_grouped.py (absent
  at base by construction), then z2767_cell.py: a minimal block-FP8 experts module on the GPU through
  forward_moe_backend_fp8, forward + backward, auto twice (A/A) and UNSLOTH_MOE_FP8_FUSED=0.
One shared venv via zoo_venv.build_venv (--venv, --head-checkout); unsloth main pinned to one commit for both arms
(--no-deps), because the dequant path calls unsloth.kernels.fp8. Observes only; JSON to --out."""
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
ap.add_argument("--venv", type = Path, required = True)
ap.add_argument("--head-checkout", required = True)
a = ap.parse_args()
a.out = a.out.resolve()
ck = str(Path(a.checkout).resolve())
out_dir = a.out.parent / f"{a.state}_artifacts"
out_dir.mkdir(parents = True, exist_ok = True)
root = Path(os.environ.get("AMD_CI_WORK") or out_dir)
shared = root / "z2767_shared"
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
UNSLOTH_SHA = "3dd8b14c58965de4a122f7ed5bfcea7a67649797"
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

obs["pytest_bank"] = zoo_venv.run_pytest(python, ck, ["tests/test_fp8_bank_release_checkpointing.py"], out_dir,
                                         f"{a.state}_bank", 1800)
save()
obs["pytest_new"] = zoo_venv.run_pytest(python, ck, ["tests/test_moe_fp8_w8a16_grouped.py"], out_dir,
                                        f"{a.state}_new", 1800)
save()

work = out_dir / "w_cell"
work.mkdir(exist_ok = True)
obs["cell"] = zoo_venv.run_cell(python, HERE / "z2767_cell.py", ["--device", "cuda"], ck, out_dir, "fp8_experts",
                                timeout = 1800,
                                extra_env = {"UNSLOTH_COMPILE_LOCATION": str(work / "ucache"), "PYTHONUTF8": "1",
                                             "UNSLOTH_DISABLE_AUTO_UPDATES": "1", "HIP_VISIBLE_DEVICES": "0",
                                             "UNSLOTH_MOE_BACKEND": "grouped_mm"})
save()
