#!/usr/bin/env python3
"""Probe for unsloth-zoo #2655 (CPU merged save / GGUF staging streamed from the live model).
Per state (checkout first on PYTHONPATH, UNSLOTH_IS_PRESENT=1, no GPU: HIP_VISIBLE_DEVICES="" because the code path
is the CPU loader): g16z_merge_cell.py builds tiny random models in-process (Llama dense fp32 / bf16, Qwen3, Qwen3MoE,
tied embeddings with LoRA on q/v and on lm_head / embed_tokens, LoRA on embed_tokens, DoRA, rslora, multiple adapters,
frozen bf16, sharded, full finetune, mixed dtype, a Llava VLM, safe_serialization=False), calls the public
save_pretrained_merged(..., "merged_16bit") and the GGUF staging save, and records per-file / per-tensor sha256,
safetensors key order, index.json, config.json and whether the streamed path engaged (head).
Linux: one shared venv via zoo_venv.build_venv (--venv, --head-checkout; CPU torch from download.pytorch.org/whl/cpu).
Windows: the workflow's interpreter (--python, or --use-current) with CPU torch + the checkout's deps installed once.
merge state: _skipped. Observes only; JSON to --out."""
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

CPU_INDEX = "https://download.pytorch.org/whl/cpu"
ap = argparse.ArgumentParser()
ap.add_argument("--state", required = True)
ap.add_argument("--checkout", required = True)
ap.add_argument("--out", required = True, type = Path)
ap.add_argument("--python", default = None)
ap.add_argument("--use-current", action = "store_true", help = "measure with this interpreter (Windows)")
ap.add_argument("--venv", type = Path, default = None)
ap.add_argument("--head-checkout", default = None)
ap.add_argument("--torch-pin", default = "torch")
ap.add_argument("--transformers", default = None, help = "optional transformers pin, e.g. 5.24.0")
ap.add_argument("--cells", default = "", help = "comma list; default every cell")
a = ap.parse_args()
a.out = a.out.resolve()
WIN = sys.platform == "win32"
ck = str(Path(a.checkout).resolve())
out_dir = a.out.parent / f"{a.state}_artifacts"
out_dir.mkdir(parents = True, exist_ok = True)
root = Path(os.environ.get("AMD_CI_WORK") or out_dir)
shared = root / "g16z_shared"
shared.mkdir(parents = True, exist_ok = True)
obs: dict = {"state": a.state, "checkout": ck, "platform": sys.platform}


def save():
    a.out.write_text(json.dumps(obs, indent = 1, default = str), encoding = "utf-8")


if a.state not in ("base", "head"):
    obs["_skipped"] = ["everything"]
    save()
    sys.exit(0)
try:
    obs["commit"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd = ck, capture_output = True, text = True,
                                   timeout = 60).stdout.strip()
except Exception as e:  # noqa: BLE001
    obs["commit"] = repr(e)


def sh(cmd, tag, timeout):
    return zoo_venv.sh(cmd, out_dir / f"{tag}.log", timeout = timeout)


if WIN or a.use_current or a.venv is None:
    python = a.python or sys.executable
    mark = shared / "stack.json"
    if not mark.exists():
        pip = [python, "-m", "pip", "install", "-q", "--disable-pip-version-check"]
        have = subprocess.run([python, "-c", "import torch, transformers, peft, safetensors, accelerate"],
                              capture_output = True, text = True, timeout = 600).returncode == 0
        steps = {"preinstalled": have}
        if not have:
            steps["torch"] = sh(pip + ["--index-url", CPU_INDEX, "--extra-index-url", zoo_venv.PYPI, a.torch_pin],
                                "stack", 2400)
            # The checkout's declared deps (transformers, peft, accelerate, safetensors, ...); its own unsloth_zoo
            # copy is shadowed by PYTHONPATH, which zoo_import records.
            steps["deps"] = sh(pip + [ck], "stack", 2400)
            steps["unsloth"] = sh(pip + ["--no-deps", "unsloth"], "stack", 1200)
        if a.transformers:
            steps["transformers"] = sh(pip + [f"transformers=={a.transformers}"], "stack", 1200)
        mark.write_text(json.dumps(steps), encoding = "utf-8")
    obs["venv"] = {"stack": json.loads(mark.read_text(encoding = "utf-8")), "python": python}
else:
    obs["venv"] = zoo_venv.build_venv(a.venv.resolve(), a.head_checkout or ck, a.torch_pin, CPU_INDEX,
                                      a.transformers, out_dir)
    if obs["venv"].get("error"):
        save()
        sys.exit(0)
    python = obs["venv"]["python"]
obs["python"] = python
env_extra = {"HIP_VISIBLE_DEVICES": "", "ROCR_VISIBLE_DEVICES": "", "CUDA_VISIBLE_DEVICES": "", "PYTHONUTF8": "1",
             "UNSLOTH_DISABLE_AUTO_UPDATES": "1", "PYTHONHASHSEED": "0"}
try:
    v = subprocess.run([python, "-c", "import json, torch, transformers, peft, safetensors; print('AMDCI' + json.dumps("
                        "{'torch': torch.__version__, 'transformers': transformers.__version__, 'peft': peft.__version__,"
                        " 'safetensors': safetensors.__version__}))"],
                       capture_output = True, text = True, timeout = 600, env = zoo_venv.state_env(ck, env_extra))
    line = next((ln for ln in v.stdout.splitlines() if ln.startswith("AMDCI")), None)
    obs["versions"] = json.loads(line[5:]) if line else {"error": (v.stdout + v.stderr)[-1500:]}
except Exception as e:  # noqa: BLE001
    obs["versions"] = {"error": repr(e)}
# The same env the cell runs with: on a ROCm host a CPU torch makes unsloth_zoo refuse to import unless HIP is hidden.
_zi = subprocess.run([python, "-c", "import unsloth_zoo; print('AMDCI' + unsloth_zoo.__file__)"], cwd = ck,
                     env = zoo_venv.state_env(ck, env_extra), capture_output = True, text = True, timeout = 900)
_zl = next((ln for ln in (_zi.stdout or "").splitlines() if ln.startswith("AMDCI")), None)
obs["zoo_import"] = {"file": _zl[5:] if _zl else None, "error": None if _zl else (_zi.stdout + _zi.stderr)[-2500:]}
save()

work = root / f"g16z_work_{a.state}"
args = ["--tree", ck, "--workdir", str(work)] + (["--cells", a.cells] if a.cells else [])
obs["cell"] = zoo_venv.run_cell(python, HERE / "g16z_merge_cell.py", args, ck, out_dir, "merge", timeout = 2400,
                                extra_env = env_extra)
save()
