#!/usr/bin/env python3
"""Probe for PR 13861 on gfx1151: Studio's causal-conv1d decision, setup's SSM step,
and a short Qwen3.5-0.8B LoRA forward/backward through Unsloth, per state.

Observes only. Each part runs in its own subprocess so one failure does not hide the rest.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

MODEL = "unsloth/Qwen3.5-0.8B"

DECISION = r"""
import importlib.metadata, json, os, sys
backend = sys.argv[1]
sys.path.insert(0, backend)
os.chdir(backend)
out = {"torch_dist_version": None, "torch_imported_before": "torch" in sys.modules}
try:
    out["torch_dist_version"] = importlib.metadata.version("torch")
except Exception as e:
    out["torch_dist_version"] = f"ERR {e}"
from utils import ssm_runtime as s
out["has_zoo_predicate"] = hasattr(s, "zoo_triton_causal_conv1d_usable")
if out["has_zoo_predicate"]:
    out["zoo_triton_causal_conv1d_usable"] = s.zoo_triton_causal_conv1d_usable()
name = sys.argv[2]
out["model_wants_causal_conv1d"] = s.model_wants_causal_conv1d(name)
out["model_is_ssm"] = s.model_is_ssm(name)
try:
    out["resolved_model_wants_causal_conv1d"] = s.resolved_model_wants_causal_conv1d(name, name, None)
except Exception as e:
    out["resolved_model_wants_causal_conv1d"] = f"ERR {type(e).__name__}: {e}"
from utils import kernel_install as k
from utils.wheel_utils import probe_torch_wheel_env
env = probe_torch_wheel_env(timeout = 300, include_windows = True)
out["wheel_env"] = env
out["resolve_wheel_url"] = {n: k.resolve_wheel_url(n, env) for n in ("causal_conv1d", "mamba_ssm")}
dists = {}
for d in ("causal-conv1d", "mamba-ssm", "causal_conv1d", "mamba_ssm"):
    try:
        dists[d] = importlib.metadata.version(d)
    except importlib.metadata.PackageNotFoundError:
        dists[d] = None
out["installed_dists"] = dists
print("DECISION_JSON " + json.dumps(out, default = str))
"""

TRAIN = r"""
import json, os, re, sys, time
import unsloth
from unsloth import FastLanguageModel
import torch
out = {"unsloth_file": unsloth.__file__, "torch": torch.__version__,
       "hip": getattr(torch.version, "hip", None), "cuda": torch.version.cuda}
try:
    import causal_conv1d
    out["causal_conv1d_importable"] = True
except Exception as e:
    out["causal_conv1d_importable"] = f"no: {type(e).__name__}"
import unsloth_zoo
out["unsloth_zoo_version"] = getattr(unsloth_zoo, "__version__", None)
props = torch.cuda.get_device_properties(0)
out["device"] = torch.cuda.get_device_name(0)
out["arch"] = getattr(props, "gcnArchName", None)
model, tok = FastLanguageModel.from_pretrained(sys.argv[1], max_seq_length = 512,
    dtype = torch.bfloat16, load_in_4bit = False)
out["model_type"] = model.config.model_type
model = FastLanguageModel.get_peft_model(model, r = 16, lora_alpha = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing = "unsloth", random_state = 0)
g = torch.Generator().manual_seed(0)
ids = torch.randint(100, 20000, (2, 256), generator = g).cuda()
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr = 1e-4)
model.train()
losses = []
kernels = set()
from torch.profiler import profile, ProfilerActivity
for step in range(4):
    if step == 0:
        with profile(activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            loss = model(input_ids = ids, labels = ids).loss
            loss.backward()
            torch.cuda.synchronize()
        for e in prof.key_averages():
            if re.search(r"conv1d|conv_depthwise|causal_conv|convolution", e.key, re.I):
                kernels.add(e.key[:120])
    else:
        loss = model(input_ids = ids, labels = ids).loss
        loss.backward()
    opt.step(); opt.zero_grad(set_to_none = True)
    losses.append(float(loss.detach().float().cpu()))
torch.cuda.synchronize()
out["losses"] = losses
out["conv_kernels"] = sorted(kernels)
out["peak_alloc_gb"] = torch.cuda.max_memory_allocated() / 1e9
print("TRAIN_JSON " + json.dumps(out, default = str))
"""


def _run(cmd, env, cwd, timeout, tag):
    try:
        p = subprocess.run(cmd, env = env, cwd = cwd, capture_output = True, text = True,
                           encoding = "utf-8", errors = "replace", timeout = timeout)
        rc, text = p.returncode, (p.stdout or "") + "\n" + (p.stderr or "")
    except subprocess.TimeoutExpired:
        rc, text = -1, "TimeoutExpired"
    payload = None
    for line in text.splitlines():
        if line.startswith(tag + " "):
            payload = json.loads(line[len(tag) + 1:])
    return {"rc": rc, "data": payload, "tail": text[-3000:]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--python", default = sys.executable)
    args = ap.parse_args()

    co = Path(args.checkout)
    backend = co / "studio" / "backend"
    obs: dict = {"state": args.state, "checkout": str(co)}
    try:
        obs["commit"] = subprocess.run(["git", "-C", str(co), "rev-parse", "HEAD"],
                                       capture_output = True, text = True).stdout.strip()
    except OSError:
        obs["commit"] = None

    stack = (co / "studio" / "install_python_stack.py").read_text(encoding = "utf-8")
    obs["setup_calls_ensure_ssm_kernels"] = stack.count("_ensure_ssm_kernels()")
    obs["setup_has_mamba_progress_step"] = '_progress("mamba kernels")' in stack

    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    obs["decision"] = _run([args.python, "-c", DECISION, str(backend), MODEL],
                           env, str(backend), 600, "DECISION_JSON")

    tenv = dict(env)
    # The state's own unsloth tree wins over the editable head install.
    tenv["PYTHONPATH"] = str(co)
    tenv["UNSLOTH_COMPILE_LOCATION"] = str(args.out.parent / f"compiled_{args.state}")
    tenv.setdefault("PYTHONHASHSEED", "0")
    obs["train"] = _run([args.python, "-c", TRAIN, MODEL], tenv, str(args.out.parent),
                        2400, "TRAIN_JSON")
    args.out.write_text(json.dumps(obs, indent = 2, default = str), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
