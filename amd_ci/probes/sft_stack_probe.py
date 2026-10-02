#!/usr/bin/env python3
"""Probe: install THIS checkout's own dependency window over the runner's ROCm torch, then train.

Observes only. For one state it creates a venv layered over the Studio venv (a .pth makes the
ROCm torch, bitsandbytes and friends visible, so pip treats them as installed), installs the
checkout with `[huggingfacenotorch]` plus an eager upgrade of transformers / trl, so the
resolver lands on the highest release the state's pins allow, and runs a short LoRA SFT on the
GPU for a dense and an MoE tiny model. Records the resolved versions, per-step losses, whether
the run used the GPU, torch.compile graph breaks, and steady step time.

Pairs with criteria/sft_stack_no_regression.py.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

MODELS = {
    "dense": "trl-internal-testing/tiny-Qwen3ForCausalLM",
    "moe": "hf-internal-testing/tiny-random-MixtralForCausalLM",
}

_TRAIN = r'''
import json, math, os, sys, time
import unsloth
from unsloth import FastLanguageModel
import torch, transformers, trl, peft
from datasets import Dataset
from trl import SFTConfig, SFTTrainer

out = {"versions": {m.__name__: m.__version__ for m in (torch, transformers, trl, peft, unsloth)},
       "hip": getattr(torch.version, "hip", None), "device": torch.cuda.get_device_name(0),
       "unsloth_file": unsloth.__file__}
model, tok = FastLanguageModel.from_pretrained(sys.argv[1], max_seq_length = 256, load_in_4bit = False,
                                               dtype = torch.bfloat16)
model = FastLanguageModel.get_peft_model(model, r = 8, lora_alpha = 16, random_state = 3407,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"])
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
ds = Dataset.from_list([{"text": f"Example {i}: the quick brown fox jumps over the lazy dog."} for i in range(64)])
times = []
class Timer(transformers.TrainerCallback):
    def on_step_begin(self, *a, **k): torch.cuda.synchronize(); self.t = time.perf_counter()
    def on_step_end(self, *a, **k): torch.cuda.synchronize(); times.append(time.perf_counter() - self.t)
cfg = SFTConfig(output_dir = "sft_out", per_device_train_batch_size = 4, max_steps = 12, learning_rate = 2e-4,
    logging_steps = 1, report_to = "none", seed = 3407, bf16 = True, dataset_text_field = "text")
trainer = SFTTrainer(model = model, processing_class = tok, train_dataset = ds, args = cfg, callbacks = [Timer()])
trainer.train()
out["losses"] = [h["loss"] for h in trainer.state.log_history if "loss" in h]
out["finite"] = all(math.isfinite(x) for x in out["losses"])
out["steady_step_ms"] = 1000 * sorted(times[2:])[len(times[2:]) // 2] if len(times) > 3 else None
from torch._dynamo.utils import counters
out["graph_breaks"] = int(sum(counters.get("graph_break", {}).values()))
out["unique_graphs"] = int(counters.get("stats", {}).get("unique_graphs", 0))
out["graph_break_reasons"] = {str(k).splitlines()[0][:160]: int(v) for k, v in counters.get("graph_break", {}).items()}
print("SFT_STACK_RESULT " + json.dumps(out))
'''


def run(cmd, env = None, timeout = 3600):
    p = subprocess.run(cmd, capture_output = True, text = True, encoding = "utf-8", errors = "replace",
                       env = env, timeout = timeout)
    return p.returncode, p.stdout, p.stderr


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--python", default = sys.executable)
    ap.add_argument("--zoo-spec", default = "",
                    help = "pip spec for unsloth_zoo installed into every state (e.g. a git ref under review)")
    args = ap.parse_args()
    obs: dict = {"state": args.state}
    root = Path(os.environ.get("RUNNER_TEMP", "/tmp")) / f"sft_stack_{args.state}"
    venv = root / "venv"
    try:
        rc, site, err = run([args.python, "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"])
        studio_site = site.strip()
        rc, _, err = run([args.python, "-m", "venv", "--without-pip", str(venv)])
        if rc:
            raise RuntimeError(f"venv: {err[-800:]}")
        py = str(venv / "Scripts" / "python.exe") if os.name == "nt" else str(venv / "bin" / "python")
        rc, vsite, err = run([py, "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"])
        Path(vsite.strip(), "_studio_layer.pth").write_text(studio_site + "\n", encoding = "utf-8")
        rc, _, err = run([py, "-m", "ensurepip", "-q"])
        spec = f"{args.checkout}[huggingfacenotorch]"
        rc, so, err = run([py, "-m", "pip", "install", "-q", "--upgrade", "--upgrade-strategy", "only-if-needed",
                           spec, "transformers", "trl"] + ([args.zoo_spec] if args.zoo_spec else []))
        obs["install_rc"] = rc
        obs["zoo_spec"] = args.zoo_spec
        if rc:
            raise RuntimeError(f"install: {(so + err)[-2000:]}")
        rc, freeze, _ = run([py, "-m", "pip", "list", "--format=json"])
        obs["installed"] = {d["name"].lower(): d["version"] for d in json.loads(freeze)
                            if d["name"].lower() in ("transformers", "trl", "peft", "datasets", "accelerate",
                                                     "unsloth", "unsloth-zoo", "unsloth_zoo", "torch")}
        env = dict(os.environ, PYTHONPATH = args.checkout, UNSLOTH_DISABLE_AUTO_UPDATES = "1",
                   UNSLOTH_COMPILE_LOCATION = str(root / "cc"))
        obs["runs"] = {}
        for name, model in MODELS.items():
            work = root / f"run_{name}"
            work.mkdir(parents = True, exist_ok = True)
            script = work / "train.py"
            script.write_text(_TRAIN, encoding = "utf-8")
            rc, so, err = run([py, str(script), model], env = dict(env, PWD = str(work)), timeout = 2400)
            line = [l for l in so.splitlines() if l.startswith("SFT_STACK_RESULT ")]
            obs["runs"][name] = json.loads(line[-1].split(" ", 1)[1]) if line else \
                {"error": f"rc={rc}: {(so + err)[-2500:]}"}
    except Exception as e:  # noqa: BLE001
        obs["error"] = f"{type(e).__name__}: {e}"
    args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
