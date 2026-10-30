#!/usr/bin/env python3
"""Probe: does Unsloth's generate wrapper drive LFM3-Omni's custom generate on this checkout (unslothai/unsloth#16096)?

Observes only. Builds a tiny random LFM3-Omni (real config, code and tokenizer from the Hub, no weights) under
transformers 5.22.0, loads it with FastVisionModel from THIS checkout, then generates, trains 3 LoRA steps and
generates again. Pairs with criteria/lfm3_generate.py.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

ARM = r'''
import json, os, sys, traceback
os.environ["UNSLOTH_IS_PRESENT"] = "1"
res = {"steps": {}}
def step(name, fn):
    try:
        res["steps"][name] = {"ok": True, "detail": fn()}
    except Exception as e:
        res["steps"][name] = {"ok": False, "detail": f"{type(e).__name__}: {str(e)[:300]}"}
import torch, unsloth, transformers
from unsloth import FastVisionModel
res.update(unsloth_file=unsloth.__file__, transformers=transformers.__version__, hip=torch.version.hip,
           device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
model, proc = FastVisionModel.from_pretrained(sys.argv[1], trust_remote_code=True, load_in_4bit=False,
                                              dtype=torch.float16, max_seq_length=256)
tok = getattr(proc, "tokenizer", proc)
ids = tok("Hello there", return_tensors="pt").to(model.device)
FastVisionModel.for_inference(model)
def gen():
    out = model.generate(**ids, max_new_tokens=6, do_sample=False)
    return list(getattr(out, "sequences", out).shape)
step("generate_base_model", gen)
model = FastVisionModel.get_peft_model(model, r=8, lora_alpha=8, finetune_vision_layers=False,
                                       finetune_language_layers=True, target_modules="all-linear")
FastVisionModel.for_training(model)
def train():
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    batch = tok(["The sum is 65. Colour ZX-RED-3."] * 2, return_tensors="pt").to(model.device)
    out = []
    for _ in range(3):
        o = model(**batch, labels=batch["input_ids"]); o.loss.backward(); opt.step(); opt.zero_grad()
        out.append(round(float(o.loss), 4))
    return out
step("lora_train_3_steps", train)
FastVisionModel.for_inference(model)
step("generate_lora_model", gen)
print("LFM3_PROBE " + json.dumps(res), flush=True)
'''


def ensure(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True)
    ap.add_argument("--checkout", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    work = Path(os.environ.get("RUNNER_TEMP") or "/tmp") / "lfm3_probe"
    overlay, tiny = work / "tf522", work / "tiny_lfm3"
    obs: dict = {"state": a.state}
    try:
        if not (overlay / "transformers").is_dir():
            ensure([sys.executable, "-m", "pip", "install", "-q", "--target", str(overlay),
                    "transformers==5.22.0", "tokenizers>=0.23.1,<0.24", "huggingface_hub>=1.0"])
        env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(overlay)]))
        if not (tiny / "config.json").is_file():
            subprocess.run([sys.executable, str(HERE.parent / "tools" / "tiny_make.py"), "LiquidAI/LFM3-Omni-2B",
                            "--trust-remote-code", "--no-check", "--out", str(tiny)], check=True, env=env)
        env["PYTHONPATH"] = os.pathsep.join([str(a.checkout.resolve()), str(overlay)])
        env["UNSLOTH_COMPILE_LOCATION"] = str(work / f"cache_{a.state}")
        p = subprocess.run([sys.executable, "-c", ARM, str(tiny)], capture_output=True, text=True, env=env,
                           cwd=str(work))
        line = [l for l in p.stdout.splitlines() if l.startswith("LFM3_PROBE ")]
        obs.update(json.loads(line[-1][11:]) if line else {"error": (p.stderr or "")[-2000:]})
        obs["checkout"] = str(a.checkout.resolve())
        obs["imports_own_checkout"] = str(obs.get("unsloth_file", "")).startswith(str(a.checkout.resolve()))
    except Exception as e:  # noqa: BLE001
        obs["error"] = f"{type(e).__name__}: {e}"
    a.out.write_text(json.dumps(obs, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
