#!/usr/bin/env python3
"""One cell of the unsloth #18282 smoke: tiny Qwen3 bf16 FULL fine-tune through SFTTrainer with bf16 stochastic
rounding (SR) on or off, one optimizer. Observes only; JSON to --out (never stdout).

Records per-step losses, the SR engagement counters the optimizer carries (_unsloth_sr_fused_steps: one-pass AdamW
kernel launches, _unsloth_sr_triton_rounds: one-pass round launches for bitsandbytes, _unsloth_bf16_stochastic_rounding:
weights wrapped), the per-device Triton probe result and any "one-pass ... unavailable" fallback message.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import random
import sys
import traceback
from pathlib import Path

MODEL = "trl-internal-testing/tiny-Qwen3ForCausalLM"
WORDS = ("the of and to in is was for on as with by he at from his an were are which this be or has had not but "
         "first one their its new after who they have two also been other more all she time year can city").split()

ap = argparse.ArgumentParser()
ap.add_argument("--optim", default = "adamw_torch")
ap.add_argument("--sr", default = "1")
ap.add_argument("--steps", type = int, default = 30)
ap.add_argument("--lr", type = float, default = 1e-4)
ap.add_argument("--seed", type = int, default = 3407)
ap.add_argument("--work", required = True, type = Path)
ap.add_argument("--out", required = True, type = Path)
a = ap.parse_args()
os.environ["UNSLOTH_BF16_STOCHASTIC_ROUNDING"] = a.sr
res: dict = {"optim": a.optim, "sr": a.sr, "stage": "start"}


def dump():
    a.out.write_text(json.dumps(res, indent = 1, default = str), encoding = "utf-8")


log = io.StringIO()
try:
    import unsloth  # noqa: F401  (before transformers / trl)
    from unsloth import FastLanguageModel
    import torch
    import transformers
    import trl
    import unsloth_zoo
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer
    import unsloth.optimizers.bf16_stochastic_rounding as srmod

    res["unsloth_file"] = unsloth.__file__
    res["sr_module_file"] = srmod.__file__
    res["has_adamw_kernel"] = hasattr(srmod, "_adamw_kernel")
    try:
        import triton
        tv = triton.__version__
    except Exception as e:  # noqa: BLE001
        tv = f"ERR {e!r}"[:200]
    res["versions"] = {"torch": torch.__version__, "hip": torch.version.hip, "cuda": torch.version.cuda,
                       "triton": tv, "transformers": transformers.__version__, "trl": trl.__version__,
                       "unsloth_zoo": unsloth_zoo.__version__, "python": sys.version.split()[0]}
    try:
        res["versions"]["device"] = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        res["versions"]["arch"] = getattr(props, "gcnArchName", None) or f"sm{props.major}{props.minor}"
    except Exception as e:  # noqa: BLE001
        res["versions"]["device"] = f"ERR {e}"
    res["stage"] = "imported"
    dump()

    model, tok = FastLanguageModel.from_pretrained(MODEL, max_seq_length = 128, load_in_4bit = False,
                                                   dtype = torch.bfloat16, full_finetuning = True)
    params = [p for p in model.parameters() if p.requires_grad]
    res["model_device"] = str(params[0].device)
    res["n_trainable"] = len(params)
    res["trainable_dtypes"] = sorted({str(p.dtype) for p in params})
    res["stage"] = "loaded"
    dump()

    rng = random.Random(0)
    texts = [" ".join(rng.choice(WORDS) for _ in range(rng.choice([12, 20, 30]))) + "." for _ in range(256)]
    ds = Dataset.from_dict({"text": texts})
    cfg = SFTConfig(output_dir = str(a.work / f"out_{a.optim}_{a.sr}"), max_steps = a.steps, per_device_train_batch_size = 4,
                    gradient_accumulation_steps = 1, learning_rate = a.lr, logging_steps = 1, seed = a.seed,
                    report_to = "none", max_length = 128, dataset_text_field = "text", optim = a.optim,
                    lr_scheduler_type = "linear", warmup_steps = 3, weight_decay = 0.01, bf16 = True,
                    save_strategy = "no")
    trainer = SFTTrainer(model = model, processing_class = tok, train_dataset = ds, args = cfg)
    res["stage"] = "training"
    dump()
    with contextlib.redirect_stdout(log):
        trainer.train()
    res["losses"] = {int(h["step"]): float(h["loss"]) for h in trainer.state.log_history if "loss" in h}
    opt = trainer.optimizer
    inner = getattr(opt, "optimizer", opt)
    res["optimizer_class"] = type(inner).__module__ + "." + type(inner).__name__
    res["sr_wrapped"] = inner.__dict__.get("_unsloth_bf16_stochastic_rounding")
    res["sr_fused_steps"] = inner.__dict__.get("_unsloth_sr_fused_steps", 0)
    res["sr_triton_rounds"] = inner.__dict__.get("_unsloth_sr_triton_rounds", 0)
    res["triton_ok"] = {k: bool(v) for k, v in getattr(srmod, "_TRITON_OK", {}).items()}
    res["finite_weights"] = all(bool(torch.isfinite(p).all()) for p in params)
    res["stage"] = "done"
except BaseException as e:  # noqa: BLE001
    res["error"] = f"{type(e).__name__}: {str(e)[:1500]}"
    res["traceback"] = traceback.format_exc()[-4000:]
text = log.getvalue()
res["fallback_messages"] = [ln for ln in text.splitlines() if "one-pass" in ln or "stochastic rounding" in ln][:10]
dump()
