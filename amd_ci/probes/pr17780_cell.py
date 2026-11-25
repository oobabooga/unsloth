#!/usr/bin/env python3
"""One phase of the unsloth #17780 non-FSDP2 smoke: single-GPU LoRA SFT on a tiny MoE with save + resume. Observes only.

--phase train   6 steps, save_steps 3; snapshots every trainable (LoRA) tensor at the step-3 save to --snap
--phase resume  fresh load + get_peft_model, Trainer.train(resume_from_checkpoint=checkpoint-3) for steps 4-6;
                at on_train_begin (after Trainer loaded the checkpoint) diffs every LoRA tensor against --snap
JSON to --out (never stdout).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import traceback
from pathlib import Path

MODEL = "trl-internal-testing/tiny-Qwen3MoeForCausalLM"
WORDS = ("the of and to in is was for on as with by he at from his an were are which this be or has had not but "
         "first one their its new after who they have two also been other more all she time year can city").split()

ap = argparse.ArgumentParser()
ap.add_argument("--phase", choices = ("train", "resume"), required = True)
ap.add_argument("--work", required = True, type = Path)
ap.add_argument("--snap", required = True, type = Path)
ap.add_argument("--out", required = True, type = Path)
a = ap.parse_args()
res: dict = {"phase": a.phase, "stage": "start"}


def dump():
    a.out.write_text(json.dumps(res, indent = 1, default = str), encoding = "utf-8")


try:
    import unsloth  # noqa: F401  (before transformers / trl)
    from unsloth import FastLanguageModel
    import torch
    import transformers
    import trl
    import peft
    import unsloth_zoo
    from datasets import Dataset
    from transformers import TrainerCallback
    from trl import SFTConfig, SFTTrainer

    res["unsloth_file"] = unsloth.__file__
    res["versions"] = {"torch": torch.__version__, "hip": torch.version.hip, "transformers": transformers.__version__,
                       "trl": trl.__version__, "peft": peft.__version__, "unsloth_zoo": unsloth_zoo.__version__,
                       "unsloth": getattr(unsloth, "__version__", None), "python": sys.version.split()[0]}
    try:
        res["versions"]["device"] = torch.cuda.get_device_name(0)
        res["versions"]["arch"] = getattr(torch.cuda.get_device_properties(0), "gcnArchName", None)
    except Exception as e:  # noqa: BLE001
        res["versions"]["device"] = f"ERR {e}"
    res["stage"] = "imported"
    dump()

    model, tok = FastLanguageModel.from_pretrained(MODEL, max_seq_length = 128, load_in_4bit = False,
                                                   dtype = torch.bfloat16)
    model = FastLanguageModel.get_peft_model(
        model, r = 8, lora_alpha = 16, lora_dropout = 0, random_state = 0,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    lora = {n: p for n, p in model.named_parameters() if p.requires_grad}
    res["model_device"] = str(next(iter(lora.values())).device)
    res["n_lora_tensors"] = len(lora)
    res["expert_lora_tensors"] = sorted(n for n in lora if "expert" in n)[:8]
    res["n_expert_lora_tensors"] = sum("expert" in n for n in lora)
    res["stage"] = "peft"
    dump()

    rng = random.Random(0)
    texts = [" ".join(rng.choice(WORDS) for _ in range(rng.choice([12, 20, 30]))) + "." for _ in range(64)]
    ds = Dataset.from_dict({"text": texts})
    out_dir = a.work / "trainer_out"

    class Snap(TrainerCallback):
        def on_save(self, args, state, control, **kw):
            if state.global_step == 3 and a.phase == "train":
                torch.save({n: p.detach().float().cpu().clone() for n, p in lora.items()}, a.snap)

        def on_train_begin(self, args, state, control, **kw):
            if a.phase == "resume":
                want = torch.load(a.snap, weights_only = True)
                diffs = {n: (p.detach().float().cpu() - want[n]).abs().max().item() for n, p in lora.items()
                         if n in want}
                res["resume_global_step"] = state.global_step
                res["resume_missing"] = sorted(set(want) - set(diffs))[:8]
                res["resume_max_abs_diff"] = max(diffs.values()) if diffs else None
                res["resume_expert_max_abs_diff"] = max(
                    [v for n, v in diffs.items() if "expert" in n], default = None)
                res["resume_n_compared"] = len(diffs)

    cfg = SFTConfig(output_dir = str(out_dir), max_steps = 6, save_steps = 3, save_strategy = "steps",
                    per_device_train_batch_size = 2, gradient_accumulation_steps = 1, learning_rate = 5e-3,
                    logging_steps = 1, seed = 0, report_to = "none", max_length = 128, dataset_text_field = "text",
                    lr_scheduler_type = "constant", warmup_steps = 0, bf16 = True, save_only_model = False)
    trainer = SFTTrainer(model = model, processing_class = tok, train_dataset = ds, args = cfg, callbacks = [Snap()])
    res["stage"] = "training"
    dump()
    if a.phase == "train":
        trainer.train()
    else:
        trainer.train(resume_from_checkpoint = str(out_dir / "checkpoint-3"))
    res["losses"] = {int(h["step"]): float(h["loss"]) for h in trainer.state.log_history if "loss" in h}
    res["checkpoints"] = sorted(p.name for p in out_dir.glob("checkpoint-*"))
    res["stage"] = "done"
except BaseException as e:  # noqa: BLE001
    res["error"] = f"{type(e).__name__}: {str(e)[:1500]}"
    res["traceback"] = traceback.format_exc()[-4000:]
dump()
