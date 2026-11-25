#!/usr/bin/env python3
"""One cell of the unsloth #18286 smoke: tiny Qwen3 LoRA + QAT (int4) through SFTTrainer, then model.eval() and a
forward. Observes only; JSON to --out (never stdout).

Records per-step losses, whether every loss / the eval logits are finite, and (when the state has it)
unsloth.models._qat_weight_memo.qat_weight_memo_stats() after every optimizer step, after training and after the
eval forward. UNSLOTH_QAT_WEIGHT_MEMO comes from the caller's environment.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import random
import sys
import traceback
from pathlib import Path

MODEL = "trl-internal-testing/small-Qwen3ForCausalLM"
WORDS = ("the of and to in is was for on as with by he at from his an were are which this be or has had not but "
         "first one their its new after who they have two also been other more all she time year can city").split()

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type = int, default = 6)
ap.add_argument("--work", required = True, type = Path)
ap.add_argument("--out", required = True, type = Path)
a = ap.parse_args()
res: dict = {"stage": "start", "memo_env": os.environ.get("UNSLOTH_QAT_WEIGHT_MEMO"), "model": MODEL}


def dump():
    a.out.write_text(json.dumps(res, indent = 1, default = str), encoding = "utf-8")


def memo_stats():
    try:
        from unsloth.models._qat_weight_memo import qat_weight_memo_stats
    except Exception as e:  # noqa: BLE001  (absent at the base)
        return {"absent": f"{type(e).__name__}: {e}"[:200]}
    return qat_weight_memo_stats()


log = io.StringIO()
try:
    import unsloth  # noqa: F401  (before transformers / trl)
    from unsloth import FastLanguageModel
    import torch
    import transformers
    import trl
    import peft
    import torchao
    import unsloth_zoo
    from datasets import Dataset
    from transformers import TrainerCallback
    from trl import SFTConfig, SFTTrainer

    res["unsloth_file"] = unsloth.__file__
    try:
        import unsloth.models._qat_weight_memo as _m
        res["memo_file"] = _m.__file__
    except Exception as e:  # noqa: BLE001
        res["memo_file"] = None
        res["memo_import"] = f"{type(e).__name__}: {e}"[:200]
    res["versions"] = {"torch": torch.__version__, "hip": torch.version.hip, "transformers": transformers.__version__,
                       "trl": trl.__version__, "peft": peft.__version__, "torchao": torchao.__version__,
                       "unsloth_zoo": unsloth_zoo.__version__, "python": sys.version.split()[0]}
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
        model, r = 8, lora_alpha = 16, lora_dropout = 0, bias = "none",
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        qat_scheme = "int4", use_gradient_checkpointing = "unsloth", random_state = 3407)
    fq = [n for n, m in model.named_modules() if getattr(m, "weight_fake_quantizer", None) is not None]
    res["fake_quantized_linears"] = len(fq)
    res["fake_quantized_sample"] = fq[:4]
    res["model_device"] = str(next(model.parameters()).device)
    res["memo_after_peft"] = memo_stats()
    res["stage"] = "loaded"
    dump()

    rng = random.Random(0)
    texts = [" ".join(rng.choice(WORDS) for _ in range(rng.choice([10, 18, 26]))) + "." for _ in range(64)]
    ds = Dataset.from_dict({"text": texts})
    per_step = {}

    class CB(TrainerCallback):
        def on_step_end(self, args, state, control, **kw):
            per_step[int(state.global_step)] = memo_stats()

    cfg = SFTConfig(output_dir = str(a.work / "out"), max_steps = a.steps, per_device_train_batch_size = 2,
                    gradient_accumulation_steps = 2, learning_rate = 2e-4, logging_steps = 1, seed = 3407,
                    report_to = "none", max_length = 64, dataset_text_field = "text", optim = "adamw_torch",
                    lr_scheduler_type = "linear", warmup_steps = 0, bf16 = True, save_strategy = "no",
                    dataloader_num_workers = 0)
    trainer = SFTTrainer(model = model, processing_class = tok, train_dataset = ds, args = cfg, callbacks = [CB()])
    res["stage"] = "training"
    dump()
    with contextlib.redirect_stdout(log):
        trainer.train()
    res["losses"] = {int(h["step"]): float(h["loss"]) for h in trainer.state.log_history if "loss" in h}
    res["finite_losses"] = bool(res["losses"]) and all(math.isfinite(v) for v in res["losses"].values())
    res["memo_per_step"] = per_step
    res["memo_after_train"] = memo_stats()
    res["stage"] = "trained"
    dump()

    model.eval()
    batch = tok(["the city was first one of the new"], return_tensors = "pt").to(model.device)
    with torch.no_grad():
        logits = model(**batch).logits
    res["eval_logits_finite"] = bool(torch.isfinite(logits).all())
    res["eval_logits_shape"] = list(logits.shape)
    res["memo_after_eval"] = memo_stats()
    res["stage"] = "done"
except BaseException as e:  # noqa: BLE001
    res["error"] = f"{type(e).__name__}: {str(e)[:1500]}"
    res["traceback"] = traceback.format_exc()[-4000:]
res["log_tail"] = log.getvalue()[-1500:]
dump()
