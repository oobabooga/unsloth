#!/usr/bin/env python3
"""One HF Trainer LoRA cell for unsloth-zoo PR 1628 (fast grad params). Observes only; writes JSON to --out.

Run in its own process per cell, PYTHONPATH = the state's unsloth-zoo checkout first.
Tiny Qwen3-MoE (trl-internal-testing/tiny-Qwen3MoeForCausalLM) through unsloth FastLanguageModel (bf16), LoRA on
attention + expert projections, transformers.Trainer with GA 2, max_grad_norm 1.0, constant LR, 6 optimizer steps
on fixed data. Records:
  * per-micro-batch loss repr (compute_loss), per-optimizer-step trainable-param digest, Trainer grad_norm logs
  * final trainable-param digest
  * fast_grad_params presence, FAST_GRAD_CALLS before / after train (zero_grad / clip / rebuild), model flagged
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import traceback
import zlib
from pathlib import Path

MODEL = "trl-internal-testing/tiny-Qwen3MoeForCausalLM"
TARGETS = r".*(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"


def _digest(tensors) -> str:
    h = hashlib.sha256()
    for n, t in tensors:
        h.update(n.encode())
        h.update(b"<none>" if t is None else t.detach().float().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--steps", type = int, default = 6)
    ap.add_argument("--model", default = MODEL)
    args = ap.parse_args()
    os.environ["UNSLOTH_COMPILE_LOCATION"] = str(args.out.parent / f"cache_{args.out.stem}")
    res: dict = {"stage": "import", "model": args.model,
                 "env": {k: os.environ.get(k) for k in ("UNSLOTH_FAST_GRAD_PARAMS", "TORCHDYNAMO_DISABLE")}}

    def dump():
        args.out.write_text(json.dumps(res, indent = 2, default = str), encoding = "utf-8")

    try:
        import unsloth  # noqa: F401  (before transformers)
        from unsloth import FastLanguageModel
        import torch
        import transformers
        import unsloth_zoo
        from transformers import Trainer, TrainingArguments, TrainerCallback
        res["unsloth_zoo_file"] = unsloth_zoo.__file__
        res["unsloth_file"] = unsloth.__file__
        res["versions"] = {"torch": torch.__version__, "hip": getattr(torch.version, "hip", None),
                           "cuda": getattr(torch.version, "cuda", None), "transformers": transformers.__version__}
        for mod in ("peft", "trl", "accelerate", "bitsandbytes"):
            try:
                res["versions"][mod] = __import__(mod).__version__
            except Exception as e:  # noqa: BLE001
                res["versions"][mod] = f"unavailable: {e!r}"
        try:
            props = torch.cuda.get_device_properties(0)
            res["versions"]["device"] = props.name
            res["versions"]["gcnArchName"] = getattr(props, "gcnArchName", None)
        except Exception as e:  # noqa: BLE001
            res["versions"]["device_error"] = repr(e)
        try:
            from unsloth_zoo import fast_grad_params as F
            res["fast_grad_params_present"] = True
        except ImportError:
            F = None
            res["fast_grad_params_present"] = False

        res["stage"] = "load"
        dump()
        model, tok = FastLanguageModel.from_pretrained(args.model, max_seq_length = 128, dtype = torch.bfloat16,
                                                       load_in_4bit = False, device_map = {"": 0})
        res["experts_class"] = sorted({type(m).__name__ for n, m in model.named_modules() if n.endswith(".experts")})
        res["stage"] = "peft"
        dump()
        model = FastLanguageModel.get_peft_model(
            model, r = 8, lora_alpha = 16, lora_dropout = 0, bias = "none", target_modules = TARGETS,
            use_gradient_checkpointing = "unsloth", random_state = 3407)
        res["model_flagged"] = bool(model.__dict__.get("_unsloth_fast_grad_params", False))
        lora = sorted((n, p) for n, p in model.named_parameters() if p.requires_grad)
        with torch.no_grad():
            for n, p in lora:
                if "lora_B" in n:
                    g = torch.Generator().manual_seed(zlib.crc32(n.encode()))
                    p.copy_((torch.randn(tuple(p.shape), generator = g) * 0.02).to(p.dtype))
        res["n_trainable_tensors"] = len(lora)
        res["n_weight_stack_params"] = sum(1 for n, _ in lora if n.endswith("weight_stack"))
        res["digest_init"] = _digest(lora)

        from datasets import Dataset
        vocab = int(model.config.vocab_size)
        g = torch.Generator().manual_seed(0)
        ids = torch.randint(4, max(5, vocab - 1), (2 * 2 * args.steps, 64), generator = g).tolist()
        ds = Dataset.from_dict({"input_ids": ids, "labels": ids, "attention_mask": [[1] * 64 for _ in ids]})

        losses, step_digests, logs = [], [], []

        class T(Trainer):
            def compute_loss(self, model, inputs, *a, **k):
                out = super().compute_loss(model, inputs, *a, **k)
                l = out[0] if isinstance(out, tuple) else out
                losses.append(repr(float(l.detach())))
                return out

        class CB(TrainerCallback):
            def on_step_end(self, a, state, control, **kw):
                step_digests.append(_digest(sorted((n, p) for n, p in model.named_parameters() if p.requires_grad)))

            def on_log(self, a, state, control, logs = None, **kw):
                if logs:
                    rec = {k: v for k, v in logs.items() if k in ("loss", "grad_norm")}
                    rec = {k: (repr(float(v)) if v is not None else None) for k, v in rec.items()}
                    if rec:
                        logs_list.append(rec)

        logs_list = logs
        targs = TrainingArguments(
            output_dir = str(args.out.parent / f"trainer_{args.out.stem}"), per_device_train_batch_size = 2,
            gradient_accumulation_steps = 2, max_steps = args.steps, learning_rate = 5e-3, lr_scheduler_type = "constant",
            warmup_steps = 0, max_grad_norm = 1.0, logging_steps = 1, save_strategy = "no", report_to = [], bf16 = True,
            seed = 3407, data_seed = 3407, optim = "adamw_torch", dataloader_num_workers = 0, remove_unused_columns = False,
        )
        from transformers import default_data_collator
        trainer = T(model = model, args = targs, train_dataset = ds, data_collator = default_data_collator,
                    callbacks = [CB()])
        c0 = dict(F.FAST_GRAD_CALLS) if F is not None else None
        res["stage"] = "train"
        dump()
        trainer.train()
        torch.cuda.synchronize()
        res["calls_delta"] = {k: F.FAST_GRAD_CALLS[k] - c0.get(k, 0) for k in F.FAST_GRAD_CALLS} if F is not None else None
        res["loss_reprs"] = losses
        res["step_digests"] = step_digests
        res["logs"] = logs
        res["final_digest"] = _digest(sorted((n, p) for n, p in model.named_parameters() if p.requires_grad))
        res["stage"] = "done"
        res["ok"] = bool(losses) and all(float(v) == float(v) and abs(float(v)) != float("inf") for v in losses)
    except Exception as e:  # noqa: BLE001
        res["error"] = f"{type(e).__name__}: {str(e)[:1500]}"
        res["traceback"] = traceback.format_exc()[-4000:]
    dump()
    return 0


if __name__ == "__main__":
    sys.exit(main())
