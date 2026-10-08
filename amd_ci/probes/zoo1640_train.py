#!/usr/bin/env python3
"""One process of LoRA training runs for unsloth-zoo PR 1640 (moe_ready_epoch). Observes only; JSON to --out.

PYTHONPATH = the state's unsloth-zoo checkout first. Tiny Qwen3-MoE through unsloth FastLanguageModel (bf16 or
4-bit NF4 via --load4bit), LoRA r=8 on q/k/v/o/gate/up/down (regex), use_gradient_checkpointing="unsloth",
transformers.Trainer (GA 2, max_grad_norm 1.0, constant LR, adamw_torch), 6 optimizer steps on fixed data.

Three sequential runs IN THIS PROCESS (fresh model each): fast/r1 (UNSLOTH_MOE_FAST_READY unset), off
(UNSLOTH_MOE_FAST_READY=0, read per call by the head), fast/r2. In-process comparison sidesteps the ROCm
cross-process non-determinism; the caller runs this process twice per state for a cross-process A/A.

Per run: per-micro-batch loss repr, per-step trainable digest, Trainer grad_norm logs, final digest,
moe_ready_epoch.COUNTS delta per optimizer step (None where the module is absent, i.e. at base), the experts class
and which forward the sparse MoE blocks run.
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
RUNS = (("fast/r1", None), ("off", "0"), ("fast/r2", None))


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
    ap.add_argument("--load4bit", action = "store_true")
    args = ap.parse_args()
    os.environ["UNSLOTH_COMPILE_LOCATION"] = str(args.out.parent / f"cache_{args.out.stem}")
    os.environ.pop("UNSLOTH_MOE_FAST_READY", None)
    res: dict = {"stage": "import", "model": args.model, "load4bit": args.load4bit, "runs": {},
                 "env": {k: os.environ.get(k) for k in ("TORCHDYNAMO_DISABLE", "UNSLOTH_MOE_FAST_READY")}}

    def dump():
        args.out.write_text(json.dumps(res, indent = 2, default = str), encoding = "utf-8")

    try:
        import unsloth  # noqa: F401  (before transformers)
        from unsloth import FastLanguageModel
        import torch
        import transformers
        import unsloth_zoo
        from transformers import Trainer, TrainingArguments, TrainerCallback, default_data_collator
        from datasets import Dataset
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
            from unsloth_zoo.temporary_patches import moe_ready_epoch as R
            res["moe_ready_epoch_present"] = True
        except ImportError:
            R = None
            res["moe_ready_epoch_present"] = False

        def counts():
            return dict(R.COUNTS) if R is not None else None

        def delta(a, b):
            return None if a is None or b is None else {k: b[k] - a.get(k, 0) for k in b}

        for key, fast_env in RUNS:
            rec: dict = {"stage": "load"}
            res["runs"][key] = rec
            if fast_env is None:
                os.environ.pop("UNSLOTH_MOE_FAST_READY", None)
            else:
                os.environ["UNSLOTH_MOE_FAST_READY"] = fast_env
            rec["UNSLOTH_MOE_FAST_READY"] = os.environ.get("UNSLOTH_MOE_FAST_READY")
            dump()
            try:
                c_load = counts()
                model, tok = FastLanguageModel.from_pretrained(
                    args.model, max_seq_length = 128, dtype = torch.bfloat16, load_in_4bit = args.load4bit,
                    device_map = {"": 0})
                rec["experts_class"] = sorted({type(m).__name__ for n, m in model.named_modules() if n.endswith(".experts")})
                rec["stage"] = "peft"
                model = FastLanguageModel.get_peft_model(
                    model, r = 8, lora_alpha = 16, lora_dropout = 0, bias = "none", target_modules = TARGETS,
                    use_gradient_checkpointing = "unsloth", random_state = 3407)
                blocks = set()
                for n, m in model.named_modules():
                    if type(m).__name__.endswith("SparseMoeBlock"):
                        f = type(m).forward
                        fi = m.__dict__.get("forward")
                        blocks.add(f"{type(m).__name__}: {getattr(f, '__module__', '?')}.{getattr(f, '__qualname__', '?')}"
                                   + (f" inst:{getattr(fi, '__module__', '?')}" if fi is not None else ""))
                rec["moe_block_forwards"] = sorted(blocks)
                lora = sorted((n, p) for n, p in model.named_parameters() if p.requires_grad)
                with torch.no_grad():
                    for n, p in lora:
                        g = torch.Generator().manual_seed(zlib.crc32(n.encode()))
                        p.copy_((torch.randn(tuple(p.shape), generator = g) * 0.02).to(p.dtype))
                rec["n_trainable_tensors"] = len(lora)
                rec["n_expert_lora_tensors"] = sum(1 for n, _ in lora if ".experts." in n or "experts" in n)
                rec["digest_init"] = _digest(lora)
                rec["counts_load_peft"] = delta(c_load, counts())

                vocab = int(model.config.vocab_size)
                g = torch.Generator().manual_seed(0)
                ids = torch.randint(4, max(5, vocab - 1), (2 * 2 * args.steps, 64), generator = g).tolist()
                ds = Dataset.from_dict({"input_ids": ids, "labels": ids, "attention_mask": [[1] * 64 for _ in ids]})
                losses, step_digests, logs, step_counts = [], [], [], []
                snap = [None]

                class T(Trainer):
                    def compute_loss(self, model, inputs, *a, **k):
                        out = super().compute_loss(model, inputs, *a, **k)
                        l = out[0] if isinstance(out, tuple) else out
                        losses.append(repr(float(l.detach())))
                        return out

                class CB(TrainerCallback):
                    def on_step_begin(self, a, state, control, **kw):
                        snap[0] = counts()

                    def on_step_end(self, a, state, control, **kw):
                        step_counts.append(delta(snap[0], counts()))
                        step_digests.append(_digest(sorted((n, p) for n, p in model.named_parameters() if p.requires_grad)))

                    def on_log(self, a, state, control, logs = None, **kw):
                        if logs:
                            r = {k: (repr(float(v)) if v is not None else None)
                                 for k, v in logs.items() if k in ("loss", "grad_norm")}
                            if r:
                                logs_list.append(r)

                logs_list = logs
                targs = TrainingArguments(
                    output_dir = str(args.out.parent / f"trainer_{args.out.stem}"), per_device_train_batch_size = 2,
                    gradient_accumulation_steps = 2, max_steps = args.steps, learning_rate = 5e-3,
                    lr_scheduler_type = "constant", warmup_steps = 0, max_grad_norm = 1.0, logging_steps = 1,
                    save_strategy = "no", report_to = [], bf16 = True, seed = 3407, data_seed = 3407,
                    optim = "adamw_torch", dataloader_num_workers = 0, remove_unused_columns = False)
                trainer = T(model = model, args = targs, train_dataset = ds, data_collator = default_data_collator,
                            callbacks = [CB()])
                rec["stage"] = "train"
                dump()
                c0 = counts()
                trainer.train()
                torch.cuda.synchronize()
                rec["counts_train_total"] = delta(c0, counts())
                rec["counts_per_step"] = step_counts
                rec["loss_reprs"] = losses
                rec["step_digests"] = step_digests
                rec["logs"] = logs
                rec["final_digest"] = _digest(sorted((n, p) for n, p in model.named_parameters() if p.requires_grad))
                rec["stage"] = "done"
                rec["ok"] = bool(losses) and all(float(v) == float(v) and abs(float(v)) != float("inf") for v in losses)
                del trainer, model
                torch.cuda.empty_cache()
            except Exception as e:  # noqa: BLE001
                rec["error"] = f"{type(e).__name__}: {str(e)[:1500]}"
                rec["traceback"] = traceback.format_exc()[-4000:]
            dump()
        os.environ.pop("UNSLOTH_MOE_FAST_READY", None)
        res["stage"] = "done"
    except Exception as e:  # noqa: BLE001
        res["error"] = f"{type(e).__name__}: {str(e)[:1500]}"
        res["traceback"] = traceback.format_exc()[-4000:]
    dump()
    return 0


if __name__ == "__main__":
    sys.exit(main())
