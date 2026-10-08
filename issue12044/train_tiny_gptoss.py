#!/usr/bin/env python3
"""Tiny GPT-OSS q_proj/v_proj LoRA (r=4, alpha=8), 5 SFT steps. Observes; writes JSON to argv[1].

Run by probe.py inside the leg 2 venv with the NVIDIA spoof active. Arm via $ARM.
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys
import time
import traceback

OUT, OUTDIR = sys.argv[1], sys.argv[2]
MODEL = "trl-internal-testing/tiny-GptOssForCausalLM"
KEYS = ("gpt", "oss", "grouped", "flex", "sink", "patch", "compile", "moe", "expert")
res: dict = {"arm": os.environ.get("ARM"), "stage": "start",
             "compile_disable_env": os.environ.get("UNSLOTH_COMPILE_DISABLE")}
records: list[str] = []


class _Grab(logging.Handler):
    def emit(self, record):
        try:
            msg = record.getMessage()
        except Exception:
            return
        if any(k in msg.lower() for k in KEYS) and len(records) < 300:
            records.append(f"{record.name}: {msg[:400]}")


def dump():
    res["log_records"] = records
    with open(OUT, "w", encoding = "utf-8") as fh:
        json.dump(res, fh, indent = 2, default = str)


root = logging.getLogger()
root.addHandler(_Grab())
root.setLevel(logging.INFO)

try:
    t0 = time.time()
    import unsloth  # noqa: F401  (first, before transformers / trl)
    from unsloth import FastLanguageModel
    import torch
    res["stage"] = "imported"
    res["import_seconds"] = round(time.time() - t0, 1)
    res["unsloth_version"] = getattr(unsloth, "__version__", None)
    res["torch_version"] = torch.__version__
    res["reported_cuda"] = torch.version.cuda
    res["reported_hip"] = getattr(torch.version, "hip", None)
    res["real_hip"] = getattr(torch.version, "_amd_ci_real_hip", None)
    res["reported_device"] = torch.cuda.get_device_name(0)
    real_name = getattr(torch.cuda, "_amd_ci_real_get_device_name", None)
    res["real_device"] = real_name(0) if real_name else None
    res["reported_capability"] = list(torch.cuda.get_device_capability(0))
    res["real_is_available"] = bool(torch.cuda.is_available())

    model, tok = FastLanguageModel.from_pretrained(
        model_name = MODEL, max_seq_length = 256, load_in_4bit = False, dtype = None)
    res["stage"] = "loaded"
    res["model_dtype"] = str(getattr(model, "dtype", None))
    layer0 = model.model.layers[0]
    experts = layer0.mlp.experts
    res["experts_class"] = f"{type(experts).__module__}.{type(experts).__name__}"
    attn_cls = type(layer0.self_attn)
    res["attention_forward_module"] = getattr(attn_cls.forward, "__module__", None)
    res["attn_implementation"] = getattr(model.config, "_attn_implementation", None)
    res["experts_implementation"] = getattr(model.config, "_experts_implementation", None)
    try:
        from unsloth_zoo.temporary_patches import gpt_oss as zgo
        res["flex_sink_installed"] = bool(getattr(zgo, "_GPT_OSS_FLEX_SINK_ATTENTION_INSTALLED", False))
    except Exception as e:  # noqa: BLE001
        res["flex_sink_installed"] = f"unreadable: {type(e).__name__}: {e}"
    res["env_flex"] = os.environ.get("UNSLOTH_ENABLE_FLEX_ATTENTION")
    res["env_force_float32"] = os.environ.get("UNSLOTH_FORCE_FLOAT32")

    model = FastLanguageModel.get_peft_model(
        model, r = 4, lora_alpha = 8, target_modules = ["q_proj", "v_proj"], lora_dropout = 0,
        bias = "none", use_gradient_checkpointing = "unsloth", random_state = 3407)
    res["stage"] = "peft"
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    res["trainable_count"] = sum(p.numel() for _, p in trainable)
    res["trainable_names_sample"] = [n for n, _ in trainable][:8]
    res["trainable_non_lora"] = [n for n, _ in trainable if "lora_" not in n]
    lora0 = {n: p.detach().float().clone() for n, p in trainable}
    exp0 = {n: p.detach().float().clone() for n, p in model.named_parameters()
            if ".mlp.experts." in n}
    res["expert_param_count"] = len(exp0)
    n_layers = len(model.base_model.model.model.layers)

    from datasets import Dataset
    from transformers import TrainerCallback
    from trl import SFTConfig, SFTTrainer
    import dataclasses

    texts = [f"Question {i}: what is {i} plus {i}? Answer: {2 * i}. " * 4 for i in range(8)]
    ds = Dataset.from_dict({"text": texts})

    grads: dict = {}

    class StepOneGrads(TrainerCallback):
        def on_pre_optimizer_step(self, args, state, control, **kw):
            if state.global_step != 0 or grads:
                return
            m = kw.get("model")
            for n, p in m.named_parameters():
                if "lora_" not in n:
                    continue
                if not (f"layers.0." in n or f"layers.{n_layers - 1}." in n):
                    continue
                grads[n] = None if p.grad is None else float(p.grad.detach().float().norm())

    want = dict(output_dir = OUTDIR, per_device_train_batch_size = 1, gradient_accumulation_steps = 1,
                max_steps = 5, learning_rate = 2e-4, lr_scheduler_type = "constant", warmup_steps = 0,
                weight_decay = 0.0, logging_steps = 1, optim = "adamw_torch", seed = 3407,
                report_to = "none", save_strategy = "no", dataset_text_field = "text",
                max_length = 128, max_seq_length = 128, dataloader_num_workers = 0,
                dataset_num_proc = 1)
    fields = {f.name for f in dataclasses.fields(SFTConfig)}
    cfg = SFTConfig(**{k: v for k, v in want.items() if k in fields})
    try:
        trainer = SFTTrainer(model = model, processing_class = tok, train_dataset = ds, args = cfg,
                             callbacks = [StepOneGrads()])
    except TypeError:
        trainer = SFTTrainer(model = model, tokenizer = tok, train_dataset = ds, args = cfg,
                             callbacks = [StepOneGrads()])
    res["stage"] = "trainer"
    res["trainer_class"] = type(trainer).__name__
    t1 = time.time()
    trainer.train()
    res["stage"] = "trained"
    res["train_seconds"] = round(time.time() - t1, 1)
    hist = trainer.state.log_history
    res["losses"] = [h["loss"] for h in hist if "loss" in h]
    res["grad_norms"] = [h.get("grad_norm") for h in hist if "loss" in h]
    res["losses_finite"] = bool(res["losses"]) and all(
        isinstance(x, (int, float)) and math.isfinite(x) for x in res["losses"])
    res["step1_lora_grad_norms"] = grads
    res["step1_lora_B_nonzero"] = {n: (g is not None and g > 0) for n, g in grads.items() if "lora_B" in n}
    after = {n: p.detach().float() for n, p in model.named_parameters() if n in lora0}
    res["lora_delta_max"] = max(float((after[n] - lora0[n]).abs().max()) for n in lora0) if lora0 else None
    exp1 = {n: p.detach().float() for n, p in model.named_parameters() if n in exp0}
    res["experts_max_change"] = max(float((exp1[n] - exp0[n]).abs().max()) for n in exp0) if exp0 else None
    try:
        res["grouped_ready_flag"] = getattr(model.base_model.model.model.layers[0].mlp.experts,
                                            "_unsloth_grouped_ready", None)
    except Exception:
        res["grouped_ready_flag"] = None
    res["ok"] = True
except BaseException as e:  # noqa: BLE001
    res["ok"] = False
    res["exc_type"] = type(e).__name__
    res["exc"] = str(e)[:2000]
    res["traceback"] = traceback.format_exc()[-6000:]
finally:
    dump()
