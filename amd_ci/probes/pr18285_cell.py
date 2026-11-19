#!/usr/bin/env python3
"""One cell of the unsloth #18285 smoke (NF4 dequant cache for QLoRA): tiny 4-bit LoRA SFT. Observes only; JSON to
--out (never stdout).

FastLanguageModel.from_pretrained(<bnb-4bit model>), get_peft_model r=16 on all 7 projections,
use_gradient_checkpointing="unsloth", SFTTrainer 8 steps, batch 2, fixed seed, fixed random-token data (every row
truncated to the same length so each step has the same matmul row count), logging_steps=1.
Records per-step losses and grad norms, torch.cuda.max_memory_allocated, the bitsandbytes version / NF4 probe, and,
when the state has it, unsloth.kernels.dequant_cache: cache_mode(), STATS, cached_count(), _BUDGET, _DISABLED.
UNSLOTH_NF4_DEQUANT_CACHE is set by the caller (probe) per cell.
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

ap = argparse.ArgumentParser()
ap.add_argument("--model", default = "unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit")
ap.add_argument("--steps", type = int, default = 8)
ap.add_argument("--seq", type = int, default = 128)
ap.add_argument("--seed", type = int, default = 3407)
ap.add_argument("--work", required = True, type = Path)
ap.add_argument("--out", required = True, type = Path)
a = ap.parse_args()
res: dict = {"model": a.model, "stage": "start", "env_cache": os.environ.get("UNSLOTH_NF4_DEQUANT_CACHE")}


def dump():
    a.out.write_text(json.dumps(res, indent = 1, default = str), encoding = "utf-8")


log = io.StringIO()
try:
    import unsloth  # noqa: F401  (before transformers / trl)
    from unsloth import FastLanguageModel, is_bfloat16_supported
    import torch
    import transformers
    import trl
    import peft
    import unsloth_zoo
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    res["unsloth_file"] = unsloth.__file__
    v = {"torch": torch.__version__, "hip": torch.version.hip, "cuda": torch.version.cuda,
         "transformers": transformers.__version__, "trl": trl.__version__, "peft": peft.__version__,
         "unsloth_zoo": unsloth_zoo.__version__, "python": sys.version.split()[0], "platform": sys.platform}
    try:
        import bitsandbytes as bnb
        v["bitsandbytes"] = bnb.__version__
    except Exception as e:  # noqa: BLE001
        v["bitsandbytes"] = f"ERR {e!r}"[:300]
    try:
        v["device"] = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        v["arch"] = getattr(props, "gcnArchName", None) or f"sm{props.major}{props.minor}"
        v["total_mem"] = props.total_memory
    except Exception as e:  # noqa: BLE001
        v["device"] = f"ERR {e}"
    res["versions"] = v
    try:
        import unsloth.kernels.dequant_cache as dc
        res["dc_module"] = dc.__file__
        res["dc_mode"] = dc.cache_mode()
    except ImportError as e:
        res["dc_module"] = None
        res["dc_import_error"] = repr(e)[:300]
        dc = None
    res["stage"] = "imported"
    dump()

    # bitsandbytes NF4 on its own: quantize / dequantize round trip on the GPU (host capability, both arms).
    try:
        import bitsandbytes.functional as F
        w = torch.randn(256, 256, device = "cuda", dtype = torch.float16)
        q, st = F.quantize_4bit(w, quant_type = "nf4")
        d = F.dequantize_4bit(q, st)
        res["bnb_nf4_roundtrip"] = {"ok": bool(torch.isfinite(d).all()),
                                    "rel_err": float((d.float() - w.float()).norm() / w.float().norm())}
    except Exception as e:  # noqa: BLE001
        res["bnb_nf4_roundtrip"] = {"ok": False, "error": f"{type(e).__name__}: {str(e)[:400]}"}
    dump()

    model, tok = FastLanguageModel.from_pretrained(a.model, max_seq_length = a.seq, load_in_4bit = True, dtype = None)
    model = FastLanguageModel.get_peft_model(
        model, r = 16, lora_alpha = 16, lora_dropout = 0, bias = "none",
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing = "unsloth", random_state = a.seed)
    params = [p for p in model.parameters() if p.requires_grad]
    res["n_trainable_tensors"] = len(params)
    res["model_device"] = str(params[0].device)
    try:
        import bitsandbytes as bnb
        res["n_linear4bit"] = sum(isinstance(m, bnb.nn.Linear4bit) for m in model.modules())
    except Exception:  # noqa: BLE001
        res["n_linear4bit"] = None
    res["stage"] = "loaded"
    dump()

    rng = random.Random(a.seed)
    vocab = min(len(tok), 30000)
    texts = [tok.decode([rng.randrange(1000, vocab) for _ in range(3 * a.seq)], skip_special_tokens = True) for _ in range(64)]
    ds = Dataset.from_dict({"text": texts})
    bf16 = is_bfloat16_supported()
    cfg = SFTConfig(output_dir = str(a.work / "out"), max_steps = a.steps, per_device_train_batch_size = 2,
                    gradient_accumulation_steps = 1, learning_rate = 2e-4, logging_steps = 1, seed = a.seed,
                    data_seed = a.seed, report_to = "none", max_length = a.seq, dataset_text_field = "text",
                    optim = "adamw_torch", lr_scheduler_type = "linear", warmup_steps = 1, weight_decay = 0.0,
                    bf16 = bf16, fp16 = not bf16, save_strategy = "no", packing = False, dataloader_num_workers = 0)
    res["bf16"] = bf16
    trainer = SFTTrainer(model = model, processing_class = tok, train_dataset = ds, args = cfg)
    res["stage"] = "training"
    dump()
    torch.cuda.reset_peak_memory_stats()
    with contextlib.redirect_stdout(log):
        trainer.train()
    res["max_memory_allocated"] = torch.cuda.max_memory_allocated()
    hist = trainer.state.log_history
    res["losses"] = {int(h["step"]): float(h["loss"]) for h in hist if "loss" in h}
    res["grad_norms"] = {int(h["step"]): float(h["grad_norm"]) for h in hist if "grad_norm" in h}
    res["trainer_class"] = type(trainer).__name__
    res["finite_lora"] = all(bool(torch.isfinite(p).all()) for p in params)
    if dc is not None:
        res["dc_stats"] = {k: (int(x) if isinstance(x, (int, float)) and float(x).is_integer() else x) for k, x in dict(dc.STATS).items()}
        res["dc_cached_count"] = dc.cached_count()
        res["dc_budget"] = {str(k): int(x) for k, x in dict(getattr(dc, "_BUDGET", {})).items()}
        res["dc_disabled"] = getattr(dc, "_DISABLED", None)
        res["dc_foreign"] = {str(k): int(x) for k, x in dict(getattr(dc, "_FOREIGN", {})).items()}
        try:
            free, total = torch.cuda.mem_get_info(0)
            res["mem_get_info_end"] = [int(free), int(total)]
        except Exception as e:  # noqa: BLE001
            res["mem_get_info_end"] = repr(e)[:200]
    res["stage"] = "done"
except BaseException as e:  # noqa: BLE001
    res["error"] = f"{type(e).__name__}: {str(e)[:1500]}"
    res["traceback"] = traceback.format_exc()[-4000:]
res["train_stdout_tail"] = log.getvalue()[-1500:]
dump()
