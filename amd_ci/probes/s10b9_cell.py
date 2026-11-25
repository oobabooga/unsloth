#!/usr/bin/env python3
"""One timed repeat: Qwen3-0.6B 16-bit LoRA, use_gradient_checkpointing="unsloth", UNSLOTH_GC_SAVE_ATTENTION UNSET
(automatic mode), fixed-length random-token batches of --seq tokens. Observes only; JSON via --out.

Records: per-step loss (float + hex, bitwise comparable), per-step seconds (synchronised), peak memory since load,
SAVED_ATTENTION_STATS after warmup and after the timed steps, saved_attention_active() as seen by a forward pre-hook on
every self_attn during the warmup steps (hooks removed before the timed steps), and which unsloth / unsloth_zoo files
were imported (the probe checks they are this arm's trees).
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--out", required = True, type = Path)
ap.add_argument("--model", default = "unsloth/Qwen3-0.6B")
ap.add_argument("--seq", type = int, default = 8192)
ap.add_argument("--warmup", type = int, default = 3)
ap.add_argument("--steps", type = int, default = 10)
args = ap.parse_args()

for k in ("UNSLOTH_GC_SAVE_ATTENTION", "UNSLOTH_GC_SAVE_ATTENTION_MIN_SEQ"):
    os.environ.pop(k, None)  # automatic mode is the thing under test

res: dict = {"stage": "start", "seq": args.seq, "warmup": args.warmup, "steps": args.steps, "model": args.model,
             "env_save_attention": os.environ.get("UNSLOTH_GC_SAVE_ATTENTION")}


def dump():
    args.out.write_text(json.dumps(res, indent = 2, default = str), encoding = "utf-8")


def stats(sa):
    return dict(getattr(sa, "SAVED_ATTENTION_STATS", {}) or {}) if sa is not None else None


try:
    dump()
    import unsloth  # noqa: F401  (before transformers)
    from unsloth import FastLanguageModel
    import torch
    import unsloth_zoo
    res["unsloth_file"] = unsloth.__file__
    res["zoo_file"] = unsloth_zoo.__file__
    p = torch.cuda.get_device_properties(0)
    res["arch"] = getattr(p, "gcnArchName", None)
    res["hip"] = torch.version.hip
    res["torch"] = torch.__version__
    try:
        sa = importlib.import_module("unsloth_zoo.saved_attention")
    except Exception as e:  # noqa: BLE001
        sa = None
        res["saved_attention_module"] = f"absent ({type(e).__name__}: {e})"
    ad = importlib.import_module("unsloth.utils.attention_dispatch")
    res["dispatch"] = {k: repr(getattr(ad, k, "<absent>")) for k in
                       ("_SAVE_ATTENTION", "_SAVE_ATTENTION_MIN_SEQ", "_ROCM_SDPA_GUARD")}
    if sa is not None:
        res["auto_active_flag"] = list(getattr(sa, "_AUTO_ACTIVE", [None]))
        res["auto_min_seq_rocm_table"] = getattr(sa, "_AUTO_MIN_SEQ_ROCM", "<absent>")
        res["selector"] = repr(getattr(sa, "_ROCM_SDPA_SELECT", None))
    res["stage"] = "load"
    dump()
    model, tok = FastLanguageModel.from_pretrained(args.model, max_seq_length = args.seq, dtype = torch.bfloat16,
                                                   load_in_4bit = False)
    model = FastLanguageModel.get_peft_model(
        model, r = 16, lora_alpha = 16, lora_dropout = 0, bias = "none", random_state = 3407,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing = "unsloth")
    model.train()
    if sa is not None:
        res["selector_after_load"] = repr(getattr(sa, "_ROCM_SDPA_SELECT", None))
    vocab = int(model.config.vocab_size)
    g = torch.Generator().manual_seed(1234)
    batches = [torch.randint(100, min(vocab, 150000) - 100, (1, args.seq), generator = g)
               for _ in range(args.warmup + args.steps)]
    torch.manual_seed(0)
    params = [q for q in model.parameters() if q.requires_grad]
    res["trainable_tensors"] = len(params)
    opt = torch.optim.AdamW(params, lr = 1e-4)
    seen = {"active": 0, "inactive": 0, "active_no_grad": 0}

    def pre_hook(mod, a, kw):
        on = bool(sa.saved_attention_active()) if sa is not None and hasattr(sa, "saved_attention_active") else False
        seen["active" if on else "inactive"] += 1
        if on and not torch.is_grad_enabled():
            seen["active_no_grad"] += 1

    handles = [m.register_forward_pre_hook(pre_hook, with_kwargs = True)
               for n, m in model.named_modules() if n.endswith("self_attn")]
    res["hooked_attn_modules"] = len(handles)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    res["mem_after_load_gb"] = torch.cuda.memory_allocated() / 2**30
    losses, hexes, secs = [], [], []
    res["stage"] = "train"
    dump()
    for i, ids in enumerate(batches):
        if i == args.warmup:
            for h in handles:
                h.remove()
            res["active_during_warmup"] = dict(seen)
            res["stats_after_warmup"] = stats(sa)
        ids = ids.cuda()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        loss = model(input_ids = ids, labels = ids).loss
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none = True)
        torch.cuda.synchronize()
        secs.append(time.perf_counter() - t0)
        v = float(loss.detach().float().item())
        losses.append(v)
        hexes.append(v.hex())
    res["losses"], res["loss_hex"], res["step_s"] = losses, hexes, secs
    res["timed_s"] = secs[args.warmup:]
    res["finite"] = all(math.isfinite(x) for x in losses)
    res["stats_final"] = stats(sa)
    res["peak_alloc_gb"] = torch.cuda.max_memory_allocated() / 2**30
    res["peak_reserved_gb"] = torch.cuda.max_memory_reserved() / 2**30
    if sa is not None:
        res["auto_active_flag_end"] = list(getattr(sa, "_AUTO_ACTIVE", [None]))
        for name, call in (("rocm_sdpa_is_kept", lambda: sa._rocm_sdpa_is_kept(torch.device("cuda", 0))),
                           ("auto_min_seq_device", lambda: sa.save_attention_auto_min_seq(torch.device("cuda", 0)))):
            try:
                res[name] = call()
            except Exception as e:  # noqa: BLE001
                res[name] = f"error {type(e).__name__}: {e}"
    res["stage"] = "done"
except Exception as e:  # noqa: BLE001
    res["error"] = f"{type(e).__name__}: {e}"
    res["traceback"] = traceback.format_exc()[-6000:]
dump()
sys.exit(0)
