#!/usr/bin/env python3
"""W5 cross-platform cells: fused CE loss, GA normalisation, offload_layers block swap and the gpt-oss MXFP4
merge_and_unload export, on tiny models. Self-contained (no workspace imports) so it runs on the AMD gfx1151
runners and on Colab / Kaggle VMs as well as the B200. Observes only: writes JSON to --out, prints nothing it parses.

One model per process (Unsloth patches leak across models):
  python w5_cells.py --model llama --out llama.json [--cells fused_ce,ga_norm,offload,offload_qlora,gptoss_export]

Checks per cell (each records its numbers; "ok" is True / False / None = not applicable):
  fused_ce        training loss through Unsloth's fused CE == fp32 CE of the same model's eval logits (gemma2:
                  final_logit_softcapping 3.0 written into a local copy so the softcap actually bites; the
                  output head is scaled so logits have std ~2, else every loss is log(vocab)).
  ga_norm         2 micro-batches with num_items_in_batch == the full batch's loss and LoRA gradient.
  offload         get_peft_model(offload_layers = 1) vs 0: equal eval logits, loss and LoRA grads, swap engaged
                  (BlockSwap installed, >= 2 host-to-device copies per swapped layer).
  offload_qlora   from_pretrained(load_in_4bit, offload_layers = 1) + Unsloth GC vs resident NF4.
  gptoss_export   16-bit tiny gpt-oss, LoRA on attention + experts, merge_and_unload().save_pretrained(): no
                  *_blocks / *_scales keys in the export (zoo fix 1176505a8; transformers 5.x only).
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import tempfile
import time
import traceback

MODELS = {
    "llama": "trl-internal-testing/tiny-LlamaForCausalLM-3.2",
    "qwen3": "trl-internal-testing/tiny-Qwen3ForCausalLM",
    "gemma2": "trl-internal-testing/tiny-Gemma2ForCausalLM",
    "qwen3moe": "trl-internal-testing/tiny-Qwen3MoeForCausalLM",
    "gptoss": "trl-internal-testing/tiny-GptOssForCausalLM",
}
DEFAULT_CELLS = {
    "llama": "fused_ce,ga_norm,offload,offload_qlora",
    "qwen3": "fused_ce,ga_norm,offload",
    "gemma2": "fused_ce,offload",
    "qwen3moe": "fused_ce,ga_norm,offload,offload_qlora",
    "gptoss": "gptoss_export,offload",
}
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def _hash(s):
    h = 0
    for c in s:
        h = (h * 131 + ord(c)) & 0xFFFFFFFF
    return h


def seed_lora_b(model, seed = 0, std = 0.02):
    import torch
    for name, mod in sorted(model.named_modules(), key = lambda t: t[0]):
        lb = getattr(mod, "lora_B", None)
        if lb is None or not hasattr(lb, "keys"):
            continue
        for k in lb.keys():
            w = lb[k].weight
            g = torch.Generator().manual_seed(seed + _hash(name) % 100003)
            with torch.no_grad():
                w.copy_((torch.randn(w.shape, generator = g) * std).to(w.dtype))


def fixed_batch(vocab, bs = 2, seq = 40, pad_right = 9, seed = 0):
    import torch
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(10, max(11, min(vocab, 32000)), (bs, seq), generator = g)
    mask = torch.ones_like(ids)
    ids[-1, seq - pad_right:] = 0
    mask[-1, seq - pad_right:] = 0
    labels = ids.clone()
    labels[mask == 0] = -100
    return {k: v.to("cuda") for k, v in dict(input_ids = ids, attention_mask = mask, labels = labels).items()}


def text_vocab(model):
    cfg = model.config
    cfg = cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg
    return int(cfg.vocab_size)


def local_copy(repo, overrides = None):
    """Snapshot into a temp dir (optionally editing config.json) so a knob can be set without touching the cache."""
    from huggingface_hub import snapshot_download
    src = snapshot_download(repo)
    if not overrides:
        return src
    import shutil
    dst = tempfile.mkdtemp(prefix = "w5_model_")
    for f in os.listdir(src):
        shutil.copy(os.path.join(src, f), os.path.join(dst, f))
    p = os.path.join(dst, "config.json")
    cfg = json.load(open(p, encoding = "utf-8"))
    cfg.update(overrides)
    json.dump(cfg, open(p, "w", encoding = "utf-8"))
    return dst


def _dtype():
    import torch
    # T4 (sm75) has no bf16: fp16 there, as a user's run would pick.
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def load(path, four_bit = False, **kw):
    from unsloth import FastModel
    model, tok = FastModel.from_pretrained(model_name = path, max_seq_length = 256, dtype = _dtype(),
                                           load_in_4bit = four_bit, **kw)
    return model


def add_lora(model, offload = 0, gc_mode = "unsloth", targets = None):
    from unsloth import FastModel
    model = FastModel.get_peft_model(model, r = 8, lora_alpha = 16, lora_dropout = 0.0, bias = "none",
                                     target_modules = targets or TARGETS, random_state = 0,
                                     use_gradient_checkpointing = gc_mode, offload_layers = offload)
    seed_lora_b(model)
    return model


def eval_logits(model, b):
    import torch
    was = model.training
    model.eval()
    with torch.no_grad():
        lg = model(input_ids = b["input_ids"], attention_mask = b["attention_mask"]).logits.float()
    if was:
        model.train()
    return lg


def scale_head(model, b, target_std = 2.0):
    import torch
    lg = eval_logits(model, b)[b["attention_mask"].bool()]
    s = float((lg - lg.mean(-1, keepdim = True)).std())
    gain = float(min(max(target_std / max(s, 1e-6), 1.0), 1e5))
    head = model.get_output_embeddings()
    with torch.no_grad():
        head.weight = torch.nn.Parameter(head.weight.detach().clone() * gain, requires_grad = False)
    return gain


def lora_grads(model):
    return {n: p.grad.detach().float().cpu().clone() for n, p in model.named_parameters()
            if p.requires_grad and p.grad is not None}


def grad_rel(g0, g1):
    import torch
    keys = sorted(set(g0) & set(g1))
    a = torch.cat([g0[k].flatten() for k in keys]) if keys else torch.zeros(1)
    b = torch.cat([g1[k].flatten() for k in keys]) if keys else torch.zeros(1)
    return float((a - b).norm() / a.norm().clamp_min(1e-12)), float(a.norm()), sorted(set(g0) ^ set(g1))[:4]


def step(model, b):
    import torch
    lg = eval_logits(model, b)
    model.train()
    model.zero_grad(set_to_none = True)
    out = model(**b)
    out.loss.backward()
    torch.cuda.synchronize()
    return lg.cpu(), float(out.loss.detach()), lora_grads(model)


def free(*objs):
    import torch
    for o in objs:
        del o
    gc.collect()
    torch.cuda.empty_cache()


def cell_fused_ce(key):
    import torch
    over = {"final_logit_softcapping": 3.0} if key == "gemma2" else None
    model = add_lora(load(local_copy(MODELS[key], over)), gc_mode = False)
    b = fixed_batch(text_vocab(model))
    gain = scale_head(model, b)
    lg = eval_logits(model, b)
    shift_lg, shift_lab = lg[:, :-1].reshape(-1, lg.shape[-1]), b["labels"][:, 1:].reshape(-1)
    ref = float(torch.nn.functional.cross_entropy(shift_lg, shift_lab, ignore_index = -100))
    model.train()
    out = model(**b)
    fused = float(out.loss.detach())
    # The fused path hands back no logits (or a placeholder) unless UNSLOTH_RETURN_LOGITS=1.
    returned = type(out.logits).__name__ + (str(tuple(out.logits.shape)) if hasattr(out.logits, "shape") else "")
    rel = abs(fused - ref) / max(abs(ref), 1e-12)
    free(model)
    return dict(ok = rel <= 2e-3, fused_loss = fused, ref_loss_fp32 = ref, rel = rel, gain = gain,
                softcap = (over or {}).get("final_logit_softcapping"), logits_returned = returned)


def cell_ga_norm(key):
    import torch
    model = add_lora(load(local_copy(MODELS[key])), gc_mode = False)
    b = fixed_batch(text_vocab(model), pad_right = 25)
    model.train()
    model.zero_grad(set_to_none = True)
    full = model(**b).loss
    full.backward()
    g_full = lora_grads(model)
    model.zero_grad(set_to_none = True)
    n_items = int((b["labels"][:, 1:] != -100).sum())
    parts = []
    for i in range(2):
        mb = {k: v[i:i + 1] for k, v in b.items()}
        lo = model(**mb, num_items_in_batch = torch.tensor(n_items, device = "cuda")).loss
        lo.backward()
        parts.append(float(lo))
    rel, norm, _ = grad_rel(g_full, lora_grads(model))
    loss_rel = abs(sum(parts) - float(full)) / max(abs(float(full)), 1e-12)
    free(model)
    return dict(ok = loss_rel <= 5e-3 and rel < 3e-2 and norm > 0, full = float(full), parts = parts,
                loss_rel = loss_rel, grad_rel = rel, grad_norm = norm)


def _swapper(model):
    m = model
    for _ in range(4):
        s = getattr(m, "_unsloth_block_swap", None)
        if s is not None:
            return s
        m = getattr(m, "base_model", None) or getattr(m, "model", None)
        if m is None:
            break
    try:
        from unsloth_zoo.block_swap import find_decoder_layers
        return getattr(find_decoder_layers(model), "_unsloth_block_swap", None)
    except Exception:
        return None


def _offload_pair(key, four_bit):
    import torch
    path = local_copy(MODELS[key])
    model = add_lora(load(path, four_bit), 0)
    b = fixed_batch(text_vocab(model))
    base = step(model, b)
    resident_swapper = _swapper(model) is not None
    free(model)
    model = None
    try:
        model = add_lora(load(path, four_bit, **({"offload_layers": 1} if four_bit else {})), 1)
    except ValueError as e:
        return dict(ok = None if "offload_layers" in str(e) else False, refused = str(e)[:400])
    swap = step(model, b)
    sw = _swapper(model)
    rec = dict(resident_has_swapper = resident_swapper, swapper = sw is not None)
    if sw is not None:
        torch.cuda.synchronize()
        st = sw.stats(reset = False)
        tot = dict(getattr(sw, "_tot", {}) or {})
        rec.update(swapped = st["swapped"], copies_total = tot.get("copies"), host_bytes = int(sw.host_bytes()),
                   timing = st.get("timing"))
    lg_rel = float((swap[0] - base[0]).norm() / base[0].norm().clamp_min(1e-12))
    loss_rel = abs(swap[1] - base[1]) / max(abs(base[1]), 1e-12)
    g_rel, g_norm, missing = grad_rel(base[2], swap[2])
    rec.update(logits_rel = lg_rel, loss_rel = loss_rel, grad_rel = g_rel, grad_norm = g_norm, grads_missing = missing,
               loss_resident = base[1], loss_offload = swap[1])
    engaged = sw is not None and not resident_swapper and rec.get("host_bytes", 0) > 0 and (
        not rec.get("timing", True) or (rec.get("copies_total") or 0) >= 2 * len(rec.get("swapped", [1])))
    rec["engaged"] = engaged
    rec["ok"] = bool(engaged and lg_rel <= 2e-3 and loss_rel <= 1e-3 and g_rel <= 2e-2 and g_norm > 0 and not missing)
    free(model)
    return rec


def cell_offload(key):
    return _offload_pair(key, False)


def cell_offload_qlora(key):
    return _offload_pair(key, True)


def cell_gptoss_export(key):
    import glob
    import transformers
    from safetensors import safe_open
    if int(transformers.__version__.split(".")[0]) < 5:
        return dict(ok = None, note = "transformers 4.x dequantizes MXFP4 at load; nothing to re-pack")
    model = load(local_copy(MODELS["gptoss"]))
    packed = [n for n, p in model.named_parameters() if "experts" in n and p.dtype.is_floating_point is False]
    model = add_lora(model, gc_mode = False)
    lora_on_experts = sorted({n.split(".lora")[0].split(".")[-1] for n, _ in model.named_parameters()
                              if "lora_" in n and "experts" in n})
    merged = model.merge_and_unload()
    d = tempfile.mkdtemp(prefix = "w5_gptoss_export_")
    merged.save_pretrained(d)
    keys = {}
    for f in glob.glob(os.path.join(d, "*.safetensors")):
        with safe_open(f, "pt") as s:
            for k in s.keys():
                if ".experts." in k:
                    keys[k] = str(s.get_slice(k).get_dtype())
    bad = sorted(k for k in keys if k.endswith(("_blocks", "_scales")) or k.endswith("$"))
    free(model, merged)
    return dict(ok = not bad and bool(keys), packed_at_load = packed[:4], lora_on_experts = lora_on_experts,
                leftover_keys = bad[:8], expert_keys = sorted(keys)[:8])


CELLS = {"fused_ce": cell_fused_ce, "ga_norm": cell_ga_norm, "offload": cell_offload,
         "offload_qlora": cell_offload_qlora, "gptoss_export": cell_gptoss_export}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required = True, choices = sorted(MODELS))
    ap.add_argument("--cells", default = None)
    ap.add_argument("--out", required = True)
    a = ap.parse_args()
    res = {"model": a.model, "repo": MODELS[a.model], "cells": {}}

    def dump():
        with open(a.out, "w", encoding = "utf-8") as fh:
            json.dump(res, fh, indent = 1, default = str)

    try:
        import torch
        import unsloth  # noqa: F401  (before transformers / peft / trl)
        import unsloth_zoo
        import transformers
        import peft
        import trl
        p = torch.cuda.get_device_properties(0)
        res["env"] = dict(torch = torch.__version__, hip = torch.version.hip, cuda = torch.version.cuda,
                          device = p.name, arch = getattr(p, "gcnArchName", None) if torch.version.hip else f"sm_{p.major}{p.minor}",
                          transformers = transformers.__version__, peft = peft.__version__, trl = trl.__version__,
                          dtype = str(_dtype()),
                          unsloth = unsloth.__file__, unsloth_zoo = unsloth_zoo.__file__)
    except Exception as e:  # noqa: BLE001
        res["import_error"] = f"{type(e).__name__}: {e}"[:2000]
        dump()
        return 0
    dump()
    for name in (a.cells or DEFAULT_CELLS[a.model]).split(","):
        t0 = time.time()
        try:
            rec = CELLS[name](a.model)
        except Exception as e:  # noqa: BLE001
            rec = dict(ok = False, error = f"{type(e).__name__}: {e}"[:1500], tb = traceback.format_exc()[-3000:])
        rec["seconds"] = round(time.time() - t0, 1)
        res["cells"][name] = rec
        dump()
    return 0


if __name__ == "__main__":
    sys.exit(main())
