#!/usr/bin/env python3
"""Diagnostic 2 (observes only): where do generate()'s NaN logits come from on gfx1151?
Load as the notebook does, then greedy 16 tokens via several generate paths, recording per-step
finiteness of the scores and the text. argv: out.json [attn_impl]"""
import json
import os
import sys
import traceback

OUT = sys.argv[1]
ATTN = sys.argv[2] if len(sys.argv) > 2 else None
out = {"env": {k: os.environ.get(k) for k in ("UNSLOTH_COMPILE_DISABLE", "UNSLOTH_MOE_ROUTED_KERNEL")}, "attn": ATTN, "variants": {}}


def dump():
    with open(OUT, "w", encoding = "utf-8") as f:
        json.dump(out, f, indent = 2, default = str)


try:
    from unsloth import FastModel
    import torch
    kw = {"attn_implementation": ATTN} if ATTN else {}
    model, tok = FastModel.from_pretrained("unsloth/gemma-4-26B-A4B-it", max_seq_length = 2048, load_in_4bit = True, **kw)
    out["config_attn"] = getattr(model.config, "_attn_implementation", None)
    tk = getattr(tok, "tokenizer", tok)
    msgs = [{"role": "user", "content": [{"type": "text", "text": "Write a poem about sloths."}]}]
    inputs = tok.apply_chat_template(msgs, add_generation_prompt = True, tokenize = True, return_dict = True,
                                     return_tensors = "pt").to("cuda")
    n_in = inputs["input_ids"].shape[-1]

    def summarize(o):
        sc = [s.float() for s in (o.scores or [])]
        return {"finite_per_step": [bool(torch.isfinite(s).all().item()) for s in sc],
                "text": tk.decode(o.sequences[0, n_in:].tolist(), skip_special_tokens = False)[:300]}

    def variant(name, fn):
        try:
            torch.cuda.synchronize()
            out["variants"][name] = summarize(fn())
        except BaseException as e:  # noqa: BLE001
            out["variants"][name] = {"error": repr(e)[:600], "tb": traceback.format_exc()[-1500:]}
        dump()

    common = dict(max_new_tokens = 16, do_sample = False, output_scores = True, return_dict_in_generate = True)
    old = getattr(model, "_old_generate", None)
    out["has_old_generate"] = old is not None
    if old is not None:
        def hf(cache):
            def f():
                with torch.inference_mode(), torch.autocast("cuda", dtype = torch.bfloat16):
                    return old(**inputs, **common, cache_implementation = cache)
            return f
        variant("hf_old_generate_dynamic", hf("dynamic"))
        variant("hf_old_generate_static", hf("static"))
    variant("unsloth_generate", lambda: model.generate(**inputs, **common))
    # manual dynamic-cache decode loop, no generate machinery
    try:
        from transformers import DynamicCache
        with torch.inference_mode():
            cache = DynamicCache(config = model.config) if "config" in DynamicCache.__init__.__code__.co_varnames else DynamicCache()
            o = model(**inputs, past_key_values = cache, use_cache = True)
            steps, ids = [], []
            nxt = o.logits[:, -1].argmax(-1, keepdim = True)
            for i in range(8):
                ids.append(int(nxt))
                am = torch.ones(1, n_in + i + 1, device = "cuda", dtype = inputs["attention_mask"].dtype)
                o = model(input_ids = nxt, attention_mask = am, past_key_values = o.past_key_values, use_cache = True)
                lg = o.logits[:, -1].float()
                steps.append(bool(torch.isfinite(lg).all().item()))
                nxt = lg.argmax(-1, keepdim = True)
            out["variants"]["manual_dynamic_loop"] = {"finite_per_step": steps, "text": tk.decode(ids)}
    except BaseException as e:  # noqa: BLE001
        out["variants"]["manual_dynamic_loop"] = {"error": repr(e)[:600], "tb": traceback.format_exc()[-1500:]}
except BaseException as e:  # noqa: BLE001
    out["error"] = repr(e)[:800]
    out["tb"] = traceback.format_exc()[-3000:]
dump()
