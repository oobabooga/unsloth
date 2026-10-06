#!/usr/bin/env python3
"""Diagnostic (observes only): load unsloth/gemma-4-26B-A4B-it 4-bit as the notebook does, greedy
text-only generate, record text / crash / routed census. Env (MoE switches) set by the caller."""
import json
import os
import sys
import time
import traceback

OUT = sys.argv[1]
NEW = int(sys.argv[2]) if len(sys.argv) > 2 else 32
out = {"env": {k: os.environ.get(k) for k in ("UNSLOTH_MOE_BACKEND", "UNSLOTH_MOE_ROUTED_KERNEL", "AMD_SERIALIZE_KERNEL",
                                             "UNSLOTH_COMPILE_DISABLE")}}


def dump():
    with open(OUT, "w", encoding = "utf-8") as f:
        json.dump(out, f, indent = 2, default = str)


try:
    from unsloth import FastModel
    import torch
    import unsloth, unsloth_zoo
    out["versions"] = {"unsloth": unsloth.__version__, "zoo": unsloth_zoo.__version__, "torch": torch.__version__}
    cen = {"calls": 0, "returned": 0}
    try:
        import unsloth_zoo.temporary_patches.moe_routed as MR
        real = MR.routed_moe_forward

        def wrapped(*a, **k):
            cen["calls"] += 1
            r = real(*a, **k)
            cen["returned"] += r is not None
            return r
        MR.routed_moe_forward = wrapped
        out["moe_routed"] = True
    except ImportError:
        out["moe_routed"] = False
    out["census"] = cen
    t0 = time.time()
    model, tok = FastModel.from_pretrained("unsloth/gemma-4-26B-A4B-it", max_seq_length = 2048, load_in_4bit = True)
    out["load_s"] = round(time.time() - t0, 1)
    try:
        exp = next(m for n, m in model.named_modules() if type(m).__name__.endswith("Experts"))
        out["experts"] = {"type": type(exp).__name__, "gate_up_proj": type(getattr(exp, "gate_up_proj", None)).__name__,
                          "has_quant_state": getattr(getattr(exp, "gate_up_proj", None), "quant_state", None) is not None}
    except StopIteration:
        out["experts"] = None
    lin = [type(m).__name__ for n, m in model.named_modules() if n.endswith("self_attn.q_proj")][:1]
    out["q_proj_type"] = lin
    dump()
    msgs = [{"role": "user", "content": [{"type": "text", "text": "Write a poem about sloths."}]}]
    inputs = tok.apply_chat_template(msgs, add_generation_prompt = True, tokenize = True, return_dict = True,
                                     return_tensors = "pt").to("cuda")
    n_in = inputs["input_ids"].shape[-1]
    tk = getattr(tok, "tokenizer", tok)
    # prefill-only logits sanity
    with torch.no_grad():
        lg = model(**inputs).logits[0, -1].float()
    torch.cuda.synchronize()
    out["prefill_logits"] = {"finite": bool(torch.isfinite(lg).all().item()), "max": lg.max().item(),
                             "top5": tk.convert_ids_to_tokens(lg.topk(5).indices.tolist())}
    dump()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    o = model.generate(**inputs, max_new_tokens = NEW, do_sample = False, use_cache = True)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    out["gen"] = {"new_tokens": int(o.shape[-1] - n_in), "seconds": round(dt, 3),
                  "tok_per_s": round((o.shape[-1] - n_in) / dt, 3),
                  "text": tk.decode(o[0, n_in:].tolist(), skip_special_tokens = False)[:500]}
    out["ok"] = True
except BaseException as e:  # noqa: BLE001
    out["error"] = repr(e)[:800]
    out["tb"] = traceback.format_exc()[-3000:]
dump()
