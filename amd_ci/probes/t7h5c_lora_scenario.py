"""T7h.5c scenario (run by t7h5c_lora_probe.py with the state's unsloth checkout first on PYTHONPATH). Observes only.

argv: OUT_JSON MODEL LOAD_IN_4BIT(0|1)
Real LoRA SFT (r16, all linear, bf16, fixed seed and fixed batch, raw loop: forward with labels -> backward ->
AdamW) for STEPS steps, then one greedy generate. Records per-step loss, grad norm, step ms, peak memory, the
generated tokens, and how often the fused LoRA order was taken (unsloth.kernels.utils._lora_fused_ok True returns;
absent at the base). Writes JSON after every section so a crash keeps what ran.
"""
import json
import math
import os
import sys
import time
import traceback

OUT, MODEL, Q4 = sys.argv[1], sys.argv[2], sys.argv[3] == "1"
STEPS = int(os.environ.get("T7H5C_STEPS", "25"))
rec = {"model": MODEL, "load_in_4bit": Q4, "flag": os.environ.get("UNSLOTH_LORA_FUSED_PASSES", "<unset>")}


def save():
    with open(OUT, "w", encoding = "utf-8") as fh:
        json.dump(rec, fh, indent = 1, default = str)


def err(e):
    return (f"{type(e).__name__}: {e}")[:800] + " | " + traceback.format_exc()[-1500:]


try:
    import unsloth  # noqa: F401
    from unsloth import FastLanguageModel

    rec["unsloth_file"] = unsloth.__file__
    import torch

    rec["torch"] = {"version": torch.__version__, "hip": getattr(torch.version, "hip", None),
                    "is_available": torch.cuda.is_available(),
                    "arch": torch.cuda.get_device_properties(0).gcnArchName if torch.cuda.is_available() else None}
    import unsloth.kernels.utils as KU

    calls = {"fused_true": 0, "fused_false": 0}
    if hasattr(KU, "_lora_fused_ok"):
        _orig = KU._lora_fused_ok

        def _count(*a, **k):
            r = _orig(*a, **k)
            calls["fused_true" if r else "fused_false"] += 1
            return r

        KU._lora_fused_ok = _count
        import unsloth.kernels.fast_lora as FL

        if hasattr(FL, "_lora_fused_ok"):
            FL._lora_fused_ok = _count
        rec["has_fused_rule"] = True
    else:
        rec["has_fused_rule"] = False
    save()

    torch.manual_seed(3407)
    model, tok = FastLanguageModel.from_pretrained(MODEL, max_seq_length = 1024, dtype = torch.bfloat16,
                                                   load_in_4bit = Q4)
    model = FastLanguageModel.get_peft_model(model, r = 16, lora_alpha = 16, lora_dropout = 0, bias = "none",
                                             target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj",
                                                               "up_proj", "down_proj"],
                                             use_gradient_checkpointing = "unsloth", random_state = 3407)
    text = ("The quick brown fox jumps over the lazy dog. " * 40)
    ids = tok([text, "Unsloth makes fine-tuning faster and uses less memory. " * 40], return_tensors = "pt",
              truncation = True, max_length = 512, padding = "max_length").input_ids.cuda()
    labels = ids.clone()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr = 1e-4)
    model.train()
    torch.cuda.reset_peak_memory_stats()
    losses, gnorms, ms = [], [], []
    for step in range(STEPS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model(input_ids = ids, labels = labels)
        out.loss.backward()
        g = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1e9)
        opt.step()
        opt.zero_grad(set_to_none = True)
        torch.cuda.synchronize()
        ms.append((time.perf_counter() - t0) * 1000)
        losses.append(float(out.loss))
        gnorms.append(float(g))
    rec["train"] = {"losses": losses, "grad_norms": gnorms, "step_ms": ms,
                    "peak_mib": torch.cuda.max_memory_allocated() / 2**20,
                    "finite": all(math.isfinite(x) for x in losses + gnorms)}
    rec["fused_calls_after_train"] = dict(calls)
    save()

    FastLanguageModel.for_inference(model)
    p = tok(["Write one sentence about foxes:"], return_tensors = "pt").to("cuda")
    with torch.no_grad():
        g = model.generate(**p, max_new_tokens = 24, do_sample = False, output_scores = True,
                           return_dict_in_generate = True)
    toks = g.sequences[0, p.input_ids.shape[1]:].tolist()
    margins = []
    for s in g.scores:
        top = torch.topk(s[0].float(), 2).values
        margins.append(float(top[0] - top[1]))
    rec["generate"] = {"tokens": toks, "margins": margins}
    rec["fused_calls"] = dict(calls)
    rec["done"] = True
except Exception as e:  # noqa: BLE001
    rec["error"] = err(e)
save()
