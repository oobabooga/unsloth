"""Same process, same compiled kernels: fused decode kernels on vs off (UNSLOTH_GPT_OSS_FUSED_DECODE's flag
toggled at runtime), order off, on, on, off. Greedy tokens + teacher-forced per-step logits.

    python -u corr.py OUT.json MODEL [--no-4bit]
"""
import json, os, sys
out_path, model_name = sys.argv[1], sys.argv[2]
no4 = "--no-4bit" in sys.argv
CTX, NEW = 1024, 80
from unsloth import FastLanguageModel  # noqa: E402
import torch  # noqa: E402
from unsloth_zoo import gpt_oss_decode_fused as FD  # noqa: E402
from transformers import DynamicCache  # noqa: E402

model, tok = FastLanguageModel.from_pretrained(model_name, max_seq_length = CTX + 160, load_in_4bit = not no4,
                                               dtype = torch.bfloat16)
FastLanguageModel.for_inference(model)
dev = torch.device("cuda")
pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
text = ("The history of numerical linear algebra begins with Gauss, who described elimination for "
        "solving the normal equations of least squares. ") * 60
real = tok(text, return_tensors = "pt").input_ids[0, :CTX]


def inputs(b):
    g = torch.Generator().manual_seed(0)
    ids = torch.randint(1000, 150000, (b, CTX), generator = g)
    ids[0, -real.shape[0]:] = real
    return ids.to(dev), torch.ones(b, CTX, dtype = torch.long, device = dev)


def greedy(ids, am):
    with torch.no_grad():
        o = model.generate(input_ids = ids, attention_mask = am, max_new_tokens = NEW, min_new_tokens = NEW,
                           do_sample = False, pad_token_id = pad, output_scores = True, return_dict_in_generate = True)
    return o.sequences[:, CTX:], torch.stack(o.scores, 1).float()


def forced(ids, am, cont):
    """Per-step logits along a fixed continuation: one prefill, then one-token decode steps."""
    cache = DynamicCache(config = model.config)
    outs = []
    with torch.no_grad():
        o = model(input_ids = ids, attention_mask = am, past_key_values = cache, use_cache = True)
        outs.append(o.logits[:, -1].float())
        for t in range(cont.shape[1] - 1):
            am = torch.cat([am, am.new_ones(am.shape[0], 1)], 1)
            o = model(input_ids = cont[:, t:t + 1], attention_mask = am, past_key_values = cache, use_cache = True)
            outs.append(o.logits[:, -1].float())
    return torch.stack(outs, 1)


res = {}
for b in (1, 4):
    ids, am = inputs(b)
    greedy(ids, am)  # warm both paths' compiles
    FD._ENABLED = False; greedy(ids, am); FD._ENABLED = True
    arms = []
    for flag in (False, True, True, False):
        FD._ENABLED = flag
        before = FD.FUSED_DECODE_CALLS["add_rms_norm"]
        toks, sc = greedy(ids, am)
        arms.append({"fused": flag, "tokens": toks, "scores": sc, "calls": FD.FUSED_DECODE_CALLS["add_rms_norm"] - before})
    ref_cont = arms[0]["tokens"]
    for a in arms:
        FD._ENABLED = a["fused"]
        a["forced"] = forced(ids, am, ref_cont)
    FD._ENABLED = True
    r = {"calls": [a["calls"] for a in arms]}
    for i in range(4):
        for j in range(i + 1, 4):
            x, y = arms[i], arms[j]
            same = (x["tokens"] == y["tokens"])
            first = [int((~row).nonzero()[0]) if (~row).any() else None for row in same]
            fx, fy = x["forced"], y["forced"]
            fin = torch.isfinite(fx) & torch.isfinite(fy)
            d = (fx - fy).abs().where(fin, torch.zeros_like(fx))
            r[f"{i}{'F' if x['fused'] else 'E'}~{j}{'F' if y['fused'] else 'E'}"] = {
                "greedy_tokens_equal": bool(same.all()), "greedy_first_diff": first,
                "forced_max_abs_logit_diff": float(d.max()),
                "forced_max_abs_logit_diff_step0": float(d[:, 0].max()),
                "forced_argmax_agree_frac": float((fx.argmax(-1) == fy.argmax(-1)).float().mean()),
                "forced_bit_identical": bool(torch.equal(fx, fy)),
            }
    res[f"b{b}"] = r
    print("CORR", b, json.dumps(r), flush = True)
json.dump(res, open(out_path, "w"), indent = 1)
print("DONE", flush = True)
