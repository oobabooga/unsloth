# SPDX-License-Identifier: AGPL-3.0-only
"""unsloth #18435 variant of vlmpack_parity.py: also counts the block-varlen route.

Packed vs per-document parity for UnslothVisionDataCollator(padding_free=True) on tiny mRoPE VLMs.

Per document: logits (packed slice vs the document alone), loss, LoRA grads (sum of per-doc CE sums vs
the packed CE sum), leakage (perturb doc A's image: doc B's packed logits bitwise unchanged), and a
negative control (text-only positions on every plane must break parity for image docs).
    python parity.py --model trl-internal-testing/tiny-Qwen2_5_VLForConditionalGeneration [--dtype fp32]
"""
import argparse
import json
import os
import sys

os.environ.setdefault("UNSLOTH_RETURN_LOGITS", "1")
os.environ.setdefault("UNSLOTH_DISABLE_AUTO_UPDATES", "1")
ap = argparse.ArgumentParser()
ap.add_argument("--model", required = True)
ap.add_argument("--dtype", default = "fp32")
ap.add_argument("--attn", default = None, help = "attn_implementation override (sdpa / eager / flash_attention_2)")
ap.add_argument("--out", default = None)
ap.add_argument("--processor", default = None, help = "processor repo when the model repo ships none")
ap.add_argument("--expect-refusal", action = "store_true", help = "base / kill switch: padding_free must raise")
a = ap.parse_args()

import unsloth  # noqa: E402,F401
from unsloth import FastVisionModel  # noqa: E402
from unsloth.trainer import UnslothVisionDataCollator  # noqa: E402
import torch  # noqa: E402
import transformers  # noqa: E402
from PIL import Image  # noqa: E402

print("IMPORT unsloth", unsloth.__file__, "transformers", transformers.__version__, "torch", torch.__version__)
dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[a.dtype]
kw = {}
if a.attn:
    kw["attn_implementation"] = a.attn
model, processor = FastVisionModel.from_pretrained(
    a.model, load_in_4bit = False, dtype = dtype, max_seq_length = 4096, **kw
)
if a.processor:
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(a.processor)
model = FastVisionModel.get_peft_model(
    model, r = 8, lora_alpha = 16, finetune_vision_layers = True, finetune_language_layers = True,
    random_state = 0,
)
# Non-zero B so LoRA paths carry signal into the logits and grads of A as well.
torch.manual_seed(0)
with torch.no_grad():
    for n, p in model.named_parameters():
        if "lora_B" in n:
            p.normal_(0, 0.02)
model.train()
print("ATTN", getattr(model.config, "_attn_implementation", None))


def img(w, h, seed):
    g = torch.Generator().manual_seed(seed)
    arr = (torch.rand(h, w, 3, generator = g) * 255).to(torch.uint8).numpy()
    return Image.fromarray(arr)


def ex(w, h, seed, q, ans):
    return {"messages": [
        {"role": "user", "content": [{"type": "image", "image": img(w, h, seed)}, {"type": "text", "text": q}]},
        {"role": "assistant", "content": [{"type": "text", "text": ans}]},
    ]}


examples = [
    ex(56, 84, 1, "Describe.", "A small picture with noise " * 3),
    ex(112, 56, 2, "What is it?", "Noise."),
    {"messages": [{"role": "user", "content": [{"type": "text", "text": "Text only question here"}]},
                  {"role": "assistant", "content": [{"type": "text", "text": "text answer " * 5}]}]},
    ex(84, 140, 3, "Caption this image please.", "It is a random image " * 6),
]
try:
    collator = UnslothVisionDataCollator(model, processor, padding_free = True)
except ValueError as e:
    print("REFUSED", str(e)[:300])
    res = {"refused": True, "msg": str(e)}
    if a.out:
        json.dump(res, open(a.out, "w", encoding = "utf-8"))
    sys.exit(0 if a.expect_refusal else 1)
if a.expect_refusal:
    print("FAIL: expected a refusal")
    sys.exit(1)

calls = {"n": 0}
ri = collator._mm_rope_index
orig_call = type(ri).__call__


def counted(self, doc):
    calls["n"] += 1
    return orig_call(self, doc)


type(ri).__call__ = counted
docs = [collator._collate_unpacked([e]) for e in examples]
packed = collator(examples)
print("ROPE_CALLS", calls["n"], "KEYS", sorted(packed.keys()))
print("POS", tuple(packed["position_ids"].shape), "LEN", packed["packed_seq_lengths"].tolist())
dev = next(model.parameters()).device


def to(b):
    return {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}


def run(batch):
    model.zero_grad(set_to_none = True)
    out = model(**to(batch))
    logits = out.logits.float()
    labels = batch["labels"].to(dev)
    sl = logits[:, :-1].reshape(-1, logits.shape[-1])
    tl = labels[:, 1:].reshape(-1)
    ce = torch.nn.functional.cross_entropy(sl, tl, ignore_index = -100, reduction = "sum")
    ce.backward()
    grads = {n: p.grad.detach().float().clone() for n, p in model.named_parameters()
             if p.requires_grad and p.grad is not None}
    return logits.detach(), ce.detach(), grads


from unsloth.utils.hf_packed_attention import HF_PACKED_ATTENTION_STATS as S  # noqa: E402
import unsloth.utils.hf_packed_attention as hpa  # noqa: E402
_orig_mbc = hpa._mask_is_block_causal
_seen = []


def _log_mbc(attention_mask, cu_seqlens, total, window = None):
    r = _orig_mbc(attention_mask, cu_seqlens, total, window)
    if len(_seen) < 3:
        _seen.append(1)
        print("MASKCHK", type(attention_mask).__name__, getattr(attention_mask, "dtype", None),
              tuple(getattr(attention_mask, "shape", ())), "cu", cu_seqlens.tolist(), "total", total, "->", r)
    return r


hpa._mask_is_block_causal = _log_mbc
from unsloth.utils.packing import HYBRID_PACKING_STATS  # noqa: E402
print("HYBRID", HYBRID_PACKING_STATS)
res = {"hybrid": dict(HYBRID_PACKING_STATS), "model": a.model, "dtype": a.dtype, "transformers": transformers.__version__, "rope_calls": calls["n"]}
route = {"tuple": 0, "leave": 0, "none": 0, "other": 0}
eager = {"n": 0}
_orig_bvk = hpa.block_varlen_kwargs


def _count_bvk(layer, hidden_states, kwargs):
    r = _orig_bvk(layer, hidden_states, kwargs)
    route["tuple" if isinstance(r, tuple) else "leave" if r == hpa._LEAVE_BLOCK else "none" if r is None else "other"] += 1
    return r


hpa.block_varlen_kwargs = _count_bvk
_inner_sdpa = hpa._sdpa_packed_varlen


def _count_eager(*args, **kwargs):
    eager["n"] += 1
    return _inner_sdpa(*args, **kwargs)


# Looked up as a module global inside the traced wrapper: replacing it before the first trace counts real calls.
hpa._sdpa_packed_varlen_eager = torch.compiler.disable(_count_eager)
res["unsloth_file"] = unsloth.__file__
res["has_leave_block"] = hasattr(hpa, "_LEAVE_BLOCK")
per = [run(d) for d in docs]
route0, eager0 = dict(route), eager["n"]
s0 = dict(S)
from torch.profiler import ProfilerActivity, profile  # noqa: E402
with profile(activities = [ProfilerActivity.CPU]) as prof:
    pl, pce, pg = run(packed)
ops = {}
for e in prof.events():
    if "attention" in e.name.lower() or "flash" in e.name.lower():
        ops[e.name[:80]] = ops.get(e.name[:80], 0) + 1
res["attn_ops"] = dict(sorted(ops.items(), key = lambda t: -t[1])[:12])
res["route"] = {k: route[k] - route0[k] for k in route}
res["route_per_doc_runs"] = route0
res["eager_wrapper_calls"] = eager["n"] - eager0
res["eager_wrapper_calls_per_doc_runs"] = eager0
try:
    from collections import Counter
    res["block_modes"] = dict(Counter(str(getattr(m, "_unsloth_block_mode", None)) for m in model.modules()
                                      if hasattr(m, "_unsloth_block_mode")))
except Exception as e:  # noqa: BLE001
    res["block_modes"] = repr(e)[:120]
print("ROUTE", res["route"], "EAGER", res["eager_wrapper_calls"], "OPS", res["attn_ops"])
res["varlen_stats_delta"] = {k: S[k] - s0[k] for k in S}
starts = [0]
for n in packed["packed_seq_lengths"].tolist():
    starts.append(starts[-1] + n)
max_logit = 0.0
for i, (l, ce, g) in enumerate(per):
    n = l.shape[1]
    d = (pl[0, starts[i]:starts[i] + n] - l[0]).abs().max().item()
    max_logit = max(max_logit, d)
    res[f"doc{i}_logit_maxabs"] = d
ce_sum = sum(c for _, c, _ in per)
res["ce_packed"], res["ce_docs"] = pce.item(), ce_sum.item()
gsum = {}
for _, _, g in per:
    for k, v in g.items():
        gsum[k] = gsum.get(k, 0) + v
rel = 0.0
for k, v in pg.items():
    ref = gsum[k]
    rel = max(rel, ((v - ref).norm() / (ref.norm() + 1e-12)).item())
res["grad_max_rel"] = rel
res["n_grad_tensors"] = len(pg)
res["vision_grad_tensors"] = sum(1 for k in pg if "visual" in k or "vision" in k)

# Leakage: perturb doc 0's image, doc 1..3 packed logits must not move.
ex0 = ex(56, 84, 99, "Describe.", "A small picture with noise " * 3)
packed2 = collator([ex0] + examples[1:])
with torch.no_grad():
    l2 = model(**to(packed2)).logits.float()
    l1 = model(**to(packed)).logits.float()
res["leak_docA_moved"] = (l2[0, :starts[1]] - l1[0, :starts[1]]).abs().max().item()
res["leak_others_maxabs"] = (l2[0, starts[1]:] - l1[0, starts[1]:]).abs().max().item()

# Negative control: text positions on every plane (what 2D position_ids would give).
bad = dict(packed)
bad["position_ids"] = packed["position_ids"][:1].expand(4, -1, -1).contiguous()
with torch.no_grad():
    lb = model(**to(bad)).logits.float()
res["negctl_doc0_maxabs"] = (lb[0, :starts[1]] - pl[0, :starts[1]]).abs().max().item()
res["max_logit_maxabs"] = max_logit
print("RESULT", json.dumps(res))
if a.out:
    json.dump(res, open(a.out, "w", encoding = "utf-8"), indent = 1)
