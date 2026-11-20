"""Real Unsloth FastModel + LoRA on a tiny random Qwen3-MoE (head_dim 128): PrefixGrouper stream vs
per-row log-probs and LoRA grads, fp32 and bf16, with the flex kernel call count (engagement)."""

import json
import os
import tempfile

import torch

import unsloth  # noqa: F401
from unsloth import FastModel
import unsloth.utils.prefix_grouper_kernel as pgk
from unsloth.utils.prefix_grouper import prefix_seg_info_applied

print("unsloth", unsloth.__file__, "MOE env", os.environ.get("UNSLOTH_GRPO_PREFIX_GROUPER_MOE"), flush=True)
CALLS = [0]
_orig = pgk.flex_shared_prefix_attention


def counted(*a, **k):
    CALLS[0] += 1
    return _orig(*a, **k)


pgk.flex_shared_prefix_attention = counted

from transformers import AutoTokenizer, Qwen3MoeConfig, Qwen3MoeForCausalLM  # noqa: E402

tmp = tempfile.mkdtemp()
cfg = Qwen3MoeConfig(vocab_size=1024, hidden_size=256, intermediate_size=512, moe_intermediate_size=128,
                     num_experts=16, num_experts_per_tok=4, num_hidden_layers=4, num_attention_heads=4,
                     num_key_value_heads=2, head_dim=128, max_position_embeddings=1024, mlp_only_layers=[],
                     decoder_sparse_step=1, norm_topk_prob=True)
torch.manual_seed(0)
m = Qwen3MoeForCausalLM(cfg)
with torch.no_grad():
    for p in m.parameters():
        if p.dim() > 1:
            torch.nn.init.normal_(p, std=0.05)
m.save_pretrained(tmp)
AutoTokenizer.from_pretrained("Qwen/Qwen3-30B-A3B").save_pretrained(tmp)
del m

g = torch.Generator().manual_seed(1)
SPEC = [(96, [40, 64, 23, 51]), (130, [33, 70, 12])]
groups = [(torch.randint(1, 1024, (P,), generator=g), [torch.randint(1, 1024, (r,), generator=g) for r in Rs])
          for P, Rs in SPEC]
weights = [[torch.randn(c.numel(), generator=g) for c in cs] for _, cs in groups]
results = {}
for dtype in (torch.float32, torch.bfloat16):
    model, tok = FastModel.from_pretrained(tmp, max_seq_length=1024, load_in_4bit=False, dtype=dtype,
                                           full_finetuning=False)
    model = FastModel.get_peft_model(model, r=8, lora_alpha=16, lora_dropout=0, bias="none",
                                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj",
                                                     "up_proj", "down_proj"], random_state=0,
                                     use_gradient_checkpointing="unsloth")
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "lora_B" in n:
                torch.nn.init.normal_(p, std=0.02)
    dev = next(model.parameters()).device
    applied = prefix_seg_info_applied(model)
    moe_ok = any(getattr(x, "_unsloth_prefix_grouper_moe_ok", False) for x in (model, getattr(model, "base_model", None), getattr(getattr(model, "base_model", None), "model", None)) if x is not None)
    seg, meta = pgk.build_seg_info_multigroup([(p.numel(), [c.numel() for c in cs]) for p, cs in groups], dev)
    ids = torch.cat([t for p, cs in groups for t in [p] + cs]).unsqueeze(0).to(dev)
    pos = torch.cat([t for p, cs in groups for t in [torch.arange(p.numel())] +
                     [torch.arange(p.numel(), p.numel() + c.numel()) for c in cs]]).unsqueeze(0).to(dev)
    model.train()

    def pg_rows():
        out = model(input_ids=ids, position_ids=pos, prefix_seg_info=seg, use_cache=False)
        lp = torch.log_softmax(out.logits[0].float(), -1)
        rows = []
        for (p, cs), gm in zip(groups, meta):
            for c, (s, e) in zip(cs, gm["suffix_slices"]):
                pred = torch.tensor([gm["prefix_last_index"]] + list(range(s, e - 1)), device=dev)
                rows.append(lp[pred].gather(-1, c.to(dev).unsqueeze(-1)).squeeze(-1))
        return rows

    def plain_rows():
        rows = []
        for p, cs in groups:
            for c in cs:
                x = torch.cat([p, c]).unsqueeze(0).to(dev)
                lp = torch.log_softmax(model(input_ids=x, use_cache=False).logits[0, p.numel() - 1:-1].float(), -1)
                rows.append(lp.gather(-1, c.to(dev).unsqueeze(-1)).squeeze(-1))
        return rows

    flat_w = [w.to(dev) for ws in weights for w in ws]

    def grads(fn):
        model.zero_grad(set_to_none=True)
        rows = fn()
        sum((r * w).sum() for r, w in zip(rows, flat_w)).backward()
        return [r.detach() for r in rows], {n: p.grad.detach().float().clone()
                                            for n, p in model.named_parameters() if p.grad is not None}

    import hashlib
    c0 = CALLS[0]
    pg_err = None
    try:
        r_pg, g_pg = grads(pg_rows)
    except Exception as e:  # noqa: BLE001
        r_pg, g_pg, pg_err = None, None, f"{type(e).__name__}: {e}"[:400]
    calls = CALLS[0] - c0
    r_pl, g_pl = grads(plain_rows)
    hp = hashlib.sha256(b"".join(r.float().cpu().numpy().tobytes() for r in r_pl)).hexdigest()[:16]
    hg = hashlib.sha256(b"".join(g_pl[k].cpu().numpy().tobytes() for k in sorted(g_pl))).hexdigest()[:16]
    res = dict(applied=applied, moe_ok=moe_ok, kernel_calls=calls, layers=cfg.num_hidden_layers, pg_error=pg_err,
               plain_logp_sha=hp, plain_grad_sha=hg, plain_logp_sum=float(sum(r.float().sum() for r in r_pl)),
               plain_grad_norm=float(sum(v.norm() ** 2 for v in g_pl.values()) ** 0.5), n_plain_grads=len(g_pl))
    if r_pg is not None:
        dlp = max(float((a - b).abs().max()) for a, b in zip(r_pg, r_pl))
        keys = sorted(set(g_pg) & set(g_pl))
        rel = max(float((g_pg[k] - g_pl[k]).norm() / (g_pl[k].norm() + 1e-12)) for k in keys)
        tot = float(sum((g_pg[k] - g_pl[k]).norm() ** 2 for k in keys) ** 0.5 / sum(g_pl[k].norm() ** 2 for k in keys) ** 0.5)
        res.update(max_abs_dlogp=dlp, max_rel_l2_grad=rel, total_rel_l2_grad=tot, n_grads=len(keys),
                   n_expert_grads=sum(1 for k in keys if "expert" in k), missing=sorted(set(g_pg) ^ set(g_pl))[:5])
    try:
        res["compile_check_failed"] = pgk.compile_check_failed() if hasattr(pgk, "compile_check_failed") else "absent"
    except Exception as e:  # noqa: BLE001
        res["compile_check_failed"] = f"ERR {type(e).__name__}: {e}"[:200]
    results[str(dtype)] = res
    print("PARITY", str(dtype), json.dumps(res, default=str), flush=True)
    del model
    torch.cuda.empty_cache()
results["env_flag"] = os.environ.get("UNSLOTH_GRPO_PREFIX_GROUPER_MOE")
results["unsloth_file"] = unsloth.__file__
with open(os.environ["CG5_OUT"], "w", encoding="utf-8") as fh:
    json.dump(results, fh, indent=1, default=str)
