#!/usr/bin/env python3
"""One cell for the unsloth-zoo PR 2106 probe (sparse flex attention on ROCm). Observes only; writes JSON via --out.

--mode on|off|fp32 runs ONE tiny random model (MiniMax-M3 or GLM-5 / glm_moe_dsa, 2 sparse attention layers, the
configs of tests/test_sparse_attention_flex.py) for a forward + backward above the indexer budget:
  on    bf16, flex allowed (the default a user gets)
  off   bf16, UNSLOTH_SPARSE_ATTN_FLEX=0 (the model's own path)
  fp32  fp32, the dense path (flex gates itself off for fp32): reference for both
Logits, loss and every parameter grad are saved to --pt. Engagement is recorded from
unsloth_zoo.flex_attention.sparse.SPARSE_ATTN_STATS and sparse_blocks._FAILED (the fallback latch).

--compare DIR FAMILY reads the three .pt files and writes the error metrics (no thresholds: criteria judge).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

CASES = ("plain", "leftpad")
SEQ = 512  # MiniMax budget 2 x 128 = 256 keys, DSA index_topk 64: both above the budget


def _minimax_config(**kw):
    from transformers.models.minimax_m3_vl.configuration_minimax_m3_vl import MiniMaxM3VLTextConfig
    base = dict(
        hidden_size = 64, intermediate_size = 64, num_hidden_layers = 2, num_attention_heads = 8,
        num_key_value_heads = 2, head_dim = 32, vocab_size = 256, index_n_heads = 2, index_head_dim = 16,
        index_block_size = 128, index_topk_blocks = 2, index_local_blocks = 1,
        layer_types = ["minimax_m3_sparse", "minimax_m3_sparse"],
        mlp_layer_types = ["dense", "dense"], dense_intermediate_size = 64,
    )
    base.update(kw)
    config = MiniMaxM3VLTextConfig(**base)
    config._attn_implementation = "sdpa"
    return config


def _dsa_config(family, **kw):
    if family == "deepseek_v32":
        from transformers.models.deepseek_v32.configuration_deepseek_v32 import DeepseekV32Config as C
    else:
        from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import GlmMoeDsaConfig as C
    base = dict(
        hidden_size = 64, intermediate_size = 64, moe_intermediate_size = 32, num_hidden_layers = 2,
        num_attention_heads = 4, num_key_value_heads = 4, qk_nope_head_dim = 16, qk_rope_head_dim = 16,
        v_head_dim = 16, kv_lora_rank = 32, q_lora_rank = 32, vocab_size = 256, index_topk = 64,
        index_n_heads = 4, index_head_dim = 32, n_routed_experts = 4, num_experts_per_tok = 2,
        first_k_dense_replace = 2, mlp_layer_types = ["dense", "dense"],
    )
    if family == "glm_moe_dsa":
        base["indexer_types"] = ["full", "shared"]
    base.update(kw)
    config = C(**base)
    config._attn_implementation = "sdpa"
    return config


def _model_cls(family):
    if family == "minimax_m3_vl":
        from transformers.models.minimax_m3_vl.modeling_minimax_m3_vl import MiniMaxM3VLForCausalLM as M
    elif family == "deepseek_v32":
        from transformers.models.deepseek_v32.modeling_deepseek_v32 import DeepseekV32ForCausalLM as M
    else:
        from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaForCausalLM as M
    return M


def run_cell(args) -> dict:
    rec: dict = {"family": args.family, "mode": args.mode, "stage": "import", "cases": {}}
    if args.mode == "off":
        os.environ["UNSLOTH_SPARSE_ATTN_FLEX"] = "0"
    import torch
    rec["versions"] = {"torch": torch.__version__, "hip": torch.version.hip, "cuda": torch.version.cuda}
    try:
        import triton
        rec["versions"]["triton"] = triton.__version__
    except Exception as e:  # noqa: BLE001
        rec["versions"]["triton"] = f"import failed: {e!r}"
    import transformers
    rec["versions"]["transformers"] = transformers.__version__
    rec["cuda_available"] = torch.cuda.is_available()
    if rec["cuda_available"]:
        p = torch.cuda.get_device_properties(0)
        rec["device"] = {"name": p.name, "arch": getattr(p, "gcnArchName", None),
                         "capability": list(torch.cuda.get_device_capability(0))}
    import unsloth_zoo
    rec["unsloth_zoo_file"] = unsloth_zoo.__file__
    from unsloth_zoo.flex_attention import sparse as gates
    from unsloth_zoo.flex_attention import sparse_blocks as sb
    from unsloth_zoo.temporary_patches import sparse_attention_flex as s
    s.patch_minimax_m3_sparse_flex()
    s.patch_dsa_sparse_flex()
    M = _model_cls(args.family)
    attn_cls = [c for c in vars(sys.modules[M.__module__]).values()
                if isinstance(c, type) and c.__name__.endswith("Attention") and hasattr(c, "forward")]
    rec["patched_attention"] = {c.__name__: bool(getattr(c.forward, "_unsloth_sparse_flex", False)) for c in attn_cls}
    q_probe = torch.zeros(1, 4, 32, 16, device = "cuda", dtype = torch.bfloat16)
    rec["gates"] = {"sparse_flex_enabled": gates.sparse_flex_enabled(), "flex_available": sb._flex_available(),
                    "device_supported": sb._device_supported(0), "sparse_flex_device_ok": sb.sparse_flex_device_ok(q_probe)}
    rec["stage"] = "build"
    if args.family == "minimax_m3_vl":
        config = _minimax_config(num_attention_heads = 8, num_key_value_heads = 2, head_dim = 64, hidden_size = 128)
    else:
        config = _dsa_config(args.family, qk_nope_head_dim = 32, qk_rope_head_dim = 32, v_head_dim = 64)
    torch.manual_seed(0)
    dtype = torch.float32 if args.mode == "fp32" else torch.bfloat16
    model = M(config).to("cuda", dtype)
    g = torch.Generator().manual_seed(1234)
    ids = torch.randint(0, config.vocab_size, (2, SEQ), generator = g).cuda()
    saved = {}
    for case in CASES:
        c: dict = {"stage": "forward"}
        rec["cases"][case] = c
        # No labels: the loss is computed here, so no patched / fused loss path can drop the logits.
        kwargs = {"input_ids": ids}
        labels = ids.clone()
        if case == "leftpad":
            am = torch.ones(2, SEQ, dtype = torch.long, device = "cuda")
            am[1, :40] = 0
            labels[1, :40] = -100
            kwargs.update(attention_mask = am)
        before = json.loads(json.dumps(gates.SPARSE_ATTN_STATS))
        failed_before = dict(sb._FAILED)
        model.zero_grad(set_to_none = True)
        t0 = time.time()
        try:
            out = model(**kwargs)
            loss = torch.nn.functional.cross_entropy(
                out.logits[:, :-1].float().reshape(-1, out.logits.shape[-1]), labels[:, 1:].reshape(-1), ignore_index = -100)
            c["logits_shape"] = list(out.logits.shape)
            c["stage"] = "backward"
            loss.backward()
            torch.cuda.synchronize()
            c["stage"] = "done"
            c["loss"] = float(loss.item())
            grads = {n: p.grad.detach().float().cpu() for n, p in model.named_parameters() if p.grad is not None}
            saved[case] = {"logits": out.logits.detach().float().cpu(), "loss": c["loss"], "grads": grads,
                           "valid_rows": (kwargs.get("attention_mask", torch.ones(2, SEQ)) > 0).cpu()}
            c["finite"] = bool(torch.isfinite(out.logits).all().item()) and all(bool(torch.isfinite(v).all()) for v in grads.values())
            c["n_grads"] = len(grads)
        except Exception as e:  # noqa: BLE001
            c["error"] = repr(e)[:1500]
            c["traceback"] = traceback.format_exc()[-4000:]
        c["seconds"] = round(time.time() - t0, 2)
        after = gates.SPARSE_ATTN_STATS
        c["stats_delta"] = {k: {m: n - before.get(k, {}).get(m, 0) for m, n in v.items() if n - before.get(k, {}).get(m, 0)}
                            for k, v in after.items()}
        c["stats_delta"] = {k: v for k, v in c["stats_delta"].items() if v}
        c["failed_new"] = {repr(k): v for k, v in sb._FAILED.items() if k not in failed_before}
    rec["stats_total"] = gates.SPARSE_ATTN_STATS
    rec["failed_latch"] = {repr(k): v for k, v in sb._FAILED.items()}
    if saved:
        torch.save(saved, args.pt)
        rec["pt"] = args.pt
    rec["stage"] = "done"
    return rec


def _rel(a, b):
    return float(((a.float() - b.float()).abs().max() / b.float().abs().max().clamp_min(1e-6)).item())


def compare(directory, family) -> dict:
    import torch
    runs = {}
    for mode in ("on", "off", "fp32"):
        path = os.path.join(directory, f"cell_{family}_{mode}.pt")
        runs[mode] = torch.load(path) if os.path.isfile(path) else None
    res: dict = {"family": family, "present": {m: r is not None for m, r in runs.items()}, "cases": {}}
    for case in CASES:
        have = {m: (r or {}).get(case) for m, r in runs.items()}
        if not all(have.values()):
            res["cases"][case] = {"missing": [m for m, v in have.items() if not v]}
            continue
        ref = have["fp32"]
        rows = ref["valid_rows"]
        names = [n for n in ref["grads"] if ref["grads"][n].numel() and ref["grads"][n].abs().max() > 0]
        m: dict = {"n_grad_tensors": len(names)}
        for a, b in (("on", "fp32"), ("off", "fp32"), ("on", "off")):
            x, y = have[a], have[b]
            gr = {n: _rel(x["grads"][n], y["grads"][n]) for n in names if n in x["grads"] and n in y["grads"]}
            worst = max(gr, key = gr.get) if gr else None
            m[f"{a}_vs_{b}"] = {
                "loss_abs": abs(x["loss"] - y["loss"]),
                "logits_rel": _rel(x["logits"][rows], y["logits"][rows]),
                "grad_rel_max": gr[worst] if worst else None, "grad_rel_worst_param": worst,
                "grads_missing": [n for n in names if n not in x["grads"] or n not in y["grads"]],
                "bit_identical_logits": bool(torch.equal(x["logits"], y["logits"])),
            }
        m["loss"] = {k: have[k]["loss"] for k in have}
        res["cases"][case] = m
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices = ("minimax_m3_vl", "glm_moe_dsa", "deepseek_v32"))
    ap.add_argument("--mode", choices = ("on", "off", "fp32"))
    ap.add_argument("--pt")
    ap.add_argument("--compare", metavar = "DIR")
    ap.add_argument("--out", required = True)
    args = ap.parse_args()
    if args.compare:
        rec = compare(args.compare, args.family)
    else:
        try:
            rec = run_cell(args)
        except Exception as e:  # noqa: BLE001
            rec = {"family": args.family, "mode": args.mode, "fatal": repr(e)[:1500],
                   "traceback": traceback.format_exc()[-4000:]}
    with open(args.out, "w", encoding = "utf-8") as fh:
        json.dump(rec, fh, indent = 2, default = str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
