#!/usr/bin/env python3
"""One decode measurement for unsloth-zoo PR 1580 (routed NF4 / BF16 MoE decode). Observes only.

Run in its own process per (model, weights), PYTHONPATH pointing at the state's unsloth-zoo
checkout. Builds a tiny Qwen3-MoE causal LM with transformers v5 3D expert stacks routed
through unsloth_zoo's experts interface:
  hub: trl-internal-testing/tiny-Qwen3MoeForCausalLM as published (hidden 8, 4 experts, top-2)
  kf : the same config, kernel-friendly shrunk sizes (hidden 256, moe_intermediate 192,
       16 experts, top-4, vocab 512), random init seed 0, decisive routers (as the PR's test)
weights bf16: experts stay BF16 3D parameters; nf4: each stack replaced by one stacked
bitsandbytes Params4bit (how the bnb loader stores 3D experts), nested, blocksize 64.

Arms (UNSLOTH_MOE_ROUTED_KERNEL read per call by the head): "0" first, then default (unset).
Per arm: prefill a fixed prompt, then greedy free-run decode of N tokens (tokens + logits);
the default arm is ALSO teacher-forced on the "0" arm's tokens so its logits are comparable
step by step. An fp32 twin (same weights, experts as the bf16 / bnb-dequantized dense fp32
stack) is teacher-forced on the same tokens with "0" as the accuracy reference.
Counts every routed_moe_forward call / non-None return / _nf4_routed / routed_bf16_moe call
(head only; the module is absent at the base). Writes JSON to --out and logits to --logits.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import traceback
from pathlib import Path

os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")
TINY_ID = "trl-internal-testing/tiny-Qwen3MoeForCausalLM"


def _sha(t) -> str:
    import torch
    return hashlib.sha256(t.detach().to(torch.float32).cpu().contiguous().numpy().tobytes()).hexdigest()[:16]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices = ["hub", "kf"], required = True)
    ap.add_argument("--weights", choices = ["bf16", "nf4"], required = True)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--logits", required = True, type = Path)
    ap.add_argument("--batch", type = int, default = 1)
    ap.add_argument("--prompt-len", type = int, default = 40)
    ap.add_argument("--new", type = int, default = 8)
    ap.add_argument("--lora", action = "store_true", help = "stash a per-expert LoRA on every experts module "
                    "(how PEFT target_parameters LoRA reaches the experts; plain BF16 stacks without it take "
                    "transformers' grouped_mm / batched_mm, not unsloth_zoo's backends)")
    args = ap.parse_args()
    res: dict = {"model": args.model, "weights": args.weights, "batch": args.batch, "lora": args.lora, "stage": "import",
                 "env_routed": os.environ.get("UNSLOTH_MOE_ROUTED_KERNEL", "<unset>")}

    def dump():
        args.out.write_text(json.dumps(res, indent = 2, default = str), encoding = "utf-8")

    try:
        import torch
        import torch.nn as nn
        res["versions"] = {"torch": torch.__version__, "hip": getattr(torch.version, "hip", None),
                           "cuda": getattr(torch.version, "cuda", None)}
        props = torch.cuda.get_device_properties(0)
        res["versions"].update(device = props.name, gcnArchName = getattr(props, "gcnArchName", None),
                               capability = list(torch.cuda.get_device_capability(0)))
        import bitsandbytes as bnb
        import transformers
        res["versions"].update(bitsandbytes = bnb.__version__, transformers = transformers.__version__)
        try:
            import triton
            res["versions"]["triton"] = triton.__version__
        except Exception as e:  # noqa: BLE001
            res["versions"]["triton"] = f"unavailable: {e!r}"
        import unsloth_zoo
        res["unsloth_zoo_file"] = unsloth_zoo.__file__
        from unsloth_zoo.temporary_patches.moe_experts_interface import patch_experts_interface
        try:
            from unsloth_zoo.temporary_patches import moe_routed as MR
        except ImportError:
            MR = None
        res["has_moe_routed"] = MR is not None
        patch_experts_interface()

        from transformers import AutoConfig, AutoModelForCausalLM
        DEV, DT = "cuda", torch.bfloat16
        res["stage"] = "build"
        dump()
        try:
            cfg = AutoConfig.from_pretrained(TINY_ID)
            res["config_source"] = TINY_ID
        except Exception as e:  # noqa: BLE001
            if args.model == "hub":
                raise
            from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
            cfg = Qwen3MoeConfig()
            res["config_source"] = f"Qwen3MoeConfig() defaults ({type(e).__name__}: {e})"
        if args.model == "kf":
            for k, v in dict(vocab_size = 512, hidden_size = 256, moe_intermediate_size = 192, intermediate_size = 512,
                             num_hidden_layers = 2, num_attention_heads = 4, num_key_value_heads = 2, head_dim = 64,
                             num_experts = 16, num_experts_per_tok = 4, max_position_embeddings = 256,
                             mlp_only_layers = [], decoder_sparse_step = 1).items():
                setattr(cfg, k, v)
        cfg._experts_implementation = "unsloth"
        torch.manual_seed(0)
        if args.model == "hub":
            model = AutoModelForCausalLM.from_pretrained(TINY_ID, config = cfg, dtype = DT)
        else:
            model = AutoModelForCausalLM.from_config(cfg, dtype = DT)
        model = model.to(DEV).eval()
        if args.model == "kf":
            with torch.no_grad():
                for p in model.parameters():
                    if p.dim() == 2 and tuple(p.shape) == (cfg.num_experts, cfg.hidden_size):
                        p.mul_(30)
        res["config"] = {k: getattr(cfg, k, None) for k in ("hidden_size", "moe_intermediate_size", "num_experts",
                                                            "num_experts_per_tok", "vocab_size", "num_hidden_layers")}
        ref = copy.deepcopy(model).float()
        experts = []
        for module, twin in zip(model.modules(), ref.modules()):
            gu = module._parameters.get("gate_up_proj") if hasattr(module, "_parameters") else None
            if gu is None or gu.dim() != 3:
                continue
            experts.append(type(module).__name__)
            if args.weights == "nf4":
                for name in ("gate_up_proj", "down_proj"):
                    w = getattr(module, name).detach()
                    p = bnb.nn.Params4bit(w.to(DT).cpu(), requires_grad = False, compress_statistics = True,
                                          quant_type = "nf4", blocksize = 64).to(DEV)
                    p._original_shape = tuple(w.shape)
                    setattr(module, name, p)
                    dq = bnb.functional.dequantize_4bit(p.data, p.quant_state).float().view(p._original_shape)
                    setattr(twin, name, nn.Parameter(dq, requires_grad = False))
            if args.lora:
                from unsloth_zoo.temporary_patches.moe_utils import moe_lora_stash_name
                gl = torch.Generator().manual_seed(3 + len(experts))
                e, n_gu, h = tuple(getattr(module.gate_up_proj, "_original_shape", module.gate_up_proj.shape))
                r, i = 8, n_gu // 2
                gu_l = (torch.randn(e, h, r, generator = gl) * 0.1, torch.randn(e, r, n_gu, generator = gl) * 0.1, 0.5, e)
                dn_l = (torch.randn(e, i, r, generator = gl) * 0.1, torch.randn(e, r, h, generator = gl) * 0.1, 2.0, e)
                for mod, cast in ((module, DT), (twin, torch.float32)):
                    for pname, term in (("gate_up_proj", gu_l), ("down_proj", dn_l)):
                        setattr(mod, moe_lora_stash_name(pname),
                                tuple(t.to(DEV, cast) if isinstance(t, torch.Tensor) else t for t in term))
        res["experts_modules"] = experts
        res["experts_impl"] = getattr(cfg, "_experts_implementation", None)
        if not experts:
            raise RuntimeError("no 3D experts stacks in this transformers version")

        counts = {"routed_moe_forward_calls": 0, "routed_moe_forward_returned": 0, "nf4_routed": 0,
                  "nf4_grouped": 0, "bf16_routed": 0}
        if MR is not None:
            real_fwd, real_r, real_g, real_b = MR.routed_moe_forward, MR._nf4_routed, MR._nf4_grouped, MR.routed_bf16_moe

            def fwd(*a, **k):
                counts["routed_moe_forward_calls"] += 1
                out = real_fwd(*a, **k)
                counts["routed_moe_forward_returned"] += out is not None
                return out

            def wrap(key, fn):
                def w(*a, **k):
                    counts[key] += 1
                    return fn(*a, **k)
                return w
            MR.routed_moe_forward = fwd
            MR._nf4_routed = wrap("nf4_routed", real_r)
            MR._nf4_grouped = wrap("nf4_grouped", real_g)
            MR.routed_bf16_moe = wrap("bf16_routed", real_b)

        g = torch.Generator().manual_seed(1234 + args.batch)
        V = cfg.vocab_size
        prompt = torch.randint(0, V, (args.batch, args.prompt_len), generator = g).to(DEV)

        def decode(m, mode, forced = None):
            if mode is None:
                os.environ.pop("UNSLOTH_MOE_ROUTED_KERNEL", None)
            else:
                os.environ["UNSLOTH_MOE_ROUTED_KERNEL"] = mode
            for k in counts:
                counts[k] = 0
            with torch.no_grad():
                out = m(prompt, use_cache = True)
                prefill = dict(counts)
                cache = out.past_key_values
                nxt = out.logits[:, -1].float().argmax(-1, keepdim = True)
                toks, logits = [], []
                for s in range(args.new):
                    feed = nxt if forced is None else forced[:, s:s + 1]
                    toks.append(feed)
                    out = m(feed, past_key_values = cache, use_cache = True)
                    cache = out.past_key_values
                    lg = out.logits[:, -1].float()
                    logits.append(lg)
                    nxt = lg.argmax(-1, keepdim = True)
            torch.cuda.synchronize()
            total = dict(counts)
            decode_counts = {k: total[k] - prefill[k] for k in total}
            return torch.cat(toks, 1), torch.stack(logits, 1), {"prefill": prefill, "decode": decode_counts}

        res["stage"] = "arm0"
        dump()
        t0, l0, c0 = decode(model, "0")
        res["arm0"] = {"tokens": t0.tolist(), "counts": c0, "logits_sha": _sha(l0), "finite": bool(torch.isfinite(l0).all())}
        res["stage"] = "arm_default"
        dump()
        t1, l1_free, c1 = decode(model, None)
        res["default_free"] = {"tokens": t1.tolist(), "counts": c1, "logits_sha": _sha(l1_free),
                               "finite": bool(torch.isfinite(l1_free).all())}
        _, l1, c1f = decode(model, None, forced = t0)
        res["default_forced"] = {"counts": c1f, "logits_sha": _sha(l1), "finite": bool(torch.isfinite(l1).all())}
        res["stage"] = "ref"
        dump()
        _, lr, _ = decode(ref, "0", forced = t0)
        os.environ.pop("UNSLOTH_MOE_ROUTED_KERNEL", None)
        d = (l1 - l0).abs()
        e0, e1 = (l0 - lr).abs(), (l1 - lr).abs()
        # Per step: does the forced default arm pick the same argmax as the "0" arm, and how
        # wide is the "0" arm's top-1 / top-2 margin there (a near tie can flip legitimately).
        top2 = l0.topk(2, dim = -1).values
        # Forced-step argmax flips, each with the "0" arm's top-1 / top-2 margin and that row's
        # max |default - 0|: a flip inside 2x the row's own difference is a near tie, not an error.
        a0, a1 = l0.argmax(-1), l1.argmax(-1)
        flips = []
        for b, s in (a0 != a1).nonzero().tolist():
            m = float(top2[b, s, 0] - top2[b, s, 1])
            rd = float(d[b, s].max())
            flips.append({"batch": b, "step": s, "margin": m, "row_max_abs_diff": rd, "near_tie": m <= 2 * rd})
        res["forced_flips"] = flips
        res["compare"] = {
            "n_forced_flips": len(flips), "n_unexplained_flips": sum(not f["near_tie"] for f in flips),
            "tokens_equal_free_run": bool(torch.equal(t0, t1)),
            "argmax_equal_forced": bool(torch.equal(l0.argmax(-1), l1.argmax(-1))),
            "min_top2_margin_arm0": float((top2[..., 0] - top2[..., 1]).min()),
            "max_abs_default_vs_0": float(d.max()), "mean_abs_default_vs_0": float(d.mean()),
            "bit_identical_default_vs_0": bool(torch.equal(l0, l1)),
            "max_abs_0_vs_fp32": float(e0.max()), "mean_abs_0_vs_fp32": float(e0.mean()),
            "max_abs_default_vs_fp32": float(e1.max()), "mean_abs_default_vs_fp32": float(e1.mean()),
            "logit_absmax": float(l0.abs().max()),
        }
        torch.save({"arm0": l0.cpu(), "default": l1.cpu(), "ref": lr.cpu(), "t0": t0.cpu(), "t1": t1.cpu()}, args.logits)
        res["stage"] = "done"
        res["ok"] = True
    except Exception as e:  # noqa: BLE001
        res["error"] = f"{type(e).__name__}: {e}"
        res["traceback"] = traceback.format_exc()[-4000:]
    dump()
    return 0


if __name__ == "__main__":
    sys.exit(main())
