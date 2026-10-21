#!/usr/bin/env python3
"""One tiny gpt-oss LoRA training cell for unsloth-zoo PR 1631 (flex attention block-mask reuse). Observes only.

Run in its own process per cell, PYTHONPATH = the state's unsloth-zoo checkout first.
trl-internal-testing/tiny-GptOssForCausalLM (2 layers: sliding 128 + full) through unsloth FastLanguageModel bf16,
LoRA on q/k/v/o, use_gradient_checkpointing="unsloth", manual loop of --steps AdamW steps on fixed data
(seq lens 256, 256, 192: the last step is a new shape). Records per step: loss repr, logits digest + sums + a slice,
LoRA grad digest + per-tensor grad norms, post-step param digest; block-mask builds (every call of
unsloth_zoo.flex_attention.utils._create_block_mask, which compiled_create_block_mask resolves at call time, at
base and head alike) and flex_attention_with_sink calls with training=True. JSON to --out.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import traceback
import zlib
from pathlib import Path

MODEL = "trl-internal-testing/tiny-GptOssForCausalLM"
SEQS = (256, 256, 192)


def _digest(items) -> str:
    h = hashlib.sha256()
    for n, t in items:
        h.update(n.encode())
        h.update(b"<none>" if t is None else t.detach().float().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--steps", type = int, default = 3)
    ap.add_argument("--model", default = MODEL)
    args = ap.parse_args()
    os.environ["UNSLOTH_COMPILE_LOCATION"] = str(args.out.parent / f"cache_{args.out.stem}")
    os.environ.setdefault("UNSLOTH_RETURN_LOGITS", "1")
    res: dict = {"stage": "import", "model": args.model,
                 "env": {k: os.environ.get(k) for k in ("UNSLOTH_FLEX_MASK_REUSE", "UNSLOTH_RETURN_LOGITS",
                                                        "TORCHDYNAMO_DISABLE", "UNSLOTH_COMPILE_DISABLE")}}

    def dump():
        args.out.write_text(json.dumps(res, indent = 2, default = str), encoding = "utf-8")

    try:
        import unsloth  # noqa: F401  (before transformers)
        from unsloth import FastLanguageModel
        import torch
        import transformers
        import unsloth_zoo
        res["unsloth_zoo_file"] = unsloth_zoo.__file__
        res["unsloth_file"] = unsloth.__file__
        res["versions"] = {"torch": torch.__version__, "hip": getattr(torch.version, "hip", None),
                           "cuda": getattr(torch.version, "cuda", None), "transformers": transformers.__version__}
        for mod in ("peft", "trl", "triton"):
            try:
                res["versions"][mod] = __import__(mod).__version__
            except Exception as e:  # noqa: BLE001
                res["versions"][mod] = f"unavailable: {e!r}"
        try:
            props = torch.cuda.get_device_properties(0)
            res["versions"]["device"] = props.name
            res["versions"]["gcnArchName"] = getattr(props, "gcnArchName", None)
        except Exception as e:  # noqa: BLE001
            res["versions"]["device_error"] = repr(e)

        from unsloth_zoo.flex_attention import utils as FU
        res["has_flex_attention"] = bool(getattr(FU, "HAS_FLEX_ATTENTION", False))
        res["reuse_fn_present"] = getattr(FU, "reused_compiled_create_block_mask", None) is not None
        counts = {"create_block_mask": 0, "shapes": []}
        if res["has_flex_attention"]:
            real_cbm = FU._create_block_mask

            def counted_cbm(mask_mod, B, H, M, N, *a, **k):
                counts["create_block_mask"] += 1
                counts["shapes"].append([getattr(mask_mod, "__name__", type(mask_mod).__name__), B, H, M, N,
                                         bool(torch.is_inference_mode_enabled())])
                return real_cbm(mask_mod, B, H, M, N, *a, **k)
            FU._create_block_mask = counted_cbm

        res["stage"] = "load"
        dump()
        model, tok = FastLanguageModel.from_pretrained(args.model, max_seq_length = 512, dtype = torch.bfloat16,
                                                       load_in_4bit = False, device_map = {"": 0})
        res["unsloth_model_name_env"] = os.environ.get("UNSLOTH_MODEL_NAME")
        res["stage"] = "peft"
        dump()
        model = FastLanguageModel.get_peft_model(
            model, r = 8, lora_alpha = 16, lora_dropout = 0, bias = "none",
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"],
            use_gradient_checkpointing = "unsloth", random_state = 3407)
        attn_cls = None
        for n, m in model.named_modules():
            if type(m).__name__ == "GptOssAttention":
                attn_cls = type(m)
                break
        res["attention_forward_module"] = getattr(getattr(attn_cls, "forward", None), "__module__", None)
        # flex_attention_with_sink calls in training (wrapped where gpt_oss.py resolves it)
        sink_calls = {"n": 0}
        try:
            import unsloth_zoo.temporary_patches.gpt_oss as GP
            fws = None
            for holder in (GP, ):
                fws = getattr(holder, "flex_attention_with_sink", None)
            res["gpt_oss_flex_sink_installed"] = getattr(GP, "_GPT_OSS_FLEX_SINK_ATTENTION_INSTALLED", None)
        except Exception as e:  # noqa: BLE001
            res["gpt_oss_import_error"] = repr(e)
        import unsloth_zoo.flex_attention as FA
        import unsloth_zoo.flex_attention.attention_sink as AS
        real_fws = AS.flex_attention_with_sink

        def counted_fws(self_attn, *a, **k):
            sink_calls["n"] += 1
            return real_fws(self_attn, *a, **k)
        # gpt_oss.forward_function imports flex_attention_with_sink from unsloth_zoo.flex_attention inside the patch
        # closure, so the patched forward's globals/closure hold it; count via the closure cell where possible.
        patched = 0
        fwd = getattr(attn_cls, "forward", None)
        seen = set()
        stack = [fwd]
        while stack:
            f = stack.pop()
            if f is None or id(f) in seen:
                continue
            seen.add(id(f))
            for cell in (getattr(f, "__closure__", None) or ()):
                try:
                    v = cell.cell_contents
                except ValueError:
                    continue
                if v is real_fws:
                    cell.cell_contents = counted_fws
                    patched += 1
                elif callable(v) and hasattr(v, "__closure__"):
                    stack.append(v)
            g = getattr(f, "__globals__", {})
            if g.get("flex_attention_with_sink") is real_fws:
                g["flex_attention_with_sink"] = counted_fws
                patched += 1
        res["sink_counter_hooks"] = patched

        lora = sorted((n, p) for n, p in model.named_parameters() if p.requires_grad)
        with torch.no_grad():
            for n, p in lora:
                if "lora_B" in n:
                    g = torch.Generator().manual_seed(zlib.crc32(n.encode()))
                    p.copy_((torch.randn(tuple(p.shape), generator = g) * 0.02).to(p.dtype))
        res["n_trainable_tensors"] = len(lora)
        res["digest_init"] = _digest(lora)
        opt = torch.optim.AdamW([p for _, p in lora], lr = 5e-3)
        vocab = int(model.config.vocab_size)
        g = torch.Generator().manual_seed(0)
        model.train()
        steps = []
        res["stage"] = "train"
        dump()
        for i in range(args.steps):
            seq = SEQS[i % len(SEQS)]
            ids = torch.randint(4, max(5, vocab - 1), (2, seq), generator = g).cuda()
            c0, s0 = counts["create_block_mask"], sink_calls["n"]
            out = model(input_ids = ids, labels = ids, attention_mask = torch.ones_like(ids))
            loss = out.loss
            loss.backward()
            torch.cuda.synchronize()
            rec = {"seq": seq, "loss": repr(float(loss.detach()))}
            lg = getattr(out, "logits", None)
            if lg is not None and torch.is_tensor(lg) and lg.numel() > 1:
                lf = lg.detach().float()
                rec["logits_digest"] = _digest([("logits", lf)])
                rec["logits_sum"] = repr(float(lf.double().sum()))
                rec["logits_abs_sum"] = repr(float(lf.double().abs().sum()))
                rec["logits_slice"] = [repr(float(x)) for x in lf[0, -1, :16].cpu()]
                rec["logits_shape"] = list(lf.shape)
            else:
                rec["logits"] = f"unavailable ({type(lg).__name__})"
            grads = [(n, p.grad) for n, p in lora]
            rec["grad_digest"] = _digest(grads)
            rec["grad_none"] = sum(1 for _, t in grads if t is None)
            rec["grad_norms"] = {n: (None if t is None else repr(float(t.detach().float().norm()))) for n, t in grads}
            rec["finite"] = bool(torch.isfinite(loss).item()) and all(t is None or bool(torch.isfinite(t).all()) for _, t in grads)
            opt.step()
            opt.zero_grad(set_to_none = True)
            rec["param_digest"] = _digest(sorted((n, p) for n, p in model.named_parameters() if p.requires_grad))
            rec["mask_builds"] = counts["create_block_mask"] - c0
            rec["sink_calls"] = sink_calls["n"] - s0
            steps.append(rec)
            res["steps"] = steps
            dump()
        res["mask_builds_total"] = counts["create_block_mask"]
        res["mask_build_shapes"] = counts["shapes"][:64]
        res["sink_calls_total"] = sink_calls["n"]
        res["final_digest"] = steps[-1]["param_digest"]
        cache = getattr(FU, "_BLOCK_MASK_CACHE", None)
        res["block_mask_cache_len"] = None if cache is None else len(cache)
        res["stage"] = "done"
        res["ok"] = bool(steps) and all(s["finite"] for s in steps)
    except Exception as e:  # noqa: BLE001
        res["error"] = f"{type(e).__name__}: {str(e)[:1500]}"
        res["traceback"] = traceback.format_exc()[-4000:]
    dump()
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
