#!/usr/bin/env python3
"""One training cell for unsloth-zoo PR 1604 (Triton grouped GEMM for every MoE behind a static arch / shape gate).
Observes only; writes JSON to --out.

Run in its own process per cell, PYTHONPATH = the state's unsloth-zoo checkout first.
Tiny Qwen3-MoE (trl-internal-testing/tiny-Qwen3MoeForCausalLM) through unsloth FastLanguageModel in bf16 (16-bit),
LoRA on attention + MLP / expert projections, N AdamW steps on a fixed batch. Records:
  * per-step loss repr (exact), per-step LoRA-grad digest, final LoRA digest
  * moe_utils._triton_grouped_mm_max_rows(i, k) for i in (0, None) and every k, under the cell's environment
    and with UNSLOTH_MOE_GROUPED_TRITON forced to "1" (head only; absent at base)
  * moe_grouped_fp16.GENERIC_CALLS before / after training (head only)
  * how often the static gate was evaluated and how often it said yes (eager cells, head only), and how often
    torch._grouped_mm itself was called (eager cells), so "gate never fired" is not vacuous
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import traceback
from pathlib import Path

MODEL = "trl-internal-testing/tiny-Qwen3MoeForCausalLM"
KINDS = ("lora", "many", "few", "dw")


def _digest(tensors) -> str:
    h = hashlib.sha256()
    for n, t in tensors:
        h.update(n.encode())
        if t is None:
            h.update(b"<none>")
        else:
            h.update(t.detach().float().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--steps", type = int, default = 4)
    ap.add_argument("--model", default = MODEL)
    ap.add_argument("--instrument", action = "store_true", help = "count gate / torch._grouped_mm calls (eager only)")
    args = ap.parse_args()
    os.environ["UNSLOTH_COMPILE_LOCATION"] = str(args.out.parent / f"cache_{args.out.stem}")
    res: dict = {"stage": "import", "model": args.model,
                 "env": {k: os.environ.get(k) for k in ("UNSLOTH_MOE_GROUPED_TRITON", "UNSLOTH_DISABLE_MOE_TRITON",
                                                       "UNSLOTH_MOE_GROUPED_TRITON_MAX_ROWS", "TORCHDYNAMO_DISABLE",
                                                       "UNSLOTH_MOE_BACKEND")}}

    def dump():
        args.out.write_text(json.dumps(res, indent = 2, default = str), encoding = "utf-8")

    try:
        import unsloth  # noqa: F401  (before transformers)
        from unsloth import FastLanguageModel
        import torch
        import transformers
        import unsloth_zoo
        from unsloth_zoo.temporary_patches import moe_utils as mu
        res["unsloth_zoo_file"] = unsloth_zoo.__file__
        res["unsloth_file"] = unsloth.__file__
        res["versions"] = {"torch": torch.__version__, "hip": getattr(torch.version, "hip", None),
                           "cuda": getattr(torch.version, "cuda", None), "transformers": transformers.__version__}
        for mod in ("peft", "trl", "triton", "bitsandbytes"):
            try:
                res["versions"][mod] = __import__(mod).__version__
            except Exception as e:  # noqa: BLE001
                res["versions"][mod] = f"unavailable: {e!r}"
        try:
            props = torch.cuda.get_device_properties(0)
            res["versions"]["device"] = props.name
            res["versions"]["gcnArchName"] = getattr(props, "gcnArchName", None)
            res["versions"]["capability"] = list(torch.cuda.get_device_capability(0))
        except Exception as e:  # noqa: BLE001
            res["versions"]["device_error"] = repr(e)
        try:
            res["moe_backend"] = mu.get_forward_moe_backend()
        except Exception as e:  # noqa: BLE001
            res["moe_backend"] = f"error {e!r}"
        try:
            res["grouped_mm_supported"] = bool(mu._check_torch_grouped_mm_supported())
        except Exception as e:  # noqa: BLE001
            res["grouped_mm_supported"] = f"error {e!r}"

        # PR 1604 symbols (head only)
        res["has_max_rows"] = hasattr(mu, "_triton_grouped_mm_max_rows")
        mg = None
        try:
            from unsloth_zoo.temporary_patches import moe_grouped_fp16 as mg
        except ImportError:
            pass
        res["has_generic_calls"] = mg is not None and hasattr(mg, "GENERIC_CALLS")
        res["triton_grouped_op_registered"] = getattr(mu, "_GROUPED_MM_TRITON_OP", "<absent>") is not None \
            if hasattr(mu, "_GROUPED_MM_TRITON_OP") else "<absent>"
        if res["has_max_rows"]:
            fn = mu._triton_grouped_mm_max_rows
            rows = {}
            for idx in (0, None):
                rows[str(idx)] = {k: fn(idx, k) for k in KINDS}
            res["max_rows_env"] = rows
            saved = os.environ.get("UNSLOTH_MOE_GROUPED_TRITON")
            forced = {}
            try:
                for mode in ("1", "auto"):
                    os.environ["UNSLOTH_MOE_GROUPED_TRITON"] = mode
                    forced[mode] = {k: fn(0, k) for k in KINDS}
            finally:
                if saved is None:
                    os.environ.pop("UNSLOTH_MOE_GROUPED_TRITON", None)
                else:
                    os.environ["UNSLOTH_MOE_GROUPED_TRITON"] = saved
            res["max_rows_forced_modes_dev0"] = forced
            try:
                res["triton_grouped_available_dev0"] = bool(mg.triton_grouped_available(torch.device("cuda", 0)))
            except Exception as e:  # noqa: BLE001
                res["triton_grouped_available_dev0"] = f"error {e!r}"
        if res["has_generic_calls"]:
            res["generic_calls_at_import"] = dict(mg.GENERIC_CALLS)

        gate = {"evaluated": 0, "true": 0}
        gmm = {"n": 0}
        if args.instrument:
            if res["has_max_rows"]:
                orig_wanted = mu._triton_grouped_mm_wanted

                def counted_wanted(inputs, weight):
                    r = orig_wanted(inputs, weight)
                    gate["evaluated"] += 1
                    gate["true"] += int(bool(r))
                    return r
                mu._triton_grouped_mm_wanted = counted_wanted
                try:
                    from unsloth_zoo.temporary_patches import moe_grouped_modulelist as mgl
                    if getattr(mgl, "_triton_grouped_mm_wanted", None) is not None:
                        mgl._triton_grouped_mm_wanted = counted_wanted
                except Exception:  # noqa: BLE001
                    pass
            if hasattr(torch, "_grouped_mm"):
                orig_gmm = torch._grouped_mm

                def counted_gmm(*a, **k):
                    gmm["n"] += 1
                    return orig_gmm(*a, **k)
                torch._grouped_mm = counted_gmm

        res["stage"] = "load"
        dump()
        dtype = torch.bfloat16
        model, tok = FastLanguageModel.from_pretrained(args.model, max_seq_length = 128, dtype = dtype,
                                                       load_in_4bit = False, device_map = {"": 0})
        cfg = model.config
        res["config"] = {k: getattr(cfg, k, None) for k in ("num_experts", "num_experts_per_tok", "hidden_size",
                                                             "moe_intermediate_size", "num_hidden_layers", "vocab_size")}
        res["mlp_class"] = sorted({type(m).__name__ for n, m in model.named_modules() if n.endswith(".mlp")})
        res["experts_class"] = sorted({type(m).__name__ for n, m in model.named_modules() if n.endswith(".experts")})
        res["stage"] = "peft"
        dump()
        model = FastLanguageModel.get_peft_model(
            model, r = 8, lora_alpha = 16, lora_dropout = 0, bias = "none",
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            use_gradient_checkpointing = "unsloth", random_state = 3407)
        torch.manual_seed(1)
        for n, p in model.named_parameters():
            if "lora_B" in n:
                p.data.normal_(0, 0.02)
        lora = sorted((n, p) for n, p in model.named_parameters() if p.requires_grad)
        expert_lora = [n for n, _ in lora if ".experts" in n or ".mlp." in n]
        res["n_lora_params"], res["n_mlp_or_expert_lora_params"] = len(lora), len(expert_lora)
        res["mlp_lora_names_sample"] = expert_lora[:6]
        res["lora_param_dtypes"] = sorted({str(p.dtype) for _, p in lora})
        res["lora_digest_init"] = _digest(lora)

        vocab = int(model.config.vocab_size)
        g = torch.Generator().manual_seed(0)
        ids = torch.randint(4, max(5, vocab - 1), (2, 64), generator = g).cuda()
        opt = torch.optim.AdamW([p for _, p in lora], lr = 1e-3)
        model.train()
        res["stage"] = "train"
        gc0 = dict(mg.GENERIC_CALLS) if res["has_generic_calls"] else None
        dump()
        loss_reprs, grad_digests, grad_norms = [], [], []
        for step in range(args.steps):
            opt.zero_grad(set_to_none = True)
            with torch.autocast("cuda", dtype = dtype):
                loss = model(input_ids = ids, labels = ids).loss
            loss.backward()
            torch.cuda.synchronize()
            grad_digests.append(_digest([(n, p.grad) for n, p in lora]))
            gn = [float(p.grad.float().norm()) for n, p in lora if p.grad is not None and n in expert_lora]
            grad_norms.append(sum(x * x for x in gn) ** 0.5 if gn else None)
            opt.step()
            torch.cuda.synchronize()
            loss_reprs.append(repr(float(loss)))
            res.update(loss_reprs = loss_reprs, grad_digests = grad_digests, mlp_lora_grad_norm = grad_norms)
            dump()
        res["losses"] = [float(x) for x in loss_reprs]
        res["lora_digest"] = _digest(lora)
        if res["has_generic_calls"]:
            res["generic_calls_delta"] = {k: mg.GENERIC_CALLS[k] - gc0.get(k, 0) for k in mg.GENERIC_CALLS}
            res["generic_calls_final"] = dict(mg.GENERIC_CALLS)
        res["gate_counts"] = gate if args.instrument else None
        res["torch_grouped_mm_calls"] = gmm["n"] if args.instrument else None
        res["instrumented"] = bool(args.instrument)
        res["stage"] = "done"
        res["ok"] = all(v == v and abs(v) != float("inf") for v in res["losses"])
    except Exception as e:  # noqa: BLE001
        res["error"] = f"{type(e).__name__}: {str(e)[:1500]}"
        res["traceback"] = traceback.format_exc()[-4000:]
    dump()
    return 0


if __name__ == "__main__":
    sys.exit(main())
