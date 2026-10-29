#!/usr/bin/env python3
"""One arm of the PR 12739 probe (bitsandbytes NF4 Linear4bit.forward override), in its own process.

PYTHONPATH puts the state's unsloth checkout first. Observes only, JSON to --out:
  * whether the override module exists, installed, Linear4bit.forward carries its mark
  * engagements: calls into bnb_override._linear (the Unsloth Triton path) and _forward
  * total Linear4bit.forward calls (so a 0-engagement count is not vacuous)
  * a standalone bnb NF4 Linear4bit forward+backward and a tiny 4-bit Llama LoRA
    forward+backward (LoRA on attention only, so MLP Linear4bit.forward runs), with
    sha256 of outputs / loss / grads for bit-identity across states.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import traceback
from pathlib import Path

MODELS = ["trl-internal-testing/tiny-LlamaForCausalLM-3.2", "hf-internal-testing/tiny-random-LlamaForCausalLM"]


def _sha(t) -> str:
    import torch
    t = t.detach().contiguous().cpu()
    if t.dtype == torch.bfloat16:
        t = t.view(torch.int16)
    return hashlib.sha256(t.numpy().tobytes()).hexdigest()[:24]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required = True, type = Path)
    args = ap.parse_args()
    res: dict = {"arm_env": os.environ.get("UNSLOTH_BNB_NF4_LINEAR", "<unset>"), "stage": "import"}

    def dump():
        args.out.write_text(json.dumps(res, indent = 2, default = str), encoding = "utf-8")

    try:
        import unsloth  # noqa: F401
        from unsloth import FastLanguageModel
        import torch
        import bitsandbytes as bnb
        res["unsloth_file"] = unsloth.__file__
        res["versions"] = {"torch": torch.__version__, "hip": torch.version.hip, "cuda": torch.version.cuda,
                           "bitsandbytes": bnb.__version__, "unsloth": getattr(unsloth, "__version__", None)}
        try:
            import triton
            res["versions"]["triton"] = triton.__version__
        except Exception as e:  # noqa: BLE001
            res["versions"]["triton"] = repr(e)
        try:
            import unsloth_zoo, transformers, peft
            res["versions"].update(unsloth_zoo = unsloth_zoo.__version__, transformers = transformers.__version__,
                                   peft = peft.__version__)
        except Exception as e:  # noqa: BLE001
            res["versions"]["other"] = repr(e)
        p = torch.cuda.get_device_properties(0)
        res["versions"]["device"] = p.name
        res["versions"]["arch"] = getattr(p, "gcnArchName", None)
        try:
            from unsloth.kernels import bnb_override as O
        except ImportError:
            O = None
        res["has_override_module"] = O is not None
        counts = {"engaged_linear": 0, "override_forward": 0, "bnb_forward_total": 0}
        if O is not None:
            res["wanted"] = bool(O._wanted())
            orig_linear, orig_forward = O._linear, O._forward

            def c_linear(*a, **k):
                counts["engaged_linear"] += 1
                return orig_linear(*a, **k)

            def c_forward(*a, **k):
                counts["override_forward"] += 1
                return orig_forward(*a, **k)
            O._linear, O._forward = c_linear, c_forward

        def wrap_total():
            inner = bnb.nn.Linear4bit.forward
            if getattr(inner, "_amdci_counted", False):
                return
            res.setdefault("forward_marks", []).append(bool(getattr(inner, "_unsloth_nf4_override", False)))

            def total(self, x, *a, **k):
                counts["bnb_forward_total"] += 1
                return inner(self, x, *a, **k)
            total._amdci_counted = True
            total._inner = inner
            bnb.nn.Linear4bit.forward = total

        # ---- tiny model (from_pretrained is where the PR installs the override)
        res["stage"] = "model"
        dump()
        model = tok = None
        for name in MODELS:
            try:
                model, tok = FastLanguageModel.from_pretrained(name, max_seq_length = 128, load_in_4bit = True,
                                                               dtype = torch.bfloat16)
                res["model"] = name
                break
            except Exception as e:  # noqa: BLE001
                res.setdefault("model_errors", {})[name] = f"{type(e).__name__}: {e}"[:500]
        if model is None:
            raise RuntimeError("no tiny model loaded")
        res["installed_mark_after_load"] = bool(getattr(bnb.nn.Linear4bit.forward, "_unsloth_nf4_override", False))
        wrap_total()
        model = FastLanguageModel.get_peft_model(model, r = 8, lora_alpha = 16, random_state = 3407,
                                                 target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"])
        n4 = sum(1 for m in model.modules() if isinstance(m, bnb.nn.Linear4bit))
        res["n_linear4bit"] = n4
        model.train()
        g = torch.Generator().manual_seed(0)
        vocab = model.config.vocab_size
        ids = torch.randint(0, vocab, (2, 64), generator = g).cuda()
        out = model(input_ids = ids, labels = ids)
        loss = out.loss
        loss.backward()
        lora = sorted((n, p.grad) for n, p in model.named_parameters() if p.requires_grad and p.grad is not None)
        res["model_probe"] = {
            "loss": repr(float(loss)), "loss_sha": _sha(loss.float().reshape(1)),
            "logits_sha": _sha(out.logits) if getattr(out, "logits", None) is not None and out.logits.numel() else None,
            "grads_sha": _sha(torch.cat([gr.float().flatten() for _, gr in lora])) if lora else None,
            "n_grads": len(lora), "counts": dict(counts),
        }
        res["stage"] = "layer"
        dump()
        # ---- standalone NF4 layer (bf16, nested), forward + backward through Linear4bit.forward
        before = dict(counts)
        torch.manual_seed(0)
        K, N = 512, 384
        lin = bnb.nn.Linear4bit(K, N, bias = False, compute_dtype = torch.bfloat16, compress_statistics = True,
                                quant_type = "nf4", quant_storage = torch.uint8)
        lin.weight = bnb.nn.Params4bit(torch.randn(N, K).to(torch.bfloat16), requires_grad = False,
                                       compress_statistics = True, quant_type = "nf4")
        lin = lin.cuda()
        x = torch.randn(4, 32, K, device = "cuda", dtype = torch.bfloat16, requires_grad = True)
        y = lin(x)
        y.float().square().mean().backward()
        with torch.no_grad():
            y1 = lin(torch.randn(1, 1, K, device = "cuda", dtype = torch.bfloat16, generator = None))
        res["layer_probe"] = {"y_sha": _sha(y), "dx_sha": _sha(x.grad), "decode_shape": list(y1.shape),
                              "counts": {k: counts[k] - before[k] for k in counts}}
        res["counts"] = counts
        res["stage"] = "done"
        res["ok"] = True
    except Exception as e:  # noqa: BLE001
        res["error"] = f"{type(e).__name__}: {e}"
        res["traceback"] = traceback.format_exc()[-4000:]
    dump()
    return 0


if __name__ == "__main__":
    sys.exit(main())
