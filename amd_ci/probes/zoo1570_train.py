#!/usr/bin/env python3
"""One training arm for unsloth-zoo PR 1570 (grouped QLoRA for gpt-oss NF4 experts).

Run in its own process per arm (UNSLOTH_GPTOSS_GROUPED unset vs "0"), with
PYTHONPATH pointing at the state's unsloth-zoo checkout. Builds the tiny NF4
gpt-oss experts module from tests/test_gpt_oss_grouped_qlora.py, wraps it with
PEFT LoRA, runs 3 SGD steps and records losses, LoRA grad norms, which path ran
(grouped forward returned a tensor vs per-expert loop calls), and gq.CALLS when
the module exists. Observes only; writes JSON to --out and step-0 grads to --grads.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--grads", required = True, type = Path)
    ap.add_argument("--steps", type = int, default = 3)
    args = ap.parse_args()
    res: dict = {"arm_env": os.environ.get("UNSLOTH_GPTOSS_GROUPED", "<unset>"), "stage": "import"}

    def dump():
        args.out.write_text(json.dumps(res, indent = 2, default = str), encoding = "utf-8")

    try:
        import torch
        res["versions"] = {"torch": torch.__version__, "hip": getattr(torch.version, "hip", None),
                           "cuda": getattr(torch.version, "cuda", None)}
        try:
            props = torch.cuda.get_device_properties(0)
            res["versions"]["device"] = props.name
            res["versions"]["gcnArchName"] = getattr(props, "gcnArchName", None)
        except Exception as e:  # noqa: BLE001
            res["versions"]["device_error"] = repr(e)
        import bitsandbytes as bnb
        import peft
        res["versions"]["bitsandbytes"] = bnb.__version__
        res["versions"]["peft"] = peft.__version__
        try:
            import triton
            res["versions"]["triton"] = triton.__version__
        except Exception as e:  # noqa: BLE001
            res["versions"]["triton"] = f"unavailable: {e!r}"
        try:
            import transformers
            res["versions"]["transformers"] = transformers.__version__
        except Exception as e:  # noqa: BLE001
            res["versions"]["transformers"] = f"unavailable: {e!r}"

        import unsloth_zoo
        res["unsloth_zoo_file"] = unsloth_zoo.__file__
        from unsloth_zoo.temporary_patches import gpt_oss as go
        from unsloth_zoo.temporary_patches.gpt_oss import GptOssExpertsBnb4bit, torch_native_forward
        from unsloth_zoo.temporary_patches.moe_utils import _check_torch_grouped_mm_supported
        try:
            from unsloth_zoo.temporary_patches import gpt_oss_grouped_qlora as gq
        except ImportError:
            gq = None
        res["has_gq_module"] = gq is not None
        res["stage"] = "probe_grouped_mm"
        dump()  # survive a segfault inside the probe below
        if res["arm_env"] != "0":   # the forced-loop arm never reaches the runtime probe in zoo either
            res["grouped_mm_supported"] = bool(_check_torch_grouped_mm_supported())
        if gq is not None:
            res["stacked_dequant_available"] = bool(gq.stacked_dequant_available(torch.device("cuda", 0)))

        DT = torch.bfloat16
        E, TOP_K, H, I = 8, 4, 256, 192

        def _linear4bit(i, o, nested, seed):
            g = torch.Generator().manual_seed(seed)
            lin = bnb.nn.Linear4bit(i, o, bias = True, compute_dtype = DT, quant_type = "nf4",
                                    compress_statistics = nested, quant_storage = torch.uint8)
            scale = 0.02 * (1 + seed % 5)
            lin.weight = bnb.nn.Params4bit((torch.randn(o, i, generator = g) * scale).to(DT), requires_grad = False,
                                           quant_type = "nf4", compress_statistics = nested, blocksize = 64)
            lin.bias = torch.nn.Parameter((torch.randn(o, generator = g) * 0.1).to(DT), requires_grad = False)
            return lin.cuda()

        counters = {"grouped_attempt": 0, "grouped_returned": 0, "loop_gate_up_calls": 0}
        orig_grouped = GptOssExpertsBnb4bit._forward_grouped_bnb4bit

        def counted_grouped(self, *a, **k):
            counters["grouped_attempt"] += 1
            out = orig_grouped(self, *a, **k)
            if out is not None:
                counters["grouped_returned"] += 1
            return out

        class _Experts(torch.nn.Module):
            _grouped_bnb4bit_ready = GptOssExpertsBnb4bit._grouped_bnb4bit_ready
            _forward_grouped_bnb4bit = counted_grouped
            forward = torch_native_forward

            def __init__(self):
                super().__init__()
                self.gate_up_projs = torch.nn.ModuleList([_linear4bit(H, 2 * I, True, e) for e in range(E)])
                self.down_projs = torch.nn.ModuleList([_linear4bit(I, H, True, 100 + e) for e in range(E)])
                self.hidden_size, self.alpha, self.limit = H, 1.702, 7.0

        res["stage"] = "build"
        ex = _Experts()
        cfg = peft.LoraConfig(r = 16, lora_alpha = 32, lora_dropout = 0.0,
                              target_modules = r".*(gate_up_projs|down_projs)\.\d+")
        ex = peft.inject_adapter_in_model(cfg, ex)
        g = torch.Generator().manual_seed(7)
        for name, p in ex.named_parameters():
            if "lora_" in name:
                p.data = (torch.randn(p.shape, generator = g) * 0.05).to(p.device, p.dtype)
        ex.train()
        for m in ex.gate_up_projs:
            m.register_forward_hook(lambda *_: counters.__setitem__(
                "loop_gate_up_calls", counters["loop_gate_up_calls"] + 1))
        lora_params = [(n, p) for n, p in ex.named_parameters() if p.requires_grad]
        res["n_lora_params"] = len(lora_params)

        T = 96
        gx = torch.Generator().manual_seed(0)
        x = (torch.randn(1, T, H, generator = gx)).to("cuda", DT)
        logits = torch.randn(T, E, generator = gx)
        vals, idx = logits.topk(TOP_K, dim = -1)
        dense = torch.zeros(T, E).scatter_(1, idx, vals.softmax(-1))
        idx, w = idx.cuda(), dense.cuda().to(DT)
        target = torch.randn(1, T, H, generator = gx).cuda() * 0.1

        opt = torch.optim.SGD([p for _, p in lora_params], lr = 0.05)
        res["stage"] = "train"
        losses, gnorms, path = [], [], []
        calls0 = dict(gq.CALLS) if gq is not None else None
        for step in range(args.steps):
            before = dict(counters)
            opt.zero_grad(set_to_none = True)
            out = ex(x, idx, w)
            loss = (out.float() - target).square().mean()
            loss.backward()
            gn = torch.sqrt(sum((p.grad.float() ** 2).sum() for _, p in lora_params if p.grad is not None))
            if step == 0:
                torch.save({n: p.grad.detach().float().cpu() for n, p in lora_params if p.grad is not None},
                           args.grads)
            opt.step()
            torch.cuda.synchronize()
            losses.append(float(loss))
            gnorms.append(float(gn))
            path.append({k: counters[k] - before[k] for k in counters})
            res["losses"], res["grad_norms"], res["per_step_path"] = losses, gnorms, path
            dump()
        res["out_dtype"] = str(out.dtype)
        res["counters"] = counters
        if gq is not None:
            res["gq_calls_delta"] = {k: gq.CALLS[k] - calls0[k] for k in gq.CALLS}
            res["gq_disabled_reason"] = getattr(gq, "_DISABLED_REASON", None)
        res["engaged"] = "grouped" if counters["grouped_returned"] > 0 and counters["loop_gate_up_calls"] == 0 \
            else ("loop" if counters["grouped_returned"] == 0 and counters["loop_gate_up_calls"] > 0 else "mixed")
        res["stage"] = "done"
        res["ok"] = all(v == v and abs(v) != float("inf") for v in losses + gnorms)
    except Exception as e:  # noqa: BLE001
        res["error"] = f"{type(e).__name__}: {e}"
        res["traceback"] = traceback.format_exc()[-4000:]
    dump()
    return 0


if __name__ == "__main__":
    sys.exit(main())
