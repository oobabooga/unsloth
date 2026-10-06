#!/usr/bin/env python3
"""One training cell for unsloth-zoo PR 1588 (float16 grouped GEMM under torch.compile). Observes only.

Run in its own process per (dtype, repeat), PYTHONPATH = the state's unsloth-zoo checkout first.
Tiny Mixtral (hf-internal-testing/tiny-random-MixtralForCausalLM), Unsloth FastModel 16-bit, LoRA on attention
and experts, compiled path at Unsloth defaults (suppress_errors untouched), 3 AdamW steps on a fixed batch.
Records: per-step loss (repr, exact), LoRA weight digest after training, expert LoRA grad stats,
torch._dynamo counters (frames total/ok, unimplemented, graph_break), dynamo/inductor WARNING+ records
(failed-to-convert frames), and torch profiler counts of unsloth_zoo::grouped_mm_fp16 / aten::_grouped_mm.
Writes JSON to --out only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import traceback
from pathlib import Path

MODEL = "hf-internal-testing/tiny-random-MixtralForCausalLM"


class _Grab(logging.Handler):
    def __init__(self):
        super().__init__(level = logging.WARNING)
        self.records = []

    def emit(self, record):
        try:
            self.records.append(f"{record.name}: {record.getMessage()}"[:1500])
        except Exception:  # noqa: BLE001
            pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", required = True, choices = ["float16", "bfloat16"])
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--steps", type = int, default = 3)
    args = ap.parse_args()
    os.environ["UNSLOTH_COMPILE_LOCATION"] = str(args.out.parent / f"cache_{args.out.stem}")
    res: dict = {"dtype": args.dtype, "stage": "import", "model": MODEL}

    def dump():
        args.out.write_text(json.dumps(res, indent = 2, default = str), encoding = "utf-8")

    grab = _Grab()
    try:
        import unsloth  # noqa: F401  (before transformers)
        from unsloth import FastModel
        import torch
        import transformers
        import unsloth_zoo
        from unsloth_zoo.temporary_patches import moe_utils as MU
        for name in ("torch._dynamo", "torch._inductor", "torch.fx", "torch._functorch"):
            logging.getLogger(name).addHandler(grab)
        res["unsloth_zoo_file"] = unsloth_zoo.__file__
        res["unsloth_file"] = unsloth.__file__
        res["versions"] = {"torch": torch.__version__, "hip": getattr(torch.version, "hip", None),
                           "transformers": transformers.__version__}
        try:
            res["versions"]["gcnArchName"] = getattr(torch.cuda.get_device_properties(0), "gcnArchName", None)
        except Exception as e:  # noqa: BLE001
            res["versions"]["device_error"] = repr(e)
        res["has_fp16_op_attr"] = hasattr(MU, "_GROUPED_MM_FP16_OP")
        res["fp16_op_registered"] = getattr(MU, "_GROUPED_MM_FP16_OP", None) is not None
        res["moe_backend_env"] = os.environ.get("UNSLOTH_MOE_BACKEND", "<unset>")
        res["stage"] = "probe_grouped_mm"
        dump()
        res["grouped_mm_supported"] = bool(MU._check_torch_grouped_mm_supported())
        dtype = getattr(torch, args.dtype)

        res["stage"] = "load"
        dump()
        model, tok = FastModel.from_pretrained(MODEL, max_seq_length = 128, dtype = dtype, load_in_4bit = False,
                                               load_in_16bit = True, device_map = {"": 0})
        model = FastModel.get_peft_model(model, r = 8, lora_alpha = 16, lora_dropout = 0, bias = "none",
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            use_gradient_checkpointing = "unsloth", random_state = 3407)
        torch.manual_seed(1)
        for n, p in model.named_parameters():
            if "lora_B" in n:
                p.data.normal_(0, 0.02)
        lora = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        expert_lora = [n for n, _ in lora if ".experts." in n]
        res["n_lora_params"], res["n_expert_lora_params"] = len(lora), len(expert_lora)
        res["expert_lora_names_sample"] = expert_lora[:4]
        res["suppress_errors"] = bool(torch._dynamo.config.suppress_errors)
        try:
            res["resolved_moe_backend"] = str(MU.select_moe_backend()) if hasattr(MU, "select_moe_backend") else None
        except Exception as e:  # noqa: BLE001
            res["resolved_moe_backend"] = f"error {e!r}"

        vocab = int(model.config.vocab_size)
        g = torch.Generator().manual_seed(0)
        ids = torch.randint(4, max(5, vocab - 1), (2, 48), generator = g).cuda()
        opt = torch.optim.AdamW([p for _, p in lora], lr = 1e-3)
        from torch._dynamo.utils import counters
        counters.clear()
        model.train()
        res["stage"] = "train"
        dump()
        losses, loss_reprs, expert_grad_max = [], [], []
        from torch.profiler import ProfilerActivity, profile
        with profile(activities = [ProfilerActivity.CPU]) as prof:
            for step in range(args.steps):
                opt.zero_grad(set_to_none = True)
                with torch.autocast("cuda", dtype = dtype):
                    loss = model(input_ids = ids, labels = ids).loss
                loss.backward()
                eg = [float(p.grad.abs().max()) for n, p in lora if ".experts." in n and p.grad is not None]
                expert_grad_max.append(max(eg) if eg else None)
                opt.step()
                torch.cuda.synchronize()
                losses.append(float(loss))
                loss_reprs.append(repr(float(loss)))
                res["losses"], res["loss_reprs"], res["expert_grad_max"] = losses, loss_reprs, expert_grad_max
                dump()
        ops: dict = {}
        for ev in prof.events():
            nm = ev.name
            if "grouped_mm" in nm:
                ops[nm] = ops.get(nm, 0) + 1
        res["profiler_grouped_ops"] = ops
        res["fp16_op_calls"] = sum(v for k, v in ops.items() if "unsloth_zoo::grouped_mm_fp16" in k)
        res["aten_grouped_mm_calls"] = sum(v for k, v in ops.items() if k == "aten::_grouped_mm")
        h = hashlib.sha256()
        for n, p in sorted(lora):
            h.update(n.encode())
            h.update(p.detach().float().cpu().numpy().tobytes())
        res["lora_digest"] = h.hexdigest()[:16]
        res["dynamo_counters"] = {k: dict(v) for k, v in counters.items() if v}
        fr = counters.get("frames", {})
        res["frames_total"], res["frames_ok"] = int(fr.get("total", 0)), int(fr.get("ok", 0))
        res["stage"] = "done"
        res["ok"] = all(v == v and abs(v) != float("inf") for v in losses)
    except Exception as e:  # noqa: BLE001
        res["error"] = f"{type(e).__name__}: {str(e)[:1500]}"
        res["traceback"] = traceback.format_exc()[-4000:]
    recs = grab.records
    res["dynamo_warning_count"] = len(recs)
    res["dynamo_failures"] = [r for r in recs if any(s in r for s in (
        "WON'T CONVERT", "won't convert", "Backend compiler", "failed", "Error", "Traceback", "Unsupported"))][:30]
    res["n_dynamo_failures"] = len(res["dynamo_failures"])
    res["grouped_mm_failure_mentions"] = sum(1 for r in recs if "grouped_mm" in r)
    res["dynamo_warnings_sample"] = recs[:15]
    dump()
    return 0


if __name__ == "__main__":
    sys.exit(main())
