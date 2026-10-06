#!/usr/bin/env python3
"""One training cell for unsloth-zoo PR 1591 (gpt-oss grouped QLoRA on float16). Observes only.

Run in its own process per (dtype, repeat), PYTHONPATH = the state's unsloth-zoo checkout first.
Tiny gpt-oss (trl-internal-testing/tiny-GptOssForCausalLM, rewritten to the split-expert bnb-4bit layout by
zoo1591_make_ckpt.py) loaded through unsloth FastLanguageModel with
load_in_4bit=True (NF4 experts, GptOssExpertsBnb4bit), LoRA on attention + every expert gate_up / down Linear4bit,
3 AdamW steps on a fixed batch under autocast(dtype). Records per-step loss (repr, exact), LoRA digest, expert LoRA
grad stats, which expert path ran (grouped_qlora CALLS deltas, LAST_DECLINE, per-expert loop call count) and the
grouped-readiness verdict. Writes JSON to --out only.
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

MODEL = "trl-internal-testing/tiny-GptOssForCausalLM"


class _Grab(logging.Handler):
    def __init__(self):
        super().__init__(level = logging.INFO)
        self.records = []

    def emit(self, record):
        try:
            self.records.append(f"{record.name}: {record.getMessage()}"[:800])
        except Exception:  # noqa: BLE001
            pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", required = True, choices = ["float16", "bfloat16"])
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--steps", type = int, default = 3)
    ap.add_argument("--model", required = True, help = "split-expert checkpoint dir from zoo1591_make_ckpt.py")
    args = ap.parse_args()
    os.environ["UNSLOTH_COMPILE_LOCATION"] = str(args.out.parent / f"cache_{args.out.stem}")
    os.environ["UNSLOTH_ENABLE_LOGGING"] = "1"   # the head logs its decline reason at INFO
    res: dict = {"dtype": args.dtype, "stage": "import", "model": args.model, "source_model": MODEL}
    try:
        res["ckpt"] = json.loads((Path(args.model) / "zoo1591_ckpt.json").read_text(encoding = "utf-8"))
    except Exception as e:  # noqa: BLE001
        res["ckpt"] = repr(e)

    def dump():
        args.out.write_text(json.dumps(res, indent = 2, default = str), encoding = "utf-8")

    grab = _Grab()
    for _ln in ("unsloth_zoo.temporary_patches", "unsloth_zoo.temporary_patches.gpt_oss_grouped_qlora"):
        logging.getLogger(_ln).setLevel(logging.INFO)
    logging.getLogger("unsloth_zoo.temporary_patches").addHandler(grab)
    try:
        import unsloth  # noqa: F401  (before transformers)
        from unsloth import FastLanguageModel
        import torch
        import transformers
        import unsloth_zoo
        from unsloth_zoo.temporary_patches import gpt_oss_grouped_qlora as gq
        from unsloth_zoo.temporary_patches import gpt_oss as go
        res["unsloth_zoo_file"] = unsloth_zoo.__file__
        res["unsloth_file"] = unsloth.__file__
        res["versions"] = {"torch": torch.__version__, "hip": getattr(torch.version, "hip", None),
                           "cuda": getattr(torch.version, "cuda", None),
                           "transformers": transformers.__version__}
        for mod in ("bitsandbytes", "peft", "trl", "triton"):
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
        res["has_last_decline"] = hasattr(gq, "LAST_DECLINE")
        try:
            from unsloth_zoo.temporary_patches import moe_grouped_fp16 as mf
            res["has_moe_grouped_fp16"] = True
            res["fp16_grouped_available"] = bool(mf.fp16_grouped_available(torch.device("cuda", 0)))
            res["fp16_unavailable_reason"] = mf.unavailable_reason() if not res["fp16_grouped_available"] else None
        except ImportError:
            res["has_moe_grouped_fp16"] = False
        res["stage"] = "probe_grouped_mm"
        dump()
        from unsloth_zoo.temporary_patches.moe_utils import _check_torch_grouped_mm_supported
        res["grouped_mm_supported"] = bool(_check_torch_grouped_mm_supported())
        res["stacked_dequant_available"] = bool(gq.stacked_dequant_available(torch.device("cuda", 0)))
        dtype = getattr(torch, args.dtype)

        res["stage"] = "load"
        dump()
        model, tok = FastLanguageModel.from_pretrained(args.model, max_seq_length = 128, dtype = dtype,
                                                       load_in_4bit = True, device_map = {"": 0})
        experts = [m for m in model.modules() if hasattr(m, "gate_up_projs")]
        res["expert_module_class"] = sorted({f"{type(m).__module__}.{type(m).__name__}" for m in experts})
        res["n_expert_modules"] = len(experts)
        res["is_bnb4bit_experts"] = bool(experts) and all(isinstance(m, go.GptOssExpertsBnb4bit) for m in experts)
        if experts:
            b0 = experts[0].gate_up_projs[0]
            d0 = experts[0].down_projs[0]
            res["expert0_dtypes"] = {
                "gate_up": [str(getattr(b0, "compute_dtype", None)), str(getattr(b0, "_pre_set_compute_dtype", None)),
                            str(getattr(getattr(b0.weight, "quant_state", None), "dtype", None)), type(b0.weight).__name__],
                "down": [str(getattr(d0, "compute_dtype", None)), str(getattr(d0, "_pre_set_compute_dtype", None)),
                         str(getattr(getattr(d0.weight, "quant_state", None), "dtype", None)), type(d0.weight).__name__],
            }
        n_exp = len(experts[0].gate_up_projs) if experts else int(model.config.num_local_experts)
        res["stage"] = "peft"
        dump()
        model = FastLanguageModel.get_peft_model(model, r = 8, lora_alpha = 16, lora_dropout = 0, bias = "none",
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"] + [f"{w}.{i}" for w in ("gate_up_projs", "down_projs")
                                                                         for i in range(n_exp)],
            use_gradient_checkpointing = "unsloth", random_state = 3407)
        torch.manual_seed(1)
        for n, p in model.named_parameters():
            if "lora_B" in n:
                p.data.normal_(0, 0.02)
        lora = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        expert_lora = [n for n, _ in lora if ".experts." in n]
        res["n_lora_params"], res["n_expert_lora_params"] = len(lora), len(expert_lora)
        res["expert_lora_names_sample"] = expert_lora[:4]
        res["lora_param_dtypes"] = sorted({str(p.dtype) for _, p in lora})
        experts = [m for m in model.modules() if hasattr(m, "gate_up_projs")]
        loop_calls = {"n": 0}
        for m in experts:
            m.gate_up_projs[0].register_forward_hook(lambda *_: loop_calls.__setitem__("n", loop_calls["n"] + 1))
        if experts and hasattr(experts[0], "_grouped_bnb4bit_ready"):
            try:
                experts[0].train()
                r = experts[0]._grouped_bnb4bit_ready()
                res["grouped_ready"] = r if isinstance(r, (bool, str)) else repr(r)
            except Exception as e:  # noqa: BLE001
                res["grouped_ready"] = f"error {e!r}"

        vocab = int(model.config.vocab_size)
        g = torch.Generator().manual_seed(0)
        ids = torch.randint(4, max(5, vocab - 1), (2, 48), generator = g).cuda()
        opt = torch.optim.AdamW([p for _, p in lora], lr = 1e-3)
        model.train()
        calls0 = dict(gq.CALLS)
        res["stage"] = "train"
        dump()
        losses, loss_reprs, expert_grad_max, per_step = [], [], [], []
        dynamo_retries = []
        res["dynamo_reset_retries"] = dynamo_retries
        for step in range(args.steps):
            c_before, l_before = dict(gq.CALLS), loop_calls["n"]
            opt.zero_grad(set_to_none = True)
            try:
                with torch.autocast("cuda", dtype = dtype):
                    loss = model(input_ids = ids, labels = ids).loss
            except (StopIteration, RuntimeError) as e:
                # torch 2.11 dynamo (ROCm runner): StopIteration in dict_keys_getitem while re-entering the
                # compiled GptOssTopKRouter_forward at step 3, base and head alike; a fresh trace after
                # torch._dynamo.reset() then dies in unsloth_zoo.utils.Version ("Could not get version").
                # The router is untouched by the PR and the expert forward under test is compiler-disabled,
                # so the step is retried once with dynamo off (eager) for the rest of the cell; recorded.
                if not isinstance(e, StopIteration) and "Could not get version" not in str(e):
                    raise
                dynamo_retries.append({"step": step, "error": f"{type(e).__name__}: {str(e)[:300]}",
                                       "where": traceback.format_exc().strip().splitlines()[-3:]})
                res["dynamo_reset_retries"] = dynamo_retries
                torch._dynamo.reset()
                torch._dynamo.config.disable = True
                gq.CALLS.update(c_before)
                loop_calls["n"] = l_before
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
            d = {k: gq.CALLS[k] - c_before.get(k, 0) for k in gq.CALLS}
            d["loop_gate_up0_calls"] = loop_calls["n"] - l_before
            per_step.append(d)
            res.update(losses = losses, loss_reprs = loss_reprs, expert_grad_max = expert_grad_max, per_step_path = per_step)
            dump()
        res["calls_delta"] = {k: gq.CALLS[k] - calls0.get(k, 0) for k in gq.CALLS}
        res["calls_final"] = dict(gq.CALLS)
        res["last_decline"] = dict(gq.LAST_DECLINE) if hasattr(gq, "LAST_DECLINE") else "<absent at this state>"
        res["disabled_reason"] = getattr(gq, "_DISABLED_REASON", None)
        res["loop_gate_up0_calls"] = loop_calls["n"]
        cd = res["calls_delta"]
        grouped = int(cd.get("forward", 0)) + int(cd.get("forward_fp16", 0))
        res["engaged"] = ("grouped" if grouped > 0 and loop_calls["n"] == 0 else
                          "loop" if grouped == 0 and loop_calls["n"] > 0 else "mixed" if grouped else "none")
        h = hashlib.sha256()
        for n, p in sorted(lora):
            h.update(n.encode())
            h.update(p.detach().float().cpu().numpy().tobytes())
        res["lora_digest"] = h.hexdigest()[:16]
        res["stage"] = "done"
        res["ok"] = all(v == v and abs(v) != float("inf") for v in losses)
    except Exception as e:  # noqa: BLE001
        res["error"] = f"{type(e).__name__}: {str(e)[:1500]}"
        res["traceback"] = traceback.format_exc()[-4000:]
    res["grouped_log"] = [r for r in grab.records if "grouped" in r or "per-expert" in r][:20]
    dump()
    return 0


if __name__ == "__main__":
    sys.exit(main())
