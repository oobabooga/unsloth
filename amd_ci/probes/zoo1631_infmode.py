#!/usr/bin/env python3
"""Inference-mode flex_attention_with_sink scenarios for unsloth-zoo PR 1631, runnable at base and head. Observes only.

Mirrors tests/test_flex_mask_reuse.py::test_inference_mode_mask_kept_apart without the head-only fixture, so the base
can run it too. --scenario:
  inf_only        one inference_mode call (training=False), fresh process
  grad_then_inf   a training call + backward (seq 256, window 128), then the same shape under inference_mode
  inf_then_grad   the PR test's order: inference_mode call, then a training call + backward
  shapes_then_inf training calls at seq 256 then 384 (create_block_mask recompiles dynamic, as after the PR test's
                  test_new_shape_builds_new_mask), then an inference_mode call at 256: the PR test-file order
JSON to --out: per call ok / error.
"""

from __future__ import annotations

import argparse
import json
import os
import traceback
import types
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--scenario", required = True, choices = ("inf_only", "grad_then_inf", "inf_then_grad", "shapes_then_inf"))
    args = ap.parse_args()
    res: dict = {"scenario": args.scenario, "env_reuse": os.environ.get("UNSLOTH_FLEX_MASK_REUSE"), "calls": []}

    def dump():
        args.out.write_text(json.dumps(res, indent = 2, default = str), encoding = "utf-8")

    try:
        import torch
        import unsloth_zoo
        from unsloth_zoo.flex_attention import utils as FU, attention_sink as AS
        res["unsloth_zoo_file"] = unsloth_zoo.__file__
        res["versions"] = {"torch": torch.__version__, "hip": getattr(torch.version, "hip", None)}
        res["has_flex"] = bool(FU.HAS_FLEX_ATTENTION)
        res["reuse_fn_present"] = getattr(FU, "reused_compiled_create_block_mask", None) is not None

        def attn(training):
            torch.manual_seed(0)
            return types.SimpleNamespace(sinks = torch.randn(4, device = "cuda", requires_grad = True),
                                         num_key_value_groups = 2, scaling = 0.125, sliding_window = 128,
                                         training = training)

        def qkv(seq = 256):
            g = torch.Generator(device = "cuda").manual_seed(1)
            return [torch.randn(2, h, seq, 64, device = "cuda", dtype = torch.bfloat16, generator = g) for h in (4, 2, 2)]

        def inf_call():
            a = attn(False)
            with torch.inference_mode():
                q, k, v = qkv()
                o = AS.flex_attention_with_sink(a, q, k, v)
            return float(o.float().abs().sum())

        def grad_call(seq = 256):
            a = attn(True)
            q, k, v = [x.requires_grad_() for x in qkv(seq)]
            o = AS.flex_attention_with_sink(a, q, k, v)
            o.float().square().sum().backward()
            torch.cuda.synchronize()
            return float(o.detach().float().abs().sum())

        seq = {"inf_only": [("inf", inf_call)], "grad_then_inf": [("grad", grad_call), ("inf", inf_call)],
               "inf_then_grad": [("inf", inf_call), ("grad", grad_call)],
               "shapes_then_inf": [("grad256", grad_call), ("grad384", lambda: grad_call(384)), ("inf", inf_call)]}[args.scenario]
        for name, fn in seq:
            rec = {"call": name}
            try:
                rec["out_abs_sum"] = repr(fn())
                rec["ok"] = True
            except Exception as e:  # noqa: BLE001
                rec["ok"] = False
                rec["error"] = f"{type(e).__name__}: {str(e)[:600]}"
                rec["traceback"] = traceback.format_exc()[-2500:]
            res["calls"].append(rec)
            dump()
        res["done"] = True
    except Exception as e:  # noqa: BLE001
        res["error"] = f"{type(e).__name__}: {str(e)[:1500]}"
        res["traceback"] = traceback.format_exc()[-3000:]
    dump()
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
