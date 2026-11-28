#!/usr/bin/env python3
"""AMD gate batch c extra cells (observe only, JSON via --out; one cell per process).

  nccl                        unsloth #19347: is_nccl_available / version.cuda, read_layout() single GPU,
                              ParallelLayout.check_backend on a duck layout, read_layout() with TP=2 WORLD_SIZE=2.
  cg_<model>_<arm>            unsloth #19339: Qwen3 LoRA 2x1024 manual loop (block compile path), block CUDA graph
                              state + per-step wall time.
     model  q06 = unsloth/Qwen3-0.6B bf16 LoRA r16;  q4b = unsloth/Qwen3-4B-unsloth-bnb-4bit QLoRA r16
     arm    auto     default env (what users get)
            unguard  auto, with _auto_platform_reason patched to None (the ROCm / torch<2.13 guard lifted):
                     the state's own ratio threshold decides; records the measured ratio
            off      as unguard with UNSLOTH_BLOCK_CUDAGRAPHS_AUTO_RATIO=1e9: ratio measured, graphs stay off
            on       UNSLOTH_BLOCK_CUDAGRAPHS=1 (forced on)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
import traceback

ap = argparse.ArgumentParser()
ap.add_argument("--x", required = True)
ap.add_argument("--out", required = True)
args = ap.parse_args()
res: dict = {"x": args.x, "stage": "start"}


def dump():
    with open(args.out, "w", encoding = "utf-8") as fh:
        json.dump(res, fh, indent = 1, default = str)


def stage(s):
    res["stage"] = s
    dump()


def scalars(d):
    out = {}
    for k, v in (d or {}).items():
        if isinstance(v, (int, float, str, bool, type(None))):
            out[k] = v
        elif isinstance(v, (list, tuple)) and len(v) <= 16 and all(isinstance(x, (int, float, str, bool, type(None))) for x in v):
            out[k] = list(v)
        elif isinstance(v, dict):
            out[k] = scalars(v) if len(v) <= 40 else f"<dict {len(v)}>"
        else:
            out[k] = f"<{type(v).__name__}>"
    return out


def nccl():
    import torch
    import torch.distributed as dist
    res["torch"] = torch.__version__
    res["hip"] = torch.version.hip
    res["cuda"] = torch.version.cuda
    res["cuda_available"] = torch.cuda.is_available()
    res["dist_available"] = dist.is_available()
    res["nccl_available"] = dist.is_nccl_available() if dist.is_available() else None
    stage("torch")
    import unsloth  # noqa: F401
    from unsloth import parallel_env as pe
    res["unsloth_file"] = unsloth.__file__
    res["has_check_backend"] = hasattr(getattr(pe, "ParallelLayout", object), "check_backend")
    for k in ("UNSLOTH_TP_SIZE", "UNSLOTH_PP_SIZE", "UNSLOTH_CP_SIZE", "WORLD_SIZE"):
        os.environ.pop(k, None)
    try:
        res["read_layout_single"] = repr(pe.read_layout())
    except Exception as e:  # noqa: BLE001
        res["read_layout_single"] = f"RAISED {type(e).__name__}: {str(e)[:400]}"
    if res["has_check_backend"]:
        class Duck:
            env_vars = "UNSLOTH_TP_SIZE=2"

            def __str__(self):
                return "tp=2"
        try:
            pe.ParallelLayout.check_backend(Duck())
            res["check_backend_duck"] = "ok"
        except Exception as e:  # noqa: BLE001
            res["check_backend_duck"] = f"RAISED {type(e).__name__}: {str(e)[:400]}"
    os.environ.update(UNSLOTH_TP_SIZE = "2", WORLD_SIZE = "2")
    try:
        res["read_layout_tp2_world2"] = repr(pe.read_layout())
    except Exception as e:  # noqa: BLE001
        res["read_layout_tp2_world2"] = f"RAISED {type(e).__name__}: {str(e)[:400]}"
    stage("done")


MODELS = {"q06": ("unsloth/Qwen3-0.6B", False, 30), "q4b": ("unsloth/Qwen3-4B-unsloth-bnb-4bit", True, 18)}


def cg(model_key, arm):
    name, four, steps = MODELS[model_key]
    if arm == "on":
        os.environ["UNSLOTH_BLOCK_CUDAGRAPHS"] = "1"
    else:
        os.environ["UNSLOTH_BLOCK_CUDAGRAPHS"] = "auto"
    if arm == "off":
        os.environ["UNSLOTH_BLOCK_CUDAGRAPHS_AUTO_RATIO"] = "1e9"
    else:
        os.environ.pop("UNSLOTH_BLOCK_CUDAGRAPHS_AUTO_RATIO", None)
    os.environ["UNSLOTH_ENABLE_LOGGING"] = "1"
    res.update(model = name, load_in_4bit = four, arm = arm, steps = steps, batch = "2x1024")
    import unsloth  # noqa: F401
    from unsloth import FastLanguageModel
    import torch
    from unsloth.models import _block_compile as bc
    res["unsloth_file"] = unsloth.__file__
    res["torch"], res["hip"] = torch.__version__, torch.version.hip
    p = torch.cuda.get_device_properties(0)
    res["device"], res["arch"] = p.name, getattr(p, "gcnArchName", None)
    res["capability"] = list(torch.cuda.get_device_capability(0))
    res["cudagraph_mode"] = getattr(bc, "_CUDAGRAPH_MODE", None)
    res["platform_reason_native"] = bc._auto_platform_reason() if hasattr(bc, "_auto_platform_reason") else "absent"
    res["threshold"] = bc._auto_ratio(0) if hasattr(bc, "_auto_ratio") else getattr(bc, "_AUTO_RATIO", None)
    res["threshold_fn"] = "_auto_ratio" if hasattr(bc, "_auto_ratio") else "_AUTO_RATIO"
    if hasattr(bc, "_auto_ratio"):  # the per-device default, without the env override the "off" arm sets
        saved, bc._AUTO_RATIO = bc._AUTO_RATIO, None
        res["native_threshold"] = bc._auto_ratio(0)
        bc._AUTO_RATIO = saved
    else:
        res["native_threshold"] = 1.2 if arm == "off" else bc._AUTO_RATIO
    if arm in ("unguard", "off"):
        bc._auto_platform_reason = lambda: None
        res["guard_lifted"] = True
    stage("imported")
    model, tok = FastLanguageModel.from_pretrained(name, max_seq_length = 1024, load_in_4bit = four,
                                                   dtype = torch.bfloat16)
    model = FastLanguageModel.get_peft_model(model, r = 16, lora_alpha = 16, lora_dropout = 0,
                                             target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj",
                                                               "up_proj", "down_proj"],
                                             use_gradient_checkpointing = "unsloth", random_state = 0)
    stage("loaded")
    g = torch.Generator().manual_seed(0)
    ids = torch.randint(100, 20000, (2, 1024), generator = g).cuda()
    batch = {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": ids.clone()}
    params = [q for q in model.parameters() if q.requires_grad]
    opt = torch.optim.AdamW(params, lr = 1e-4)
    model.train()
    losses, ms = [], []
    for s in range(steps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model(**batch)
        out.loss.backward()
        opt.step()
        opt.zero_grad(set_to_none = True)
        torch.cuda.synchronize()
        ms.append((time.perf_counter() - t0) * 1e3)
        losses.append(float(out.loss.detach()))
        if s in (0, 4, steps - 1):
            res["auto_state"] = scalars(bc._CUDAGRAPH_STATE.get("auto"))
            res["cg_state"] = scalars({k: v for k, v in bc._CUDAGRAPH_STATE.items() if k != "auto"})
        res["ms"], res["losses"] = [round(x, 2) for x in ms], losses
        dump()
    tailms = ms[8:]
    res["ms_median_tail"] = round(statistics.median(tailms), 2)
    res["ms_min_tail"] = round(min(tailms), 2)
    res["finite"] = all(math.isfinite(x) for x in losses)
    res["peak_alloc_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
    try:
        from torch._dynamo.utils import counters
        res["dynamo"] = {"unique_graphs": dict(counters.get("stats", {})).get("unique_graphs", 0),
                         "graph_breaks": int(sum(counters.get("graph_break", {}).values())),
                         "inductor_cudagraph": {str(k): v for k, v in dict(counters.get("inductor", {})).items()
                                                if "cudagraph" in str(k)}}
    except Exception as e:  # noqa: BLE001
        res["dynamo"] = repr(e)[:200]
    stage("done")


try:
    if args.x == "nccl":
        nccl()
    elif args.x.startswith("cg_"):
        _, m, a = args.x.split("_")
        cg(m, a)
    else:
        raise SystemExit(f"unknown cell {args.x}")
except BaseException as e:  # noqa: BLE001
    res["error"] = f"{type(e).__name__}: {str(e)[:600]}"
    res["traceback"] = traceback.format_exc()[-3000:]
    dump()
    raise
