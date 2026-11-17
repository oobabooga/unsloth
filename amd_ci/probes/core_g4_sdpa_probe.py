#!/usr/bin/env python3
"""Probe: Gemma 4 attention through plain ``import unsloth`` (no Studio code), per state. Observes only (judged by
criteria/core_g4_sdpa_win.py and criteria/core_g4_sdpa_lnx.py).

Each task runs in its own process with PYTHONPATH = the state's checkout, so ``import unsloth`` is that tree's (recorded
and checked), and a native crash in a broken kernel is an exit code rather than a lost probe:

  fwd             import unsloth, then transformers' Gemma 4 forward at 2048 tokens with attn_implementation "eager"
                  and "sdpa": top-1 agreement, KL, next-token loss; plus a profiled sdpa forward recording which aten
                  attention op ran at which head dim (engagement of the guard: math vs fused at head 512).
  train_sdpa      FastLanguageModel (falls back to FastModel) bf16 LoRA, 20 steps of a fixed 1024-token stream,
  train_eager     plain AdamW loop, loss per step; the same with "eager"; and with no attn_implementation (Unsloth's
  train_default   own choice, recorded).

torch.compile stays off (TORCHDYNAMO_DISABLE=1): the Windows boxes have no MSVC. A train task that fails is retried once
with UNSLOTH_COMPILE_DISABLE=1 as well, and the retry is recorded. JSON to --out only.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

MODEL = "unsloth/gemma-4-E2B-it"
FLAG = "_unsloth_windows_rocm_wide_head_guard"
TRAIN_STEPS = 20
TRAIN_SEQ = 1024
FWD_SEQ = 2048
SDPA_OPS = (
    "aten::_scaled_dot_product_flash_attention",
    "aten::_scaled_dot_product_efficient_attention",
    "aten::_scaled_dot_product_attention_math",
    "aten::_scaled_dot_product_cudnn_attention",
    "aten::_scaled_dot_product_flash_attention_for_cpu",
)

PARAGRAPH = (
    "The lighthouse keeper climbed the spiral stairs every evening at dusk, counting one hundred and twelve steps "
    "before he reached the lamp room. He wiped the brass fittings, trimmed the wick, and wrote the weather in a "
    "leather notebook: wind from the north west, a falling barometer, fog banks moving in from the outer islands. "
    "Ships passed on the horizon with their lights blinking red and green, and once a year a supply boat brought "
    "flour, lamp oil, letters from his sister, and a crate of oranges that he rationed carefully until spring. "
    "In the long winter nights he taught himself astronomy from a borrowed atlas, learning the names of the "
    "constellations and the dates of eclipses, and he kept a second notebook of every star he could identify. "
)


def _write(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent = 2, default = str), encoding = "utf-8")


def _env_facts(res: dict, checkout: str) -> None:
    import torch
    import torch.nn.functional as F

    import unsloth

    res["unsloth_file"] = getattr(unsloth, "__file__", None)
    res["unsloth_from_checkout"] = bool(res["unsloth_file"]) and os.path.normcase(
        os.path.abspath(res["unsloth_file"])
    ).startswith(os.path.normcase(os.path.abspath(checkout)))
    fn, guarded = F.scaled_dot_product_attention, False
    for _ in range(32):
        if fn is None:
            break
        if getattr(fn, FLAG, False):
            guarded = True
            break
        fn = getattr(fn, "__wrapped__", None)
    res["guard_installed"] = guarded
    res["torch"] = torch.__version__
    res["hip"] = getattr(torch.version, "hip", None)
    res["platform"] = sys.platform
    res["aotriton_experimental"] = os.environ.get("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL")
    if torch.cuda.is_available():
        res["arch"] = getattr(torch.cuda.get_device_properties(0), "gcnArchName", None)
    try:
        import transformers

        res["transformers"] = transformers.__version__
    except Exception:  # noqa: BLE001
        pass


def _load_hf(transformers, torch, impl):
    for cls in ("AutoModelForCausalLM", "AutoModelForImageTextToText"):
        try:
            model = getattr(transformers, cls).from_pretrained(
                MODEL, dtype = torch.bfloat16, attn_implementation = impl, token = False
            )
            return model, cls
        except Exception:  # noqa: BLE001
            continue
    raise RuntimeError(f"no transformers class loaded {MODEL} with {impl}")


def _op_counts(prof) -> dict:
    counts: dict = {}
    for ev in prof.events():
        if ev.name in SDPA_OPS:
            shapes = getattr(ev, "input_shapes", None) or []
            head = shapes[0][-1] if shapes and shapes[0] else None
            key = f"{ev.name.split('::')[-1]}@{head}"
            counts[key] = counts.get(key, 0) + 1
    return counts


def task_fwd(res: dict) -> None:
    import torch
    import transformers

    tok = transformers.AutoTokenizer.from_pretrained(MODEL, token = False)
    dev = "cuda" if torch.cuda.is_available() else "cpu"  # cpu: harness smoke only
    ids = tok(PARAGRAPH * 40, return_tensors = "pt").input_ids[:, :FWD_SEQ].to(dev)
    out = {"tokens": int(ids.shape[1])}
    logits = {}
    for impl in ("eager", "sdpa"):
        t0 = time.time()
        model, cls = _load_hf(transformers, torch, impl)
        out["class"] = type(model).__name__
        model = model.to(dev).eval()
        out[f"impl_{impl}"] = getattr(model.config, "_attn_implementation", None)
        with torch.no_grad():
            logits[impl] = model(input_ids = ids).logits[0].float().cpu()
            if impl == "sdpa":
                # Which attention op ran, by head dim: the guard's engagement count, not inferred from the outcome.
                from torch.profiler import ProfilerActivity, profile

                with profile(activities = [ProfilerActivity.CPU], record_shapes = True) as prof:
                    model(input_ids = ids[:, :256])
                out["sdpa_ops"] = _op_counts(prof)
        out[f"{impl}_s"] = round(time.time() - t0, 1)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    e, s = logits["eager"], logits["sdpa"]
    lp_e, lp_s = e.log_softmax(-1), s.log_softmax(-1)
    out["finite"] = bool(torch.isfinite(s).all() and torch.isfinite(e).all())
    out["rel_l2"] = round(float((s - e).norm() / e.norm()), 6)
    out["top1_agree"] = round(float((e.argmax(-1) == s.argmax(-1)).float().mean()), 5)
    kl = (lp_e.exp() * (lp_e - lp_s)).sum(-1)
    out["kl_eager_sdpa_mean"] = round(float(kl.mean()), 6)
    out["kl_eager_sdpa_max"] = round(float(kl.max()), 6)
    tgt = ids[0, 1:].cpu()
    out["nll_eager"] = round(float(torch.nn.functional.cross_entropy(e[:-1], tgt)), 5)
    out["nll_sdpa"] = round(float(torch.nn.functional.cross_entropy(s[:-1], tgt)), 5)
    res["fwd"] = out


def task_train(res: dict, impl: str | None) -> None:
    import torch

    import unsloth  # noqa: F401
    from unsloth import FastLanguageModel, FastModel

    kwargs = dict(
        model_name = MODEL,
        max_seq_length = TRAIN_SEQ,
        dtype = torch.bfloat16,
        load_in_4bit = False,
        load_in_8bit = False,
        full_finetuning = False,
        token = None,
    )
    if impl is not None:
        kwargs["attn_implementation"] = impl
    out: dict = {"requested": impl}
    loader = FastLanguageModel
    try:
        model, tok = FastLanguageModel.from_pretrained(**kwargs)
    except Exception as exc:  # noqa: BLE001
        out["fastlanguagemodel_error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
        loader = FastModel
        model, tok = FastModel.from_pretrained(**kwargs)
    out["loader"] = loader.__name__
    cfgs = [model.config] + [getattr(model.config, n, None) for n in ("text_config",)]
    out["attn_impl"] = [getattr(c, "_attn_implementation", None) for c in cfgs if c is not None]
    peft_kwargs = dict(
        r = 8,
        lora_alpha = 16,
        lora_dropout = 0,
        bias = "none",
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
    )
    if loader is FastModel:
        peft_kwargs.update(finetune_vision_layers = False, finetune_language_layers = True,
                           finetune_attention_modules = True, finetune_mlp_modules = True)
    torch.manual_seed(3407)
    model = loader.get_peft_model(model, **peft_kwargs)
    text_tok = getattr(tok, "tokenizer", tok)
    stream = text_tok(PARAGRAPH * 60, return_tensors = "pt", add_special_tokens = False).input_ids[0]
    device = next(p for p in model.parameters() if p.requires_grad).device
    params = [p for p in model.parameters() if p.requires_grad]
    out["trainable"] = int(sum(p.numel() for p in params))
    opt = torch.optim.AdamW(params, lr = 2e-4, weight_decay = 0.0)
    model.train()
    losses, extra = [], {}
    t0 = time.time()
    for step in range(TRAIN_STEPS):
        start = (step * 97) % (stream.numel() - TRAIN_SEQ)
        ids = stream[start : start + TRAIN_SEQ][None].to(device)
        batch = {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": ids}
        try:
            loss = model(**batch, **extra).loss
        except Exception as exc:  # noqa: BLE001
            if "token_type_ids" not in str(exc) or extra:
                raise
            extra = {"token_type_ids": torch.zeros_like(ids)}
            out["needed_token_type_ids"] = True
            loss = model(**batch, **extra).loss
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none = True)
        losses.append(round(float(loss.detach()), 5))
        print(f"step {step + 1} loss {losses[-1]}", flush = True)
    out["losses"] = losses
    out["train_s"] = round(time.time() - t0, 1)
    out["finite"] = all(x == x and abs(x) != float("inf") for x in losses)
    res["train"] = out


def child(args) -> int:
    res: dict = {"task": args.child, "compile_disabled": os.environ.get("UNSLOTH_COMPILE_DISABLE")}
    try:
        _env_facts(res, args.checkout)
        _write(args.out, res)
        if args.child == "fwd":
            task_fwd(res)
        else:
            task_train(res, {"train_sdpa": "sdpa", "train_eager": "eager", "train_default": None}[args.child])
    except Exception as exc:  # noqa: BLE001
        res["error"] = f"{type(exc).__name__}: {str(exc)[:600]}"
        res["traceback"] = traceback.format_exc()[-4000:]
    _write(args.out, res)
    return 0 if "error" not in res else 3


def run_task(args, task: str, env: dict) -> dict:
    out = args.out.with_name(f"{args.out.stem}_{task}.json")
    log = args.out.with_name(f"{args.out.stem}_{task}.log")
    cmd = [sys.executable, __file__, "--state", args.state, "--checkout", args.checkout, "--out", str(out),
           "--child", task, "--model", args.model]
    t0 = time.time()
    with open(log, "ab") as fh:
        rc = subprocess.run(cmd, stdout = fh, stderr = subprocess.STDOUT, env = env, cwd = args.workdir).returncode
    try:
        got = json.loads(out.read_text(encoding = "utf-8"))
    except Exception:  # noqa: BLE001
        got = {"error": f"task process exited {rc} with no result"}
    got["_rc"] = rc
    got["_s"] = round(time.time() - t0, 1)
    return got


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--tasks", default = "fwd,train_sdpa,train_eager,train_default")
    ap.add_argument("--child", default = None, help = argparse.SUPPRESS)
    ap.add_argument("--model", default = MODEL, help = "harness smoke tests only: a tiny Gemma 4 copy")
    args = ap.parse_args()
    globals()["MODEL"] = args.model
    args.checkout = str(Path(args.checkout).resolve())
    args.out = args.out.resolve()
    args.workdir = str(args.out.parent)
    if args.child:
        return child(args)

    env = dict(os.environ)
    env["PYTHONPATH"] = args.checkout + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.setdefault("TORCHDYNAMO_DISABLE", "1")
    env.setdefault("UNSLOTH_DISABLE_AUTO_UPDATES", "1")
    env.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env["UNSLOTH_COMPILE_LOCATION"] = str(Path(args.workdir) / f"unsloth_compiled_cache_{args.state}")
    for k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        env.pop(k, None)
    res: dict = {"state": args.state, "checkout": args.checkout, "tasks": {}}
    _write(args.out, res)
    for task in [t for t in args.tasks.split(",") if t]:
        got = run_task(args, task, env)
        if task.startswith("train") and got.get("error"):
            retry_env = dict(env, UNSLOTH_COMPILE_DISABLE = "1")
            first = got
            got = run_task(args, task, retry_env)
            got["_first_attempt_error"] = first.get("error")
            got["_retried_with_compile_disable"] = True
        res["tasks"][task] = got
        for k in ("unsloth_file", "unsloth_from_checkout", "guard_installed", "torch", "hip", "platform", "arch",
                  "transformers", "aotriton_experimental"):
            if k in got and k not in res:
                res[k] = got[k]
        _write(args.out, res)
    return 0


if __name__ == "__main__":
    sys.exit(main())
