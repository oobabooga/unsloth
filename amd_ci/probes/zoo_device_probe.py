#!/usr/bin/env python3
"""Probe: what does an unsloth-zoo checkout decide about THIS host's device, and
does a short real LoRA run through it still train?

Written for unsloth-zoo changes that add a backend branch (NPU, XPU, ...) to code
every other backend runs too, where the question on AMD is "is this a strict no-op
on HIP". Observes only; criteria/zoo_device_no_regression.py judges.

The checkout under test is an unsloth-zoo SOURCE tree. It is put first on
PYTHONPATH of a FRESH child interpreter per leg, so the state's `unsloth_zoo`
shadows whatever zoo the environment has installed, with torch, transformers,
TRL and unsloth held identical across states. The child records
`unsloth_zoo.__file__` so the criteria can refuse a run where the overlay did not
take (both states silently importing the installed zoo would compare it to itself).

Legs, each its own subprocess:

  device  import unsloth_zoo; DEVICE_TYPE, DEVICE_TYPE_TORCH, DEVICE_COUNT,
          device_is_bf16_supported(), is_hip(), npu_is_available() if present;
          the module-level dispatch choices the change touches
          (gradient_checkpointing.torch_gpu_stream, _amp_device_type,
          loss_utils.current_device); initialize_unsloth_gradient_checkpointing()
          on the real GPU with buffer count / shape / dtype / device and stream and
          event types; reset_unsloth_gradient_checkpointing_buffers() and the same
          summary after; device_synchronize() / device_empty_cache(); and
          vllm_utils.get_mem_info() / _device_empty_cache() when vllm_utils imports
          (its import error is recorded, never raised: vLLM is usually absent).

  train   (--train) unsloth (whatever the environment installed) + the state's zoo:
          16-bit LoRA on --model, use_gradient_checkpointing="unsloth", --steps SFT
          steps via trainer.train(), losses from log_history; then the same number
          of steps through unsloth_zoo.training_utils.unsloth_train (the loop whose
          device the change now takes from a variable), losses parsed from its
          "step, loss" lines.

Writes JSON to --out, never stdout (import banners corrupt stdout).
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path


def _err(e: BaseException) -> str:
    return f"{type(e).__name__}: {e}"[:2000]


def _qual(obj) -> str | None:
    if obj is None:
        return None
    mod = getattr(obj, "__module__", None) or type(obj).__module__
    name = getattr(obj, "__qualname__", None) or type(obj).__qualname__
    return f"{mod}.{name}"


def _tensor(t) -> dict | None:
    if t is None:
        return None
    try:
        return {"shape": list(t.shape), "numel": int(t.numel()), "dtype": str(t.dtype),
                "device": str(t.device), "pinned": bool(t.is_pinned()) if t.device.type == "cpu" else None}
    except Exception as e:  # noqa: BLE001
        return {"error": _err(e)}


def _zoo_file_check(checkout: str) -> dict:
    import unsloth_zoo
    f = str(Path(unsloth_zoo.__file__).resolve())
    return {"file": f, "from_checkout": f.startswith(str(Path(checkout).resolve()) + os.sep)}


def _gc_summary(gcm) -> dict:
    out: dict = {}
    for name in ("GPU_BUFFERS", "GPU_BUFFERS_B"):
        bufs = getattr(gcm, name, None)
        out[name] = None if bufs is None else [_tensor(b) for b in bufs]
    cpu = getattr(gcm, "CPU_BUFFERS", None) or []
    out["CPU_BUFFERS"] = {"count": len(cpu), "first": _tensor(cpu[0]) if cpu else None}
    for name in ("BUFFER_EVENTS_A", "BUFFER_EVENTS_B", "EXTRA_STREAMS", "MAIN_STREAMS"):
        v = getattr(gcm, name, None)
        out[name] = None if v is None else {"count": len(v), "types": sorted({_qual(x) for x in v}),
                                            "devices": [str(getattr(x, "device", "?")) for x in v]}
    for name in ("USE_DOUBLE_BUFFER", "USE_UNSLOTH_GC", "BACKWARD_PASS", "MINIMUM_SIZE",
                 "NEXT_BUFFER_SLOT", "CPU_INDEX", "FIRST_PASS", "LAST_GC_INDEX"):
        v = getattr(gcm, name, "absent")
        out[name] = v if isinstance(v, (bool, int, float, str, list, type(None))) else repr(v)
    return out


def child_device(checkout: str) -> dict:
    import torch
    obs: dict = {"torch": {"version": torch.__version__, "hip": getattr(torch.version, "hip", None),
                           "cuda": getattr(torch.version, "cuda", None),
                           "cuda_available": torch.cuda.is_available(),
                           "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0}}
    if torch.cuda.is_available():
        with contextlib.suppress(Exception):
            obs["torch"]["device_name"] = torch.cuda.get_device_name(0)
        with contextlib.suppress(Exception):
            obs["torch"]["gcn_arch"] = getattr(torch.cuda.get_device_properties(0), "gcnArchName", None)
    with contextlib.redirect_stdout(io.StringIO()):
        import unsloth_zoo  # noqa: F401
    obs["zoo"] = _zoo_file_check(checkout)
    from unsloth_zoo import device_type as dt
    obs["DEVICE_TYPE"] = dt.DEVICE_TYPE
    obs["DEVICE_TYPE_TORCH"] = dt.DEVICE_TYPE_TORCH
    obs["DEVICE_COUNT"] = dt.DEVICE_COUNT
    for fn in ("device_is_bf16_supported", "is_hip", "npu_is_available"):
        f = getattr(dt, fn, None)
        if f is None:
            obs[fn] = "absent"
            continue
        try:
            obs[fn] = f()
        except Exception as e:  # noqa: BLE001
            obs[fn] = {"error": _err(e)}
    for fn in ("device_synchronize", "device_empty_cache"):
        f = getattr(dt, fn, None)
        try:
            obs[fn] = "absent" if f is None else (f(), "ok")[1]
        except Exception as e:  # noqa: BLE001
            obs[fn] = {"error": _err(e)}

    from unsloth_zoo import gradient_checkpointing as gcm
    from unsloth_zoo import loss_utils
    obs["dispatch"] = {
        "gradient_checkpointing.torch_gpu_stream": _qual(getattr(gcm, "torch_gpu_stream", None)),
        "gradient_checkpointing._amp_device_type": getattr(gcm, "_amp_device_type", "absent"),
        "loss_utils.current_device": _qual(getattr(loss_utils, "current_device", None)),
    }
    gc_obs: dict = {}
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            gcm.initialize_unsloth_gradient_checkpointing()
        gc_obs["init"] = _gc_summary(gcm)
    except Exception as e:  # noqa: BLE001
        gc_obs["init_error"] = _err(e)
        gc_obs["init_traceback"] = traceback.format_exc()[-2000:]
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            gcm.reset_unsloth_gradient_checkpointing_buffers()
        gc_obs["after_reset"] = _gc_summary(gcm)
    except Exception as e:  # noqa: BLE001
        gc_obs["reset_error"] = _err(e)
        gc_obs["reset_traceback"] = traceback.format_exc()[-2000:]
    obs["gc"] = gc_obs

    vl: dict = {}
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            from unsloth_zoo import vllm_utils
        vl["import"] = "ok"
        try:
            free, total = vllm_utils.get_mem_info()
            vl["get_mem_info"] = {"ok": True, "total": int(total), "free_positive": int(free) > 0}
        except Exception as e:  # noqa: BLE001
            vl["get_mem_info"] = {"ok": False, "error": _err(e)}
        f = getattr(vllm_utils, "_device_empty_cache", None)
        try:
            vl["_device_empty_cache"] = "absent" if f is None else (f(), "ok")[1]
        except Exception as e:  # noqa: BLE001
            vl["_device_empty_cache"] = {"error": _err(e)}
    except BaseException as e:  # noqa: BLE001
        vl["import"] = _err(e)
    obs["vllm_utils"] = vl
    return obs


_STEP_LOSS = re.compile(r"^\s*(\d+),\s*([-+0-9.eEnaNinf]+)\s*$")


def child_train(checkout: str, model_name: str, steps: int, seed: int, work: str) -> dict:
    obs: dict = {"model": model_name, "steps": steps, "seed": seed}
    t0 = time.time()
    import unsloth  # noqa: F401  (must precede transformers / trl)
    from unsloth import FastLanguageModel, is_bf16_supported
    obs["unsloth_file"] = unsloth.__file__
    obs["unsloth_version"] = getattr(unsloth, "__version__", None)
    obs["zoo"] = _zoo_file_check(checkout)
    import torch
    import transformers
    import trl
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer
    obs["versions"] = {"torch": torch.__version__, "transformers": transformers.__version__,
                       "trl": trl.__version__}
    from unsloth_zoo.device_type import DEVICE_TYPE
    obs["DEVICE_TYPE"] = DEVICE_TYPE

    model, tok = FastLanguageModel.from_pretrained(
        model_name = model_name, max_seq_length = 256, dtype = None,
        load_in_4bit = False, load_in_8bit = False, full_finetuning = False)
    model = FastLanguageModel.get_peft_model(
        model, r = 8, lora_alpha = 16, lora_dropout = 0, bias = "none",
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing = "unsloth", random_state = seed)
    obs["model_dtype"] = str(next(model.parameters()).dtype)
    rows = [f"Question {i}: what is {i} plus {i + 3}? Answer: {2 * i + 3}. "
            f"The quick brown fox jumps over the lazy dog number {i}." for i in range(64)]
    ds = Dataset.from_dict({"text": rows})
    bf16 = bool(is_bf16_supported())
    cfg = dict(output_dir = os.path.join(work, "sft"), max_steps = steps,
               per_device_train_batch_size = 2, gradient_accumulation_steps = 2,
               learning_rate = 2e-4, warmup_steps = 0, logging_steps = 1, seed = seed,
               report_to = "none", save_strategy = "no", dataset_text_field = "text",
               bf16 = bf16, fp16 = not bf16, optim = "adamw_torch",
               dataloader_num_workers = 0)
    try:
        args = SFTConfig(max_length = 256, **cfg)
    except TypeError:
        args = SFTConfig(max_seq_length = 256, **cfg)
    trainer = SFTTrainer(model = model, processing_class = tok, train_dataset = ds, args = args)
    import torch.utils.checkpoint as tuc
    obs["checkpoint_fn"] = getattr(tuc.checkpoint, "__name__", None)
    trainer.train()
    obs["sft_losses"] = [float(h["loss"]) for h in trainer.state.log_history if "loss" in h]
    obs["checkpoint_fn_after"] = getattr(tuc.checkpoint, "__name__", None)
    from unsloth_zoo import gradient_checkpointing as gcm
    obs["gc_after_train"] = _gc_summary(gcm)

    ut: dict = {}
    try:
        from unsloth_zoo.training_utils import unsloth_train
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            unsloth_train(trainer)
        losses = []
        for line in buf.getvalue().replace("\r", "\n").splitlines():
            m = _STEP_LOSS.match(line)
            if m:
                losses.append(float(m.group(2)))
        ut = {"ok": True, "losses": losses, "log_tail": buf.getvalue()[-1500:]}
    except BaseException as e:  # noqa: BLE001
        ut = {"ok": False, "error": _err(e), "traceback": traceback.format_exc()[-2500:]}
    obs["unsloth_train"] = ut
    obs["seconds"] = round(time.time() - t0, 1)
    return obs


def run_child(leg: str, args, work: Path, timeout: int) -> dict:
    out = work / f"{leg}.json"
    log = Path(args.out).with_name(f"probe_{args.state}_{leg}.log")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(args.checkout) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.setdefault("UNSLOTH_COMPILE_LOCATION", str(work / "unsloth_compiled_cache"))
    env["UNSLOTH_DISABLE_AUTO_UPDATES"] = "1"
    env["PYTHONHASHSEED"] = str(args.seed)
    if leg == "device":
        # Device leg imports unsloth_zoo WITHOUT unsloth; this is unsloth_zoo's own guard.
        env["UNSLOTH_IS_PRESENT"] = "1"
    cmd = [sys.executable, os.path.abspath(__file__), "--child", leg, "--state", args.state,
           "--checkout", str(args.checkout), "--out", str(out), "--model", args.model,
           "--steps", str(args.steps), "--seed", str(args.seed)]
    t0 = time.time()
    with open(log, "wb") as fh:
        try:
            rc = subprocess.run(cmd, stdout = fh, stderr = subprocess.STDOUT, env = env,
                                cwd = str(work), timeout = timeout).returncode
        except subprocess.TimeoutExpired:
            rc = "timeout"
    res: dict = {"rc": rc, "seconds": round(time.time() - t0, 1), "log": str(log)}
    if out.is_file():
        try:
            res.update(json.loads(out.read_text(encoding = "utf-8")))
        except Exception as e:  # noqa: BLE001
            res["parse_error"] = _err(e)
    else:
        res["missing_output"] = True
        with contextlib.suppress(Exception):
            res["log_tail"] = log.read_text(encoding = "utf-8", errors = "replace")[-3000:]
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True, type = Path)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--train", action = "store_true", help = "also run the LoRA training leg")
    ap.add_argument("--model", default = "unsloth/Llama-3.2-1B-Instruct")
    ap.add_argument("--steps", type = int, default = 3)
    ap.add_argument("--seed", type = int, default = 3407)
    ap.add_argument("--unsloth-dir", type = Path, default = None,
                    help = "checkout the environment's unsloth was installed from; its commit is recorded")
    ap.add_argument("--child", choices = ("device", "train"), default = None)
    args = ap.parse_args()
    args.checkout = args.checkout.resolve()

    if args.child:
        work = str(args.out.parent)
        try:
            if args.child == "device":
                obs = child_device(str(args.checkout))
            else:
                obs = child_train(str(args.checkout), args.model, args.steps, args.seed, work)
        except BaseException as e:  # noqa: BLE001
            obs = {"error": _err(e), "traceback": traceback.format_exc()[-3000:]}
        args.out.write_text(json.dumps(obs, indent = 2, default = str), encoding = "utf-8")
        return 0

    def _git(d: Path | None) -> str | None:
        if d is None:
            return None
        p = subprocess.run(["git", "-C", str(d), "rev-parse", "HEAD"], capture_output = True,
                           text = True, encoding = "utf-8")
        return p.stdout.strip() or None

    obs: dict = {"state": args.state, "checkout": str(args.checkout), "commit": _git(args.checkout),
                 "unsloth_commit": _git(args.unsloth_dir), "python": sys.executable,
                 "env": {k: os.environ.get(k) for k in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES",
                                                        "CUDA_VISIBLE_DEVICES", "HSA_OVERRIDE_GFX_VERSION")}}
    with tempfile.TemporaryDirectory(prefix = f"zoo_{args.state}_",
                                     dir = os.environ.get("RUNNER_TEMP") or None) as d:
        work = Path(d)
        obs["device"] = run_child("device", args, work, timeout = 900)
        obs["train"] = run_child("train", args, work, timeout = 3600) if args.train else {"skipped": True}
    args.out.write_text(json.dumps(obs, indent = 2, default = str), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
