#!/usr/bin/env python3
"""Probe: single-GPU LoRA training speed, peak memory, loss and compile behaviour of
an unsloth OR unsloth-zoo checkout on THIS host.

Written for merged capability / arch-detection changes (per-device caches, bf16 /
compile / TMA gates) where the question on AMD is "did training get slower, use more
memory, change its losses or add graph breaks". Observes only;
criteria/train_perf_no_regression.py judges.

The checkout under test is a SOURCE tree of `--package` (unsloth or unsloth_zoo). It
is put first on PYTHONPATH of a FRESH child interpreter per repeat, so the state's
package shadows the installed copy while torch, transformers, TRL and the OTHER
package stay identical across states. The child records both packages' `__file__`
so the criteria can refuse a run where the overlay did not take.

Each repeat (default 2, so A/A noise is measured per state) is its own process with
its own UNSLOTH_COMPILE_LOCATION and TRITON_CACHE_DIR: 16-bit LoRA SFT on --model,
use_gradient_checkpointing="unsloth", deterministic synthetic data, seed 3407,
batch 2 x grad accum 4, --steps optimizer steps. Recorded per repeat: per-step wall
time (torch.cuda.synchronize at both ends of every optimizer step), losses,
max_memory_allocated / max_memory_reserved since reset after model load, dynamo
graph-break and unique-graph counters, SUPPORTS_BFLOAT16, model dtype, the torch
capability Unsloth sees, and unsloth_zoo's _detect_gfx_arch() /
get_amd_attention_implementation() where importable.

Writes JSON to --out, never stdout (import banners corrupt stdout).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

PACKAGES = ("unsloth", "unsloth_zoo")


def _err(e: BaseException) -> str:
    return f"{type(e).__name__}: {e}"[:2000]


def _jsonable(v):
    try:
        json.dumps(v)
        return v
    except Exception:  # noqa: BLE001
        return repr(v)[:500]


def _pkg_info(name: str, checkout: str | None) -> dict:
    mod = sys.modules.get(name)
    if mod is None:
        return {"imported": False}
    f = str(Path(mod.__file__).resolve())
    d = {"imported": True, "file": f, "version": getattr(mod, "__version__", None)}
    if checkout:
        d["from_checkout"] = f.startswith(str(Path(checkout).resolve()) + os.sep)
    # Installed distribution (the copy the overlay shadows, or the one held constant):
    # a git+https install records its commit in direct_url.json.
    with contextlib.suppress(Exception):
        from importlib import metadata
        dist = metadata.distribution(name.replace("_", "-"))
        d["dist_version"] = dist.version
        du = dist.read_text("direct_url.json")
        if du:
            d["dist_vcs"] = (json.loads(du).get("vcs_info") or {}).get("commit_id")
    return d


def _call(fn, *a):
    try:
        return _jsonable(fn(*a))
    except Exception as e:  # noqa: BLE001
        return {"error": _err(e)}


def child_train(args) -> dict:
    obs: dict = {"model": args.model, "steps": args.steps, "seed": args.seed,
                 "repeat": args.repeat, "batch": args.batch, "grad_accum": args.grad_accum,
                 "package": args.package}
    t_start = time.time()
    import unsloth  # noqa: F401  (must precede transformers / trl)
    from unsloth import FastLanguageModel
    import unsloth_zoo  # noqa: F401
    import torch
    import transformers
    import trl
    from datasets import Dataset
    from transformers import TrainerCallback
    from trl import SFTConfig, SFTTrainer

    checkout = str(args.checkout)
    obs["unsloth"] = _pkg_info("unsloth", checkout if args.package == "unsloth" else None)
    obs["unsloth_zoo"] = _pkg_info("unsloth_zoo", checkout if args.package == "unsloth_zoo" else None)
    obs["overlay"] = obs[args.package]
    obs["versions"] = {"torch": torch.__version__, "hip": getattr(torch.version, "hip", None),
                       "transformers": transformers.__version__, "trl": trl.__version__}
    dev: dict = {"cuda_available": torch.cuda.is_available()}
    if torch.cuda.is_available():
        dev["count"] = torch.cuda.device_count()
        dev["name"] = _call(torch.cuda.get_device_name, 0)
        dev["gcn_arch"] = getattr(torch.cuda.get_device_properties(0), "gcnArchName", None)
        dev["capability"] = _call(torch.cuda.get_device_capability)
        dev["capability_per_device_cache"] = bool(
            getattr(torch.cuda.get_device_capability, "_unsloth_per_device_cache", False))
    obs["device"] = dev

    flags: dict = {}
    try:
        from unsloth.models import _utils as uu
        flags["SUPPORTS_BFLOAT16"] = getattr(uu, "SUPPORTS_BFLOAT16", "absent")
    except Exception as e:  # noqa: BLE001
        flags["SUPPORTS_BFLOAT16"] = {"error": _err(e)}
    flags["is_bf16_supported()"] = _call(unsloth.is_bf16_supported)
    try:
        from unsloth_zoo import device_type as zdt
        flags["zoo DEVICE_TYPE"] = getattr(zdt, "DEVICE_TYPE", "absent")
        for fn in ("_detect_gfx_arch", "get_amd_attention_implementation"):
            f = getattr(zdt, fn, None)
            flags[fn] = "absent" if f is None else _call(f)
        f = getattr(zdt, "get_amd_flash_attn_func", None)
        if f is None:
            flags["get_amd_flash_attn_func"] = "absent"
        else:
            r = _call(f)
            flags["get_amd_flash_attn_func"] = r if isinstance(r, dict) or r is None else (
                f"{getattr(r, '__module__', '?')}.{getattr(r, '__qualname__', repr(r))}")
    except Exception as e:  # noqa: BLE001
        flags["zoo device_type import"] = _err(e)
    obs["flags"] = flags

    model, tok = FastLanguageModel.from_pretrained(
        model_name = args.model, max_seq_length = args.max_seq_length, dtype = None,
        load_in_4bit = False, load_in_8bit = False, full_finetuning = False)
    model = FastLanguageModel.get_peft_model(
        model, r = 16, lora_alpha = 16, lora_dropout = 0, bias = "none",
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing = "unsloth", random_state = args.seed)
    obs["model_dtype"] = str(next(model.parameters()).dtype)
    obs["attn_implementation"] = getattr(getattr(model, "config", None), "_attn_implementation", None)
    obs["load_seconds"] = round(time.time() - t_start, 1)

    # Deterministic synthetic data, ~150-250 tokens per row so steps are not launch-bound.
    rows = []
    for i in range(256):
        body = " ".join(f"Item {i}-{j}: {(i * 7 + j * 13) % 97} plus {(i + j) % 31} equals "
                        f"{(i * 7 + j * 13) % 97 + (i + j) % 31}." for j in range(10))
        rows.append(f"Question {i}: add the pairs below.\n{body}\nThe quick brown fox number {i} "
                    f"jumps over the lazy dog.")
    ds = Dataset.from_dict({"text": rows})
    bf16 = bool(unsloth.is_bf16_supported())

    sync = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)
    step_times: list[float] = []

    class StepTimer(TrainerCallback):
        def on_step_begin(self, a, state, control, **kw):
            sync()
            self.t0 = time.perf_counter()

        def on_step_end(self, a, state, control, **kw):
            sync()
            step_times.append(time.perf_counter() - self.t0)

    cfg = dict(output_dir = os.path.join(args.work, "sft"), max_steps = args.steps,
               per_device_train_batch_size = args.batch,
               gradient_accumulation_steps = args.grad_accum,
               learning_rate = 2e-4, warmup_steps = 0, logging_steps = 1, seed = args.seed,
               data_seed = args.seed, report_to = "none", save_strategy = "no",
               dataset_text_field = "text", bf16 = bf16, fp16 = not bf16, optim = "adamw_torch",
               dataloader_num_workers = 0, packing = False)
    try:
        targs = SFTConfig(max_length = args.max_seq_length, **cfg)
    except TypeError:
        targs = SFTConfig(max_seq_length = args.max_seq_length, **cfg)
    trainer = SFTTrainer(model = model, processing_class = tok, train_dataset = ds, args = targs,
                         callbacks = [StepTimer()])
    obs["trainer_class"] = f"{type(trainer).__module__}.{type(trainer).__qualname__}"

    from torch._dynamo.utils import counters
    counters.clear()
    if torch.cuda.is_available():
        sync()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        obs["mem_before_train_gb"] = torch.cuda.memory_allocated() / 1024 ** 3
    t0 = time.time()
    trainer.train()
    sync()
    obs["train_seconds"] = round(time.time() - t0, 2)
    if torch.cuda.is_available():
        obs["peak_allocated_gb"] = torch.cuda.max_memory_allocated() / 1024 ** 3
        obs["peak_reserved_gb"] = torch.cuda.max_memory_reserved() / 1024 ** 3
    obs["step_times"] = step_times
    obs["losses"] = [float(h["loss"]) for h in trainer.state.log_history if "loss" in h]
    obs["graph_breaks"] = int(sum(counters["graph_break"].values()))
    obs["graph_break_reasons"] = {str(k)[:300]: int(v) for k, v in
                                  sorted(counters["graph_break"].items(), key = lambda kv: -kv[1])[:20]}
    obs["unique_graphs"] = int(counters["stats"].get("unique_graphs", 0))
    obs["recompiles"] = {str(k)[:200]: int(v) for k, v in counters["recompiles"].items()} \
        if "recompiles" in counters else {}
    obs["seconds"] = round(time.time() - t_start, 1)
    return obs


def run_child(args, repeat: int, root: Path, timeout: int) -> dict:
    work = root / f"r{repeat}"
    work.mkdir(parents = True, exist_ok = True)
    out = work / "train.json"
    log = Path(args.out).with_name(f"probe_{args.state}_train_r{repeat}.log")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(args.checkout) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    # Per repeat, never inherited: a cache compiled by one state / repeat must not serve another.
    env["UNSLOTH_COMPILE_LOCATION"] = str(work / "unsloth_compiled_cache")
    env["TRITON_CACHE_DIR"] = str(work / "triton_cache")
    env["TORCHINDUCTOR_CACHE_DIR"] = str(work / "inductor_cache")
    env["UNSLOTH_DISABLE_AUTO_UPDATES"] = "1"
    env["PYTHONHASHSEED"] = str(args.seed)
    root_env = os.environ.get("AMD_CI_WORK") or os.environ.get("RUNNER_TEMP")
    if root_env:
        # Shared across states and repeats (one model download), not across jobs.
        env.setdefault("HF_HOME", os.path.join(root_env, "hf_home"))
        env.setdefault("XDG_CACHE_HOME", os.path.join(root_env, "xdg_cache"))
    cmd = [sys.executable, os.path.abspath(__file__), "--child", "--state", args.state,
           "--checkout", str(args.checkout), "--out", str(out), "--package", args.package,
           "--model", args.model, "--steps", str(args.steps), "--seed", str(args.seed),
           "--batch", str(args.batch), "--grad-accum", str(args.grad_accum),
           "--max-seq-length", str(args.max_seq_length), "--repeat", str(repeat)]
    t0 = time.time()
    with open(log, "wb") as fh:
        try:
            rc = subprocess.run(cmd, stdout = fh, stderr = subprocess.STDOUT, env = env,
                                cwd = str(work), timeout = timeout).returncode
        except subprocess.TimeoutExpired:
            rc = "timeout"
    res: dict = {"rc": rc, "wall_seconds": round(time.time() - t0, 1), "log": str(log)}
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


def _git(d: Path | None) -> str | None:
    if d is None:
        return None
    p = subprocess.run(["git", "-C", str(d), "rev-parse", "HEAD"], capture_output = True,
                       text = True, encoding = "utf-8")
    return p.stdout.strip() or None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True, type = Path)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--package", required = True, choices = PACKAGES,
                    help = "which package the checkout is a source tree of")
    ap.add_argument("--model", default = "unsloth/Llama-3.2-1B-Instruct")
    ap.add_argument("--steps", type = int, default = 30)
    ap.add_argument("--seed", type = int, default = 3407)
    ap.add_argument("--batch", type = int, default = 2)
    ap.add_argument("--grad-accum", type = int, default = 4)
    ap.add_argument("--max-seq-length", type = int, default = 512)
    ap.add_argument("--repeats", type = int, default = 2)
    ap.add_argument("--timeout", type = int, default = 1800, help = "per repeat, seconds")
    ap.add_argument("--env-dir", type = Path, default = None,
                    help = "checkout the environment was installed from; its commit is recorded")
    ap.add_argument("--dry-run", action = "store_true",
                    help = "write the observation skeleton without spawning training children")
    ap.add_argument("--child", action = "store_true")
    ap.add_argument("--repeat", type = int, default = 0)
    args = ap.parse_args()
    args.checkout = args.checkout.resolve()

    if args.child:
        args.work = str(args.out.parent)
        try:
            obs = child_train(args)
        except BaseException as e:  # noqa: BLE001
            obs = {"error": _err(e), "traceback": traceback.format_exc()[-3000:]}
        args.out.write_text(json.dumps(obs, indent = 2, default = str), encoding = "utf-8")
        return 0

    pkg_dir = args.checkout / args.package
    obs: dict = {"state": args.state, "checkout": str(args.checkout), "commit": _git(args.checkout),
                 "package": args.package, "package_dir_exists": pkg_dir.is_dir(),
                 "env_commit": _git(args.env_dir), "python": sys.executable,
                 "config": {"model": args.model, "steps": args.steps, "seed": args.seed,
                            "batch": args.batch, "grad_accum": args.grad_accum,
                            "max_seq_length": args.max_seq_length, "repeats": args.repeats},
                 "env": {k: os.environ.get(k) for k in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES",
                                                        "CUDA_VISIBLE_DEVICES", "HSA_OVERRIDE_GFX_VERSION")},
                 "runs": []}
    if args.dry_run:
        obs["dry_run"] = True
    else:
        with tempfile.TemporaryDirectory(prefix = f"perf_{args.state}_",
                                         dir = os.environ.get("RUNNER_TEMP") or None) as d:
            for r in range(1, args.repeats + 1):
                obs["runs"].append(run_child(args, r, Path(d), args.timeout))
    args.out.write_text(json.dumps(obs, indent = 2, default = str), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
