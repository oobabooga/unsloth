#!/usr/bin/env python3
"""Probe: a short real LoRA SFT run through an unsloth OR unsloth-zoo checkout on THIS
host, recording the device / dtype / MoE grouped_mm decisions Unsloth made on the way.

Written for merged detection changes whose AMD question is "is this a strict no-op on
HIP": bf16 detection and the CUDA_DEVICE_ORDER hint (unsloth), the torch._grouped_mm
support probe and its ROCm crash-probe child (unsloth-zoo). Observes only; the
criteria modules criteria/unsloth_bf16_order_no_regression.py and
criteria/zoo_grouped_mm_no_regression.py judge.

Environment. Without `--studio-ref` the probe trains with the interpreter it runs
under (the Linux template's Studio venv, installed from the head worktree). With
`--studio-ref SHA` it builds ONE Studio environment for every state, from
unslothai/unsloth at SHA, with the same installer users run (`install.sh --local`
on Linux, `install.ps1 --local` on Windows), under $AMD_CI_WORK/probe_env, and reuses
it for the next state. That is how a `--no-studio` job (a PR outside unslothai/unsloth)
and the Windows template get a ROCm torch at all. The install runs with any HIP
device multiplier / NVIDIA identity variables stripped, so the installer sees the real
host; the training children get them back.

The checkout under test is a SOURCE tree of `--package`. It goes first on PYTHONPATH of
a FRESH child interpreter per repeat, so the state's package shadows the installed copy
while torch, transformers, TRL and the OTHER package stay identical across states. The
child records both packages' `__file__` so the criteria can refuse a run where the
overlay did not take.

Per repeat (default 2, so the A/A loss spread is observed): import unsloth, record the
bf16 answers (SUPPORTS_BFLOAT16, is_bfloat16_supported(), torch.cuda.is_bf16_supported(),
zoo device_is_bf16_supported(), decision._amp_dtype(cuda:0), cuda_order_mismatch_names()
where it exists), load --model with dtype=None, LoRA r=16 on attention + MLP / experts,
--steps SFT steps with bf16/fp16 chosen from is_bfloat16_supported(), losses from
log_history. Every `subprocess.run` whose code is the zoo grouped_mm crash probe is
recorded (argv index, child HIP/CUDA_VISIBLE_DEVICES, return code, phase). After
training the zoo moe_utils module state is read (_TORCH_GROUPED_MM_SUPPORTED,
_GROUPED_MM_SURVIVES, per-kind dicts, selected backend, Triton scratch buffer), then
grouped_mm support is asked for EVERY visible device (`_grouped_mm_supported_on` at a
tree that has it, the no-arg check otherwise), still recording children. The child's
combined stdout/stderr is scanned for the CUDA_DEVICE_ORDER hint. CUDA_DEVICE_ORDER
is removed from the child environment so the hint is not silenced by the runner.

Writes JSON to --out, never stdout (import banners corrupt stdout).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

PACKAGES = ("unsloth", "unsloth_zoo")
UPSTREAM_UNSLOTH = "https://github.com/unslothai/unsloth"
# Variables lib/device_multiplier.py and lib/nvidia_identity.py activate through.
SPOOF_VARS = ("LD_PRELOAD", "AMD_CI_SPOOFED_DEVICES", "AMD_CI_SPOOFED_VENDOR")
ORDER_MARKERS = ("CUDA_DEVICE_ORDER", "visible GPUs differ")


def _err(e: BaseException) -> str:
    return f"{type(e).__name__}: {e}"[:2000]


def _jsonable(v):
    try:
        json.dumps(v)
        return v
    except Exception:  # noqa: BLE001
        return repr(v)[:500]


def _call(fn, *a):
    try:
        return _jsonable(fn(*a))
    except Exception as e:  # noqa: BLE001
        return {"error": _err(e)}


def _pkg_info(name: str, checkout: str | None) -> dict:
    mod = sys.modules.get(name)
    if mod is None:
        return {"imported": False}
    f = str(Path(mod.__file__).resolve())
    d = {"imported": True, "file": f, "version": getattr(mod, "__version__", None)}
    if checkout:
        d["from_checkout"] = f.startswith(str(Path(checkout).resolve()) + os.sep)
    return d


def _root() -> Path:
    r = os.environ.get("AMD_CI_WORK") or os.environ.get("RUNNER_TEMP") or tempfile.gettempdir()
    return Path(r)


# ----------------------------------------------------------------------------- child


def child_train(args) -> dict:
    phase = ["import"]
    children: list[dict] = []
    _orig_run = subprocess.run

    def _run(*a, **k):
        argv = a[0] if a else k.get("args")
        is_probe = (isinstance(argv, (list, tuple)) and len(argv) >= 3 and argv[1] == "-c"
                    and "_grouped_mm" in str(argv[2]))
        t0 = time.time()
        try:
            r = _orig_run(*a, **k)
        except BaseException as e:  # noqa: BLE001
            if is_probe:
                children.append({"phase": phase[0], "index_arg": str(argv[-1]), "exception": _err(e)})
            raise
        if is_probe:
            env = k.get("env")
            src = env if env is not None else os.environ
            children.append({"phase": phase[0], "index_arg": str(argv[-1]), "rc": r.returncode,
                             "env_overridden": env is not None,
                             "HIP_VISIBLE_DEVICES": src.get("HIP_VISIBLE_DEVICES"),
                             "CUDA_VISIBLE_DEVICES": src.get("CUDA_VISIBLE_DEVICES"),
                             "seconds": round(time.time() - t0, 1)})
        return r

    subprocess.run = _run

    obs: dict = {"model": args.model, "steps": args.steps, "seed": args.seed,
                 "repeat": args.repeat, "package": args.package, "crash_probe_children": children}
    t_start = time.time()
    import unsloth  # noqa: F401  (must precede transformers / trl)
    from unsloth import FastLanguageModel
    import unsloth_zoo  # noqa: F401
    import torch
    import transformers
    import trl
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    checkout = str(args.checkout)
    obs["unsloth"] = _pkg_info("unsloth", checkout if args.package == "unsloth" else None)
    obs["unsloth_zoo"] = _pkg_info("unsloth_zoo", checkout if args.package == "unsloth_zoo" else None)
    obs["overlay"] = obs[args.package]
    obs["versions"] = {"torch": torch.__version__, "hip": getattr(torch.version, "hip", None),
                       "cuda": getattr(torch.version, "cuda", None),
                       "transformers": transformers.__version__, "trl": trl.__version__}
    obs["env"] = {k: os.environ.get(k) for k in ("CUDA_DEVICE_ORDER", "UNSLOTH_MOE_BACKEND",
                                                 "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES",
                                                 "ROCR_VISIBLE_DEVICES", "AMD_CI_SPOOFED_DEVICES",
                                                 "AMD_CI_SPOOFED_VENDOR")}
    dev: dict = {"cuda_available": torch.cuda.is_available()}
    if torch.cuda.is_available():
        dev["count"] = torch.cuda.device_count()
        dev["names"] = [_call(torch.cuda.get_device_name, i) for i in range(dev["count"])]
        dev["gcn_arch"] = getattr(torch.cuda.get_device_properties(0), "gcnArchName", None)
        dev["capability"] = _call(torch.cuda.get_device_capability)
    obs["device"] = dev

    flags: dict = {}
    with contextlib.suppress(Exception):
        from unsloth.models import _utils as uu
        flags["SUPPORTS_BFLOAT16"] = _jsonable(getattr(uu, "SUPPORTS_BFLOAT16", "absent"))
    for name in ("is_bfloat16_supported", "is_bf16_supported"):
        f = getattr(unsloth, name, None)
        flags[f"unsloth.{name}()"] = "absent" if f is None else _call(f)
    flags["torch.cuda.is_bf16_supported()"] = _call(torch.cuda.is_bf16_supported)
    try:
        from unsloth import device_type as udt
        flags["unsloth DEVICE_TYPE"] = getattr(udt, "DEVICE_TYPE", "absent")
        f = getattr(udt, "cuda_order_mismatch_names", None)
        flags["cuda_order_mismatch_names()"] = "absent" if f is None else _call(f)
    except Exception as e:  # noqa: BLE001
        flags["unsloth device_type import"] = _err(e)
    try:
        from unsloth_zoo import device_type as zdt
        flags["zoo DEVICE_TYPE"] = getattr(zdt, "DEVICE_TYPE", "absent")
        f = getattr(zdt, "device_is_bf16_supported", None)
        flags["zoo device_is_bf16_supported()"] = "absent" if f is None else _call(f)
    except Exception as e:  # noqa: BLE001
        flags["zoo device_type import"] = _err(e)
    try:
        from unsloth.models import decision
        f = getattr(decision, "_amp_dtype", None)
        flags["decision._amp_dtype(cuda:0)"] = "absent" if f is None else str(_call(f, torch.device("cuda", 0)))
    except Exception as e:  # noqa: BLE001
        flags["decision._amp_dtype(cuda:0)"] = {"error": _err(e)}
    obs["flags"] = flags

    phase[0] = "load"
    model, tok = FastLanguageModel.from_pretrained(
        model_name = args.model, max_seq_length = args.max_seq_length, dtype = None,
        load_in_4bit = False, load_in_8bit = False, full_finetuning = False)
    obs["model_dtype"] = str(next(model.parameters()).dtype)
    model = FastLanguageModel.get_peft_model(
        model, r = 16, lora_alpha = 16, lora_dropout = 0, bias = "none",
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing = "unsloth", random_state = args.seed)
    trainable = [(n, p.numel()) for n, p in model.named_parameters() if p.requires_grad]
    obs["trainable_params"] = int(sum(c for _, c in trainable))
    obs["trainable_tensors"] = len(trainable)
    obs["expert_lora_tensors"] = sum(1 for n, _ in trainable if "expert" in n)
    obs["load_seconds"] = round(time.time() - t_start, 1)

    rows = []
    for i in range(64):
        body = " ".join(f"Item {i}-{j}: {(i * 7 + j * 13) % 97} plus {(i + j) % 31} equals "
                        f"{(i * 7 + j * 13) % 97 + (i + j) % 31}." for j in range(6))
        rows.append(f"Question {i}: add the pairs below.\n{body}\nThe quick brown fox {i}.")
    ds = Dataset.from_dict({"text": rows})
    bf16 = bool(unsloth.is_bfloat16_supported())
    obs["trainer_bf16"], obs["trainer_fp16"] = bf16, not bf16
    cfg = dict(output_dir = os.path.join(args.work, "sft"), max_steps = args.steps,
               per_device_train_batch_size = 2, gradient_accumulation_steps = 1,
               learning_rate = 2e-4, warmup_steps = 0, logging_steps = 1, seed = args.seed,
               data_seed = args.seed, report_to = "none", save_strategy = "no",
               dataset_text_field = "text", bf16 = bf16, fp16 = not bf16, optim = "adamw_torch",
               dataloader_num_workers = 0, packing = False)
    try:
        targs = SFTConfig(max_length = args.max_seq_length, **cfg)
    except TypeError:
        targs = SFTConfig(max_seq_length = args.max_seq_length, **cfg)
    trainer = SFTTrainer(model = model, processing_class = tok, train_dataset = ds, args = targs)
    obs["trainer_class"] = f"{type(trainer).__module__}.{type(trainer).__qualname__}"
    phase[0] = "train"
    t0 = time.time()
    trainer.train()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    obs["train_seconds"] = round(time.time() - t0, 2)
    obs["losses"] = [float(h["loss"]) for h in trainer.state.log_history if "loss" in h]
    obs["grad_norms"] = [float(h["grad_norm"]) for h in trainer.state.log_history
                         if isinstance(h.get("grad_norm"), (int, float))]

    mu = sys.modules.get("unsloth_zoo.temporary_patches.moe_utils")
    moe: dict = {"module_loaded": mu is not None}
    if mu is not None:
        for name in ("_TORCH_GROUPED_MM_AVAILABLE", "_TORCH_GROUPED_MM_SUPPORTED", "_GROUPED_MM_SURVIVES",
                     "_TORCH_GROUPED_MM_SUPPORTED_BY_KIND", "_GROUPED_MM_SURVIVES_BY_KIND",
                     "_GROUPED_MM_FOREIGN", "_GROUPED_MM_HOME", "_SELECTED_MOE_BACKEND",
                     "_TRANSPOSED_VIEW_GROUPED_MM_SAFE"):
            v = getattr(mu, name, "absent")
            moe[name] = _jsonable(v if not isinstance(v, dict) else {repr(k): repr(x) for k, x in v.items()})
        buf = getattr(mu, "_PERSISTENT_BUFFER", "absent")
        moe["_PERSISTENT_BUFFER"] = (
            {"type": "dict", "keys": [repr(k) for k in buf]} if isinstance(buf, dict)
            else {"type": type(buf).__name__})
        with contextlib.suppress(Exception):
            dump = torch._C._dispatch_dump("aten::_grouped_mm").splitlines()
            moe["python_grouped_mm_fallback"] = any(
                ln.startswith("CUDA:") and ".cpp:" not in ln for ln in dump)
        moe["children_after_train"] = len(children)
        phase[0] = "query"
        answers = []
        per_device = getattr(mu, "_grouped_mm_supported_on", None)
        for i in range(torch.cuda.device_count() if torch.cuda.is_available() else 0):
            if per_device is not None:
                answers.append(_call(per_device, torch.device("cuda", i)))
            else:
                answers.append(_call(mu._check_torch_grouped_mm_supported))
        moe["per_device_answers"] = answers
        moe["per_device_api"] = per_device is not None
        moe["children_after_query"] = len(children)
        for name in ("_TORCH_GROUPED_MM_SUPPORTED_BY_KIND", "_GROUPED_MM_SURVIVES_BY_KIND", "_GROUPED_MM_FOREIGN"):
            v = getattr(mu, name, "absent")
            moe[name + " after query"] = (
                {repr(k): repr(x) for k, x in v.items()} if isinstance(v, dict) else _jsonable(v))
        sel = getattr(mu, "select_moe_backend", None)
        moe["select_moe_backend()"] = "absent" if sel is None else _call(sel)
    obs["moe"] = moe
    obs["seconds"] = round(time.time() - t_start, 1)
    return obs


# ----------------------------------------------------------------------------- parent


def bootstrap(ref: str, log_dir: Path) -> dict:
    """One Studio environment for every state, built with the installer users run."""
    root = _root() / "probe_env"
    marker = root / "python.txt"
    info: dict = {"ref": ref, "root": str(root)}
    if marker.is_file():
        info.update(json.loads(marker.read_text(encoding = "utf-8")))
        info["reused"] = True
        return info
    root.mkdir(parents = True, exist_ok = True)
    src = root / "src"
    log = log_dir / "probe_env_install.log"
    env = {k: v for k, v in os.environ.items() if k not in SPOOF_VARS}
    # nvidia_identity puts its directory first on PATH / PYTHONPATH; the installer must see the real host.
    shim = str(_root() / "nvidia")
    for var in ("PATH", "PYTHONPATH"):
        if env.get(var):
            env[var] = os.pathsep.join(p for p in env[var].split(os.pathsep) if not p.startswith(shim))
    env["UNSLOTH_STUDIO_HOME"] = str(root / "studio")
    env["UV_EXTRA_INDEX_URL"] = "https://pypi.org/simple"
    env["PYTHONUTF8"] = "1"
    env["UNSLOTH_SKIP_AUTOSTART"] = "1"   # never launch the Studio server from inside a CI job
    t0 = time.time()
    with open(log, "wb") as fh:
        def sh(cmd, cwd = None, timeout = 600):
            fh.write(("\n$ " + " ".join(cmd) + "\n").encode("utf-8"))
            fh.flush()
            return subprocess.run(cmd, cwd = cwd, env = env, stdout = fh, stderr = subprocess.STDOUT,
                                  timeout = timeout).returncode
        try:
            rc = sh(["git", "init", "-q", str(src)])
            rc = rc or sh(["git", "-C", str(src), "fetch", "-q", "--depth", "1", UPSTREAM_UNSLOTH, ref])
            rc = rc or sh(["git", "-C", str(src), "checkout", "-q", "FETCH_HEAD"])
            if rc == 0:
                if os.name == "nt":
                    cmd = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                           str(src / "install.ps1"), "--local", "--skip-autostart"]
                else:
                    cmd = ["bash", str(src / "install.sh"), "--local"]
                rc = sh(cmd, cwd = str(src), timeout = 3600)
        except subprocess.TimeoutExpired as e:
            rc = f"timeout: {e}"
        except Exception as e:  # noqa: BLE001
            rc = _err(e)
    info["install_rc"] = rc
    info["install_seconds"] = round(time.time() - t0, 1)
    info["install_log"] = str(log)
    studio = root / "studio"
    cands = []
    for d in [studio / "unsloth_studio", *sorted(studio.glob(".venv_t5_*")), *sorted(studio.glob("*venv*"))]:
        cands += [d / "bin" / "python", d / "Scripts" / "python.exe"]
    py = next((str(c) for c in cands if c.is_file()), None)
    info["python"] = py
    if py is None or rc != 0:
        with contextlib.suppress(Exception):
            info["install_log_tail"] = log.read_text(encoding = "utf-8", errors = "replace")[-3000:]
        return info
    marker.write_text(json.dumps({k: info[k] for k in ("python", "install_rc", "install_seconds", "install_log")}),
                      encoding = "utf-8")
    return info


def run_child(args, python: str, repeat: int, root: Path, timeout: int) -> dict:
    work = root / f"r{repeat}"
    work.mkdir(parents = True, exist_ok = True)
    out = work / "train.json"
    log = Path(args.out).with_name(f"probe_{args.state}_train_r{repeat}.log")
    env = dict(os.environ)
    env.pop("CUDA_DEVICE_ORDER", None)
    env["PYTHONPATH"] = str(args.checkout) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["UNSLOTH_COMPILE_LOCATION"] = str(work / "unsloth_compiled_cache")
    env["TRITON_CACHE_DIR"] = str(work / "triton_cache")
    env["TORCHINDUCTOR_CACHE_DIR"] = str(work / "inductor_cache")
    env["UNSLOTH_DISABLE_AUTO_UPDATES"] = "1"
    env["PYTHONHASHSEED"] = str(args.seed)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env.setdefault("HF_HOME", str(_root() / "hf_home"))
    env.setdefault("XDG_CACHE_HOME", str(_root() / "xdg_cache"))
    cmd = [python, os.path.abspath(__file__), "--child", "--state", args.state,
           "--checkout", str(args.checkout), "--out", str(out), "--package", args.package,
           "--model", args.model, "--steps", str(args.steps), "--seed", str(args.seed),
           "--max-seq-length", str(args.max_seq_length), "--repeat", str(repeat)]
    t0 = time.time()
    with open(log, "wb") as fh:
        try:
            rc = subprocess.run(cmd, stdout = fh, stderr = subprocess.STDOUT, env = env,
                                cwd = str(work), timeout = timeout).returncode
        except subprocess.TimeoutExpired:
            rc = "timeout"
    res: dict = {"repeat": repeat, "rc": rc, "wall_seconds": round(time.time() - t0, 1), "log": str(log)}
    text = ""
    with contextlib.suppress(Exception):
        text = log.read_text(encoding = "utf-8", errors = "replace")
    res["order_hint_lines"] = [ln.strip()[:400] for ln in text.splitlines()
                               if any(m in ln for m in ORDER_MARKERS)][:10]
    if out.is_file():
        try:
            res.update(json.loads(out.read_text(encoding = "utf-8")))
            res["repeat"] = repeat
        except Exception as e:  # noqa: BLE001
            res["parse_error"] = _err(e)
    else:
        res["missing_output"] = True
        res["log_tail"] = text[-3000:]
    res["losses_finite"] = bool(res.get("losses")) and all(
        isinstance(x, (int, float)) and math.isfinite(x) for x in res.get("losses") or [])
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
    ap.add_argument("--package", required = True, choices = PACKAGES)
    ap.add_argument("--model", default = "trl-internal-testing/tiny-Qwen3ForCausalLM")
    ap.add_argument("--steps", type = int, default = 3)
    ap.add_argument("--seed", type = int, default = 3407)
    ap.add_argument("--max-seq-length", type = int, default = 256)
    ap.add_argument("--repeats", type = int, default = 2)
    ap.add_argument("--timeout", type = int, default = 1500, help = "per repeat, seconds")
    ap.add_argument("--studio-ref", default = None,
                    help = "build one Studio environment from unslothai/unsloth at this commit "
                           "(install.sh / install.ps1 --local) and train with its python")
    ap.add_argument("--expect-head", default = None,
                    help = "merge commit the head state must resolve to (recorded; the criteria gate on it, "
                           "because states.py finds a merged PR by grepping `(#N)`, which a later revert matches)")
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

    obs: dict = {"state": args.state, "checkout": str(args.checkout), "commit": _git(args.checkout),
                 "package": args.package, "package_dir_exists": (args.checkout / args.package).is_dir(),
                 "platform": sys.platform,
                 "config": {"model": args.model, "steps": args.steps, "seed": args.seed,
                            "max_seq_length": args.max_seq_length, "repeats": args.repeats,
                            "studio_ref": args.studio_ref, "expect_head": args.expect_head},
                 "spoofed_devices": os.environ.get("AMD_CI_SPOOFED_DEVICES"),
                 "runs": []}
    python = sys.executable
    if args.studio_ref:
        b = bootstrap(args.studio_ref, Path(args.out).parent)
        obs["bootstrap"] = b
        python = b.get("python")
    obs["python"] = python
    if python:
        base_tmp = os.environ.get("RUNNER_TEMP") or None
        with tempfile.TemporaryDirectory(prefix = f"lora_{args.state}_", dir = base_tmp,
                                         ignore_cleanup_errors = True) as d:
            for r in range(1, args.repeats + 1):
                obs["runs"].append(run_child(args, python, r, Path(d), args.timeout))
    args.out.write_text(json.dumps(obs, indent = 2, default = str), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
