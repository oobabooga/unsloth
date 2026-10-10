#!/usr/bin/env python3
"""Probe: can THIS checkout's Studio install vLLM on this AMD GPU and generate with it?

Observes only, writes JSON to --out. Per state: host facts, vLLM / SGLang support verdicts,
then (only when the checkout says vLLM is supported) the real managed install through
engine_install._install, and loads + generations through ManagedEngine, the path Studio's
orchestrator uses. criteria/vllm_rocm_install.py judges.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
import threading
import time
import traceback
from pathlib import Path

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
PROMPT = "What is the capital of France? Answer with one word."
# Which quantized formats vLLM's ROCm build serves here decides what Studio may offer on AMD.
CHECKPOINTS = (
    "Qwen/Qwen2.5-0.5B-Instruct-AWQ",
    "Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4",
    "RedHatAI/Qwen2.5-0.5B-Instruct-FP8-dynamic",
    "RedHatAI/Qwen2.5-0.5B-Instruct-quantized.w8a8",
    "RedHatAI/Qwen3-0.6B-quantized.w4a16",
    "mgoin/Qwen3-0.6B-NVFP4",
)


def tree_bytes(root: Path) -> int:
    seen, total = set(), 0
    for dirpath, _, files in os.walk(root):
        for name in files:
            try:
                st = os.lstat(os.path.join(dirpath, name))
            except OSError:
                continue
            if (st.st_dev, st.st_ino) in seen:
                continue
            seen.add((st.st_dev, st.st_ino))
            total += st.st_blocks * 512
    return total


def host_facts() -> dict:
    facts = {"python": sys.version.split()[0], "glibc": platform.libc_ver()[1]}
    try:
        facts["rocm_version"] = Path("/opt/rocm/.info/version").read_text(encoding = "utf-8").strip()
    except OSError as e:
        facts["rocm_version"] = f"unreadable: {e}"
    facts["kfd_rw"] = os.access("/dev/kfd", os.R_OK | os.W_OK)
    facts["cc"] = os.environ.get("CC") or shutil.which("gcc") or shutil.which("clang")
    facts["uv"] = shutil.which("uv")
    try:
        import torch
        facts["torch"] = torch.__version__
        facts["torch_hip"] = getattr(torch.version, "hip", None)
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            facts["arch"] = getattr(props, "gcnArchName", None)
            facts["device"] = torch.cuda.get_device_name(0)
    except Exception as e:  # noqa: BLE001
        facts["torch_error"] = f"{type(e).__name__}: {e}"
    return facts


class _Load:
    """The fields of a load request validate_load reads."""

    def __init__(self, precision: str, model: str = MODEL, gpu_ids = None):
        self.gpu_ids = gpu_ids
        self.model_path = model
        self.gguf_variant = None
        self.engine_precision = precision
        self.engine_parallelism = "tensor"


def gpus(managed_engine, precision: str = "auto", model: str = MODEL):
    # The physical GPUs Studio itself would launch on: inside the parent's visible set.
    return managed_engine.validate_load("vllm", _Load(precision, model))


def generate(managed_engine, engine_install, precision: str, model: str = MODEL) -> dict:
    out = {"precision": precision, "model": model}
    engine = managed_engine.ManagedEngine("vllm")
    t0 = time.monotonic()
    try:
        gpu_ids = gpus(managed_engine, precision, model)
        out["gpu_ids"] = gpu_ids
        engine.start(model, 2048, gpu_ids, dict(os.environ), None, {"precision": precision})
        out["load_s"] = round(time.monotonic() - t0, 1)
        t1 = time.monotonic()
        stats: dict = {}
        text = "".join(
            engine.generate(
                messages = [{"role": "user", "content": PROMPT}],
                temperature = 0.0,
                max_new_tokens = 32,
                stats_holder = stats,
            )
        )
        out["gen_s"] = round(time.monotonic() - t1, 2)
        out["text"] = text
        out["usage"] = (stats.get("stats") or {}).get("usage")
    except Exception as e:  # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {e}"
        out["traceback"] = traceback.format_exc()[-3000:]
        out["engine_tail"] = list(getattr(engine, "_tail", []))[-60:]
    finally:
        try:
            engine.stop()
        except Exception as e:  # noqa: BLE001
            out["stop_error"] = f"{type(e).__name__}: {e}"
        # The next load budgets from the driver's free memory, which trails a just-stopped engine.
        time.sleep(30)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True, type = Path)
    ap.add_argument("--out", required = True, type = Path)
    args = ap.parse_args()

    obs: dict = {"state": args.state, "host": host_facts()}
    # Studio's own uv sits beside its interpreter or in the Studio home.
    home = os.environ.get("UNSLOTH_STUDIO_HOME", "")
    # install.sh puts uv in ~/.local/bin (or UV_INSTALL_DIR), which a login shell has on PATH.
    extra = [str(Path(sys.executable).parent), str(Path.home() / ".local" / "bin")]
    extra += [os.environ[k] for k in ("UV_INSTALL_DIR", "XDG_BIN_HOME") if os.environ.get(k)]
    extra += [str(Path(home) / "bin")] if home else []
    os.environ["PATH"] = os.pathsep.join([*extra, os.environ.get("PATH", "")])
    obs["host"]["uv_after_path"] = shutil.which("uv")
    backend = args.checkout / "studio" / "backend"
    sys.path.insert(0, str(backend))
    os.chdir(backend)
    try:
        from core.inference import engine_install, managed_engine
    except Exception as e:  # noqa: BLE001
        obs["import_error"] = f"{type(e).__name__}: {e}"
        args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
        return 0

    def ask(name, *a, **k):
        try:
            return getattr(engine_install, name)(*a, **k)
        except Exception as e:  # noqa: BLE001
            return f"raised {type(e).__name__}: {e}"

    obs["gpu_platform"] = ask("gpu_platform") if hasattr(engine_install, "gpu_platform") else None
    obs["vllm_reason"] = ask("support_reason", "vllm")
    obs["sglang_reason"] = ask("support_reason", "sglang")
    status = ask("status", "vllm")
    obs["vllm_status"] = (
        {k: status.get(k) for k in ("version", "installed", "unsupported_reason", "precisions", "platform", "download_bytes")}
        if isinstance(status, dict)
        else status
    )
    obs["lock"] = ask("profile", "vllm").get("lock") if isinstance(ask("profile", "vllm"), dict) else None

    if obs["vllm_reason"] is None:
        t0 = time.monotonic()
        try:
            engine_install._install("vllm", threading.Event())
        except Exception as e:  # noqa: BLE001
            obs["install_error"] = f"{type(e).__name__}: {e}"[-4000:]
        obs["install_s"] = round(time.monotonic() - t0, 1)
        job = dict(engine_install._jobs.get("vllm") or {})
        obs["install_job"] = {k: job.get(k) for k in ("state", "phase", "message")}
        obs["install_log_tail"] = (job.get("log") or [])[-25:]
        # _install reports failure through its job record, not by raising.
        obs["install_ok"] = job.get("state") == "success" and "install_error" not in obs
        if job.get("state") == "error":
            obs.setdefault("install_error", job.get("message"))
        info = engine_install.installed("vllm")
        obs["installed_info"] = (
            {k: info.get(k) for k in ("version", "platform", "shared", "python", "directory")} if info else None
        )
        if info:
            obs["env_bytes"] = tree_bytes(Path(info["path"]))
            uv_cache = Path(engine_install.install_environment()["UV_CACHE_DIR"])
            obs["uv_cache_bytes"] = tree_bytes(uv_cache) if uv_cache.is_dir() else None
            obs["generations"] = [generate(managed_engine, engine_install, p) for p in ("auto", "fp8")]
            obs["checkpoints"] = [
                generate(managed_engine, engine_install, "auto", model) for model in CHECKPOINTS
            ]
            # 4-bit load-time conversion is refused on AMD before anything is unloaded.
            try:
                managed_engine.validate_load("vllm", _Load("int4"))
                obs["int4_refusal"] = None
            except Exception as e:  # noqa: BLE001
                obs["int4_refusal"] = f"{type(e).__name__}: {e}"

    args.out.write_text(json.dumps(obs, indent = 2, default = str), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
