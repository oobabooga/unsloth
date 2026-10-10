#!/usr/bin/env python3
"""Probe (Windows): can THIS checkout's Studio run vLLM on the AMD GPU through its private WSL distro?

Observes only, writes JSON to --out. Per state: Windows / WSL / GPU facts, the vLLM and SGLang
support verdicts, and, only when the checkout says vLLM is supported, the real managed install
(engine_install._install: WSL distro, ROCm + librocdxg, the vLLM lock) followed by a load through
ManagedEngine.start and one greedy chat completion against the engine's own endpoint.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
PROMPT = "What is the capital of France? Answer with one word."


def run(argv, timeout = 120) -> dict:
    try:
        p = subprocess.run(argv, capture_output = True, timeout = timeout)
        raw = p.stdout + p.stderr
        text = raw.decode("utf-16-le", "replace") if raw[1:2] == b"\x00" else raw.decode("utf-8", "replace")
        return {"rc": p.returncode, "out": text.replace("\x00", "")[-2000:]}
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}


def host_facts() -> dict:
    facts = {"python": sys.version.split()[0]}
    try:
        facts["admin"] = bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception as e:  # noqa: BLE001
        facts["admin"] = f"{type(e).__name__}: {e}"
    try:
        import torch
        facts["torch"] = torch.__version__
        facts["torch_hip"] = getattr(torch.version, "hip", None)
        facts["torch_gpu"] = torch.cuda.is_available() and torch.cuda.get_device_name(0)
    except Exception as e:  # noqa: BLE001
        facts["torch_error"] = f"{type(e).__name__}: {e}"
    facts["wsl_status"] = run(["wsl.exe", "--status"])
    facts["wsl_version"] = run(["wsl.exe", "--version"])
    facts["wsl_list"] = run(["wsl.exe", "-l", "-v"])
    facts["video"] = run(
        ["powershell", "-NoProfile", "-Command",
         "Get-CimInstance Win32_VideoController | Select-Object Name,DriverVersion,AdapterRAM | ConvertTo-Json"]
    )
    facts["virtualization"] = run(
        ["powershell", "-NoProfile", "-Command",
         "(Get-CimInstance Win32_ComputerSystem).HypervisorPresent"]
    )
    return facts


def chat(engine) -> dict:
    import httpx

    t0 = time.monotonic()
    r = httpx.post(
        engine.base_url + "/v1/chat/completions",
        headers = engine.headers,
        json = {
            "model": engine.model,
            "messages": [{"role": "user", "content": PROMPT}],
            "temperature": 0.0,
            "max_tokens": 32,
        },
        timeout = 600,
    )
    out = {"http": r.status_code, "gen_s": round(time.monotonic() - t0, 2)}
    try:
        body = r.json()
        out["text"] = body["choices"][0]["message"]["content"]
        out["usage"] = body.get("usage")
    except Exception:  # noqa: BLE001
        out["body"] = r.text[:2000]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True, type = Path)
    ap.add_argument("--out", required = True, type = Path)
    args = ap.parse_args()

    obs: dict = {"state": args.state, "host": host_facts()}
    backend = args.checkout / "studio" / "backend"
    sys.path.insert(0, str(backend))
    os.chdir(backend)
    try:
        from core.inference import engine_install, managed_engine, wsl_host
    except Exception as e:  # noqa: BLE001
        obs["import_error"] = f"{type(e).__name__}: {e}"
        obs["import_traceback"] = traceback.format_exc()[-3000:]
        args.out.write_text(json.dumps(obs, indent = 2, default = str), encoding = "utf-8")
        return 0

    def ask(fn, *a, **k):
        try:
            return fn(*a, **k)
        except Exception as e:  # noqa: BLE001
            return f"raised {type(e).__name__}: {e}"

    obs["wsl_active"] = ask(wsl_host.active)
    obs["gpu_platform"] = ask(engine_install.gpu_platform) if hasattr(engine_install, "gpu_platform") else None
    obs["vllm_reason"] = ask(engine_install.support_reason, "vllm")
    obs["sglang_reason"] = ask(engine_install.support_reason, "sglang")
    obs["download_bytes"] = ask(engine_install.download_bytes, "vllm")

    if obs["vllm_reason"] is None:
        t0 = time.monotonic()
        try:
            engine_install._install("vllm", threading.Event())
        except Exception as e:  # noqa: BLE001
            obs["install_error"] = f"{type(e).__name__}: {e}"[-4000:]
        obs["install_s"] = round(time.monotonic() - t0, 1)
        job = dict(engine_install._jobs.get("vllm") or {})
        obs["install_job"] = {k: job.get(k) for k in ("state", "phase", "message")}
        obs["install_log_tail"] = (job.get("log") or [])[-30:]
        # _install reports failure through its job record, not by raising.
        obs["install_ok"] = job.get("state") == "success" and "install_error" not in obs
        if job.get("state") == "error":
            obs.setdefault("install_error", job.get("message"))
        info = engine_install.installed("vllm")
        obs["installed_info"] = (
            {k: info.get(k) for k in ("version", "platform", "host", "python", "directory")} if info else None
        )
        if info:
            engine = managed_engine.ManagedEngine("vllm")
            gen = {"precision": "auto"}
            t1 = time.monotonic()
            try:
                class _Load:
                    gpu_ids = None
                    model_path = MODEL
                    gguf_variant = None
                    engine_precision = "auto"
                    engine_parallelism = "tensor"

                gpu_ids = managed_engine.validate_load("vllm", _Load())
                gen["gpu_ids"] = gpu_ids
                engine.start(MODEL, 2048, gpu_ids, dict(os.environ), None, {"precision": "auto"})
                gen["load_s"] = round(time.monotonic() - t1, 1)
                gen.update(chat(engine))
            except Exception as e:  # noqa: BLE001
                gen["error"] = f"{type(e).__name__}: {e}"
                gen["traceback"] = traceback.format_exc()[-3000:]
                gen["engine_tail"] = list(getattr(engine, "_tail", []))[-60:]
            finally:
                try:
                    engine.stop()
                except Exception as e:  # noqa: BLE001
                    gen["stop_error"] = f"{type(e).__name__}: {e}"
            obs["generations"] = [gen]

    args.out.write_text(json.dumps(obs, indent = 2, default = str), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
