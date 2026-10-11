#!/usr/bin/env python3
"""Why do FP8 checkpoints fail under Studio's managed vLLM on gfx1151? Observes only.

Installs the engine through the head checkout's engine_install (the path Studio uses), then runs
each case in a fresh engine-python process: raw torch FP8 GEMM support, vLLM's own FP8 platform
answers, and real loads + one generation of FP8 checkpoints, with and without AITER.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

PROMPT = "What is the capital of France? Answer with one word."

TORCH_FACTS = r"""
import json, torch
out = {"torch": torch.__version__, "hip": getattr(torch.version, "hip", None)}
p = torch.cuda.get_device_properties(0)
out["arch"] = getattr(p, "gcnArchName", None)
a = torch.randn(64, 64, device="cuda")
for name in ("float8_e4m3fn", "float8_e4m3fnuz"):
    dt = getattr(torch, name, None)
    if dt is None:
        out[name] = "absent"; continue
    try:
        x = a.to(dt); w = a.to(dt).t()
        one = torch.ones((), device="cuda")
        torch._scaled_mm(x, w, scale_a=one, scale_b=one, out_dtype=torch.bfloat16)
        out[name] = "scaled_mm ok"
    except Exception as e:
        out[name] = f"{type(e).__name__}: {str(e)[:300]}"
    try:
        (a.to(dt).to(torch.bfloat16) @ a.to(torch.bfloat16))
        out[name + "_cast"] = "cast+bf16 matmul ok"
    except Exception as e:
        out[name + "_cast"] = f"{type(e).__name__}: {str(e)[:300]}"
try:
    from vllm.platforms import current_platform as cp
    out["vllm_supports_fp8"] = cp.supports_fp8()
    out["vllm_fp8_fnuz"] = cp.is_fp8_fnuz()
    out["vllm_fp8_dtype"] = str(cp.fp8_dtype())
except Exception as e:
    out["vllm_platform_error"] = f"{type(e).__name__}: {e}"
try:
    import aiter  # noqa: F401
    out["aiter_import"] = "ok"
except Exception as e:
    out["aiter_import"] = f"{type(e).__name__}: {str(e)[:300]}"
print("RESULT " + json.dumps(out))
"""

LOAD = r"""
import json, sys, time
from vllm import LLM, SamplingParams
model, mem = sys.argv[1], float(sys.argv[2])
t = time.monotonic()
llm = LLM(model=model, max_model_len=2048, enforce_eager=True, gpu_memory_utilization=mem,
          limit_mm_per_prompt={"image": 0, "video": 0} if "Qwen3.8" in model else None)
load = time.monotonic() - t
msgs = [{"role": "user", "content": sys.argv[3]}]
o = llm.chat(msgs, SamplingParams(max_tokens=24, temperature=0),
             chat_template_kwargs={"enable_thinking": False})
print("RESULT " + json.dumps({"load_s": round(load, 1), "answer": o[0].outputs[0].text}))
"""

AITER = {"VLLM_ROCM_USE_AITER": "1", "VLLM_ROCM_USE_AITER_LINEAR": "1"}
CASES = [
    ("unsloth/Qwen3.8-27B-FP8, default", "unsloth/Qwen3.8-27B-FP8", {}, 0.6, 2400),
    ("per-token FP8 0.5B, default", "RedHatAI/Qwen2.5-0.5B-Instruct-FP8-dynamic", {}, 0.3, 900),
    ("per-token FP8 0.5B, AITER on", "RedHatAI/Qwen2.5-0.5B-Instruct-FP8-dynamic", AITER, 0.3, 900),
    ("unquantized 0.5B, AITER on", "Qwen/Qwen2.5-0.5B-Instruct", AITER, 0.3, 1800),
    ("AWQ 0.5B, AITER on", "Qwen/Qwen2.5-0.5B-Instruct-AWQ", AITER, 0.3, 1800),
    ("unsloth/Qwen3.8-27B-FP8, AITER on", "unsloth/Qwen3.8-27B-FP8", AITER, 0.6, 3600),
    ("unsloth/Qwen3.8-27B-FP8, AITER on, warm", "unsloth/Qwen3.8-27B-FP8", AITER, 0.6, 3600),
]


LOG_DIR = None


def run(python: str, code: str, args: list[str], env: dict, timeout: int, label: str = "") -> dict:
    t = time.monotonic()
    try:
        p = subprocess.run([python, "-c", code, *args], env=env, capture_output=True, text=True,
                           timeout=timeout)
        text = p.stdout + p.stderr
        rc = p.returncode
    except subprocess.TimeoutExpired as e:
        text = (e.stdout or b"").decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
        rc = "timeout"
    res = {"rc": rc, "wall_s": round(time.monotonic() - t, 1)}
    if LOG_DIR is not None:
        name = "".join(ch if ch.isalnum() else "_" for ch in (label or "torch_facts"))
        (LOG_DIR / f"{name}.log").write_text(text, encoding="utf-8")
    import re as _re
    res["errors"] = sorted({_re.sub(r"^.*\[core.py:\d+\]\s*", "", l)[:400] for l in text.splitlines()
                            if _re.search(r"\b\w*(Error|Exception)\b: |requires|Selected \w+ for", l)})[:30]
    for line in text.splitlines():
        if line.startswith("RESULT "):
            res.update(json.loads(line[7:]))
    if "answer" not in res and "torch" not in res:
        lines = text.splitlines()
        reasons = [i for i, l in enumerate(lines) if "Reasons:" in l or "Error" in l]
        start = max(0, (reasons[0] - 3) if reasons else len(lines) - 60)
        res["tail"] = "\n".join(lines[start:start + 80])[-8000:]
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkout", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    global LOG_DIR
    LOG_DIR = a.out.parent / "fp8_logs"
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    obs: dict = {}
    home = os.environ.get("UNSLOTH_STUDIO_HOME", "")
    extra = [str(Path(sys.executable).parent), str(Path.home() / ".local" / "bin")]
    extra += [str(Path(home) / "bin")] if home else []
    os.environ["PATH"] = os.pathsep.join([*extra, os.environ.get("PATH", "")])
    backend = a.checkout / "studio" / "backend"
    sys.path.insert(0, str(backend))
    os.chdir(backend)
    from core.inference import engine_install

    obs["support_reason"] = engine_install.support_reason("vllm")
    t = time.monotonic()
    engine_install._install("vllm", threading.Event())
    job = dict(engine_install._jobs.get("vllm") or {})
    obs["install"] = {"state": job.get("state"), "s": round(time.monotonic() - t, 1)}
    info = engine_install.installed("vllm")
    if not info:
        obs["install_log_tail"] = (job.get("log") or [])[-30:]
        a.out.write_text(json.dumps(obs, indent=2), encoding="utf-8")
        return 0
    python = str(Path(info["path"]) / "bin" / "python")
    env = {k: v for k, v in os.environ.items() if k not in ("LD_LIBRARY_PATH", "VIRTUAL_ENV", "PYTHONPATH")}
    env.update({"HIP_VISIBLE_DEVICES": "0", "PYTHONNOUSERSITE": "1",
                "PATH": os.pathsep.join([str(Path(info["path"]) / "bin"), env.get("PATH", "")])})
    obs["torch_facts"] = run(python, TORCH_FACTS, [], env, 600)
    a.out.write_text(json.dumps(obs, indent=2), encoding="utf-8")
    obs["cases"] = []
    for label, model, extra_env, mem, timeout in CASES:
        r = run(python, LOAD, [model, str(mem), PROMPT], {**env, **extra_env}, timeout, label)
        r.update({"label": label, "model": model, "env": extra_env})
        obs["cases"].append(r)
        print(json.dumps({k: v for k, v in r.items() if k != "tail"}), flush=True)
        a.out.write_text(json.dumps(obs, indent=2), encoding="utf-8")
        time.sleep(20)
    return 0


if __name__ == "__main__":
    sys.exit(main())
