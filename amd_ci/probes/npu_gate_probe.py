#!/usr/bin/env python3
"""Probe: drive Studio's AMD NPU backend on this host and record what each step did.

Observes only (criteria/npu_gate.py judges). Steps: detection, the UNSLOTH_STUDIO_ENABLE_NPU=0
off switch, runtime enable, catalog, display-name resolution (if the state has it), download,
load, one greedy chat, converted-model listing (if the state has it), unload.
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

_WORKER = r"""
import json, os, sys, time, traceback, urllib.request
sys.path.insert(0, os.getcwd())
obs = {"steps": {}}
def step(name, fn):
    t = time.time()
    try:
        value = fn()
        obs["steps"][name] = {"ok": True, "s": round(time.time() - t, 1), "value": value}
        return value
    except Exception as exc:
        obs["steps"][name] = {"ok": False, "s": round(time.time() - t, 1),
                              "error": f"{type(exc).__name__}: {exc}"[:600],
                              "trace": traceback.format_exc()[-1500:]}
        return None
from utils.hardware.npu import detect_amd_npu
hw = step("detect", detect_amd_npu)
if os.environ.get("NPU_PROBE_DETECT_ONLY"):
    print("NPU_PROBE_JSON " + json.dumps(obs, default=str)); sys.exit(0)
from core.inference import npu_backend as nb
b = nb.get_npu_backend()
step("status_before", lambda: {k: v for k, v in b.status().items() if k in ("supported", "ready", "state", "error")})
step("enable", lambda: {k: v for k, v in b.enable().items() if k in ("supported", "ready", "state", "error", "versions")})
cat = step("catalog", lambda: [{"id": m.id, "hf_repo": m.hf_repo, "size_gb": m.size_gb, "downloaded": m.downloaded,
                                "local": getattr(m, "local", None)} for m in b.catalog()])
obs["has_display_names"] = hasattr(b, "model_path_for_name")
obs["has_local_q4nx"] = hasattr(b, "_local_rows")
want = os.environ.get("NPU_PROBE_MODEL", "qwen3-0.6b-FLM")
row = next((m for m in cat or [] if m["id"] == want), None)
if row is not None:
    if not row["downloaded"]:
        step("download", lambda: sum(1 for _ in b.download(want)))
    step("load", lambda: b.load(want, context_length=4096).id)
    def chat():
        up = b.upstream()
        body = json.dumps({"model": up.model, "temperature": 0, "max_tokens": 24,
                           "messages": [{"role": "user", "content": "Name the capital of France in one word."}]}).encode()
        req = urllib.request.Request(up.base_url + "/chat/completions", data=body,
                                     headers={"Content-Type": "application/json", "Authorization": f"Bearer {up.api_key}"})
        with urllib.request.urlopen(req, timeout=300) as r:
            return json.loads(r.read())["choices"][0]["message"]["content"]
    step("chat", chat)
    if obs["has_display_names"] and row.get("hf_repo"):
        name = row["hf_repo"].rsplit("/", 1)[-1]
        step("resolve_display_name", lambda: [b.model_path_for_name(name, start_runtime=False),
                                              b.model_path_for_name(name.upper(), start_runtime=False),
                                              b.model_path_for_name("Qwen3-9B-Nope-NPU2", start_runtime=False),
                                              b.model_path_for_name("unsloth/Qwen3-0.6B-GGUF", start_runtime=False)])
    step("unload", lambda: b.unload())
else:
    obs["steps"]["load"] = {"ok": False, "error": f"{want} not in catalog"}
step("shutdown", lambda: b.shutdown())
print("NPU_PROBE_JSON " + json.dumps(obs, default=str))
"""


def _run(python: str, workdir: Path, env: dict, timeout: int) -> dict:
    try:
        proc = subprocess.run([python, "-c", _WORKER], cwd = str(workdir), env = env,
                              capture_output = True, text = True, encoding = "utf-8",
                              errors = "replace", timeout = timeout)
    except subprocess.TimeoutExpired:
        return {"error": f"worker timed out after {timeout}s"}
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith("NPU_PROBE_JSON "):
            out = json.loads(line[len("NPU_PROBE_JSON "):])
            out["rc"] = proc.returncode
            return out
    return {"error": "worker printed no result", "rc": proc.returncode,
            "stderr_tail": proc.stderr[-2000:], "stdout_tail": proc.stdout[-1000:]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--python", default = sys.executable)
    ap.add_argument("--timeout", type = int, default = 2400)
    args = ap.parse_args()

    workdir = Path(args.checkout) / "studio" / "backend"
    obs: dict = {"state": args.state}
    try:
        # The Windows job's venv has only pytest; the Linux one is the Studio venv.
        probe = subprocess.run([args.python, "-c", "import jwt, structlog, fastapi"], capture_output = True)
        if probe.returncode != 0:
            reqs = workdir / "requirements" / "studio.txt"
            inst = subprocess.run([args.python, "-m", "pip", "install", "-q", "-r", str(reqs),
                                   "structlog", "psutil", "httpx", "sqlalchemy", "cryptography",
                                   "python-multipart", "aiofiles", "pyyaml", "requests"],
                                  capture_output = True, text = True, encoding = "utf-8", errors = "replace")
            obs["deps_install"] = {"rc": inst.returncode, "tail": inst.stderr[-1500:]}
        base_home = Path(os.environ.get("UNSLOTH_STUDIO_HOME") or (Path(args.checkout).parent / "studio"))
        env = dict(os.environ)
        # One Studio home per state: the base never sees the head's registrations.
        env["UNSLOTH_STUDIO_HOME"] = str(base_home.parent / f"npu_home_{args.state}")
        env.pop("UNSLOTH_STUDIO_ENABLE_NPU", None)
        obs["enabled"] = _run(args.python, workdir, env, args.timeout)
        off = dict(env, UNSLOTH_STUDIO_ENABLE_NPU = "0", NPU_PROBE_DETECT_ONLY = "1")
        obs["disabled"] = _run(args.python, workdir, off, 300)
    except Exception:  # noqa: BLE001 -- a probe records, never raises
        obs["error"] = traceback.format_exc()[-2000:]
    obs["finished_at"] = time.time()
    args.out.write_text(json.dumps(obs, indent = 2, default = str), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
