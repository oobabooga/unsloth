"""Probe: Studio chat speed on the gfx1151 iGPU (llama.cpp GGUF) vs the XDNA2 NPU (FastFlowLM).

Observes only: drives an installed Studio over its own HTTP API and writes JSON to --out.
Devices are interleaved, never concurrent (they share the APU's memory bandwidth).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import secrets
import shutil
import socket
import statistics
import subprocess
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import quote

PROMPT_PARAGRAPH = (
    "The history of the printing press begins in the fifteenth century, when movable metal type "
    "made it possible to reproduce texts quickly and cheaply. Before that, books were copied by "
    "hand, which made them rare and expensive. The spread of printing changed how knowledge moved "
    "across Europe, helped standardize languages, and supported the growth of science and "
    "literacy. "
)
PROMPT = (PROMPT_PARAGRAPH * 7) + (
    "Write a detailed, multi-paragraph essay that continues this history up to the present day, "
    "covering newspapers, the telegraph, radio, television and the internet."
)
MAX_TOKENS = 256
REPEATS = 5
CTX = 4096


def sh(cmd: str, timeout: int = 60) -> str:
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return (r.stdout + r.stderr)[-6000:]
    except Exception as exc:  # noqa: BLE001
        return f"ERROR {exc}"


def detect() -> dict:
    return {
        "dev_accel": sorted(glob.glob("/dev/accel/accel*")),
        "sys_class_accel": sh("ls -la /sys/class/accel 2>&1; for d in /sys/class/accel/*; do echo $d; cat $d/device/vendor $d/device/device 2>/dev/null; done"),
        "lsmod_amdxdna": sh("lsmod | grep -i amdxdna || echo none"),
        "xrt_smi": sh("command -v xrt-smi && xrt-smi examine 2>&1 | head -40 || echo 'xrt-smi absent'"),
        "lspci_npu": sh("lspci -nn 2>/dev/null | grep -iE 'signal processing|npu|1022:17f0|1022:1502' || echo none"),
        "cpu": sh("grep -m1 'model name' /proc/cpuinfo"),
        "rocminfo_gfx": sh("rocminfo 2>/dev/null | grep -E 'Marketing Name|gfx' | head -10 || echo none"),
        "amd_smi": sh("amd-smi static --asic 2>/dev/null | head -20 || echo none"),
        "kernel": sh("uname -a"),
        "meminfo": sh("grep -E 'MemTotal|MemAvailable' /proc/meminfo"),
    }


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class Studio:
    def __init__(self, base: str):
        self.base = base
        self.token = None

    def req(self, method, path, payload=None, timeout=600, raw=False):
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        data = None if payload is None else json.dumps(payload).encode()
        r = urllib.request.Request(self.base + path, data=data, headers=headers, method=method)
        resp = urllib.request.urlopen(r, timeout=timeout)
        if raw:
            return resp
        with resp:
            body = resp.read()
        return json.loads(body or b"{}")


def http_error_text(exc: Exception) -> str:
    if isinstance(exc, urllib.error.HTTPError):
        try:
            return f"HTTP {exc.code}: {exc.read().decode()[:1500]}"
        except Exception:  # noqa: BLE001
            return f"HTTP {exc.code}"
    return f"{type(exc).__name__}: {exc}"


def login(st: Studio, home: Path) -> None:
    boot = (home / "auth" / ".bootstrap_password").read_text().strip()
    tok = st.req("POST", "/api/auth/login", {"username": "unsloth", "password": boot})
    st.token = tok["access_token"]
    new = "Bench-" + secrets.token_hex(8)
    tok = st.req("POST", "/api/auth/change-password", {"current_password": boot, "new_password": new})
    st.token = tok["access_token"]


def stream_chat(st: Studio, model: str, max_tokens: int = MAX_TOKENS, nonce: str = "") -> dict:
    # A distinct first line per request: a shared prefix would be a prompt-cache hit.
    content = f"Request {nonce}.\n" + PROMPT if max_tokens == MAX_TOKENS else "Say hi."
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "stream": True,
        "max_tokens": max_tokens,
        "temperature": 0,
        "enable_thinking": False,
    }
    t0 = time.perf_counter()
    first = None
    timings = usage = finish = None
    n_text = 0
    resp = st.req("POST", "/v1/chat/completions", payload, timeout=900, raw=True)
    with resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body == "[DONE]":
                break
            try:
                chunk = json.loads(body)
            except ValueError:
                continue
            if chunk.get("timings"):
                timings = chunk["timings"]
            if chunk.get("usage"):
                usage = chunk["usage"]
            for ch in chunk.get("choices") or []:
                delta = ch.get("delta") or {}
                text = (delta.get("content") or "") + (delta.get("reasoning_content") or "")
                if text:
                    n_text += 1
                    if first is None:
                        first = time.perf_counter()
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
    t1 = time.perf_counter()
    out = {
        "wall_s": t1 - t0,
        "client_ttft_ms": None if first is None else (first - t0) * 1000,
        "text_chunks": n_text,
        "finish_reason": finish,
        "timings": timings,
        "usage": usage,
    }
    if timings:
        out["prefill_tps"] = timings.get("prompt_per_second")
        out["decode_tps"] = timings.get("predicted_per_second")
        out["prompt_n"] = timings.get("prompt_n")
        out["predicted_n"] = timings.get("predicted_n")
        out["server_ttft_ms"] = timings.get("prompt_ms")
    if first is not None and timings and timings.get("predicted_n"):
        decode_wall = t1 - first
        if decode_wall > 0:
            out["client_decode_tps"] = (timings["predicted_n"] - 1) / decode_wall
    return out


def load(st: Studio, model_path: str, variant=None) -> dict:
    payload = {"model_path": model_path, "max_seq_length": CTX}
    if variant:
        payload["gguf_variant"] = variant
    t0 = time.perf_counter()
    body = st.req("POST", "/api/inference/load", payload, timeout=3600)
    return {"load_s": time.perf_counter() - t0, "response": {k: body.get(k) for k in list(body)[:25]}}


def npu_download(st: Studio, model_id: str, log: list) -> None:
    resp = st.req("POST", f"/api/npu/models/{quote(model_id, safe='')}/download", None, timeout=7200, raw=True)
    done = False
    with resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            try:
                ev = json.loads(line[5:])
            except ValueError:
                continue
            if isinstance(ev, dict):
                if ev.get("event") in ("error",) or ev.get("error"):
                    raise RuntimeError(f"download {model_id}: {ev}")
                if ev.get("event") == "complete":
                    done = True
    log.append(f"downloaded {model_id} complete={done}")
    if not done:
        raise RuntimeError(f"download of {model_id} ended early")


def summarize(runs: list) -> dict:
    def stat(key):
        vals = [r[key] for r in runs if isinstance(r.get(key), (int, float))]
        if not vals:
            return None
        return {"median": statistics.median(vals), "min": min(vals), "max": max(vals), "n": len(vals)}

    return {k: stat(k) for k in ("server_ttft_ms", "client_ttft_ms", "prefill_tps", "decode_tps", "client_decode_tps", "wall_s")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--home", required=True)
    args = ap.parse_args()
    out_path = Path(args.out)
    home = Path(args.home)
    result: dict = {"detect": detect(), "steps": [], "cells": {}, "errors": []}

    def flush():
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")

    flush()
    unsloth = next((str(p) for p in [home / "bin" / "unsloth", *sorted(home.glob(".venv_t5_*/bin/unsloth")), home / "unsloth_studio" / "bin" / "unsloth"] if p.exists()), None)
    if unsloth is None:
        result["errors"].append(f"no unsloth binary under {home}")
        flush()
        return 1
    port = free_port()
    log_path = out_path.parent / "studio_server.log"
    result["studio_log"] = str(log_path)
    logf = open(log_path, "w")
    proc = subprocess.Popen([unsloth, "studio", "-p", str(port)], stdout=logf, stderr=subprocess.STDOUT, preexec_fn=os.setsid, env={**os.environ, "UNSLOTH_STUDIO_HOME": str(home)})
    st = Studio(f"http://127.0.0.1:{port}")
    try:
        deadline = time.time() + 600
        while time.time() < deadline:
            try:
                urllib.request.urlopen(st.base + "/api/health", timeout=5).read()
                break
            except Exception:  # noqa: BLE001
                if proc.poll() is not None:
                    raise RuntimeError(f"studio exited {proc.returncode}")
                time.sleep(2)
        login(st, home)
        result["steps"].append("logged in")
        flush()

        # NPU side setup
        npu_models: dict = {}
        try:
            result["npu_status_before"] = st.req("GET", "/api/npu/status", timeout=120)
            flush()
            status = result["npu_status_before"]
            if status.get("supported") and not status.get("ready"):
                status = st.req("POST", "/api/npu/enable", None, timeout=1800)
            result["npu_status_after_enable"] = status
            flush()
            if status.get("ready"):
                listing = st.req("GET", "/api/npu/models", timeout=300).get("models") or []
                result["npu_catalog"] = [{k: m.get(k) for k in ("id", "size_gb", "downloaded", "vision", "tools")} for m in listing]
                ids = {m.get("id"): m for m in listing if isinstance(m, dict)}
                small = next((i for i in ids if re.fullmatch(r"qwen3-0\.6b-flm", i or "", re.I)), None)
                big = None
                for pat in (r"qwen3-4b-flm", r"qwen3-8b-flm", r"qwen3-1\.7b-flm"):
                    big = next((i for i in ids if re.fullmatch(pat, i or "", re.I)), None)
                    if big:
                        break
                for size, mid in (("0.6B", small), ("big", big)):
                    if not mid:
                        continue
                    if not ids[mid].get("downloaded"):
                        npu_download(st, mid, result["steps"])
                    npu_models[size] = mid
                flush()
        except Exception as exc:  # noqa: BLE001
            result["errors"].append("npu setup: " + http_error_text(exc))
            result["npu_traceback"] = traceback.format_exc()[-3000:]
            flush()

        big_size = None
        if "big" in npu_models:
            big_size = re.search(r"qwen3-([\d.]+b)", npu_models["big"], re.I).group(1).upper()
        sizes = [("0.6B", "unsloth/Qwen3-0.6B-GGUF")]
        sizes.append((big_size or "4B", f"unsloth/Qwen3-{big_size or '4B'}-GGUF"))
        for size_label, gguf_repo in sizes:
            npu_id = npu_models.get("0.6B" if size_label == "0.6B" else "big")
            arms = [("gpu", gguf_repo, "Q4_K_M")]
            if npu_id:
                arms.insert(0, ("npu", f"lemonade:{npu_id}", None))
            for device, model_path, variant in arms:
                result["cells"].setdefault(f"{size_label}|{device}", {"model": model_path, "variant": variant, "runs": [], "warmup": None})
            # Interleave: load each arm in turn per repeat (a load per run keeps them strictly
            # sequential and the measurement order alternating).
            for rep in range(REPEATS + 1):
                for device, model_path, variant in arms:
                    cell = result["cells"][f"{size_label}|{device}"]
                    try:
                        info = load(st, model_path, variant)
                        if rep == 0:
                            cell["first_load"] = info
                        cell.setdefault("mem_after_load", sh("grep MemAvailable /proc/meminfo"))
                        # Untimed: the first request after a load pays one-off setup on both devices.
                        stream_chat(st, model_path, max_tokens = 8)
                        run = stream_chat(st, model_path, nonce = f"{secrets.token_hex(4)}-{rep}-{device}")
                        if rep == 0:
                            cell["warmup"] = run
                        else:
                            cell["runs"].append(run)
                    except Exception as exc:  # noqa: BLE001
                        cell.setdefault("errors", []).append(f"rep {rep}: " + http_error_text(exc))
                    cell["summary"] = summarize(cell["runs"])
                    flush()
        result["versions"] = {
            "npu": (result.get("npu_status_after_enable") or {}).get("versions"),
            "llama_cpp": sh(f"ls {home}/llama.cpp 2>/dev/null | head; cat {home}/llama.cpp/*.json 2>/dev/null | head -c 1500"),
        }
        flush()
    except Exception as exc:  # noqa: BLE001
        result["errors"].append("fatal: " + http_error_text(exc))
        result["traceback"] = traceback.format_exc()[-4000:]
        flush()
    finally:
        try:
            os.killpg(os.getpgid(proc.pid), 15)
        except Exception:  # noqa: BLE001
            pass
        logf.close()
        try:
            shutil.copy(log_path, out_path.parent / "studio_server_copy.log")
        except Exception:  # noqa: BLE001
            pass
        flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
