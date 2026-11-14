#!/usr/bin/env python3
"""Probe: drive Unsloth Studio's NPU backend on the runner's real XDNA2 NPU. Observes only.

Runs inside <checkout>/studio/backend with the Studio venv: enable (validate) the Lemonade runtime,
list the catalog, download a small model, load it on Auto and chat once, load it with a pin above
FastFlowLM's 32k card limit, and, where the checkout's load() takes flm_args, load with FastFlowLM
settings. Each step records ok / value / error; nothing is judged here. Models are kept in a
directory shared by every state so the second state does not download again.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

INNER = r'''
import inspect, json, os, sys, time, traceback
import httpx
out = {"steps": {}}
def step(name, fn):
    t = time.monotonic()
    try:
        value = fn()
        out["steps"][name] = {"ok": True, "value": value, "s": round(time.monotonic() - t, 1)}
        return value
    except BaseException as exc:
        out["steps"][name] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:2000],
                              "s": round(time.monotonic() - t, 1)}
        return None
from utils.hardware.npu import detect_amd_npu
from core.inference import npu_backend as nb
step("detect", detect_amd_npu)
b = nb.get_npu_backend()
os.makedirs(os.environ["NPU_SHARED_MODELS"], exist_ok = True)
flm = b.root / "flm"
if not flm.exists():
    flm.parent.mkdir(parents = True, exist_ok = True)
    try:
        flm.symlink_to(os.environ["NPU_SHARED_MODELS"], target_is_directory = True)
    except OSError as exc:
        out["shared_models"] = f"not shared: {exc}"
status = step("enable", lambda: {k: v for k, v in b.enable().items() if k in ("state", "error", "validation", "versions", "memory_window_gb")})
MODEL = os.environ.get("NPU_MODEL", "qwen3-0.6b-FLM")
def catalog():
    rows = [m.to_json() for m in b.catalog()]
    pick = [r for r in rows if r.get("id") == MODEL]
    return {"n": len(rows), "keys": sorted(pick[0].keys()) if pick else None, "model": pick[0] if pick else None}
step("catalog", lambda: json.loads(json.dumps(catalog(), default = str)))
def download():
    events = {}
    for e in b.download(MODEL):
        events[e.get("event", "?")] = events.get(e.get("event", "?"), 0) + 1
    return events
step("download", download)
def chat():
    r = b.resident()
    resp = httpx.post(f"{r.base_url}/v1/chat/completions", headers = {"Authorization": f"Bearer {r.api_key}"},
                      json = {"model": r.model.id, "messages": [{"role": "user", "content": "Say hello in one word."}],
                              "max_tokens": 16, "stream": False}, timeout = 300)
    body = resp.json()
    return {"status": resp.status_code, "text": (body.get("choices") or [{}])[0].get("message", {}).get("content", str(body))[:200]}
def load(ctx = None, **kw):
    b.load(MODEL, ctx, **kw)
    r = b.resident()
    return {"context_length": r.context_length, "requested": r.requested_context_length,
            "fitted": getattr(r, "fitted_context_length", "absent"),
            "flm_args": getattr(getattr(b, "_loaded", None), "flm_args", "absent")}
step("load_auto", load)
step("chat_auto", chat)
step("load_pin_40960", lambda: load(40960))
step("chat_after_pin", chat)
if "flm_args" in inspect.signature(b.load).parameters:
    step("load_flm_args", lambda: load(None, flm_args = "--pmode turbo --prefill-chunk-len 1024"))
    step("chat_flm_args", chat)
else:
    out["steps"]["load_flm_args"] = {"ok": None, "error": "load() takes no flm_args at this state"}
step("unload", b.unload)
def corrupt_load():
    import glob, shutil
    hits = [p for p in glob.glob(str(b.root / "flm" / "**" / "model.q4nx"), recursive = True) if "Qwen3-0.6B" in p]
    if not hits:
        return {"skipped": "model.q4nx not found"}
    path = hits[0]
    backup = path + ".bak"
    shutil.move(path, backup)
    try:
        try:
            b.load(MODEL)
            return {"loaded": True}
        except Exception as exc:
            return {"error_shown": f"{type(exc).__name__}: {exc}"[:1500]}
    finally:
        shutil.move(backup, path)
        try:
            b.unload()
        except Exception:
            pass
step("load_corrupt", corrupt_load)
def failing_load(prepare, restore, ctx = None):
    undo = prepare()
    try:
        try:
            b.load(MODEL, ctx)
            return {"loaded": True, "context_length": b.resident().context_length}
        except Exception as exc:
            return {"error_shown": f"{type(exc).__name__}: {exc}"[:1500]}
    finally:
        restore(undo)
        try:
            b.unload()
        except Exception:
            pass
step("load_huge_ctx", lambda: failing_load(lambda: None, lambda _: None, 1 << 22))
def _break_config():
    import glob
    hits = [p for p in glob.glob(str(b.root / "flm" / "**" / "config.json"), recursive = True) if "Qwen3-0.6B" in p]
    if not hits:
        return None
    text = open(hits[0], "rb").read()
    open(hits[0], "wb").write(text[: len(text) // 3])
    return hits[0], text
def _fix_config(undo):
    if undo:
        open(undo[0], "wb").write(undo[1])
step("load_bad_config", lambda: failing_load(_break_config, _fix_config))
if "cancel" in inspect.signature(b.download).parameters:
    import threading
    SECOND = os.environ.get("NPU_SECOND_MODEL", "qwen3-1.7b-FLM")
    def cancel_then_resume():
        cancel = threading.Event()
        seen = []
        try:
            for e in b.download(SECOND, cancel = cancel):
                seen.append(e.get("event"))
                if seen.count("progress") >= 5:
                    cancel.set()
            first = "completed without cancelling"
        except Exception as exc:
            first = f"{type(exc).__name__}: {exc}"[:300]
        resumed = {}
        for e in b.download(SECOND):
            resumed[e.get("event", "?")] = resumed.get(e.get("event", "?"), 0) + 1
        return {"first": first, "first_events": sorted(set(seen)), "resumed_events": resumed}
    step("download_cancel_resume", cancel_then_resume)
if hasattr(b, "model_folders"):
    step("model_folders", lambda: [[str(p.name), c] for p, c in b.model_folders()])
def q4nx_roundtrip():
    """Stock Qwen3.5-0.8B GGUF -> Q4NX through Studio's converter, swapped in for FastFlowLM's own
    file, then loaded and asked once on the NPU. Observes whether FastFlowLM can run the export."""
    import glob, shutil
    from pathlib import Path
    from huggingface_hub import hf_hub_download
    from core.export import q4nx
    res = {}
    work = Path(os.environ["NPU_SHARED_MODELS"]).parent / "q4nx_e2e" / os.environ.get("NPU_STATE", "x")
    gguf = hf_hub_download("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf", token = False,
                           cache_dir = str(work.parent / "hf"))
    try:
        out = q4nx.convert_existing_gguf(Path(gguf), "unsloth/Qwen3.5-0.8B", work / "out")
    except Exception as exc:
        return {"convert_error": f"{type(exc).__name__}: {exc}"[:1500]}
    res["files"] = sorted(p.name for p in out.iterdir())
    res["converted_bytes"] = (out / "model.q4nx").stat().st_size
    row = next((m for m in b.catalog() if m.checkpoint == "qwen3.5:0.8b"), None)
    if row is None:
        res["skipped"] = "qwen3.5:0.8b not in the catalog"
        return res
    for e in b.download(row.id):
        pass
    hits = [p for p in glob.glob(str(b.root / "flm" / "**" / "model.q4nx"), recursive = True) if "Qwen3.5-0.8B" in p]
    if not hits:
        res["skipped"] = "official Qwen3.5-0.8B model.q4nx not found"
        return res
    official = hits[0]
    res["official_bytes"] = os.path.getsize(official)
    try:
        b.load(row.id)
        res["official_chat"] = chat()
        b.unload()
    except Exception as exc:
        res["official_error"] = f"{type(exc).__name__}: {exc}"[:800]
    shutil.move(official, official + ".bak")
    shutil.copyfile(out / "model.q4nx", official)
    try:
        try:
            b.load(row.id)
            res["converted_chat"] = chat()
        except Exception as exc:
            res["converted_error"] = f"{type(exc).__name__}: {exc}"[:1500]
    finally:
        try:
            b.unload()
        except Exception:
            pass
        os.replace(official + ".bak", official)
    return res
if os.environ.get("NPU_Q4NX_E2E", "1") == "1":
    step("q4nx_roundtrip", q4nx_roundtrip)
def gating(home, absent):
    real_detect, real_get = nb.detect_amd_npu, nb.get_npu_backend
    if absent:
        nb.detect_amd_npu = lambda: {"present": False, "supported": False, "family": None, "name": None}
    try:
        probe = b if home is None else nb.LemonadeNpuBackend(b.root.parent / home)
        nb.get_npu_backend = lambda: probe
        st = probe.status()
        res = {k: st.get(k) for k in ("supported", "ready", "state", "runtime_installed")}
        try:
            from utils.q4nx_export_gate import q4nx_export_allowed
            res["q4nx_export_allowed"] = q4nx_export_allowed()
        except ImportError:
            res["q4nx_export_allowed"] = "absent"
        return res
    finally:
        nb.detect_amd_npu, nb.get_npu_backend = real_detect, real_get
step("gating_real", lambda: gating(None, False))
step("gating_npu_no_runtime", lambda: gating("npu_gate_fresh", False))
step("gating_no_npu", lambda: gating("npu_gate_absent", True))
b.shutdown()
log_dir = b.root.parent / "logs" / "lemonade"
out["lemonade_logs"] = sorted(p.name for p in log_dir.glob("*.log")) if log_dir.is_dir() else []
print("NPU_E2E_JSON " + json.dumps(out, default = str))
'''


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--python", default = sys.executable)
    ap.add_argument("--timeout", type = int, default = 3600)
    args = ap.parse_args()
    workdir = Path(args.checkout) / "studio" / "backend"
    work = Path(os.environ.get("AMD_CI_WORK", "/tmp"))
    env = dict(os.environ)
    env["UNSLOTH_STUDIO_HOME"] = str(work / f"npu_home_{args.state}")
    env["NPU_SHARED_MODELS"] = str(work / "npu_models")
    env["PYTHONPATH"] = str(workdir)
    env["NPU_STATE"] = args.state
    obs: dict = {"state": args.state}
    try:
        proc = subprocess.run([args.python, "-c", INNER], cwd = workdir, env = env, capture_output = True,
                              text = True, timeout = args.timeout)
        obs["rc"] = proc.returncode
        line = next((l for l in proc.stdout.splitlines() if l.startswith("NPU_E2E_JSON ")), None)
        if line:
            obs.update(json.loads(line[len("NPU_E2E_JSON "):]))
        else:
            obs["error"] = "no result line"
        obs["stderr_tail"] = proc.stderr[-3000:]
    except subprocess.TimeoutExpired as exc:
        obs["error"] = f"timeout after {args.timeout}s"
        obs["stderr_tail"] = (exc.stderr or b"")[-3000:].decode("utf-8", "replace") if isinstance(exc.stderr, bytes) else str(exc.stderr)[-3000:]
    args.out.parent.mkdir(parents = True, exist_ok = True)
    args.out.write_text(json.dumps(obs, indent = 2, default = str), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
