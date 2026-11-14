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
def _lemonade_tail(n = 60):
    """Lines of the newest Lemonade logs that say why FastFlowLM stopped."""
    import re
    logs = sorted((b.root.parent / "logs" / "lemonade").glob("*.log"), key = lambda p: p.stat().st_mtime)[-2:]
    keep = re.compile(r"error|fail|abort|weight|not found|terminat|exception|flm|fastflowlm|q4nx|tensor", re.I)
    out = []
    for log in logs:
        out += [l for l in log.read_text(encoding = "utf-8", errors = "replace").splitlines() if keep.search(l)]
    return out[-n:]
def _q4k_variants(work):
    """Converter trees for the q4k path: the pin, the pin with the bf16 prefill copies written
    (ROCm/FLM_Q4NX_Converter#22), and #26's head (bf16 copies + Q4_K re-encode)."""
    import shutil, tarfile, urllib.request
    from pathlib import Path
    from core.export import q4nx
    from utils.paths.storage_roots import studio_root
    pinned = q4nx._installer().install(studio_root() / "q4nx_converter", name = "q4k")
    out = {"pinned": pinned}
    copy = work / "v_bf16_copies"
    if not copy.exists():
        shutil.copytree(pinned.parent, copy)
    src = copy / "q4nx" / "models" / "qwen35.py"
    text = src.read_text(encoding = "utf-8")
    old = ("# new_name = self.forward_name_map[gguf_tensor.name]", "# new_name = new_name.replace(", "# self.q4nx_tensors[new_name] = w  sdfsdf")
    for o in old:
        text = text.replace(o, o.replace("# ", "", 1).replace("  sdfsdf", ""))
    src.write_text(text, encoding = "utf-8")
    out["bf16_copies"] = copy / pinned.name
    if os.environ.get("NPU_Q4NX_ONLY_NEW") == "1":
        out.pop("bf16_copies")
    pr26 = work / "v_pr26"
    if not (pr26 / "convert.py").exists():
        arc = work / "pr26.tar.gz"
        urllib.request.urlretrieve("https://codeload.github.com/Eaumelette/FLM_Q4NX_Converter/tar.gz/ea1df7ec736fdf2d45dcbc99a79b6300c4301e07", arc)
        with tarfile.open(arc) as t:
            t.extractall(work / "pr26_x")
        shutil.move(str(next((work / "pr26_x").iterdir())), pr26)
    out["pr26"] = pr26 / "convert.py"
    # #26 switches only the 9B config to Q4_K; FastFlowLM's own Qwen3.5 files are all q4k.
    q4k = work / "v_pr26_q4k"
    if not q4k.exists():
        shutil.copytree(pr26, q4k)
        for cfg in (q4k / "configs").glob("qwen3.5_*.json"):
            text = cfg.read_text(encoding = "utf-8")
            cfg.write_text(text.replace('"default_tensor_type": "Q4_1"', '"default_tensor_type": "Q4_K"'), encoding = "utf-8")
    out["pr26_q4k"] = q4k / "convert.py"
    return out
def q4nx_roundtrip():
    """Stock Qwen3.5 GGUFs -> Q4NX through Studio's converter (its own pin; candidate trees too when
    NPU_Q4NX_VARIANTS=1), each swapped in for FastFlowLM's file, loaded and asked once on the NPU."""
    import glob, shutil
    from pathlib import Path
    from huggingface_hub import hf_hub_download
    from core.export import q4nx
    res = {}
    work = Path(os.environ["NPU_SHARED_MODELS"]).parent / "q4nx_e2e" / os.environ.get("NPU_STATE", "x")
    work.mkdir(parents = True, exist_ok = True)
    sizes = os.environ.get("NPU_Q4NX_SIZES", "0.8B,2B").split(",")
    for size in sizes:
        r = res.setdefault(size, {})
        gguf = hf_hub_download(f"unsloth/Qwen3.5-{size}-GGUF", f"Qwen3.5-{size}-Q4_K_M.gguf", token = False,
                               cache_dir = str(work.parent / "hf"))
        row = next((m for m in b.catalog() if m.checkpoint == f"qwen3.5:{size.lower()}"), None)
        if row is None:
            r["skipped"] = f"qwen3.5:{size.lower()} not in the catalog"
            continue
        for e in b.download(row.id):
            pass
        hits = [p for p in glob.glob(str(b.root / "flm" / "**" / "model.q4nx"), recursive = True)
                if f"Qwen3.5-{size}-" in p]
        if not hits:
            r["skipped"] = "official model.q4nx not found"
            continue
        official = hits[0]
        r["official_bytes"] = os.path.getsize(official)
        try:
            b.load(row.id)
            r["official_chat"] = chat()
            b.unload()
        except Exception as exc:
            r["official_error"] = f"{type(exc).__name__}: {exc}"[:800]
        variants = {"pinned": None}
        if os.environ.get("NPU_Q4NX_VARIANTS") == "1":
            try:
                variants.update(_q4k_variants(work))
            except Exception as exc:
                r["variants_error"] = f"{type(exc).__name__}: {exc}"[:800]
        inst = q4nx._installer()
        real_install = inst.install
        for name, script in variants.items():
            v = r.setdefault(name, {})
            if script is not None:
                inst.install = lambda root, name = "q4nx", _s = script, **kw: _s if name == "q4k" else real_install(root, name = name, **kw)
            try:
                out = q4nx.convert_existing_gguf(Path(gguf), f"unsloth/Qwen3.5-{size}", work / f"out_{size}_{name}")
            except Exception as exc:
                v["convert_error"] = f"{type(exc).__name__}: {exc}"[:1500]
                continue
            finally:
                inst.install = real_install
            v["converted_bytes"] = (out / "model.q4nx").stat().st_size
            shutil.move(official, official + ".bak")
            shutil.copyfile(out / "model.q4nx", official)
            try:
                try:
                    b.load(row.id)
                    v["chat"] = chat()
                except Exception as exc:
                    v["load_error"] = f"{type(exc).__name__}: {exc}"[:800]
                    v["lemonade_tail"] = _lemonade_tail()
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
