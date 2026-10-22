#!/usr/bin/env python3
"""Probe: which vision projector does THIS checkout pick on a ROCm llama.cpp install, and does
the bundled llama-server read images correctly with it? (unslothai/unsloth#14679)

The checkout's own selection code is executed: the projector ranking helpers are lifted from
studio/backend/utils/models/model_config.py and core/inference/llama_cpp.py (no backend imports,
so the runner needs no Studio venv), with the installed-backend reader answering what the ROCm
bundle's asset name records. A checkout without the helpers is the old F16-first rule.

Then each model is served on the GPU with that projector (Studio's flags) and asked to read a
rendered number from 3 fresh images; answers and the server's image-encode times are recorded.

Observes only; criteria/mmproj_precision.py judges. Stdlib only. JSON to --out, never stdout.
"""

from __future__ import annotations

import argparse
import ast
import base64
import json
import os
import re
import shutil
import socket
import struct
import subprocess
import sys
import tarfile
import time
import types
import urllib.request
import zlib
from pathlib import Path

MODELS = [
    ("mimo_vl_7b", "unsloth/MiMo-VL-7B-RL-GGUF", "MiMo-VL-7B-RL-UD-Q4_K_XL.gguf"),
    ("qwen25vl_7b", "unsloth/Qwen2.5-VL-7B-Instruct-GGUF", "Qwen2.5-VL-7B-Instruct-UD-Q4_K_XL.gguf"),
    ("qwen3vl_4b", "unsloth/Qwen3-VL-4B-Instruct-GGUF", "Qwen3-VL-4B-Instruct-UD-Q4_K_XL.gguf"),
    ("qwen35_4b", "unsloth/Qwen3.5-4B-GGUF", "Qwen3.5-4B-UD-Q4_K_XL.gguf"),
    ("gemma3_4b", "unsloth/gemma-3-4b-it-GGUF", "gemma-3-4b-it-UD-Q4_K_XL.gguf"),
    ("gemma4_e2b", "unsloth/gemma-4-E2B-it-GGUF", "gemma-4-E2B-it-UD-Q4_K_XL.gguf"),
]
LISTING = ["mmproj-BF16.gguf", "mmproj-F16.gguf", "mmproj-F32.gguf"]
PROMPT = "What number is written in this image? Reply with the number only."
ENC_RE = re.compile(r"image slice encoded in (\d+) ms")

# 5x7 digit glyphs, drawn without PIL (stdlib only on the runner).
GLYPHS = {
    "0": "01110100011001110101110011000101110", "1": "00100011000010000100001000010001110",
    "2": "01110100010000100010001000100011111", "3": "11111000100010000010000011000101110",
    "4": "00010001100101010010111110001000010", "5": "11111100001111000001000011000101110",
    "6": "00110010001000011110100011000101110", "7": "11111000010001000100010000100001000",
    "8": "01110100011000101110100011000101110", "9": "01110100011000101111000010001001100",
}


def png_number(text: str, scale: int = 18) -> str:
    w, h = (len(text) * 6 + 2) * scale, 11 * scale
    rows = []
    for y in range(h):
        row = bytearray(b"\xff" * w)
        gy = y // scale - 2
        if 0 <= gy < 7:
            for i, ch in enumerate(text):
                bits = GLYPHS[ch][gy * 5:(gy + 1) * 5]
                for gx, b in enumerate(bits):
                    if b == "1":
                        x0 = (1 + i * 6 + gx) * scale
                        row[x0:x0 + scale] = b"\x00" * scale
        rows.append(b"\x00" + bytes(row))
    raw = zlib.compress(b"".join(rows), 9)

    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)

    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0))
           + chunk(b"IDAT", raw) + chunk(b"IEND", b""))
    return "data:image/png;base64," + base64.b64encode(png).decode()


def lift(path: Path, names: set) -> str:
    src = path.read_text(encoding = "utf-8")
    tree = ast.parse(src)
    out = []
    for node in tree.body:
        targets = set()
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            targets = {node.name}
        elif isinstance(node, ast.Assign):
            targets = {t.id for t in node.targets if isinstance(t, ast.Name)}
        if targets & names:
            out.append(ast.get_source_segment(src, node))
    return "\n\n".join(out)


def selection(checkout: Path, backend: str) -> dict:
    """{model name: picked projector} using the checkout's own code."""
    be = checkout / "studio" / "backend"
    mc = lift(be / "utils" / "models" / "model_config.py",
              {"_F16_UNSAFE_MMPROJ_RE", "_MMPROJ_PRECISION_RE", "MMPROJ_ORDER_BF16", "MMPROJ_ORDER_NO_F16",
               "mmproj_f16_unsafe", "device_mmproj_precision", "mmproj_order", "mmproj_order_rank"})
    pick = lift(be / "core" / "inference" / "llama_cpp.py", {"_pick_mmproj"})
    # Fake modules the lifted code imports lazily; the backend answer is what the bundle records.
    fake_hw = types.ModuleType("utils.hardware.hardware")
    fake_hw._installed_llama_backend = lambda: backend
    fake_gguf = types.ModuleType("hub.utils.gguf")
    fake_gguf.drop_shadowed_appledouble_names = lambda names: names
    ns_mc = {"__name__": "lifted_model_config"}
    fake_mc = types.ModuleType("utils.models.model_config")
    for name, mod in (("utils", types.ModuleType("utils")), ("utils.hardware", types.ModuleType("utils.hardware")),
                      ("utils.hardware.hardware", fake_hw), ("utils.models", types.ModuleType("utils.models")),
                      ("utils.models.model_config", fake_mc), ("hub", types.ModuleType("hub")),
                      ("hub.utils", types.ModuleType("hub.utils")), ("hub.utils.gguf", fake_gguf)):
        sys.modules[name] = mod
    import functools
    import logging
    import platform
    ns_mc.update(_re = re, re = re, functools = functools, platform = platform, Path = Path,
                 Optional = __import__("typing").Optional, logger = logging.getLogger("lifted"))
    has_rule = "def mmproj_order" in mc
    if has_rule:
        exec(compile(mc, "model_config(lifted)", "exec"), ns_mc)
        for k, v in ns_mc.items():
            setattr(fake_mc, k, v)
    ns_pick = {"Path": Path, "Optional": __import__("typing").Optional}
    exec(compile(pick, "llama_cpp(lifted)", "exec"), ns_pick)
    out = {"has_device_rule": has_rule,
           "device_precision": ns_mc["device_mmproj_precision"]() if has_rule else None, "picks": {}}
    for name, repo, _ in MODELS:
        order = ns_mc["mmproj_order"](repo) if has_rule else None
        out["picks"][name] = ns_pick["_pick_mmproj"](LISTING, order = order) if order else ns_pick["_pick_mmproj"](LISTING)
    return out


def fetch(url: str, dest: Path) -> Path:
    if not dest.exists():
        dest.parent.mkdir(parents = True, exist_ok = True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        with urllib.request.urlopen(url, timeout = 1800) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f, 1 << 20)
        tmp.replace(dest)
    return dest


def bundle_dir(cache: Path, release: str, gfx: str) -> Path:
    out = cache / f"bundle_{gfx}"
    if (out / ".done").exists():
        return out
    arc = fetch(f"https://github.com/unslothai/llama.cpp/releases/download/{release}/"
                f"app-{release}-linux-x64-rocm-{gfx}.tar.gz", cache / f"bundle_{gfx}.tar.gz")
    shutil.rmtree(out, ignore_errors = True)
    out.mkdir(parents = True)
    with tarfile.open(arc) as t:
        t.extractall(out, **({"filter": "data"} if hasattr(tarfile, "data_filter") else {}))
    (out / ".done").write_text("", encoding = "utf-8")
    return out


def hf(repo: str, name: str, cache: Path) -> Path:
    return fetch(f"https://huggingface.co/{repo}/resolve/main/{name}", cache / repo.replace("/", "__") / name)


def post(url: str, payload: dict, timeout: int = 900) -> dict:
    req = urllib.request.Request(url, data = json.dumps(payload).encode(), headers = {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout = timeout) as r:
        return json.loads(r.read().decode())


def serve(bundle: Path, model: Path, mmproj: Path, log: Path, codes: list) -> dict:
    exe = next(p for p in bundle.rglob("llama-server") if p.is_file())
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    cmd = [str(exe), "-m", str(model), "--mmproj", str(mmproj), "--host", "127.0.0.1", "--port", str(port),
           "-c", "8192", "-ngl", "-1", "--parallel", "1", "--no-context-shift", "--jinja", "--flash-attn", "on",
           "--no-cache-prompt", "--fit", "off", "-lv", "4"]
    env = dict(os.environ, LD_LIBRARY_PATH = f"{exe.parent}:{os.environ.get('LD_LIBRARY_PATH', '')}")
    res: dict = {"answers": [], "ok": [], "ready": False}
    with open(log, "w", encoding = "utf-8") as lf:
        proc = subprocess.Popen(cmd, stdout = lf, stderr = subprocess.STDOUT, env = env, cwd = str(exe.parent))
        base = f"http://127.0.0.1:{port}"
        try:
            for _ in range(1200):
                if proc.poll() is not None:
                    break
                try:
                    with urllib.request.urlopen(base + "/health", timeout = 2) as r:
                        if r.status == 200:
                            res["ready"] = True
                            break
                except Exception:  # noqa: BLE001
                    pass
                time.sleep(0.5)
            for n in codes if res["ready"] else []:
                body = {"messages": [{"role": "user", "content": [
                            {"type": "text", "text": PROMPT},
                            {"type": "image_url", "image_url": {"url": png_number(str(n))}}]}],
                        "max_tokens": 32, "temperature": 0, "top_k": 1, "seed": 3407,
                        "chat_template_kwargs": {"enable_thinking": False}}
                try:
                    msg = (post(base + "/v1/chat/completions", body).get("choices") or [{}])[0].get("message") or {}
                    ans = ((msg.get("content") or "") + (msg.get("reasoning_content") or ""))[-80:]
                except Exception as e:  # noqa: BLE001
                    ans = f"EXC {e!r}"[:120]
                res["answers"].append(ans)
                res["ok"].append(str(n) in ans)
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout = 20)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
    text = log.read_text(encoding = "utf-8", errors = "replace")
    res["encode_ms"] = [int(x) for x in ENC_RE.findall(text)]
    res["rocm_model_buffer"] = bool(re.search(r"ROCm\d+ model buffer size", text))
    res["exit_code"] = proc.returncode
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True, type = Path)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--release", default = "b11798-mix-f8a7950")
    ap.add_argument("--gfx", default = "gfx1151")
    args = ap.parse_args()
    root = Path(os.environ.get("AMD_CI_WORK") or args.out.parent)
    cache = root / "mmproj_cache"
    cache.mkdir(parents = True, exist_ok = True)
    asset = f"app-{args.release}-linux-x64-rocm-{args.gfx}.tar.gz"
    backend = "rocm" if "-rocm-" in asset else None
    obs: dict = {"state": args.state, "release": args.release, "asset": asset, "models": {}}
    try:
        obs["selection"] = selection(args.checkout, backend)
    except Exception as e:  # noqa: BLE001
        obs["selection_error"] = repr(e)[:600]
        args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
        return 0
    bundle = bundle_dir(cache, args.release, args.gfx)
    code = 4100 if args.state == "base" else 7300
    for name, repo, weight in MODELS:
        picked = obs["selection"]["picks"][name]
        row: dict = {"repo": repo, "mmproj": picked}
        try:
            model = hf(repo, weight, cache)
            mmproj = hf(repo, picked, cache)
            codes = [code + i * 37 for i in range(3)]
            code += 200
            row.update(serve(bundle, model, mmproj, args.out.parent / f"{args.state}_{name}.log", codes))
        except Exception as e:  # noqa: BLE001
            row["error"] = repr(e)[:400]
        obs["models"][name] = row
        args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
    args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
