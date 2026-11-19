#!/usr/bin/env python3
"""Probe (#18001): the paged-load --fit-target a Windows Vulkan iGPU gets, and where llama.cpp's
fitter then places a small model. Observes only."""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

GGUF_URL = "https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q4_K_M.gguf"


def _cache() -> Path:
    root = Path(os.environ.get("RUNNER_TEMP") or os.environ.get("TEMP") or ".") / "i18001_cache"
    root.mkdir(parents = True, exist_ok = True)
    return root


def _fetch(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers = {"User-Agent": "amd-ci-probe"})
    with urllib.request.urlopen(req, timeout = 600) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    tmp.replace(dest)
    return dest


def _llama_vulkan(obs: dict) -> Path:
    root = _cache() / "llama_vulkan"
    exe = next(root.rglob("llama-server.exe"), None) if root.exists() else None
    if exe:
        return exe
    # The reporter's build (Settings > About: b12347-mix-241a0f3).
    obs["llama_release"] = "unslothai/llama.cpp b12347-mix-241a0f3"
    name = "app-b12347-mix-241a0f3-windows-x64-vulkan.zip"
    url = f"https://github.com/unslothai/llama.cpp/releases/download/b12347-mix-241a0f3/{name}"
    z = _fetch(url, _cache() / name)
    with zipfile.ZipFile(z) as zf:
        zf.extractall(root)
    return next(root.rglob("llama-server.exe"))


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _run_server(exe: Path, gguf: Path, fit_args: list[str], out_dir: Path, tag: str) -> dict:
    log = out_dir / f"llama_{tag}.log"
    help_text = subprocess.run([str(exe), "--help"], capture_output = True, text = True,
                               timeout = 60).stdout
    load_mode = ["--load-mode", "mmap"] if "--load-mode" in help_text else []
    argv = [str(exe), "-m", str(gguf), "--port", str(_free_port()), "-c", "4096",
            "--fit", "on", *fit_args, *load_mode, "-lv", "4"]
    res: dict = {"argv": argv[1:]}
    with open(log, "w", encoding = "utf-8") as f:
        p = subprocess.Popen(argv, stdout = f, stderr = subprocess.STDOUT)
    deadline = time.time() + 240
    text = ""
    while time.time() < deadline:
        time.sleep(2)
        text = log.read_text(encoding = "utf-8", errors = "replace")
        if "server is listening" in text or "exiting due to model loading error" in text \
                or p.poll() is not None:
            break
    if p.poll() is None:
        p.kill()
    p.wait(timeout = 30)
    text = log.read_text(encoding = "utf-8", errors = "replace")
    res["listening"] = "server is listening" in text
    res["alloc_failed"] = "ErrorOutOfDeviceMemory" in text or "failed to allocate" in text
    m = re.findall(r"- (Vulkan\d+) \(([^)]*)\): (\d+) layers,\s+(\d+) MiB used,\s+(\d+) MiB free", text)
    res["fit_result"] = [list(x) for x in m]
    res["offloaded"] = re.findall(r"offloaded (\d+)/(\d+) layers to GPU", text)
    res["device_info"] = re.findall(r"- (Vulkan\d+ : [^\n]*)", text)
    res["model_buffers"] = re.findall(r"load_tensors:\s+(\S+) model buffer size =\s+([\d.]+) MiB", text)
    res["log_tail"] = text[-1500:]
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True, type = Path)
    ap.add_argument("--out", required = True, type = Path)
    args = ap.parse_args()
    obs: dict = {"state": args.state, "platform": sys.platform}
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "httpx", "structlog"],
                       check = False, timeout = 600)
        exe = _llama_vulkan(obs)
        obs["binary"] = str(exe)
        backend = args.checkout / "studio" / "backend"
        sys.path.insert(0, str(backend))
        import core.inference.llama_cpp as m  # noqa: PLC0415
        B = m.LlamaCppBackend
        obs["module"] = m.__file__
        obs["raw_probe"] = B._run_vulkan_probe(str(exe))
        rows = B._get_gpu_free_memory_vulkan(str(exe))
        obs["rows"] = [list(r) for r in rows]
        igpu = {idx for idx, _f, total in rows if total <= 0}
        obs["igpu_ids"] = sorted(igpu)
        try:
            import utils.hardware.hardware as hw  # noqa: PLC0415
            recs = hw._windows_amd_adapter_records_by_luid(hw._AMD_PCI_VENDOR_ID,
                                                            distinguish_failure = True)
            obs["dxgi_amd_records"] = {str(k): v for k, v in (recs or {}).items()}
        except Exception as e:  # noqa: BLE001
            obs["dxgi_error"] = repr(e)
        obs["carve_out_bytes"] = B._igpu_dedicated_memory_bytes(
            sorted(igpu)[:1] or None, ordinals_are_vulkan = True)
        detected = [(idx, free) for idx, free, _t in rows]
        argv = ["--fit", "on", "--fit-target", "512"]
        if hasattr(B, "_paged_fit_headroom_mib"):
            h = B._paged_fit_headroom_mib(detected, igpu)
            flags = B._paged_fit_headroom_flags(argv, {}, [], gpu_count = len(detected),
                                               headroom_mib = h)
        else:
            h = m._IGPU_PAGED_FIT_HEADROOM_MIB
            flags = B._paged_fit_headroom_flags(argv, {}, [], gpu_count = len(detected))
        obs["headroom_mib"] = h
        obs["headroom_flags"] = flags
        # Studio's argv keeps its own margin first and appends the paging pair (last wins).
        fit_args = ["--fit-target", "512", *flags]
        obs["effective_fit_target"] = int(float(fit_args[-1]))
        gguf = _fetch(GGUF_URL, _cache() / "Qwen3-0.6B-Q4_K_M.gguf")
        obs["server"] = _run_server(exe, gguf, fit_args, args.out.parent, args.state)
    except Exception as e:  # noqa: BLE001
        import traceback
        obs["error"] = f"{type(e).__name__}: {e}"
        obs["traceback"] = traceback.format_exc()[-3000:]
    args.out.write_text(json.dumps(obs, indent = 2, default = str), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
