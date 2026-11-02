# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import traceback
import urllib.request
import zipfile


RELEASE = "b11985-mix-c116c32"


def make_model(path):
    import gguf
    import numpy as np

    rng = np.random.default_rng(3407)
    writer = gguf.GGUFWriter(path, "llama")
    writer.add_name("random-llama-startup-control")
    for name, value in (
        ("context_length", 8192), ("embedding_length", 64), ("block_count", 2),
        ("feed_forward_length", 128), ("head_count", 4), ("head_count_kv", 2),
        ("rope_dimension_count", 16),
    ):
        getattr(writer, "add_" + name)(value)
    writer.add_layer_norm_rms_eps(1e-5)
    writer.add_tokenizer_model("llama")
    writer.add_token_list(["<unk>", "<s>", "</s>"] + [f"<0x{i:02X}>" for i in range(256)])
    writer.add_token_types([2, 3, 3] + [6] * 256)
    writer.add_token_scores([0.0] * 259)
    writer.add_unk_token_id(0)
    writer.add_bos_token_id(1)
    writer.add_eos_token_id(2)
    tensors = {"token_embd.weight": (259, 64), "output_norm.weight": (64,), "output.weight": (259, 64)}
    for i in range(2):
        tensors.update({f"blk.{i}." + name: shape for name, shape in {
            "attn_norm.weight": (64,), "attn_q.weight": (64, 64),
            "attn_k.weight": (32, 64), "attn_v.weight": (32, 64),
            "attn_output.weight": (64, 64), "ffn_norm.weight": (64,),
            "ffn_gate.weight": (128, 64), "ffn_up.weight": (128, 64),
            "ffn_down.weight": (64, 128),
        }.items()})
    for name, shape in tensors.items():
        values = np.ones(shape, dtype=np.float32) if len(shape) == 1 else rng.normal(0, .02, shape).astype(np.float32)
        writer.add_tensor(name, values)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def binary(cache):
    folder = cache / "vulkan"
    matches = list(folder.glob("**/llama-server.exe")) if folder.exists() else []
    if not matches:
        archive = cache / "vulkan.zip"
        url = f"https://github.com/unslothai/llama.cpp/releases/download/{RELEASE}/app-{RELEASE}-windows-x64-vulkan.zip"
        with urllib.request.urlopen(url, timeout=180) as src, archive.open("wb") as dst:
            shutil.copyfileobj(src, dst)
        folder.mkdir()
        with zipfile.ZipFile(archive) as source:
            for entry in source.infolist():
                if not (folder / entry.filename).resolve().is_relative_to(folder.resolve()):
                    raise RuntimeError("archive member leaves the extraction directory")
            source.extractall(folder)
        matches = list(folder.glob("**/llama-server.exe"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one server, found {len(matches)}")
    return matches[0].resolve()


def request(url, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as response:
        return json.load(response)


def completion(url, headers=None):
    payload = {"prompt": "A", "n_predict": 8, "temperature": 0, "seed": 3407, "ignore_eos": True}
    req = urllib.request.Request(url + "/completion", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json", **(headers or {})})
    with urllib.request.urlopen(req, timeout=60) as response:
        result = json.load(response)
    return {k: result.get(k) for k in ("content", "tokens_predicted", "stop", "stopped_limit")}


def native(exe, model, mode, output):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    cmd = [str(exe), "-m", str(model), "--host", "127.0.0.1", "--port", str(port),
           "--fit", mode, "-c", "8192", "-np", "1", "-ngl", "999", "-lv", "4"]
    result = {"cmd": cmd, "ready": False}
    with output.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=exe.parent)
        try:
            deadline = time.monotonic() + 180
            while time.monotonic() < deadline and proc.poll() is None:
                try:
                    request(f"http://127.0.0.1:{port}/health")
                    result["ready"] = True
                    result["completion"] = completion(f"http://127.0.0.1:{port}")
                    break
                except (OSError, ValueError):
                    time.sleep(.5)
        finally:
            if proc.poll() is None:
                proc.terminate()
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)
    text = output.read_text(encoding="utf-8", errors="replace")
    result["gpu_evidence"] = [line for line in text.splitlines()
                              if "offloaded" in line or ("Vulkan" in line and "buffer size" in line)]
    return result


def backend_recovery(checkout, exe, model, output):
    os.environ["UNSLOTH_STUDIO_HOME"] = str(output / "home")
    sys.path.insert(0, str(checkout / "studio/backend"))
    from core.inference import llama_cpp
    from core.inference.llama_cpp import GgufLoadIntent, LlamaCppBackend
    from unittest.mock import patch

    backend = LlamaCppBackend()
    backend._find_llama_server_binary = lambda **_kwargs: str(exe)
    backend._get_gpu_memory = lambda *_args, **_kwargs: []
    backend._get_gpu_free_memory = lambda *_args, **_kwargs: []
    backend._can_estimate_kv = lambda: False
    launches = []
    injected = False
    original = subprocess.Popen

    def popen(cmd, *args, **kwargs):
        nonlocal injected
        if isinstance(cmd, (list, tuple)) and Path(str(cmd[0])) == exe and "-m" in cmd:
            launches.append(list(cmd))
            fit_modes = [cmd[i + 1] for i, flag in enumerate(cmd[:-1]) if flag in ("--fit", "-fit")]
            if len(launches) == 1 and (not fit_modes or fit_modes[-1] == "on"):
                injected = True
                crash = ("import ctypes; print('common_params_fit_impl: getting device memory data "
                         "for initial parameters:', flush=True); ctypes.windll.kernel32.ExitProcess(0xC0000005)")
                return original([sys.executable, "-c", crash], *args, **kwargs)
        return original(cmd, *args, **kwargs)

    result = {"launches": launches, "loaded": False}
    try:
        with patch.object(llama_cpp.subprocess, "Popen", popen):
            result["loaded"] = backend.load_model(GgufLoadIntent(
                model_identifier="random-llama", gguf_path=str(model), n_ctx=0,
                gpu_memory_mode="auto", n_threads=2, n_parallel=1,
            ))
            if result["loaded"]:
                result["completion"] = completion(backend.base_url, backend._auth_headers)
            result["cpu_fallback_reason"] = backend._cpu_fallback_reason
            result["log"] = list(backend._stdout_lines)
    except Exception:
        result["error"] = traceback.format_exc()
        result["log"] = list(backend._stdout_lines)
    finally:
        result["fault_injected"] = injected
        log_path = getattr(backend, "_llama_log_path", None)
        if log_path and Path(log_path).is_file():
            result["server_log"] = Path(log_path).read_text(encoding="utf-8", errors="replace")
        backend.unload_model()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkout", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args, _ = parser.parse_known_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result = {"platform": sys.platform, "native_driver_fault_reproduced": False}
    try:
        cache = args.out.parent.parent / "vulkan-fixture"
        cache.mkdir(exist_ok=True)
        marker = Path(sys.prefix) / "issue16395_real_deps"
        if not marker.exists():
            subprocess.run([sys.executable, "-m", "pip", "install", "-r",
                            str(args.checkout / "studio/backend/requirements/studio.txt"),
                            "gguf==0.19.0", "numpy", "psutil", "python-multipart"],
                           check=True, timeout=1200)
            marker.write_text("ready", encoding="utf-8")
        exe = binary(cache)
        model = cache / "random-llama.gguf"
        if not model.exists():
            make_model(model)
        result["release"] = RELEASE
        result["model_bytes"] = model.stat().st_size
        devices = subprocess.run([str(exe), "--list-devices"], cwd=exe.parent,
                                 capture_output=True, text=True, timeout=60)
        result["devices"] = devices.stdout + devices.stderr
        result["native"] = {mode: native(exe, model, mode, args.out.with_suffix(f".{mode}.log"))
                            for mode in ("on", "off")}
        result["backend"] = backend_recovery(args.checkout, exe, model, args.out.parent)
    except Exception:
        result["error"] = traceback.format_exc()
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
