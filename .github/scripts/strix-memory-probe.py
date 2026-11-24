"""Compare the Unsloth GGUF load path with its own llama-server binary."""

import contextlib
import json
import os
from pathlib import Path
import re
import secrets
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request


WORK = Path(sys.argv[1])
OUT = WORK / "out"
HOME = Path(os.environ["UNSLOTH_STUDIO_HOME"])
MODEL = WORK / "Qwen3.8-27B-UD-Q4_K_XL.gguf"
CONTEXT = 32768
EVIDENCE = re.compile(
    r"offload|memory breakdown|model buffer|memory detected|using device|"
    r"n_ctx|context length|ROCm[0-9]|Vulkan[0-9]|out of memory|failed to allocate",
    re.IGNORECASE,
)


def record(name, value):
    body = json.dumps(value, indent=2)
    (OUT / f"{name}.json").write_text(body + "\n")
    print(name, body, flush=True)


def snapshot(name):
    memory = Path("/proc/meminfo").read_text().splitlines()
    devices = {}
    for device in sorted(Path("/sys/class/drm").glob("card[0-9]*/device")):
        if not device.joinpath("mem_info_vram_total").exists():
            continue
        files = [*device.glob("mem_info_*"), device / "device", device / "vendor"]
        devices[device.parent.name] = {
            entry.name: entry.read_text().strip() for entry in files if entry.is_file()
        }
    record(name, {
        "memory": [line for line in memory if line.startswith(("MemTotal:", "MemAvailable:", "SwapFree:"))],
        "devices": devices,
    })
    return devices


def request(base, path, body=None, token=None, timeout=30):
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(
        base + path, headers=headers, data=json.dumps(body).encode() if body is not None else None
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def free_port():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


@contextlib.contextmanager
def server(command, port, health, log, cwd=None):
    with log.open("w") as output:
        process = subprocess.Popen(command, cwd=cwd, stdout=output, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        base = f"http://127.0.0.1:{port}"
        try:
            for _ in range(180):
                if process.poll() is not None:
                    raise RuntimeError(f"Server exited with {process.returncode}. See {log.name}.")
                try:
                    request(base, health, timeout=2)
                except (OSError, ValueError):
                    time.sleep(2)
                else:
                    break
            else:
                raise RuntimeError(f"Server health timed out. See {log.name}.")
            yield base
        finally:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait()


def login(base):
    password = (HOME / "auth" / ".bootstrap_password").read_text().strip()
    credentials = request(base, "/api/auth/login", {"username": "unsloth", "password": password})
    token = credentials["access_token"]
    if credentials.get("must_change_password"):
        changed = request(base, "/api/auth/change-password", {
            "current_password": password,
            "new_password": secrets.token_urlsafe(24) + "Aa1!",
        }, token)
        token = changed.get("access_token") or token
    return token


def capture_lines(name, paths):
    lines = [line for path in paths for line in path.read_text(errors="replace").splitlines()
             if EVIDENCE.search(line) and "Starting llama-server:" not in line]
    record(name, lines[-80:])


def run_unsloth():
    port = free_port()
    backend = WORK / "source" / "studio" / "backend"
    backend_log = WORK / "unsloth-backend.log"
    snapshot("unsloth-idle")
    command = [sys.executable, "run.py", "--api-only", "--host", "127.0.0.1", "--port", str(port)]
    try:
        with server(command, port, "/api/health", backend_log, cwd=backend) as base:
            token = login(base)
            result = request(base, "/api/inference/load", {
                "model_path": str(MODEL), "max_seq_length": CONTEXT,
                "gpu_memory_mode": "auto", "speculative_type": "off", "disable_vision": True,
            }, token, timeout=480)
            if result.get("status") in {"error", "failed"}:
                raise RuntimeError(str(result.get("detail") or result.get("status")))
            status = request(base, "/api/inference/status", token=token)
            record("unsloth-load", {"requested_context": CONTEXT, "response": result, "status": status})
            snapshot("unsloth-loaded")
            request(base, "/api/inference/unload", {"model_path": str(MODEL)}, token, timeout=90)
    finally:
        capture_lines("unsloth-placement", [backend_log, *sorted((HOME / "logs" / "llama-server").glob("*.log"))])


def run_standalone(binary):
    port = free_port()
    log = OUT / "llama-server.log"
    snapshot("llama-server-idle")
    command = [
        str(binary), "-m", str(MODEL), "--host", "127.0.0.1", "--port", str(port),
        "--ctx-size", str(CONTEXT), "--parallel", "1", "--fit", "on",
    ]
    try:
        with server(command, port, "/health", log) as base:
            props = request(base, "/props")
            record("llama-server-load", {
                "requested_context": CONTEXT,
                "default_generation_settings": props.get("default_generation_settings"),
            })
            snapshot("llama-server-loaded")
    finally:
        capture_lines("llama-server-placement", [log])


def main():
    devices = snapshot("host")
    if not any(data.get("device") == "0x1586" or data.get("device") == "0x158e"
               for data in devices.values()):
        print("Check the device list for the Strix Halo GPU PCI ID.", flush=True)
    from core.inference.llama_cpp import LlamaCppBackend

    binary = LlamaCppBackend._find_llama_server_binary()
    if not binary:
        raise RuntimeError("The Unsloth installation did not provide llama-server.")
    record("backend-memory-idle", LlamaCppBackend._get_gpu_memory(binary))
    for flag in ("--version", "--list-devices"):
        result = subprocess.run([binary, flag], capture_output=True, text=True, timeout=45)
        record(flag[2:], {"exit": result.returncode, "output": result.stdout + result.stderr})
    failures = []
    for name, probe in (("Unsloth", run_unsloth), ("llama.cpp", lambda: run_standalone(binary))):
        try:
            probe()
        except urllib.error.HTTPError as error:
            failures.append(f"{name}: HTTP {error.code} {error.reason}")
        except (OSError, RuntimeError, ValueError, KeyError) as error:
            failures.append(f"{name}: {error}")
    record("failures", failures)
    return bool(failures)


if __name__ == "__main__":
    sys.path.insert(0, str(WORK / "source" / "studio" / "backend"))
    sys.exit(main())
