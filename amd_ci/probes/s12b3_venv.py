#!/usr/bin/env python3
"""Minimal ROCm torch venv for gfx1151 probes (torch from AMD's gfx1151 index + pytest), built once per pin under
$AMD_CI_WORK and shared by every state. Observes only."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

AMD_GFX1151_INDEX = "https://repo.amd.com/rocm/whl/gfx1151/"
PYPI = "https://pypi.org/simple"
ENV_CODE = (
    "import json, torch\n"
    "ok = torch.cuda.is_available()\n"
    "p = torch.cuda.get_device_properties(0) if ok else None\n"
    "smoke = None\n"
    "if ok:\n"
    "    a = torch.randn(256, 256, device='cuda')\n"
    "    smoke = float((a @ a).abs().sum().item())\n"
    "try:\n"
    "    import triton; tv = triton.__version__\n"
    "except Exception as e:\n"
    "    tv = 'import failed: ' + repr(e)[:200]\n"
    "print('AMDCI' + json.dumps({'torch': torch.__version__, 'hip': torch.version.hip, 'cuda': torch.version.cuda,\n"
    "    'triton': tv, 'cuda_available': ok, 'device': p.name if p else None,\n"
    "    'arch': getattr(p, 'gcnArchName', None) if p else None, 'torch_file': torch.__file__,\n"
    "    'matmul_smoke': smoke}))\n"
)


def sh(cmd, log: Path, timeout = 3600, env = None, cwd = None):
    with open(log, "ab") as fh:
        fh.write(("\n$ " + " ".join(map(str, cmd)) + "\n").encode())
        fh.flush()
        try:
            return subprocess.run(cmd, stdout = fh, stderr = subprocess.STDOUT, timeout = timeout, env = env,
                                  cwd = cwd).returncode
        except subprocess.TimeoutExpired:
            return "timeout"
        except FileNotFoundError as e:
            fh.write(f"{e!r}\n".encode())
            return "not_found"


def env_info(py: str) -> dict:
    p = subprocess.run([py, "-c", ENV_CODE], capture_output = True, text = True, timeout = 600)
    line = next((ln for ln in (p.stdout or "").splitlines() if ln.startswith("AMDCI")), None)
    return json.loads(line[5:]) if line else {"error": ((p.stdout or "") + (p.stderr or ""))[-2000:]}


def build(venv: Path, specs: list, log: Path, python_req: str | None = None) -> dict:
    py = venv / "bin" / "python"
    info: dict = {"venv": str(venv), "python": str(py), "specs": specs}
    marker = venv / ".amd_ci_ready"
    if marker.is_file():
        info["reused"] = True
    else:
        uv = shutil.which("uv")
        info["uv"] = uv
        interp = python_req or sys.executable
        info["python_request"] = interp
        if uv:
            steps = [("venv", [uv, "venv", "-q", "--seed", "--python", interp, str(venv)]),
                     ("install", [uv, "pip", "install", "--python", str(py), "--index-url", AMD_GFX1151_INDEX,
                                  "--extra-index-url", PYPI, "--index-strategy", "unsafe-best-match", *specs,
                                  "pytest", "pytest-timeout"])]
        else:
            steps = [("venv", [sys.executable, "-m", "venv", str(venv)]),
                     ("install", [str(py), "-m", "pip", "install", "-q", "--index-url", AMD_GFX1151_INDEX,
                                  "--extra-index-url", PYPI, *specs, "pytest", "pytest-timeout"])]
        info["steps"] = {}
        for name, cmd in steps:
            rc = sh(cmd, log, timeout = 3000)
            info["steps"][name] = rc
            if rc != 0:
                info["error"] = f"step {name} rc={rc}"
                info["log_tail"] = log.read_text(encoding = "utf-8", errors = "replace")[-3000:]
                return info
        marker.write_text("ok", encoding = "utf-8")
    info["env"] = env_info(str(py))
    pl = subprocess.run([str(py), "-m", "pip", "list"], capture_output = True, text = True)
    info["pip_list"] = [ln for ln in (pl.stdout or "").splitlines()
                        if ln.split(" ")[0].lower() in ("torch", "triton", "pytorch-triton-rocm", "pytest",
                                                        "numpy", "rocm", "rocm-sdk-core")]
    return info


def clean_env(extra: dict | None = None) -> dict:
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTEST_ADDOPTS", "PYTHONSTARTUP")}
    env.update(extra or {})
    return env
