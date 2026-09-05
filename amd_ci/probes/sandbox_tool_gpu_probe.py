#!/usr/bin/env python3
"""Probe: does Studio's Python tool still run, and still see the GPU, at this checkout?

Observes only. For PR 10285 (OS isolation for the Python/Terminal tools) the
question is not "do the new tests pass" but "what happens to a plain tool call
on a real GPU workstation". So this probe drives the checkout's own
``core.inference.tools._python_exec`` exactly as the chat loop does, in its
default mode (Required at the head, unsandboxed at the base), and records what
the tool printed. A host-side torch control and an explicit Full-access control
(``disable_sandbox=True``) are recorded beside it so the criteria can tell
"the sandbox hid the GPU" from "this box has no working torch".

Writes JSON via --out, never stdout (the backend prints import banners).
Pairs with criteria/sandbox_tool_gpu.py.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path

TOOL_CODE = r"""
import os, sys, json
info = {}
try:
    import torch
    info["torch"] = torch.__version__
    info["hip"] = getattr(torch.version, "hip", None)
    info["cuda_available"] = bool(torch.cuda.is_available())
    info["device_count"] = int(torch.cuda.device_count()) if info["cuda_available"] else 0
    info["device_name"] = torch.cuda.get_device_name(0) if info["cuda_available"] else None
    if info["cuda_available"]:
        x = torch.ones(64, 64, device="cuda"); torch.cuda.synchronize()
        info["matmul_ok"] = bool((x @ x).sum().item() == 64 * 64 * 64)
except Exception as exc:
    info["torch_error"] = f"{type(exc).__name__}: {exc}"
info["dev_kfd"] = os.path.exists("/dev/kfd")
info["dev_dri"] = os.path.exists("/dev/dri")
info["dev_nvidia"] = any(n.startswith("nvidia") for n in (os.listdir("/dev") if os.path.isdir("/dev") else []))
info["env_visible"] = sorted(k for k in os.environ if k.startswith(("HIP_", "ROCR_", "CUDA_", "HSA_")))
info["cwd"] = os.getcwd()
print("PROBE_JSON=" + json.dumps(info))
"""


def _read(path: str) -> str | None:
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def _host_torch() -> dict:
    out: dict = {}
    try:
        import torch  # noqa: PLC0415

        out["torch"] = torch.__version__
        out["hip"] = getattr(torch.version, "hip", None)
        out["cuda_available"] = bool(torch.cuda.is_available())
        out["device_count"] = int(torch.cuda.device_count()) if out["cuda_available"] else 0
        out["device_name"] = torch.cuda.get_device_name(0) if out["cuda_available"] else None
    except Exception as exc:  # noqa: BLE001
        out["torch_error"] = f"{type(exc).__name__}: {exc}"
    return out


def _parse_tool(text: str) -> dict:
    """Split the tool's returned string into the JSON it printed and everything else."""
    res: dict = {"raw": text[-3000:]}
    for line in text.splitlines():
        if line.startswith("PROBE_JSON="):
            try:
                res["parsed"] = json.loads(line[len("PROBE_JSON="):])
            except Exception as exc:  # noqa: BLE001
                res["parse_error"] = str(exc)
    res["ran"] = "parsed" in res
    res["fail_closed"] = "OS_ISOLATION_UNAVAILABLE" in text or "SandboxUnavailable" in text
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True, type = Path)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--tool-timeout", type = int, default = 240)
    args = ap.parse_args()
    args.out = args.out.resolve()
    args.checkout = args.checkout.resolve()

    obs: dict = {"state": args.state, "checkout": str(args.checkout), "platform": sys.platform,
                 "python": sys.executable}
    obs["host"] = {
        "bwrap": shutil.which("bwrap"),
        "sandbox_exec": shutil.which("sandbox-exec"),
        "apparmor_restrict_unprivileged_userns": _read("/proc/sys/kernel/apparmor_restrict_unprivileged_userns"),
        "unprivileged_userns_clone": _read("/proc/sys/kernel/unprivileged_userns_clone"),
        "dockerenv": os.path.exists("/.dockerenv"),
        "uid": getattr(os, "getuid", lambda: None)(),
        "dev_kfd": os.path.exists("/dev/kfd"),
        "dev_dri": os.path.exists("/dev/dri"),
        "visible_env": {k: v for k, v in os.environ.items()
                        if k.startswith(("HIP_", "ROCR_", "CUDA_", "HSA_"))},
    }
    obs["host_torch"] = _host_torch()

    backend = args.checkout / "studio" / "backend"
    if not backend.is_dir():
        obs["error"] = f"no backend at {backend}"
        args.out.write_text(json.dumps(obs, indent = 2))
        return 0

    # Keep the tool's home and sandbox root under the runner's reclaimed temp.
    scratch = Path(tempfile.mkdtemp(prefix = f"amd-ci-sbx-{args.state}-"))
    os.environ["UNSLOTH_STUDIO_HOME"] = str(scratch / "home")
    os.environ["UNSLOTH_STUDIO_SANDBOX_HOME"] = str(scratch / "sandbox")
    os.environ.setdefault("UNSLOTH_ALLOW_CPU", "1")
    os.chdir(backend)
    sys.path.insert(0, str(backend))
    for stale in [m for m in sys.modules if m.split(".")[0] in ("core", "loggers", "utils", "state", "storage", "routes", "models", "auth")]:
        del sys.modules[stale]

    try:
        from core.inference import tools  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        obs["import_error"] = traceback.format_exc()[-3000:]
        args.out.write_text(json.dumps(obs, indent = 2))
        return 0
    obs["has_os_sandbox_module"] = (backend / "core" / "inference" / "os_sandbox.py").is_file()

    if obs["has_os_sandbox_module"]:
        try:
            from core.inference.os_sandbox import capability_snapshot  # noqa: PLC0415

            t0 = time.time()
            cap = capability_snapshot(force = True)
            obs["capability"] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in cap.__dict__.items()}
            obs["capability_seconds"] = round(time.time() - t0, 3)
        except Exception:  # noqa: BLE001
            obs["capability_error"] = traceback.format_exc()[-2000:]

    plain = "print('PROBE_JSON=' + __import__('json').dumps({'plain': True}))"
    for label, code, kwargs in (("tool_plain", plain, {}), ("tool_default", TOOL_CODE, {}),
                                ("tool_full", TOOL_CODE, {"disable_sandbox": True})):
        t0 = time.time()
        try:
            text = tools._python_exec(code, session_id = f"amd-ci-{args.state}",
                                      timeout = args.tool_timeout, **kwargs)
            obs[label] = _parse_tool(str(text))
        except Exception as exc:  # noqa: BLE001
            obs[label] = {"ran": False, "exception": f"{type(exc).__name__}: {exc}",
                          "fail_closed": "SandboxUnavailable" in type(exc).__name__}
        obs[label]["seconds"] = round(time.time() - t0, 3)

    args.out.write_text(json.dumps(obs, indent = 2, default = str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
