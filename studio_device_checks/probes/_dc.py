"""Helpers shared by the probes. Each probe runs in a fresh process under the installed Studio's Python, with
PYTHONPATH pointing at one backend tree (the installed release, the PR base or the PR head)."""

from __future__ import annotations

import json
import os
import sys
import time
import traceback


def tree_backend() -> str:
    """The backend tree this arm was started for (set by the driver)."""
    return os.environ.get("DEVCHECKS_BACKEND", "")


def enter_tree() -> dict:
    """Put the arm's backend first on sys.path, chdir into it, and prove `core` comes from it."""
    backend = tree_backend()
    info = {"backend": backend, "core_file": None, "own_tree": None}
    if not backend:
        return info
    if backend not in sys.path:
        sys.path.insert(0, backend)
    try:
        os.chdir(backend)
    except OSError:
        pass
    try:
        import core  # noqa: F401

        core_file = os.path.abspath(getattr(core, "__file__", None) or list(core.__path__)[0])
        info["core_file"] = core_file
        info["own_tree"] = os.path.normcase(core_file).startswith(os.path.normcase(os.path.abspath(backend)))
    except Exception as exc:  # noqa: BLE001
        info["core_error"] = f"{type(exc).__name__}: {exc}"[:400]
        info["own_tree"] = False
    return info


def windows_compile_gate(rec: dict) -> None:
    """Studio's inference worker runs this before torch.compile on Windows; the probes do the same."""
    if os.name != "nt":
        return
    try:
        import logging

        from core._msvc_env import gate_torch_compile_on_windows

        gate_torch_compile_on_windows(logging.getLogger("devchecks"))
        rec["windows_compile_gate"] = "ran"
    except Exception as exc:  # noqa: BLE001
        rec["windows_compile_gate"] = f"{type(exc).__name__}: {exc}"[:300]


def err(exc: BaseException, limit: int = 600) -> str:
    return f"{type(exc).__name__}: {exc}"[:limit]


def tb(limit: int = 3000) -> str:
    return traceback.format_exc()[-limit:]


def write(path: str, rec: dict) -> None:
    rec.setdefault("finished_at", time.strftime("%Y-%m-%dT%H:%M:%S"))
    tmp = path + ".tmp"
    with open(tmp, "w", encoding = "utf-8") as fh:
        json.dump(rec, fh, indent = 1, default = str)
    os.replace(tmp, path)


def out_path() -> str:
    """Every probe takes `--out FILE` as its first two arguments."""
    if len(sys.argv) < 3 or sys.argv[1] != "--out":
        raise SystemExit("usage: <probe> --out FILE [args]")
    return sys.argv[2]


def cuda_info() -> dict:
    out = {"cuda_available": False}
    try:
        import torch

        out["torch"] = torch.__version__
        out["torch_cuda"] = getattr(torch.version, "cuda", None)
        out["torch_hip"] = getattr(torch.version, "hip", None)
        out["cuda_available"] = bool(torch.cuda.is_available())
        if out["cuda_available"]:
            out["device_count"] = torch.cuda.device_count()
            out["gpu"] = torch.cuda.get_device_name(0)
            out["capability"] = list(torch.cuda.get_device_capability(0))
            free, total = torch.cuda.mem_get_info(0)
            out["vram_total_gb"] = round(total / 2**30, 2)
            out["vram_free_gb"] = round(free / 2**30, 2)
    except Exception as exc:  # noqa: BLE001
        out["torch_error"] = err(exc)
    return out
