#!/usr/bin/env python3
"""Probe: what each state's installer decides about THIS machine's real GPU.

No mocks. The state's own `studio/install_python_stack.py` is loaded from its
worktree and asked the questions it asks during an install: what architectures
are on this host, what ROCm version is here, which index would serve them. The
answers come from rocminfo / amd-smi / the KFD topology sysfs of the machine the
probe is running on.

It also records `torch.cuda.get_arch_list()` for the wheel actually installed in
this environment, which is the measurement the change's architecture table is a
claim about.

Observes only, and never judges: it does not decide whether a difference between
the states matters. Loading the installer module is done with `__name__` set to
something other than `__main__`, so its CLI entry point does not run.
"""

from __future__ import annotations

import argparse
import importlib.util
import io
import contextlib
import json
import sys
import traceback
from pathlib import Path


def _load(stack_path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, stack_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    # Import banners go to a buffer: this probe's stdout is a log, but a module
    # that printed on import has corrupted a probe's JSON before.
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        spec.loader.exec_module(mod)
    return mod, buf.getvalue()


def _call(obs: dict, key: str, fn) -> object:
    try:
        value = fn()
    except Exception as e:  # noqa: BLE001
        obs[key] = None
        obs.setdefault("_errors", {})[key] = f"{type(e).__name__}: {e}"
        return None
    obs[key] = value
    return value


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True, type = Path)
    args = ap.parse_args()

    obs: dict = {"state": args.state, "checkout": args.checkout}

    # The wheel in this environment, and what it actually carries. Independent of
    # the state under test: one environment serves every leg, so this is the same
    # measurement each time and any difference between legs would be a red flag.
    try:
        import torch
        obs["torch_version"] = torch.__version__
        obs["torch_hip"] = torch.version.hip
        obs["wheel_arch_list"] = sorted(
            a.split(":")[0] for a in (torch.cuda.get_arch_list() or []))
        obs["torch_cuda_available"] = bool(torch.cuda.is_available())
        obs["device_archs"] = [
            (getattr(torch.cuda.get_device_properties(i), "gcnArchName", "") or "").split(":")[0]
            for i in range(torch.cuda.device_count() if torch.cuda.is_available() else 0)
        ]
    except Exception as e:  # noqa: BLE001
        obs["torch_error"] = f"{type(e).__name__}: {e}"

    stack = Path(args.checkout) / "studio" / "install_python_stack.py"
    if not stack.is_file():
        obs["error"] = f"no installer at {stack}"
        args.out.write_text(json.dumps(obs, indent = 2, default = str))
        return 0

    try:
        mod, banner = _load(stack, f"amd_ci_stack_{args.state}")
    except Exception:  # noqa: BLE001
        obs["error"] = "could not load the installer module"
        obs["traceback"] = traceback.format_exc()[-3000:]
        args.out.write_text(json.dumps(obs, indent = 2, default = str))
        return 0
    obs["import_banner"] = banner[-500:]

    have = {n: hasattr(mod, n) for n in (
        "_detect_amd_gfx_codes", "_detect_rocm_version", "_infer_linux_amd_gfx_arch",
        "_kfd_gfx_targets", "_amd_arch_index_url", "_strix_needs_amd_arch_index",
        "_runtime_gfx_target", "_generic_rocm_wheel_lacks_kernels",
        "_GENERIC_ROCM_WHEEL_GFX", "_GFX_TO_AMD_INDEX_ARCH",
    )}
    obs["symbols"] = have

    codes = _call(obs, "gfx_codes", lambda: list(mod._detect_amd_gfx_codes())) \
        if have["_detect_amd_gfx_codes"] else None
    ver = _call(obs, "rocm_version", mod._detect_rocm_version) \
        if have["_detect_rocm_version"] else None
    if have["_infer_linux_amd_gfx_arch"]:
        _call(obs, "inferred_gfx", mod._infer_linux_amd_gfx_arch)
    if have["_kfd_gfx_targets"]:
        _call(obs, "kfd_targets", lambda: list(mod._kfd_gfx_targets()))

    # The target the installer would route for. The head resolves this through a
    # dedicated function; the base has no such function, so its target is the
    # probe's first code, which is what the base's own callers use.
    target = None
    if have["_runtime_gfx_target"]:
        res = _call(obs, "runtime_gfx_target_raw", lambda: mod._runtime_gfx_target(
            mod._infer_linux_amd_gfx_arch() if have["_infer_linux_amd_gfx_arch"] else None))
        if isinstance(res, (list, tuple)) and res:
            target = res[0]
    if target is None and codes:
        target = codes[0]
    obs["target_gfx"] = (target or "").split(":")[0] or None

    if have["_amd_arch_index_url"]:
        _call(obs, "amd_arch_index_url", lambda: mod._amd_arch_index_url(obs["target_gfx"]))
    if have["_strix_needs_amd_arch_index"] and isinstance(ver, (list, tuple)):
        _call(obs, "strix_needs_amd_arch_index",
              lambda: bool(mod._strix_needs_amd_arch_index(tuple(ver))))
    if have["_generic_rocm_wheel_lacks_kernels"]:
        _call(obs, "generic_lacks_kernels_for_target",
              lambda: bool(mod._generic_rocm_wheel_lacks_kernels(
                  obs["target_gfx"], tuple(ver) if isinstance(ver, (list, tuple)) else None)))
    if have["_GENERIC_ROCM_WHEEL_GFX"]:
        obs["declared_generic_wheel_gfx"] = sorted(mod._GENERIC_ROCM_WHEEL_GFX)
    if have["_GFX_TO_AMD_INDEX_ARCH"]:
        obs["amd_index_leaf_for_target"] = dict(mod._GFX_TO_AMD_INDEX_ARCH).get(
            obs["target_gfx"] or "")

    args.out.write_text(json.dumps(obs, indent = 2, default = str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
