#!/usr/bin/env python3
"""Observe which Torch wheel index the installer would choose here, for PR 9829.

Observes only. Judgement lives in criteria/rocm_index_route.py.

The routing decision is assembled inside `_ensure_rocm_torch()`; there is no pure
function that returns "the index this host would pick". So this probe drives that
function at both states with the pip entry points replaced by recorders, which is
what the PR's own test does, and reads the choice out of the recorded call. Two
things are simulated and both are named in the report:

  * `_probe_torch_runtime` answers "CPU-only torch". That is the documented
    precondition of the whole function ("reinstall torch with ROCm wheels when the
    venv received CPU-only torch"), and without it a runner that already has working
    ROCm torch makes the function return before it decides anything.
  * in the `runtime_only` scenario, `rocminfo` and `amd-smi` are removed from PATH.
    That is the runtime-only ROCm install this PR's KFD fallback exists for. The KFD
    topology read underneath is the REAL kernel's view of the REAL GPU; only
    executable discovery is masked.

Everything else - GPU enumeration, KFD sysfs, product-name inference, the ROCm
version - is the live host. Nothing is installed.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

SCENARIOS = ("natural", "runtime_only", "runtime_only_unnamed_product")

# Anything that steers wheel selection. The primary comparison must not be decided by
# something this job inherited from the runner.
_STRIP_ENV = (
    "UNSLOTH_ROCM_GFX_ARCH", "UNSLOTH_TORCH_BACKEND", "UNSLOTH_ROCM_TORCH_INSTALLED",
    "UNSLOTH_AMD_ROCM_MIRROR", "UNSLOTH_ROCM_WINDOWS_MIRROR", "UNSLOTH_TORCH_INDEX_URL",
    "HSA_OVERRIDE_GFX_VERSION", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES",
    "CUDA_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL",
)


# --------------------------------------------------------------------------- worker


def _load(checkout: str):
    studio = Path(checkout) / "studio"
    for p in (str(studio), str(studio / "backend")):
        if p not in sys.path:
            sys.path.insert(0, p)
    spec = importlib.util.spec_from_file_location(
        "amd_ci_install_python_stack", studio / "install_python_stack.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _mask_userland_probes() -> list[str]:
    """Remove rocminfo and amd-smi from PATH by rebuilding it without their directories.

    Not `pkill`, not a rename: nothing on this host is modified. A directory is dropped
    only when it actually holds one of the two, so the rest of PATH survives."""
    removed = []
    keep = []
    for d in os.environ.get("PATH", "").split(os.pathsep):
        if not d:
            continue
        if any(os.path.exists(os.path.join(d, exe)) for exe in ("rocminfo", "amd-smi")):
            removed.append(d)
            continue
        keep.append(d)
    os.environ["PATH"] = os.pathsep.join(keep)
    return removed


def _observe(mod, scenario: str) -> dict:
    out: dict = {"scenario": scenario}

    out["rocminfo_on_path"] = shutil.which("rocminfo")
    out["amd_smi_on_path"] = shutil.which("amd-smi")

    def _call(name, fn, *a):
        try:
            return fn(*a)
        except Exception as e:  # noqa: BLE001
            out.setdefault("errors", {})[name] = f"{type(e).__name__}: {e}"
            return None

    out["detected_gfx"] = _call("detect", mod._detect_amd_gfx_codes, False)
    out["winning_probe"] = getattr(mod, "_LAST_AMD_GFX_PROBE", None)
    out["kfd_targets"] = _call("kfd", mod._kfd_gfx_targets)
    out["inferred_product_gfx"] = _call("infer", mod._infer_linux_amd_gfx_arch)
    out["has_rocm_gpu"] = _call("has_rocm_gpu", mod._has_rocm_gpu)
    ver = _call("rocm_version", mod._detect_rocm_version)
    out["rocm_version"] = list(ver) if ver else None
    out["strix_needs_arch_index"] = (
        _call("strix", mod._strix_needs_amd_arch_index, ver) if ver else None)

    # Head-only helpers. Their absence at the base is the observation, not a failure.
    rt = getattr(mod, "_runtime_gfx_target", None)
    out["has_runtime_gfx_target"] = rt is not None
    if rt is not None:
        res = _call("runtime_gfx_target", rt, out["inferred_product_gfx"])
        if res is not None:
            out["runtime_gfx_target"] = res[0]
            out["runtime_gfx_devices"] = list(res[1])
            out["runtime_physical_gfx"] = res[2]
    out["has_generic_wheel_lacks_kernels"] = hasattr(mod, "_generic_rocm_wheel_lacks_kernels")
    out["has_index_url_join"] = hasattr(mod, "_index_url_join")
    out["generic_wheel_gfx"] = sorted(getattr(mod, "_GENERIC_ROCM_WHEEL_GFX", ()) or ())

    # URL construction for the target this host actually is, independent of the drive below.
    for label, gfx in (("kfd", (out.get("kfd_targets") or [None])[0]),
                       ("detected", (out.get("detected_gfx") or [None])[0])):
        out[f"index_url_for_{label}"] = _call(
            f"url_{label}", mod._amd_arch_index_url, gfx) if gfx else None

    # ---- drive the real decision
    calls: list[dict] = []

    def rec(label, *args, **kwargs):
        argv = [str(a) for a in args]
        idx = argv.index("--index-url") + 1 if "--index-url" in argv else None
        calls.append({"label": str(label), "args": argv,
                      "index_url": argv[idx] if idx is not None and idx < len(argv) else None})
        return True

    mod.pip_install = rec
    mod.pip_install_try = rec
    mod._install_bnb_windows_rocm = lambda *a, **k: True
    mod._probe_torch_runtime = lambda *a, **k: (True, True, "2.10.0+cpu", None, None)
    try:
        mod._ensure_rocm_torch()
    except Exception as e:  # noqa: BLE001
        out.setdefault("errors", {})["ensure_rocm_torch"] = f"{type(e).__name__}: {e}"
    out["install_calls"] = calls
    out["chosen_index_url"] = next(
        (c["index_url"] for c in reversed(calls) if c["index_url"]), None)
    out["chosen_label"] = calls[-1]["label"] if calls else None
    return out


def worker(checkout: str, scenario: str) -> dict:
    for name in _STRIP_ENV:
        os.environ.pop(name, None)
    removed: list[str] = []
    if scenario.startswith("runtime_only"):
        removed = _mask_userland_probes()
    mod = _load(checkout)
    if scenario == "runtime_only_unnamed_product":
        # A host whose product name does not identify the GPU, which is most AMD boxes
        # that are not a named Strix part. Supplementary evidence only.
        mod._infer_linux_amd_gfx_arch = lambda *a, **k: None
    obs = _observe(mod, scenario)
    obs["path_dirs_removed"] = removed
    return obs


# --------------------------------------------------------------------------- driver


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default = "")
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--scenario", default = None, choices = SCENARIOS)
    ap.add_argument("--python", default = sys.executable)
    args = ap.parse_args()
    args.out.parent.mkdir(parents = True, exist_ok = True)

    if args.scenario:
        try:
            payload = worker(args.checkout, args.scenario)
        except Exception as e:  # noqa: BLE001
            import traceback
            payload = {"scenario": args.scenario,
                       "worker_error": f"{type(e).__name__}: {e}",
                       "traceback": traceback.format_exc()[-2000:]}
        args.out.write_text(json.dumps(payload, indent = 2))
        return 0

    obs: dict = {"state": args.state, "checkout": args.checkout}
    for scenario in SCENARIOS:
        print(f"-- scenario {scenario}", flush = True)
        # Its own process: the module runs host probes at import and caches the winning
        # one in a global, so a masked run must not inherit an unmasked module.
        sub = args.out.parent / f"{args.out.stem}.{scenario}.json"
        log = sub.with_suffix(".log")
        with open(log, "w") as fh:
            rc = subprocess.run(
                [args.python, os.path.abspath(__file__), "--scenario", scenario,
                 "--checkout", args.checkout, "--out", str(sub)],
                stdout = fh, stderr = subprocess.STDOUT).returncode
        rec: dict = {"rc": rc}
        if sub.is_file():
            try:
                rec.update(json.loads(sub.read_text()))
            except Exception as e:  # noqa: BLE001
                rec["parse_error"] = f"{type(e).__name__}: {e}"
        else:
            rec["no_output"] = True
            rec["log_tail"] = log.read_text(errors = "replace")[-1500:] if log.is_file() else ""
        obs[scenario] = rec
    args.out.write_text(json.dumps(obs, indent = 2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
