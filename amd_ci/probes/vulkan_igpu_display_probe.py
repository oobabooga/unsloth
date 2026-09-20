#!/usr/bin/env python3
"""Probe: what does this host's integrated GPU report, and what does this checkout's
Studio System panel render from it?

Three observations, in order of how much of the stack they involve:

  1. ggml's own device type, read by running THIS checkout's `_vulkan_probe.py`
     against a Vulkan llama.cpp. `is_igpu` is the enum the whole display change
     keys on, so it is read at the bottom rather than inferred from a name.
  2. This checkout's `get_vulkan_inference_gpu_info()`, plus the ROCm view of the
     same APU (`get_backend_visible_gpu_info`, `get_visible_gpu_utilization`).
     The ROCm view is observed even though it is not the subject: the change must
     leave it alone, and that cannot be checked without reading it.
  3. The React render of this checkout's real Resources tab and floating monitor,
     fed the readings from (2). This exists because the display hook is new at the
     head: comparing "the hook imports" against "it does not" would be vacuous,
     while rendering the component each state actually ships is not.

Observes only. Whether the rendered text is the old one or the new one is the
criteria module's call.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def import_hardware(checkout: Path):
    """Import THIS checkout's hardware module, the way the app does."""
    backend = checkout / "studio" / "backend"
    if not backend.is_dir():
        raise SystemExit(f"no backend at {backend}")
    sys.path.insert(0, str(backend))
    for stale in [m for m in sys.modules if m.startswith(("utils.", "core."))]:
        del sys.modules[stale]
    import utils.hardware.hardware as hw  # noqa: PLC0415
    return hw


def raw_ggml_devices(checkout: Path, lib_dir: str) -> dict:
    """Run the checkout's own ctypes probe against a Vulkan llama.cpp lib dir.

    Its stdout is one tab-separated row per device:
    index, free bytes, is_igpu, total bytes, name, type_known.
    """
    probe = checkout / "studio" / "backend" / "core" / "inference" / "_vulkan_probe.py"
    if not probe.is_file():
        return {"error": f"no _vulkan_probe.py at {probe}"}
    try:
        proc = subprocess.run(
            [sys.executable, str(probe), lib_dir],
            capture_output = True, text = True, timeout = 120, encoding = "utf-8",
        )
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}
    rows = []
    for line in (proc.stdout or "").splitlines():
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 4 or not parts[0].strip().isdigit():
            continue
        rows.append({
            "index": int(parts[0]),
            "free_bytes": int(parts[1]),
            "is_igpu": parts[2] == "1",
            "total_bytes": int(parts[3]),
            "name": parts[4].strip() if len(parts) >= 5 else "",
            "type_known": len(parts) >= 6 and parts[5].strip() == "1",
        })
    return {"rc": proc.returncode, "rows": rows, "stderr": (proc.stderr or "")[-2000:]}


def render_cases(checkout: Path, harness: Path, cases: list[dict], node: str,
                 out_dir: Path) -> dict:
    """Render this checkout's real components against the captured readings."""
    frontend = checkout / "studio" / "frontend"
    if not frontend.is_dir():
        return {"error": f"no frontend at {frontend}"}
    if not shutil.which(node) and not Path(node).is_file():
        return {"error": f"node not found: {node}"}
    install: dict = {}
    if not (frontend / "node_modules" / "react").is_dir():
        # Per state, because each state is its own checkout and jobs share no disk
        # or npm cache with each other. `npm ci` from the lockfile, so the two
        # states differ only where the repository does.
        npm = os.environ.get("AMD_CI_NPM", "npm")
        try:
            proc = subprocess.run(
                [npm, "ci", "--no-audit", "--no-fund"], cwd = str(frontend),
                capture_output = True, text = True, timeout = 2400, encoding = "utf-8",
                shell = (os.name == "nt"),
            )
            install = {"rc": proc.returncode, "tail": (proc.stdout or "")[-500:],
                       "stderr": (proc.stderr or "")[-1000:]}
        except Exception as e:  # noqa: BLE001
            return {"error": f"npm ci failed: {type(e).__name__}: {e}"}
        if not (frontend / "node_modules" / "react").is_dir():
            return {"error": "npm ci did not produce node_modules", "install": install}
    target = frontend / "tests" / "amd_ci_render_probe.ts"
    shutil.copyfile(harness, target)
    cases_path = out_dir / "render_cases.json"
    cases_path.write_text(json.dumps(cases, indent = 2), encoding = "utf-8")
    result_path = out_dir / "render_result.json"
    try:
        proc = subprocess.run(
            [node, "--experimental-strip-types", str(target),
             "--cases", str(cases_path), "--out", str(result_path)],
            cwd = str(frontend), capture_output = True, text = True, timeout = 600,
            encoding = "utf-8",
        )
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}
    out: dict = {"rc": proc.returncode, "stderr": (proc.stderr or "")[-3000:],
                 "npm_install": install}
    if result_path.is_file():
        try:
            out.update(json.loads(result_path.read_text(encoding = "utf-8")))
        except Exception as e:  # noqa: BLE001
            out["parse_error"] = f"{type(e).__name__}: {e}"
    return out


def system_info(gpu: dict | None, inference_gpu: dict | None, backend: str) -> dict:
    """The shape useSystemInfo hands a component, around the real GPU readings."""
    info = {
        "status": "ready",
        "platform": sys.platform,
        "python_version": "%d.%d" % sys.version_info[:2],
        "device_backend": backend,
        "uptime_seconds": 60,
        "cpu": {"logical_count": os.cpu_count() or 1,
                "physical_count": (os.cpu_count() or 2) // 2,
                "usage_percent": 5, "frequency_mhz": 3000},
        "memory": {"total_gb": 0, "available_gb": 0, "percent_used": 0, "process_used_mb": 0},
        "disk": {"total_gb": 0, "free_gb": 0, "percent_used": 0},
        "gpu": gpu or {"available": False, "devices": []},
        "ml_packages": {},
    }
    try:
        import psutil  # noqa: PLC0415
        vm = psutil.virtual_memory()
        info["memory"] = {
            "total_gb": round(vm.total / 1024 ** 3, 2),
            "available_gb": round(vm.available / 1024 ** 3, 2),
            "percent_used": vm.percent,
            "process_used_mb": 0,
        }
    except Exception:  # noqa: BLE001
        pass
    if inference_gpu:
        info["inference_gpu"] = inference_gpu
    return info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True, type = Path)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--vulkan-lib-dir", default = os.environ.get("AMD_CI_VULKAN_LIB_DIR", ""))
    ap.add_argument("--llama-server", default = os.environ.get("LLAMA_SERVER_PATH", ""))
    ap.add_argument("--node", default = os.environ.get("AMD_CI_NODE", "node"))
    ap.add_argument("--harness", default = os.environ.get("AMD_CI_RENDER_HARNESS", ""))
    args = ap.parse_args()

    out_dir = args.out.parent
    out_dir.mkdir(parents = True, exist_ok = True)
    obs: dict = {"state": args.state, "platform": sys.platform}

    if args.llama_server:
        os.environ["LLAMA_SERVER_PATH"] = args.llama_server
    if args.vulkan_lib_dir:
        obs["ggml_devices"] = raw_ggml_devices(args.checkout, args.vulkan_lib_dir)
    else:
        obs["ggml_devices"] = {"error": "no --vulkan-lib-dir given"}

    try:
        hw = import_hardware(args.checkout)
        obs["hardware_file"] = hw.__file__
    except Exception as e:  # noqa: BLE001
        obs["hardware_error"] = f"{type(e).__name__}: {e}"
        hw = None

    if hw is not None:
        for name, call in (
            ("vulkan_gpu", "get_vulkan_inference_gpu_info"),
            ("backend_gpu", "get_backend_visible_gpu_info"),
            ("utilization", "get_visible_gpu_utilization"),
            ("summary", "get_gpu_summary"),
        ):
            try:
                obs[name] = getattr(hw, call)()
            except Exception as e:  # noqa: BLE001
                obs[f"{name}_error"] = f"{type(e).__name__}: {e}"

    # The cases are built from what this host actually reported. A case whose
    # readings are missing is recorded as missing rather than filled in.
    cases: list[dict] = []
    vulkan = obs.get("vulkan_gpu")
    backend_gpu = obs.get("backend_gpu")
    if isinstance(vulkan, dict) and vulkan.get("devices"):
        cases.append({"name": "measured_vulkan_only",
                      "systemInfo": system_info(vulkan, None, "cpu")})
        if isinstance(backend_gpu, dict) and backend_gpu.get("devices"):
            cases.append({"name": "measured_rocm_with_vulkan_inference",
                          "systemInfo": system_info(backend_gpu, vulkan, "rocm")})
    if isinstance(backend_gpu, dict) and backend_gpu.get("devices"):
        cases.append({"name": "measured_rocm_only",
                      "systemInfo": system_info(backend_gpu, None, "rocm")})
    obs["case_names"] = [c["name"] for c in cases]

    if cases and args.harness:
        obs["render"] = render_cases(args.checkout, Path(args.harness), cases,
                                     args.node, out_dir)
    elif not cases:
        obs["render"] = {"error": "no device readings, so nothing was rendered"}
    else:
        obs["render"] = {"error": "no --harness given"}

    args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
