#!/usr/bin/env python3
"""Probe: which stable-diffusion.cpp build does this checkout install when torch is CPU-only?

Observes only, on the host's REAL detection (sysfs / registry inventory, the real Vulkan loader
and ICDs). Run under a venv whose torch is the CPU wheel, which is what a Radeon with no ROCm
wheel gets (#13666). For each checkout: the diffusion device backend, image_install_accelerator()
for it, then the checkout's own install_sd_cpp_prebuilt.install() into a scratch dir with that
accelerator, and the installed sd-cli's --list-devices.

Writes JSON to --out, never stdout.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

SCRUB = (
    "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES",
    "GGML_VK_VISIBLE_DEVICES", "UNSLOTH_DIFFUSION_SD_CPP", "UNSLOTH_DIFFUSION_SD_CPP_DEVICE",
    "UNSLOTH_DIFFUSION_ENGINE", "UNSLOTH_DIFFUSION_SD_CPP_VULKAN_FALLBACK", "UNSLOTH_SD_CPP_TAG",
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True, type = Path)
    ap.add_argument("--out", required = True, type = Path)
    args = ap.parse_args()
    args.out = args.out.resolve()
    args.checkout = args.checkout.resolve()

    obs: dict = {"state": args.state, "platform": sys.platform}
    try:
        for k in SCRUB:
            os.environ.pop(k, None)
        work = Path(tempfile.mkdtemp(prefix = f"sdcpp_{args.state}_",
                                     dir = os.environ.get("RUNNER_TEMP")))
        os.environ["UNSLOTH_STUDIO_HOME"] = str(work / "studio_home")
        backend_dir = args.checkout / "studio" / "backend"
        sys.path.insert(0, str(backend_dir))
        os.chdir(backend_dir)

        import torch

        obs["torch"] = torch.__version__
        obs["torch_gpu"] = bool(torch.cuda.is_available())

        from core.inference.diffusion_device import resolve_diffusion_device_target
        from core.inference import diffusion_engine_router as router

        backend = resolve_diffusion_device_target().backend
        obs["backend"] = backend
        inv = router._physical_inventory()
        obs["inventory"] = {
            "unknown": inv.get("unknown"),
            "unanswered": list(inv.get("unanswered") or ()),
            "devices": [{k: d.get(k) for k in ("vendor", "name", "index", "xpu_class")}
                        for d in inv.get("devices") or () if isinstance(d, dict)],
        }
        accel = router.image_install_accelerator(backend)
        obs["image_install_accelerator"] = accel

        spec = importlib.util.spec_from_file_location(
            f"sdcpp_installer_{args.state}", args.checkout / "studio" / "install_sd_cpp_prebuilt.py")
        installer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(installer)
        try:
            _repo, _rel, chosen = installer._resolve_with_fallback(accel, None, None)
            obs["asset"] = chosen
            obs["asset_repo"] = _repo
        except Exception:  # noqa: BLE001
            obs["asset_error"] = traceback.format_exc()[-1500:]
        root = work / "sd"
        t = time.time()
        try:
            cli = installer.install(install_dir = root, accelerator = accel)
            obs["install_s"] = round(time.time() - t, 1)
            obs["sd_cli"] = str(cli)
            obs["install_record"] = installer.read_install_record(root)
        except Exception:  # noqa: BLE001
            obs["install_error"] = traceback.format_exc()[-2000:]
            cli = None
        if cli:
            bindir = Path(cli).parent
            env = dict(os.environ)
            if sys.platform != "win32":
                env["LD_LIBRARY_PATH"] = str(bindir) + os.pathsep + env.get("LD_LIBRARY_PATH", "")
            else:
                env["PATH"] = str(bindir) + os.pathsep + env.get("PATH", "")
            r = subprocess.run([str(cli), "--list-devices"], env = env, capture_output = True,
                               text = True, encoding = "utf-8", errors = "replace", timeout = 180)
            obs["list_devices_rc"] = r.returncode
            obs["list_devices"] = (r.stdout + "\n" + r.stderr)[-3000:]
    except Exception:  # noqa: BLE001
        obs["error"] = traceback.format_exc()[-3000:]
    args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
