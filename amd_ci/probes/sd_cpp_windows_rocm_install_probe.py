#!/usr/bin/env python3
"""Probe: what a checkout's sd.cpp installer and backend do on a real Windows AMD host WITHOUT the
AMD HIP SDK runtime on PATH, when ROCm is asked for while a working Vulkan build is installed.

The reported failure: the upstream (leejet) win-rocm zip imports hipblas.dll and does not ship it;
the installer extracted it, swept the working Vulkan build as "superseded", and the new binaries
then died in the Windows loader (exit 0xC0000135), leaving nothing runnable.

Real prebuilts, real loader. The only arrangement: PATH entries holding hipblas.dll are dropped from
this process's PATH (recorded), so a box with the HIP SDK still reproduces a host without it.

Per state, against that checkout's own modules:
  1. seed: install(accelerator="vulkan") from the pinned mirror release; start its sd-cli / sd-server;
  2. install(accelerator="rocm") over it; record what raised, whether the Vulkan binaries survive,
     and how every binary left in the tree starts;
  3. backend: preferred_accelerator("rocm") on this host, the Windows HIP DLL check when present;
  4. fresh tree: ensure_sd_server_binary(accelerator="rocm"); record what it returns and how it starts.

Observes only; criteria/sd_cpp_windows_rocm_install.py judges. Writes JSON via --out only.
Public GitHub release downloads only, no token is used or sent.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path


def _hide_hip_sdk() -> dict:
    parts = [p for p in os.environ.get("PATH", "").split(os.pathsep) if p]
    hidden = [p for p in parts if os.path.isfile(os.path.join(p.strip('"'), "hipblas.dll"))]
    os.environ["PATH"] = os.pathsep.join(p for p in parts if p not in hidden)
    system32 = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32")
    return {
        "hidden_path_entries": hidden,
        "hip_path_env": os.environ.get("HIP_PATH"),
        "hipblas_in_system32": os.path.isfile(os.path.join(system32, "hipblas.dll")),
        "amdhip64_in_system32": sorted(
            n for n in os.listdir(system32) if n.lower().startswith("amdhip64")
        )
        if os.path.isdir(system32)
        else [],
        "hipblas_on_path_after_hide": any(
            os.path.isfile(os.path.join(p.strip('"'), "hipblas.dll"))
            for p in os.environ.get("PATH", "").split(os.pathsep)
            if p
        ),
    }


def _start(binary) -> dict:
    """``binary --help`` with the binary's dir first on PATH, loader dialogs suppressed."""
    if not binary or not os.path.isfile(str(binary)):
        return {"path": str(binary) if binary else None, "exists": False}
    env = dict(os.environ)
    env["PATH"] = str(Path(binary).parent) + os.pathsep + env.get("PATH", "")
    previous = None
    try:
        import ctypes

        previous = ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002 | 0x8000)
    except Exception:  # noqa: BLE001
        pass
    try:
        proc = subprocess.run(
            [str(binary), "--help"],
            capture_output = True,
            text = True,
            encoding = "utf-8",
            errors = "replace",
            timeout = 60,
            env = env,
            stdin = subprocess.DEVNULL,
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        rc = proc.returncode
        out = (proc.stdout or "") + (proc.stderr or "")
        err = None
    except Exception as exc:  # noqa: BLE001
        rc, out, err = None, "", repr(exc)
    finally:
        if previous is not None:
            try:
                import ctypes

                ctypes.windll.kernel32.SetErrorMode(previous)
            except Exception:  # noqa: BLE001
                pass
    code = None if rc is None else (rc + (1 << 32) if rc < 0 else rc)
    return {
        "path": str(binary),
        "exists": True,
        "rc": rc,
        "rc_hex": None if code is None else f"0x{code:08X}",
        "dll_not_found": code == 0xC0000135,
        "output_head": out[:300],
        "error": err,
    }


def _binaries(root: Path) -> list[str]:
    out = []
    for name in ("sd-cli.exe", "sd-server.exe", "sd-cli", "sd-server"):
        out += [str(p) for p in root.rglob(name) if p.is_file()]
    return sorted(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True)
    ap.add_argument("--cache", default = None, help = "shared download cache (state-independent)")
    args = ap.parse_args()

    obs: dict = {"state": args.state, "platform": sys.platform}
    out_path = Path(args.out)

    def dump():
        out_path.write_text(json.dumps(obs, indent = 2, default = str), encoding = "utf-8")

    for var in ("GH_TOKEN", "GITHUB_TOKEN", "HF_TOKEN"):
        os.environ.pop(var, None)
    os.environ.pop("UNSLOTH_SD_CPP_WINDOWS_ROCM", None)
    os.environ.pop("UNSLOTH_SD_CPP_VERIFY_INSTALL", None)
    os.environ.pop("UNSLOTH_DIFFUSION_SD_CPP_VULKAN_FALLBACK", None)
    obs["host"] = _hide_hip_sdk()

    pip = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "httpx", "structlog"],
        capture_output = True,
        text = True,
    )
    obs["pip_rc"] = pip.returncode

    temp = Path(os.environ.get("RUNNER_TEMP") or os.environ.get("TEMP") or ".")
    work = temp / "amd-ci-moss" / args.state
    if work.exists():
        shutil.rmtree(work, ignore_errors = True)
    work.mkdir(parents = True)
    cache = Path(args.cache) if args.cache else temp / "amd-ci-moss" / "cache"
    cache.mkdir(parents = True, exist_ok = True)

    checkout = Path(args.checkout)
    sys.path.insert(0, str(checkout / "studio" / "backend"))
    sys.path.insert(0, str(checkout / "studio"))
    os.environ["UNSLOTH_STUDIO_HOME"] = str(work / "home1" / "studio")
    (work / "home1" / "studio").mkdir(parents = True)

    try:
        import install_sd_cpp_prebuilt as inst

        obs["installer_file"] = inst.__file__
        obs["default_tag"] = inst.DEFAULT_TAG
        real_download = inst._download

        def cached_download(url, dest, *a, **k):
            hit = cache / url.rsplit("/", 1)[-1]
            if not hit.is_file():
                real_download(url, hit, *a, **k)
            shutil.copyfile(hit, dest)

        inst._download = cached_download

        # 1. seed the working Vulkan build
        root = inst.default_install_dir()
        obs["root"] = str(root)
        try:
            inst.install(install_dir = root, accelerator = "vulkan")
            obs["seed_error"] = None
        except Exception as exc:  # noqa: BLE001
            obs["seed_error"] = repr(exc)
        seeded = _binaries(root)
        obs["seed_binaries"] = seeded
        obs["seed_starts"] = [_start(b) for b in seeded]
        obs["seed_record"] = inst.read_install_record(root)
        dump()

        # 2. the ROCm install over it
        try:
            inst.install(install_dir = root, accelerator = "rocm")
            obs["rocm_install"] = {"raised": None}
        except Exception as exc:  # noqa: BLE001
            obs["rocm_install"] = {"raised": type(exc).__name__, "message": str(exc)[:600]}
        obs["after_rocm_seed_binaries_left"] = [b for b in seeded if os.path.isfile(b)]
        left = _binaries(root)
        obs["after_rocm_binaries"] = left
        obs["after_rocm_starts"] = [_start(b) for b in left]
        obs["after_rocm_record"] = inst.read_install_record(root)
        located = inst._locate_sd_server(root)
        obs["after_rocm_located_server"] = str(located) if located else None
        dump()

        # 3. the backend's view of this host
        import core.inference.sd_cpp_backend as bk

        obs["backend_file"] = bk.__file__
        obs["preferred_for_rocm"] = bk.preferred_accelerator("rocm")
        fn = getattr(bk, "windows_hip_runtime_missing_dlls", None)
        obs["hip_missing_dlls"] = fn() if fn else "absent at this state"
        if located:
            obs["backend_says_server_runnable"] = bk._server_binary_runnable(str(located))
        dump()

        # 4. fresh tree, the backend's own ensure for a ROCm request
        os.environ["UNSLOTH_STUDIO_HOME"] = str(work / "home2" / "studio")
        (work / "home2" / "studio").mkdir(parents = True)
        root2 = inst.default_install_dir()
        obs["root2"] = str(root2)
        try:
            server = bk.ensure_sd_server_binary(allow_install = True, accelerator = "rocm")
            obs["ensure_error"] = None
        except Exception as exc:  # noqa: BLE001
            server = None
            obs["ensure_error"] = repr(exc)
        obs["ensure_server"] = server
        obs["ensure_server_start"] = _start(server) if server else None
        obs["ensure_record"] = inst.read_install_record(root2)
        obs["ensure_binaries"] = _binaries(root2)
        obs["preferred_for_rocm_after_ensure"] = bk.preferred_accelerator("rocm")
    except Exception as exc:  # noqa: BLE001
        obs["probe_error"] = repr(exc)
        obs["probe_traceback"] = traceback.format_exc()[-2000:]
    dump()
    shutil.rmtree(work, ignore_errors = True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
