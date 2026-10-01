#!/usr/bin/env python3
"""Probe: what does this checkout's llama.cpp SOURCE build do on a ROCm host?

Observes only. Loads the state's unsloth_zoo/llama_cpp.py by file (no package
import), forces the source build (UNSLOTH_LLAMA_FORCE_COMPILE=1), and calls
install_llama_cpp(gpu_support=True) into a fresh folder. Records every shell
command install_llama_cpp ran (through its own try_execute), the cmake configure
line, the outcome, the CMakeCache GPU keys before the build dir is removed, and
what the produced binaries link against. Judging is criteria/llama_cpp_hip_build.py.

Pairs with criteria/llama_cpp_hip_build.py.
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path


def _tail(s: str, n: int) -> str:
    s = s or ""
    return s if len(s) <= n else "...[truncated]...\n" + s[-n:]


def _run(cmd: list[str], timeout: int = 60) -> dict:
    try:
        p = subprocess.run(cmd, capture_output = True, text = True, timeout = timeout,
                           stdin = subprocess.DEVNULL)
        return {"rc": p.returncode, "out": _tail((p.stdout or "") + (p.stderr or ""), 4000)}
    except Exception as e:  # noqa: BLE001
        return {"rc": None, "out": f"{type(e).__name__}: {e}"}


def _cmake_cache_keys(cache: Path) -> dict:
    keys = {}
    want = ("GGML_HIP", "GGML_CUDA", "GPU_TARGETS", "AMDGPU_TARGETS", "CMAKE_HIP_COMPILER",
            "CMAKE_HIP_ARCHITECTURES", "CMAKE_CUDA_COMPILER", "hip_DIR", "hipblas_DIR", "rocblas_DIR")
    try:
        for line in cache.read_text(encoding = "utf-8", errors = "replace").splitlines():
            name = line.split(":", 1)[0].split("=", 1)[0]
            if name in want:
                keys[name] = line[:400]
    except Exception as e:  # noqa: BLE001
        keys["_error"] = f"{type(e).__name__}: {e}"
    return keys


def toolkit_facts() -> dict:
    f: dict = {}
    rp = os.environ.get("ROCM_PATH")
    f["env_ROCM_PATH"] = rp
    f["env_HIP_PATH"] = os.environ.get("HIP_PATH")
    root = rp or "/opt/rocm"
    f["rocm_root_checked"] = root
    f["rocm_root_exists"] = os.path.isdir(root)
    try:
        f["rocm_root_ls"] = sorted(os.listdir(root))[:60] if os.path.isdir(root) else None
    except Exception as e:  # noqa: BLE001
        f["rocm_root_ls"] = f"{type(e).__name__}: {e}"
    f["opt_rocm_realpath"] = os.path.realpath("/opt/rocm") if os.path.exists("/opt/rocm") else None
    f["hip_clang_exists"] = os.path.exists(os.path.join(root, "llvm", "bin", "clang"))
    for tool in ("hipconfig", "hipcc", "nvcc", "cmake", "gcc", "rocminfo", "amd-smi", "rocm-sdk"):
        f[f"which_{tool}"] = shutil.which(tool)
    venv_sdk = os.path.join(os.path.dirname(sys.executable), "rocm-sdk")
    f["venv_rocm_sdk"] = venv_sdk if os.path.exists(venv_sdk) else None
    if shutil.which("hipconfig"):
        f["hipconfig_version"] = _run(["hipconfig", "--version"])
        f["hipconfig_path"] = _run(["hipconfig", "--path"])
    if shutil.which("cmake"):
        f["cmake_version"] = _run(["cmake", "--version"])["out"].splitlines()[:1]
    cmake_cfgs = []
    for base in {root, "/opt/rocm", "/usr"}:
        for pat in ("lib/cmake/hip/hip-config.cmake", "lib/cmake/hipblas/hipblas-config.cmake",
                    "lib/cmake/rocblas/rocblas-config.cmake", "lib*/cmake/hip/hip-config.cmake"):
            cmake_cfgs += glob.glob(os.path.join(base, pat))
    f["rocm_cmake_configs"] = sorted(set(cmake_cfgs))
    return f


def torch_facts(mod) -> dict:
    t: dict = {}
    try:
        import torch  # noqa: PLC0415
        t["torch"] = torch.__version__
        t["hip"] = getattr(torch.version, "hip", None)
        t["cuda"] = getattr(torch.version, "cuda", None)
        t["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            t["gcnArchName"] = getattr(torch.cuda.get_device_properties(0), "gcnArchName", None)
    except Exception as e:  # noqa: BLE001
        t["error"] = f"{type(e).__name__}: {e}"
    try:
        t["detect_gpu_target"] = list(mod._detect_gpu_target() or []) or None
    except Exception as e:  # noqa: BLE001
        t["detect_gpu_target_error"] = f"{type(e).__name__}: {e}"
    t["has__gpu_cmake_flags"] = hasattr(mod, "_gpu_cmake_flags")
    return t


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True, type = Path)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--work-root", default = None)
    args = ap.parse_args()

    obs: dict = {"state": args.state, "checkout": str(args.checkout)}
    try:
        obs["checkout_head"] = subprocess.run(["git", "-C", str(args.checkout), "rev-parse", "HEAD"],
                                              capture_output = True, text = True).stdout.strip()
    except Exception:  # noqa: BLE001
        pass

    work_root = Path(args.work_root or os.environ.get("TMPDIR") or tempfile.gettempdir())
    work_root.mkdir(parents = True, exist_ok = True)
    arm_dir = Path(tempfile.mkdtemp(prefix = f"{args.state}_", dir = str(work_root)))
    home = arm_dir / "home"
    home.mkdir()
    folder = arm_dir / "llama.cpp"  # does not exist yet: fresh clone + build
    obs["llama_cpp_folder"] = str(folder)

    # Before the module loads: UNSLOTH_HOME is derived from HOME at import.
    os.environ["HOME"] = str(home)
    os.environ["UNSLOTH_LLAMA_FORCE_COMPILE"] = "1"
    os.environ["UNSLOTH_IS_PRESENT"] = "1"
    os.environ.setdefault("VIRTUAL_ENV", sys.prefix)
    os.chdir(arm_dir)

    module_path = args.checkout / "unsloth_zoo" / "llama_cpp.py"
    spec = importlib.util.spec_from_file_location(f"llama_cpp_probe_{args.state}", module_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception as e:  # noqa: BLE001
        obs["load_error"] = _tail(f"{type(e).__name__}: {e}\n{traceback.format_exc()}", 4000)
        args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
        return 0
    obs["module_file"] = str(module_path)

    obs["toolkit"] = toolkit_facts()
    obs["torch"] = torch_facts(mod)
    try:
        obs["build_requirements_missing"] = list(mod.check_build_requirements())
    except Exception as e:  # noqa: BLE001
        obs["build_requirements_error"] = f"{type(e).__name__}: {e}"

    commands: list[dict] = []
    obs["commands"] = commands
    orig = mod.try_execute

    def recording_try_execute(command, *a, **kw):
        cwd = kw.get("cwd")
        if cwd is None and len(a) >= 4:
            cwd = a[3]
        entry = {"cmd": _tail(str(command), 3000), "cwd": cwd}
        build = Path(cwd or ".") / "build"
        if str(command).strip() == "rm -rf build" and (build / "CMakeCache.txt").is_file():
            obs["cmake_cache_before_rm"] = _cmake_cache_keys(build / "CMakeCache.txt")
            obs["ggml_hip_dir_built"] = (build / "ggml" / "src" / "ggml-hip").is_dir()
            obs["ggml_cuda_dir_built"] = (build / "ggml" / "src" / "ggml-cuda").is_dir()
            obs["libggml_hip_archives"] = [str(p.relative_to(build)) for p in build.rglob("libggml-hip*")][:10]
        t0 = time.time()
        try:
            orig(command, *a, **kw)
            entry["ok"] = True
        except BaseException as e:
            entry["ok"] = False
            entry["error"] = _tail(f"{type(e).__name__}: {e}", 6000)
            if "-B build" in str(command) and (build / "CMakeCache.txt").is_file():
                obs["cmake_cache_after_fail"] = _cmake_cache_keys(build / "CMakeCache.txt")
            raise
        finally:
            entry["secs"] = round(time.time() - t0, 1)
            commands.append(entry)

    mod.try_execute = recording_try_execute

    t0 = time.time()
    try:
        ret = mod.install_llama_cpp(llama_cpp_folder = str(folder), gpu_support = True, print_output = True)
        obs["install_returned"] = [str(x) for x in ret] if isinstance(ret, (tuple, list)) else str(ret)
        obs["install_ok"] = True
    except BaseException as e:  # noqa: BLE001
        msg = f"{type(e).__name__}: {e}"
        obs["install_ok"] = False
        obs["exception_head"] = msg[:2500]
        obs["exception_tail"] = _tail(msg, 5000)
    obs["install_secs"] = round(time.time() - t0, 1)

    cfg = [c for c in commands if str(c["cmd"]).lstrip().startswith("cmake . -B build")]
    obs["configure_line"] = cfg[0]["cmd"] if cfg else None
    obs["configure_ok"] = cfg[0].get("ok") if cfg else None
    blds = [c for c in commands if str(c["cmd"]).lstrip().startswith("cmake --build build")]
    obs["build_cmd_ok"] = blds[0].get("ok") if blds else None
    obs["build_secs"] = blds[0].get("secs") if blds else None

    q = folder / "llama-quantize"
    obs["llama_quantize_exists"] = q.is_file()
    obs["binaries"] = sorted(p.name for p in folder.glob("llama-*") if p.is_file())[:30] if folder.is_dir() else []
    if q.is_file():
        ldd = _run(["ldd", str(q)])
        obs["ldd_llama_quantize"] = ldd
        out = ldd["out"]
        obs["links"] = {k: (k in out) for k in ("libamdhip64", "libhipblas", "librocblas", "libcudart", "libcublas")}
        srv = folder / "llama-server"
        if srv.is_file():
            obs["llama_server_ldd_hip"] = "libamdhip64" in _run(["ldd", str(srv)])["out"]
            obs["list_devices"] = _run([str(srv), "--list-devices"], timeout = 120)
    obs["toolkit_after"] = {"which_hipconfig": shutil.which("hipconfig"), "which_nvcc": shutil.which("nvcc")}

    args.out.write_text(json.dumps(obs, indent = 2, default = str), encoding = "utf-8")
    # Free disk for the next arm; the observations hold the evidence.
    shutil.rmtree(arm_dir / "llama.cpp" / "build", ignore_errors = True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
