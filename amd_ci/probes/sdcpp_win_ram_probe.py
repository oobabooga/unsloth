#!/usr/bin/env python3
"""Probe: Studio's sd.cpp offload CPU VAE tile on THIS host, from the state's own host RAM readings.

Observes only; criteria/sdcpp_win_ram*.py judge.

Per state and per Python condition:
  * "nopsutil": psutil made unimportable (sys.modules["psutil"] = None), the bare venv the PR 18999 run used;
  * "psutil":   psutil==7.2.2 importable, as Studio's installer pins it (no-torch-runtime.txt);
the chooser subprocess records the raw OS readings (psutil, GlobalMemoryStatusEx, K32GetPerformanceInfo,
/proc/meminfo), every reading the state's code returns (host_memory.system_*_mib, diffusion_memory._system_memory_mib,
diffusion_small_host.host_ram_mib / host_commit_mib), the spare MiB and tile `_cpu_vae_tile_size` derives, and the
flags `_offload_vae_flags(OFFLOAD_MODEL, ..., klass="vulkan")` emits. Then one sd-cli render per distinct flag set at
--sizes, plus the untiled Vulkan reference of the same latent (cached by argv across states / conditions).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sdcpp_whole_vae_probe as W  # noqa: E402

PSUTIL_PIN = "psutil==7.2.2"
CONDITIONS = ("nopsutil", "psutil")

CHOOSER = r'''
import json, logging, os, sys, types
cond = sys.argv[9]
if cond == "nopsutil":
    sys.modules["psutil"] = None
sys.path.insert(0, sys.argv[1])
klass, binary, desc, dit, vae, llm = sys.argv[2:8]
res = {"cond": cond, "platform": sys.platform, "klass": klass}
MIB = 1 << 20

def raw():
    out = {}
    try:
        import psutil
        vm = psutil.virtual_memory()
        out["psutil"] = {"version": psutil.__version__, "total_mib": vm.total // MIB, "available_mib": vm.available // MIB}
    except Exception as e:
        out["psutil"] = f"{type(e).__name__}: {e}"
    if sys.platform == "win32":
        import ctypes
        class MS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_uint32), ("dwMemoryLoad", ctypes.c_uint32)] + [
                (n, ctypes.c_ulonglong) for n in ("TotalPhys", "AvailPhys", "TotalPageFile", "AvailPageFile",
                                                 "TotalVirtual", "AvailVirtual", "AvailExtendedVirtual")]
        s = MS(); s.dwLength = ctypes.sizeof(MS)
        ok = ctypes.WinDLL("kernel32").GlobalMemoryStatusEx(ctypes.byref(s))
        out["GlobalMemoryStatusEx"] = {"ok": bool(ok), "TotalPhys_mib": s.TotalPhys // MIB, "AvailPhys_mib": s.AvailPhys // MIB,
                                       "TotalPageFile_mib": s.TotalPageFile // MIB, "AvailPageFile_mib": s.AvailPageFile // MIB}
        class PI(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_uint32)] + [(n, ctypes.c_size_t) for n in (
                "CommitTotal", "CommitLimit", "CommitPeak", "PhysicalTotal", "PhysicalAvailable", "SystemCache",
                "KernelTotal", "KernelPaged", "KernelNonpaged", "PageSize")] + [
                (n, ctypes.c_uint32) for n in ("HandleCount", "ProcessCount", "ThreadCount")]
        p = PI(); p.cb = ctypes.sizeof(PI)
        ok = ctypes.WinDLL("kernel32").K32GetPerformanceInfo(ctypes.byref(p), p.cb)
        pg = p.PageSize or 4096
        out["K32GetPerformanceInfo"] = {"ok": bool(ok), "CommitLimit_mib": p.CommitLimit * pg // MIB,
                                        "CommitTotal_mib": p.CommitTotal * pg // MIB,
                                        "PhysicalAvailable_mib": p.PhysicalAvailable * pg // MIB}
    else:
        try:
            mi = {l.split(":")[0]: int(l.split()[1]) // 1024 for l in open("/proc/meminfo", encoding="utf-8")}
            out["proc_meminfo"] = {"MemTotal_mib": mi.get("MemTotal"), "MemAvailable_mib": mi.get("MemAvailable")}
        except Exception as e:
            out["proc_meminfo"] = f"{type(e).__name__}: {e}"
    return out

res["raw"] = raw()

def call(name, fn):
    try:
        v = fn()
        res["state_readings"][name] = list(v) if isinstance(v, tuple) else v
    except Exception as e:
        res["state_readings"][name] = f"{type(e).__name__}: {str(e)[:200]}"

res["state_readings"] = {}
try:
    from utils import host_memory as hm
    call("host_memory.system_total_mib", hm.system_total_mib)
    call("host_memory.system_available_mib", hm.system_available_mib)
    res["has_windows_physical_mib"] = hasattr(hm, "windows_physical_mib")
except Exception as e:
    res["state_readings"]["utils.host_memory"] = f"{type(e).__name__}: {e}"
from core.inference import diffusion_memory as dm, diffusion_small_host as sh
call("diffusion_memory._system_memory_mib", dm._system_memory_mib)
call("diffusion_memory._available_system_memory_mib", dm._available_system_memory_mib)
call("diffusion_memory._host_ram_capacity_mib", dm._host_ram_capacity_mib)
call("diffusion_small_host.host_ram_mib", sh.host_ram_mib)
call("diffusion_small_host.host_commit_mib", sh.host_commit_mib)
import core.inference.sd_cpp_args as a
from core.inference.sd_cpp_multigpu import TEXT_ENCODER_FIELDS, file_mib
files = types.SimpleNamespace(diffusion_model = dit, vae = vae, llm = llm)
res["tile_in_pixels"] = bool(a.sd_cpp_vae_tile_size_in_pixels(binary))
weights = sum(file_mib(getattr(files, f, None)) for f in ("diffusion_model", "vae", *TEXT_ENCODER_FIELDS))
res["weights_mib"] = weights
try:
    total, avail = sh.host_ram_mib()
    commit = sh.host_commit_mib()
    use = avail if commit is None or avail is None else min(avail, commit)
    res["reserve_mib"] = sh.host_reserve_mib(total)
    res["spare_mib"] = None if use is None else use - weights - res["reserve_mib"]
except Exception as e:
    res["spare_mib"] = f"{type(e).__name__}: {e}"
logs = []
class H(logging.Handler):
    def emit(self, r):
        logs.append(r.getMessage())
logging.getLogger().addHandler(H()); logging.getLogger().setLevel(logging.INFO)
from core.inference import sd_cpp_backend as b
res["cpu_vae_tile_size"] = b._cpu_vae_tile_size(files, binary)
res["flags"] = b._offload_vae_flags(a.OFFLOAD_MODEL, files, binary, klass, None, desc)
res["log"] = [m for m in logs if "sd_cpp.vae" in m][-4:]
print("CHOOSER " + json.dumps(res, default = str))
'''


def ensure_psutil() -> str:
    p = subprocess.run([sys.executable, "-m", "pip", "install", "-q", PSUTIL_PIN], capture_output = True, text = True,
                       errors = "replace", env = W.clean_env())
    return f"pip install {PSUTIL_PIN}: rc={p.returncode} {(p.stdout + p.stderr)[-300:]}"


def choose(checkout: Path, klass: str, binary: str, desc, models: dict, cond: str) -> dict:
    backend = checkout / "studio" / "backend"
    p = subprocess.run([sys.executable, "-c", CHOOSER, str(backend), klass, binary, desc or "", models["dit"],
                        models["vae"], models["llm"], "flux.2-klein", cond], capture_output = True, text = True,
                       encoding = "utf-8", errors = "replace", cwd = str(backend), env = W.clean_env(), timeout = 600)
    line = [x for x in p.stdout.splitlines() if x.startswith("CHOOSER ")]
    if not line:
        return {"cond": cond, "error": (p.stdout + p.stderr)[-2000:]}
    return json.loads(line[-1][8:])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True, type = Path)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--shared", default = "")
    ap.add_argument("--tag", default = "master-982-0518b34-u27b9ddc")
    ap.add_argument("--sizes", default = "2048")
    ap.add_argument("--steps", type = int, default = 4)
    a = ap.parse_args()
    for k in W.TOKEN_VARS:
        os.environ.pop(k, None)
    shared = Path(a.shared or (Path(os.environ.get("AMD_CI_WORK") or a.out.parent) / "sdcpp_win_ram_shared")).resolve()
    sizes = [int(s) for s in a.sizes.split(",") if s]
    obs: dict = {"state": a.state, "platform": sys.platform, "sizes": sizes, "steps": a.steps, "arms": []}
    try:
        prep = W.prepare(shared, a.tag, False, a.checkout)
    except Exception as e:  # noqa: BLE001
        prep = {"error": f"{type(e).__name__}: {e}"}
    obs["prepared"] = prep
    obs["psutil_install"] = ensure_psutil()
    bins = prep.get("binaries") or {}
    if not prep.get("models_ok") or "vulkan" not in bins:
        W.wjson(a.out, obs)
        return 0
    desc = (prep["bundles"]["vulkan"] or {}).get("description")
    obs["choice"] = {c: choose(a.checkout, "vulkan", bins["vulkan"], desc, prep["models"], c) for c in CONDITIONS}
    W.wjson(a.out, obs)
    runner = W.Runner(shared, prep["models"], bins, a.steps)
    for size in sizes:
        ref_flags = W.untiled_of(W.GPU_COMMON)
        ref = runner.run("vulkan", size, ref_flags, 0)
        obs.setdefault("runs", {})[ref["key"]] = ref
        for cond in CONDITIONS:
            flags = (obs["choice"][cond] or {}).get("flags")
            if not flags:
                continue
            rec = runner.run("vulkan", size, list(flags), 0)
            obs["runs"][rec["key"]] = rec
            obs["arms"].append({
                "name": f"state_{a.state}_{cond}", "cond": cond, "size": size, "flags": flags,
                "tile_passed": W.tile_of(flags), "key": rec["key"], "ref_key": ref["key"],
                "psnr_vs_untiled": (W.psnr(runner.dir / rec["png"], runner.dir / ref["png"])
                                    if rec.get("png") and ref.get("png") else None),
            })
            W.wjson(a.out, obs)
    keep = a.out.parent / "win_ram_runs"
    keep.mkdir(parents = True, exist_ok = True)
    for k in set(obs.get("runs", {})):
        for ext in ("log", "png"):
            src = runner.dir / f"{k}.{ext}"
            if src.is_file() and not (keep / src.name).is_file():
                (keep / src.name).write_bytes(src.read_bytes())
    W.wjson(a.out, obs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
