#!/usr/bin/env python3
"""Probe (Windows gfx1151): what llama.cpp's --fit believes about device memory versus what the
machine has, and where a full offload's device buffers land (RAM, commit, GPU counters).

Runs Unsloth's prebuilt llama-server (ROCm gfx1151) on one GGUF at a few context sizes; the
large ones are sized from this box's free RAM so the device plan lands between
"available RAM - --fit-target" and "available RAM". Stdlib only; torch readings if importable."""

from __future__ import annotations

import argparse
import ctypes
import ctypes.wintypes as wt
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

LLAMA_URL = (
    "https://github.com/unslothai/llama.cpp/releases/download/b12347-mix-241a0f3/"
    "app-b12347-mix-241a0f3-windows-x64-rocm-gfx1151.zip"
)
MODEL_URL = "https://huggingface.co/unsloth/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf"
KV_BYTES_PER_TOKEN = 36 * 2 * 8 * 128 * 2  # Qwen3-8B, f16 K + V
MIB = 1024**2


class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", wt.DWORD), ("dwMemoryLoad", wt.DWORD),
        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
    _fields_ = [
        ("cb", wt.DWORD), ("PageFaultCount", wt.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
        ("PrivateUsage", ctypes.c_size_t), ("PrivateWorkingSetSize", ctypes.c_size_t),
        ("SharedCommitUsage", ctypes.c_ulonglong),
    ]


def sysmem() -> dict:
    m = MEMORYSTATUSEX()
    m.dwLength = ctypes.sizeof(m)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return {
        "total_phys_mib": m.ullTotalPhys // MIB, "avail_phys_mib": m.ullAvailPhys // MIB,
        "commit_limit_mib": m.ullTotalPageFile // MIB, "commit_avail_mib": m.ullAvailPageFile // MIB,
    }


def procmem(pid: int) -> dict:
    h = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
    if not h:
        return {}
    try:
        c = PROCESS_MEMORY_COUNTERS_EX()
        c.cb = ctypes.sizeof(c)
        if not ctypes.windll.kernel32.K32GetProcessMemoryInfo(h, ctypes.byref(c), c.cb):
            return {}
        return {"private_mib": c.PrivateUsage // MIB, "ws_mib": c.WorkingSetSize // MIB,
                "peak_ws_mib": c.PeakWorkingSetSize // MIB,
                "private_ws_mib": c.PrivateWorkingSetSize // MIB}
    finally:
        ctypes.windll.kernel32.CloseHandle(h)


def ps(cmd: str, timeout: int = 60) -> str:
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output = True,
                           text = True, timeout = timeout, encoding = "utf-8", errors = "replace")
        return (r.stdout + r.stderr).strip()
    except Exception as e:  # noqa: BLE001
        return f"ERR {e}"


def gpu_counters(pid: int | None = None) -> str:
    paths = ["'\\GPU Adapter Memory(*)\\Dedicated Usage'", "'\\GPU Adapter Memory(*)\\Shared Usage'"]
    if pid:
        paths += [f"'\\GPU Process Memory(pid_{pid}*)\\Dedicated Usage'",
                  f"'\\GPU Process Memory(pid_{pid}*)\\Shared Usage'"]
    return ps("(Get-Counter -Counter " + ",".join(paths) + " -ErrorAction SilentlyContinue)"
              ".CounterSamples | ForEach-Object { \"$($_.Path)=$([math]::Round($_.CookedValue/1MB))MiB\" }")


def facts() -> dict:
    f = {"sysmem": sysmem()}
    f["physical_dimms"] = ps("Get-CimInstance Win32_PhysicalMemory | ForEach-Object { \"$($_.Capacity/1GB) GiB $($_.PartNumber)\" }")
    f["visible_kib"] = ps("(Get-CimInstance Win32_OperatingSystem).TotalVisibleMemorySize")
    f["adapters"] = ps(
        "Get-ItemProperty 'HKLM:\\SYSTEM\\CurrentControlSet\\Control\\Class\\{4d36e968-e325-11ce-bfc1-08002be10318}\\0*' "
        "-ErrorAction SilentlyContinue | ForEach-Object { \"$($_.DriverDesc) dedicated=$([math]::Round($_.'HardwareInformation.qwMemorySize'/1MB))MiB driver=$($_.DriverVersion)\" }")
    f["pagefile"] = ps("Get-CimInstance Win32_PageFileUsage | ForEach-Object { \"$($_.Name) alloc=$($_.AllocatedBaseSize)MB\" }; (Get-CimInstance Win32_ComputerSystem).AutomaticManagedPagefile")
    f["gpu_counters"] = gpu_counters()
    try:
        import torch  # noqa: PLC0415

        free, total = torch.cuda.mem_get_info(0)
        f["torch"] = {"version": torch.__version__, "free_mib": free // MIB, "total_mib": total // MIB,
                      "props_total_mib": torch.cuda.get_device_properties(0).total_memory // MIB}
    except Exception as e:  # noqa: BLE001
        f["torch"] = {"error": f"{type(e).__name__}: {e}"}
    return f


def fetch(url: str, dest: Path) -> None:
    if dest.is_file() and dest.stat().st_size > 0:
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout = 120) as r, open(tmp, "wb") as fh:
        while True:
            b = r.read(16 * MIB)
            if not b:
                break
            fh.write(b)
    tmp.replace(dest)


def find_server(root: Path) -> Path:
    hits = sorted(root.rglob("llama-server.exe"))
    if not hits:
        raise SystemExit(f"no llama-server.exe under {root}")
    return hits[0]


_KEEP = re.compile(r"ggml_cuda_init|Device 0:|MiB free|projected to use|will leave|offloaded|"
                   r"memory_breakdown|common_params_fit|fit|KV buffer size|model buffer size|"
                   r"compute buffer size|error|failed|out of memory", re.I)


def run_config(server: Path, model: Path, name: str, args: list[str], logdir: Path,
               port: int) -> dict:
    log = logdir / f"{name}.log"
    before = sysmem()
    cmd = [str(server), "-m", str(model), "--host", "127.0.0.1", "--port", str(port), *args]
    res: dict = {"name": name, "args": args, "before": before, "before_gpu": gpu_counters()}
    t0 = time.time()
    with open(log, "wb") as fh:
        p = subprocess.Popen(cmd, stdout = fh, stderr = subprocess.STDOUT,
                             creationflags = 0x00000200)
    peak_private = 0
    status = "timeout"
    min_avail = before["avail_phys_mib"]
    min_commit = before["commit_avail_mib"]
    while time.time() - t0 < 600:
        if p.poll() is not None:
            status = f"exited {p.returncode}"
            break
        pm = procmem(p.pid)
        peak_private = max(peak_private, pm.get("private_mib", 0))
        s = sysmem()
        min_avail = min(min_avail, s["avail_phys_mib"])
        min_commit = min(min_commit, s["commit_avail_mib"])
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout = 5) as r:
                if r.status == 200:
                    status = "ready"
                    break
        except Exception:  # noqa: BLE001
            pass
        time.sleep(2)
    res["status"] = status
    res["load_s"] = round(time.time() - t0, 1)
    if status == "ready":
        try:
            body = json.dumps({"prompt": "Hello", "n_predict": 8}).encode()
            req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", data = body,
                                         headers = {"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout = 300) as r:
                res["completion_ok"] = r.status == 200
        except Exception as e:  # noqa: BLE001
            res["completion_ok"] = f"ERR {e}"
        time.sleep(3)
        res["loaded_proc"] = procmem(p.pid)
        res["loaded_sys"] = sysmem()
        res["loaded_gpu"] = gpu_counters(p.pid)
        time.sleep(60)
        res["idle60_proc"] = procmem(p.pid)
        res["idle60_sys"] = sysmem()
    res["peak_private_mib"] = peak_private
    res["min_avail_phys_mib"] = min_avail
    res["min_commit_avail_mib"] = min_commit
    subprocess.run(["taskkill", "/PID", str(p.pid), "/T", "/F"], capture_output = True)
    try:
        p.wait(timeout = 60)
    except Exception:  # noqa: BLE001
        pass
    text = log.read_text(encoding = "utf-8", errors = "replace")
    res["log_lines"] = [ln for ln in text.splitlines() if _KEEP.search(ln)][:120]
    res["log_tail"] = text.splitlines()[-25:]
    time.sleep(15)
    res["after_sys"] = sysmem()
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--work", default = os.environ.get("AMD_CI_WORK") or os.environ.get("RUNNER_TEMP") or ".")
    a = ap.parse_args()
    work = Path(a.work) / "igpu_fit_cache"
    work.mkdir(parents = True, exist_ok = True)
    out: dict = {"state": a.state}
    try:
        out["facts"] = facts()
        zpath = work / "llama.zip"
        fetch(LLAMA_URL, zpath)
        ldir = work / "llama"
        if not ldir.is_dir():
            zipfile.ZipFile(zpath).extractall(ldir)
        server = find_server(ldir)
        out["server"] = str(server)
        model = work / "Qwen3-8B-Q4_K_M.gguf"
        fetch(MODEL_URL, model)
        r = subprocess.run([str(server), "--list-devices"], capture_output = True, text = True,
                           timeout = 120, errors = "replace")
        out["list_devices"] = (r.stdout + r.stderr).splitlines()[-12:]
        avail = sysmem()["avail_phys_mib"]
        weights = model.stat().st_size // MIB

        def ctx_for(device_mib: int) -> int:
            kv = max(0, device_mib - weights - 1024) * MIB
            return max(8192, (kv // KV_BYTES_PER_TOKEN) // 1024 * 1024)

        base = ["-c", "8192", "--parallel", "1", "-fa", "on", "--fit", "on"]
        configs = [
            ("full_none", [*base, "--load-mode", "none"]),
            ("full_mmap", [*base, "--load-mode", "mmap"]),
            ("full_mmap_lazy", [*base, "--load-mode", "mmap", "--lazy-mode", "on"]),
            ("full_none_again", [*base, "--load-mode", "none"]),
        ]
        mid = 0
        out["sizing"] = {"avail_at_start_mib": avail, "weights_mib": weights, "mid_ctx": mid,
                         "mid_kv_mib": mid * KV_BYTES_PER_TOKEN // MIB}
        logdir = Path(a.out).parent / f"llama_logs_{a.state}"
        logdir.mkdir(parents = True, exist_ok = True)
        out["runs"] = [run_config(server, model, n, args, logdir, 18316 + i)
                       for i, (n, args) in enumerate(configs)]
    except BaseException as e:  # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {e}"
    a.out.write_text(json.dumps(out, indent = 2), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
