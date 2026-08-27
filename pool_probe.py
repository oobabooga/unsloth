#!/usr/bin/env python3
"""Measure the two memory pools a ROCm APU launch can choose between.

Answers, on real gfx1151 hardware, the questions review of unslothai/unsloth#9884
turns on:

1. What do props.total_memory / mem_get_info / MemAvailable read at rest.
2. What the PR's own helpers answer on this host.
3. Whether another process occupying the carve-out lowers the device's FREE
   reading while leaving total_memory and host MemAvailable untouched.
4. Whether exhausting each pool fails the same way: a hipMalloc over-allocation
   versus a touched hipMallocManaged over-allocation.
"""

import argparse
import ctypes
import glob
import json
import os
import subprocess
import sys
import time

MIB = 1024 * 1024
GIB = 1024 ** 3


def meminfo_mib(key):
    try:
        with open("/proc/meminfo", encoding = "utf-8") as f:
            for line in f:
                if line.startswith(key + ":"):
                    return int(line.split()[1]) // 1024
    except Exception:
        pass
    return None


def sysfs_vram():
    out = {}
    for name in ("mem_info_vram_total", "mem_info_vram_used", "mem_info_gtt_total"):
        for path in sorted(glob.glob(f"/sys/class/drm/card*/device/{name}")):
            try:
                with open(path, encoding = "utf-8") as f:
                    out.setdefault(name, []).append(int(f.read().strip()) // MIB)
            except Exception:
                pass
    return out


def torch_devices():
    import torch

    rows = []
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        free, total = torch.cuda.mem_get_info(i)
        rows.append({
            "ordinal": i,
            "name": getattr(p, "name", None),
            "gcnArchName": getattr(p, "gcnArchName", None),
            "is_integrated": bool(getattr(p, "is_integrated", False)),
            "props_total_mib": int(getattr(p, "total_memory", 0)) // MIB,
            "mem_get_info_free_mib": free // MIB,
            "mem_get_info_total_mib": total // MIB,
        })
    return rows


def repo_helpers(repo):
    backend = os.path.join(repo, "studio", "backend")
    sys.path.insert(0, backend)
    from core.inference.llama_cpp import LlamaCppBackend as B

    return {
        "_rocm_selected_pool_mib()": B._rocm_selected_pool_mib(),
        "_rocm_selected_pool_mib([0])": B._rocm_selected_pool_mib([0]),
        "_available_system_memory_mib()": B._available_system_memory_mib(),
        "_unified_memory_would_help()": B._unified_memory_would_help(),
        "_amd_apu_wants_unified_memory()": B._amd_apu_wants_unified_memory(),
        "_rocm_unified_memory_gpu_ids()": sorted(B._rocm_unified_memory_gpu_ids()),
    }


# ---------------------------------------------------------------- child modes

def child_hold(gib, ready_path):
    """Hold ``gib`` of plain device memory until killed."""
    import torch

    buf = torch.empty(int(gib * GIB), dtype = torch.uint8, device = "cuda")
    buf[0] = 1
    torch.cuda.synchronize()
    with open(ready_path, "w", encoding = "utf-8") as f:
        f.write("ready\n")
    time.sleep(600)


def child_hipmalloc(mib):
    """Over-allocate plain device memory; report how it fails."""
    import torch

    try:
        torch.empty(int(mib) * MIB, dtype = torch.uint8, device = "cuda")
    except Exception as exc:  # noqa: BLE001 - the failure mode IS the result
        print(f"RESULT hipMalloc raised {type(exc).__name__}: {str(exc)[:200]}")
        return 0
    print("RESULT hipMalloc unexpectedly succeeded")
    return 0


def child_managed(mib):
    """Over-allocate MANAGED memory and touch it; report how it fails."""
    try:
        with open("/proc/self/oom_score_adj", "w", encoding = "utf-8") as f:
            f.write("1000")
    except Exception as exc:  # noqa: BLE001
        print(f"note: could not raise oom_score_adj: {exc}")

    import torch  # initializes HIP before we reach into libamdhip64

    torch.cuda.is_available()
    lib = ctypes.CDLL("libamdhip64.so")
    lib.hipMallocManaged.argtypes = [
        ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_uint
    ]
    lib.hipMallocManaged.restype = ctypes.c_int
    ptr = ctypes.c_void_p()
    size = int(mib) * MIB
    rc = lib.hipMallocManaged(ctypes.byref(ptr), ctypes.c_size_t(size), ctypes.c_uint(1))
    print(f"RESULT hipMallocManaged({mib} MiB) rc={rc} ptr={ptr.value}")
    if rc != 0 or not ptr.value:
        return 0
    step = 512 * MIB
    done = 0
    while done < size:
        n = min(step, size - done)
        ctypes.memset(ptr.value + done, 1, n)
        done += n
        print(f"touched {done // MIB} MiB, MemAvailable={meminfo_mib('MemAvailable')} MiB",
              flush = True)
    print("RESULT managed allocation fully touched without being killed")
    return 0


# ---------------------------------------------------------------------- driver

def run_child(mode, arg, extra = None):
    cmd = [sys.executable, os.path.abspath(__file__), "--child", mode, "--arg", str(arg)]
    if extra:
        cmd += extra
    proc = subprocess.run(cmd, capture_output = True, text = True, timeout = 1800)
    return {
        "returncode": proc.returncode,
        "stdout": proc.stdout[-4000:],
        "stderr": proc.stderr[-2000:],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required = True)
    ap.add_argument("--repo", default = None)
    ap.add_argument("--child", default = None)
    ap.add_argument("--arg", default = "0")
    ap.add_argument("--ready", default = "/tmp/hold_ready")
    args = ap.parse_args()

    if args.child == "hold":
        return child_hold(float(args.arg), args.ready)
    if args.child == "hipmalloc":
        return child_hipmalloc(float(args.arg))
    if args.child == "managed":
        return child_managed(float(args.arg))

    report = {}

    def emit():
        with open(args.out, "w", encoding = "utf-8") as f:
            json.dump(report, f, indent = 2, default = str)

    report["host"] = {
        "MemTotal_mib": meminfo_mib("MemTotal"),
        "MemAvailable_mib": meminfo_mib("MemAvailable"),
        "sysfs": sysfs_vram(),
        "rocm_version": (open("/opt/rocm/.info/version", encoding = "utf-8").read().strip()
                         if os.path.exists("/opt/rocm/.info/version") else None),
        "kernel": os.uname().release,
    }
    emit()

    import torch
    report["torch"] = {
        "version": torch.__version__,
        "hip": getattr(torch.version, "hip", None),
        "is_available": torch.cuda.is_available(),
        "devices": torch_devices(),
    }
    emit()

    if args.repo:
        try:
            report["repo_helpers_at_pr_head"] = repo_helpers(args.repo)
        except Exception as exc:  # noqa: BLE001
            report["repo_helpers_at_pr_head"] = {"error": f"{type(exc).__name__}: {exc}"}
        emit()

    # -- 3. does an occupied carve-out move the readings the decision uses? ----
    ready = "/tmp/hold_ready"
    if os.path.exists(ready):
        os.unlink(ready)
    hold = subprocess.Popen(
        [sys.executable, os.path.abspath(__file__), "--out", "/dev/null",
         "--child", "hold", "--arg", "24", "--ready", ready],
        stdout = subprocess.PIPE, stderr = subprocess.STDOUT, text = True,
    )
    occupied = {"waited_s": None, "child_alive": None}
    for i in range(180):
        if os.path.exists(ready):
            occupied["waited_s"] = i
            break
        if hold.poll() is not None:
            occupied["child_died"] = hold.communicate()[0][-2000:]
            break
        time.sleep(1)
    if os.path.exists(ready):
        occupied["devices"] = torch_devices()
        occupied["MemAvailable_mib"] = meminfo_mib("MemAvailable")
        occupied["sysfs"] = sysfs_vram()
    hold.kill()
    try:
        hold.wait(timeout = 30)
    except Exception:
        pass
    report["with_24gib_of_the_carve_out_held_by_another_process"] = occupied
    emit()
    time.sleep(10)  # let the driver reclaim before the exhaustion probes

    # -- 4. how does each pool fail when over-allocated? ----------------------
    devs = torch_devices()
    free_mib = devs[0]["mem_get_info_free_mib"] if devs else 0
    avail_mib = meminfo_mib("MemAvailable") or 0
    report["exhaustion"] = {
        "free_vram_mib_at_probe": free_mib,
        "MemAvailable_mib_at_probe": avail_mib,
        "hipMalloc_over_by_8gib": run_child("hipmalloc", free_mib + 8192),
    }
    emit()
    report["exhaustion"]["hipMallocManaged_over_by_6gib"] = run_child("managed", avail_mib + 6144)
    emit()
    print("probe complete", flush = True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
