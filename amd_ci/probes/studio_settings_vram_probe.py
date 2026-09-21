#!/usr/bin/env python3
"""Probe: what does Settings > System actually show for VRAM on this APU?

unsloth#7449 defect 1, "VRAM readout wrong on an APU". Three PRs have been
merged at it -- #8863 (join the WDDM adapter counters to torch devices on the
LUID), #9314 (take the APU total from the driver, not the BIOS carve-out) and
#11366 (count a unified-memory APU's window as shared host memory). #11366's own
evidence table carries the row "Windows APU (flag already set): unchanged",
which is a claim that Windows was already correct. Reading the code supports the
narrow version of that claim -- `hardware.py:2393` sets `shared_memory` only on
Windows, and #11366 touched no backend file -- and supports nothing about the
NUMBERS. Nobody has run the readout on a Windows Strix Halo.

This probe is that run. It observes and judges nothing.

What it captures, per state:

  READOUT    the whole chain `/api/system` walks, twice: once idle and once with
             a hold resident. `get_backend_visible_gpu_info` supplies the totals
             and the `shared_memory` / `shared_memory_host_backed_gb` flags,
             `get_visible_gpu_utilization` supplies used, and `main.py` derives
             free as total minus used. Also `get_gpu_summary` /
             `get_gpu_memory_info`, which feed `/api/system/hardware` (the About
             tab) through a SEPARATE code path that disagrees with the first on
             an APU by construction. A report that quotes only one of them
             cannot say which number the reporter was looking at.

  LABEL      the Settings string itself, reproduced here as a port of
             `gpu-vram.ts::gpuMemoryTotalsGb` and `resources-tab.tsx:445-451`.
             Every intermediate (total, used, free, host_backed, dedicated,
             shared) is recorded separately, because the interesting failure is
             `shared_memory_host_backed_gb` coming back None, which collapses
             the dedicated bucket to zero and prints the whole pool as "shared".

  GROUND     read by this probe itself, through nothing Studio owns. Linux: DRM
  TRUTH      sysfs `mem_info_{vram,gtt}_{total,used}` plus `/proc/meminfo`.
             Windows: the WDDM `GPU Adapter Memory` counters via Get-Counter,
             the DirectX registry's dedicated/shared bytes, and
             GlobalMemoryStatusEx. NOT amd-smi on Windows: `amd-smi.exe` is on
             PATH on these boxes but answers `Error LoadLibraryA` and exits
             non-zero, and Studio does not use it there either
             (`amd.py::_amd_smi_allowed` is False on Windows without a HIP SDK),
             so reading the same counters Studio reads is the faithful thing as
             well as the only working one. It is still SPAWNED once and its
             failure recorded, so "amd-smi is broken here" is in the artifact
             rather than assumed.

  HOLD       N GiB (default 8) allocated and TOUCHED on the GPU, alive across
             the second readout and the held ground-truth sample. The failure
             this exists to catch is invisible at idle: `Dedicated Usage`
             saturates at the BIOS carve-out and the overflow lands in `Shared`,
             so a loaded APU reads as almost entirely free (measured on a
             gfx1151: 31.637 GB dedicated, 33.887 GB shared, holding 64 GiB).
             Idle, every arm of that agrees. If torch cannot allocate, that is
             recorded and the run continues with no hold rather than aborting --
             the criteria gates on the hold, so a vacuous reading fails loudly
             instead of passing quietly.

No counter is ever coerced to 0. "the counter declined" and "the GPU holds
nothing" are the two answers this probe exists to tell apart, so a reader that
could not answer leaves its key absent or None and never a zero.

Pairs with criteria/settings_vram_matches_the_machine.py.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

GIB = 1024 ** 3

# Run in a CHILD so each state gets a fresh interpreter. hardware.py caches its
# device classification and its physical inventory at module level, and two
# states imported into one process would both report the first one's answer.
_MEASURE = r'''
import ctypes, glob, json, os, platform, re, subprocess, sys, time

GIB = 1024 ** 3
MIB = 1024 * 1024

checkout = sys.argv[1]
hold_gib = float(sys.argv[2])
backend_dir = os.path.join(checkout, "studio", "backend")
sys.path.insert(0, backend_dir)

out = {
    "backend_dir": backend_dir,
    "backend_dir_exists": os.path.isdir(backend_dir),
    "os_name": os.name,
    "sys_platform": sys.platform,
    "platform_system": platform.system(),
    "hold_gib_requested": hold_gib,
}


def call(bucket, name, fn, *a, **kw):
    """Record a value or an error string. One failed expression must not cost
    the run every other expression."""
    try:
        bucket[name] = {"ok": True, "value": fn(*a, **kw)}
    except BaseException as e:
        bucket[name] = {"ok": False, "error": (type(e).__name__ + ": " + str(e))[:400]}
    return bucket[name].get("value")


def is_windows():
    return os.name == "nt" or sys.platform.startswith("win")


# ── ground truth: read here, by this probe, through nothing Studio owns ────────
# Every reader returns bytes or None. A key is absent when its source declined
# and present-but-zero only when the source really answered zero.

def linux_ground_truth():
    gt = {"source": "linux-drm-sysfs+proc-meminfo", "errors": []}
    fields = ("vram_total", "vram_used", "gtt_total", "gtt_used",
              "vis_vram_total", "vis_vram_used")
    card = None
    for path in sorted(glob.glob("/sys/class/drm/card*/device/mem_info_vram_total")):
        card = os.path.dirname(path)
        break
    if card is None:
        gt["errors"].append("no /sys/class/drm/card*/device/mem_info_vram_total")
    else:
        gt["card"] = card
        for field in fields:
            try:
                with open(os.path.join(card, "mem_info_" + field), "r", encoding = "utf-8") as fh:
                    gt[field + "_bytes"] = int(fh.read().strip())
            except (OSError, ValueError) as e:
                gt["errors"].append("mem_info_" + field + ": " + type(e).__name__)
    try:
        with open("/proc/meminfo", "r", encoding = "utf-8") as fh:
            for line in fh:
                key, _sep, rest = line.partition(":")
                if key in ("MemTotal", "MemFree", "MemAvailable"):
                    try:
                        gt["host_" + key.lower() + "_bytes"] = int(rest.split()[0]) * 1024
                    except (IndexError, ValueError):
                        pass
    except OSError as e:
        gt["errors"].append("/proc/meminfo: " + type(e).__name__)

    # The two halves Studio's Windows split is named after, so both OSes report
    # the same shape and the criteria needs no per-OS arithmetic.
    if "vram_used_bytes" in gt:
        gt["dedicated_used_bytes"] = gt["vram_used_bytes"]
    if "gtt_used_bytes" in gt:
        gt["shared_used_bytes"] = gt["gtt_used_bytes"]
    if "vram_total_bytes" in gt:
        gt["dedicated_total_bytes"] = gt["vram_total_bytes"]
    if "gtt_total_bytes" in gt:
        gt["shared_total_bytes"] = gt["gtt_total_bytes"]
    if "host_memtotal_bytes" in gt:
        gt["host_total_bytes"] = gt["host_memtotal_bytes"]
    if "host_memavailable_bytes" in gt:
        gt["host_available_bytes"] = gt["host_memavailable_bytes"]
    return gt


# Dedicated Usage and Shared Usage are kept APART and never summed here: their
# separation IS the "in the carve-out or in host memory" distinction the whole
# defect turns on, and a probe that handed the criteria one number would have
# decided the question itself.
_PS_COUNTERS = (
    "$ErrorActionPreference='SilentlyContinue';"
    "foreach ($c in @('\\GPU Adapter Memory(*)\\Dedicated Usage',"
    "'\\GPU Adapter Memory(*)\\Shared Usage')) {"
    "  try { foreach ($s in (Get-Counter -Counter $c -ErrorAction Stop).CounterSamples) {"
    "    Write-Output ($c + '|' + $s.InstanceName + '|' + [int64]$s.CookedValue) } }"
    "  catch { Write-Output ($c + '|ERROR|' + $_.Exception.Message) } }"
)


def windows_counters(errors):
    """Per-adapter Dedicated and Shared Usage, or {} when the counter declined."""
    try:
        p = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", _PS_COUNTERS],
            capture_output = True, text = True, timeout = 120,
            encoding = "utf-8", errors = "replace", stdin = subprocess.DEVNULL)
    except BaseException as e:
        errors.append("Get-Counter: " + (type(e).__name__ + ": " + str(e))[:200])
        return {}
    totals = {"dedicated": 0, "shared": 0}
    seen = {"dedicated": False, "shared": False}
    instances = []
    for line in p.stdout.splitlines():
        parts = line.strip().split("|")
        if len(parts) != 3:
            continue
        counter, instance, value = parts
        key = "dedicated" if "Dedicated" in counter else "shared" if "Shared" in counter else None
        if key is None:
            continue
        if instance == "ERROR":
            errors.append((counter + ": " + value)[:200])
            continue
        try:
            n = int(value)
        except ValueError:
            continue
        seen[key] = True
        totals[key] += n
        instances.append({"counter": key, "instance": instance, "bytes": n})
    got = {}
    # A counter that answered with nothing but zeros is a real zero; one that
    # never answered stays absent, so the two remain distinguishable.
    if seen["dedicated"]:
        got["dedicated_used_bytes"] = totals["dedicated"]
    if seen["shared"]:
        got["shared_used_bytes"] = totals["shared"]
    if instances:
        got["counter_instances"] = instances
    if not any(seen.values()):
        errors.append("no WDDM GPU Adapter Memory counter answered")
    return got


_AMD_VENDOR_ID = 0x1002
_DIRECTX_KEY = r"SOFTWARE\Microsoft\DirectX"


def windows_directx_registry(errors):
    """DirectX's own per-adapter memory record. This is where Studio's
    `shared_memory_host_backed_gb` comes from, so reading it directly says
    whether a None there is the registry's fault or Studio's."""
    if not is_windows():
        return []
    try:
        import winreg
    except ImportError as e:
        errors.append("winreg: " + type(e).__name__ + ": " + str(e))
        return []
    adapters = []
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, _DIRECTX_KEY) as dx:
            for i in range(winreg.QueryInfoKey(dx)[0]):
                sub = winreg.EnumKey(dx, i)
                if not (sub.startswith("{") and sub.endswith("}")):
                    continue
                rec = {"subkey": sub}
                try:
                    with winreg.OpenKey(dx, sub) as key:
                        for value_name, out_name in (
                                ("VendorId", "vendor_id"),
                                ("AdapterLuid", "adapter_luid"),
                                ("Description", "description"),
                                ("AdapterFamily", "adapter_family"),
                                ("DedicatedVideoMemory", "dedicated_video_memory_bytes"),
                                ("DedicatedSystemMemory", "dedicated_system_memory_bytes"),
                                ("SharedSystemMemory", "shared_system_memory_bytes")):
                            try:
                                v, _t = winreg.QueryValueEx(key, value_name)
                            except OSError:
                                continue
                            rec[out_name] = int(v) if isinstance(v, int) else str(v)
                except OSError as e:
                    rec["error"] = type(e).__name__ + ": " + str(e)
                if rec.get("vendor_id") == _AMD_VENDOR_ID or "error" in rec:
                    adapters.append(rec)
    except BaseException as e:
        errors.append("DirectX registry: " + (type(e).__name__ + ": " + str(e))[:200])
    return adapters


def windows_host_memory(errors):
    try:
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        st = MEMORYSTATUSEX()
        st.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
            errors.append("GlobalMemoryStatusEx returned FALSE")
            return {}
        return {"host_total_bytes": int(st.ullTotalPhys),
                "host_available_bytes": int(st.ullAvailPhys)}
    except BaseException as e:
        errors.append("GlobalMemoryStatusEx: " + (type(e).__name__ + ": " + str(e))[:200])
        return {}


def windows_ground_truth():
    gt = {"source": "windows-wddm-getcounter+directx-registry+globalmemorystatusex",
          "errors": []}
    gt.update(windows_counters(gt["errors"]))
    gt.update(windows_host_memory(gt["errors"]))
    gt["directx_adapters"] = windows_directx_registry(gt["errors"])
    # The carve-out, as DirectX reports it, for comparison against Studio's
    # dedicated/shared split. Absent rather than zero when no record carried it.
    dedicated = [a.get("dedicated_video_memory_bytes", 0) + a.get("dedicated_system_memory_bytes", 0)
                 for a in gt["directx_adapters"]
                 if "dedicated_video_memory_bytes" in a or "dedicated_system_memory_bytes" in a]
    if dedicated:
        gt["directx_dedicated_bytes"] = max(dedicated)
    shared = [a["shared_system_memory_bytes"] for a in gt["directx_adapters"]
              if "shared_system_memory_bytes" in a]
    if shared:
        gt["directx_shared_system_bytes"] = max(shared)
    return gt


def ground_truth():
    return windows_ground_truth() if is_windows() else linux_ground_truth()


def amd_smi_reality():
    """Spawned once, as an OBSERVATION and never as ground truth. On the Windows
    runners amd-smi.exe is on PATH and answers `Error LoadLibraryA` with a
    non-zero exit, so recording the failure keeps that in the artifact instead
    of in a comment."""
    import shutil
    name = "amd-smi.exe" if is_windows() else "amd-smi"
    exe = shutil.which(name)
    rec = {"on_path": bool(exe), "path": exe, "used_as_ground_truth": False}
    if not exe:
        return rec
    try:
        p = subprocess.run([exe, "static", "--json"], capture_output = True, text = True,
                           timeout = 120, encoding = "utf-8", errors = "replace",
                           stdin = subprocess.DEVNULL)
        rec["rc"] = p.returncode
        rec["stdout_head"] = p.stdout[:600]
        rec["stderr_head"] = p.stderr[:600]
    except BaseException as e:
        rec["error"] = (type(e).__name__ + ": " + str(e))[:300]
    return rec


# ── the hold ──────────────────────────────────────────────────────────────────

def take_hold(gib):
    """Allocate and TOUCH gib GiB on the GPU, returning (chunks, info).

    Touched, not merely allocated: an uncommitted allocation is not what a
    resident model is and may never reach the driver's counters at all. A
    failure is recorded and returned as no hold; the criteria gates on the hold
    having actually held, so an un-held run fails a gate rather than passing on
    a vacuous reading."""
    info = {"requested_gib": gib, "held": False, "held_bytes": None}
    if gib <= 0:
        info["skipped"] = "hold-gib <= 0"
        return None, info
    try:
        import torch
        if not torch.cuda.is_available():
            info["error"] = "torch.cuda.is_available() is False"
            return None, info
        chunks = []
        want = int(gib * GIB)
        held = 0
        step = 256 * MIB
        while held < want:
            t = torch.empty(step, dtype = torch.uint8, device = "cuda")
            t.fill_(1)
            chunks.append(t)
            held += step
        torch.cuda.synchronize()
        info["held"] = True
        info["held_bytes"] = held
        info["allocated_bytes"] = int(torch.cuda.memory_allocated())
        info["reserved_bytes"] = int(torch.cuda.memory_reserved())
        return chunks, info
    except BaseException as e:
        info["error"] = (type(e).__name__ + ": " + str(e))[:400]
        return None, info


def release_hold(chunks):
    rec = {"released": False}
    if chunks is None:
        rec["skipped"] = "nothing was held"
        return rec
    try:
        del chunks[:]
        import torch
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        rec["released"] = True
    except BaseException as e:
        rec["error"] = (type(e).__name__ + ": " + str(e))[:300]
    return rec


# ── section A: the environment gates ──────────────────────────────────────────

env = {}
out["env"] = env
env["platform_system"] = platform.system()
env["sys_platform"] = sys.platform

try:
    import torch
    env["torch_version"] = torch.__version__
    env["torch_version_hip"] = getattr(torch.version, "hip", None)
    env["torch_version_cuda"] = getattr(torch.version, "cuda", None)
    env["cuda_available"] = bool(torch.cuda.is_available())
    env["device_count"] = int(torch.cuda.device_count()) if env["cuda_available"] else 0
except BaseException as e:
    torch = None
    env["torch_error"] = (type(e).__name__ + ": " + str(e))[:400]

try:
    import utils.hardware.hardware as hw
    out["hardware_module"] = getattr(hw, "__file__", None)
except BaseException as e:
    hw = None
    out["hardware_import_error"] = (type(e).__name__ + ": " + str(e))[:600]

try:
    import utils.hardware.amd as amd_mod
except BaseException as e:
    amd_mod = None
    out["amd_import_error"] = (type(e).__name__ + ": " + str(e))[:400]

if hw is not None:
    call(env, "hw_IS_ROCM", lambda: bool(hw.IS_ROCM))
    call(env, "hip_runtime_version", lambda: hw._hip_runtime_version())
    call(env, "rocm_windows_free_is_untrusted", lambda: hw.rocm_windows_free_is_untrusted())
if amd_mod is not None:
    call(env, "amd_smi_allowed", lambda: bool(amd_mod._amd_smi_allowed()))
call(env, "amd_smi_reality", amd_smi_reality)

# ── section B: the classifier every later branch is gated on ──────────────────

cls = {}
out["classify"] = cls
props = None
if torch is not None and env.get("cuda_available"):
    props = call(cls, "get_device_properties", lambda: torch.cuda.get_device_properties(0))
if props is not None:
    cls["props_name"] = getattr(props, "name", None)
    cls["props_gcnArchName"] = getattr(props, "gcnArchName", None)
    cls["props_is_integrated"] = getattr(props, "is_integrated", None)
    cls["props_total_memory"] = int(getattr(props, "total_memory", 0) or 0)
    cls.pop("get_device_properties", None)

    def _classify():
        from core.training.worker import _rocm_classify_unified_memory
        return list(_rocm_classify_unified_memory(props))

    call(cls, "classify_unified_memory", _classify)
    if hw is not None:
        call(cls, "positively_unified", lambda: hw._rocm_props_are_positively_unified(props))
        call(cls, "total_is_carve_out", lambda: hw._rocm_props_total_is_carve_out(props))
        call(cls, "unified_status", lambda: hw._rocm_props_unified_status(props))
        call(cls, "cuda_props_are_integrated",
             lambda: hw._cuda_props_are_integrated(props, "cuda"))


# ── sections C through G, plus the Settings label ─────────────────────────────

def settings_label(info, util):
    """Port of gpu-vram.ts::gpuMemoryTotalsGb and resources-tab.tsx:445-451,
    fed by main.py::_get_cached_system_gpu_info's merge.

    Single-device only, and `device_count` is recorded so a multi-device host is
    visibly out of this port's scope rather than silently misreported."""
    r = {"device_count": None, "total_gb": None, "used_gb": None, "free_gb": None,
         "host_backed_gb": None, "host_backed_known": None, "shares_host_memory": None,
         "dedicated_gb": None, "shared_gb": None, "label": None, "used_known": None}
    devices = (info or {}).get("devices") or []
    r["device_count"] = len(devices)
    if not devices:
        r["error"] = "get_backend_visible_gpu_info() reported no devices"
        return r
    d = devices[0]
    r["shared_memory"] = d.get("shared_memory")
    r["unified_memory"] = d.get("unified_memory")
    r["memory_total_gb"] = d.get("memory_total_gb")
    u = {}
    for row in (util or {}).get("devices") or []:
        if row.get("index") == d.get("index"):
            u = row
            break
    r["util_row_found"] = bool(u)
    # main.py:2129-2141, verbatim in structure.
    total = u.get("vram_total_gb") or d.get("memory_total_gb") or 0
    used = u.get("vram_used_gb", d.get("vram_used_gb"))
    reported_free = u.get("vram_free_gb", d.get("vram_free_gb"))
    if reported_free is not None:
        free = reported_free
    elif total and used is not None:
        free = round(total - used, 2)
    else:
        free = None
    r["total_gb"] = total
    r["used_gb"] = used
    r["used_known"] = used is not None
    r["free_gb"] = free
    r["vram_used_gb_aggregate"] = (util or {}).get("vram_used_gb_aggregate")

    hb = d.get("shared_memory_host_backed_gb")
    hb_known = isinstance(hb, (int, float)) and not isinstance(hb, bool) and hb >= 0
    r["host_backed_gb"] = hb
    r["host_backed_known"] = hb_known
    hb_eff = min(total, hb) if hb_known else total
    # gpu-vram.ts:150, sharesHostMemory: shared_memory OR unified_memory.
    shares = bool(d.get("shared_memory")) or bool(d.get("unified_memory"))
    r["shares_host_memory"] = shares
    if shares:
        shared_gb = round(hb_eff, 2)
        dedicated_gb = round(total - hb_eff, 2) if hb_known else 0.0
    else:
        shared_gb = 0.0
        dedicated_gb = round(total, 2)
    r["shared_gb"] = shared_gb
    r["dedicated_gb"] = dedicated_gb
    if shared_gb > 0:
        r["label"] = "{0:.2f} GiB VRAM + {1:.2f} GiB shared".format(dedicated_gb, shared_gb)
    else:
        r["label"] = "{0:.2f} GiB".format(total)
    r["used_label"] = "Unknown" if used is None else "{0:.2f} GiB".format(used)
    r["line"] = "VRAM  " + r["used_label"] + " / " + r["label"]
    return r


def studio_readout(tag):
    """Everything /api/system's GPU tile is built from, plus the About tab's
    separate path, plus the Windows-only sources underneath both."""
    rd = {"tag": tag, "t": round(time.time(), 3)}
    if hw is None:
        rd["error"] = "utils.hardware.hardware did not import"
        return rd

    # C: the two totals. mem_get_info attaches a HIP primary context the process
    # never gives back, which is why the idle pass is taken first and both
    # passes pay it identically.
    if torch is not None and env.get("cuda_available"):
        call(rd, "mem_get_info", lambda: [int(x) for x in torch.cuda.mem_get_info(0)])
        call(rd, "trusted_mem_get_info", lambda: [int(x) for x in hw.trusted_mem_get_info(0)])
        call(rd, "props_total_memory",
             lambda: int(torch.cuda.get_device_properties(0).total_memory))

    # D: the Windows-only sources, inert off Windows by their own guards.
    call(rd, "dx_records",
         lambda: hw._windows_amd_adapter_records_by_luid(distinguish_failure = True))
    call(rd, "counters_dedicated",
         lambda: hw._rocm_windows_perf_counter_vram_by_adapter("Dedicated Usage"))
    call(rd, "counters_shared",
         lambda: hw._rocm_windows_perf_counter_vram_by_adapter("Shared Usage"))
    call(rd, "unified_used_bytes", lambda: hw._rocm_windows_unified_used_bytes())
    call(rd, "per_device_vram", lambda: hw._rocm_windows_per_device_vram([0]))
    call(rd, "noise_floor_bytes", lambda: int(hw._ROCM_WIN_ADAPTER_MIN_BYTES))

    # E: the inventory carrying the flag under test, and the host-backed figure.
    inv = call(rd, "inventory", lambda: hw._torch_get_device_inventory([0]))
    call(rd, "windows_shared_pool_gb",
         lambda: hw._windows_rocm_shared_pool_host_gb_by_index(inv or []))
    call(rd, "linux_shared_pool_gb",
         lambda: hw._rocm_linux_shared_pool_host_gb_by_index(inv or []))

    # F: the two payloads /api/system merges (main.py:2078).
    info = call(rd, "visible_gpu_info", lambda: hw.get_backend_visible_gpu_info())
    util = call(rd, "visible_gpu_util", lambda: hw.get_visible_gpu_utilization())
    call(rd, "gpu_utilization", lambda: hw.get_gpu_utilization())

    # G: the OTHER tab. /api/system/hardware walks a separate path that
    # disagrees with the above on an APU by construction, and a report quoting
    # one of them cannot say which number the user was looking at.
    call(rd, "gpu_summary", lambda: hw.get_gpu_summary())
    call(rd, "gpu_memory_info", lambda: hw.get_gpu_memory_info())

    rd["settings"] = settings_label(info, util)
    return rd


# ── the run: idle, hold, held, release ────────────────────────────────────────

out["ground_truth_idle"] = ground_truth()
out["readout_idle"] = studio_readout("idle")

chunks, hold_info = take_hold(hold_gib)
out["hold"] = hold_info

out["ground_truth_held"] = ground_truth()
out["readout_held"] = studio_readout("held")

out["release"] = release_hold(chunks)
chunks = None
out["ground_truth_released"] = ground_truth()

print(json.dumps(out, default = str))
'''


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--hold-gib", type = float, default = 8.0,
                    help = "GiB to allocate and touch on the GPU across the second readout; "
                           "the plateau failure mode is invisible at idle")
    ap.add_argument("--timeout", type = int, default = 1800)
    args = ap.parse_args()

    obs: dict = {"state": args.state, "checkout": args.checkout,
                 "platform": sys.platform, "hold_gib": args.hold_gib}

    env = dict(os.environ)
    # Read by the paths under test. An inherited value would answer for the
    # harness rather than for the machine: UNSLOTH_ENABLE_AMD_SMI flips
    # `_amd_smi_allowed` on Windows, and a visibility mask changes which device
    # the readout is even about.
    for name in ("UNSLOTH_ENABLE_AMD_SMI", "UNSLOTH_COMPILE_DISABLE",
                 "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES",
                 "CUDA_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL"):
        env.pop(name, None)

    try:
        proc = subprocess.run(
            [sys.executable, "-c", _MEASURE, args.checkout, str(args.hold_gib)],
            capture_output = True, text = True, timeout = args.timeout, env = env,
            encoding = "utf-8", errors = "replace", stdin = subprocess.DEVNULL)
        obs["rc"] = proc.returncode
        obs["stderr_tail"] = proc.stderr[-4000:]
        stdout = proc.stdout
    except subprocess.TimeoutExpired as e:
        # A timeout is a reading too: the criteria gates on the probe having
        # returned one, so this fails a gate rather than vanishing.
        obs["rc"] = None
        obs["timed_out"] = args.timeout
        obs["stderr_tail"] = str(e)[-4000:]
        stdout = ""

    # Import banners share stdout, so take the LAST JSON object on it.
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                obs.update(json.loads(line))
                break
            except ValueError:
                continue
    else:
        obs["child_failed"] = True
        obs["stdout_tail"] = stdout[-4000:]

    args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
