#!/usr/bin/env python3
"""Probe: on this gfx1151, where do a model's weights ACTUALLY land?

unsloth#7449 defect 2 and unsloth#8945's remainder are both "the weights are in
system RAM, not the GPU pool". Everything merged so far (#9884) changes a
DECISION: whether Studio exports ``GGML_CUDA_ENABLE_UNIFIED_MEMORY=1``. Nobody
has measured the consequence on this part. A decision differential on its own
cannot say whether the decision is the right one, because it never looks at a
byte of memory.

So this observes two separate things and judges neither:

  DECISION   what the state's own code chooses for the model actually on disk,
             read through the same helpers the launch path calls.

  RESIDENCY  what llama.cpp does with that model, measured twice in EVERY state
             with the flag forced OFF and forced ON. Three independent readings
             of "where did it go", because each one can lie alone:

               * llama.cpp's own `load_tensors: <dev> model buffer size` lines.
                 The most direct answer, and the one the workflow doc says to
                 trust over `system_info`, which names ROCm even for a run that
                 touched no GPU.
               * the driver's VRAM-used counter, sampled across the run. On an
                 APU a managed allocation shows up as GTT rather than VRAM, so
                 both are sampled on Linux where both are exposed.
               * host free memory across the run.

             and tok/s, because #8945's log is a throughput report (0.28 t/s
             decode against 118 t/s prefill) and a residency claim that does not
             reproduce that number has not explained that issue.

Forcing the flag BOTH ways in EVERY state is the point. The binary is identical
in both states, so the residency readings are a property of the flag and of this
hardware, not of the checkout; measuring them per state is what makes them a
control on each other. If flag_on and flag_off place the weights identically,
then the flag is not the mechanism and #7449 defect 2 is something else, and
that conclusion is only available because both legs ran.

Pairs with criteria/weights_land_where_decided.py.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import urllib.request
import zipfile
from pathlib import Path

MIB = 1024 * 1024
GIB = 1024 ** 3

PROMPT = (
    "Write a detailed technical explanation of how a modern operating system "
    "schedules threads across multiple CPU cores, covering run queues, load "
    "balancing and priority inheritance."
)

# `eval time = ... ( 9.64 ms per token, 103.72 tokens per second)`. The lookbehind
# is load-bearing: `prompt eval time` CONTAINS ` eval time`, is printed first, and
# without it every decode reading is silently the PREFILL rate.
_EVAL_RE = re.compile(
    r"(?<!prompt )eval time\s*=.*?\(\s*[\d.]+\s*ms per token,\s*([\d.]+)\s*tokens per second\)")
_PROMPT_EVAL_RE = re.compile(
    r"prompt eval time\s*=.*?\(\s*[\d.]+\s*ms per token,\s*([\d.]+)\s*tokens per second\)")
# `load_tensors:        ROCm0 model buffer size = 12345.67 MiB`
_BUFFER_RE = re.compile(
    r"load_tensors:\s+(\S+)\s+model buffer size\s*=\s*([\d.]+)\s*MiB")
# `load_tensors: layer 12 assigned to device ROCm0, is_swa = 0`, needs -v.
_LAYER_DEV_RE = re.compile(r"load_tensors:\s+layer\s+\d+\s+assigned to device\s+([A-Za-z0-9_]+)")
_OFFLOADED_RE = re.compile(r"offloaded\s+(\d+)\s*/\s*(\d+)\s+layers to GPU")


def _is_windows() -> bool:
    return os.name == "nt" or sys.platform.startswith("win")


def _exe(name: str) -> str:
    return f"{name}.exe" if _is_windows() else name


# ── memory sampling ────────────────────────────────────────────────────────────
# Every reader returns MiB or None. None is never coerced to 0: a zero would read
# as "nothing was allocated", which is the exact finding this probe exists to
# report, and a missing counter must not be able to manufacture it.

def _linux_drm_mem() -> dict:
    out: dict = {}
    for field in ("vram_total", "vram_used", "gtt_total", "gtt_used",
                  "vis_vram_total", "vis_vram_used"):
        for path in sorted(glob.glob(f"/sys/class/drm/card*/device/mem_info_{field}")):
            try:
                with open(path, "r", encoding = "utf-8") as fh:
                    value = int(fh.read().strip())
            except (OSError, ValueError):
                continue
            # First readable card wins; these boxes have exactly one.
            out[field + "_mib"] = value // MIB
            break
    return out


def _linux_host_mem() -> dict:
    out: dict = {}
    try:
        with open("/proc/meminfo", "r", encoding = "utf-8") as fh:
            for line in fh:
                key, _, rest = line.partition(":")
                if key in ("MemTotal", "MemFree", "MemAvailable"):
                    try:
                        out[key.lower() + "_mib"] = int(rest.split()[0]) // 1024
                    except (IndexError, ValueError):
                        pass
    except OSError:
        pass
    return out


def _windows_host_mem() -> dict:
    try:
        import ctypes

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
            return {}
        return {"memtotal_mib": int(st.ullTotalPhys) // MIB,
                "memavailable_mib": int(st.ullAvailPhys) // MIB,
                "memfree_mib": int(st.ullAvailPhys) // MIB}
    except Exception:  # noqa: BLE001
        return {}


_AMD_SMI_NUM = re.compile(r"([A-Z_]+)\s*[:=]\s*([\d]+)")


def _amd_smi_mem(errors: list) -> dict:
    """VRAM used/total in MiB from amd-smi, JSON first, then the plain table.

    Records the parse failure rather than returning zeros: on Windows this is the
    only VRAM counter available, so "amd-smi said nothing" and "the GPU holds
    nothing" have to stay distinguishable.
    """
    exe = shutil.which(_exe("amd-smi"))
    if not exe:
        errors.append("amd-smi not on PATH")
        return {}
    for argv in ([exe, "metric", "-g", "0", "--mem-usage", "--json"],
                 [exe, "metric", "--mem-usage", "--json"]):
        try:
            p = subprocess.run(argv, capture_output = True, text = True, timeout = 60,
                               encoding = "utf-8", errors = "replace",
                               stdin = subprocess.DEVNULL)
        except Exception as e:  # noqa: BLE001
            errors.append(f"{' '.join(argv)}: {type(e).__name__}: {e}"[:200])
            continue
        blob = p.stdout
        start = blob.find("[")
        if start < 0:
            start = blob.find("{")
        if start < 0:
            continue
        try:
            doc = json.loads(blob[start:])
        except ValueError:
            continue
        rows = doc if isinstance(doc, list) else [doc]
        for row in rows:
            usage = (row or {}).get("mem_usage") or row or {}
            got: dict = {}
            for key, name in (("total_vram", "vram_total_mib"),
                              ("used_vram", "vram_used_mib"),
                              ("free_vram", "vram_free_mib"),
                              ("total_visible_vram", "vis_vram_total_mib"),
                              ("used_visible_vram", "vis_vram_used_mib"),
                              ("total_gtt", "gtt_total_mib"),
                              ("used_gtt", "gtt_used_mib")):
                v = usage.get(key)
                if isinstance(v, dict):
                    v = v.get("value")
                if isinstance(v, (int, float)):
                    got[name] = int(v)
            if got:
                return got
    # Plain table, last resort.
    try:
        p = subprocess.run([exe, "metric", "-g", "0", "--mem-usage"],
                           capture_output = True, text = True, timeout = 60,
                           encoding = "utf-8", errors = "replace",
                           stdin = subprocess.DEVNULL)
        got = {}
        for key, name in (("TOTAL_VRAM", "vram_total_mib"), ("USED_VRAM", "vram_used_mib"),
                          ("FREE_VRAM", "vram_free_mib"), ("TOTAL_GTT", "gtt_total_mib"),
                          ("USED_GTT", "gtt_used_mib")):
            m = re.search(key + r"\s*[:=]?\s*(\d+)", p.stdout)
            if m:
                got[name] = int(m.group(1))
        if got:
            return got
        errors.append("amd-smi --mem-usage produced no parsable counters")
    except Exception as e:  # noqa: BLE001
        errors.append(f"amd-smi table: {type(e).__name__}: {e}"[:200])
    return {}


# The WDDM counters, which on Windows are the ONLY working VRAM reading here:
# amd-smi.exe is on PATH but answers `Error LoadLibraryA` and exits non-zero
# (measured on ephemeral-devlab-x2-la01-s04-d01, run 35587502769). Studio does
# not use it there either -- `amd.py::_amd_smi_allowed` returns False on Windows
# without a HIP SDK -- so reading the same counters Studio reads is also the
# faithful thing to do, not just the available one.
#
# Dedicated Usage vs Shared Usage IS the "in VRAM or in system RAM" distinction
# this probe exists to measure, so the two are kept apart and never summed here.
_PS_COUNTERS = (
    "$ErrorActionPreference='SilentlyContinue';"
    "foreach ($c in @('\\GPU Adapter Memory(*)\\Dedicated Usage',"
    "'\\GPU Adapter Memory(*)\\Shared Usage','\\GPU Adapter Memory(*)\\Total Committed')) {"
    "  try { foreach ($s in (Get-Counter -Counter $c -ErrorAction Stop).CounterSamples) {"
    "    Write-Output ($c + '|' + $s.InstanceName + '|' + [int64]$s.CookedValue) } }"
    "  catch { Write-Output ($c + '|ERROR|' + $_.Exception.Message) } }"
)


def _windows_gpu_mem(errors: list) -> dict:
    try:
        p = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", _PS_COUNTERS],
            capture_output = True, text = True, timeout = 120,
            encoding = "utf-8", errors = "replace", stdin = subprocess.DEVNULL)
    except Exception as e:  # noqa: BLE001
        errors.append(f"Get-Counter: {type(e).__name__}: {e}"[:200])
        return {}
    totals = {"dedicated": 0, "shared": 0, "committed": 0}
    seen = {"dedicated": False, "shared": False, "committed": False}
    instances: list = []
    for line in p.stdout.splitlines():
        parts = line.strip().split("|")
        if len(parts) != 3:
            continue
        counter, instance, value = parts
        key = ("dedicated" if "Dedicated" in counter
               else "shared" if "Shared" in counter
               else "committed" if "Committed" in counter else None)
        if key is None:
            continue
        if instance == "ERROR":
            errors.append(f"{counter}: {value}"[:200])
            continue
        try:
            n = int(value)
        except ValueError:
            continue
        seen[key] = True
        totals[key] += n
        if n > 0:
            instances.append({"counter": key, "instance": instance, "bytes": n})
    out: dict = {}
    # A counter that answered with nothing but zeros is a real zero; one that never
    # answered stays absent, so the criteria can tell them apart.
    if seen["dedicated"]:
        out["vram_used_mib"] = totals["dedicated"] // MIB
    if seen["shared"]:
        out["gtt_used_mib"] = totals["shared"] // MIB
    if seen["committed"]:
        out["committed_mib"] = totals["committed"] // MIB
    if instances:
        out["counter_instances"] = instances
    if not any(seen.values()):
        errors.append("no WDDM GPU Adapter Memory counter answered")
    return out


def sample_memory(errors: list) -> dict:
    """One instant of every memory counter this OS exposes."""
    snap: dict = {"t": round(time.time(), 3)}
    if _is_windows():
        # Not amd-smi: it is broken on these boxes and its failure is recorded once
        # by the workflow's ground-truth step rather than on every sample.
        snap.update(_windows_gpu_mem(errors))
        snap.update(_windows_host_mem())
    else:
        snap.update(_amd_smi_mem(errors))
        # amd-smi first, then sysfs OVERRIDES it: on Linux mem_info_vram_used and
        # mem_info_gtt_used come straight from amdgpu and need no parsing, while
        # amd-smi's output format has changed between releases. amd-smi is kept
        # because it is the only source for counters sysfs does not expose.
        drm = _linux_drm_mem()
        snap.update({k: v for k, v in drm.items() if v is not None})
        snap.update(_linux_host_mem())
    return snap


class Sampler(threading.Thread):
    """Samples memory on an interval for the life of a child process."""

    def __init__(self, interval: float = 1.0):
        super().__init__(daemon = True)
        self.interval = interval
        self.samples: list[dict] = []
        self.errors: list[str] = []
        # NOT `self._stop`. `threading.Thread._stop` is a real private METHOD, and
        # `Thread.join` calls it once the thread's state lock clears. Shadowing it
        # with an Event makes join() raise `TypeError: 'Event' object is not
        # callable` -- and only sometimes, because join skips that call when the
        # thread has already finished, so it passed on Linux and failed on Windows.
        self._stop_event = threading.Event()

    def run(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.samples.append(sample_memory(self.errors))
            except Exception as e:  # noqa: BLE001
                self.errors.append(f"{type(e).__name__}: {e}"[:200])
            self._stop_event.wait(self.interval)

    def stop(self) -> None:
        self._stop_event.set()

    def peak(self, key: str):
        vals = [s[key] for s in self.samples if isinstance(s.get(key), (int, float))]
        return max(vals) if vals else None

    def trough(self, key: str):
        vals = [s[key] for s in self.samples if isinstance(s.get(key), (int, float))]
        return min(vals) if vals else None


# ── artefacts ──────────────────────────────────────────────────────────────────

def download(url: str, dest: Path, log: list) -> bool:
    if dest.is_file() and dest.stat().st_size > 0:
        log.append({"download": url, "cached": True, "bytes": dest.stat().st_size})
        return True
    dest.parent.mkdir(parents = True, exist_ok = True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    t0 = time.time()
    try:
        req = urllib.request.Request(url, headers = {"User-Agent": "amd-ci/1.0"})
        with urllib.request.urlopen(req, timeout = 600) as r, open(tmp, "wb") as fh:
            shutil.copyfileobj(r, fh, 1 << 20)
        tmp.replace(dest)
    except Exception as e:  # noqa: BLE001
        log.append({"download": url, "error": f"{type(e).__name__}: {e}"[:300]})
        return False
    log.append({"download": url, "cached": False, "bytes": dest.stat().st_size,
                "seconds": round(time.time() - t0, 1)})
    return True


def extract(archive: Path, dest: Path, log: list) -> bool:
    marker = dest / ".extracted"
    if marker.is_file():
        log.append({"extract": str(archive), "cached": True})
        return True
    dest.mkdir(parents = True, exist_ok = True)
    try:
        if archive.name.endswith(".zip"):
            with zipfile.ZipFile(archive) as z:
                z.extractall(dest)
        else:
            # encoding names how MEMBER NAMES are decoded, and tarfile defaults to
            # the locale, which is not utf-8 everywhere. Same rule as every
            # other text read here.
            with tarfile.open(archive, "r:gz", encoding = "utf-8") as t:
                t.extractall(dest)
    except Exception as e:  # noqa: BLE001
        log.append({"extract": str(archive), "error": f"{type(e).__name__}: {e}"[:300]})
        return False
    if not _is_windows():
        for p in dest.rglob("llama-*"):
            if p.is_file():
                try:
                    p.chmod(0o755)
                except OSError:
                    pass
    marker.write_text("ok", encoding = "utf-8")
    try:
        archive.unlink()
    except OSError:
        pass
    log.append({"extract": str(archive), "cached": False})
    return True


def find_binary(root: Path, name: str) -> Path | None:
    direct = root / _exe(name)
    if direct.is_file():
        return direct
    for p in root.rglob(_exe(name)):
        if p.is_file():
            return p
    return None


# ── the state's own decision ───────────────────────────────────────────────────

_DECIDE = r'''
import json, os, sys

out = {}
checkout, need_bytes_csv = sys.argv[1], sys.argv[2]
sys.path.insert(0, os.path.join(checkout, "studio", "backend"))

try:
    import torch
    out["torch_version"] = torch.__version__
    out["torch_hip"] = getattr(torch.version, "hip", None)
    out["torch_cuda"] = getattr(torch.version, "cuda", None)
    out["cuda_available"] = bool(torch.cuda.is_available())
    devices = []
    if out["cuda_available"]:
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            devices.append({
                "ordinal": i,
                "name": getattr(p, "name", None),
                "gcnArchName": getattr(p, "gcnArchName", None),
                "is_integrated": getattr(p, "is_integrated", None),
                "total_memory": int(getattr(p, "total_memory", 0) or 0),
                "total_mib": int(getattr(p, "total_memory", 0) or 0) // (1024 * 1024),
            })
    out["devices"] = devices
except BaseException as e:
    out["torch_error"] = f"{type(e).__name__}: {e}"[:400]

try:
    from core.inference.llama_cpp import LlamaCppBackend as B
except BaseException as e:
    out["backend_error"] = f"{type(e).__name__}: {e}"[:600]
    print(json.dumps(out, default = str)); raise SystemExit(0)


def call(name, *args, **kw):
    fn = getattr(B, name, None)
    if fn is None:
        return {"present": False}
    try:
        value = fn(*args, **kw)
        if isinstance(value, set):
            value = sorted(value)
        return {"present": True, "value": value}
    except BaseException as e:
        return {"present": True, "error": f"{type(e).__name__}: {e}"[:300]}


out["torch_is_rocm"] = call("_torch_is_rocm")
out["rocm_classification_answered"] = call("_rocm_classification_answered")
out["rocm_unified_memory_gpu_ids"] = call("_rocm_unified_memory_gpu_ids")
out["rocm_selected_pool_mib"] = call("_rocm_selected_pool_mib", [0])
out["available_system_memory_mib"] = call("_available_system_memory_mib")
out["amd_apu_wants_unified_memory"] = call("_amd_apu_wants_unified_memory", [0])

decisions = []
for raw in need_bytes_csv.split(","):
    if not raw.strip():
        continue
    need_bytes = int(raw)
    answer = {"need_bytes": need_bytes, "need_gib": round(need_bytes / (1024 ** 3), 3)}
    for name, args, kw in (
        ("_unified_memory_for_launch", ([0], need_bytes), {}),
        ("_unified_memory_would_help", ([0],), {"need_bytes": need_bytes}),
        ("_unified_memory_would_help", ([0],), {}),
        ("_amd_apu_wants_unified_memory", ([0],), {}),
    ):
        fn = getattr(B, name, None)
        if fn is None:
            continue
        try:
            answer["decided_by"] = name
            answer["unified_memory"] = bool(fn(*args, **kw))
            break
        except TypeError as e:
            answer["last_type_error"] = f"{name}: {e}"[:200]
            continue
        except BaseException as e:
            answer["error"] = f"{name}: {type(e).__name__}: {e}"[:300]
            break
    decisions.append(answer)
out["decisions"] = decisions

# #7449 defect 1 rides along: the same run can say what Settings > System shows.
try:
    from utils.hardware.hardware import get_gpu_summary, get_gpu_memory_info
    out["gpu_summary"] = get_gpu_summary()
    out["gpu_memory_info"] = get_gpu_memory_info()
except BaseException as e:
    out["gpu_readout_error"] = f"{type(e).__name__}: {e}"[:400]

print(json.dumps(out, default = str))
'''


def run_decision(python: str, checkout: str, need_bytes: list[int], timeout: int) -> dict:
    env = dict(os.environ)
    for name in ("GGML_CUDA_ENABLE_UNIFIED_MEMORY", "UNSLOTH_DISABLE_UNIFIED_MEMORY",
                 "UNSLOTH_ENABLE_UNIFIED_MEMORY"):
        env.pop(name, None)
    try:
        p = subprocess.run(
            [python, "-c", _DECIDE, checkout, ",".join(str(n) for n in need_bytes)],
            capture_output = True, text = True, timeout = timeout, env = env,
            encoding = "utf-8", errors = "replace", stdin = subprocess.DEVNULL)
    except Exception as e:  # noqa: BLE001
        return {"child_failed": True, "error": f"{type(e).__name__}: {e}"[:300]}
    rec: dict = {"rc": p.returncode, "stderr_tail": p.stderr[-3000:]}
    for line in reversed(p.stdout.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                rec.update(json.loads(line))
                return rec
            except ValueError:
                continue
    rec["child_failed"] = True
    rec["stdout_tail"] = p.stdout[-3000:]
    return rec


# ── one residency leg ──────────────────────────────────────────────────────────

def run_leg(binary: Path, model: Path, args, flag_on: bool, log: list) -> dict:
    """Load and generate once, sampling memory throughout.

    Baseline is taken BEFORE the child starts and again after it exits, and both
    are reported. A single baseline cannot tell a leak or a co-tenant from an
    allocation, and on an APU the counters do not always return to where they
    started.
    """
    env = dict(os.environ)
    lib_dir = str(binary.parent)
    if _is_windows():
        env["PATH"] = lib_dir + os.pathsep + env.get("PATH", "")
    else:
        env["LD_LIBRARY_PATH"] = lib_dir + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    # ggml tests PRESENCE, not truth, so "0" would enable what it looks like it
    # disables (#8651). Only absence is off.
    env.pop("GGML_CUDA_ENABLE_UNIFIED_MEMORY", None)
    if flag_on:
        env["GGML_CUDA_ENABLE_UNIFIED_MEMORY"] = "1"

    errors: list[str] = []
    before = sample_memory(errors)
    sampler = Sampler(interval = args.sample_interval)
    sampler.start()
    argv = [str(binary), "-m", str(model), "-ngl", "999", "-c", str(args.ctx),
            "-n", str(args.n_gen), "-p", PROMPT, "-st", "--no-warmup", "--perf",
            "-s", "1234", "--temp", "0.7", "--top-k", "40",
            "-t", str(args.threads), "-v"]
    t0 = time.time()
    try:
        p = subprocess.run(argv, capture_output = True, text = True, timeout = args.timeout,
                           env = env, cwd = lib_dir, encoding = "utf-8", errors = "replace",
                           stdin = subprocess.DEVNULL)
        rc, out, err, timed_out = p.returncode, p.stdout, p.stderr, False
    except subprocess.TimeoutExpired as e:
        rc, timed_out = None, True
        out = e.stdout if isinstance(e.stdout, str) else (e.stdout or b"").decode("utf-8", "replace")
        err = e.stderr if isinstance(e.stderr, str) else (e.stderr or b"").decode("utf-8", "replace")
    seconds = round(time.time() - t0, 2)
    sampler.stop()
    sampler.join(timeout = 10)
    after = sample_memory(errors)

    blob = (err or "") + (out or "")
    buffers: dict[str, float] = {}
    for dev, mib in _BUFFER_RE.findall(blob):
        buffers[dev] = buffers.get(dev, 0.0) + float(mib)
    layers: dict[str, int] = {}
    for dev in _LAYER_DEV_RE.findall(blob):
        layers[dev] = layers.get(dev, 0) + 1
    tg = _EVAL_RE.search(blob)
    pp = _PROMPT_EVAL_RE.search(blob)
    off = _OFFLOADED_RE.search(blob)

    def delta(key: str):
        base, peak = before.get(key), sampler.peak(key)
        if not isinstance(base, (int, float)) or not isinstance(peak, (int, float)):
            return None
        return peak - base

    def host_drop(key: str):
        base, low = before.get(key), sampler.trough(key)
        if not isinstance(base, (int, float)) or not isinstance(low, (int, float)):
            return None
        return base - low

    rec = {
        "flag": "GGML_CUDA_ENABLE_UNIFIED_MEMORY=1" if flag_on else "(unset)",
        "flag_on": flag_on,
        "rc": rc, "timed_out": timed_out, "seconds": seconds,
        "tg_ts": float(tg.group(1)) if tg else None,
        "pp_ts": float(pp.group(1)) if pp else None,
        "offloaded_layers": (int(off.group(1)), int(off.group(2))) if off else None,
        "model_buffer_mib_by_device": buffers,
        "layer_devices": layers,
        "mem_before": before,
        "mem_after": after,
        "mem_peak": {k: sampler.peak(k) for k in
                     ("vram_used_mib", "gtt_used_mib", "vis_vram_used_mib")},
        "mem_trough": {k: sampler.trough(k) for k in
                       ("memavailable_mib", "memfree_mib")},
        "vram_used_delta_mib": delta("vram_used_mib"),
        "gtt_used_delta_mib": delta("gtt_used_mib"),
        "host_available_drop_mib": host_drop("memavailable_mib"),
        "samples": len(sampler.samples),
        "sample_errors": sorted(set(sampler.errors + errors))[:10],
    }
    log.append({"leg": rec["flag"], "rc": rc, "seconds": seconds, "argv": argv,
                "stderr_tail": (err or "")[-6000:]})
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--llama-repo", default = "unslothai/llama.cpp")
    ap.add_argument("--tag", required = True,
                    help = "ONE llama.cpp release tag, used in every state: the binary "
                           "is not what differs between them")
    ap.add_argument("--model-repo", default = "unsloth/Qwen3-4B-Instruct-2507-GGUF")
    ap.add_argument("--model-file", default = "Qwen3-4B-Instruct-2507-Q4_K_M.gguf")
    ap.add_argument("--cache-dir", default = None)
    ap.add_argument("--n-gen", type = int, default = 128)
    ap.add_argument("--ctx", type = int, default = 4096)
    ap.add_argument("--threads", type = int, default = 8)
    # Windows has no cheap counter read: every sample spawns a PowerShell, which
    # costs a few hundred ms, so it samples less often. Linux reads sysfs files.
    ap.add_argument("--sample-interval", type = float,
                    default = 2.0 if _is_windows() else 0.5)
    ap.add_argument("--settle", type = float, default = 5.0,
                    help = "seconds between legs, for the driver counters to settle")
    ap.add_argument("--timeout", type = int, default = 3600)
    ap.add_argument("--decision-timeout", type = int, default = 900)
    ap.add_argument("--python", default = sys.executable,
                    help = "interpreter for the DECISION child; must be the Studio venv")
    args = ap.parse_args()

    log: list = []
    obs: dict = {
        "state": args.state, "checkout": args.checkout,
        "platform": platform.platform(),
        "os": "windows" if _is_windows() else sys.platform,
        "tag": args.tag,
        "model": f"{args.model_repo}/{args.model_file}",
        "settings": {"n_gen": args.n_gen, "ctx": args.ctx, "threads": args.threads,
                     "sample_interval": args.sample_interval},
    }

    cache = Path(args.cache_dir) if args.cache_dir else Path(
        os.environ.get("AMD_CI_WORK") or os.environ.get("RUNNER_TEMP") or ".") / "residency-cache"
    cache = cache.expanduser().resolve()
    cache.mkdir(parents = True, exist_ok = True)
    obs["cache_dir"] = str(cache)

    model = cache / args.model_file
    model_url = f"https://huggingface.co/{args.model_repo}/resolve/main/{args.model_file}"
    if not download(model_url, model, log):
        obs["error"] = "model download failed"
        obs["_log"] = log
        args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
        return 0
    obs["model_bytes"] = model.stat().st_size
    obs["model_gib"] = round(model.stat().st_size / GIB, 3)

    # The decision, asked about the file actually on disk plus a ladder either
    # side of it, so the carve-out boundary is visible in the table.
    ladder = [obs["model_bytes"]] + [int(g * GIB) for g in (1, 4, 16, 40, 90)]
    obs["decision"] = run_decision(args.python, args.checkout, ladder, args.decision_timeout)

    asset = (f"app-{args.tag}-windows-x64-rocm-gfx1151.zip" if _is_windows()
             else f"app-{args.tag}-linux-x64-rocm-gfx1151.tar.gz")
    url = f"https://github.com/{args.llama_repo}/releases/download/{args.tag}/{asset}"
    archive = cache / asset
    root = cache / f"x-{args.tag}"
    if not (root / ".extracted").is_file():
        if not download(url, archive, log):
            obs["error"] = f"could not download {asset}"
            obs["_log"] = log
            args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
            return 0
    if not extract(archive, root, log):
        obs["error"] = f"could not extract {asset}"
        obs["_log"] = log
        args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
        return 0

    comp = find_binary(root, "llama-completion")
    obs["llama_completion"] = str(comp) if comp else None
    if comp is None:
        obs["error"] = "bundle is missing llama-completion"
        obs["_log"] = log
        args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
        return 0

    obs["idle_memory"] = sample_memory([])
    legs: dict = {}
    for name, flag_on in (("flag_off", False), ("flag_on", True)):
        legs[name] = run_leg(comp, model, args, flag_on, log)
        # Let the driver's counters settle before the next baseline; an APU's
        # VRAM-used does not drop the instant the process exits, and a baseline
        # taken too early makes the next leg's delta look small.
        time.sleep(args.settle)
    obs["legs"] = legs
    obs["_log"] = log

    args.out.write_text(json.dumps(obs, indent = 2, default = str), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
