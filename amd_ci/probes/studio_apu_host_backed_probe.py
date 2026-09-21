#!/usr/bin/env python3
"""Probe: what host-backed split does the backend publish for this APU, and what
does the Settings > System tile then READ?

unsloth#11451. The change is two lines of consequence in
`studio/backend/utils/hardware/hardware.py`:

  `_rocm_linux_shared_pool_host_gb_by_index` used to OMIT an index whose sysfs
  total was readable and whose torch total did not exceed it by more than 10%.
  An omitted index is "unknown". It now publishes 0.0, a MEASURED zero.

  `get_backend_visible_gpu_info` used to set `shared_memory = True` for every
  index present in that map. It now sets it only when the host-backed figure is
  above zero, and publishes the figure either way.

Neither line is observable from the backend alone as a defect: both states
return a well-formed device dict. The defect is what the CONSUMER does with
them, and the consumer is `studio/frontend/src/hooks/gpu-vram.ts`
`gpuMemoryTotalsGb`, where an absent/null `shared_memory_host_backed_gb` means
`hostBackedKnown = false`, so `hostBacked` becomes the whole total, the reserved
bucket stays empty and the dedicated bucket collapses to zero. A 64 GiB carve-out
then prints as all shared and no VRAM.

So this probe captures four layers, and judges none of them:

  DEVICE DICTS  `get_backend_visible_gpu_info()` called on the real gfx1151, in
                this checkout, through the app's own import path. The six fields
                the frontend reads are lifted out separately, and a null
                `shared_memory_host_backed_gb` is recorded as null WITH a
                `_present` boolean beside it, never coerced to 0. "unknown" and
                "a measured zero" are the two answers this whole run exists to
                tell apart, so the probe must not blur them at the first hop.

  RAW INPUTS    the two numbers the changed predicate compares, read here rather
                than taken from the function's word: torch's
                `get_device_properties(i).total_memory` in bytes, and DRM sysfs
                `mem_info_vram_total` for the card's PCI id. Plus
                `mem_info_vram_used`, `mem_info_gtt_total`, `mem_info_gtt_used`
                and the KFD `gfx_target_version` (gfx1151 encodes as 110501), so
                a reader can recompute the branch that was taken instead of
                trusting the verdict. The module's own helper maps
                (`_rocm_linux_sysfs_vram_by_index`,
                `_rocm_linux_shared_pool_host_gb_by_index`) are captured too,
                which is the only way to see an index that was OMITTED: the
                published dict cannot distinguish "absent from the map" from
                "present and null".

  RENDERED TILE the real `gpuMemoryTotalsGb`, executed. `gpu-vram.ts` has no
                imports, so node >= 22 can import the TypeScript directly under
                type stripping; the probe feeds it the device dicts it just read
                off the backend and records what comes back. A Python port of
                the same function runs unconditionally beside it, because node
                is not guaranteed on either half of the pool, and the two are
                recorded separately so the criteria can gate on their agreeing
                rather than on whichever one answered. The label strings
                themselves are ported: `formatGiB` and the `vramWithShared`
                expression live in `resources-tab.tsx`, which is JSX with
                imports and cannot be loaded this way, so its source text is
                captured verbatim into the artifact for audit instead.

  FRONTEND SHA  sha256 and line counts of the frontend files the port mirrors.
                The port is only faithful to the source it was written against,
                and a criteria gate compares the hashes across states: this PR
                touches no frontend file, so a difference would mean the port is
                describing something that moved.

Everything is recorded per reader, and a reader that declined leaves an error
string rather than a zero. No counter is ever coerced.

Pairs with criteria/apu_host_backed_zero_is_measured.py (Linux, differential)
and criteria/apu_host_backed_windows_unchanged.py (Windows, control).
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

GIB = 1024 ** 3

# The six fields gpu-vram.ts and resources-tab.tsx actually read off a device row.
DEVICE_FIELDS = ("index", "name", "memory_total_gb", "shared_memory",
                 "shared_memory_host_backed_gb", "unified_memory")


def call(bucket: dict, name: str, fn, *a, **kw):
    """Record a value or an error string. One failed reader must not cost the run
    every other reader."""
    try:
        bucket[name] = {"ok": True, "value": fn(*a, **kw)}
    except BaseException as e:  # noqa: BLE001
        bucket[name] = {"ok": False, "error": f"{type(e).__name__}: {e}"[:500]}
    return bucket[name].get("value")


# ── the frontend rule, ported ────────────────────────────────────────────────
# A transliteration of gpu-vram.ts::gpuMemoryTotalsGb and ::sharesHostMemory, and
# of resources-tab.tsx::formatGiB plus the vramCapacityLabel expression. Written
# to be boring and line-for-line rather than idiomatic, because the only property
# that matters is that it does what the TypeScript does. It is cross-checked
# against the real TypeScript whenever node can run it.

def _round_to_device_precision(value: float) -> float:
    """JS `Math.round(value * 100) / 100`. Math.round is half-UP, Python's round
    is half-to-even, so 2.345 would differ. Do it the JS way."""
    if not math.isfinite(value):
        return value
    scaled = value * 100.0
    return math.floor(scaled + 0.5) / 100.0 if scaled >= 0 else math.ceil(scaled - 0.5) / 100.0


def _finite_number(v) -> bool:
    """JS `Number.isFinite`. A bool is not a number here; Python says it is."""
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _size(device: dict) -> float:
    total = device.get("memory_total_gb")
    total = 0 if total is None else total
    return float(total) if _finite_number(total) and total > 0 else 0.0


def _shares_host_memory(device: dict) -> bool:
    return device.get("shared_memory") is True or device.get("unified_memory") is True


def gpu_memory_totals_gb(devices: list[dict]) -> dict:
    dedicated_devices = _round_to_device_precision(
        sum(_size(d) for d in devices if not _shares_host_memory(d)))
    host_backed = 0.0
    per_device = 0.0
    reserved = 0.0
    for d in devices:
        if not _shares_host_memory(d):
            continue
        total = _size(d)
        reported = d.get("shared_memory_host_backed_gb")
        known = _finite_number(reported) and reported >= 0
        hb = min(total, float(reported)) if known else total
        if d.get("shared_memory") is True:
            host_backed = max(host_backed, hb)
        else:
            per_device += hb
        reserved += (total - hb) if known else 0.0
    shared = _round_to_device_precision(host_backed + per_device)
    dedicated = _round_to_device_precision(dedicated_devices + reserved)
    return {"dedicated": dedicated, "shared": shared,
            "total": _round_to_device_precision(dedicated + shared)}


def _to_fixed(value: float, digits: int) -> str:
    """JS `Number.prototype.toFixed`, which rounds half away from zero. Python's
    format rounds half to even, so 0.125 would print 0.12 there and 0.13 here."""
    if not math.isfinite(value):
        return str(value)
    factor = 10 ** digits
    scaled = value * factor
    rounded = math.floor(scaled + 0.5) if scaled >= 0 else math.ceil(scaled - 0.5)
    out = rounded / factor
    return f"{out:.{digits}f}"


def format_gib_resources_tab(value) -> str:
    """resources-tab.tsx:83. Note the trailing-zero trim: it is why a dedicated
    figure of exactly 64 prints `64 GiB` on that surface and not `64.00 GiB`."""
    safe = max(0.0, float(value)) if _finite_number(value) else 0.0
    digits = 1 if safe >= 10 else 2
    text = re.sub(r"\.?0+$", "", _to_fixed(safe, digits))
    return f"{text} GiB"


def format_gib_floating_monitor(value) -> str:
    """floating-monitor.tsx:412. The same tile logic on the always-on monitor,
    with no trim, so the same totals read differently there."""
    v = float(value) if _finite_number(value) else 0.0
    return f"{_to_fixed(v, 1 if v >= 10 else 2)} GiB"


def format_gib_two_dp(value) -> str:
    """Not a surface: a fixed-2dp rendering of the same totals, so the figures
    can be quoted unambiguously without a reader having to undo a trim."""
    v = float(value) if _finite_number(value) else 0.0
    return f"{_to_fixed(v, 2)} GiB"


def render_labels(totals: dict) -> dict:
    """resources-tab.tsx:445-451 with the `en` locale string
    `settings.resources.environment.vramWithShared` = "{vram} VRAM + {shared} shared"."""
    out = {}
    for tag, fmt in (("resources_tab", format_gib_resources_tab),
                     ("floating_monitor", format_gib_floating_monitor),
                     ("two_dp", format_gib_two_dp)):
        out[tag] = (f"{fmt(totals['dedicated'])} VRAM + {fmt(totals['shared'])} shared"
                    if totals["shared"] > 0 else fmt(totals["total"]))
    return out


# ── ground truth, read here, through nothing Studio owns ─────────────────────

def _read_int(path: str):
    with open(path, "r", encoding = "utf-8") as fh:
        return int(fh.read().strip())


def _read_text(path: str) -> str:
    with open(path, "r", encoding = "utf-8", errors = "replace") as fh:
        return fh.read()


def sysfs_cards() -> list[dict]:
    """Every DRM card's memory counters, keyed by the PCI id hardware.py joins on.

    Absent keys where a counter declined. `mem_info_vram_total` is the number the
    changed predicate compares torch's budget against."""
    cards = []
    for card in sorted(glob.glob("/sys/class/drm/card*")):
        if re.search(r"card\d+-", os.path.basename(card)):
            continue  # a connector, not a card
        dev = os.path.join(card, "device")
        if not os.path.isdir(dev):
            continue
        entry: dict = {"card": os.path.basename(card), "errors": []}
        try:
            entry["pci_id"] = os.path.basename(os.path.realpath(dev))
        except OSError as e:
            entry["errors"].append(f"pci_id: {e}")
        for key in ("vendor", "device", "subsystem_vendor", "subsystem_device"):
            try:
                entry[key] = _read_text(os.path.join(dev, key)).strip()
            except OSError:
                pass
        for key in ("mem_info_vram_total", "mem_info_vram_used",
                    "mem_info_gtt_total", "mem_info_gtt_used",
                    "mem_info_vis_vram_total", "mem_info_vis_vram_used"):
            try:
                value = _read_int(os.path.join(dev, key))
            except (OSError, ValueError) as e:
                entry["errors"].append(f"{key}: {type(e).__name__}")
                continue
            entry[key] = value
            entry[key + "_gib"] = value / GIB
        cards.append(entry)
    return cards


def kfd_nodes() -> list[dict]:
    """KFD topology. `gfx_target_version` is 110501 on gfx1151; the CPU node has
    no simd_count, so both are recorded and neither is filtered out here."""
    nodes = []
    for props in sorted(glob.glob("/sys/class/kfd/kfd/topology/nodes/*/properties")):
        node: dict = {"node": os.path.basename(os.path.dirname(props))}
        try:
            text = _read_text(props)
        except OSError as e:
            node["error"] = str(e)
            nodes.append(node)
            continue
        wanted = {"gfx_target_version", "simd_count", "cpu_cores_count",
                  "location_id", "domain", "vendor_id", "device_id"}
        for line in text.splitlines():
            bits = line.split()
            if len(bits) == 2 and bits[0] in wanted:
                node[bits[0]] = bits[1]
        name_path = os.path.join(os.path.dirname(props), "name")
        try:
            node["name"] = _read_text(name_path).strip()
        except OSError:
            pass
        nodes.append(node)
    return nodes


def proc_meminfo() -> dict:
    """`/proc/meminfo`, verbatim plus the two lines that matter.

    Decisive for reading the result rather than for producing it. Strix Halo has
    two opposite shapes. With a large BIOS carve-out the amdgpu VRAM bar is
    DISJOINT from `MemTotal` -- the kernel never saw those pages -- so
    `MemTotal + mem_info_vram_total` sums to the fitted RAM and calling the whole
    torch budget dedicated is sound. With the default small carve, GTT is the
    real pool and OVERLAPS `MemTotal`, so the same addition double counts. The
    probe does not decide which; it records the raw kB lines so the arithmetic
    can be done and disagreed with."""
    out: dict = {"raw_lines": {}}
    text = _read_text("/proc/meminfo")
    out["full"] = text[:8000]
    for line in text.splitlines():
        key = line.split(":", 1)[0]
        if key in ("MemTotal", "MemAvailable", "MemFree", "Cma Total", "CmaTotal",
                   "CmaFree", "HugePages_Total", "Hugetlb", "Shmem"):
            out["raw_lines"][key] = line
            m = re.search(r"(\d+)\s*kB", line)
            if m:
                out[key + "_bytes"] = int(m.group(1)) * 1024
                out[key + "_gib"] = int(m.group(1)) * 1024 / GIB
    return out


def psutil_memory() -> dict:
    import psutil  # noqa: PLC0415
    vm = psutil.virtual_memory()
    return {"total_bytes": int(vm.total), "available_bytes": int(vm.available),
            "total_gib": vm.total / GIB, "available_gib": vm.available / GIB}


def dmi_physical_memory() -> dict:
    """Installed DIMM capacity, which `MemTotal` understates by whatever firmware
    reserved. There is no root on these runners, so this is expected to decline;
    the decline is recorded rather than assumed."""
    out: dict = {}
    sysfs_total = 0
    found = False
    for path in sorted(glob.glob("/sys/devices/system/memory/memory*/state")):
        found = True
        break
    out["sysfs_memory_blocks_present"] = found
    block_size = "/sys/devices/system/memory/block_size_bytes"
    if os.path.exists(block_size):
        try:
            size = int(_read_text(block_size).strip(), 16)
            blocks = len(glob.glob("/sys/devices/system/memory/memory*"))
            sysfs_total = size * blocks
            out["block_size_bytes"] = size
            out["blocks"] = blocks
            out["sysfs_total_bytes"] = sysfs_total
            out["sysfs_total_gib"] = sysfs_total / GIB
        except (OSError, ValueError) as e:
            out["sysfs_error"] = f"{type(e).__name__}: {e}"
    dmidecode = shutil.which("dmidecode")
    out["dmidecode"] = dmidecode
    if dmidecode:
        p = subprocess.run([dmidecode, "-t", "memory"],
                           capture_output = True, text = True, timeout = 180)
        out["dmidecode_rc"] = p.returncode
        out["dmidecode_out"] = (p.stdout or "")[:4000]
        out["dmidecode_err"] = (p.stderr or "")[:500]
    return out


def _run_tool(name: str, args: list[str], timeout: int = 300) -> dict:
    """Spawn a vendor tool and record what it said, including a failure. "amd-smi
    is broken on this host" belongs in the artifact, not in an assumption."""
    exe = shutil.which(name) or shutil.which(name + ".exe")
    out: dict = {"path": exe, "argv": [name, *args]}
    if not exe:
        out["error"] = f"{name} is not on PATH"
        return out
    p = subprocess.run([exe, *args], capture_output = True, text = True, timeout = timeout)
    out["rc"] = p.returncode
    out["stdout"] = (p.stdout or "")[:12000]
    out["stderr"] = (p.stderr or "")[:2000]
    return out


def windows_physical_memory() -> dict:
    """The Windows counterpart of MemTotal plus the WDDM budget split, so the
    same disjoint-vs-overlapping question can be asked there."""
    ps = shutil.which("powershell") or shutil.which("powershell.exe")
    if not ps:
        return {"error": "no powershell on PATH"}
    script = (
        "$cs = Get-CimInstance Win32_ComputerSystem; "
        "$os = Get-CimInstance Win32_OperatingSystem; "
        "$phys = Get-CimInstance Win32_PhysicalMemory | "
        "Measure-Object -Property Capacity -Sum; "
        "[pscustomobject]@{ "
        "total_physical_memory = $cs.TotalPhysicalMemory; "
        "dimm_capacity_sum = $phys.Sum; "
        "total_visible_memory_kb = $os.TotalVisibleMemorySize; "
        "free_physical_memory_kb = $os.FreePhysicalMemory } | "
        "ConvertTo-Json -Compress")
    p = subprocess.run([ps, "-NoProfile", "-NonInteractive", "-Command", script],
                       capture_output = True, text = True, timeout = 180)
    if p.returncode != 0:
        return {"error": f"powershell rc={p.returncode}: {(p.stderr or '')[:300]}"}
    return json.loads((p.stdout or "").strip() or "{}")


def windows_wddm_budget() -> dict:
    """DedicatedVideoMemory / SharedSystemMemory off the DirectX registry, which
    is the source `_windows_rocm_shared_pool_host_gb_by_index` reads, plus the
    live WDDM adapter counters."""
    ps = shutil.which("powershell") or shutil.which("powershell.exe")
    if not ps:
        return {"error": "no powershell on PATH"}
    script = (
        "$rows = @(); "
        "$k = 'HKLM:\\SOFTWARE\\Microsoft\\DirectX'; "
        "if (Test-Path $k) { Get-ChildItem $k | ForEach-Object { "
        "$p = Get-ItemProperty $_.PSPath; "
        "$rows += [pscustomobject]@{ subkey = $_.PSChildName; "
        "description = $p.Description; "
        "dedicated_video = $p.DedicatedVideoMemory; "
        "dedicated_system = $p.DedicatedSystemMemory; "
        "shared_system = $p.SharedSystemMemory } } } ; "
        "$counters = @(); "
        "try { (Get-Counter '\\GPU Adapter Memory(*)\\Dedicated Usage' "
        "-ErrorAction Stop).CounterSamples | ForEach-Object { "
        "$counters += [pscustomobject]@{ path = $_.Path; value = $_.CookedValue } } } "
        "catch { $counters += [pscustomobject]@{ error = $_.Exception.Message } } ; "
        "[pscustomobject]@{ registry = $rows; dedicated_usage = $counters } | "
        "ConvertTo-Json -Depth 5 -Compress")
    p = subprocess.run([ps, "-NoProfile", "-NonInteractive", "-Command", script],
                       capture_output = True, text = True, timeout = 300)
    if p.returncode != 0:
        return {"error": f"powershell rc={p.returncode}: {(p.stderr or '')[:300]}"}
    return json.loads((p.stdout or "").strip() or "{}")


def windows_adapter_memory() -> list[dict]:
    """The Windows counterpart of the sysfs read: what the OS says each adapter
    has. Not used by the Linux branch under test, and recorded on Windows only so
    the control leg's artifact is not empty of ground truth."""
    ps = shutil.which("powershell") or shutil.which("powershell.exe")
    if not ps:
        return [{"error": "no powershell on PATH"}]
    script = (
        "$out = @(); "
        "Get-CimInstance Win32_VideoController | ForEach-Object { "
        "$out += [pscustomobject]@{ name = $_.Name; adapter_ram = $_.AdapterRAM; "
        "driver = $_.DriverVersion; status = $_.Status } }; "
        "$out | ConvertTo-Json -Compress")
    p = subprocess.run([ps, "-NoProfile", "-NonInteractive", "-Command", script],
                       capture_output = True, text = True, timeout = 180)
    if p.returncode != 0:
        return [{"error": f"powershell rc={p.returncode}: {(p.stderr or '')[:300]}"}]
    raw = (p.stdout or "").strip()
    if not raw:
        return [{"error": "empty output"}]
    parsed = json.loads(raw)
    return parsed if isinstance(parsed, list) else [parsed]


# ── the backend, as the app imports it ───────────────────────────────────────

def import_hardware(checkout: Path):
    backend = checkout / "studio" / "backend"
    if not backend.is_dir():
        raise SystemExit(f"no backend at {backend}")
    sys.path.insert(0, str(backend))
    for stale in [m for m in sys.modules if m.startswith("utils.")]:
        del sys.modules[stale]
    import utils.hardware.hardware as hw  # noqa: PLC0415
    return hw


def _jsonable(value):
    """Helper maps are keyed by int and valued by tuple; JSON needs neither."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def frontend_sources(checkout: Path) -> dict:
    """The source the port mirrors, hashed and quoted. A port is faithful to a
    revision, not in general, so the artifact has to carry the revision."""
    out: dict = {}
    files = {
        "gpu_vram_ts": "studio/frontend/src/hooks/gpu-vram.ts",
        "resources_tab_tsx": "studio/frontend/src/features/settings/tabs/resources-tab.tsx",
        "floating_monitor_tsx": "studio/frontend/src/components/floating-monitor.tsx",
        "locale_en_ts": "studio/frontend/src/i18n/locales/en.ts",
    }
    for tag, rel in files.items():
        path = checkout / rel
        entry: dict = {"path": rel, "exists": path.is_file()}
        if path.is_file():
            data = path.read_bytes()
            entry["sha256"] = hashlib.sha256(data).hexdigest()
            entry["bytes"] = len(data)
            text = data.decode("utf-8", "replace")
            entry["lines"] = text.count("\n") + 1
            if tag == "gpu_vram_ts":
                m = re.search(r"export function gpuMemoryTotalsGb\(.*?\n\}\n", text, re.S)
                entry["gpuMemoryTotalsGb_source"] = m.group(0) if m else None
            if tag == "resources_tab_tsx":
                m = re.search(r"function formatGiB\(.*?\n\}\n", text, re.S)
                entry["formatGiB_source"] = m.group(0) if m else None
                m = re.search(r"const vramCapacityLabel =.*?;\n", text, re.S)
                entry["vramCapacityLabel_source"] = m.group(0) if m else None
            if tag == "locale_en_ts":
                m = re.search(r"vramWithShared: \"(.*?)\"", text)
                entry["vramWithShared"] = m.group(1) if m else None
        out[tag] = entry
    return out


def drive_real_typescript(checkout: Path, devices: list[dict], work: Path) -> dict:
    """Execute the checkout's own gpuMemoryTotalsGb, unmodified.

    `gpu-vram.ts` imports nothing, so node >= 22.6 can import the .ts file itself
    under type stripping; no bundler, no transpile step, no chance of the port
    being what is under test. Writes JSON to a FILE, because node prints warnings
    ("Type Stripping is an experimental feature") to stderr and a banner on
    stdout is exactly how a probe's JSON gets corrupted."""
    out: dict = {"attempted": True}
    node = shutil.which("node") or shutil.which("node.exe")
    out["node"] = node
    if not node:
        out["attempted"] = False
        out["error"] = "node is not on PATH; the Python port is the only rendering"
        return out
    ver = subprocess.run([node, "--version"], capture_output = True, text = True, timeout = 120)
    out["node_version"] = (ver.stdout or ver.stderr or "").strip()
    ts = checkout / "studio" / "frontend" / "src" / "hooks" / "gpu-vram.ts"
    if not ts.is_file():
        out["error"] = f"no {ts}"
        return out
    work.mkdir(parents = True, exist_ok = True)
    devices_json = work / "devices.json"
    devices_json.write_text(json.dumps(devices), encoding = "utf-8")
    result_json = work / "ts_totals.json"
    driver = work / "drive_gpu_vram.mjs"
    driver.write_text(
        "import { readFileSync, writeFileSync } from 'node:fs';\n"
        f"const mod = await import({json.dumps(ts.as_posix())});\n"
        f"const devices = JSON.parse(readFileSync({json.dumps(devices_json.as_posix())}, 'utf8'));\n"
        "const totals = mod.gpuMemoryTotalsGb(devices);\n"
        f"writeFileSync({json.dumps(result_json.as_posix())}, JSON.stringify({{\n"
        "  totals,\n"
        "  aggregate: mod.aggregateGpuMemoryTotalGb(devices),\n"
        "  sharedHost: mod.gpuSharedHostMemoryGb(devices),\n"
        "  sharesHostMemory: devices.map((d) => mod.sharesHostMemory({\n"
        "    sharedMemory: d.shared_memory, unifiedMemory: d.unified_memory })),\n"
        "}));\n", encoding = "utf-8")
    p = subprocess.run([node, "--experimental-strip-types", str(driver)],
                       capture_output = True, text = True, timeout = 300)
    out["rc"] = p.returncode
    out["stdout_tail"] = (p.stdout or "")[-1500:]
    out["stderr_tail"] = (p.stderr or "")[-1500:]
    if result_json.is_file():
        out.update(json.loads(result_json.read_text(encoding = "utf-8")))
        out["ok"] = True
    else:
        out["ok"] = False
        out.setdefault("error", "node produced no output file")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True, type = Path)
    ap.add_argument("--out", required = True, type = Path)
    args = ap.parse_args()

    checkout = args.checkout
    obs: dict = {
        "state": args.state,
        "checkout": str(checkout),
        "platform_system": platform.system(),
        "sys_platform": sys.platform,
        "hostname": platform.node(),
        "python": sys.version.split()[0],
    }

    # ── raw inputs, read independently of hardware.py ────────────────────────
    raw: dict = {}
    obs["raw"] = raw
    call(raw, "sysfs_cards", sysfs_cards)
    call(raw, "kfd_nodes", kfd_nodes)
    # Host memory, for the disjoint-versus-overlapping question: does the carve-out
    # stand BESIDE MemTotal or INSIDE it? Recorded, never resolved here.
    if os.path.exists("/proc/meminfo"):
        call(raw, "proc_meminfo", proc_meminfo)
        call(raw, "physical_memory", dmi_physical_memory)
    pm = call(raw, "psutil_memory", psutil_memory)
    # The `memory` block of the /api/system payload, reproduced from its own source:
    # main.py:2389 rounds psutil.virtual_memory() to 2dp and publishes nothing else.
    # Quoted beside the VRAM tile because the two are read together by a human
    # deciding whether the machine's RAM and its VRAM are the same pages.
    if isinstance(pm, dict):
        raw["api_system_memory_block"] = {
            "total_gb": round(pm["total_bytes"] / GIB, 2),
            "available_gb": round(pm["available_bytes"] / GIB, 2),
        }
    # Vendor tools, recorded including their failures.
    call(raw, "amd_smi_static", _run_tool, "amd-smi", ["static"])
    call(raw, "amd_smi_metric", _run_tool, "amd-smi", ["metric", "-m"])
    call(raw, "rocm_smi_meminfo", _run_tool, "rocm-smi", ["--showmeminfo", "vram", "gtt"])
    if os.name == "nt" or sys.platform.startswith("win"):
        call(raw, "windows_adapters", windows_adapter_memory)
        call(raw, "windows_physical_memory", windows_physical_memory)
        call(raw, "windows_wddm_budget", windows_wddm_budget)

    torch_info: dict = {}
    obs["torch"] = torch_info
    try:
        import torch  # noqa: PLC0415
        torch_info["version"] = torch.__version__
        torch_info["hip"] = getattr(torch.version, "hip", None)
        torch_info["cuda"] = getattr(torch.version, "cuda", None)
        torch_info["is_available"] = bool(torch.cuda.is_available())
        torch_info["device_count"] = int(torch.cuda.device_count())
        devs = []
        for i in range(torch_info["device_count"]):
            d: dict = {"index": i}
            call(d, "properties", lambda i = i: {
                "name": torch.cuda.get_device_properties(i).name,
                "gcnArchName": getattr(torch.cuda.get_device_properties(i), "gcnArchName", None),
                "total_memory_bytes": int(torch.cuda.get_device_properties(i).total_memory),
                "total_memory_gib": torch.cuda.get_device_properties(i).total_memory / GIB,
                "is_integrated": getattr(
                    torch.cuda.get_device_properties(i), "is_integrated", None),
                "multi_processor_count": getattr(
                    torch.cuda.get_device_properties(i), "multi_processor_count", None),
            })
            devs.append(d)
        torch_info["devices"] = devs
    except BaseException as e:  # noqa: BLE001
        torch_info["error"] = f"{type(e).__name__}: {e}"[:500]

    # ── the backend, in THIS checkout ────────────────────────────────────────
    backend: dict = {}
    obs["backend"] = backend
    hw = None
    try:
        hw = import_hardware(checkout)
        backend["hardware_file"] = hw.__file__
        backend["IS_ROCM"] = bool(getattr(hw, "IS_ROCM", False))
        backend["get_device"] = str(call(backend, "_get_device", hw.get_device))
    except BaseException as e:  # noqa: BLE001
        backend["import_error"] = f"{type(e).__name__}: {e}"[:500]

    devices: list[dict] = []
    if hw is not None:
        info = call(backend, "get_backend_visible_gpu_info", hw.get_backend_visible_gpu_info)
        if isinstance(info, dict):
            backend["get_backend_visible_gpu_info"]["value"] = _jsonable(info)
            devices = list(info.get("devices") or [])

        # The module's OWN view of the inputs, so an omitted index is visible.
        # A published device dict cannot distinguish "index absent from the map"
        # from "index present with a null figure": that distinction is the PR.
        internals: dict = {}
        backend["internals"] = internals

        def _inventory():
            ids = hw.get_parent_visible_gpu_ids()
            if ids:
                indices, kind = ids, "physical"
            else:
                indices = list(range(hw._torch_get_physical_gpu_count() or 0))
                kind = "relative"
            return {"index_kind": kind, "indices": _jsonable(indices),
                    "inventory": _jsonable(hw._torch_get_device_inventory(indices))}

        inv = call(internals, "torch_inventory", _inventory)
        raw_inv = []
        if isinstance(inv, dict):
            try:
                ids = hw.get_parent_visible_gpu_ids()
                indices = ids or list(range(hw._torch_get_physical_gpu_count() or 0))
                raw_inv = hw._torch_get_device_inventory(indices)
            except BaseException as e:  # noqa: BLE001
                internals["inventory_reuse_error"] = f"{type(e).__name__}: {e}"[:300]

        call(internals, "sysfs_vram_by_pci_gb",
             lambda: _jsonable(hw._rocm_linux_sysfs_vram_by_pci_gb()))
        call(internals, "kfd_gpu_pci_ids", lambda: _jsonable(hw._rocm_kfd_gpu_pci_ids()))
        if raw_inv:
            call(internals, "sysfs_vram_by_index", lambda: _jsonable(
                hw._rocm_linux_sysfs_vram_by_index(raw_inv, allow_numeric_mask = True)))
            m = call(internals, "shared_pool_host_gb_by_index", lambda: _jsonable(
                hw._rocm_linux_shared_pool_host_gb_by_index(raw_inv)))
            # Spelled out, because "which indices does the map contain" IS the change.
            if isinstance(m, dict):
                internals["shared_pool_indices_present"] = sorted(m.keys())
            internals["inventory_known_unified"] = {
                str(d.get("index")): bool(d.get("_rocm_known_unified")) for d in raw_inv}
            internals["inventory_total_gb"] = {
                str(d.get("index")): d.get("total_gb") for d in raw_inv}

    # ── the six fields the frontend reads, lifted out and never coerced ──────
    lifted = []
    for d in devices:
        row = {f: d.get(f) for f in DEVICE_FIELDS}
        # `absent` and `null` behave identically in gpu-vram.ts (Number.isFinite
        # rejects both), but they mean different things upstream, so both are kept.
        row["shared_memory_host_backed_gb_key_present"] = \
            "shared_memory_host_backed_gb" in d
        row["shared_memory_host_backed_gb_is_null"] = \
            d.get("shared_memory_host_backed_gb") is None
        lifted.append(row)
    obs["devices"] = lifted

    # ── what the tile then shows ─────────────────────────────────────────────
    render: dict = {}
    obs["render"] = render
    totals = call(render, "totals_python_port", gpu_memory_totals_gb, lifted)
    if isinstance(totals, dict):
        render["labels_python_port"] = render_labels(totals)
    render["typescript"] = drive_real_typescript(
        checkout, lifted, args.out.parent / f"node_{args.state}")
    ts_totals = (render["typescript"] or {}).get("totals")
    if isinstance(ts_totals, dict):
        render["labels_typescript"] = render_labels(ts_totals)
        render["port_matches_typescript"] = (
            isinstance(totals, dict)
            and all(abs(float(ts_totals[k]) - float(totals[k])) < 1e-9
                    for k in ("dedicated", "shared", "total")))

    obs["frontend_sources"] = frontend_sources(checkout)

    args.out.parent.mkdir(parents = True, exist_ok = True)
    args.out.write_text(json.dumps(obs, indent = 2, default = repr), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
