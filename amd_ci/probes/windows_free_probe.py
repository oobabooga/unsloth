#!/usr/bin/env python3
"""Probe: what does a free-VRAM query report on Windows while another process holds VRAM?

AMD documents the hipMemGetInfo free half as accounting only for the calling
process on Windows, "and may be optimistic", because WDDM hands a process its own
budget rather than the card's residency. Studio encodes that in
`rocm_windows_free_is_untrusted()` and caps the figure in `trusted_mem_get_info()`.
This records, side by side and with an 8 GiB bystander resident:

  raw            what `mem_get_info` returns to this process
  guard-facing   what a guard in THIS checkout would be handed, which is
                 `trusted_mem_get_info()` where that function exists and the raw
                 figure where it does not
  DirectX        the adapter record's dedicated and shared memory values, which
                 are what the carve-out advice reads on Windows

Observes only. Whether "raw is near total" is the documented over-report or a
correct reading of an empty card is not the probe's call: the fixture size and
the comparison live in the criteria module.

Pairs with probes/vram_holder_fixture.py and
criteria/windows_free_is_untrusted.py.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

GIB = 1024 ** 3
DIRECTX_KEY = r"SOFTWARE\Microsoft\DirectX"
AMD_PCI_VENDOR_ID = 4098
# Both dedicated values, because that is the sum hardware.py charges as the
# firmware carve-out, plus the shared figure so the split is visible rather than
# inferred.
ADAPTER_VALUES = ("DedicatedVideoMemory", "DedicatedSystemMemory", "SharedSystemMemory")


def import_hardware(checkout: Path):
    """Import THIS checkout's hardware module, the way the app does."""
    backend = checkout / "studio" / "backend"
    if not backend.is_dir():
        raise SystemExit(f"no backend at {backend}")
    sys.path.insert(0, str(backend))
    for stale in [m for m in sys.modules if m.startswith("utils.")]:
        del sys.modules[stale]
    import utils.hardware.hardware as hw  # noqa: PLC0415
    return hw


def directx_adapters() -> dict:
    """The DirectX adapter records, read straight from the registry.

    Deliberately independent of the checkout: the point is to see the same bytes
    the code reads without going through the code, so a state whose reader is
    absent or broken still contributes a figure.
    """
    if platform.system() != "Windows":
        return {"available": False, "why": f"not Windows ({platform.system()})"}
    try:
        import winreg  # noqa: PLC0415
    except ImportError as e:  # noqa: BLE001
        return {"available": False, "why": f"no winreg: {e}"}

    adapters: list[dict] = []
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, DIRECTX_KEY) as dx:
            for index in range(winreg.QueryInfoKey(dx)[0]):
                subkey = winreg.EnumKey(dx, index)
                # Adapter records are GUID-named; ShaderCache and friends are not.
                if not (subkey.startswith("{") and subkey.endswith("}")):
                    continue
                with winreg.OpenKey(dx, subkey) as key:
                    row: dict = {"subkey": subkey}
                    for name in ("VendorId", "AdapterLuid", "Description", "AdapterFamily"):
                        try:
                            row[name] = winreg.QueryValueEx(key, name)[0]
                        except OSError:
                            pass
                    for name in ADAPTER_VALUES:
                        try:
                            value = int(winreg.QueryValueEx(key, name)[0])
                        except (OSError, TypeError, ValueError):
                            continue
                        row[name] = value
                        row[f"{name}_gib"] = value / GIB
                    adapters.append(row)
    except Exception as e:  # noqa: BLE001
        return {"available": False, "why": f"{type(e).__name__}: {e}"}

    amd = [a for a in adapters if a.get("VendorId") == AMD_PCI_VENDOR_ID]
    out: dict = {"available": True, "adapters": adapters, "amd_count": len(amd)}
    dedicated = sum(int(a.get(n, 0)) for a in amd
                    for n in ("DedicatedVideoMemory", "DedicatedSystemMemory"))
    shared = sum(int(a.get("SharedSystemMemory", 0)) for a in amd)
    if amd:
        out["amd_dedicated_bytes"] = dedicated
        out["amd_dedicated_gib"] = dedicated / GIB
        out["amd_shared_bytes"] = shared
        out["amd_shared_gib"] = shared / GIB
    return out


def carveout_reading(checkout: Path) -> dict:
    """What the carve-out advice would read on this box, through its own reader.

    Separate from the registry dump above on purpose: the advice does not quote
    every adapter it finds, it fails closed when the inventory is incomplete, so
    "the registry says 8 GiB" and "the advice would say 8 GiB" are two different
    observations and both are worth recording.
    """
    out: dict = {}
    try:
        from core.inference.llama_cpp import LlamaCppBackend  # noqa: PLC0415
        value = LlamaCppBackend._igpu_dedicated_memory_bytes()
        out["igpu_dedicated_memory_bytes"] = value
        out["igpu_dedicated_memory_gib"] = None if value is None else value / GIB
    except Exception as e:  # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {e}"
    return out


def planner_readings(hw, total_bytes: int | None) -> dict:
    """The figures the LAUNCH path budgets with, not the primitive underneath it.

    `trusted_mem_get_info` says in its own docstring that it cannot see another
    process, so it is a ceiling rather than a measurement. What decides a
    llama-server placement is `_get_gpu_memory`, which on a device the ROCm branch
    calls unified applies a SECOND cap: `min(free, _available_system_memory_mib())`
    before the host reserve comes off. Recording only the primitive would grade
    the wrong number.

    Every entry is separate and separately failable, because which branch of
    `_get_gpu_memory` answered is itself an observation: amd-smi runs before the
    torch fallback and does not need a HIP context, so on a box with amd-smi.exe
    on PATH the ROCm torch branch may never be reached.
    """
    out: dict = {}
    try:
        from core.inference.llama_cpp import LlamaCppBackend  # noqa: PLC0415
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}

    def record(name, fn):
        try:
            out[name] = fn()
        except Exception as e:  # noqa: BLE001
            out[f"{name}_error"] = f"{type(e).__name__}: {e}"

    # As the planner consumes it. for_llama_server=True is the placement caller's
    # spelling, so both are recorded rather than assuming they agree.
    record("get_gpu_memory", lambda: [list(row) for row in LlamaCppBackend._get_gpu_memory()])
    record("get_gpu_memory_for_llama_server",
           lambda: [list(row) for row in
                    LlamaCppBackend._get_gpu_memory(for_llama_server = True)])
    record("get_gpu_memory_amd_smi",
           lambda: [list(row) for row in
                    LlamaCppBackend._get_gpu_memory_amd_smi(None, for_llama_server = True)])
    record("unified_ids", lambda: sorted(LlamaCppBackend._rocm_unified_memory_gpu_ids()))
    record("available_system_memory_mib", LlamaCppBackend._available_system_memory_mib)
    record("llama_server_binary", LlamaCppBackend._find_llama_server_binary)

    # The reading get_gpu_summary uses on a Windows unified part: WDDM performance
    # counters, out of process, which is the one path here that can see another
    # process's residency at all.
    if total_bytes:
        record("context_free_unified_bytes",
               lambda: hw._context_free_cuda_memory_info(0, int(total_bytes), unified = True))
        out["context_free_unified_gib"] = (
            None if out.get("context_free_unified_bytes") is None
            else out["context_free_unified_bytes"] / GIB)
    record("rocm_windows_unified_used_bytes", hw._rocm_windows_unified_used_bytes)
    return out


def self_allocation_check(hw, gib: float = 2.0) -> dict:
    """Read raw and trusted again with an allocation in THIS process.

    Observation, not judgement: it records what the cap does when the memory it
    can see is its own. Whether that is the same protection as against another
    process is the criteria module's question, not the probe's.
    """
    out: dict = {"requested_gib": gib}
    try:
        import torch  # noqa: PLC0415
        block = torch.empty(int(gib * GIB), dtype = torch.uint8, device = "cuda")
        block.fill_(1)
        torch.cuda.synchronize()
        free_b, total_b = torch.cuda.mem_get_info()
        out["raw_free_gib"] = free_b / GIB
        out["raw_total_gib"] = total_b / GIB
        out["allocated_gib"] = torch.cuda.memory_allocated() / GIB
        out["reserved_gib"] = torch.cuda.memory_reserved() / GIB
        t_free, _t_total = hw.trusted_mem_get_info()
        out["trusted_free_gib"] = t_free / GIB
        del block
    except Exception as e:  # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {e}"
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True, type = Path)
    ap.add_argument("--out", required = True, type = Path)
    args = ap.parse_args()

    obs: dict = {"state": args.state, "platform": platform.system()}

    try:
        import torch  # noqa: PLC0415
        obs["torch_version"] = torch.__version__
        obs["torch_hip"] = getattr(torch.version, "hip", None)
        obs["torch_cuda"] = getattr(torch.version, "cuda", None)
        obs["torch_cuda_available"] = bool(torch.cuda.is_available())
        free_b, total_b = torch.cuda.mem_get_info()
        obs["raw_free_gib"] = free_b / GIB
        obs["raw_total_gib"] = total_b / GIB
        obs["raw_free_fraction"] = (free_b / total_b) if total_b else None
        props = torch.cuda.get_device_properties(0)
        obs["props_total_gib"] = props.total_memory / GIB
        obs["arch"] = getattr(props, "gcnArchName", None)
        obs["is_integrated"] = getattr(props, "is_integrated", None)
        obs["device_name"] = props.name
        # The observer must be a bystander, or it is measuring its own allocation.
        obs["observer_allocated_gib"] = torch.cuda.memory_allocated() / GIB
        obs["observer_reserved_gib"] = torch.cuda.memory_reserved() / GIB
    except Exception as e:  # noqa: BLE001
        obs["torch_error"] = f"{type(e).__name__}: {e}"

    hw = None
    try:
        hw = import_hardware(args.checkout)
        obs["hardware_file"] = hw.__file__
        # `IS_ROCM` is a module global that only `detect_hardware()` sets, and
        # `rocm_windows_free_is_untrusted()` reads it. A probe that imports the
        # module and asks straight away gets the False it was initialised with and
        # reports the cap as inactive on a host where the app would have it active.
        # The app runs detection at startup, so the probe runs it too, and records
        # the flag on both sides of that call so the artifact shows the difference
        # rather than hiding it.
        obs["is_rocm_before_detect"] = getattr(hw, "IS_ROCM", None)
        obs["detected_device"] = str(hw.detect_hardware())
        obs["is_rocm_after_detect"] = getattr(hw, "IS_ROCM", None)
        obs["has_untrusted_predicate"] = hasattr(hw, "rocm_windows_free_is_untrusted")
        obs["has_trusted_mem_get_info"] = hasattr(hw, "trusted_mem_get_info")
        if obs["has_untrusted_predicate"]:
            obs["free_is_untrusted"] = bool(hw.rocm_windows_free_is_untrusted())
        obs["is_rocm_flag"] = getattr(hw, "IS_ROCM", None)
        if obs["has_trusted_mem_get_info"]:
            t_free, t_total = hw.trusted_mem_get_info()
            obs["trusted_free_gib"] = t_free / GIB
            obs["trusted_total_gib"] = t_total / GIB
        # What a guard in this checkout is actually handed. Where the capping
        # function does not exist the answer is the raw figure, which is the
        # whole point of comparing the two states.
        obs["guard_free_gib"] = obs.get("trusted_free_gib", obs.get("raw_free_gib"))
    except Exception as e:  # noqa: BLE001
        obs["hardware_error"] = f"{type(e).__name__}: {e}"
        obs.setdefault("guard_free_gib", obs.get("raw_free_gib"))

    try:
        # Not folded into the block above: a summary that throws must not cost the
        # trusted/raw pair, which is the measurement this probe exists for.
        obs["summary"] = hw.get_gpu_summary() if hw is not None else None
    except Exception as e:  # noqa: BLE001
        obs["summary_error"] = f"{type(e).__name__}: {e}"

    obs["directx"] = directx_adapters()
    obs["carveout"] = carveout_reading(args.checkout)
    if hw is not None:
        total_gib = obs.get("raw_total_gib")
        obs["planner"] = planner_readings(
            hw, None if total_gib is None else int(total_gib * GIB))

    # LAST, and under its own key prefix. Everything above is a bystander reading,
    # which is what the comparison uses; this allocates in the probe's OWN process
    # to show what the cap does when it has something to cap against. The two are
    # not interchangeable: `trusted_mem_get_info` bounds free by what THIS process
    # has reserved, so an allocation here moves it and another process's does not.
    if hw is not None and obs.get("raw_free_gib") is not None:
        obs["selfcheck"] = self_allocation_check(hw)

    args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
