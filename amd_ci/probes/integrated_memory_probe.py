#!/usr/bin/env python3
"""Probe: what memory budget does this checkout report on a unified-memory GPU?

PR 6785 rewrites `_get_gpu_memory` into an nvidia-smi leg plus a torch leg, and
replaces `_available_system_memory_mib`'s cgroup source. Both are read on every
AMD APU too, and gfx1151 (Strix Halo) IS an integrated unified-memory part, so
this runner exercises the rewritten code even though the reported defect is
NVIDIA-only.

Observes only. Every figure that a judgment could be built on is recorded next to
the raw driver/kernel figure it should track, so the criteria module can compare
an OFFSET rather than an absolute: MemAvailable drifts between two probe runs,
"backend figure minus kernel figure" does not.

`mem_get_info` is sampled either side of the backend call for the same reason:
it brackets the moment the backend read it.

Pairs with criteria/integrated_memory_no_regression.py.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

MIB = 1024 * 1024


def import_backend(checkout: Path):
    """Import THIS checkout's llama.cpp backend, the way its own tests do."""
    backend = checkout / "studio" / "backend"
    if not backend.is_dir():
        raise SystemExit(f"no backend at {backend}")
    sys.path.insert(0, str(backend))
    # States are probed in one process each, but drop anything cached anyway:
    # importing the base checkout's `core.` and then the head's must not return
    # the first one.
    for stale in [m for m in sys.modules if m.split(".")[0] in ("core", "utils")]:
        del sys.modules[stale]
    os.chdir(backend)
    from core.inference.llama_cpp import LlamaCppBackend  # noqa: PLC0415
    return LlamaCppBackend


def read_meminfo() -> dict:
    out: dict = {}
    try:
        with open("/proc/meminfo", encoding = "utf-8") as f:
            for line in f:
                key, _, rest = line.partition(":")
                if key in ("MemTotal", "MemFree", "MemAvailable", "Cached", "SwapFree"):
                    out[key] = int(rest.split()[0]) // 1024  # kB -> MiB
    except Exception as e:  # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {e}"
    return out


def read_cgroup() -> dict:
    """The raw kernel side of the budget, so a disagreement between the two
    checkouts' cgroup helpers can be attributed rather than guessed at."""
    out: dict = {}
    try:
        out["proc_self_cgroup"] = Path("/proc/self/cgroup").read_text(encoding = "utf-8").strip()
    except Exception as e:  # noqa: BLE001
        out["proc_self_cgroup_error"] = f"{type(e).__name__}: {e}"
    rel = ""
    for line in str(out.get("proc_self_cgroup", "")).splitlines():
        fields = line.split(":", 2)
        if len(fields) == 3 and fields[0] == "0" and fields[1] == "":
            rel = fields[2]
    levels = []
    parts = [p for p in rel.split("/") if p]
    for i in range(len(parts), -1, -1):
        base = "/sys/fs/cgroup" + ("/" + "/".join(parts[:i]) if parts[:i] else "")
        lvl = {"dir": base}
        for name in ("memory.max", "memory.current"):
            try:
                lvl[name] = Path(f"{base}/{name}").read_text(encoding = "utf-8").strip()
            except Exception as e:  # noqa: BLE001
                lvl[name] = f"unreadable: {type(e).__name__}"
        levels.append(lvl)
    out["v2_levels"] = levels
    return out


def call(obj, name: str, *args):
    """Call an optional backend helper, recording absence and failure distinctly:
    a helper that does not exist at a state is not the same as one that raised."""
    fn = getattr(obj, name, None)
    if fn is None:
        return {"absent": True}
    try:
        value = fn(*args)
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}
    if isinstance(value, set):
        value = sorted(value)
    return {"value": value}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True, type = Path)
    ap.add_argument("--out", required = True, type = Path)
    args = ap.parse_args()
    # Resolved before import_backend chdir's into the checkout.
    out_path = args.out.resolve()
    checkout = args.checkout.resolve()

    obs: dict = {"state": args.state}
    obs["nvidia_smi_on_path"] = shutil.which("nvidia-smi")
    obs["meminfo_before"] = read_meminfo()
    obs["cgroup"] = read_cgroup()
    obs["env_masks"] = {
        k: os.environ.get(k)
        for k in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")
    }

    devices: list[dict] = []
    try:
        import torch
        obs["torch_version"] = torch.__version__
        obs["torch_hip"] = getattr(torch.version, "hip", None)
        obs["torch_cuda"] = getattr(torch.version, "cuda", None)
        obs["cuda_available"] = bool(torch.cuda.is_available())
        obs["device_count"] = int(torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            d: dict = {"ordinal": i}
            try:
                props = torch.cuda.get_device_properties(i)
                d["name"] = getattr(props, "name", None)
                d["arch"] = getattr(props, "gcnArchName", None)
                # Both spellings: torch renamed `integrated` to `is_integrated`.
                d["is_integrated"] = getattr(props, "is_integrated", None)
                d["integrated"] = getattr(props, "integrated", None)
                d["props_total_mib"] = getattr(props, "total_memory", 0) // MIB
            except Exception as e:  # noqa: BLE001
                d["props_error"] = f"{type(e).__name__}: {e}"
            try:
                free_b, total_b = torch.cuda.mem_get_info(i)
                d["mem_get_info_free_mib"] = free_b // MIB
                d["mem_get_info_total_mib"] = total_b // MIB
            except Exception as e:  # noqa: BLE001
                d["mem_get_info_error"] = f"{type(e).__name__}: {e}"
            devices.append(d)
    except Exception as e:  # noqa: BLE001
        obs["torch_error"] = f"{type(e).__name__}: {e}"
    obs["devices_before"] = devices

    try:
        backend = import_backend(checkout)
        obs["backend_file"] = sys.modules["core.inference.llama_cpp"].__file__
        # The figure the context fit is handed.
        obs["get_gpu_memory"] = call(backend, "_get_gpu_memory")
        # The figure the AMD APU unified-memory path prices RAM against.
        obs["available_system_memory_mib"] = call(backend, "_available_system_memory_mib")
        obs["total_system_memory_mib"] = call(backend, "_total_system_memory_mib")
        # Why the host is or is not treated as unified memory.
        obs["rocm_unified_memory_gpu_ids"] = call(backend, "_rocm_unified_memory_gpu_ids")
        obs["amd_apu_wants_unified_memory"] = call(backend, "_amd_apu_wants_unified_memory", None)
        obs["integrated_cuda_gpu_ids"] = call(backend, "_integrated_cuda_gpu_ids")
        obs["resolve_visible_physical_ids"] = call(backend, "_resolve_visible_physical_ids")
        # Helpers that exist at only one of the two states; `absent` is the answer
        # at the other, and that is itself part of the comparison.
        obs["gpu_is_integrated_0"] = call(backend, "_gpu_is_integrated", 0)
        obs["get_gpu_memory_via_torch"] = call(backend, "_get_gpu_memory_via_torch")
        obs["get_gpu_memory_via_nvidia_smi"] = call(backend, "_get_gpu_memory_via_nvidia_smi")
        obs["cgroup_available_memory_mib"] = call(backend, "_cgroup_available_memory_mib")
        obs["cgroup_memory_limit_mib"] = call(backend, "_cgroup_memory_limit_mib")
        obs["cgroup_memory_mib"] = call(backend, "_cgroup_memory_mib")
        obs["host_memory_mib"] = call(backend, "_host_memory_mib")
        obs["system_memory_budget_mib"] = call(backend, "_system_memory_budget_mib")
    except Exception as e:  # noqa: BLE001
        obs["backend_error"] = f"{type(e).__name__}: {e}"

    # Second bracket: the backend read the driver somewhere between these two.
    after: list[dict] = []
    try:
        import torch
        for i in range(torch.cuda.device_count()):
            free_b, total_b = torch.cuda.mem_get_info(i)
            after.append({"ordinal": i,
                          "mem_get_info_free_mib": free_b // MIB,
                          "mem_get_info_total_mib": total_b // MIB})
    except Exception as e:  # noqa: BLE001
        obs["torch_after_error"] = f"{type(e).__name__}: {e}"
    obs["devices_after"] = after
    obs["meminfo_after"] = read_meminfo()

    out_path.parent.mkdir(parents = True, exist_ok = True)
    out_path.write_text(json.dumps(obs, indent = 2, default = str), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
