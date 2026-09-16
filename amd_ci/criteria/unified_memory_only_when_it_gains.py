#!/usr/bin/env python3
"""Criteria: does Studio still push a fitting model into system RAM on this APU?

unsloth#7449, the half #9884 was meant to close. The defect is a model that FITS
the GPU carve-out being launched with GGML_CUDA_ENABLE_UNIFIED_MEMORY anyway,
which is what puts the weights in system RAM.

    base -> the commit before #9884, where any AMD APU got the flag
    head -> #9884's merge commit

The distinguishing case is deliberately the SMALL model, not the large one. A
criterion written on the large model would pass at both states, because managed
allocation is the right answer there and always was; only the small one separates
"decided" from "always said yes".

Pairs with probes/studio_unified_memory_probe.py.
"""

from __future__ import annotations

TITLE = "unsloth#7449: unified memory on a Windows Strix Halo APU"
MODE = "differential"

# Authored. The change is a ROCm APU memory policy, the report is Windows, and
# the same code path has a discrete-GPU and a multi-GPU branch this host cannot
# reach.
NEEDS = ["rocm", "gpu", "integrated_gpu", "windows", "windows_rocm_wddm",
         "discrete_gpu", "multi_gpu"]

# The model that must NOT get managed allocation: comfortably inside any
# plausible carve-out on a 96 GB-class Strix Halo, and inside a 16 GB one too.
SMALL_GIB = 4.0


def _decision(state: dict, gib: float):
    for d in state.get("decisions") or []:
        if abs(float(d.get("need_gib", -1)) - gib) < 1e-6:
            return d
    return None


def _pool_mib(state: dict):
    return (state.get("rocm_selected_pool_mib") or {}).get("value")


def _host_mib(state: dict):
    return (state.get("available_system_memory_mib") or {}).get("value")


def _device(state: dict) -> dict:
    devices = state.get("devices") or []
    return devices[0] if devices else {}


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out: list[tuple[str, bool, str]] = []
    for name in ("base", "head"):
        st = obs.get(name)
        if not st:
            continue
        out.append((f"{name} probe returned a reading", not st.get("child_failed"),
                    f"rc={st.get('rc')}; "
                    + str(st.get("torch_error") or st.get("backend_error")
                          or st.get("stderr_tail", ""))[-180:]))
        out.append((f"{name} torch is a ROCm build", bool(st.get("torch_hip")),
                    f"torch={st.get('torch_version')} hip={st.get('torch_hip')}"))
        dev = _device(st)
        arch = str(dev.get("gcnArchName") or "")
        out.append((f"{name} the device is the gfx1151 APU", "gfx1151" in arch,
                    f"{dev.get('name')} arch={arch or 'unreadable'} "
                    f"is_integrated={dev.get('is_integrated')}"))
        # Without a readable carve-out every decision below is False by default
        # rather than by judgement, and "no flag set" would mean nothing.
        out.append((f"{name} the carve-out is readable", bool(_pool_mib(st)),
                    f"_rocm_selected_pool_mib([0]) = {_pool_mib(st)} MiB; "
                    f"HIP total_memory = {dev.get('total_mib')} MiB"))
        out.append((f"{name} host RAM is readable", bool(_host_mib(st)),
                    f"_available_system_memory_mib() = {_host_mib(st)} MiB"))
        small = _decision(st, SMALL_GIB)
        out.append((f"{name} the decision answered for a {SMALL_GIB:.0f} GiB model",
                    bool(small) and "unified_memory" in small,
                    f"decided_by={None if not small else small.get('decided_by')}; "
                    f"{None if not small else small.get('error', '')}"))
    return out


def table(obs: dict) -> str:
    rows = ["| state | device | arch | integrated | carve-out MiB | host RAM MiB | "
            "decided by | 4 GiB model | 90 GiB model |", "|---|---|---|---|---|---|---|---|---|"]
    for name in ("base", "head", "merge"):
        st = obs.get(name)
        if not st:
            continue
        dev = _device(st)
        small, large = _decision(st, SMALL_GIB), _decision(st, 90.0)
        rows.append(
            f"| {name} | {dev.get('name')} | {dev.get('gcnArchName')} | "
            f"{dev.get('is_integrated')} | {_pool_mib(st)} | {_host_mib(st)} | "
            f"`{None if not small else small.get('decided_by')}` | "
            f"{None if not small else small.get('unified_memory')} | "
            f"{None if not large else large.get('unified_memory')} |")
    rows.append("")
    rows.append("`True` means the launch gets `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1`, "
                "which is what places the weights in system RAM.")
    for name in ("base", "head"):
        st = obs.get(name) or {}
        readout = st.get("gpu_summary") or st.get("gpu_readout_error")
        if readout:
            rows.append(f"- `{name}` Studio's own GPU readout: `{str(readout)[:400]}`")
    return "\n".join(rows)


def base_shows_defect(base: dict) -> tuple[bool, str]:
    d = _decision(base, SMALL_GIB)
    if not d or "unified_memory" not in d:
        return False, "the base state produced no decision for the small model"
    return bool(d["unified_memory"]), (
        f"a {SMALL_GIB:.0f} GiB model that fits the {_pool_mib(base)} MiB carve-out "
        f"{'DOES' if d['unified_memory'] else 'does not'} get managed allocation at the "
        f"base (decided by `{d.get('decided_by')}`)")


def head_is_fixed(head: dict) -> tuple[bool, str]:
    d = _decision(head, SMALL_GIB)
    if not d or "unified_memory" not in d:
        return False, "the head state produced no decision for the small model"
    return not bool(d["unified_memory"]), (
        f"at the head the same model {'still gets' if d['unified_memory'] else 'does not get'} "
        f"managed allocation (decided by `{d.get('decided_by')}`)")
