#!/usr/bin/env python3
"""Criteria for probes/hip_context_probe.py: no Studio step may open a KFD process, hold more GPU memory or
newly error at head. Gated on the plain torch steps (a KFD process after mem_get_info, a 256 MiB allocation
seen by the counter), or a blind counter would pass every comparison."""

from __future__ import annotations

TITLE = "GPU (KFD) process and VRAM per metadata call, base versus head"
MODE = "regression"
NEEDS = ["gpu", "nvidia", "multi_gpu"]
MIB = 1024 ** 2
TOL_BYTES = 64 * MIB
TORCH_STEPS = ("import_torch", "is_available", "device_count", "get_device_properties",
               "properties_then_mem_get_info", "allocate_256mib")


def _states(obs: dict) -> dict:
    return {k: v for k, v in obs.items() if not k.startswith("_")}


def _end(runs: list) -> tuple[bool, int, bool]:
    """(opened a KFD process in any repeat, max VRAM bytes, any repeat errored)."""
    opened = any((r.get("end") or {}).get("kfd_process") for r in runs)
    vram = max([(r.get("end") or {}).get("gpu_bytes", 0) or 0 for r in runs] or [0])
    errored = any(r.get("error") for r in runs)
    return opened, vram, errored


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out = []
    for name, o in _states(obs).items():
        steps = o.get("steps") or {}
        out.append((f"{name}: KFD sysfs readable", bool(o.get("kfd_sysfs")), str(o.get("kfd_sysfs"))))
        hip = [r.get("torch_hip") for r in steps.get("import_torch", [])]
        out.append((f"{name}: torch is a ROCm build", any(hip), str(hip)))
        count = [r.get("value") for r in steps.get("device_count", [])]
        out.append((f"{name}: a GPU is visible", any((c or 0) > 0 for c in count), str(count)))
        errs = [s for s in TORCH_STEPS if _end(steps.get(s, []))[2]]
        out.append((f"{name}: plain torch steps ran", not errs, ", ".join(errs) or "all ran"))
        ctl = _end(steps.get("properties_then_mem_get_info", []))
        out.append((f"{name}: positive control opened a KFD process", ctl[0],
                    f"{ctl[1] / MIB:.0f} MiB"))
        # The increase across the allocation, read after the context exists: an absolute total would
        # pass a counter blind to the allocation whenever the context alone reads 200 MiB.
        grew = max([((r.get("end") or {}).get("gpu_bytes", 0) or 0)
                    - ((r.get("before_alloc") or {}).get("gpu_bytes", 0) or 0)
                    for r in steps.get("allocate_256mib", []) if r.get("before_alloc")] or [0])
        out.append((f"{name}: a 256 MiB allocation is seen by the counter", grew >= 200 * MIB,
                    f"+{grew / MIB:.0f} MiB across the allocation"))
    return out


def table(obs: dict) -> str:
    states = _states(obs)
    names = list(states)
    rows = ["| step | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    steps = []
    for o in states.values():
        steps += [s for s in (o.get("steps") or {}) if s not in steps]
    for step in steps:
        cells = []
        for o in states.values():
            runs = (o.get("steps") or {}).get(step, [])
            opened, vram, errored = _end(runs)
            cell = (f"KFD process, {vram / MIB:.0f} MiB" if opened else "no KFD process")
            if errored:
                cell += "; error: " + next(str(r["error"])[:80] for r in runs if r.get("error"))
            cells.append(cell)
        rows.append(f"| {step} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def head_is_worse(base: dict, head: dict) -> tuple[bool, str]:
    bad = []
    for step, hruns in (head.get("steps") or {}).items():
        if not step.startswith("studio_"):
            continue
        b = _end((base.get("steps") or {}).get(step, []))
        h = _end(hruns)
        if h[0] and not b[0]:
            bad.append(f"{step} opens a KFD process the base did not")
        if h[1] > b[1] + TOL_BYTES:
            bad.append(f"{step} holds {h[1] / MIB:.0f} MiB vs base {b[1] / MIB:.0f} MiB")
        if h[2] and not b[2]:
            bad.append(f"{step} errors at head only")
    if bad:
        return True, "; ".join(bad)
    return False, "every Studio step holds no more GPU state at head than at base"
