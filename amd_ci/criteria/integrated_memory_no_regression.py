#!/usr/bin/env python3
"""Criteria: does PR 6785 change what an AMD unified-memory APU is told it has?

The defect the PR fixes is NVIDIA-only (GB10 / DGX Spark reports `[N/A]` to
nvidia-smi). It cannot reproduce here, so this is deliberately a REGRESSION
comparison and not a differential: on a gfx1151 Strix Halo the honest question is
whether the rewrite leaves the integrated AMD path exactly where it was. Running
it as a differential would be VOID by construction, and a VOID is not evidence.

Two claims the PR makes are testable on this host, and only on hardware:

  1. ROCm keeps `mem_get_info`. The override is guarded by `not is_rocm`, on the
     stated grounds that AMD APUs have their own unified path scoped to
     gfx1150/gfx1151. This runner IS gfx1151, and torch reports AMD APUs as
     integrated too, so it is precisely the device the guard exists for.
  2. `_available_system_memory_mib` is unchanged in value. The PR reimplements
     its cgroup source as an ancestor walk. That helper feeds the AMD APU RAM
     pricing, so a walk that binds tighter than the one in place would shrink an
     APU's budget -- a regression nobody is looking for, on hardware nobody
     tested it on.

Both are judged as OFFSETS against the kernel/driver figure read either side of
the backend call, never as absolutes: MemAvailable drifts between two probe runs
seconds apart, and "backend figure minus kernel figure" does not. A tolerance
loose enough to absorb drift would also absorb the defect.

Pairs with probes/integrated_memory_probe.py.
"""

from __future__ import annotations

TITLE = "AMD unified-memory APU (gfx1151): memory budget, base versus head"
MODE = "regression"

# Everything the change touches, not everything this host has. `nvidia`,
# `discrete_gpu`, `docker`, `multi_gpu` and `windows` are the parts of PR 6785
# this runner cannot speak to at all, and they belong in the report.
NEEDS = [
    "gpu", "rocm", "linux", "integrated_gpu",
    "nvidia", "discrete_gpu", "multi_gpu", "mig", "xpu", "mlx",
    "docker", "windows",
]

# MiB. The figures compared are offsets against a bracketed kernel reading, so
# ordinary drift is already cancelled; this absorbs rounding and the handful of
# MiB the probe itself allocates.
TOL = 512


def _val(state: dict, key: str):
    entry = state.get(key) or {}
    return entry.get("value") if isinstance(entry, dict) else None


def _rows(state: dict) -> list[list]:
    v = _val(state, "get_gpu_memory")
    return [list(r) for r in v] if isinstance(v, list) else []


def _driver_bracket(state: dict, ordinal: int) -> tuple[int | None, int | None, int | None, int | None]:
    """(free_lo, free_hi, total_lo, total_hi) across the readings either side."""
    frees, totals = [], []
    for key in ("devices_before", "devices_after"):
        for d in state.get(key) or []:
            if d.get("ordinal") != ordinal:
                continue
            if d.get("mem_get_info_free_mib") is not None:
                frees.append(int(d["mem_get_info_free_mib"]))
            if d.get("mem_get_info_total_mib") is not None:
                totals.append(int(d["mem_get_info_total_mib"]))
    if not frees or not totals:
        return None, None, None, None
    return min(frees), max(frees), min(totals), max(totals)


def _bracket_offset(value, lo, hi):
    """How far outside [lo, hi] the value sits; 0 when inside. None if unknown."""
    if value is None or lo is None or hi is None:
        return None
    if value < lo:
        return value - lo
    if value > hi:
        return value - hi
    return 0


def _mem_available_bracket(state: dict) -> tuple[int | None, int | None]:
    vals = []
    for key in ("meminfo_before", "meminfo_after"):
        m = state.get(key) or {}
        if m.get("MemAvailable") is not None:
            vals.append(int(m["MemAvailable"]))
    return (min(vals), max(vals)) if vals else (None, None)


def _gpu_free_offset(state: dict):
    """Reported free minus the driver's own free, for the first device."""
    rows = _rows(state)
    if not rows:
        return None
    idx, free = int(rows[0][0]), int(rows[0][1])
    lo, hi, _, _ = _driver_bracket(state, idx)
    return _bracket_offset(free, lo, hi)


def _gpu_total_offset(state: dict):
    rows = _rows(state)
    if not rows or len(rows[0]) < 3:
        return None
    idx, total = int(rows[0][0]), int(rows[0][2])
    _, _, lo, hi = _driver_bracket(state, idx)
    return _bracket_offset(total, lo, hi)


def _budget_offset(state: dict):
    """`_available_system_memory_mib` minus the kernel's MemAvailable."""
    lo, hi = _mem_available_bracket(state)
    return _bracket_offset(_val(state, "available_system_memory_mib"), lo, hi)


def _integrated(state: dict) -> bool:
    for d in state.get("devices_before") or []:
        if d.get("is_integrated") or d.get("integrated"):
            return True
    return bool(_val(state, "rocm_unified_memory_gpu_ids"))


def _states(obs: dict) -> list[tuple[str, dict]]:
    return [(n, v) for n, v in obs.items() if not n.startswith("_") and isinstance(v, dict)]


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out: list[tuple[str, bool, str]] = []
    pairs = _states(obs)

    broken = [n for n, v in pairs if v.get("backend_error")]
    out.append(("every state imported the backend", not broken,
                "; ".join(f"{n}: {obs[n]['backend_error']}" for n in broken) or "ok"))

    hips = {n: v.get("torch_hip") for n, v in pairs}
    out.append(("torch is a ROCm build at every state", all(hips.values()),
                ", ".join(f"{n}={h or 'None'}" for n, h in hips.items())))

    # Without this the run says nothing about integrated GPUs, which is the whole
    # question. A gfx1151 that reported itself discrete would be a finding in its
    # own right, so it fails the run rather than quietly weakening the claim.
    integ = {n: _integrated(v) for n, v in pairs}
    archs = {n: [d.get("arch") for d in (v.get("devices_before") or [])] for n, v in pairs}
    out.append(("the runner is a unified-memory integrated GPU", all(integ.values()),
                ", ".join(f"{n}: integrated={integ[n]} arch={archs[n]}" for n, _ in pairs)))

    empty = [n for n, v in pairs if not _rows(v)]
    out.append(("every state produced a GPU memory reading", not empty,
                ", ".join(f"{n}={_rows(obs[n])}" for n, _ in pairs)))

    budgets = {n: _val(v, "available_system_memory_mib") for n, v in pairs}
    out.append(("every state produced a system RAM budget",
                all(b is not None for b in budgets.values()),
                ", ".join(f"{n}={b}" for n, b in budgets.items())))

    # The offsets compare device to device; if the reported ids are not torch's
    # ordinals the two sides are not the same device and the comparison is void.
    aligned = True
    detail = []
    for n, v in pairs:
        ords = {d.get("ordinal") for d in (v.get("devices_before") or [])}
        ids = {int(r[0]) for r in _rows(v)}
        aligned = aligned and bool(ids) and ids == ords
        detail.append(f"{n}: reported={sorted(ids)} torch={sorted(ords)}")
    out.append(("reported GPU ids line up with torch ordinals", aligned, "; ".join(detail)))

    offsets = {n: _gpu_free_offset(v) for n, v in pairs}
    out.append(("every state's free figure could be bracketed against the driver",
                all(o is not None for o in offsets.values()),
                ", ".join(f"{n}={o}" for n, o in offsets.items())))
    return out


def table(obs: dict) -> str:
    rows = ["| state | arch | integrated | reported free/total MiB | driver free MiB "
            "| free offset | RAM budget MiB | MemAvailable MiB | budget offset |",
            "|---|---|---|---|---|---|---|---|---|"]
    for name, v in _states(obs):
        gpus = _rows(v)
        first = gpus[0] if gpus else [None, None, None]
        lo, hi, _, _ = _driver_bracket(v, int(first[0])) if gpus else (None, None, None, None)
        ml, mh = _mem_available_bracket(v)
        devs = v.get("devices_before") or []
        rows.append(
            f"| {name} | {devs[0].get('arch') if devs else '?'} "
            f"| {_integrated(v)} "
            f"| {first[1]}/{first[2]} "
            f"| {lo}-{hi} "
            f"| {_gpu_free_offset(v)} "
            f"| {_val(v, 'available_system_memory_mib')} "
            f"| {ml}-{mh} "
            f"| {_budget_offset(v)} |")

    extra = ["", "Helper availability per state (an `absent` helper is the PR's own "
                 "restructuring, not an error):", "",
             "| state | _gpu_is_integrated(0) | _cgroup_memory_mib | "
             "_cgroup_available_memory_mib | nvidia-smi on PATH |", "|---|---|---|---|---|"]
    for name, v in _states(obs):
        def cell(key: str) -> str:
            e = v.get(key) or {}
            if e.get("absent"):
                return "absent"
            if "error" in e:
                return f"error: {e['error']}"
            return f"`{e.get('value')}`"
        extra.append(f"| {name} | {cell('gpu_is_integrated_0')} | {cell('cgroup_memory_mib')} "
                     f"| {cell('cgroup_available_memory_mib')} "
                     f"| {v.get('nvidia_smi_on_path') or 'no'} |")
    return "\n".join(rows + extra)


def head_is_worse(base: dict, head: dict) -> tuple[bool, str]:
    findings: list[str] = []

    fb, fh = _gpu_free_offset(base), _gpu_free_offset(head)
    if fb is not None and fh is not None and abs(fh - fb) > TOL:
        findings.append(
            f"the free figure handed to the context fit moved relative to the driver: "
            f"offset {fb:+d} MiB at the base, {fh:+d} MiB at the head. On ROCm the PR "
            f"states mem_get_info is passed through untouched, so this is the "
            f"`not is_rocm` guard failing to hold on a real gfx1151 APU")

    tb, th = _gpu_total_offset(base), _gpu_total_offset(head)
    if tb is not None and th is not None and abs(th - tb) > TOL:
        findings.append(f"the reported total moved relative to the driver: "
                        f"{tb:+d} MiB at the base, {th:+d} MiB at the head")

    bb, bh = _budget_offset(base), _budget_offset(head)
    if bb is not None and bh is not None and abs(bh - bb) > TOL:
        direction = "tighter" if bh < bb else "looser"
        findings.append(
            f"_available_system_memory_mib moved relative to MemAvailable: offset "
            f"{bb:+d} MiB at the base, {bh:+d} MiB at the head, i.e. {direction}. That "
            f"helper prices the AMD APU unified-memory path, so the rewritten cgroup "
            f"source changes what this integrated GPU is allowed to use")

    if findings:
        return True, "; ".join(findings)

    detail = (f"on this gfx1151 APU the head reports the same budget as the base: GPU free "
              f"offset {fb}->{fh} MiB against the driver, system RAM budget offset "
              f"{bb}->{bh} MiB against MemAvailable, both within {TOL} MiB")
    if _val(head, "gpu_is_integrated_0") is not None:
        detail += (f". The PR's new `_gpu_is_integrated(0)` answers "
                   f"`{_val(head, 'gpu_is_integrated_0')}` on this part, and the ROCm guard "
                   f"kept mem_get_info regardless")
    return False, detail
