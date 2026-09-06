#!/usr/bin/env python3
"""Criteria: does the HIP runtime cap a single allocation below the memory the
machine actually has, and does the head state lift that cap?

Pairs with probes/gpu_memory_report_probe.py. The states are two HIP runtimes,
not two llama.cpp builds: `base` loads the shipped amdhip64_7.dll and `head`
loads the one under test.

The defect is PAL reporting a constant rather than a measurement. ROCm/rocm-systems
Sep 4 2026 (`fix(clr): report the full unified memory pool on large-memory APUs`)
clamps `maxAllocSize_` at 64 GiB and derives the pool from the largest single
heap; the fix raises the clamp and credits the aperture as
`gart - min(gart / 4, 4 GiB)`. So on an APU whose unified pool exceeds 64 GiB,
the stock runtime refuses an allocation the hardware can hold.

Two things this is careful about:

  * A cap is only a defect if there is memory beyond it. On a machine whose
    carve-out IS 64 GiB the clamp is invisible, and calling that a pass would be
    reading a coincidence as a fix. `gates` therefore requires a pool larger
    than the clamp before any comparison is shown.
  * A hipMalloc that returns a pointer is not evidence the memory exists; the
    probe writes and reads back every candidate, and a state whose confirmation
    passes were unstable is reported rather than averaged away.
"""

from __future__ import annotations

GIB = 1024 ** 3
TITLE = "single-allocation cap on gfx1151: stock HIP runtime vs the one under test"
MODE = "differential"
NEEDS = ["gpu", "rocm", "integrated_gpu", "windows", "windows_rocm_wddm",
         "discrete_gpu", "nvidia", "multi_gpu", "multi_gpu_amd"]

# The constant PAL clamps to on the stock runtime. Not a tolerance: a cap that
# lands here is the clamp, and a cap elsewhere is something else and is named.
CLAMP_GIB = 64.0
CLAMP_TOL_GIB = 1.0


def _sec(state: dict, name: str) -> dict:
    return ((state or {}).get("sections") or {}).get(name) or {}


def _cap(state: dict) -> float | None:
    v = _sec(state, "alloc_cap").get("max_ok_gib")
    return float(v) if v is not None else None


def _hip_total_gib(state: dict) -> float | None:
    devices = _sec(state, "hip").get("devices") or []
    if not devices:
        return None
    total = devices[0].get("total_bytes") or devices[0].get("device_total_bytes")
    return round(total / GIB, 2) if total else None


def _host_pool_gib(state: dict) -> float | None:
    """The memory a single allocation could plausibly be backed by: the carve-out
    the registry records plus the host RAM the aperture maps."""
    host = _sec(state, "host")
    vram = 0
    for adapter in host.get("adapters") or []:
        size = adapter.get("qw_memory_size") or 0
        if isinstance(size, int):
            vram = max(vram, size)
    ram = host.get("total_phys_bytes") or host.get("memtotal_bytes") or 0
    if not vram and not ram:
        return None
    return round((vram + ram) / GIB, 2)


def _stable(state: dict) -> bool:
    cap = _sec(state, "alloc_cap")
    return bool(cap.get("boundary_stable", True))


def _dll(state: dict) -> str:
    return ((state or {}).get("hip_dll_file") or {}).get("sha256", "")[:12] or "?"


def gates(obs: dict) -> list:
    """Non-vacuity: measure nothing until the machine could show the defect."""
    out = []
    for name in ("base", "head"):
        st = obs.get(name) or {}
        cap = _sec(st, "alloc_cap")
        out.append((f"{name}: the cap search completed",
                    cap.get("max_ok_gib") is not None,
                    cap.get("error") or f"max_ok_gib={cap.get('max_ok_gib')}"))
        out.append((f"{name}: HIP reported a device",
                    bool(_sec(st, "hip").get("devices")),
                    _sec(st, "hip").get("error", "no devices")))
    base, head = obs.get("base") or {}, obs.get("head") or {}
    pool = _host_pool_gib(base)
    out.append(("this host has memory beyond the clamp, so a cap at it is visible",
                bool(pool and pool > CLAMP_GIB + CLAMP_TOL_GIB),
                f"carve-out plus host RAM = {pool} GiB against a {CLAMP_GIB} GiB clamp"))
    # Two states that loaded the same runtime cannot differ; that is a
    # mis-wired matrix, not a result about the runtime.
    out.append(("the two states loaded different HIP runtimes",
                _dll(base) != _dll(head) and "?" not in (_dll(base), _dll(head)),
                f"base {_dll(base)}, head {_dll(head)}"))
    return out


def base_shows_defect(base: dict) -> tuple[bool, str]:
    cap = _cap(base)
    if cap is None:
        return False, "the base cap search produced no boundary"
    pool = _host_pool_gib(base)
    if pool and cap >= pool - CLAMP_TOL_GIB:
        return False, (f"the base allocated {cap} GiB against a {pool} GiB pool, so nothing "
                       f"is being refused")
    if abs(cap - CLAMP_GIB) <= CLAMP_TOL_GIB:
        return True, (f"the base capped at {cap} GiB, which is the {CLAMP_GIB} GiB PAL clamp, "
                      f"on a host holding {pool} GiB")
    return True, (f"the base capped at {cap} GiB, below the {pool} GiB the host holds; this is "
                  f"not the {CLAMP_GIB} GiB clamp, so the cause is something else and the "
                  f"number is the finding")


def head_is_fixed(head: dict) -> tuple[bool, str]:
    """Judged on its own, because the engine applies this to every non-base arm,
    including controls, and an arm is only "fixed" if it clears the clamp itself."""
    h = _cap(head)
    if h is None:
        return False, "the cap search produced no boundary"
    if not _stable(head):
        return False, (f"reached {h} GiB, but the boundary did not reproduce in both directions, "
                       f"so it is a fragmentation reading rather than a cap")
    if h <= CLAMP_GIB + CLAMP_TOL_GIB:
        return False, f"still capped at {h} GiB, at or below the {CLAMP_GIB} GiB clamp"
    pool = _host_pool_gib(head)
    return True, f"allocated {h} GiB, past the {CLAMP_GIB} GiB clamp, on a {pool} GiB host"


def table(obs: dict) -> str:
    rows = ["| state | HIP dll | HIP total | registry VRAM + RAM | max single alloc | stable |",
            "|---|---|---|---|---|---|"]
    for name in ("base", "head", *[k for k in obs if k not in ("base", "head")
                                   and not k.startswith("_")]):
        st = obs.get(name)
        if not isinstance(st, dict):
            continue
        rows.append(f"| {name} | `{_dll(st)}` | {_hip_total_gib(st)} GiB | "
                    f"{_host_pool_gib(st)} GiB | {_cap(st)} GiB | "
                    f"{'yes' if _stable(st) else 'NO'} |")
    return "\n".join(rows)
