#!/usr/bin/env python3
"""Criteria: does the HIP runtime cap a single allocation below the memory the
machine actually has, and does the head state lift that cap?

Pairs with probes/gpu_memory_report_probe.py. The states are two HIP runtimes,
not two llama.cpp builds: `base` loads the shipped amdhip64_7.dll and `head`
loads the one under test.

The defect is PAL refusing an allocation the pool can hold. The rule the stock
runtime applies is a formula, not a constant:

    stock:    max(dedicated_VRAM_heap, 0.75 x shared_GART_heap)
    patched:  max(dedicated_VRAM_heap, 1.00 x shared_GART_heap)

so the factor only bites where the GART branch wins, i.e. a small carve-out
against a large aperture. An earlier version of this file asserted a flat 64 GiB
clamp; the devlab box measured 110.2 GiB, which is the GART branch, and the
constant is what a fixed number would have hidden.

Three things this is careful about:

  * A cap is only a defect if there is memory beyond it. `gates` requires a pool
    larger than the clamp before any comparison is shown.
  * A hipMalloc that returns a pointer is not evidence the memory exists; the
    probe writes and reads back every candidate, and a state whose confirmation
    passes were unstable is reported rather than averaged away.
  * The search is deliberately bounded below the machine, because committing
    near all of RAM can bugcheck the host. A state that clears the whole range
    demonstrates no cap, so a base that clears it makes the run VOID rather
    than CONFIRMED, and a head that clears it is the fix showing.
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


def _single(state: dict) -> dict | None:
    """The single-probe reading, when the run asked one question instead of searching.

    A bisecting search tries its own ceiling, and on a runtime that lifts the cap that
    attempt SUCCEEDS near the whole machine and starves the OS -- one devlab host was
    lost that way. Where the stock boundary is already known, one attempt just above it
    answers the same question and allocates nothing larger.
    """
    cap = _sec(state, "alloc_cap")
    return cap if cap.get("mode") == "single_probe" else None


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
        one = _single(st)
        if one is not None:
            out.append((f"{name}: the probe ran and its 1 GiB floor passed",
                        one.get("ok") is not None and not one.get("error"),
                        one.get("error") or f"{one.get('probe_gib')} GiB ok={one.get('ok')}"))
        else:
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
    one = _single(base)
    if one is not None:
        # The refusal IS the defect: the shipped runtime turning down memory the
        # machine holds. A base that succeeds has nothing for the head to lift.
        if one.get("error"):
            return False, str(one["error"])
        if one.get("any_ok"):
            return False, (f"the shipped runtime allocated {one.get('probe_gib')} GiB, so it "
                           f"refuses nothing here and there is no cap to lift")
        return True, (f"the shipped runtime refused {one.get('probe_gib')} GiB on a "
                      f"{_host_pool_gib(base)} GiB host, over {one.get('reps')} attempts")
    cap = _cap(base)
    if cap is None:
        return False, "the base cap search produced no boundary"
    if _sec(base, "alloc_cap").get("capped") is False:
        return False, (f"the base allocated the whole {cap} GiB search range, so it refused "
                       f"nothing and there is no cap here to lift")
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


def head_is_fixed(head: dict, base: dict | None = None) -> tuple[bool, str]:
    """A lifted cap is a CHANGE, so this reads the base when the engine offers it.

    Judging an arm against a fixed clamp alone is wrong on a host whose stock cap
    already sits above that clamp: the devlab box refuses at 110.2 GiB, so an
    inert runtime would clear a 64 GiB bar and read as fixed. The clamp rule is
    kept for the single-arm call, where there is nothing better to compare to.
    """
    one = _single(head)
    if one is not None:
        if one.get("error"):
            return False, str(one["error"])
        if not one.get("stable"):
            return False, (f"{one.get('probe_gib')} GiB succeeded on some attempts and not "
                           f"others, which is fragmentation rather than a lifted cap")
        if one.get("ok"):
            return True, (f"allocated {one.get('probe_gib')} GiB, which the shipped runtime "
                          f"refused, on every one of {one.get('reps')} attempts")
        return False, f"still refused {one.get('probe_gib')} GiB, so the cap did not move"
    h = _cap(head)
    if h is None:
        return False, "the cap search produced no boundary"
    pool = _host_pool_gib(head)
    # The search stops short of the machine on purpose, so clearing the range is
    # the strongest statement available and the one the bound was chosen to make.
    if _sec(head, "alloc_cap").get("capped") is False:
        return True, (f"allocated the whole {h} GiB search range without being refused, on a "
                      f"{pool} GiB host; the range stops below the machine deliberately")
    if not _stable(head):
        return False, (f"reached {h} GiB, but the boundary did not reproduce in both directions, "
                       f"so it is a fragmentation reading rather than a cap")
    b = _cap(base) if base else None
    if b is not None:
        step = float(_sec(head, "alloc_cap").get("resolution_gib") or 0.5)
        if h > b + step:
            return True, f"allocated {h} GiB where the shipped runtime refused past {b} GiB"
        return False, (f"capped at {h} GiB against the shipped runtime's {b} GiB, which is "
                       f"inside the {step} GiB search resolution: nothing moved")
    if h <= CLAMP_GIB + CLAMP_TOL_GIB:
        return False, f"still capped at {h} GiB, at or below the {CLAMP_GIB} GiB clamp"
    return True, f"allocated {h} GiB, past the {CLAMP_GIB} GiB clamp, on a {pool} GiB host"


def table(obs: dict) -> str:
    rows = ["| state | HIP dll | HIP total | registry VRAM + RAM | single allocation | stable |",
            "|---|---|---|---|---|---|"]
    for name in ("base", "head", *[k for k in obs if k not in ("base", "head")
                                   and not k.startswith("_")]):
        st = obs.get(name)
        if not isinstance(st, dict):
            continue
        one = _single(st)
        if one is not None:
            verdict = "refused" if not one.get("any_ok") else (
                "ok" if one.get("ok") else "ok on some attempts")
            reading = f"{one.get('probe_gib')} GiB {verdict}"
            stable = one.get("stable", False)
        else:
            reading = f"{_cap(st)} GiB max"
            stable = _stable(st)
        rows.append(f"| {name} | `{_dll(st)}` | {_hip_total_gib(st)} GiB | "
                    f"{_host_pool_gib(st)} GiB | {reading} | "
                    f"{'yes' if stable else 'NO'} |")
    return "\n".join(rows)
