#!/usr/bin/env python3
"""Criteria: on Windows ROCm, is the cap the only thing between a plan and a spill?

The defect is the PLATFORM's, not the checkout's. AMD documents hipMemGetInfo's
free half on Windows as accounting only for the calling process and "may be
optimistic": WDDM gives a process its own budget, so a fresh process is told the
card is nearly empty however much another process has resident. An over-commit
built on that figure does not fail, WDDM satisfies it from host RAM, and the
process quietly runs out of the card instead.

So the two halves of the question are asked separately.

  base_shows_defect  the CHARACTERISATION. With the holder resident, does the raw
                     reading still claim the card is nearly free? If instead free
                     drops by roughly what is held, this silicon does not have the
                     problem the cap exists for, and the run is VOID: nothing about
                     the cap can be concluded from a host that never over-reports.

  head_is_fixed      the GUARD. Is the figure a guard in this checkout is handed
                     bounded by what the card can still hold, `total - held`,
                     within tolerance?

Tolerance follows the Linux criteria: the tighter of 2% of total and 25% of the
holder, so a large unified pool cannot make a half-cap look like a cap.

Non-vacuity: the holder must really be resident, the probe must be a bystander,
and this must be a Windows ROCm host with a readable reading. Everything else is
INCONCLUSIVE, which is the honest answer for "the question was never reached" and
in particular for a Windows box carrying no ROCm torch.

Pairs with probes/windows_free_probe.py and probes/vram_holder_fixture.py.
"""

from __future__ import annotations

GIB = 1024 ** 3
TITLE = "Windows ROCm free VRAM against a resident bystander (Strix Halo)"
MODE = "differential"
# Authored for the CHANGE, not for the host. The cap decides a budget on every
# platform the backend runs on, so each one it could be wrong on is declared and
# the report bounds itself to the one that ran.
NEEDS = ["gpu", "rocm", "windows", "windows_rocm_wddm", "integrated_gpu",
         "discrete_gpu", "multi_gpu", "nvidia", "mig", "xpu", "mlx"]

# The fraction of total above which a reading is an over-report rather than an
# honest one. With 8 GiB of a 64 GiB pool held, an honest free is at most ~0.88
# of total, so 0.9 cannot be met by accident.
OVER_REPORT_FRACTION = 0.9

_CTX: dict = {"held": 0.0, "tol": 0.5, "total": 0.0}


def _held(obs: dict) -> float:
    return float((obs.get("_fixture") or {}).get("allocated_gib") or 0.0)


def _states(obs: dict) -> dict:
    return {n: v for n, v in obs.items() if not n.startswith("_")}


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    held = _held(obs)
    states = _states(obs)
    ref = states.get("head") or (next(iter(states.values())) if states else {})
    total = float(ref.get("raw_total_gib") or 0.0)
    _CTX["held"] = held
    _CTX["total"] = total
    _CTX["tol"] = min(max(0.5, 0.02 * total), max(0.5, 0.25 * held)) if held else 0.5

    out: list[tuple[str, bool, str]] = []
    out.append(("holder really held >= 2 GiB", held >= 2.0, f"{held:.2f} GiB"))

    # A ROCm torch on Windows is the whole precondition. The hardware gate in the
    # workflow fails first when it is absent; this repeats the check from the
    # observation so an artifact read on its own still says why.
    is_win = str(ref.get("platform") or "") == "Windows"
    hip = ref.get("torch_hip")
    out.append(("Windows with a ROCm torch", is_win and bool(hip),
                f"platform={ref.get('platform') or '-'} torch={ref.get('torch_version', '-')} "
                f"hip={hip or 'none'} error={ref.get('torch_error', '-')}"))

    untrusted = ref.get("free_is_untrusted")
    out.append(("the checkout's own predicate says free is untrusted here",
                untrusted is True,
                f"rocm_windows_free_is_untrusted()={untrusted} IS_ROCM={ref.get('is_rocm_flag')}"))

    readable = all(v.get("raw_free_gib") is not None and v.get("guard_free_gib") is not None
                   for v in states.values())
    out.append(("every state produced a raw and a guard-facing reading", readable,
                "" if readable else "a state failed to read one of the two"))

    bystanders = {n: v.get("observer_allocated_gib", 0) for n, v in states.items()}
    out.append(("the probe stayed a bystander",
                all((v or 0) < 0.5 for v in bystanders.values()),
                ", ".join(f"{k}={(v or 0):.2f}" for k, v in bystanders.items())))
    return out


def table(obs: dict) -> str:
    held, total, tol = _CTX["held"], _CTX["total"], _CTX["tol"]
    ceiling = max(0.0, total - held)
    rows = ["| state | raw free | raw/total | guard-facing free | ceiling (total - held) | "
            "trusted present | props total |",
            "|---|---|---|---|---|---|---|"]
    for name, v in _states(obs).items():
        raw = v.get("raw_free_gib")
        guard = v.get("guard_free_gib")
        if raw is None:
            rows.append(f"| {name} | error: {v.get('torch_error') or v.get('hardware_error')} "
                        f"| | | | | |")
            continue
        frac = v.get("raw_free_fraction")
        rows.append(
            f"| {name} | {raw:.2f} | {'-' if frac is None else f'{frac:.3f}'} | "
            f"{'-' if guard is None else f'{guard:.2f}'} | {ceiling:.2f} | "
            f"{v.get('has_trusted_mem_get_info')} | {v.get('props_total_gib', 0):.2f} |")
    rows.append("")
    rows.append(f"Holder took {held:.2f} GiB of a {total:.2f} GiB pool; tolerance {tol:.2f} GiB "
                f"(the tighter of 2% of total and 25% of the holder). A reading above "
                f"{OVER_REPORT_FRACTION:.2f} of total is the documented WDDM over-report.")

    dx = next((v.get("directx") for v in _states(obs).values() if (v.get("directx") or {}).get("available")), None)
    if dx:
        rows += ["", "DirectX adapter record, which is where the Windows carve-out figure "
                     "comes from:", "", "| figure | GiB |", "|---|---|"]
        for key, label in (("amd_dedicated_gib", "AMD dedicated (video + system)"),
                           ("amd_shared_gib", "AMD shared system memory")):
            if dx.get(key) is not None:
                rows.append(f"| {label} | {dx[key]:.2f} |")
        rows.append(f"| AMD adapters found | {dx.get('amd_count', 0)} |")
    carve = next((v.get("carveout") for v in _states(obs).values() if v.get("carveout")), None)
    if carve and carve.get("igpu_dedicated_memory_gib") is not None:
        rows.append(f"| carve-out advice would read | "
                    f"{carve['igpu_dedicated_memory_gib']:.2f} |")
    return "\n".join(rows)


def base_shows_defect(base: dict) -> bool:
    """Does the raw reading over-report while the holder is resident?"""
    raw, total = base.get("raw_free_gib"), base.get("raw_total_gib")
    if raw is None or not total:
        return False
    return (raw / total) >= OVER_REPORT_FRACTION and _CTX["held"] >= 2.0


def head_is_fixed(head: dict) -> bool:
    """Is the guard-facing figure bounded by what the card can still hold?"""
    guard, total = head.get("guard_free_gib"), head.get("raw_total_gib")
    if guard is None or not total:
        return False
    return guard <= (total - _CTX["held"]) + _CTX["tol"]
