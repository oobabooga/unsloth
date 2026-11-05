#!/usr/bin/env python3
"""Criteria: does a cold (no torch) vLLM status poll under HSA_OVERRIDE_GFX_VERSION reach the
verdict torch gives, on real gfx1151? Base defect: cold poll stuck at "Checking" for a readable
override. Fixed: cold == hot for every readable override, torch's gcnArchName == the override
name, and the poll never imports torch. Pairs with probes/rocm_override_status_probe.py."""

from __future__ import annotations

TITLE = "Cold vLLM status under HSA_OVERRIDE_GFX_VERSION, base versus head"
MODE = "differential"
NEEDS = ["gpu", "rocm", "multi_gpu", "nvidia", "windows"]

CHECKING = "Checking for a supported AMD GPU."
READABLE = {"'11.0.0'": "gfx1100", "'10.3.0'": "gfx1030", "'11.5.1'": "gfx1151", "'9.0.10'": "gfx90a"}


def _states(obs):
    return {n: v for n, v in obs.items() if not n.startswith("_")}


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out = []
    for name, st in _states(obs).items():
        cases = st.get("cases") or {}
        errs = [f"{c}/{m}: {r.get('error')}" for c, modes in cases.items()
                for m, r in modes.items() if r.get("error")]
        out.append((f"{name}: every probe ran", bool(cases) and not errs, "; ".join(errs)[:400]))
        base_case = (cases.get("''") or {}).get("hot") or {}
        kfd = base_case.get("kfd_targets") or []
        out.append((f"{name}: real gfx1151 in KFD, rocm platform",
                    "gfx1151" in kfd and base_case.get("platform") == "rocm",
                    f"kfd={kfd} platform={base_case.get('platform')}"))
        leaked = [c for c, m in cases.items() if (m.get("cold_prereq") or {}).get("torch_imported_by_poll")]
        out.append((f"{name}: cold poll never imports torch", not leaked, f"leaked in {leaked}"))
        arches = {c: (cases.get(c) or {}).get("hot", {}).get("torch_arches") for c in READABLE}
        out.append((f"{name}: torch reports a device under each readable override",
                    all(arches.values()), str(arches)))
    return out


def table(obs: dict) -> str:
    rows = ["| state | override | torch gcnArchName | cold (real) | cold (prereq stubbed) | hot (torch) |",
            "|---|---|---|---|---|---|"]
    for name, st in _states(obs).items():
        for case, modes in (st.get("cases") or {}).items():
            def a(m):
                r = modes.get(m) or {}
                return r.get("error") or str((r.get("answers") or [None])[0])
            rows.append(f"| {name} | {case} | {(modes.get('hot') or {}).get('torch_arches')} "
                        f"| {a('cold')} | {a('cold_prereq')} | {a('hot')} |")
    return "\n".join(rows)


def _first(r):
    return (r.get("answers") or [None])[0]


def base_shows_defect(base: dict) -> bool:
    cases = base.get("cases") or {}
    return all(
        (cases.get(c) or {}).get("cold_prereq", {}).get("answers") == [CHECKING] * 3
        and _first((cases.get(c) or {}).get("hot", {})) != CHECKING
        for c in READABLE
    )


def head_is_fixed(head: dict) -> bool:
    cases = head.get("cases") or {}
    for c, name in READABLE.items():
        modes = cases.get(c) or {}
        cold, hot = modes.get("cold_prereq") or {}, modes.get("hot") or {}
        if set(hot.get("torch_arches") or [None]) != {name}:
            return False
        if cold.get("answers") != [_first(hot)] * 3 or cold.get("answers_gpu0") != hot.get("answers_gpu0"):
            return False
    plain = cases.get("''") or {}
    if _first(plain.get("cold_prereq") or {}) != _first(plain.get("hot") or {}):
        return False
    return (cases.get("'11.0.0 '") or {}).get("cold_prereq", {}).get("answers") == [CHECKING] * 3
