#!/usr/bin/env python3
"""Criteria for #10656: base routes a Windows gfx1151 host to Vulkan, head keeps ROCm."""

from __future__ import annotations

TITLE = "Automatic llama.cpp backend on a Windows Strix Halo host, base versus head"
MODE = "differential"
NEEDS: list[str] = ["windows", "gpu"]
RUNS = ("detected", "forwarded_gfx1151", "remembered_gfx1151")
# The judged run: setup.ps1 and the updater forward the arch (--rocm-gfx), so this is the
# path a Desktop install takes. Without it this runner has no HIP SDK, detection reports no
# ROCm and both states take the #7357 Vulkan route; that run must merely agree across states.
JUDGED = "forwarded_gfx1151"
ALSO_JUDGED = ("remembered_gfx1151",)


def _auto(o: dict, run: str):
    return ((((o or {}).get("runs") or {}).get(run) or {}).get("auto") or {}).get("resolved_backend")


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out = []
    for name in ("base", "head"):
        o = obs.get(name) or {}
        host = o.get("host") or {}
        out.append((f"{name} sees a Windows host", host.get("is_windows") is True, str(host)))
        out.append((f"{name} has an AMD Vulkan driver registered",
                    host.get("amd_vulkan_icd_present") is True, str(host)))
        for run in RUNS:
            r = (o.get("runs") or {}).get(run) or {}
            out.append((f"{name} {run} resolved an auto backend", _auto(o, run) is not None,
                        r.get("error") or f"rc={r.get('rc')} auto={r.get('auto')}"))
        rocm = ((o.get("runs") or {}).get(JUDGED) or {}).get("rocm") or {}
        out.append((f"{name} {JUDGED} has a ROCm bundle for this host",
                    rocm.get("available") is True, str(rocm)))
    base, head = obs.get("base") or {}, obs.get("head") or {}
    out.append(("unforwarded run identical across states (untouched path)",
                _auto(base, "detected") == _auto(head, "detected"),
                f"base={_auto(base, 'detected')} head={_auto(head, 'detected')}"))
    return out


def table(obs: dict) -> str:
    rows = ["| state | run | auto resolves to | auto asset | rocm asset |", "|---|---|---|---|---|"]
    for name in ("base", "head"):
        o = obs.get(name) or {}
        for run in RUNS:
            r = (o.get("runs") or {}).get(run) or {}
            rows.append(f"| {name} | {run} | {_auto(o, run)} | {(r.get('auto') or {}).get('asset')} "
                        f"| {(r.get('rocm') or {}).get('asset')} |")
    return "\n".join(rows)


def base_shows_defect(base: dict) -> tuple[bool, str]:
    got = {r: _auto(base, r) for r in (JUDGED, *ALSO_JUDGED)}
    return all(v == "vulkan" for v in got.values()), f"base auto: {got}"


def head_is_fixed(head: dict) -> tuple[bool, str]:
    got = {r: _auto(head, r) for r in (JUDGED, *ALSO_JUDGED)}
    return all(v == "rocm" for v in got.values()), f"head auto: {got}"
