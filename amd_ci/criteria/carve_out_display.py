#!/usr/bin/env python3
"""Criteria (#17509): on a carve-out APU (reserved part >= host-backed part) the floating monitor and VRAM tile
read the carve-out alone on head, carve-out plus shared on base."""

from __future__ import annotations

TITLE = "Strix Halo VRAM display: carve-out vs carve-out + shared RAM"
MODE = "differential"
NEEDS = ["gpu", "multi_gpu", "nvidia"]


def _states(obs):
    return {k: v for k, v in obs.items() if not k.startswith("_")}


def _apu(state):
    for d in (state.get("gpu") or {}).get("devices", []):
        if d.get("shared_memory") or d.get("unified_memory"):
            return d
    return None


def _split(state):
    t = ((state.get("display") or {}).get("totals")) or {}
    return t.get("dedicated"), t.get("shared")


def gates(obs):
    st = _states(obs)
    out = [("every state rendered a display", all(v.get("display") for v in st.values()),
            "; ".join(f"{k}: {v.get('payload_error') or v.get('display_error') or v.get('node_stderr', '')[-300:] or 'ok'}" for k, v in st.items()))]
    base = st.get("base") or {}
    apu = _apu(base)
    out.append(("backend reports a shared/unified APU", apu is not None, str(apu)[:400]))
    ded, sh = _split(base)
    out.append(("APU is a carve-out (reserved >= host-backed > 0)",
                bool(ded and sh and ded >= sh), f"dedicated={ded} shared={sh}"))
    return out


def table(obs):
    rows = ["| state | platform | devices (name, total, host-backed, used) | floating total | tile capacity | carve-out |", "|---|---|---|---|---|---|"]
    for k, v in _states(obs).items():
        devs = "; ".join(f"{d.get('name')} {d.get('memory_total_gb')} hb={d.get('shared_memory_host_backed_gb')} used={d.get('vram_used_gb')}"
                         for d in (v.get("gpu") or {}).get("devices", []))
        disp = v.get("display") or {}
        rows.append(f"| {k} | {v.get('platform')} | {devs} | {disp.get('floating_total_gb')} | {disp.get('tile_capacity')} | {disp.get('carve_out')} |")
    return "\n".join(rows)


def base_shows_defect(base):
    disp = base.get("display") or {}
    ded, sh = _split(base)
    return bool(ded and sh) and abs((disp.get("floating_total_gb") or 0) - (ded + sh)) < 0.02


def head_is_fixed(head):
    disp = head.get("display") or {}
    ded, _ = _split(head)
    carve = disp.get("carve_out") or {}
    return bool(ded) and abs((disp.get("floating_total_gb") or 0) - ded) < 0.02 and abs((carve.get("totalGb") or 0) - ded) < 0.02 \
        and disp.get("tile_capacity") == f"{carve.get('totalGb')}"
