#!/usr/bin/env python3
"""Criteria: Studio's sd.cpp default speed profile forces --diffusion-conv-direct on Vulkan / ROCm. Is that a
measured loss on gfx1151, and does the head drop it there while keeping --diffusion-fa?

Pairs with probes/sdcpp_conv_probe.py. Differential.

Per (accelerator, model) the shared A/B matrix gives conv-direct ON and OFF runs, interleaved. noise = the larger
relative spread (max - min) / median of the two arms, floored at 2%. A cell is
  SLOWER  median(on) > median(off) * (1 + noise)
  FASTER  median(on) < median(off) * (1 - noise)
  NEUTRAL otherwise.

base shows the defect: on an accelerator whose sd-cli lists the GPU, the base's real sd-cli argv carries
conv-direct, at least one model is SLOWER with it and none FASTER.
head is fixed: on every such accelerator the head's sd-cli AND sd-server argv carry --diffusion-fa and not
conv-direct, and the head's argv rendered (rc 0) on the card.
"""

from __future__ import annotations

import statistics

TITLE = "sd.cpp --diffusion-conv-direct on gfx1151 (Vulkan / ROCm builds Studio installs)"
MODE = "differential"
NEEDS = ["rocm", "gpu", "linux", "discrete_gpu", "windows", "nvidia"]
NOISE_FLOOR = 0.02


def _sh(obs: dict) -> dict:
    for name in ("base", "head"):
        s = (obs.get(name) or {}).get("shared") if name in obs else None
        if s:
            return s
    return (obs or {}).get("shared") or {}


def _times(runs: list) -> list:
    return [r["sampling_s"] for r in runs or [] if r.get("rc") == 0 and isinstance(r.get("sampling_s"), (int, float))]


def cell(m: dict) -> dict:
    on, off = _times(m.get("on")), _times(m.get("off"))
    if len(on) < 2 or len(off) < 2:
        return {"ok": False, "on": on, "off": off}
    mon, moff = statistics.median(on), statistics.median(off)
    noise = max(NOISE_FLOOR, (max(on) - min(on)) / mon, (max(off) - min(off)) / moff)
    if mon > moff * (1 + noise):
        verdict = "SLOWER"
    elif mon < moff * (1 - noise):
        verdict = "FASTER"
    else:
        verdict = "NEUTRAL"
    return {"ok": True, "on": on, "off": off, "med_on": mon, "med_off": moff, "ratio": mon / moff,
            "noise": noise, "verdict": verdict}


def _cells(sh: dict) -> dict:
    out = {}
    for accel, models in (sh.get("matrix") or {}).items():
        if not isinstance(models, dict) or "skipped" in models:
            continue
        for model, m in models.items():
            if isinstance(m, dict) and "skipped" not in m:
                out[(accel, model)] = cell(m)
    return out


def _gpu_accels(sh: dict) -> list:
    return [a for a, r in (sh.get("installs") or {}).items() if r.get("gpu_listed")]


def gates(obs: dict) -> list:
    sh = _sh(obs)
    accels = _gpu_accels(sh)
    cells = _cells(sh)
    out = [("an installed sd.cpp build lists the AMD GPU", bool(accels),
            ", ".join(f"{a}: rc={r.get('rc')} listed={r.get('gpu_listed')}" for a, r in (sh.get("installs") or {}).items()))]
    z = [k for k, c in cells.items() if k[1] == "zimage" and c["ok"]]
    out.append(("Z-Image conv on/off measured (>=2 ok runs per arm) on a GPU build", bool(z),
                ", ".join(f"{a}/{m}: on {c['on']} off {c['off']}" for (a, m), c in cells.items())))
    for name in ("base", "head"):
        st = (obs.get(name) or {}).get("studio_argv") or {}
        got = [a for a in accels if (st.get(a) or {}).get("cli_cmd")]
        out.append((f"{name}: the checkout's own SdCppEngine argv captured on every GPU build", bool(accels)
                    and set(got) == set(accels), f"captured for {got or 'none'}; errors "
                    f"{ {a: (st.get(a) or {}).get('error') for a in accels} }"))
    return out


def table(obs: dict) -> str:
    sh = _sh(obs)
    rows = ["| accel | model | conv ON sampling s | conv OFF sampling s | ON/OFF | noise | verdict |",
            "|---|---|---|---|---|---|---|"]
    for (a, m), c in sorted(_cells(sh).items()):
        if not c["ok"]:
            rows.append(f"| {a} | {m} | {c['on']} | {c['off']} | - | - | not measured |")
            continue
        rows.append(f"| {a} | {m} | {c['med_on']:.2f} {c['on']} | {c['med_off']:.2f} {c['off']} | "
                    f"{c['ratio']:.3f} | {c['noise']:.3f} | {c['verdict']} |")
    rows += ["", "| state | accel | binary accelerator | sd-cli conv-direct | sd-cli fa | sd-server conv-direct | "
             "sd-server fa | state argv sampling s (rc) |", "|---|---|---|---|---|---|---|---|"]
    for name in ("base", "head"):
        for a, e in sorted(((obs.get(name) or {}).get("studio_argv") or {}).items()):
            runs = e.get("runs") or []
            rows.append(f"| {name} | {a} | {e.get('binary_accelerator')} | {e.get('cli_conv_direct')} | "
                        f"{e.get('cli_fa')} | {e.get('server_conv_direct')} | {e.get('server_fa')} | "
                        f"{[(r.get('sampling_s'), r.get('rc')) for r in runs]} |")
    return "\n".join(rows)


def base_shows_defect(base: dict):
    sh = base.get("shared") or {}
    cells = _cells(sh)
    st = base.get("studio_argv") or {}
    hits = []
    for a in _gpu_accels(sh):
        if not (st.get(a) or {}).get("cli_conv_direct"):
            continue
        mine = [c for (acc, _m), c in cells.items() if acc == a and c["ok"]]
        if any(c["verdict"] == "SLOWER" for c in mine) and not any(c["verdict"] == "FASTER" for c in mine):
            hits.append(a)
    return bool(hits), f"base forces conv-direct and it is measured slower on: {hits or 'none'}"


def head_is_fixed(head: dict):
    sh = head.get("shared") or {}
    st = head.get("studio_argv") or {}
    bad = []
    for a in _gpu_accels(sh):
        e = st.get(a) or {}
        runs = e.get("runs") or []
        ok = (e.get("cli_conv_direct") is False and e.get("cli_fa") is True
              and e.get("server_conv_direct") in (False, None) and e.get("server_fa") in (True, None)
              and runs and all(r.get("rc") == 0 for r in runs))
        if not ok:
            bad.append(a)
    return not bad, f"head still forces conv-direct / failed on: {bad or 'none'}"
