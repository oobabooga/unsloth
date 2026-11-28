#!/usr/bin/env python3
"""Criteria for probes/c2d_probe.py (FLUX.2 VAE Triton conv2d route on gfx1151): measurement only on the head state
(arms pick the main or the branch tree themselves). Gates prove each listed item ran on gfx1151 and every arm
finished; the numbers and the SHIP / KILL line are read from obs_head.json offline."""
from __future__ import annotations

TITLE = "gfx1151: FLUX.2 / FLUX.3 klein VAE decoder, per-shape Triton conv2d route vs MIOpen (main vs branch)"
MODE = "regression"
NEEDS = ["gpu", "rocm", "exclusive_gpu"]


def gates(obs: dict) -> list:
    h = obs.get("head") or {}
    arch = (h.get("env") or {}).get("arch") or ""
    items = [x for x in ((h.get("args") or {}).get("items") or "").split(",") if x]
    out = [("gfx1151", arch.startswith("gfx1151"), arch), ("items listed", bool(items), ",".join(items))]
    for it in items:
        if h.get(f"{it}_error"):
            out.append((f"{it} no exception", False, str(h[f"{it}_error"])[-300:]))
    if "smoke" in items:
        s = h.get("smoke") or {}
        out.append(("smoke self-check passed on the GPU", bool(s.get("selfcheck")), str(s.get("selfcheck_error"))))
        out.append(("smoke timed every shape", len(s.get("shapes") or []) == 11, str(len(s.get("shapes") or []))))
        wc = s.get("warmup_check") or {}
        out.append(("warmup loads the launch's exact variant", bool(wc.get("same_variant")), str(wc)))
    if "e2e" in items:
        arms = (h.get("e2e") or {}).get("arms") or []
        bad = [a.get("tag") for a in arms if a.get("verdict") != "ok"]
        out.append(("every e2e arm finished", len(arms) == 8 and not bad, f"{len(arms)} arms, failed {bad}"))
        routed = sum(int(((r.get("conv2d_counts_delta") or {}).get("routed") or 0))
                     for a in arms if a.get("arm") == "B" for r in a.get("renders") or [])
        out.append(("branch arms routed calls (engagement)", routed > 0, str(routed)))
    return out


def head_is_worse(base: dict, head: dict) -> tuple:
    return False, "measurement only (head state; main vs branch arms in one job)"


def table(obs: dict) -> str:
    h = obs.get("head") or {}
    return f"- env {h.get('env')}\n- item seconds {h.get('item_s')}\n- numbers: obs_head.json (judged offline)"
