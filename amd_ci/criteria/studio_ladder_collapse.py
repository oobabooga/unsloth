#!/usr/bin/env python3
"""Criteria: does the reported 60 -> 5 fps collapse reproduce on this host?

Judges only. Observations come from probes/studio_ladder_probe.py.

The claim under test is the user's, in their words: Unsloth Studio goes from 60 fps to 5 fps as
a thread grows from 0K to 100K to 500K tokens.

Three things decide it, and the order matters.

1. **Was the workload real?** This is the trap that made every headless-Chromium number
   worthless: there, all four rungs and the null sat at 60.0 fps with a 1-2% busy main thread,
   which is not evidence that the symptom is absent, it is evidence that nothing was loaded. So
   a flat frame rate is only allowed to mean "no collapse" when the rung is shown to have
   loaded the page: the DOM has to grow with the rung, and the busy percentage has to grow with
   it. If it did not, the verdict is VOID and says the venue is unloaded, exactly as a null
   control that never moved would.

2. **Is the difference bigger than the same rung's own spread?** Nothing is under test here,
   so the control is not a base arm, it is the same rung measured again. A rung-to-rung
   difference smaller than the repeat spread is not a difference. With no repeat there is no
   floor at all, and the verdict is INCONCLUSIVE rather than a number.

3. **Only then, the frame rate.** Read from GdkFrameClock::after-paint, one emission per
   presented frame of the toplevel, NOT from requestAnimationFrame. Under a headless X server
   rAF is not vsync locked and reports gaps of 8-9 ms, so it measures main-thread availability
   and would read "120 Hz" on a server with no refresh rate. Both are recorded; only the
   presented series can carry a claim about frames a user sees.
"""

from __future__ import annotations

TITLE = "Does the 60 to 5 fps collapse reproduce in real WebKitGTK on the AMD CI runner?"
MODE = "capability"

NEEDS = [
    "linux", "drm_render_node", "egl_hardware_gl", "webkitgtk", "headless_display_server",
    "gpu_browser_compositing", "studio_production_bundle", "studiobench_ladder",
    "discrete_gpu", "nvidia", "windows", "mlx",
]

# The rung ladder, low to high. "0K" is synthetic (an empty thread); studiobench's own RUNGS
# start at 1K.
ORDER = ["0K", "1K", "10K", "100K", "500K", "1M"]

# A frame rate this far below the reference rung, in percent, is a movement worth naming. It is
# not the bar for the CLAIM: the claim is 60 -> 5, which is a 92% drop. This is only the point
# below which a difference stops being called flat.
MATERIAL_DROP_PCT = 10.0

# What the user reported. Used to say whether a real drop is the reported one or a smaller one.
REPORTED_FLOOR_FPS = 5.0

# The DOM at the top rung has to be this many times the empty rung's before the seeding counts
# as having put anything on the page.
MIN_DOM_GROWTH = 5.0

# And the main thread has to be at least this busy at the top rung, in percent, before a flat
# frame rate is allowed to mean "no collapse" rather than "nothing was loaded". Headless
# Chromium sat at 1-2%.
MIN_TOP_BUSY_PCT = 15.0


def _runs(obs: dict) -> list[dict]:
    return [r for r in (obs.get("runs") or []) if isinstance(r, dict)]


def _ok_runs(obs: dict) -> list[dict]:
    out = []
    for r in _runs(obs):
        p = r.get("payload") or {}
        if p.get("ok") and p.get("marks"):
            out.append(r)
    return out


def _draw_ts(payload: dict) -> list[float]:
    wd = payload.get("widget_draws") or {}
    first, gaps = wd.get("first"), wd.get("gaps_ms") or []
    if first is None:
        return []
    ts = [float(first)]
    for g in gaps:
        ts.append(ts[-1] + float(g) / 1000.0)
    return ts


def _phase_fps(payload: dict) -> dict[str, float | None]:
    """Presented frames per second, per phase, cut on the scene's wall-clock marks."""
    ts = _draw_ts(payload)
    marks = payload.get("marks") or []
    out: dict[str, float | None] = {}
    for i, m in enumerate(marks):
        if i + 1 >= len(marks):
            break
        a, b = m["wall_ms"] / 1000.0, marks[i + 1]["wall_ms"] / 1000.0
        n = sum(1 for t in ts if a <= t <= b)
        out[m["name"]] = (1000.0 * n / ((b - a) * 1000.0)) if b > a else None
    return out


def _busy(payload: dict, phase: str):
    for ph in payload.get("phases") or []:
        if ph.get("phase") == phase:
            return (ph.get("busy") or {}).get("busy_pct")
    return None


def _elements(payload: dict):
    return (payload.get("final") or {}).get("elements")


def _by_rung(obs: dict) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for r in _ok_runs(obs):
        out.setdefault(r.get("rung", "?"), []).append(r)
    return out


def _rungs_sorted(obs: dict) -> list[str]:
    return sorted(_by_rung(obs), key = lambda r: ORDER.index(r) if r in ORDER else 99)


def _agg(rs: list[dict], phase: str) -> dict:
    vals = []
    for r in rs:
        f = _phase_fps(r["payload"]).get(phase)
        if f is not None:
            vals.append(f)
    busy = [b for b in (_busy(r["payload"], phase) for r in rs) if b is not None]
    els = [e for e in (_elements(r["payload"]) for r in rs) if e is not None]
    return {"fps": vals, "n": len(vals),
            "fps_min": min(vals) if vals else None, "fps_max": max(vals) if vals else None,
            "spread": (max(vals) - min(vals)) if len(vals) >= 2 else None,
            "busy": (sum(busy) / len(busy)) if busy else None,
            "elements": (sum(els) / len(els)) if els else None}


def _top_and_base(obs: dict) -> tuple[str | None, str | None]:
    rs = _rungs_sorted(obs)
    return (rs[0] if rs else None), (rs[-1] if rs else None)


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out: list[tuple[str, bool, str]] = []

    x = obs.get("xserver") or {}
    out.append(("a display server was obtained", bool(x.get("display")),
                obs.get("fatal") or f"{x.get('binary')} on {x.get('display')}"))

    d = obs.get("dist") or {}
    inst = obs.get("install") or {}
    out.append(("Studio installed and built a PRODUCTION bundle", bool(d.get("index_html"))
                and (d.get("asset_files") or 0) > 0,
                f"install rc={inst.get('rc')} in {inst.get('seconds')}s; "
                f"dist={d.get('path')} assets={d.get('asset_files')}"))

    runs, ok = _runs(obs), _ok_runs(obs)
    out.append(("every ladder run completed", bool(runs) and len(ok) == len(runs),
                f"{len(ok)}/{len(runs)} ok" + ("" if len(ok) == len(runs) else
                "; " + "; ".join(
                    f"{r.get('rung')}#{r.get('rep')}: "
                    f"{str((r.get('payload') or {}).get('error') or r.get('error') or r.get('rc'))[:120]}"
                    for r in runs if r not in ok))))

    engines = {(r["payload"].get("engine_probe") or {}).get("is_webkit_gtk_ua") for r in ok}
    out.append(("every run really was WebKitGTK", bool(ok) and engines == {True},
                f"is_webkit_gtk_ua={engines}"))

    bundles = {(r["payload"].get("run_meta") or {}).get("bundle_hash") for r in ok}
    out.append(("one bundle across every rung", len(bundles) == 1,
                f"{sorted(b or '?' for b in bundles)}"))

    base, top = _top_and_base(obs)
    grew = False
    detail = "no runs"
    if base and top and base != top:
        b_el = _agg(_by_rung(obs)[base], "idle")["elements"]
        t_el = _agg(_by_rung(obs)[top], "idle")["elements"]
        if b_el and t_el:
            grew = (t_el / b_el) >= MIN_DOM_GROWTH
            detail = f"{base}: {b_el:,.0f} elements -> {top}: {t_el:,.0f} ({t_el / b_el:.1f}x)"
    out.append((f"the seeded thread really mounted (DOM grew >= {MIN_DOM_GROWTH:.0f}x)",
                grew, detail))

    have_repeat = any(len(rs) >= 2 for rs in _by_rung(obs).values())
    out.append(("at least one rung was measured twice, so a difference has a floor",
                have_repeat,
                ", ".join(f"{k}x{len(v)}" for k, v in sorted(_by_rung(obs).items())) or "none"))
    return out


def table(obs: dict) -> str:
    rows = ["| rung | reps | DOM elements | idle fps | scroll fps | stream fps | "
            "idle busy | scroll busy | stream busy |", "|---|---|---|---|---|---|---|---|---|"]
    by = _by_rung(obs)
    for rung in _rungs_sorted(obs):
        rs = by[rung]
        cells = []
        for phase in ("idle", "scroll", "stream"):
            a = _agg(rs, phase)
            cells.append("-" if not a["fps"] else
                         (f"{a['fps_min']:.1f}" if a["n"] == 1 else
                          f"{a['fps_min']:.1f}-{a['fps_max']:.1f}"))
        busies = []
        for phase in ("idle", "scroll", "stream"):
            b = _agg(rs, phase)["busy"]
            busies.append("null" if b is None else f"{b:.0f}%")
        el = _agg(rs, "idle")["elements"]
        rows.append(f"| {rung} | {len(rs)} | {'-' if el is None else f'{el:,.0f}'} | "
                    + " | ".join(cells) + " | " + " | ".join(busies) + " |")

    rows.append("")
    rows.append("Frame rate is the PRESENTED series (GdkFrameClock::after-paint, one emission "
                "per rendered frame of the toplevel), not requestAnimationFrame. Under a "
                "headless X server rAF is not vsync locked and reports 8-9 ms gaps on an idle "
                "page, so it measures main-thread availability and cannot carry a claim about "
                "frames a user sees.")
    rows.append("")
    rows.append("Repeat spread at the same rung, which is the only floor a rung-to-rung "
                "difference in this table has:")
    rows.append("")
    rows.append("| rung | phase | readings | spread |")
    rows.append("|---|---|---|---|")
    for rung in _rungs_sorted(obs):
        for phase in ("idle", "scroll", "stream"):
            a = _agg(_by_rung(obs)[rung], phase)
            if a["spread"] is None:
                continue
            rows.append(f"| {rung} | {phase} | "
                        f"{', '.join(f'{v:.1f}' for v in a['fps'])} | {a['spread']:.1f} fps |")

    # Anything that failed, named rather than dropped.
    bad = [r for r in _runs(obs) if r not in _ok_runs(obs)]
    if bad:
        rows.append("")
        rows.append("Runs that did not complete:")
        rows.append("")
        for r in bad:
            p = r.get("payload") or {}
            rows.append(f"- {r.get('rung')} rep {r.get('rep')}: rc={r.get('rc')} "
                        f"phase={p.get('phase')} {str(p.get('error') or r.get('error'))[:200]}")
    return "\n".join(rows)


def _worst_drop(obs: dict) -> tuple[str | None, str | None, float | None, float | None, float | None]:
    """The largest fall from the lowest rung, over any phase. (phase, rung, base, top, spread)"""
    by = _by_rung(obs)
    order = _rungs_sorted(obs)
    if len(order) < 2:
        return None, None, None, None, None
    base = order[0]
    worst = (None, None, None, None, None)
    worst_pct = 0.0
    for phase in ("idle", "scroll", "stream"):
        b = _agg(by[base], phase)
        if not b["fps"]:
            continue
        for rung in order[1:]:
            t = _agg(by[rung], phase)
            if not t["fps"]:
                continue
            drop_pct = 100.0 * (b["fps_max"] - t["fps_min"]) / b["fps_max"]
            floor = max(x for x in (b["spread"] or 0.0, t["spread"] or 0.0, 0.0))
            if drop_pct > worst_pct:
                worst_pct = drop_pct
                worst = (phase, rung, b["fps_max"], t["fps_min"], floor)
    return worst


def verdict(obs: dict) -> tuple[str, str]:
    by = _by_rung(obs)
    order = _rungs_sorted(obs)
    base, top = order[0], order[-1]
    phase, rung, base_fps, top_fps, floor = _worst_drop(obs)

    top_busy = max((_agg(by[top], p)["busy"] or 0.0) for p in ("idle", "scroll", "stream"))
    base_busy = max((_agg(by[base], p)["busy"] or 0.0) for p in ("idle", "scroll", "stream"))
    top_el = _agg(by[top], "idle")["elements"] or 0
    base_el = _agg(by[base], "idle")["elements"] or 0

    if phase is None:
        return "INCONCLUSIVE", ("no phase produced a presented-frame reading at two different "
                                "rungs, so nothing can be compared")

    drop_pct = 100.0 * (base_fps - top_fps) / base_fps
    real = drop_pct > MATERIAL_DROP_PCT and (base_fps - top_fps) > floor

    if real:
        reached = (f"which is at or below the {REPORTED_FLOOR_FPS:.0f} fps the report names"
                   if top_fps <= REPORTED_FLOOR_FPS else
                   f"which is well above the {REPORTED_FLOOR_FPS:.0f} fps the report names, so "
                   f"this is a smaller effect than the one described")
        return "COLLAPSE_REPRODUCED", (
            f"presented frame rate fell from {base_fps:.1f} fps at {base} to {top_fps:.1f} fps "
            f"at {rung} during the {phase} phase, a {drop_pct:.0f}% drop against a same-rung "
            f"repeat spread of {floor:.1f} fps, {reached}. The main thread was {top_busy:.0f}% "
            f"busy at {top} against {base_busy:.0f}% at {base}, over {top_el:,.0f} DOM elements "
            f"against {base_el:,.0f}")

    # Flat. Now the question is whether that is a fact about the app or about the venue.
    if top_busy < MIN_TOP_BUSY_PCT or (base_el and top_el / base_el < MIN_DOM_GROWTH):
        return "VOID", (
            f"the frame rate is flat ({base_fps:.1f} fps at {base}, {top_fps:.1f} fps at {top}) "
            f"but the venue was not loaded: the main thread reached only {top_busy:.0f}% busy at "
            f"{top} over {top_el:,.0f} DOM elements. A flat reading on an unloaded page is "
            f"evidence of no load, not evidence of no effect, which is exactly what made every "
            f"headless Chromium number VOID")

    return "NO_COLLAPSE", (
        f"the presented frame rate does NOT collapse: {base_fps:.1f} fps at {base} against "
        f"{top_fps:.1f} fps at {top} in the {phase} phase, a {drop_pct:.0f}% difference against "
        f"a same-rung repeat spread of {floor:.1f} fps. And the venue WAS loaded, which is what "
        f"makes that a finding rather than a null: the main thread went from {base_busy:.0f}% "
        f"busy at {base} to {top_busy:.0f}% at {top}, over {top_el:,.0f} DOM elements against "
        f"{base_el:,.0f}. On this host, in the engine Studio actually uses, a thread this size "
        f"loads the main thread heavily and the compositor still presents at the display rate")


def observed_capabilities(obs: dict) -> dict[str, bool]:
    d = obs.get("dist") or {}
    ok = _ok_runs(obs)
    return {
        "webkitgtk": bool(ok),
        "headless_display_server": bool((obs.get("xserver") or {}).get("display")),
        "studio_production_bundle": bool(d.get("index_html")) and (d.get("asset_files") or 0) > 0,
        "studiobench_ladder": len(_by_rung(obs)) >= 2,
    }
