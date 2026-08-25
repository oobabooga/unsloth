#!/usr/bin/env python3
"""Criteria: does PR 9477 piece 2 help at the one rung where the defect exists?

Judges only. Observations come from probes/studio_9477_probe.py.

**The VOID rule is implemented here rather than inherited.** lib/differential.py owns it for
PR-shaped runs driven by lib/states.py, and this is not one: piece 2 is not an open PR, and both
states are built from the same upstream commit with a patch as the only difference. So the rule is
re-stated explicitly and it is not negotiable here either:

    if the BASE arm does not exhibit the defect, the verdict is VOID, not a pass.

A green head arm with no demonstrated base failure shows only that the harness ran. That is the
whole reason this run is at r500K and not at r100K: at r100K on this host the base arm sits at
25-27% busy with a 209 ms worst frame, so it does not exhibit the defect and a differential there
would be VOID by construction. At r500K the base arm collapses to 2.4 fps at 94% busy.

Two non-vacuity gates beyond that:

  * the two arms must have produced DIFFERENT bundle hashes. Two arms that built byte-identical
    frontends are one arm measured twice and would read as a perfect null;
  * the patch must have applied cleanly and completely. A half-applied patch produces an arm that
    is neither base nor head, and it installs and measures perfectly happily.

Every frame rate is quoted with its busy figure, because a frame rate without one cannot tell a
fast page from a blind instrument, which is how a 60.0 fps reading at 94% busy was once published
as a decisive result.
"""

from __future__ import annotations

TITLE = "PR 9477 piece 2 at r500K, in real WebKitGTK on the gfx1151: base vs head"
MODE = "capability"

NEEDS = [
    "linux", "drm_render_node", "egl_hardware_gl", "webkitgtk", "headless_display_server",
    "gpu_browser_compositing", "studio_production_bundle", "studiobench_ladder",
    "discrete_gpu", "nvidia", "windows", "mlx",
]

# The base arm must be at least this busy, and below this frame rate, before it counts as
# exhibiting the defect. Anchored on the measured base: 2.4 fps at 94% busy.
DEFECT_MIN_BUSY_PCT = 60.0
DEFECT_MAX_FPS = 15.0

# A head/base ratio has to clear this AND the arms' own rep spread before it is a real move.
MIN_RATIO = 1.15
PHASES = ("scroll", "stream", "idle")


def _runs(obs: dict) -> list[dict]:
    return [r for r in (obs.get("runs") or []) if (r.get("payload") or {}).get("ok")]


def _arm(obs: dict, arm: str) -> list[dict]:
    return [r for r in _runs(obs) if r.get("arm") == arm]


def _phase(payload: dict, name: str) -> dict:
    for ph in payload.get("phases") or []:
        if ph.get("phase") == name:
            return ph
    return {}


def _fps(payload: dict, name: str):
    ph = _phase(payload, name)
    n = (ph.get("raf") or {}).get("n")
    el = ph.get("elapsed_ms")
    return (1000.0 * n / el) if (n and el) else None


def _busy(payload: dict, name: str):
    return ((_phase(payload, name).get("busy")) or {}).get("busy_pct")


def _worst(payload: dict, name: str):
    return (_phase(payload, name).get("raf") or {}).get("max_ms")


def _vals(obs: dict, arm: str, name: str, fn):
    return [v for v in (fn(r["payload"], name) for r in _arm(obs, arm)) if v is not None]


def _mean(xs):
    return (sum(xs) / len(xs)) if xs else None


def _spread(xs):
    return (max(xs) - min(xs)) if len(xs) >= 2 else None


def _bundles(obs: dict, arm: str) -> set:
    return {(r["payload"].get("run_meta") or {}).get("bundle_hash") for r in _arm(obs, arm)}


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out: list[tuple[str, bool, str]] = []
    st = obs.get("states") or {}
    runs, ok = obs.get("runs") or [], _runs(obs)

    out.append(("a display server was obtained", bool((obs.get("xserver") or {}).get("display")),
                obs.get("fatal") or str((obs.get("xserver") or {}).get("display"))))

    pa = (st.get("head") or {}).get("patch_apply") or {}
    out.append(("the piece 2 patch applied cleanly and completely",
                bool((st.get("head") or {}).get("patch_ok")),
                f"rc={pa.get('rc')}; {str(pa.get('stderr') or pa.get('stdout') or '')[:160]}"))

    both = all((st.get(n) or {}).get("dist", {}).get("index_html") for n in ("base", "head"))
    out.append(("both states installed a PRODUCTION bundle", both,
                "; ".join(f"{n}: install rc={(st.get(n) or {}).get('install', {}).get('rc')} "
                          f"assets={(st.get(n) or {}).get('dist', {}).get('asset_files')}"
                          for n in ("base", "head"))))

    out.append(("every measurement completed", bool(runs) and len(ok) == len(runs),
                f"{len(ok)}/{len(runs)} ok" + ("" if len(ok) == len(runs) else "; " + "; ".join(
                    f"{r.get('arm')}#{r.get('rep')}: "
                    f"{str((r.get('payload') or {}).get('error') or r.get('rc'))[:110]}"
                    for r in runs if r not in ok))))

    bb, hb = _bundles(obs, "base"), _bundles(obs, "head")
    differ = bool(bb) and bool(hb) and not (bb & hb)
    out.append(("the two arms are DIFFERENT builds", differ,
                f"base={sorted(x or '?' for x in bb)} head={sorted(x or '?' for x in hb)}"))

    engines = {(r["payload"].get("engine_probe") or {}).get("is_webkit_gtk_ua") for r in ok}
    out.append(("every measurement really was WebKitGTK", bool(ok) and engines == {True},
                f"is_webkit_gtk_ua={engines}"))

    # THE VOID RULE, as a gate on the base arm.
    bf = _mean(_vals(obs, "base", "scroll", _fps))
    bb_ = _mean(_vals(obs, "base", "scroll", _busy))
    shows = (bf is not None and bb_ is not None
             and bf <= DEFECT_MAX_FPS and bb_ >= DEFECT_MIN_BUSY_PCT)
    out.append((f"the BASE arm exhibits the defect (<= {DEFECT_MAX_FPS:.0f} fps at "
                f">= {DEFECT_MIN_BUSY_PCT:.0f}% busy on scroll)", shows,
                "no base reading" if bf is None else
                f"base scroll {bf:.1f} fps at {bb_:.0f}% busy"))
    return out


def table(obs: dict) -> str:
    rows = ["| phase | arm | fps | busy | worst frame | head/base |", "|---|---|---|---|---|---|"]
    for name in PHASES:
        ratio = None
        b, h = _mean(_vals(obs, "base", name, _fps)), _mean(_vals(obs, "head", name, _fps))
        if b and h:
            ratio = h / b
        for arm in ("base", "head"):
            f = _mean(_vals(obs, arm, name, _fps))
            bu = _mean(_vals(obs, arm, name, _busy))
            w = _vals(obs, arm, name, _worst)
            rows.append("| " + " | ".join([
                name, arm,
                "-" if f is None else f"**{f:.1f}**",
                "-" if bu is None else f"{bu:.0f}%",
                "-" if not w else f"{max(w):,.0f} ms",
                ("-" if arm == "base" or ratio is None else f"**{ratio:.3f}x**"),
            ]) + " |")

    rows += ["", "Per repetition, so a mean cannot hide a disagreement, and so the arms' own "
             "spread is visible as the floor any ratio has to clear:", "",
             "| phase | arm | fps per rep | busy per rep | spread |", "|---|---|---|---|---|"]
    for name in PHASES:
        for arm in ("base", "head"):
            fs = _vals(obs, arm, name, _fps)
            bs = _vals(obs, arm, name, _busy)
            sp = _spread(fs)
            rows.append(f"| {name} | {arm} | "
                        f"{', '.join(f'{x:.1f}' for x in fs) or '-'} | "
                        f"{', '.join(f'{x:.0f}%' for x in bs) or '-'} | "
                        f"{'-' if sp is None else f'{sp:.1f} fps'} |")

    st = obs.get("states") or {}
    rows += ["", f"Both arms are upstream `{(obs.get('merge_base') or '')[:12]}`; head is that "
             f"checkout plus a {obs.get('patch_bytes')}-byte patch "
             f"(`git diff base..62564233c`, 15 files). The only difference between the arms is "
             f"the patch, by construction. Bundle hashes: "
             f"base {sorted(x or '?' for x in _bundles(obs, 'base'))}, "
             f"head {sorted(x or '?' for x in _bundles(obs, 'head'))}.",
             "",
             "Arms were interleaved base, head, base, head, so drift over the job cannot land "
             "entirely on whichever arm ran second."]
    return "\n".join(rows)


def verdict(obs: dict) -> tuple[str, str]:
    b = _mean(_vals(obs, "base", "scroll", _fps))
    h = _mean(_vals(obs, "head", "scroll", _fps))
    bb = _mean(_vals(obs, "base", "scroll", _busy))
    hb = _mean(_vals(obs, "head", "scroll", _busy))
    if b is None or h is None:
        return "INCONCLUSIVE", "one of the arms produced no scroll reading"

    floor = max(_spread(_vals(obs, "base", "scroll", _fps)) or 0.0,
                _spread(_vals(obs, "head", "scroll", _fps)) or 0.0)
    ratio = h / b
    moved = ratio >= MIN_RATIO and (h - b) > floor

    others = []
    for name in ("stream", "idle"):
        ob, oh = _mean(_vals(obs, "base", name, _fps)), _mean(_vals(obs, "head", name, _fps))
        if ob and oh:
            others.append(f"{name} {oh / ob:.3f}x ({ob:.1f} -> {oh:.1f} fps, "
                          f"{_mean(_vals(obs, 'base', name, _busy)):.0f}% -> "
                          f"{_mean(_vals(obs, 'head', name, _busy)):.0f}% busy)")
    tail = "; ".join(others) or "no other phase produced both arms"

    if moved:
        return "HELPS", (
            f"at r500K, the one rung where the base arm exhibits the defect, piece 2 takes the "
            f"scroll from {b:.1f} fps at {bb:.0f}% busy to {h:.1f} fps at {hb:.0f}% busy, "
            f"{ratio:.3f}x, against an arm rep spread of {floor:.1f} fps. Other phases: {tail}")

    return "NO_BENEFIT", (
        f"at r500K, the one rung where the base arm exhibits the defect, piece 2 measures "
        f"{ratio:.3f}x on scroll: {b:.1f} fps at {bb:.0f}% busy against {h:.1f} fps at "
        f"{hb:.0f}% busy, a difference no larger than the arms' own rep spread of {floor:.1f} "
        f"fps. Other phases: {tail}. This is a measured absence of benefit at the rung where the "
        f"defect lives, not an absence of measurement")


def observed_capabilities(obs: dict) -> dict[str, bool]:
    st = obs.get("states") or {}
    return {
        "webkitgtk": bool(_runs(obs)),
        "headless_display_server": bool((obs.get("xserver") or {}).get("display")),
        "studio_production_bundle": all(
            (st.get(n) or {}).get("dist", {}).get("index_html") for n in ("base", "head")),
        "studiobench_ladder": bool(_runs(obs)),
    }
