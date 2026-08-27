#!/usr/bin/env python3
"""Criteria: what mechanism burns the frame budget during a 500K scroll?

Judges only. Observations come from probes/studio_ablate_probe.py.

The measured target on this host, corpus 23cd2464: scroll at rung 500K is 2.4 fps at 94% busy
with a 3,092 ms worst frame, while IDLE at the same rung is 61 fps at 9% busy. So the mechanism
is in the scroll path and any hypothesis predicting idle degradation is refuted before it starts.

**Attribution here is by ABLATION, not by trace.** Twice in this campaign a trace nominated a
culprit that ablation cleared: a 153.0 ms traced handler measured ~26 ms when timed directly with
tracing off, and a fix aimed at a candidate owning 21.4% of a traced commit measured completely
flat. Traces nominate; removal convicts.

**Two controls gate every reading, and both must behave before ANY arm is believed.** This is the
same discipline that caught the frame-clock trap, where a channel that could not fall reported
60.0 fps at 94% busy and produced a clean-looking NO_COLLAPSE.

  POSITIVE (`detach_messages`): remove all but two messages. The frame rate MUST recover. If it
  does not, this harness cannot detect a win and no other arm means anything: INCONCLUSIVE.

  NEGATIVE (`noop_touch`): walk the same nodes, add a class that styles nothing, change no
  geometry. The frame rate must NOT recover. If it does, the arms are measuring the act of
  mutating the DOM rather than what they removed, and every recovery is an artefact:
  INCONCLUSIVE.

  DRIFT (`baseline_repeat`): the baseline arm again, after every reversible arm has been applied
  and reverted. If it does not match the first baseline within the same-arm spread, the readings
  are page drift.
"""

from __future__ import annotations

TITLE = "What mechanism burns the frame budget during a 500K scroll?"
MODE = "capability"

NEEDS = [
    "linux", "drm_render_node", "egl_hardware_gl", "webkitgtk", "headless_display_server",
    "gpu_browser_compositing", "studio_production_bundle", "studiobench_ladder",
    "discrete_gpu", "nvidia", "windows", "mlx",
]

BASELINE = "baseline"
BASELINE_REPEAT = "baseline_repeat"
POSITIVE = "detach_messages"
NEGATIVE = "noop_touch"

# The positive control has to restore at least this share of the gap between the collapsed
# baseline and the same page's own idle rate. Anything less and "we can detect a win" is not
# established.
POSITIVE_MIN_RECOVERY = 0.60

# The negative control must stay within this multiple of the baseline. It removes nothing, so
# anything more is the harness reacting to being touched.
NEGATIVE_MAX_RATIO = 1.25

# An arm has to beat the baseline by more than this, AND by more than the baseline's own
# rep-to-rep spread, before it is called a real recovery.
ARM_MIN_RATIO = 1.30


def _runs(obs: dict) -> list[dict]:
    return [r for r in (obs.get("runs") or []) if (r.get("payload") or {}).get("ok")]


def _arms(obs: dict) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for r in _runs(obs):
        for a in r["payload"].get("arms") or []:
            out.setdefault(a.get("name", "?"), []).append(a)
    return out


def _fps(arms: list[dict]) -> list[float]:
    return [a["eff_fps"] for a in arms if a.get("eff_fps") is not None and a.get("ok")]


def _mean(xs):
    return (sum(xs) / len(xs)) if xs else None


def _busy(arms: list[dict]):
    v = [(a.get("busy") or {}).get("busy_pct") for a in arms]
    v = [x for x in v if x is not None]
    return _mean(v)


def _worst(arms: list[dict]):
    v = [(a.get("raf") or {}).get("max_ms") for a in arms]
    v = [x for x in v if x is not None]
    return max(v) if v else None


def _idle_fps(obs: dict):
    return _mean([r["payload"].get("idle", {}).get("eff_fps") for r in _runs(obs)
                  if (r["payload"].get("idle") or {}).get("eff_fps") is not None])


def _spread(xs):
    return (max(xs) - min(xs)) if len(xs) >= 2 else None


def _ratio(obs: dict, name: str):
    a, b = _mean(_fps(_arms(obs).get(name) or [])), _mean(_fps(_arms(obs).get(BASELINE) or []))
    return (a / b) if (a and b) else None


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out: list[tuple[str, bool, str]] = []
    arms = _arms(obs)
    runs = obs.get("runs") or []
    ok = _runs(obs)

    out.append(("a display server was obtained", bool((obs.get("xserver") or {}).get("display")),
                obs.get("fatal") or str((obs.get("xserver") or {}).get("display"))))
    d = obs.get("dist") or {}
    out.append(("Studio installed and built a PRODUCTION bundle",
                bool(d.get("index_html")) and (d.get("asset_files") or 0) > 0,
                f"install rc={(obs.get('install') or {}).get('rc')}; "
                f"assets={d.get('asset_files')}"))
    out.append(("every ablation run completed", bool(runs) and len(ok) == len(runs),
                f"{len(ok)}/{len(runs)} ok" + ("" if len(ok) == len(runs) else "; " + "; ".join(
                    str((r.get('payload') or {}).get('error') or r.get('rc'))[:120]
                    for r in runs if r not in ok))))

    base_fps, idle = _fps(arms.get(BASELINE) or []), _idle_fps(obs)
    collapsed = bool(base_fps) and idle is not None and _mean(base_fps) < idle * 0.5
    out.append(("the baseline arm still exhibits the collapse", collapsed,
                f"baseline {_mean(base_fps):.1f} fps at {_busy(arms.get(BASELINE) or []):.0f}% busy "
                f"against idle {idle:.1f} fps" if base_fps and idle is not None
                else "no baseline reading"))

    # POSITIVE CONTROL: the harness must be able to show a win.
    pos = _mean(_fps(arms.get(POSITIVE) or []))
    b = _mean(base_fps)
    recovered = None
    if pos is not None and b is not None and idle is not None and idle > b:
        recovered = (pos - b) / (idle - b)
    out.append((f"POSITIVE control recovers >= {POSITIVE_MIN_RECOVERY:.0%} of the gap to idle",
                recovered is not None and recovered >= POSITIVE_MIN_RECOVERY,
                "no positive control arm" if pos is None else
                f"detach_messages {pos:.1f} fps against baseline {b:.1f} and idle {idle:.1f} "
                f"= {recovered:.0%} of the gap"))

    # NEGATIVE CONTROL: touching without removing must not help.
    neg_ratio = _ratio(obs, NEGATIVE)
    out.append((f"NEGATIVE control does not recover (<= {NEGATIVE_MAX_RATIO:.2f}x baseline)",
                neg_ratio is not None and neg_ratio <= NEGATIVE_MAX_RATIO,
                "no negative control arm" if neg_ratio is None else
                f"noop_touch {neg_ratio:.2f}x baseline"))

    # DRIFT.
    rep_ratio = _ratio(obs, BASELINE_REPEAT)
    sp = _spread(base_fps) or 0.0
    drift_ok = rep_ratio is not None and abs(rep_ratio - 1.0) * (_mean(base_fps) or 0) <= max(sp, 2.0)
    out.append(("the page did not drift (baseline_repeat matches baseline)", bool(drift_ok),
                "no repeat arm" if rep_ratio is None else
                f"baseline_repeat {rep_ratio:.2f}x baseline; baseline rep spread {sp:.1f} fps"))

    out.append(("at least two repetitions, so an arm has a floor", len(ok) >= 2,
                f"{len(ok)} runs"))
    return out


def table(obs: dict) -> str:
    arms = _arms(obs)
    base = _mean(_fps(arms.get(BASELINE) or []))
    idle = _idle_fps(obs)
    order = [a.get("name") for r in _runs(obs) for a in (r["payload"].get("arms") or [])]
    seen, ordered = set(), []
    for n in order:
        if n not in seen:
            seen.add(n)
            ordered.append(n)

    rows = ["| arm | fps | vs baseline | busy | worst frame | elements after | highlight spans "
            "after | what it removed |", "|---|---|---|---|---|---|---|---|"]
    for n in ordered:
        a = arms.get(n) or []
        f, bu, w = _mean(_fps(a)), _busy(a), _worst(a)
        ratio = (f / base) if (f and base) else None
        c = (a[0].get("census_applied") or {}) if a else {}
        tag = ""
        if n == POSITIVE:
            tag = " **[POSITIVE CONTROL]**"
        elif n == NEGATIVE:
            tag = " **[NEGATIVE CONTROL]**"
        elif n == BASELINE_REPEAT:
            tag = " **[DRIFT CHECK]**"
        rows.append("| " + " | ".join([
            f"`{n}`{tag}",
            "-" if f is None else f"**{f:.1f}**",
            "-" if ratio is None else f"{ratio:.2f}x",
            "-" if bu is None else f"{bu:.0f}%",
            "-" if w is None else f"{w:,.0f} ms",
            f"{c.get('elements'):,}" if c.get("elements") else "-",
            f"{c.get('highlight_spans'):,}" if c.get("highlight_spans") is not None else "-",
            str((a[0].get("apply_detail") if a else "") or "")[:70],
        ]) + " |")

    rows += ["",
             f"Idle at this rung, same page, same session: **{idle:.1f} fps**. That is the "
             f"ceiling an ablation can restore to, and the reason the standing DOM is not itself "
             f"the mechanism." if idle is not None else "",
             "",
             "fps is the EFFECTIVE rate: rAF callbacks delivered over the arm's wall time. Never "
             "1000/p50, which reported 16.0 ms both jammed and unjammed when a bursty block was "
             "applied deliberately. Every fps here is quoted with its busy figure because a frame "
             "rate without one cannot distinguish a fast page from a blind instrument."]

    # Per-rep detail, so a single-rep fluke cannot hide inside a mean.
    rows += ["", "Per repetition, so a mean cannot hide a disagreement:", "",
             "| arm | fps per rep | busy per rep |", "|---|---|---|"]
    for n in ordered:
        a = arms.get(n) or []
        fs = ", ".join(f"{x:.1f}" for x in _fps(a)) or "-"
        bs = ", ".join(str((x.get("busy") or {}).get("busy_pct")) for x in a) or "-"
        rows.append(f"| `{n}` | {fs} | {bs} |")
    return "\n".join(rows)


def verdict(obs: dict) -> tuple[str, str]:
    arms = _arms(obs)
    base = _mean(_fps(arms.get(BASELINE) or []))
    idle = _idle_fps(obs)
    base_spread = _spread(_fps(arms.get(BASELINE) or [])) or 0.0

    candidates = []
    for name, a in arms.items():
        if name in (BASELINE, BASELINE_REPEAT, POSITIVE, NEGATIVE):
            continue
        f = _mean(_fps(a))
        if f is None or not base:
            continue
        if f / base >= ARM_MIN_RATIO and (f - base) > base_spread:
            candidates.append((f / base, name, f, _busy(a), _worst(a)))
    candidates.sort(reverse = True)

    if not candidates:
        near = sorted(((_mean(_fps(a)) or 0) / base, n) for n, a in arms.items()
                      if n not in (BASELINE, BASELINE_REPEAT, POSITIVE, NEGATIVE) and base)
        best = f"{near[-1][1]} at {near[-1][0]:.2f}x" if near else "none"
        return "NOT_ATTRIBUTED", (
            f"no candidate ablation recovered the frame rate. The baseline collapses to "
            f"{base:.1f} fps against {idle:.1f} fps idle, and the positive control shows the "
            f"harness can see a recovery, so the cost is real and is NOT any of the mechanisms "
            f"removed here (best was {best}). The mechanism is elsewhere and this run narrows it "
            f"rather than naming it")

    ratio, name, f, bu, w = candidates[0]
    others = "; ".join(f"{n} {r:.2f}x" for r, n, *_ in candidates[1:]) or "no other arm qualified"
    return "ATTRIBUTED", (
        f"removing `{name}` takes the 500K scroll from {base:.1f} fps to {f:.1f} fps "
        f"({ratio:.2f}x) at {bu:.0f}% busy with a {w:,.0f} ms worst frame, against a baseline "
        f"rep spread of {base_spread:.1f} fps and an idle ceiling of {idle:.1f} fps. The negative "
        f"control did not recover and the positive control did, so this is the removed work and "
        f"not the act of removing it. Other qualifying arms: {others}")


def observed_capabilities(obs: dict) -> dict[str, bool]:
    d = obs.get("dist") or {}
    return {
        "webkitgtk": bool(_runs(obs)),
        "headless_display_server": bool((obs.get("xserver") or {}).get("display")),
        "studio_production_bundle": bool(d.get("index_html")) and (d.get("asset_files") or 0) > 0,
        "studiobench_ladder": bool(_runs(obs)),
    }
