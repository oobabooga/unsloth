#!/usr/bin/env python3
"""Criteria: does the RenderLayer attribution for the 500K scroll transfer to a REAL GPU?

Judges only. Observations come from probes/studio_layers_probe.py.

THE CLAIM UNDER TEST. On the local llvmpipe WebKitGTK rig the 500K scroll cost is attributed to
`RenderLayer::recursiveUpdateLayerPositionsAfterScroll`, which recurses over every child
unconditionally with exactly one early-out (`!m_hasVisibleDescendant && !m_hasVisibleContent`).
`RenderBoxModelObject::requiresLayer()` is true for `isPositioned()`, and `isPositioned()` is
`position != static`, so `position: relative` alone buys a RenderLayer. Locally
`[data-role] *{position:static}` removes 78% of the cost and `.katex *{position:static}` 79%.

On a real GPU `usesCompositedScrolling()` is true, which adds `setNeedsCompositingGeometryUpdate()`,
`setDescendantsNeedUpdateBackingAndHierarchyTraversal()` and `updateCompositingLayersAfterScroll()`
-- a SECOND full descendant walk per scroll event that llvmpipe never takes. The prediction is
therefore not merely that the attribution transfers, but that the SAME ablations remove MORE here.

THE PRIMARY METRIC IS BLOCKED MILLISECONDS PER FRAME, not busy%. The gesture issues exactly one
1 px scroll per painted frame, so blocked-ms-per-frame is the cost of ONE SCROLL EVENT. `busy%` is
a share of WALL time: when frames get cheaper more of them fit into the same second, so busy% stays
high while the work per event falls. In this campaign that arithmetic already hid a 62% work
reduction as a 19% change.

EVERY ARM IS SCORED AGAINST ITS OWN TWO NEIGHBOURING BASELINES, because a baseline runs between
every arm. A before/after pair cannot separate an arm from page drift, and the local replication of
the earlier ablation failed on precisely that: baseline 17.5 fps against baseline_repeat 30.1 fps,
a first-visit cost that made every ratio in that run meaningless.

THE DRIFT GATE HAS ALREADY FIRED ONCE HERE, on run 32865232787, and its threshold is unchanged
since. That run read baseline 261.4 ms blocked per scroll event against a 162.6 / 193.9 / 197.5 /
184.5 cluster, and the DOM grew by 11,205 elements across the session while `.katex` (1,027),
`.katex *` (101,306) and `pre` (330) all stayed pinned. The cause was the harness: the park
position was recomputed as `scrollHeight * 0.5` per window, so an arm that made the thread 7.3%
taller moved the window onto content that had never been visited, and mounting it was billed to
whichever arm moved the scroller there. The fix is a FIXED park pixel plus a DISCARDED warm-up run
to quiescence, both gated below. Widening this threshold instead would have switched off the one
gate that was telling the truth.

TWO ARMS ARE EXPECTED TO BE DISQUALIFIED ON scrollHeight, and that is a property of the ablation
rather than a fault in the run: KaTeX uses relative positioning for vertical alignment, so forcing
its descendants to static necessarily reflows the thread (measured: +7.3% for `[data-role] *`,
+3.8% for `.katex *`). Their numbers are printed, clearly marked, and cannot carry the headline.
`.katex{visibility:hidden}` changes no layout at all, which is why it is the arm that can.

GATES THAT ARE ABOUT THE INSTRUMENT FAIL THE RUN. Conditions that are about ONE ARM disqualify that
arm and are printed, because a single bad arm should not throw away the others:
  * an arm whose declaration was DROPPED by the engine (checked by re-reading getComputedStyle on
    a sample of the very nodes the selector names). Three `overflow-anchor` arms in this campaign
    were vacuous exactly this way and their clean nulls proved nothing;
  * an arm that changed scrollHeight, which pins the gesture. One such arm reported 111 fps at
    184% busy and meant nothing at all.
"""

from __future__ import annotations

TITLE = ("Does the ~22,000-RenderLayer attribution for the 500K scroll transfer to a real GPU, "
         "and is it larger there?")
MODE = "capability"

NEEDS = [
    "linux", "drm_render_node", "egl_hardware_gl", "webkitgtk", "headless_display_server",
    "gpu_browser_compositing", "studio_production_bundle", "studiobench_ladder",
    "discrete_gpu", "nvidia", "windows", "mlx",
]

BASELINE_ARM = "baseline"
BASELINE_REPEAT = "baseline_repeat"
FIRST_BASELINE = "baseline"
POSITIVE = "detach_messages"
NEGATIVE = "noop_touch"
FLOOR = "still_no_scroll"
CANDIDATES = ["position_static_all", "katex_static", "visibility_hidden_offscreen"]

# The jam must be visible at idle. 0b prices this at 61.2 -> 17.2 fps on the correct channel and a
# pinned 60.0 on the broken one, so 25% is a floor and not a target.
LIVENESS_MIN_DROP = 0.25
# baseline_repeat against the FIRST baseline, on the primary metric.
DRIFT_MAX = 0.25
# The negative control removes nothing, so any apparent saving is the harness reacting to being
# touched.
NEGATIVE_MAX_SAVING = 0.15
# The positive control must recover most of the way to the measured floor.
POSITIVE_MIN_RECOVERY = 0.60
# Elements the page may still mount during a SCORED window before that window is measuring
# somebody else's deferred work rather than the gesture.
MAX_SCORED_WINDOW_MUTATIONS = 200
# An arm must beat its neighbouring baselines by more than this on blocked-ms-per-frame.
ARM_MIN_SAVING = 0.20
# scrollHeight change that pins the gesture.
MAX_SCROLL_HEIGHT_DELTA = 0.02


# ── payload access ────────────────────────────────────────────────────────────────────────────

def _runs(obs: dict) -> list[dict]:
    return [r for r in (obs.get("runs") or []) if (r.get("payload") or {}).get("ok")]


def _all_runs(obs: dict) -> list[dict]:
    return list(obs.get("runs") or [])


def _windows(obs: dict) -> list[tuple[int, list[dict]]]:
    """Per repetition, the arm windows IN ORDER. Order is what makes neighbours meaningful."""
    return [(r.get("rep"), list(r["payload"].get("arms") or [])) for r in _runs(obs)]


def _bmpf(a: dict):
    return a.get("blocked_ms_per_frame")


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return (sum(xs) / len(xs)) if xs else None


def _neighbour_baseline(seq: list[dict], i: int, key):
    """Mean of the nearest baseline window before and after index i, on `key`."""
    vals = []
    for j in range(i - 1, -1, -1):
        if seq[j].get("arm") == BASELINE_ARM and key(seq[j]) is not None:
            vals.append(key(seq[j]))
            break
    for j in range(i + 1, len(seq)):
        if seq[j].get("arm") == BASELINE_ARM and key(seq[j]) is not None:
            vals.append(key(seq[j]))
            break
    return _mean(vals)


def _scored(obs: dict) -> dict[str, list[dict]]:
    """For every non-baseline window, its own local comparison. One entry per repetition."""
    out: dict[str, list[dict]] = {}
    for rep, seq in _windows(obs):
        for i, a in enumerate(seq):
            if a.get("arm") == BASELINE_ARM and a.get("name") != BASELINE_REPEAT:
                continue
            base_b = _neighbour_baseline(seq, i, _bmpf)
            base_f = _neighbour_baseline(seq, i, lambda x: x.get("eff_fps"))
            b, f = _bmpf(a), a.get("eff_fps")
            entry = {
                "rep": rep, "name": a.get("name"), "arm": a.get("arm"), "ok": a.get("ok"),
                "bmpf": b, "fps": f, "busy": (a.get("busy") or {}).get("busy_pct"),
                "frames": a.get("frames"), "worst_ms": (a.get("raf") or {}).get("max_ms"),
                "base_bmpf": base_b, "base_fps": base_f,
                # SAVING on the primary metric: the share of the per-scroll-event cost removed.
                "saving": (1 - b / base_b) if (b is not None and base_b) else None,
                "work_ratio": (base_b / b) if (b and base_b) else None,
                "fps_ratio": (f / base_f) if (f and base_f) else None,
                "fired": a.get("fired"),
                "scroll_height_delta": a.get("scroll_height_delta"),
                "mutations": a.get("mutations") or {},
                "gesture": a.get("gesture") or {},
            }
            out.setdefault(a.get("name"), []).append(entry)
    return out


def _disqualified(name: str, entries: list[dict]) -> str | None:
    """Reasons this ARM cannot be scored. Not run-level: the other arms are unaffected."""
    if name in (POSITIVE, NEGATIVE, FLOOR, BASELINE_REPEAT):
        return None
    for e in entries:
        fired = e.get("fired")
        if isinstance(fired, dict) and fired.get("fired") is False:
            return (f"the declaration did not take: only {fired.get('matching')}/"
                    f"{fired.get('sampled')} sampled `{fired.get('selector')}` nodes report "
                    f"{fired.get('prop')}={fired.get('want')} "
                    f"(saw {fired.get('non_matching_examples')}). An arm whose CSS the engine "
                    f"dropped is vacuous and its null proves nothing")
        d = e.get("scroll_height_delta")
        if d is not None and abs(d) > MAX_SCROLL_HEIGHT_DELTA:
            return (f"scrollHeight moved {d:+.1%}, so this arm changed the thing the gesture "
                    f"scrolls and its window is not the same window as the baseline's")
        if not e.get("ok"):
            return "the arm did not complete"
    return None


def _liveness(obs: dict) -> list[dict]:
    return [r["payload"].get("liveness") or {} for r in _runs(obs)]


# ── gates ─────────────────────────────────────────────────────────────────────────────────────

def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out: list[tuple[str, bool, str]] = []
    runs, ok = _all_runs(obs), _runs(obs)
    scored = _scored(obs)

    out.append(("a display server was obtained", bool((obs.get("xserver") or {}).get("display")),
                obs.get("fatal") or str((obs.get("xserver") or {}).get("display"))))
    d = obs.get("dist") or {}
    out.append(("Studio installed and built a PRODUCTION bundle",
                bool(d.get("index_html")) and (d.get("asset_files") or 0) > 0,
                f"install rc={(obs.get('install') or {}).get('rc')}; "
                f"assets={d.get('asset_files')}"))
    out.append(("every repetition completed", bool(runs) and len(ok) == len(runs),
                f"{len(ok)}/{len(runs)} ok" + ("" if len(ok) == len(runs) else "; " + "; ".join(
                    str((r.get('payload') or {}).get('error') or r.get('rc'))[:160]
                    for r in runs if r not in ok))))
    out.append(("at least two repetitions, so a per-rep disagreement is visible", len(ok) >= 2,
                f"{len(ok)} runs"))

    # The channel must be able to report a blocked main thread AT ALL. Priced at idle, where it
    # has room to fall; pricing a control where the page is already saturated is how a control
    # with ample power gets thrown away.
    liv = _liveness(obs)
    drops = [l.get("drop_fraction") for l in liv if l.get("drop_fraction") is not None]
    out.append((f"LIVENESS: a deliberate main-thread jam drops idle fps by >= "
                f"{LIVENESS_MIN_DROP:.0%}",
                bool(drops) and min(drops) >= LIVENESS_MIN_DROP,
                "; ".join(f"rep: {l.get('clean_fps')} -> {l.get('jammed_fps')} fps "
                          f"({(l.get('drop_fraction') or 0):.0%} fall), blocked/frame "
                          f"{l.get('clean_blocked_ms_per_frame')} -> "
                          f"{l.get('jammed_blocked_ms_per_frame')} ms" for l in liv)
                or "no liveness window"))

    clamps = [(r["payload"].get("clamp") or {}) for r in ok]
    out.append(("the busy clamp calibrated, so blocked-ms is defined",
                bool(clamps) and all(c.get("clamp_ms") is not None for c in clamps),
                "; ".join(str(c.get("clamp_ms") or c.get("reason")) for c in clamps)
                or "no clamp"))

    # `scroll-behavior: smooth` on the viewport turns an assignment into an ANIMATION, so a
    # same-turn read-back reports a real gesture as inert.
    guards = [(r["payload"].get("guard") or {}) for r in ok]
    out.append(("scroll-behavior was forced to auto on the scroller",
                bool(guards) and all(g.get("ok") for g in guards),
                "; ".join(f"{g.get('scroll_behavior_before')} -> {g.get('scroll_behavior_after')} "
                          f"on {g.get('scroller_tag')}.{(g.get('scroller_class') or '')[:40]}"
                          for g in guards) or "no guard record"))

    # The gesture must have ACHIEVED what it commanded. A probe that reports what it asked for
    # and samples only the endpoints will agree with itself perfectly while measuring nothing.
    tf, sb = [], []
    for _, seq in _windows(obs):
        for a in seq:
            g = a.get("gesture") or {}
            if a.get("arm") == BASELINE_ARM and g.get("travel_fraction") is not None:
                tf.append(g["travel_fraction"])
                sb.append(g.get("snapback_frames") or 0)
    out.append(("the 1 px gesture actually moved the scroller on every baseline window",
                bool(tf) and min(tf) >= 0.9,
                f"travel fraction min {min(tf):.2f} over {len(tf)} baseline windows, "
                f"snapback frames max {max(sb)}" if tf else "no baseline gesture recorded"))

    # DRIFT. Reported first in the summary because a drift-corrected number is the only honest
    # one, and if this fails the correct output is "it drifted", not a ratio against a
    # contaminated baseline.
    drift_txt, drift_ok = [], True
    for rep, seq in _windows(obs):
        first = next((x for x in seq if x.get("name") == FIRST_BASELINE), None)
        rep_w = next((x for x in seq if x.get("name") == BASELINE_REPEAT), None)
        if not first or not rep_w or not _bmpf(first) or _bmpf(rep_w) is None:
            drift_ok = False
            drift_txt.append(f"rep {rep}: missing a baseline window")
            continue
        rel = _bmpf(rep_w) / _bmpf(first) - 1
        allb = [_bmpf(x) for x in seq if x.get("arm") == BASELINE_ARM and _bmpf(x) is not None]
        drift_txt.append(f"rep {rep}: baseline {_bmpf(first)} -> baseline_repeat "
                         f"{_bmpf(rep_w)} ms blocked/frame ({rel:+.0%}); all baselines "
                         f"{[round(x, 1) for x in allb]}")
        if abs(rel) > DRIFT_MAX:
            drift_ok = False
    out.append((f"DRIFT: baseline_repeat within {DRIFT_MAX:.0%} of the first baseline",
                drift_ok, "; ".join(drift_txt) or "no baseline windows"))

    # THE DEFERRED-FENCE RESERVOIR. `code-fence-defer.tsx` ships `SHIP_DEFAULT = "defer"`, and
    # `useFenceReached` latches on an IntersectionObserver with a 100% root margin -- which
    # re-delivers on LAYOUT CHANGE, no scroll needed. So any arm that changes the thread's height
    # mounts markup, and that mounting is permanent and contaminates every window after it. The
    # warm-up drains it through the app's own `beforeprint` -> `upgradeEverythingForPrint` path.
    # If anything is left deferred, an arm can still trip it.
    latches = [(r["payload"].get("fence_latch") or {}) for r in ok]
    left = [((l.get("after") or {}).get("fences_deferred")) for l in latches]
    out.append(("the deferred-fence reservoir was drained before scoring",
                bool(latches) and all(x == 0 for x in left),
                "; ".join(
                    f"rep: {(l.get('before') or {}).get('fences_deferred')} deferred -> "
                    f"{(l.get('after') or {}).get('fences_deferred')} left, "
                    f"{l.get('fences_latched')} latched, +{l.get('elements_added')} elements "
                    f"(+{l.get('highlight_spans_added')} `pre span`), dispatched="
                    f"{l.get('dispatched')} {l.get('error') or ''}" for l in latches)
                or "no fence-latch record"))

    # THE WARM-UP. A first-visit cost paid inside the first scored window is exactly what failed
    # the previous run, so the window that absorbs it is discarded and its convergence is gated.
    warm = [r["payload"].get("warmup") or {} for r in ok]
    out.append(("the discarded warm-up reached quiescence before any scoring began",
                bool(warm) and all(w.get("quiesced") for w in warm),
                "; ".join(
                    f"rep: {len(w.get('rounds') or [])} rounds, quiesced={w.get('quiesced')}, "
                    f"absorbed {(w.get('mut_total') or {}).get('added_elements')} added elements "
                    f"in {w.get('elapsed_ms')} ms, per-round element deltas "
                    f"{[r_.get('element_delta') for r_ in (w.get('rounds') or [])]}"
                    for w in warm) or "no warm-up window"))

    # And it must STAY quiescent, or a scored window is measuring deferred mounting.
    noisy = []
    for rep_i, seq in _windows(obs):
        for a in seq:
            if a.get("arm") == POSITIVE:
                continue  # detaching 28 messages is supposed to mutate the DOM
            m = (a.get("mutations") or {}).get("added_elements")
            if m is not None and m > MAX_SCORED_WINDOW_MUTATIONS:
                noisy.append(f"rep {rep_i} {a.get('name')}: +{m} elements "
                             f"{(a.get('mutations') or {}).get('buckets')}")
    out.append((f"no scored window mounted more than {MAX_SCORED_WINDOW_MUTATIONS} elements",
                not noisy, "; ".join(noisy) if noisy else "every scored window was quiescent"))

    # Every window must have looked at the SAME content, which is what a fixed park pixel buys.
    parks = {r["payload"].get("park") for r in ok}
    park_ok = True
    for _, seq in _windows(obs):
        for a in seq:
            g = a.get("gesture") or {}
            if g and g.get("park") is not None and g["park"] not in parks:
                park_ok = False
    out.append(("every window parked at the same fixed pixel", park_ok and bool(parks),
                f"park positions {sorted(x for x in parks if x is not None)}"))

    # The defect must still be present, or there is nothing to ablate. Priced against the FLOOR
    # measured on this very page (`still`), not against an assumed 60 fps.
    floor = _mean([e["bmpf"] for e in scored.get(FLOOR, []) if e.get("bmpf") is not None])
    base_all = _mean([_bmpf(x) for _, seq in _windows(obs) for x in seq
                      if x.get("arm") == BASELINE_ARM])
    out.append(("the baseline still exhibits the collapse (a 1 px scroll costs far more than "
                "assigning the same scrollTop)",
                bool(floor is not None and base_all and base_all > max(2.0, floor * 3)),
                f"baseline {base_all:.1f} ms blocked per scroll event against a `still` floor of "
                f"{floor:.1f} ms" if (base_all and floor is not None)
                else f"baseline={base_all} floor={floor}"))

    neg = _mean([e["saving"] for e in scored.get(NEGATIVE, []) if e.get("saving") is not None])
    out.append((f"NEGATIVE control removes <= {NEGATIVE_MAX_SAVING:.0%} of the cost",
                neg is not None and neg <= NEGATIVE_MAX_SAVING,
                f"noop_touch {neg:+.0%}" if neg is not None else "no negative control window"))

    pos = _mean([e["bmpf"] for e in scored.get(POSITIVE, []) if e.get("bmpf") is not None])
    rec = None
    if pos is not None and base_all and floor is not None and base_all > floor:
        rec = (base_all - pos) / (base_all - floor)
    out.append((f"POSITIVE control recovers >= {POSITIVE_MIN_RECOVERY:.0%} of the way to the floor",
                rec is not None and rec >= POSITIVE_MIN_RECOVERY,
                f"detach_messages {pos:.1f} ms blocked/frame against baseline {base_all:.1f} and "
                f"floor {floor:.1f} = {rec:.0%}" if rec is not None
                else "no positive control window"))
    return out


# ── report ────────────────────────────────────────────────────────────────────────────────────

def table(obs: dict) -> str:
    scored = _scored(obs)
    order = []
    for _, seq in _windows(obs):
        for a in seq:
            if a.get("name") not in order:
                order.append(a.get("name"))

    rows = ["The primary metric is **blocked ms per frame**. The gesture issues exactly one 1 px "
            "scroll per painted frame, so this is the cost of ONE SCROLL EVENT. `busy%` is a "
            "share of wall time and is shown only for continuity with earlier tables.", "",
            "| window | blocked ms/frame | vs its two neighbouring baselines | cost removed | "
            "fps | busy | worst frame | scrollHeight | elements mounted | declaration took |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for name in order:
        es = scored.get(name)
        if es is None:
            # a baseline window: show it as its own row, unscored
            vals = [_bmpf(a) for _, seq in _windows(obs) for a in seq if a.get("name") == name]
            fps = [a.get("eff_fps") for _, seq in _windows(obs) for a in seq
                   if a.get("name") == name]
            mut = [(a.get("mutations") or {}).get("added_elements") or 0
                   for _, seq in _windows(obs) for a in seq if a.get("name") == name]
            rows.append(f"| `{name}` (baseline) | "
                        + ", ".join(f"{v:.1f}" for v in vals if v is not None)
                        + " | - | - | " + ", ".join(f"{v:.1f}" for v in fps if v is not None)
                        + f" | - | - | - | {max(mut) if mut else 0} | - |")
            continue
        tag = {POSITIVE: " **[POSITIVE CONTROL]**", NEGATIVE: " **[NEGATIVE CONTROL]**",
               FLOOR: " **[FLOOR]**", BASELINE_REPEAT: " **[DRIFT CHECK]**"}.get(name, "")
        b = _mean([e["bmpf"] for e in es])
        wr = _mean([e["work_ratio"] for e in es])
        sv = _mean([e["saving"] for e in es])
        f = _mean([e["fps"] for e in es])
        bu = _mean([e["busy"] for e in es])
        wo = max([e["worst_ms"] for e in es if e.get("worst_ms") is not None], default = None)
        dq = _disqualified(name, es)
        sh = _mean([e["scroll_height_delta"] for e in es])
        fired = next((e.get("fired") for e in es if isinstance(e.get("fired"), dict)), None)
        rows.append("| " + " | ".join([
            f"`{name}`{tag}" + ("  **[DISQUALIFIED]**" if dq else ""),
            "-" if b is None else f"**{b:.1f}**",
            "-" if wr is None else f"{wr:.2f}x less work",
            "-" if sv is None else f"{sv:+.0%}",
            "-" if f is None else f"{f:.1f}",
            "-" if bu is None else f"{bu:.0f}%",
            "-" if wo is None else f"{wo:,.0f} ms",
            "-" if sh is None else f"{sh:+.2%}",
            str(max((e["mutations"].get("added_elements") or 0) for e in es)),
            "-" if not fired else ("yes" if fired.get("fired") else
                                   f"NO ({fired.get('matching')}/{fired.get('sampled')})"),
        ]) + " |")

    dqs = [(n, _disqualified(n, es)) for n, es in scored.items() if _disqualified(n, es)]
    if dqs:
        rows += ["", "Disqualified arms, and why (the other arms are unaffected):", ""]
        rows += [f"- `{n}`: {why}" for n, why in dqs]

    # Per repetition, because a mean over a bimodal draw is a report of how many slow draws
    # happened to land in that arm.
    rows += ["", "Per repetition, so a mean cannot hide a disagreement:", "",
             "| window | blocked ms/frame per rep | fps per rep | frames per rep |",
             "|---|---|---|---|"]
    for name in order:
        vals, fps, fr = [], [], []
        for _, seq in _windows(obs):
            for a in seq:
                if a.get("name") == name:
                    vals.append(_bmpf(a))
                    fps.append(a.get("eff_fps"))
                    fr.append(a.get("frames"))
        rows.append(f"| `{name}` | " + ", ".join(str(v) for v in vals) + " | "
                    + ", ".join(str(v) for v in fps) + " | "
                    + ", ".join(str(v) for v in fr) + " |")

    # THE DISCARDED WARM-UP, reported. The whole point of it is the first-visit work it absorbs,
    # so throwing the window away without saying what was in it would repeat the mistake at one
    # remove.
    warm = [r["payload"].get("warmup") or {} for r in _runs(obs)]
    if any(warm):
        rows += ["", "The discarded warm-up, which is where the first-visit cost is now paid:", "",
                 "| rep | rounds | quiesced | elements mounted | ms | per-round element delta | "
                 "where they landed |", "|---|---|---|---|---|---|---|"]
        for i, w in enumerate(warm, 1):
            mt = w.get("mut_total") or {}
            rows.append(f"| {i} | {len(w.get('rounds') or [])} | "
                        f"{'yes' if w.get('quiesced') else 'NO'} | "
                        f"{mt.get('added_elements')} | {w.get('elapsed_ms')} | "
                        f"{[r_.get('element_delta') for r_ in (w.get('rounds') or [])]} | "
                        f"{mt.get('buckets')} |")
        lat = warm[0].get("fence_latch") or {}
        lb, la = lat.get("before") or {}, lat.get("after") or {}
        if lb and la:
            rows += ["", f"Draining the deferred-fence reservoir through the app's own "
                         f"`beforeprint` path latched **{lat.get('fences_latched')}** fences and "
                         f"added **{lat.get('elements_added'):,}** elements, of which "
                         f"{lat.get('highlight_spans_added'):,} are `pre span`, while `pre` "
                         f"stayed at {la.get('code_blocks')} and `.katex` at "
                         f"{la.get('katex_roots'):,}. That is the 11,205-element growth run "
                         f"32865232787 could not name, and it is now paid before anything is "
                         f"scored rather than by whichever arm happened to change the layout "
                         f"height. **The caveat this buys, stated as a number:** the page is now "
                         f"{la.get('elements'):,} elements against the {lb.get('elements'):,} the "
                         f"local llvmpipe rig measured, so the local 78% / 79% / 86% figures are "
                         f"for a page with {lb.get('highlight_spans'):,} `pre span` and this one "
                         f"has {la.get('highlight_spans'):,}. None of those spans is positioned, "
                         f"so the RenderLayer population under test is unchanged: `.katex` roots "
                         f"and descendants are identical either way."]
        cb = warm[0].get("census_before") or {}
        ca = warm[0].get("census_after") or {}
        if cb and ca:
            rows += ["", f"Across the warm-up the page went from {cb.get('elements'):,} to "
                         f"{ca.get('elements'):,} elements, `pre span` "
                         f"{cb.get('highlight_spans')} -> {ca.get('highlight_spans')}, "
                         f"`span` {cb.get('all_spans')} -> {ca.get('all_spans')}, `button` "
                         f"{cb.get('buttons')} -> {ca.get('buttons')}, `[data-slot]` "
                         f"{cb.get('data_slots')} -> {ca.get('data_slots')}, while `.katex` "
                         f"stayed at {ca.get('katex_roots')} and `pre` at "
                         f"{ca.get('code_blocks')}."]

    # What is in the page, because the whole attribution is content-dependent.
    cen = next((r["payload"].get("baseline_census") for r in _runs(obs)), None)
    pos = next((r["payload"].get("positioned") for r in _runs(obs)), None)
    if cen:
        rows += ["", f"The page: {cen.get('elements'):,} elements, {cen.get('messages')} messages, "
                     f"{cen.get('katex_roots'):,} `.katex` roots with "
                     f"{cen.get('katex_descendants'):,} descendants, "
                     f"{cen.get('code_blocks'):,} code blocks, scroller "
                     f"{cen.get('scroller_scroll_height'):,} px tall."]
    if pos:
        md, kd = pos.get("message_descendants") or {}, pos.get("katex_descendants") or {}
        rows += [f"Sampled positioned elements (`position != static`, which is exactly what "
                 f"`RenderBoxModelObject::requiresLayer()` keys on): "
                 f"{md.get('non_static')}/{md.get('sampled')} of `[data-role] *` "
                 f"(~{md.get('estimate'):,} of {md.get('total'):,}), "
                 f"{kd.get('non_static')}/{kd.get('sampled')} of `.katex *` "
                 f"(~{kd.get('estimate'):,} of {kd.get('total'):,})."]
    return "\n".join(rows)


LOCAL = {"position_static_all": 0.78, "katex_static": 0.79,
         "visibility_hidden_offscreen": 0.86}


def verdict(obs: dict) -> tuple[str, str]:
    scored = _scored(obs)
    hits = []
    for name in CANDIDATES:
        es = scored.get(name)
        if not es or _disqualified(name, es):
            continue
        sv = _mean([e["saving"] for e in es if e.get("saving") is not None])
        wr = _mean([e["work_ratio"] for e in es if e.get("work_ratio") is not None])
        if sv is not None and sv >= ARM_MIN_SAVING:
            hits.append((sv, wr, name))
    hits.sort(reverse = True)

    if not hits:
        near = [(_mean([e["saving"] for e in (scored.get(n) or [])
                        if e.get("saving") is not None]), n) for n in CANDIDATES]
        near = [(s, n) for s, n in near if s is not None]
        best = max(near)[0] if near else None
        return "DOES_NOT_TRANSFER", (
            "the RenderLayer attribution does NOT transfer to this venue. The baseline still "
            "collapses on a ONE PIXEL scroll and both controls behave, so the cost is real and "
            "the harness can see a win, but removing the positioned descendants "
            f"(`[data-role] *{{position:static}}`, `.katex *{{position:static}}`) and reaching the "
            f"walk's single early-out (`.katex{{visibility:hidden}}`) recovered at most "
            + (f"{best:+.0%}" if best is not None else "nothing measurable")
            + " of the per-scroll-event cost, against 78-86% on the local llvmpipe rig. Whatever "
              "owns the cost on a real GPU is not the descendant RenderLayer walk alone")

    sv, wr, name = hits[0]
    local = LOCAL.get(name)
    bigger = (local is not None and sv > local)
    others = "; ".join(f"`{n}` {s:+.0%}" for s, _, n in hits[1:]) or "no other arm qualified"
    return "TRANSFERS", (
        f"the attribution TRANSFERS. `{name}` removes {sv:.0%} of the per-scroll-event cost on "
        f"the real gfx1151 ({wr:.2f}x less blocked time per 1 px scroll), scored against its own "
        f"two neighbouring baselines, with the negative control flat and the positive control "
        f"recovering. Against {local:.0%} for the same arm on the local llvmpipe rig, that is "
        + ("LARGER here, which is what the compositing hypothesis predicts: on a real GPU "
           "`usesCompositedScrolling()` is true and adds a second full descendant walk per scroll "
           "that llvmpipe never takes"
           if bigger else
           "NOT larger here, so the second compositing walk is not adding to the cost in the way "
           "the hypothesis predicts, even though the underlying attribution holds")
        + f". Other qualifying arms: {others}")


def observed_capabilities(obs: dict) -> dict[str, bool]:
    d = obs.get("dist") or {}
    return {
        "webkitgtk": bool(_runs(obs)),
        "headless_display_server": bool((obs.get("xserver") or {}).get("display")),
        "studio_production_bundle": bool(d.get("index_html")) and (d.get("asset_files") or 0) > 0,
        "studiobench_ladder": bool(_runs(obs)),
    }
