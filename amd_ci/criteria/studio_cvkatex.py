#!/usr/bin/env python3
"""Criteria: what does `content-visibility: auto` on KaTeX buy, and where can it not reach?

Judges only. Observations come from probes/studio_cvkatex_probe.py driving
probes/studio_ladder/cvk_layers.js.

THE CLAIM UNDER TEST. Run 32869180652 measured, on this venue, a 500K thread whose ONE PIXEL
scroll costs 287.6 ms blocked per scroll event at 3.2 fps, and `.katex{visibility:hidden}` taking
that to 10.0 ms at 61.0 fps, a 97% removal. That arm is a MECHANISM PROBE and can never ship: it
works by making the maths invisible. `content-visibility: auto` is the shippable form of the same
idea -- off-screen maths generates no boxes and no layers, on-screen maths renders exactly as it
does today -- and the only question worth asking here is how much of that 97% a rule which keeps
the maths visible can actually reach.

THE CEILING IS SET BY THE CORPUS, NOT BY THE ENGINE'S GENEROSITY. This thread carries 1,027
`.katex` roots of which only 117 are `.katex-display`. Their 101,306 descendants split 77,832
inline against 23,591 display. The positioned boxes -- the population
`RenderBoxModelObject::requiresLayer()` actually keys on, since `isPositioned()` is
`position != static` -- split ~15,291 under inline maths against ~4,919 under display maths.
`content-visibility` requires SIZE CONTAINMENT, size containment does not apply to a non-atomic
inline-level box (css-contain-2 #containment-size, enforced in WebKit by
`Style::ContainmentChecker::shouldApplySizeContainment`), and shipped `katex.css` gives `.katex` no
`display` at all, so inline maths is `display: inline` and permanently out of reach. The shippable
rule can therefore address roughly a quarter of the layers the upper-bound probe deletes, and a
verdict that reports a saving without saying that is selling the number.

`fired` IS NOT `took_effect`, AND FOR THIS FIX THAT DISTINCTION IS THE WHOLE FILE.
`getComputedStyle(el).contentVisibility` returns the SPECIFIED value whether or not the engine is
skipping anything, so an arm can read `fired: true` on 400 sampled inline `.katex` roots that the
engine is ignoring completely. The behavioural signal is the height probe the scene records: a
skipped root is size-contained, so its used height becomes the `contain-intrinsic-size` placeholder
instead of the height its contents would produce. An arm that is `fired: true` with
`took_effect.changed == 0` is the VACUOUS ARM case -- the declaration was accepted and the engine
did nothing with it -- and its clean null proves nothing. Three `overflow-anchor` arms in this
campaign were already vacuous in the cruder `fired` sense; this one would be invisible without the
height probe.

THE HEADLINE COMPARISON. `content_visibility_katex_display` (THE SHIPPABLE ARM, byte for byte the
rule the PR adds) and `content_visibility_katex_all` (the same rule with `.katex` added to the
selector) should read the SAME if the spec reading above is right, because 910 of the 1,027 roots
are inline and the declaration is silently inert on them. That is measured here rather than
asserted: if the all-selector arm reads MATERIALLY BETTER, the reading is wrong and the shipped
rule is leaving something on the table, and this file says so.

THE REFERENCE UPPER BOUND IS NOT A CANDIDATE. `visibility_hidden_offscreen` is re-measured in the
same session for one reason: to say whether this session reproduces the 97% that run 32869180652
measured. If it does not, the candidates cannot be read against that published number, because the
session is not the same experiment. It is gated as such, and it is never allowed to carry a
verdict, because it is not shippable.

MULTI-RUNG. The probe measures several rungs in one observation file and every per-window
computation is grouped BY RUNG. An arm at 100K scored against a baseline at 500K would be a ratio
between two different pages. Gates about the instrument are evaluated per rung and fail the run if
ANY rung fails, naming the rung.

THE SHORT RUNG IS A RESULT, NOT A FAILURE. At 0K the thread is empty: no maths, no scrollable
range, `no_scroll_range: true` and an empty arm list. That leg is what answers "does this rule cost
anything at short context", and the answer it gives is structural rather than statistical -- the
selector matches nothing there, so the rule cannot cost anything. It is reported with its census
and its idle numbers, and gated only on idle frame rate not being worse than at the long rung.

EVERY ARM IS SCORED AGAINST ITS OWN TWO NEIGHBOURING BASELINES, because a baseline runs between
every arm. A before/after pair cannot separate an arm from page drift, and the local replication of
the earlier ablation failed on precisely that: baseline 17.5 fps against baseline_repeat 30.1 fps.

THE DRIFT GATE HAS ALREADY FIRED ONCE HERE, on run 32865232787, which read baseline 261.4 ms
blocked per scroll event against a 162.6 / 193.9 / 197.5 / 184.5 cluster while the DOM grew by
11,205 elements. The cause was a park position recomputed per window; the fix is a FIXED park pixel
plus a DISCARDED warm-up run to quiescence, both gated below. The threshold is unchanged.

GATES THAT ARE ABOUT THE INSTRUMENT FAIL THE RUN. Conditions that are about ONE ARM disqualify that
arm and are printed, because a single bad arm should not throw away the others:
  * an arm whose declaration was DROPPED by the engine (`fired`);
  * an arm whose declaration was ACCEPTED and did nothing (`took_effect`), which is the failure
    mode `content-visibility` on inline maths produces;
  * an arm that changed scrollHeight enough to pin the gesture. One such arm in this campaign
    reported 111 fps at 184% busy and meant nothing at all.
"""

from __future__ import annotations

TITLE = ("What does `content-visibility: auto` on KaTeX buy on a real GPU, and how much of the "
         "97% `visibility: hidden` upper bound can a shippable rule reach?")
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

CANDIDATES = ["content_visibility_katex_display", "content_visibility_katex_all",
              "content_visibility_math_blocks"]
# NOT a candidate. `.katex{visibility:hidden}` makes the maths invisible, so it can never ship; it
# is kept in the sequence only to say whether this session reproduces the published 97%.
UPPER_BOUND = "visibility_hidden_offscreen"
# The one arm a verdict may be built on: it is the rule the PR adds.
SHIPPABLE = "content_visibility_katex_display"
# The same rule with `.katex` added, which exists to prove the inline-maths claim rather than
# assert it.
ALL_SELECTOR = "content_visibility_katex_all"
# NOT SHIPPABLE AS WRITTEN: it hoists the declaration to a block ancestor, which is what a
# renderer-side change would have to do. Labelled EXPLORATORY everywhere it is printed.
EXPLORATORY = "content_visibility_math_blocks"

# Low to high. An unknown rung sorts last rather than crashing the report.
RUNG_ORDER = ["0K", "1K", "10K", "50K", "100K", "500K"]

# The jam must be visible at idle. 0b prices this at 61.2 -> 17.2 fps on the correct channel and a
# pinned 60.0 on the broken one, so 25% is a floor and not a target.
LIVENESS_MIN_DROP = 0.25
# baseline_repeat against the FIRST baseline of the SAME rung, on the primary metric.
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
# scrollHeight change that pins the gesture. NOT widened for the content-visibility arms: a
# placeholder height is EXPECTED to move the thread a little, and the whole point of the threshold
# is that a fraction of a percent on a 300,000 px document cannot pin a one-pixel gesture while
# several percent can.
MAX_SCROLL_HEIGHT_DELTA = 0.02
# Share of the sampled off-screen boxes whose height must actually move before a
# `content-visibility` arm counts as having been ACTED on rather than merely accepted.
TOOK_EFFECT_MIN_FRACTION = 0.5
# The reference upper bound has to reproduce, or this session is not the experiment that produced
# the published 97% and no candidate can be read against it.
UPPER_BOUND_MIN_SAVING = 0.80
# How much better the all-selector arm may read before the spec reading is wrong.
KATEX_ALL_MAX_EXCESS = 0.05
# Idle frame rate at the short rung, against the long rung. The short rung has fewer elements and
# no maths, so it cannot legitimately be slower.
SHORT_RUNG_IDLE_TOLERANCE = 0.05

# Run 32869180652, this venue, 500K, same scene minus the three new arms.
PUBLISHED_RUN = "32869180652"
PUBLISHED_BASE_MS, PUBLISHED_BASE_FPS = 287.6, 3.2
PUBLISHED_UPPER_MS, PUBLISHED_UPPER_FPS, PUBLISHED_UPPER_SAVING = 10.0, 61.0, 0.97


# ── payload access, GROUPED BY RUNG ───────────────────────────────────────────────────────────

def _all_runs(obs: dict) -> list[dict]:
    return list(obs.get("runs") or [])


def _runs(obs: dict) -> list[dict]:
    return [r for r in _all_runs(obs) if (r.get("payload") or {}).get("ok")]


def _rung_of(run: dict) -> str:
    return str(run.get("rung") or (run.get("payload") or {}).get("rung") or "?")


def _by_rung(obs: dict) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for r in _runs(obs):
        out.setdefault(_rung_of(r), []).append(r)
    return out


def _rungs(obs: dict) -> list[str]:
    by = _by_rung(obs)
    return sorted(by, key = lambda r: (RUNG_ORDER.index(r) if r in RUNG_ORDER else 99, r))


def _is_short(payload: dict) -> bool:
    """No scrollable range, or nothing scored in it. A RESULT, not a failure."""
    return bool(payload.get("no_scroll_range")) or not (payload.get("arms") or [])


def _short_rungs(obs: dict) -> list[str]:
    by = _by_rung(obs)
    return [r for r in _rungs(obs) if all(_is_short(x["payload"]) for x in by[r])]


def _scored_rungs(obs: dict) -> list[str]:
    short = set(_short_rungs(obs))
    return [r for r in _rungs(obs) if r not in short]


def _long_rung(obs: dict) -> str | None:
    """The highest rung that carries scored windows. This is where every claim is made."""
    rs = _scored_rungs(obs)
    return rs[-1] if rs else None


def _rung_runs(obs: dict, rung: str) -> list[dict]:
    return _by_rung(obs).get(rung) or []


def _windows(obs: dict, rung: str) -> list[tuple[int, list[dict]]]:
    """Per repetition of ONE rung, the arm windows IN ORDER. Order makes neighbours meaningful."""
    return [(r.get("rep"), list((r["payload"].get("arms") or []))) for r in _rung_runs(obs, rung)]


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


def _scored(obs: dict, rung: str) -> dict[str, list[dict]]:
    """For every non-baseline window OF ONE RUNG, its own local comparison, one per repetition."""
    out: dict[str, list[dict]] = {}
    for rep, seq in _windows(obs, rung):
        for i, a in enumerate(seq):
            if a.get("arm") == BASELINE_ARM and a.get("name") != BASELINE_REPEAT:
                continue
            base_b = _neighbour_baseline(seq, i, _bmpf)
            base_f = _neighbour_baseline(seq, i, lambda x: x.get("eff_fps"))
            b, f = _bmpf(a), a.get("eff_fps")
            entry = {
                "rung": rung, "rep": rep, "name": a.get("name"), "arm": a.get("arm"),
                "ok": a.get("ok"), "bmpf": b, "fps": f, "frames": a.get("frames"),
                "busy": (a.get("busy") or {}).get("busy_pct"),
                "worst_ms": (a.get("raf") or {}).get("max_ms"),
                "base_bmpf": base_b, "base_fps": base_f,
                # SAVING on the primary metric: the share of the per-scroll-event cost removed.
                "saving": (1 - b / base_b) if (b is not None and base_b) else None,
                "work_ratio": (base_b / b) if (b and base_b) else None,
                "fps_ratio": (f / base_f) if (f and base_f) else None,
                "fired": a.get("fired"),
                # The engine ACTED on it, as opposed to having accepted it.
                "took_effect": a.get("took_effect"),
                "probe_before": a.get("probe_before"), "probe_after": a.get("probe_after"),
                "scroll_height_delta": a.get("scroll_height_delta"),
                "mutations": a.get("mutations") or {},
                "gesture": a.get("gesture") or {},
            }
            out.setdefault(a.get("name"), []).append(entry)
    return out


def _entries(obs: dict, rung: str | None, name: str) -> list[dict]:
    return (_scored(obs, rung).get(name) or []) if rung else []


def _saving(obs: dict, rung: str | None, name: str):
    return _mean([e["saving"] for e in _entries(obs, rung, name) if e.get("saving") is not None])


def _took_effect(entries: list[dict]) -> dict | None:
    for e in entries:
        t = e.get("took_effect")
        if isinstance(t, dict) and t.get("fraction_changed") is not None:
            return t
    return None


def _idle(payload: dict) -> tuple[float | None, float | None]:
    """(idle fps, idle blocked ms per frame). The liveness record carries both by construction."""
    idle = payload.get("idle") or {}
    liv = payload.get("liveness") or {}
    fps = idle.get("eff_fps")
    if fps is None:
        fps = liv.get("clean_fps")
    blocked = idle.get("blocked_ms_per_frame")
    if blocked is None:
        blocked = liv.get("clean_blocked_ms_per_frame")
    return fps, blocked


def _is_candidate(name: str) -> bool:
    return name in CANDIDATES


def _cv_arm(name: str) -> bool:
    return name.startswith("content_visibility")


def _label(name: str) -> str:
    """Every printed name carries what it is, so no reader has to remember which arm can ship."""
    if name == SHIPPABLE:
        return " **[SHIPPABLE]**"
    if name == UPPER_BOUND:
        return " **[REFERENCE UPPER BOUND, NOT SHIPPABLE]**"
    if name == EXPLORATORY:
        return " **[EXPLORATORY, NOT SHIPPABLE AS WRITTEN]**"
    if name == ALL_SELECTOR:
        return " **[SELECTOR PROBE, NOT A FIX]**"
    return {POSITIVE: " **[POSITIVE CONTROL]**", NEGATIVE: " **[NEGATIVE CONTROL]**",
            FLOOR: " **[FLOOR]**", BASELINE_REPEAT: " **[DRIFT CHECK]**"}.get(name, "")


# ── per-arm disqualifiers ─────────────────────────────────────────────────────────────────────

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
        # THE VACUOUS ARM. `fired` is a getComputedStyle re-read and only says the declaration was
        # ACCEPTED; `took_effect` is the height probe and says the engine ACTED on it. For
        # `content-visibility` the two come apart by design: size containment does not apply to a
        # non-atomic inline-level box, so the rule is silently inert on `display: inline` KaTeX,
        # which is 910 of this corpus's 1,027 `.katex` roots.
        t = e.get("took_effect")
        if (_is_candidate(name) and isinstance(t, dict)
                and t.get("fraction_changed") is not None
                and t["fraction_changed"] < TOOK_EFFECT_MIN_FRACTION):
            return (f"the declaration was ACCEPTED and did nothing on the population it was "
                    f"probed against: `{t.get('selector')}` reports the property back through "
                    f"getComputedStyle, but only {t.get('changed')}/{t.get('compared')} of the "
                    f"sampled off-screen boxes changed height ({t['fraction_changed']:.0%}, mean "
                    f"{t.get('mean_height_before')} -> {t.get('mean_height_after')} px), so the "
                    f"engine skipped nothing there. `content-visibility` requires SIZE "
                    f"CONTAINMENT and size containment does not apply to a non-atomic "
                    f"inline-level box, so the rule is silently inert on `display: inline` KaTeX "
                    f"-- 910 of this corpus's 1,027 `.katex` roots. A vacuous arm's null proves "
                    f"nothing and cannot carry a verdict")
        d = e.get("scroll_height_delta")
        if d is not None and abs(d) > MAX_SCROLL_HEIGHT_DELTA:
            if _cv_arm(name):
                return (f"scrollHeight moved {d:+.1%}, above the {MAX_SCROLL_HEIGHT_DELTA:.0%} "
                        f"gate. A scrollHeight change is EXPECTED for this fix -- "
                        f"`contain-intrinsic-size` replaces an unrendered block's real height with "
                        f"a placeholder, so the thread's height moves by the error in that "
                        f"placeholder -- and the gate is about MAGNITUDE, not about existence: a "
                        f"fraction of a percent on a 300,000 px document cannot pin a one-pixel "
                        f"gesture, several percent can. {abs(d):.1%} is several percent, so this "
                        f"window is not the same window as the baseline's and the placeholder "
                        f"needs to be closer to the measured block heights")
            return (f"scrollHeight moved {d:+.1%}, so this arm changed the thing the gesture "
                    f"scrolls and its window is not the same window as the baseline's")
        if not e.get("ok"):
            return "the arm did not complete"
    return None


# ── gates ─────────────────────────────────────────────────────────────────────────────────────

def _per_rung(obs: dict, fn, *, scored_only: bool = False) -> tuple[bool, str]:
    """Evaluate a per-rung condition. FAIL if ANY rung fails, and the text names the rung.

    A rung with no scrollable range has no windows, so a window gate has nothing to say about it
    and says so instead of failing it. If no rung has windows at all, that is reported by the
    scoring-rung gate rather than by every window gate at once.
    """
    rungs = _rungs(obs)
    if not rungs:
        return False, "no repetition produced a payload"
    short = set(_short_rungs(obs))
    oks, txt = [], []
    for rung in rungs:
        if scored_only and rung in short:
            txt.append(f"{rung}: no scrollable range, so no window to score here")
            continue
        ok, t = fn(rung, _rung_runs(obs, rung))
        oks.append(ok)
        txt.append(f"{rung}: {t}" + ("" if ok else "  <- FAILS AT THIS RUNG"))
    return (all(oks) if oks else True), "; ".join(txt)


def _gate_records(obs: dict) -> list[dict]:
    runs, ok = _all_runs(obs), _runs(obs)
    by = _by_rung(obs)
    long_rung = _long_rung(obs)
    out: list[dict] = []

    def g(name: str, passed: bool, evidence: str, instrument: bool = True) -> None:
        out.append({"name": name, "ok": bool(passed), "evidence": evidence,
                    "instrument": instrument})

    g("a display server was obtained", bool((obs.get("xserver") or {}).get("display")),
      obs.get("fatal") or str((obs.get("xserver") or {}).get("display")))

    d = obs.get("dist") or {}
    g("Studio installed and built a PRODUCTION bundle",
      bool(d.get("index_html")) and (d.get("asset_files") or 0) > 0,
      f"install rc={(obs.get('install') or {}).get('rc')}; assets={d.get('asset_files')}")

    # A rung with no scrollable range still COMPLETED. Failing it here would report the
    # short-context leg as a broken run instead of as the flat leg it is asked for.
    g("every repetition completed", bool(runs) and len(ok) == len(runs),
      f"{len(ok)}/{len(runs)} ok across rungs "
      + str({k: len(v) for k, v in sorted(by.items())})
      + ("" if len(ok) == len(runs) else "; " + "; ".join(
          f"{_rung_of(r)}#{r.get('rep')}: "
          + str((r.get('payload') or {}).get('error') or r.get('rc'))[:160]
          for r in runs if r not in ok)))

    g("every rung was measured at least twice, so a per-rep disagreement is visible",
      bool(by) and all(len(v) >= 2 for v in by.values()),
      ", ".join(f"{k}x{len(v)}" for k, v in sorted(by.items())) or "no runs")

    # Without one there is nothing to ablate and no claim to make; the short rungs alone cannot
    # answer the question, they can only bound its cost.
    g("at least one rung carried a scrollable range to score", long_rung is not None,
      f"scored rungs {_scored_rungs(obs)}, short rungs {_short_rungs(obs)}")

    # LIVENESS. The channel must be able to report a blocked main thread AT ALL. Priced at idle,
    # where it has room to fall; pricing a control where the page is already saturated is how a
    # control with ample power gets thrown away.
    def _liveness(rung, rs):
        liv = [r["payload"].get("liveness") or {} for r in rs]
        drops = [l.get("drop_fraction") for l in liv if l.get("drop_fraction") is not None]
        txt = "; ".join(f"{l.get('clean_fps')} -> {l.get('jammed_fps')} fps "
                        f"({(l.get('drop_fraction') or 0):.0%} fall), blocked/frame "
                        f"{l.get('clean_blocked_ms_per_frame')} -> "
                        f"{l.get('jammed_blocked_ms_per_frame')} ms" for l in liv)
        return (bool(drops) and min(drops) >= LIVENESS_MIN_DROP), (txt or "no liveness window")

    g(f"LIVENESS: a deliberate main-thread jam drops idle fps by >= {LIVENESS_MIN_DROP:.0%} at "
      f"every rung", *_per_rung(obs, _liveness))

    def _clamp(rung, rs):
        cs = [(r["payload"].get("clamp") or {}) for r in rs]
        return (bool(cs) and all(c.get("clamp_ms") is not None for c in cs),
                "; ".join(str(c.get("clamp_ms") or c.get("reason")) for c in cs) or "no clamp")

    g("the busy clamp calibrated at every rung, so blocked-ms is defined", *_per_rung(obs, _clamp))

    # `scroll-behavior: smooth` on the viewport turns an assignment into an ANIMATION, so a
    # same-turn read-back reports a real gesture as inert.
    def _guard(rung, rs):
        gs = [(r["payload"].get("guard") or {}) for r in rs]
        return (bool(gs) and all(x.get("ok") for x in gs),
                "; ".join(f"{x.get('scroll_behavior_before')} -> {x.get('scroll_behavior_after')} "
                          f"on {x.get('scroller_tag')}.{(x.get('scroller_class') or '')[:40]}"
                          for x in gs) or "no guard record")

    g("scroll-behavior was forced to auto on the scroller at every rung", *_per_rung(obs, _guard))

    # The gesture must have ACHIEVED what it commanded. A probe that reports what it asked for and
    # samples only the endpoints will agree with itself perfectly while measuring nothing.
    def _gesture(rung, rs):
        tf, sb = [], []
        for _, seq in _windows(obs, rung):
            for a in seq:
                ge = a.get("gesture") or {}
                if a.get("arm") == BASELINE_ARM and ge.get("travel_fraction") is not None:
                    tf.append(ge["travel_fraction"])
                    sb.append(ge.get("snapback_frames") or 0)
        if not tf:
            return False, "no baseline gesture recorded"
        return (min(tf) >= 0.9,
                f"travel fraction min {min(tf):.2f} over {len(tf)} baseline windows, snapback "
                f"frames max {max(sb)}")

    g("the 1 px gesture actually moved the scroller on every baseline window",
      *_per_rung(obs, _gesture, scored_only = True))

    # DRIFT, per rung, because baseline_repeat at 100K says nothing about 500K. Reported first in
    # the summary because a drift-corrected number is the only honest one: if this fails the
    # correct output is "it drifted", not a ratio against a contaminated baseline.
    def _drift(rung, rs):
        txt, okd = [], True
        for rep, seq in _windows(obs, rung):
            first = next((x for x in seq if x.get("name") == FIRST_BASELINE), None)
            rep_w = next((x for x in seq if x.get("name") == BASELINE_REPEAT), None)
            if not first or not rep_w or not _bmpf(first) or _bmpf(rep_w) is None:
                okd = False
                txt.append(f"rep {rep}: missing a baseline window")
                continue
            rel = _bmpf(rep_w) / _bmpf(first) - 1
            allb = [_bmpf(x) for x in seq if x.get("arm") == BASELINE_ARM and _bmpf(x) is not None]
            txt.append(f"rep {rep}: baseline {_bmpf(first)} -> baseline_repeat {_bmpf(rep_w)} ms "
                       f"blocked/frame ({rel:+.0%}); all baselines {[round(x, 1) for x in allb]}")
            if abs(rel) > DRIFT_MAX:
                okd = False
        return (okd and bool(txt)), ("; ".join(txt) or "no baseline windows")

    g(f"DRIFT: baseline_repeat within {DRIFT_MAX:.0%} of the first baseline of the same rung",
      *_per_rung(obs, _drift, scored_only = True))

    # THE DEFERRED-FENCE RESERVOIR. `code-fence-defer.tsx` ships `SHIP_DEFAULT = "defer"` and
    # `useFenceReached` latches on an IntersectionObserver with a 100% root margin, which
    # re-delivers on LAYOUT CHANGE with no scroll at all. So any arm that changes the thread's
    # height mounts markup, permanently, into every window after it -- and a placeholder height is
    # exactly such a change. The warm-up drains it through the app's own `beforeprint` path.
    def _fence(rung, rs):
        lat = [(r["payload"].get("fence_latch") or {}) for r in rs]
        left = [((l.get("after") or {}).get("fences_deferred")) for l in lat]
        return (bool(lat) and all(x == 0 for x in left),
                "; ".join(f"{(l.get('before') or {}).get('fences_deferred')} deferred -> "
                          f"{(l.get('after') or {}).get('fences_deferred')} left, "
                          f"{l.get('fences_latched')} latched, +{l.get('elements_added')} elements "
                          f"(+{l.get('highlight_spans_added')} `pre span`), dispatched="
                          f"{l.get('dispatched')} {l.get('error') or ''}" for l in lat)
                or "no fence-latch record")

    g("the deferred-fence reservoir was drained before scoring, at every rung",
      *_per_rung(obs, _fence))

    # THE WARM-UP. A first-visit cost paid inside the first scored window is exactly what failed
    # run 32865232787, so the window that absorbs it is discarded and its convergence is gated.
    def _warm(rung, rs):
        ws = [r["payload"].get("warmup") or {} for r in rs]
        return (bool(ws) and all(w.get("quiesced") for w in ws),
                "; ".join(f"{len(w.get('rounds') or [])} rounds, quiesced={w.get('quiesced')}, "
                          f"absorbed {(w.get('mut_total') or {}).get('added_elements')} added "
                          f"elements in {w.get('elapsed_ms')} ms, per-round element deltas "
                          f"{[r_.get('element_delta') for r_ in (w.get('rounds') or [])]}"
                          for w in ws) or "no warm-up window")

    g("the discarded warm-up reached quiescence before any scoring began, at every rung",
      *_per_rung(obs, _warm))

    # And it must STAY quiescent, or a scored window is measuring deferred mounting.
    def _quiet(rung, rs):
        noisy = []
        for rep_i, seq in _windows(obs, rung):
            for a in seq:
                if a.get("arm") == POSITIVE:
                    continue  # detaching 28 messages is supposed to mutate the DOM
                m = (a.get("mutations") or {}).get("added_elements")
                if m is not None and m > MAX_SCORED_WINDOW_MUTATIONS:
                    noisy.append(f"rep {rep_i} {a.get('name')}: +{m} elements "
                                 f"{(a.get('mutations') or {}).get('buckets')}")
        return (not noisy), ("; ".join(noisy) if noisy else "every scored window was quiescent")

    g(f"no scored window mounted more than {MAX_SCORED_WINDOW_MUTATIONS} elements",
      *_per_rung(obs, _quiet, scored_only = True))

    # Every window of a rung must have looked at the SAME content, which is what a fixed park
    # pixel buys. The park differs BETWEEN rungs by construction, which is why this is per rung.
    def _park(rung, rs):
        parks = {r["payload"].get("park") for r in rs}
        okp = bool(parks)
        for _, seq in _windows(obs, rung):
            for a in seq:
                ge = a.get("gesture") or {}
                if ge and ge.get("park") is not None and ge["park"] not in parks:
                    okp = False
        return okp, f"park positions {sorted(x for x in parks if x is not None)}"

    g("every window parked at the same fixed pixel within its rung",
      *_per_rung(obs, _park, scored_only = True))

    # The defect must still be present, or there is nothing to ablate. Priced against the FLOOR
    # measured on this very page (`still`), not against an assumed 60 fps.
    def _collapse(rung, rs):
        sc = _scored(obs, rung)
        floor = _mean([e["bmpf"] for e in sc.get(FLOOR, []) if e.get("bmpf") is not None])
        base = _mean([_bmpf(x) for _, seq in _windows(obs, rung) for x in seq
                      if x.get("arm") == BASELINE_ARM])
        if base is None or floor is None:
            return False, f"baseline={base} floor={floor}"
        return (base > max(2.0, floor * 3),
                f"baseline {base:.1f} ms blocked per scroll event against a `still` floor of "
                f"{floor:.1f} ms")

    g("the baseline still exhibits the collapse (a 1 px scroll costs far more than assigning the "
      "same scrollTop)", *_per_rung(obs, _collapse, scored_only = True))

    def _neg(rung, rs):
        s = _saving(obs, rung, NEGATIVE)
        return (s is not None and s <= NEGATIVE_MAX_SAVING,
                f"noop_touch {s:+.0%}" if s is not None else "no negative control window")

    g(f"NEGATIVE control removes <= {NEGATIVE_MAX_SAVING:.0%} of the cost",
      *_per_rung(obs, _neg, scored_only = True))

    def _pos(rung, rs):
        sc = _scored(obs, rung)
        floor = _mean([e["bmpf"] for e in sc.get(FLOOR, []) if e.get("bmpf") is not None])
        base = _mean([_bmpf(x) for _, seq in _windows(obs, rung) for x in seq
                      if x.get("arm") == BASELINE_ARM])
        p = _mean([e["bmpf"] for e in sc.get(POSITIVE, []) if e.get("bmpf") is not None])
        if p is None or base is None or floor is None or base <= floor:
            return False, "no positive control window"
        rec = (base - p) / (base - floor)
        return (rec >= POSITIVE_MIN_RECOVERY,
                f"detach_messages {p:.1f} ms blocked/frame against baseline {base:.1f} and floor "
                f"{floor:.1f} = {rec:.0%}")

    g(f"POSITIVE control recovers >= {POSITIVE_MIN_RECOVERY:.0%} of the way to the floor",
      *_per_rung(obs, _pos, scored_only = True))

    # THE REFERENCE UPPER BOUND. Not a candidate and not shippable: it removes the cost by making
    # the maths invisible. Its only job is to say whether this session is the same experiment as
    # the one that published +97%, because every candidate number below is read against that.
    ub = _saving(obs, long_rung, UPPER_BOUND)
    ub_entries = _entries(obs, long_rung, UPPER_BOUND)
    ub_txt = (f"`{UPPER_BOUND}` removes {ub:+.0%} at {long_rung} "
              f"({_mean([e['bmpf'] for e in ub_entries]):.1f} ms blocked/frame against "
              f"{_mean([e['base_bmpf'] for e in ub_entries]):.1f} baseline), against "
              f"+{PUBLISHED_UPPER_SAVING:.0%} in run {PUBLISHED_RUN} "
              f"({PUBLISHED_BASE_MS} ms at {PUBLISHED_BASE_FPS} fps -> {PUBLISHED_UPPER_MS} ms at "
              f"{PUBLISHED_UPPER_FPS} fps)"
              if ub is not None else
              (f"no `{UPPER_BOUND}` window at {long_rung}" if long_rung is not None else
               "no rung carried a scrollable range, so the reference upper bound was never "
               "measured in this session"))
    if long_rung is not None and (ub is None or ub < UPPER_BOUND_MIN_SAVING):
        ub_txt += (". This session did NOT reproduce the published upper bound, so it is not the "
                   "same experiment and the candidate savings below cannot be read against the "
                   "published 97%")
    g(f"the reference upper bound reproduces: `{UPPER_BOUND}` removes >= "
      f"{UPPER_BOUND_MIN_SAVING:.0%} of the cost at the long rung",
      ub is not None and ub >= UPPER_BOUND_MIN_SAVING, ub_txt)

    # THE HEADLINE COMPARISON. The two arms should read the SAME if inline maths is out of reach.
    disp, alls = _saving(obs, long_rung, SHIPPABLE), _saving(obs, long_rung, ALL_SELECTOR)
    if long_rung is None:
        excess_txt = ("no rung carried a scrollable range, so neither arm has a window and the "
                      "comparison this run exists for was never made")
        excess_ok = False
    elif disp is None or alls is None:
        excess_txt = (f"missing an arm at {long_rung}: `{SHIPPABLE}`="
                      f"{'-' if disp is None else format(disp, '+.0%')}, `{ALL_SELECTOR}`="
                      f"{'-' if alls is None else format(alls, '+.0%')}")
        excess_ok = False
    else:
        excess = alls - disp
        te_d, te_a = _took_effect(_entries(obs, long_rung, SHIPPABLE)), \
            _took_effect(_entries(obs, long_rung, ALL_SELECTOR))
        excess_ok = excess <= KATEX_ALL_MAX_EXCESS
        excess_txt = (f"`{SHIPPABLE}` {disp:+.0%} against `{ALL_SELECTOR}` {alls:+.0%} at "
                      f"{long_rung}, a difference of {excess:+.1%}"
                      + (f"; height probe on the display roots {te_d.get('changed')}/"
                         f"{te_d.get('compared')} changed" if te_d else "")
                      + (f", on the INLINE roots {te_a.get('changed')}/{te_a.get('compared')} "
                         f"changed" if te_a else ""))
        if not excess_ok:
            excess_txt += (f". Adding `.katex` bought MORE than {KATEX_ALL_MAX_EXCESS:.0%}, so the "
                           f"spec reading is WRONG: size containment is reaching inline maths here "
                           f"and the shipped rule is leaving something on the table")
    g("adding `.katex` to the selector buys nothing, which is what says inline maths is out of "
      "reach", excess_ok, excess_txt, instrument = False)

    # THE SHORT-CONTEXT CLAIM. Structural, not statistical: the selector matches nothing at 0K.
    shorts = _short_rungs(obs)
    if not shorts or long_rung is None:
        g("idle frame rate at the short rung is not worse than at the long rung", True,
          f"short rungs {shorts or 'none'}, long rung {long_rung}: the short-context claim is not "
          f"made by this observation")
    else:
        parts, okc = [], True
        lf = _mean([_idle(r["payload"])[0] for r in _rung_runs(obs, long_rung)])
        for rung in shorts:
            sf = _mean([_idle(r["payload"])[0] for r in _rung_runs(obs, rung)])
            sb = _mean([_idle(r["payload"])[1] for r in _rung_runs(obs, rung)])
            cen = next((r["payload"].get("baseline_census") or {}
                        for r in _rung_runs(obs, rung)), {})
            if sf is None or lf is None:
                okc = False
                parts.append(f"{rung}: no idle reading")
                continue
            if sf < lf * (1 - SHORT_RUNG_IDLE_TOLERANCE):
                okc = False
            sb_txt = "-" if sb is None else f"{sb:.1f}"
            parts.append(f"{rung}: idle {sf:.1f} fps at {sb_txt} ms blocked/frame against "
                         f"{lf:.1f} fps at {long_rung}, with {cen.get('katex_roots')} `.katex` "
                         f"roots and {cen.get('katex_display')} `.katex-display`")
        g("idle frame rate at the short rung is not worse than at the long rung", okc,
          "; ".join(parts))
    return out


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    return [(r["name"], r["ok"], r["evidence"]) for r in _gate_records(obs)]


# ── report ────────────────────────────────────────────────────────────────────────────────────

def _rung_table(obs: dict, rung: str) -> list[str]:
    scored = _scored(obs, rung)
    order: list[str] = []
    for _, seq in _windows(obs, rung):
        for a in seq:
            if a.get("name") not in order:
                order.append(a.get("name"))

    rows = [f"### Rung {rung}", "",
            "| window | blocked ms/frame | vs its two neighbouring baselines | cost removed | "
            "fps | busy | worst frame | scrollHeight | elements mounted | declaration accepted | "
            "engine acted |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name in order:
        es = scored.get(name)
        if es is None:
            vals = [_bmpf(a) for _, seq in _windows(obs, rung) for a in seq
                    if a.get("name") == name]
            fps = [a.get("eff_fps") for _, seq in _windows(obs, rung) for a in seq
                   if a.get("name") == name]
            mut = [(a.get("mutations") or {}).get("added_elements") or 0
                   for _, seq in _windows(obs, rung) for a in seq if a.get("name") == name]
            rows.append(f"| `{name}` (baseline) | "
                        + ", ".join(f"{v:.1f}" for v in vals if v is not None)
                        + " | - | - | " + ", ".join(f"{v:.1f}" for v in fps if v is not None)
                        + f" | - | - | - | {max(mut) if mut else 0} | - | - |")
            continue
        b = _mean([e["bmpf"] for e in es])
        wr = _mean([e["work_ratio"] for e in es])
        sv = _mean([e["saving"] for e in es])
        f = _mean([e["fps"] for e in es])
        bu = _mean([e["busy"] for e in es])
        wo = max([e["worst_ms"] for e in es if e.get("worst_ms") is not None], default = None)
        dq = _disqualified(name, es)
        sh = _mean([e["scroll_height_delta"] for e in es])
        fired = next((e.get("fired") for e in es if isinstance(e.get("fired"), dict)), None)
        te = _took_effect(es)
        rows.append("| " + " | ".join([
            f"`{name}`{_label(name)}" + ("  **[DISQUALIFIED]**" if dq else ""),
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
            "-" if not te else (f"{te.get('changed')}/{te.get('compared')} boxes "
                                f"({(te.get('fraction_changed') or 0):.0%})"),
        ]) + " |")

    # scrollHeight FOR EVERY ARM, whether or not it disqualified anything, because for this fix a
    # small movement is the expected signature of `contain-intrinsic-size` and printing it only
    # when it fails would make the expected case invisible.
    rows += ["", f"scrollHeight, measured on every arm (the gate is "
                 f"{MAX_SCROLL_HEIGHT_DELTA:.0%}; `contain-intrinsic-size` replacing an unrendered "
                 f"block's real height with a placeholder is EXPECTED to move it a little):", ""]
    for name in order:
        es = scored.get(name)
        vals = ([e["scroll_height_delta"] for e in es] if es else
                [a.get("scroll_height_delta") for _, seq in _windows(obs, rung) for a in seq
                 if a.get("name") == name])
        vals = [v for v in vals if v is not None]
        if not vals:
            rows.append(f"- `{name}`: not recorded")
            continue
        worst = max(vals, key = abs)
        over = abs(worst) > MAX_SCROLL_HEIGHT_DELTA
        exempt = name in (POSITIVE, NEGATIVE, FLOOR, BASELINE_REPEAT)
        rows.append(f"- `{name}`{_label(name)}: " + ", ".join(f"{v:+.2%}" for v in vals)
                    + ("" if not over else
                       ("  <- over the gate, and exempt from it: this arm changes the thread on "
                        "purpose" if exempt else "  <- over the gate")))

    dqs = [(n, _disqualified(n, es)) for n, es in scored.items() if _disqualified(n, es)]
    if dqs:
        rows += ["", "Disqualified arms, and why (the other arms are unaffected):", ""]
        rows += [f"- `{n}`{_label(n)}: {why}" for n, why in dqs]

    # Per repetition, because a mean over a bimodal draw is a report of how many slow draws
    # happened to land in that arm.
    rows += ["", "Per repetition, so a mean cannot hide a disagreement:", "",
             "| window | blocked ms/frame per rep | fps per rep | frames per rep |",
             "|---|---|---|---|"]
    for name in order:
        vals, fps, fr = [], [], []
        for _, seq in _windows(obs, rung):
            for a in seq:
                if a.get("name") == name:
                    vals.append(_bmpf(a))
                    fps.append(a.get("eff_fps"))
                    fr.append(a.get("frames"))
        rows.append(f"| `{name}` | " + ", ".join(str(v) for v in vals) + " | "
                    + ", ".join(str(v) for v in fps) + " | "
                    + ", ".join(str(v) for v in fr) + " |")
    return rows


def _short_section(obs: dict, rung: str) -> list[str]:
    """What the short rung establishes. It is a RESULT, and it is a structural one."""
    rs = _rung_runs(obs, rung)
    cen = next((r["payload"].get("baseline_census") or {} for r in rs), {})
    mb = next((r["payload"].get("math_blocks") or {} for r in rs), {})
    fps = _mean([_idle(r["payload"])[0] for r in rs])
    blocked = _mean([_idle(r["payload"])[1] for r in rs])
    px = next((r["payload"].get("scrollable_px") for r in rs), None)
    long_rung = _long_rung(obs)
    long_fps = _mean([_idle(r["payload"])[0] for r in _rung_runs(obs, long_rung)]) \
        if long_rung else None
    return [
        f"### Rung {rung}: no scrollable range, and that is the short-context answer", "",
        f"The thread at {rung} has {px} px of scrollable range, so no gesture window is scored "
        f"here and the arm table is empty. That is a RESULT, not a failed leg. What it "
        f"establishes is structural rather than statistical: the selector the shipped rule uses "
        f"MATCHES NOTHING at this rung, so the rule cannot cost anything at short context.", "",
        f"- `.katex` roots: **{cen.get('katex_roots')}**",
        f"- `.katex-display` roots, which is what the shipped rule selects: "
        f"**{cen.get('katex_display')}**",
        f"- maths-bearing blocks (`.cvk-mathblock`, the exploratory arm's selector): "
        f"**{cen.get('math_blocks')}**" + (f" (scene counted {mb.get('count')})"
                                           if mb.get("count") is not None else ""),
        f"- elements in the page: **{(cen.get('elements') or 0):,}**",
        f"- idle frame rate: **{fps if fps is None else round(fps, 1)} fps** at "
        f"**{blocked if blocked is None else round(blocked, 2)} ms blocked per frame**"
        + (f", against {long_fps:.1f} fps at {long_rung}" if long_fps is not None else ""),
        "",
        "A rule whose selector matches zero elements has no work to do and no memory to hold, so "
        "the only way it could cost anything here is by slowing the engine down for everyone, "
        "which is what the idle comparison above is for.",
    ]


def table(obs: dict) -> str:
    long_rung = _long_rung(obs)
    rows = ["The primary metric is **blocked ms per frame**. The gesture issues exactly one 1 px "
            "scroll per painted frame, so this is the cost of ONE SCROLL EVENT. `busy%` is a "
            "share of wall time and is shown only for continuity with earlier tables. Every "
            "figure is grouped BY RUNG: an arm at one rung scored against a baseline at another "
            "would be a ratio between two different pages.", "",
            f"`{SHIPPABLE}` is THE SHIPPABLE ARM. `{UPPER_BOUND}` is a REFERENCE UPPER BOUND and "
            f"cannot ship, because it works by making the maths invisible. `{ALL_SELECTOR}` is a "
            f"probe of the spec reading rather than a fix: it exists to show whether the inline "
            f"roots can be reached at all. `{EXPLORATORY}` is EXPLORATORY and is not shippable as "
            f"written: it hoists the declaration to a block ancestor, which is what a "
            f"renderer-side change would have to do.", ""]

    short = set(_short_rungs(obs))
    for rung in _rungs(obs):
        rows += (_short_section(obs, rung) if rung in short else _rung_table(obs, rung)) + [""]

    # THE COMPARISON THE RUN EXISTS FOR.
    if long_rung is not None:
        disp_e = _entries(obs, long_rung, SHIPPABLE)
        all_e = _entries(obs, long_rung, ALL_SELECTOR)
        d_sv, a_sv = _saving(obs, long_rung, SHIPPABLE), _saving(obs, long_rung, ALL_SELECTOR)
        te_d, te_a = _took_effect(disp_e), _took_effect(all_e)
        rows += [f"### Does adding `.katex` to the selector buy anything? ({long_rung})", "",
                 "The two arms should read the SAME if size containment cannot apply to a "
                 "non-atomic inline-level box: 910 of this corpus's 1,027 `.katex` roots are "
                 "`display: inline`, so on that reading the extra selector is inert. The height "
                 "probes are the evidence, and they are probed on DIFFERENT populations on "
                 "purpose -- the display arm on `.katex-display`, the all arm on the INLINE roots, "
                 "because those are the ones the claim is about.", "",
                 "| arm | cost removed | height probe | off-screen boxes whose height changed | "
                 "mean height before -> after |", "|---|---|---|---|---|"]
        for nm, sv, te in ((SHIPPABLE, d_sv, te_d), (ALL_SELECTOR, a_sv, te_a)):
            rows.append("| " + " | ".join([
                f"`{nm}`{_label(nm)}",
                "-" if sv is None else f"**{sv:+.0%}**",
                "-" if not te else f"`{te.get('selector')}`",
                "-" if not te else (f"{te.get('changed')}/{te.get('compared')} "
                                    f"({(te.get('fraction_changed') or 0):.0%})"),
                "-" if not te else f"{te.get('mean_height_before')} -> "
                                   f"{te.get('mean_height_after')} px",
            ]) + " |")
        if d_sv is not None and a_sv is not None:
            excess = a_sv - d_sv
            if excess > KATEX_ALL_MAX_EXCESS:
                rows += ["", f"**Adding `.katex` bought {excess:+.1%}, which is more than "
                             f"{KATEX_ALL_MAX_EXCESS:.0%}.** The spec reading above is WRONG: "
                             f"size containment is reaching inline maths in this engine, and the "
                             f"shipped rule is leaving something on the table."]
            else:
                rows += ["", f"**Adding `.katex` bought {excess:+.1%}, which is nothing.** Inline "
                             f"maths is out of reach of `content-visibility` as a MEASURED fact "
                             f"rather than as a reading of the spec, so the shipped rule is not "
                             f"leaving anything on the table by selecting only `.katex-display`."]
        rows.append("")

        # The reference upper bound, printed as what it is.
        ub = _saving(obs, long_rung, UPPER_BOUND)
        if ub is not None:
            rows += [f"### The reference upper bound ({long_rung})", "",
                     f"`{UPPER_BOUND}` removes **{ub:+.0%}** here, against +"
                     f"{PUBLISHED_UPPER_SAVING:.0%} in run {PUBLISHED_RUN} ({PUBLISHED_BASE_MS} ms "
                     f"blocked per scroll event at {PUBLISHED_BASE_FPS} fps -> "
                     f"{PUBLISHED_UPPER_MS} ms at {PUBLISHED_UPPER_FPS} fps). It is NOT a "
                     f"candidate and cannot ship: it removes the cost by making the maths "
                     f"invisible. Its only job is to say whether this session is the same "
                     f"experiment as the published one, so that the candidate numbers above can "
                     f"be read against that 97%.", ""]

    # THE DISCARDED WARM-UP, reported per rung. The whole point of it is the first-visit work it
    # absorbs, so throwing the window away without saying what was in it would repeat the mistake
    # at one remove.
    rows += ["### The discarded warm-up, which is where the first-visit cost is now paid", "",
             "| rung | rep | rounds | quiesced | elements mounted | ms | per-round element delta | "
             "where they landed |", "|---|---|---|---|---|---|---|---|"]
    for rung in _rungs(obs):
        for i, r in enumerate(_rung_runs(obs, rung), 1):
            w = r["payload"].get("warmup") or {}
            mt = w.get("mut_total") or {}
            rows.append(f"| {rung} | {r.get('rep', i)} | {len(w.get('rounds') or [])} | "
                        f"{'yes' if w.get('quiesced') else 'NO'} | {mt.get('added_elements')} | "
                        f"{w.get('elapsed_ms')} | "
                        f"{[r_.get('element_delta') for r_ in (w.get('rounds') or [])]} | "
                        f"{mt.get('buckets')} |")
    rows.append("")

    if long_rung is not None:
        lat = next((r["payload"].get("fence_latch") or {} for r in _rung_runs(obs, long_rung)), {})
        lb, la = lat.get("before") or {}, lat.get("after") or {}
        if lb and la:
            rows += [f"Draining the deferred-fence reservoir at {long_rung} through the app's own "
                     f"`beforeprint` path latched **{lat.get('fences_latched')}** fences and added "
                     f"**{(lat.get('elements_added') or 0):,}** elements, of which "
                     f"{(lat.get('highlight_spans_added') or 0):,} are `pre span`, while `pre` "
                     f"stayed at {la.get('code_blocks')} and `.katex` at "
                     f"{(la.get('katex_roots') or 0):,}. That is the 11,205-element growth run "
                     f"32865232787 could not name, and it is now paid before anything is scored "
                     f"rather than by whichever arm happened to change the layout height -- which "
                     f"matters more for this fix than for any ablation before it, because "
                     f"`contain-intrinsic-size` changes the layout height BY DESIGN.", ""]

        cen = next((r["payload"].get("baseline_census") or {}
                    for r in _rung_runs(obs, long_rung)), {})
        pos = next((r["payload"].get("positioned") or {}
                    for r in _rung_runs(obs, long_rung)), {})
        if cen:
            rows += [f"The page at {long_rung}: {(cen.get('elements') or 0):,} elements, "
                     f"{cen.get('messages')} messages, {(cen.get('katex_roots') or 0):,} `.katex` "
                     f"roots of which {cen.get('katex_display')} are `.katex-display`, "
                     f"{(cen.get('katex_descendants') or 0):,} descendants of which "
                     f"{(cen.get('katex_display_descendants') or 0):,} are under display maths, "
                     f"{cen.get('math_blocks')} maths-bearing blocks, "
                     f"{(cen.get('code_blocks') or 0):,} code blocks, scroller "
                     f"{(cen.get('scroller_scroll_height') or 0):,} px tall.", ""]
        if pos:
            dd = pos.get("katex_display_descendants") or {}
            ii = pos.get("katex_inline_descendants") or {}
            dr = pos.get("katex_display_roots") or {}
            ir = pos.get("katex_inline_roots") or {}
            rows += ["**The split the whole fix turns on.** Sampled positioned elements "
                     "(`position != static`, which is exactly what "
                     "`RenderBoxModelObject::requiresLayer()` keys on):", "",
                     f"- under DISPLAY maths, which `content-visibility` CAN reach: "
                     f"{dd.get('non_static')}/{dd.get('sampled')} sampled, "
                     f"~{(dd.get('estimate') or 0):,} of {(dd.get('total') or 0):,} descendants, "
                     f"plus ~{(dr.get('estimate') or 0):,} of {(dr.get('total') or 0):,} roots",
                     f"- under INLINE maths, which it provably CANNOT: {ii.get('non_static')}/"
                     f"{ii.get('sampled')} sampled, ~{(ii.get('estimate') or 0):,} of "
                     f"{(ii.get('total') or 0):,} descendants, plus ~{(ir.get('estimate') or 0):,} "
                     f"of {(ir.get('total') or 0):,} roots", ""]
    return "\n".join(rows)


def _display_share(obs: dict, rung: str | None) -> tuple[float | None, int, int]:
    """(share of positioned maths boxes under DISPLAY maths, display estimate, inline estimate)."""
    if rung is None:
        return None, 0, 0
    pos = next((r["payload"].get("positioned") or {} for r in _rung_runs(obs, rung)), {})
    dd = (pos.get("katex_display_descendants") or {}).get("estimate") or 0
    ii = (pos.get("katex_inline_descendants") or {}).get("estimate") or 0
    tot = dd + ii
    return ((dd / tot) if tot else None), dd, ii


def _per_rep(entries: list[dict]) -> str:
    out = []
    for e in entries:
        sv = e.get("saving")
        out.append(f"rep {e.get('rep')}: {e.get('bmpf')} ms blocked/frame against "
                   f"{e.get('base_bmpf'):.1f} baseline"
                   f"{'' if sv is None else f' ({sv:+.0%})'}"
                   if e.get("base_bmpf") else f"rep {e.get('rep')}: {e.get('bmpf')} ms")
    return "; ".join(out) or "no per-rep figures"


def verdict(obs: dict) -> tuple[str, str]:
    # A failed gate means "we could not tell", which is not the same as "the fix does not work",
    # and the two must never be reported as each other.
    bad = [g for g in _gate_records(obs) if not g["ok"]]
    if bad:
        return "INCONCLUSIVE", (
            "this run could not tell, because "
            + ("; ".join(f'the gate "{g["name"]}" failed ({g["evidence"][:200]})' for g in bad[:3]))
            + (f"; and {len(bad) - 3} further gates" if len(bad) > 3 else "")
            + ". That is not the same as the fix not working, and no arm number below a failed "
              "instrument gate is readable")

    rung = _long_rung(obs)
    if rung is None:
        return "INCONCLUSIVE", ("no rung carried a scrollable range, so nothing was scored. The "
                                "short rungs can bound what this rule COSTS, never what it BUYS")

    ship = _entries(obs, rung, SHIPPABLE)
    dq = _disqualified(SHIPPABLE, ship) if ship else "the shippable arm produced no window"
    sv = _saving(obs, rung, SHIPPABLE)
    wr = _mean([e["work_ratio"] for e in ship if e.get("work_ratio") is not None])
    sh = _mean([e["scroll_height_delta"] for e in ship if e.get("scroll_height_delta") is not None])
    ub = _saving(obs, rung, UPPER_BOUND)
    alls = _saving(obs, rung, ALL_SELECTOR)
    expl = _saving(obs, rung, EXPLORATORY)
    expl_dq = _disqualified(EXPLORATORY, _entries(obs, rung, EXPLORATORY))
    share, dd, ii = _display_share(obs, rung)
    shorts = _short_rungs(obs)

    bound = (f"The {PUBLISHED_UPPER_SAVING:.0%} upper bound comes from "
             f"`.katex{{visibility:hidden}}`, which is a PROBE and not shippable: it removes the "
             f"cost by deleting the maths from the page"
             + (f" (re-measured here at {ub:+.0%})" if ub is not None else ""))
    structural = (
        f"the fix cannot reach that bound for a structural reason, counted rather than assumed: "
        f"only {share:.0%} of the positioned boxes under maths live under DISPLAY maths "
        f"(~{dd:,} against ~{ii:,} under inline maths), and `content-visibility` needs size "
        f"containment, which does not apply to a non-atomic inline-level box, so the ~{ii:,} "
        f"inline ones are out of reach of any selector"
        if share is not None else
        "the positioned split between display and inline maths was not recorded, so the "
        "structural ceiling on this fix cannot be stated from this run")
    short_txt = (f"At the short rung{'s' if len(shorts) > 1 else ''} "
                 f"({', '.join(shorts)}) the selector matches nothing -- zero `.katex-display` "
                 f"roots -- so the rule cannot cost anything there"
                 if shorts else "No short rung was measured, so the cost at short context is not "
                                "established by this run")
    exploratory_txt = (
        f"`{EXPLORATORY}` (EXPLORATORY, NOT SHIPPABLE AS WRITTEN: it hoists the declaration to a "
        f"block ancestor, which is what a renderer-side change would have to do) reads "
        + ("no window" if expl is None else f"{expl:+.0%}")
        + (f", but it is DISQUALIFIED ({expl_dq})" if expl_dq else "")
        + ", which is what reaching inline maths would be worth")

    if ship and not dq and sv is not None and sv >= ARM_MIN_SAVING:
        return "HELPS", (
            f"the shippable rule WORKS at {rung}. `{SHIPPABLE}` removes {sv:.0%} of the "
            f"per-scroll-event cost"
            + (f" ({wr:.2f}x less blocked time per 1 px scroll)" if wr else "")
            + f", scored against its own two neighbouring baselines, with the negative control "
              f"flat and the positive control recovering. Per rep: {_per_rep(ship)}. "
            + bound + ", and " + structural + ". Adding `.katex` to the selector bought "
            + ("no window" if alls is None else f"{alls - sv:+.1%}")
            + f", which is the measured form of that claim. The rule moves scrollHeight by "
            + ("no recorded amount" if sh is None else f"{sh:+.2%}")
            + f", which is `contain-intrinsic-size` standing in for the height of blocks the "
              f"engine skipped and is inside the {MAX_SCROLL_HEIGHT_DELTA:.0%} gate that says the "
              f"gesture was not pinned. " + short_txt + ". For scale, " + exploratory_txt)

    why = dq or (f"it removes only {sv:+.0%}, under the {ARM_MIN_SAVING:.0%} bar"
                 if sv is not None else "it produced no scored window")
    return "NO_BENEFIT", (
        f"the shippable rule DOES NOT WORK at {rung}: `{SHIPPABLE}` cannot carry a claim because "
        f"{why}. Per rep: {_per_rep(ship)}. " + bound + ", and " + structural
        + ". Adding `.katex` to the selector "
        + ("produced no window" if (alls is None or sv is None)
           else f"changed the reading by {alls - sv:+.1%}")
        + ". The rule moves scrollHeight by "
        + ("no recorded amount" if sh is None else f"{sh:+.2%}")
        + ". " + short_txt + ". Instead, " + exploratory_txt
        + ". This is a measured absence of benefit at the rung where the defect lives, not an "
          "absence of measurement: the controls behave and the reference upper bound reproduced")


def observed_capabilities(obs: dict) -> dict[str, bool]:
    d = obs.get("dist") or {}
    return {
        "webkitgtk": bool(_runs(obs)),
        "headless_display_server": bool((obs.get("xserver") or {}).get("display")),
        "studio_production_bundle": bool(d.get("index_html")) and (d.get("asset_files") or 0) > 0,
        "studiobench_ladder": len(_by_rung(obs)) >= 1,
    }
