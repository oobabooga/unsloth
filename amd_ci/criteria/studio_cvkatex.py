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

THE REFERENCE UPPER BOUND IS NOT A CANDIDATE. `katex_root_visibility_hidden` is re-measured in the
same session for one reason: to say whether this session reproduces the 97% that run 32869180652
measured. If it does not, the candidates cannot be read against that published number, because the
session is not the same experiment. It is gated as such, and it is never allowed to carry a
verdict, because it is not shippable. It now runs TWICE, early and late in the sequence
(`katex_root_visibility_hidden` and `katex_root_visibility_hidden_late`), because run 32876363634
read 55% from it at r500K where the published run read 97% and the only thing that visibly differed
was where in the sequence it ran. The two readings are compared per rung and reported: a
disagreement between them is a statement about the SESSION, not about the arm.

THE MEAN IS PRIMARY AND THE MEAN IS FRAGILE, AND BOTH OF THOSE ARE TRUE AT ONCE. On this venue at
r500K the app blocks the main thread for about 8.6 SECONDS every 30 SECONDS. Re-scored from the
untouched rAF gap series of runs 32869180652 and 32876363634, every window longer than about 3 s
catches exactly one of those stalls and every window shorter than that catches none. Because
`blocked_ms_per_frame` is a MEAN, one 8.6 s frame spread over ~90 frames adds about 90 ms per
frame on its own. `.katex{visibility:hidden}` completes in 1.8 s when it works, so the published
run missed the stall and read 10.0 ms per frame while the next session caught one and read 112.9 --
an 11x difference produced by a SINGLE FRAME, with both sessions reporting an identical p50 of
17 ms and p95 of 18 ms. Drop the single worst frame from each window and the two sessions agree to
within 2% on every window. The stall does not occur at r100K (0 frames over 1 s in 1,392 frames in
the same session) and does not occur on the local llvmpipe rig at all. So the scene records a
`robust` figure per window with its single worst tick and worst rAF gap removed, this file reports
it ALONGSIDE the mean rather than instead of it, and a stall-dominated window is disqualified.
Recording that quantity and never gating on it would be defect #48 (a measured quantity nobody
checks) on top of defect #8 (a metric unstable by construction).

WHAT COUNTS AS STALL-DOMINATED TOOK ONE RUN TO CALIBRATE, and the first version of it was wrong in
a way worth keeping written down. Run 32882865551 failed on `still_no_scroll` at r100K reading a
mean of 0.5 ms/frame against a robust 0.2, a ratio of 2.5x -- while its own evidence line said ZERO
frames over one second and a worst gap of 48 ms. A 48 ms frame is not an 8.6 s stall, and a ratio
between two numbers a fifth of a millisecond apart is arithmetic, not a measurement. The condition
now requires all three of: at least STALL_MIN_FRAMES_OVER_1S frame over one second (0 against the
real stall's 1, with 48 ms against 8,338-8,930 ms, a factor of about 174 of daylight); a mean of at
least STALL_MIN_MEAN_MS (the false positive is 0.5, twenty times below it, and the smallest genuine
stalled window in that run is 102.2, twenty times above it); and the MEAN_OVER_ROBUST_MAX ratio.

WHOSE WINDOW IT IS DECIDES WHETHER THE RUN DIES, and that is a deliberate change of blast radius
rather than a loosening. A stall-dominated window is disqualified exactly as a dropped declaration
or a moved scrollHeight is. It fails the RUN only when the window carries a conclusion: a CANDIDATE
arm, either window of the reference upper bound, or a BASELINE, since a contaminated baseline
contaminates every arm scored against it. For a CONTROL (`still_no_scroll`, `noop_touch`,
`detach_messages`) the disqualification is printed and the control's own gate is then evaluated on
the ROBUST figure, which is the same statistic with the app's periodic stall removed, and the
evidence says so. The alternative is discarding a 26 minute exclusive GPU run because the floor arm
caught a stall it has no control over: on this host at r500K essentially every window longer than
about three seconds catches one, so exempting nothing voids most 500K runs at random, which is a
worse instrument than the one it replaces.

THE FLOOR IS A FLOOR ON THE GESTURE, NOT ON EVERY ARM. `still_no_scroll` measures what its name
says -- its MEDIAN rAF gap is 17 ms at r500K in both reps, identical to the positive control's 16
and the reference arm's 17, against 246-250 ms for the baselines -- but it is not a lower bound
that no arm may pass. `Element::setScrollTop` calls
`document->updateLayoutIgnorePendingStylesheets(...)` BEFORE it does anything else
(Source/WebCore/dom/Element.cpp:1808-1813), so assigning the SAME scrollTop still forces a full
synchronous document layout every frame, and that flush scales with DOM size: the floor reads 0.2
ms/frame robust at r100K and 10.3 at r500K, while `detach_messages` reads 0.9 and 1.9 because its
DOM is two messages. An arm that also deletes the DOM can legitimately go BELOW the floor, so the
reported recovery is capped at 100% with the raw ratio printed beside it, and that is stated as
expected rather than reported as an instrument fault.

p50 IS THE ONE STATISTIC A STALL CANNOT MOVE, and on this host it settled a question the mean got
wrong. At r500K in run 32882865551 the baselines read 246-250 ms, `katex_root_visibility_hidden` 17
early and 17 late, `content_visibility_katex_display` 216-217, `content_visibility_katex_all`
217-218, `content_visibility_math_blocks` 29, `still_no_scroll` 17, `detach_messages` 16. The two
`content-visibility` arms are the SAME on p50 (216 against 217) while the mean and the robust
figure disagreed by 30 points purely on where stalls landed. So p50 is a column in every table, it
is reported per candidate against its own neighbouring baselines, and the "adding `.katex` buys
nothing" comparison is decided ON p50 at the unchanged 5 percentage point threshold.

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
# is kept in the sequence only to say whether this session reproduces the published 97%. Renamed
# from `visibility_hidden_offscreen`, which in `wq_final.js` names a DIFFERENT arm that hides
# off-screen MESSAGES: two arms sharing one label is defect #40, and the label is what a report
# quotes.
UPPER_BOUND = "katex_root_visibility_hidden"
# The same arm, run a second time late in the sequence. Not a candidate either. It exists to
# separate "the arm reads differently depending on what ran before it" from "this session differs
# from the published one", which run 32876363634 could not do.
UPPER_BOUND_LATE = "katex_root_visibility_hidden_late"
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
# How far a window's MEAN may sit above the same window with its single worst frame removed. Two
# times is already enormous: it means one frame out of ninety is worth as much as the other
# eighty-nine put together, which on this venue means the window caught one of the app's 8.6 s
# stalls and is reporting where that landed rather than what the arm did.
MEAN_OVER_ROBUST_MAX = 2.0
# ... but a ratio alone fired on `still_no_scroll` at r100K in run 32882865551: mean 0.5 ms/frame
# against a robust 0.2, 2.5x, with ZERO frames over one second and a worst gap of 48 ms. Two more
# conditions have to hold, and each one alone separates that window from a real stall.
#
# The window must contain a frame over one second. The false positive had none; the genuine ones
# had one apiece, at 8,338-8,930 ms. That is a factor of about 174 between 48 ms and the stall, so
# this discriminator is not close to its threshold in either direction.
STALL_MIN_FRAMES_OVER_1S = 1
# And the mean must be large enough for a ratio to mean anything. Below this, a "2.5x" is two
# numbers a fifth of a millisecond apart. The false positive is 0.5 ms/frame, twenty times below
# this floor; the smallest genuinely stalled window in that run is 102.2, twenty times above it.
STALL_MIN_MEAN_MS = 5.0
# REPORTING ONLY, gates nothing: how far the early and late readings of the reference upper bound
# may diverge before the report calls them a disagreement about the session.
EARLY_LATE_MAX_RATIO = 2.0
# How much of a selector to quote in a table cell.
SELECTOR_CHARS = 46

# Run 32869180652, this venue, 500K, same scene minus the three new arms.
PUBLISHED_RUN = "32869180652"
PUBLISHED_BASE_MS, PUBLISHED_BASE_FPS = 287.6, 3.2
PUBLISHED_UPPER_MS, PUBLISHED_UPPER_FPS, PUBLISHED_UPPER_SAVING = 10.0, 61.0, 0.97
# Run 32876363634, the session that caught the stalls: the same upper-bound arm read 55% at r500K
# and 90% at r100K, with a floor of 6.0 ms and a positive control of 2.0 ms.
STALL_RUN = "32876363634"


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


def _p50(a: dict):
    """The window's median rAF gap. One stalled frame cannot move a median."""
    return (a.get("raf") or {}).get("p50_ms")


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
            base_p = _neighbour_baseline(seq, i, _p50)
            b, f, p = _bmpf(a), a.get("eff_fps"), _p50(a)
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
                # p50 AND ITS OWN SAVING. A median cannot be moved by one stalled frame, which is
                # why the `.katex` comparison is decided on this and not on the mean.
                "p50": p, "base_p50": base_p,
                "p50_saving": (1 - p / base_p) if (p is not None and base_p) else None,
                "fired": a.get("fired"),
                # What the arm actually applied, quoted rather than summarised by its label.
                "selector": a.get("selector") or a.get("apply_detail"),
                # The same window with its single worst frame removed. Reported ALONGSIDE the
                # mean, never instead of it.
                "robust": a.get("robust") or {},
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


def _p50_saving(obs: dict, rung: str | None, name: str):
    """The same saving computed on the MEDIAN rAF gap, which one stalled frame cannot move."""
    return _mean([e["p50_saving"] for e in _entries(obs, rung, name)
                  if e.get("p50_saving") is not None])


def _p50_of(obs: dict, rung: str | None, name: str):
    v = _mean([e["p50"] for e in _entries(obs, rung, name) if e.get("p50") is not None])
    return "-" if v is None else f"{v:.0f}"


def _base_p50_of(obs: dict, rung: str | None, name: str):
    v = _mean([e["base_p50"] for e in _entries(obs, rung, name) if e.get("base_p50") is not None])
    return "-" if v is None else f"{v:.0f}"


def _selector_of(window_or_entry: dict) -> str | None:
    return window_or_entry.get("selector") or window_or_entry.get("apply_detail")


def _sel_txt(x, chars: int = SELECTOR_CHARS) -> str:
    """Quote what the arm applied. A label is a claim about a selector; the selector is the fact."""
    s = str(x or "").strip().replace("|", "/").replace("\n", " ")
    if not s:
        return "-"
    return "`" + (s if len(s) <= chars else s[:chars - 3] + "...") + "`"


def _arm_selector(entries: list[dict]) -> str | None:
    for e in entries:
        s = _selector_of(e)
        if s:
            return s
    return None


def _rb(window_or_entry: dict) -> dict:
    """The window with its single worst tick and worst rAF gap removed."""
    r = window_or_entry.get("robust")
    return r if isinstance(r, dict) else {}


CONTROL_WINDOWS = (FLOOR, NEGATIVE, POSITIVE)

_STALL_WHY = ("This host blocks the main thread for about 8.6 s every 30 s, and a mean over ~90 "
              "frames turns one such frame into ~90 ms/frame of apparent cost, so such a window "
              "is reporting where a stall landed rather than what it was measuring")

# Said in full every time a control gate switches statistic, because a number quietly computed a
# different way from the one beside it is how a report stops being readable.
_ROBUST_NOTE = (". NOTE: the {what} is read on its ROBUST figure, with its single worst frame "
                "dropped, because that window caught one of this host's ~8.6 s main-thread stalls "
                "and its mean is a report of where the stall landed. It is the same statistic over "
                "the same window minus one frame, and the alternative is voiding the whole run "
                "over a stall the control cannot influence")


def _stall_dominated(a: dict) -> str | None:
    """Is this window's MEAN a report of where one 8.6 s stall landed? Returns why, or None.

    All three conditions, because the ratio alone fired on a 0.5 ms/frame window with no frame
    over one second and a 48 ms worst gap. See the constants for where each number came from.
    """
    rb = _rb(a)
    # Accepts a raw scene window (`blocked_ms_per_frame`) or a scored entry (`bmpf`), because the
    # gate reads raw windows and the disqualifier reads entries, and they must agree.
    m = a.get("blocked_ms_per_frame")
    if m is None:
        m = a.get("bmpf")
    r = rb.get("blocked_ms_per_frame")
    if m is None or r is None:
        return None
    stalls = rb.get("stall_frames_over_1s")
    if stalls is None or stalls < STALL_MIN_FRAMES_OVER_1S:
        return None
    if m < STALL_MIN_MEAN_MS:
        return None
    if m <= MEAN_OVER_ROBUST_MAX * r:
        return None
    ratio = "infinitely more" if not r else f"{m / r:.1f}x"
    # The FACTS only. The explanation is `_STALL_WHY`, said once per gate rather than once per
    # window: this fires on several windows at a time and a paragraph repeated six times is how a
    # report stops being read.
    return (f"its mean is one frame: {m} ms/frame against {r} ms/frame with the single worst "
            f"frame dropped ({ratio}), {stalls} frame{'' if stalls == 1 else 's'} over 1 s, worst "
            f"gap {rb.get('worst_gap_ms')} ms")


def _carries_conclusion(a: dict) -> bool:
    """Would a stall in THIS window corrupt a conclusion, or only a control?

    A baseline counts, and counts hardest: every arm is scored against its neighbouring baselines,
    so one contaminated baseline contaminates the arms on both sides of it.
    """
    name, arm = a.get("name"), a.get("arm")
    if arm == BASELINE_ARM:
        return True
    if name in (UPPER_BOUND, UPPER_BOUND_LATE) or arm in (UPPER_BOUND, UPPER_BOUND_LATE):
        return True
    return name in CANDIDATES or arm in CANDIDATES


def _control_bmpf(obs: dict, rung: str, name: str) -> tuple[float | None, bool]:
    """A control's cost, on the ROBUST figure if its window caught a stall. (value, used_robust).

    A control that caught a stall is disqualified as a reading of the mean, but the same window
    with the stall removed is still a measurement of the same thing, and throwing away a 26 minute
    exclusive GPU run because the floor arm caught a stall it cannot influence is a worse trade.
    """
    vals, used = [], False
    for _, seq in _windows(obs, rung):
        for a in seq:
            if a.get("name") != name:
                continue
            if _stall_dominated(a):
                v = _rb(a).get("blocked_ms_per_frame")
                used = True
            else:
                v = _bmpf(a)
            if v is not None:
                vals.append(v)
    return _mean(vals), used


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
    if name == UPPER_BOUND_LATE:
        return " **[REFERENCE UPPER BOUND, RE-RUN LATE, NOT SHIPPABLE]**"
    if name == EXPLORATORY:
        return " **[EXPLORATORY, NOT SHIPPABLE AS WRITTEN]**"
    if name == ALL_SELECTOR:
        return " **[SELECTOR PROBE, NOT A FIX]**"
    return {POSITIVE: " **[POSITIVE CONTROL]**", NEGATIVE: " **[NEGATIVE CONTROL]**",
            FLOOR: " **[FLOOR]**", BASELINE_REPEAT: " **[DRIFT CHECK]**"}.get(name, "")


# ── per-arm disqualifiers ─────────────────────────────────────────────────────────────────────

def _disqualified(name: str, entries: list[dict]) -> str | None:
    """Reasons this ARM cannot be scored. Not run-level: the other arms are unaffected."""
    # A STALL DISQUALIFIES ANY WINDOW, controls included, because the mean it produced is a
    # statement about where the stall landed. Whether it also fails the RUN is a separate
    # question, decided by `_carries_conclusion` in the gate: a control that caught one is
    # re-read on its robust figure rather than throwing the whole run away.
    for e in entries:
        why = _stall_dominated(e)
        if why:
            return (f"the window caught one of this host's main-thread stalls, so {why}. "
                    + _STALL_WHY
                    + (f". As a CONTROL its own gate is re-evaluated on the robust figure "
                       f"({_rb(e).get('blocked_ms_per_frame')} ms/frame) rather than voiding the "
                       f"run" if name in CONTROL_WINDOWS else ""))
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

    # THE MEAN MUST NOT BE ONE FRAME -- WHERE A CONCLUSION DEPENDS ON IT. On this venue at r500K
    # the app blocks the main thread for about 8.6 s every 30 s, and `blocked_ms_per_frame` is a
    # MEAN, so one such frame over ~90 frames adds ~90 ms per frame by itself. That is how run
    # 32869180652 read 10.0 ms for the upper-bound arm and run 32876363634 read 112.9 for the same
    # arm on the same corpus, both at a p50 of 17 ms.
    #
    # THE BLAST RADIUS IS DELIBERATE. Every stall-dominated window is DISQUALIFIED and printed.
    # Only a window that carries a conclusion -- a candidate arm, either reference upper-bound
    # window, or a baseline -- fails the RUN, because a contaminated baseline contaminates every
    # arm scored against it. A CONTROL that catches one has its own gate re-evaluated on the
    # robust figure instead, which is the same statistic with the stall removed. Run 32882865551
    # is why: `still_no_scroll` at r500K read 102.3 mean against 10.3 robust with one 8,595 ms
    # frame, and voiding a 26 minute exclusive GPU run because the FLOOR arm caught a stall it
    # cannot influence is a worse instrument than reading the floor without that frame. At r500K
    # essentially every window over about three seconds catches one, so exempting nothing voids
    # most 500K runs at random.
    def _stall(rung, rs):
        fatal, controls = [], []
        for rep_i, seq in _windows(obs, rung):
            for a in seq:
                why = _stall_dominated(a)
                if not why:
                    continue
                line = f"rep {rep_i} `{a.get('name')}`: {why}"
                if _carries_conclusion(a):
                    fatal.append(line)
                else:
                    controls.append(f"rep {rep_i} `{a.get('name')}` (CONTROL, disqualified but not "
                                    f"fatal; its gate is re-read on "
                                    f"{_rb(a).get('blocked_ms_per_frame')} ms/frame): {why}")
        txt = "; ".join(fatal + controls)
        if not fatal and not controls:
            return True, "every window's mean survives dropping its worst frame"
        txt += ". " + _STALL_WHY
        if not fatal:
            return True, ("no conclusion-carrying window caught a stall, but: " + txt)
        return False, txt

    g(f"no window that carries a conclusion has a mean dominated by a single frame (a candidate "
      f"arm, either upper-bound window, or a baseline; mean >= {STALL_MIN_MEAN_MS:.0f} ms/frame "
      f"and > {MEAN_OVER_ROBUST_MAX:.0f}x its robust figure with >= "
      f"{STALL_MIN_FRAMES_OVER_1S} frame over 1 s)",
      *_per_rung(obs, _stall, scored_only = True))

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
        floor, floor_robust = _control_bmpf(obs, rung, FLOOR)
        base = _mean([_bmpf(x) for _, seq in _windows(obs, rung) for x in seq
                      if x.get("arm") == BASELINE_ARM])
        if base is None or floor is None:
            return False, f"baseline={base} floor={floor}"
        return (base > max(2.0, floor * 3),
                f"baseline {base:.1f} ms blocked per scroll event against a `still` floor of "
                f"{floor:.1f} ms" + (_ROBUST_NOTE.format(what = "floor") if floor_robust else ""))

    g("the baseline still exhibits the collapse (a 1 px scroll costs far more than assigning the "
      "same scrollTop)", *_per_rung(obs, _collapse, scored_only = True))

    def _neg(rung, rs):
        n, n_robust = _control_bmpf(obs, rung, NEGATIVE)
        base = _mean([e["base_bmpf"] for e in _entries(obs, rung, NEGATIVE)
                      if e.get("base_bmpf") is not None])
        s = (1 - n / base) if (n is not None and base) else None
        return (s is not None and s <= NEGATIVE_MAX_SAVING,
                (f"noop_touch {s:+.0%} ({n:.1f} ms/frame against {base:.1f} on its two "
                 f"neighbouring baselines)" + (_ROBUST_NOTE.format(what = "negative control")
                                               if n_robust else ""))
                if s is not None else "no negative control window")

    g(f"NEGATIVE control removes <= {NEGATIVE_MAX_SAVING:.0%} of the cost",
      *_per_rung(obs, _neg, scored_only = True))

    # THE FLOOR IS NOT A LOWER BOUND ON EVERY ARM. `Element::setScrollTop` calls
    # `document->updateLayoutIgnorePendingStylesheets(...)` before it does anything else
    # (Source/WebCore/dom/Element.cpp:1808-1813), so assigning the SAME scrollTop still forces a
    # full synchronous document layout every frame, and that flush scales with DOM size: the floor
    # reads 0.2 ms/frame robust at r100K and 10.3 at r500K while `detach_messages` reads 0.9 and
    # 1.9, because its DOM is two messages. An arm that DELETES the DOM can therefore legitimately
    # go below the floor and produce a recovery over 100%, which is a property of the arithmetic
    # and not an instrument fault. Reported capped, with the raw ratio beside it.
    def _pos(rung, rs):
        floor, floor_robust = _control_bmpf(obs, rung, FLOOR)
        p, p_robust = _control_bmpf(obs, rung, POSITIVE)
        base = _mean([_bmpf(x) for _, seq in _windows(obs, rung) for x in seq
                      if x.get("arm") == BASELINE_ARM])
        if p is None or base is None or floor is None or base <= floor:
            return False, "no positive control window"
        raw = (base - p) / (base - floor)
        rec = min(raw, 1.0)
        note = ""
        if raw > 1.0:
            note = (f" (capped from a raw {raw:.0%}: the positive control went BELOW the floor, "
                    f"which is expected rather than an instrument fault -- `still_no_scroll` still "
                    f"pays a full synchronous layout flush per frame via "
                    f"`updateLayoutIgnorePendingStylesheets` and that flush scales with DOM size, "
                    f"while this arm has deleted all but two messages)")
        for what, used in (("floor", floor_robust), ("positive control", p_robust)):
            if used:
                note += _ROBUST_NOTE.format(what = what)
        return (rec >= POSITIVE_MIN_RECOVERY,
                f"detach_messages {p:.1f} ms blocked/frame against baseline {base:.1f} and floor "
                f"{floor:.1f} = {rec:.0%}" + note)

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

    # THE HEADLINE COMPARISON, DECIDED ON p50. The two arms should read the SAME if inline maths is
    # out of reach. It is decided on the median rather than the mean because a median cannot be
    # moved by one stalled frame and the mean can: in run 32882865551 the two arms sat at 216 and
    # 217 ms on p50 -- the same arm, twice -- while their means disagreed by 30 points purely on
    # where the stalls happened to land. The 5 percentage point threshold is unchanged.
    disp, alls = _p50_saving(obs, long_rung, SHIPPABLE), _p50_saving(obs, long_rung, ALL_SELECTOR)
    disp_mean, alls_mean = _saving(obs, long_rung, SHIPPABLE), _saving(obs, long_rung, ALL_SELECTOR)
    if long_rung is None:
        excess_txt = ("no rung carried a scrollable range, so neither arm has a window and the "
                      "comparison this run exists for was never made")
        excess_ok = False
    elif disp is None or alls is None:
        excess_txt = (f"missing a p50 reading at {long_rung}: `{SHIPPABLE}`="
                      f"{'-' if disp is None else format(disp, '+.0%')}, `{ALL_SELECTOR}`="
                      f"{'-' if alls is None else format(alls, '+.0%')}")
        excess_ok = False
    else:
        excess = alls - disp
        te_d, te_a = _took_effect(_entries(obs, long_rung, SHIPPABLE)), \
            _took_effect(_entries(obs, long_rung, ALL_SELECTOR))
        excess_ok = excess <= KATEX_ALL_MAX_EXCESS
        excess_txt = (f"ON p50: `{SHIPPABLE}` {disp:+.0%} "
                      f"({_p50_of(obs, long_rung, SHIPPABLE)} ms median rAF gap against "
                      f"{_base_p50_of(obs, long_rung, SHIPPABLE)} on its neighbouring baselines) "
                      f"against `{ALL_SELECTOR}` {alls:+.0%} "
                      f"({_p50_of(obs, long_rung, ALL_SELECTOR)} ms against "
                      f"{_base_p50_of(obs, long_rung, ALL_SELECTOR)}) at {long_rung}, a difference "
                      f"of {excess:+.1%}. On the MEAN, which one stalled frame can move, the same "
                      f"pair reads "
                      + ("-" if disp_mean is None else f"{disp_mean:+.0%}") + " against "
                      + ("-" if alls_mean is None else f"{alls_mean:+.0%}")
                      + (f"; height probe on the display roots {te_d.get('changed')}/"
                         f"{te_d.get('compared')} changed" if te_d else "")
                      + (f", on the INLINE roots {te_a.get('changed')}/{te_a.get('compared')} "
                         f"changed" if te_a else ""))
        if not excess_ok:
            excess_txt += (f". Adding `.katex` bought MORE than {KATEX_ALL_MAX_EXCESS:.0%} on the "
                           f"metric a stall cannot corrupt, so the spec reading is WRONG: size "
                           f"containment is reaching inline maths here and the shipped rule is "
                           f"leaving something on the table")
    g("adding `.katex` to the selector buys nothing (on p50, the metric a stall cannot move), "
      "which is what says inline maths is out of reach", excess_ok, excess_txt,
      instrument = False)

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

    def raw(name: str) -> list[dict]:
        return [a for _, seq in _windows(obs, rung) for a in seq if a.get("name") == name]

    rows = [f"### Rung {rung}", "",
            f"**blocked ms/frame is the MEAN and stays the primary metric.** The column beside it "
            f"is the SAME window with its single worst tick and worst rAF gap removed, reported "
            f"ALONGSIDE the mean and never instead of it, because on this venue at r500K the app "
            f"blocks the main thread for about 8.6 s every 30 s and one such frame spread over "
            f"~90 frames is worth ~90 ms/frame on its own. A window that caught one is "
            f"DISQUALIFIED, and fails the run only if it carries a conclusion; a CONTROL that "
            f"caught one is re-read on its robust figure instead. `stalls > 1 s` is the count of "
            f"frames over one second in the window, and **p50 is the median rAF gap, which one "
            f"stalled frame cannot move** -- where p50 and the mean disagree, p50 is the one to "
            f"read.", "",
            "| window | selector | blocked ms/frame (MEAN, primary) | robust, worst frame dropped "
            "| p50 rAF gap | stalls > 1 s | vs its two neighbouring baselines | cost removed | "
            "fps | busy | worst frame | scrollHeight | elements mounted | declaration accepted | "
            "engine acted |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name in order:
        es = scored.get(name)
        ws = raw(name)
        sel = _sel_txt(next((_selector_of(a) for a in ws if _selector_of(a)), None))
        rb_vals = [_rb(a).get("blocked_ms_per_frame") for a in ws]
        rb_txt = ", ".join(f"{v:.1f}" for v in rb_vals if v is not None) or "-"
        st_vals = [_rb(a).get("stall_frames_over_1s") for a in ws]
        st_txt = ", ".join(str(v) for v in st_vals if v is not None) or "-"
        p50_txt = ", ".join(f"{_p50(a):.0f}" for a in ws if _p50(a) is not None) or "-"
        if es is None:
            vals = [_bmpf(a) for a in ws]
            fps = [a.get("eff_fps") for a in ws]
            mut = [(a.get("mutations") or {}).get("added_elements") or 0 for a in ws]
            stalled = any(_stall_dominated(a) for a in ws)
            rows.append(f"| `{name}` (baseline)"
                        + ("  **[DISQUALIFIED: caught a stall]**" if stalled else "")
                        + f" | {sel} | "
                        + (", ".join(f"{v:.1f}" for v in vals if v is not None) or "-")
                        + f" | {rb_txt} | {p50_txt} | {st_txt} | - | - | "
                        + (", ".join(f"{v:.1f}" for v in fps if v is not None) or "-")
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
            sel,
            "-" if b is None else f"**{b:.1f}**",
            rb_txt,
            p50_txt,
            st_txt,
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
        ws = raw(name)
        sel = _sel_txt(next((_selector_of(a) for a in ws if _selector_of(a)), None))
        vals = ([e["scroll_height_delta"] for e in es] if es
                else [a.get("scroll_height_delta") for a in ws])
        vals = [v for v in vals if v is not None]
        if not vals:
            rows.append(f"- `{name}` ({sel}): not recorded")
            continue
        worst = max(vals, key = abs)
        over = abs(worst) > MAX_SCROLL_HEIGHT_DELTA
        exempt = name in (POSITIVE, NEGATIVE, FLOOR, BASELINE_REPEAT)
        rows.append(f"- `{name}`{_label(name)} ({sel}): "
                    + ", ".join(f"{v:+.2%}" for v in vals)
                    + ("" if not over else
                       ("  <- over the gate, and exempt from it: this arm changes the thread on "
                        "purpose" if exempt else "  <- over the gate")))

    dqs = [(n, _disqualified(n, es)) for n, es in scored.items() if _disqualified(n, es)]
    if dqs:
        rows += ["", "Disqualified arms, and why (the other arms are unaffected):", ""]
        rows += [f"- `{n}`{_label(n)} ({_sel_txt(_arm_selector(scored[n]))}): {why}"
                 for n, why in dqs]

    rows += _p50_lines(obs, rung)
    rows += _early_late_lines(obs, rung)

    # Per repetition, because a mean over a bimodal draw is a report of how many slow draws
    # happened to land in that arm. The robust and p50 columns are repeated here per rep because a
    # stall lands in ONE repetition, and a mean over two reps is exactly where that disappears.
    rows += ["", "Per repetition, so a mean cannot hide a disagreement:", "",
             "| window | selector | blocked ms/frame per rep | robust per rep | p50 per rep | "
             "stalls > 1 s per rep | fps per rep | frames per rep |",
             "|---|---|---|---|---|---|---|---|"]
    for name in order:
        ws = raw(name)
        sel = _sel_txt(next((_selector_of(a) for a in ws if _selector_of(a)), None))
        rows.append(f"| `{name}` | {sel} | "
                    + ", ".join(str(_bmpf(a)) for a in ws) + " | "
                    + ", ".join(str(_rb(a).get("blocked_ms_per_frame")) for a in ws) + " | "
                    + ", ".join(str(_p50(a)) for a in ws) + " | "
                    + ", ".join(str(_rb(a).get("stall_frames_over_1s")) for a in ws) + " | "
                    + ", ".join(str(a.get("eff_fps")) for a in ws) + " | "
                    + ", ".join(str(a.get("frames")) for a in ws) + " |")
    return rows


def _p50_lines(obs: dict, rung: str) -> list[str]:
    """Each candidate's MEDIAN rAF gap against its own neighbouring baselines' median.

    This is a REPORT, not a gate (except for the `.katex` comparison, which is decided on it).
    It exists because p50 is the one statistic on this host a stall cannot move, and in run
    32882865551 it settled a question the mean got wrong: the display and all-selector arms sat at
    216 and 217 ms against baselines of 246-250, which is the same arm measured twice, while their
    means disagreed by 30 points purely on where stalls landed.
    """
    scored = _scored(obs, rung)
    names = [n for n in CANDIDATES + [UPPER_BOUND, UPPER_BOUND_LATE, FLOOR, POSITIVE, NEGATIVE]
             if n in scored]
    rows_ = []
    for n in names:
        es = scored[n]
        p = _mean([e["p50"] for e in es if e.get("p50") is not None])
        bp = _mean([e["base_p50"] for e in es if e.get("base_p50") is not None])
        sv = _mean([e["p50_saving"] for e in es if e.get("p50_saving") is not None])
        if p is None or bp is None:
            continue
        rows_.append(f"| `{n}`{_label(n)} | {_sel_txt(_arm_selector(es))} | {p:.0f} ms | "
                     f"{bp:.0f} ms | " + ("-" if sv is None else f"**{sv:+.0%}**") + " |")
    if not rows_:
        return []
    return ["", "**p50, the one statistic a stall cannot move.** A median over ~90 frames is not "
                "moved by one 8.6 s frame, so where this disagrees with the mean it is the mean "
                "that is reporting stall placement. Each arm against its OWN neighbouring "
                "baselines, same as everything else here:", "",
            "| window | selector | p50 rAF gap | neighbouring baselines' p50 | cost removed on "
            "p50 |", "|---|---|---|---|---|"] + rows_


def _early_late_lines(obs: dict, rung: str, brief: bool = False) -> list[str]:
    """The reference upper bound ran twice. Whether the two agree is a fact about the SESSION.

    Run 32876363634 read 55% from this arm at r500K where run 32869180652 published 97% for the
    same arm on the same corpus and the same venue, with every control behaving in both. The only
    visible difference was WHERE IN THE SEQUENCE it ran: tenth, after three arms that skip and
    unskip large subtrees, rather than eighth after two that force `position: static`. Running it
    in both positions in one session is the only way to settle that, and if the two readings agree
    the position hypothesis is dead and the difference lives between sessions.
    """
    scored = _scored(obs, rung)
    early, late = scored.get(UPPER_BOUND) or [], scored.get(UPPER_BOUND_LATE) or []
    if not early or not late:
        return []
    eb, lb = _mean([e["bmpf"] for e in early]), _mean([e["bmpf"] for e in late])
    es_, ls_ = _saving(obs, rung, UPPER_BOUND), _saving(obs, rung, UPPER_BOUND_LATE)
    sel = _sel_txt(_arm_selector(early) or _arm_selector(late))
    if eb is None or lb is None or not eb or not lb:
        return ["", f"The reference upper bound ran twice at {rung} ({sel}) but one of the two "
                    f"windows produced no reading, so the order-dependence question is open."]
    ratio = max(eb, lb) / min(eb, lb)
    if brief:
        # The full paragraph lives in the rung's own section; repeating it here would be noise,
        # but a reader who came for the bound has to be told whether it is stable.
        return ["", f"It ran twice in this session: early **{eb:.1f}**, late **{lb:.1f}** ms "
                    f"blocked/frame, {ratio:.1f}x apart -- they "
                    + ("DISAGREE, so read the rung section above before using this number as a "
                       "bound" if ratio > EARLY_LATE_MAX_RATIO
                       else "agree, so the bound is stable within this session") + "."]
    head = (f"The reference upper bound, run TWICE at {rung} ({sel}): early "
            f"**{eb:.1f}** ms blocked/frame"
            + ("" if es_ is None else f" ({es_:+.0%})")
            + f", late **{lb:.1f}** ms blocked/frame"
            + ("" if ls_ is None else f" ({ls_:+.0%})")
            + f", a factor of {ratio:.1f}x between them.")
    if ratio > EARLY_LATE_MAX_RATIO:
        tail = (f" **They DISAGREE.** The same declaration, on the same page, in the same session, "
                f"read {ratio:.1f}x differently depending on where in the sequence it ran. That is "
                f"a statement about this SESSION and not about the arm: either the arms that ran "
                f"between them left the page in a different state, or a window caught one of this "
                f"venue's 8.6 s stalls. Read the robust column above before reading either number, "
                f"and do not use this arm as a bound until they agree.")
    else:
        tail = (f" They AGREE to within {EARLY_LATE_MAX_RATIO:.0f}x, so this arm's reading does "
                f"not depend on what ran before it, and the gap between run {PUBLISHED_RUN} "
                f"({PUBLISHED_UPPER_SAVING:.0%}) and run {STALL_RUN} (55% at r500K) is a "
                f"difference between SESSIONS rather than an artefact of arm order.")
    return ["", head + tail]


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
        d_sv = _p50_saving(obs, long_rung, SHIPPABLE)
        a_sv = _p50_saving(obs, long_rung, ALL_SELECTOR)
        d_mean = _saving(obs, long_rung, SHIPPABLE)
        a_mean = _saving(obs, long_rung, ALL_SELECTOR)
        te_d, te_a = _took_effect(disp_e), _took_effect(all_e)
        rows += [f"### Does adding `.katex` to the selector buy anything? ({long_rung})", "",
                 "The two arms should read the SAME if size containment cannot apply to a "
                 "non-atomic inline-level box: 910 of this corpus's 1,027 `.katex` roots are "
                 "`display: inline`, so on that reading the extra selector is inert. The height "
                 "probes are the evidence, and they are probed on DIFFERENT populations on "
                 "purpose -- the display arm on `.katex-display`, the all arm on the INLINE roots, "
                 "because those are the ones the claim is about.", "",
                 "**Decided on p50**, because a median cannot be moved by one stalled frame and a "
                 "mean can. The mean is printed beside it: where the two disagree, the mean is "
                 "reporting where this host's 8.6 s stalls happened to land.", "",
                 "| arm | selector | cost removed on p50 (decides) | p50 rAF gap | cost removed on "
                 "the MEAN | height probe | off-screen boxes whose height changed | mean height "
                 "before -> after |", "|---|---|---|---|---|---|---|---|"]
        for nm, sv, mn, te in ((SHIPPABLE, d_sv, d_mean, te_d), (ALL_SELECTOR, a_sv, a_mean, te_a)):
            rows.append("| " + " | ".join([
                f"`{nm}`{_label(nm)}",
                _sel_txt(_arm_selector(_entries(obs, long_rung, nm))),
                "-" if sv is None else f"**{sv:+.0%}**",
                f"{_p50_of(obs, long_rung, nm)} ms vs {_base_p50_of(obs, long_rung, nm)} baseline",
                "-" if mn is None else f"{mn:+.0%}",
                "-" if not te else f"`{te.get('selector')}`",
                "-" if not te else (f"{te.get('changed')}/{te.get('compared')} "
                                    f"({(te.get('fraction_changed') or 0):.0%})"),
                "-" if not te else f"{te.get('mean_height_before')} -> "
                                   f"{te.get('mean_height_after')} px",
            ]) + " |")
        if d_sv is not None and a_sv is not None:
            excess = a_sv - d_sv
            mean_excess = ("-" if (d_mean is None or a_mean is None)
                           else f"{a_mean - d_mean:+.1%}")
            if excess > KATEX_ALL_MAX_EXCESS:
                rows += ["", f"**Adding `.katex` bought {excess:+.1%} on p50, which is more than "
                             f"{KATEX_ALL_MAX_EXCESS:.0%}** (on the mean, {mean_excess}). The spec "
                             f"reading above is WRONG: size containment is reaching inline maths "
                             f"in this engine, and the shipped rule is leaving something on the "
                             f"table."]
            else:
                rows += ["", f"**Adding `.katex` bought {excess:+.1%} on p50, which is nothing** "
                             f"(on the mean, {mean_excess}). Inline maths is out of reach of "
                             f"`content-visibility` as a MEASURED fact rather than as a reading of "
                             f"the spec, so the shipped rule is not leaving anything on the table "
                             f"by selecting only `.katex-display`."]
        rows.append("")

        # The reference upper bound, printed as what it is.
        ub = _saving(obs, long_rung, UPPER_BOUND)
        if ub is not None:
            rows += [f"### The reference upper bound ({long_rung})", "",
                     f"`{UPPER_BOUND}` "
                     f"({_sel_txt(_arm_selector(_entries(obs, long_rung, UPPER_BOUND)))}) removes "
                     f"**{ub:+.0%}** here, against +"
                     f"{PUBLISHED_UPPER_SAVING:.0%} in run {PUBLISHED_RUN} ({PUBLISHED_BASE_MS} ms "
                     f"blocked per scroll event at {PUBLISHED_BASE_FPS} fps -> "
                     f"{PUBLISHED_UPPER_MS} ms at {PUBLISHED_UPPER_FPS} fps). It is NOT a "
                     f"candidate and cannot ship: it removes the cost by making the maths "
                     f"invisible. Its only job is to say whether this session is the same "
                     f"experiment as the published one, so that the candidate numbers above can "
                     f"be read against that 97%."]
            rows += _early_late_lines(obs, long_rung, brief = True)
            rows.append("")

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
    # The `.katex` comparison is DECIDED on p50, so the verdict quotes p50 first and the mean
    # after it, rather than quoting a number the gate did not use.
    sv_p50, alls_p50 = _p50_saving(obs, rung, SHIPPABLE), _p50_saving(obs, rung, ALL_SELECTOR)
    katex_txt = (("no window" if (alls_p50 is None or sv_p50 is None)
                  else f"{alls_p50 - sv_p50:+.1%} on p50")
                 + ("" if (alls is None or sv is None) else f" ({alls - sv:+.1%} on the mean)"))
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
    # Quote what each arm APPLIED, not only what it is called. Three times in this campaign a
    # label has been trusted over the selector it names, most recently when two different arms in
    # two scenes shared the name `visibility_hidden_offscreen`.
    ship_sel = _sel_txt(_arm_selector(ship))
    expl_sel = _sel_txt(_arm_selector(_entries(obs, rung, EXPLORATORY)))
    exploratory_txt = (
        f"`{EXPLORATORY}` ({expl_sel}, EXPLORATORY, NOT SHIPPABLE AS WRITTEN: it hoists the "
        f"declaration to a block ancestor, which is what a renderer-side change would have to do) "
        f"reads "
        + ("no window" if expl is None else f"{expl:+.0%}")
        + (f", but it is DISQUALIFIED ({expl_dq})" if expl_dq else "")
        + ", which is what reaching inline maths would be worth")

    if ship and not dq and sv is not None and sv >= ARM_MIN_SAVING:
        return "HELPS", (
            f"the shippable rule WORKS at {rung}. `{SHIPPABLE}` ({ship_sel}) removes {sv:.0%} of "
            f"the per-scroll-event cost"
            + (f" ({wr:.2f}x less blocked time per 1 px scroll)" if wr else "")
            + f", scored against its own two neighbouring baselines, with the negative control "
              f"flat and the positive control recovering. Per rep: {_per_rep(ship)}. "
            + bound + ", and " + structural + ". Adding `.katex` to the selector bought "
            + katex_txt
            + f", which is the measured form of that claim. The rule moves scrollHeight by "
            + ("no recorded amount" if sh is None else f"{sh:+.2%}")
            + f", which is `contain-intrinsic-size` standing in for the height of blocks the "
              f"engine skipped and is inside the {MAX_SCROLL_HEIGHT_DELTA:.0%} gate that says the "
              f"gesture was not pinned. " + short_txt + ". For scale, " + exploratory_txt)

    why = dq or (f"it removes only {sv:+.0%}, under the {ARM_MIN_SAVING:.0%} bar"
                 if sv is not None else "it produced no scored window")
    return "NO_BENEFIT", (
        f"the shippable rule DOES NOT WORK at {rung}: `{SHIPPABLE}` ({ship_sel}) cannot carry a "
        f"claim because {why}. Per rep: {_per_rep(ship)}. " + bound + ", and " + structural
        + ". Adding `.katex` to the selector changed the reading by " + katex_txt
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
