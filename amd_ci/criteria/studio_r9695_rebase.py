#!/usr/bin/env python3
"""Criteria: does PR 9695's 5.02x survive a rebase onto today's main?

Judges only. Observations come from probes/studio_r9695_probe.py.

THE CLAIM UNDER RE-EXAMINATION. PR 9695 renders fenced code inside reasoning panes plain, and on
this venue it was measured at **5.021x** on `action:reasoning_toggle_all` at r100K (AMD runs
`32833058576` and `32841817590`), with the reasoning span census falling 74,250 -> 10,917. Since
then main merged PR 9799 (the idle grammar pre-warm that was warming on an EMPTY STRING), PR 9731
and PR 9787, all on this path, and main's per-fence deferral now ships on by default, so a fence
nobody has scrolled to already renders the plain shell on `main` too.

**A DEAD WIN IS THE CORRECT ANSWER IF THAT IS WHAT THE BOX SAYS.** There is a live precedent:
PR 9731's +92% turned out to be stale for exactly this reason. This module is written so that
`WIN_ABSORBED` is as reachable as `WIN_SURVIVES`, is a FINDING rather than a failure, and keeps
the job green. What must never happen is a number surviving because the harness could not tell.

FOUR WAYS THIS RUN CAN PRODUCE A CONFIDENT WRONG ANSWER, AND WHAT STOPS EACH.

1. **The idle control.** This is the one this question is most likely to get wrong, and it has
   already happened. At r500K a `plain` arm read 4.16 fps against 2.77 on the arm it was being
   compared with -- +50%, and reading the median table alone would have published it -- while its
   IDLE window was already stalling in 3 repetitions of 5: idle fps 18.6, 14.8 and 4.9 against
   61-62 on every other arm, with 7.9-9.3 second frames inside the idle window. An arm whose idle
   window is already stalling is not in a comparable state. Every repetition here carries its own
   idle window, a repetition that fails it is DISCARDED BY NAME, and the count of discards per arm
   is printed in the report rather than buried.

2. **The jammed positive control.** Two of the three plausible frame channels are blind:
   `GdkFrameClock::after-paint` reads 60.0 fps with the main thread 80% blocked, and `1000/p50` of
   rAF gaps read 62.5 -> 62.5 under a jam that took the correct channel from 62.0 to 16.6. So
   every session opens a deliberately jammed window, and a repetition whose channel did not see it
   is discarded too.

   And the jam is priced against the IDLE window, which this module has already required to be
   quiet. That closes the instrument defect run `33040070879` was VOIDed by:
   `drop_fraction = 1 - jammed/clean` cannot express "the clean window was slower than the jammed
   one", and there the page at 500K was blocking the main thread about 170x harder than a
   deliberate 80%-duty spinner. Here such a repetition fails the IDLE gate first and is discarded
   as not-comparable, so the jam is never asked to price a window that was not quiet.

3. **A `busy_pct` of null.** The clamp calibration refuses above `MAX_CLAMP_MS = 10.0` ms and the
   clamp measured on this venue is 8 ms, so two milliseconds of drift makes it null. `None >= 50.0`
   is False, so a single-expression "the base arm exhibits the defect" test would report a page
   rendering one frame every eight seconds as **"main did NOT exhibit the reported collapse"**.
   That does not read as an instrument fault, it reads as the complaint never having been real.
   A null busy is carried as MISSING, never averaged as zero, and a cell with no readable busy at
   all returns **INCONCLUSIVE naming the clamp** rather than a claim about the subject.

4. **The VOID rule.** If the `main` arm does not exhibit the defect at a rung, that rung is VOID,
   not a pass. A fast head arm against a base that was never slow shows only that the harness ran.
   Each rung is gated by its OWN controls; r100K may never license r500K and vice versa.

NEVER `raf.fps_p50`. It is reported so the disagreement stays visible and it is never a headline.

WHAT IS REPORTED, and why the median sits beside the mean everywhere. This host blocks for about
8.6 seconds every 30 at r500K. A mean over ~100 frames cannot survive one 8.6 s frame; a median
across repetitions can. Both are printed for every cell, together with the stall-stripped figure
(the same window with its single worst tick dropped), so a cell whose numbers disagree announces
that it caught a stall instead of quietly reporting where the stall landed.
"""

from __future__ import annotations

import statistics

TITLE = ("PR 9695 (plain code in reasoning panes) re-measured after rebase onto main 0be140dbd, "
         "action:reasoning_toggle_all at r100K and r500K, real WebKitGTK/gfx1151")
MODE = "capability"

NEEDS = [
    "linux", "drm_render_node", "egl_hardware_gl", "webkitgtk", "headless_display_server",
    "gpu_browser_compositing", "studio_production_bundle", "studiobench_ladder",
    "discrete_gpu", "nvidia", "windows", "mlx",
]

#: THE SCORED GESTURE, quoted rather than named. `reasoning_toggle` names TWO gestures in this
#: campaign -- this scene's opens the FIRST pane, the scored one opens EVERY pane -- and the same
#: mechanism read 1.9% on one and 1.913x on the other, both correctly.
ACTION = "reasoning_toggle_all"
SECONDARY_ACTION = "reasoning_toggle"
#: not a performance window at all: it exists so the DOM can settle before fidelity is priced
FIDELITY_ACTION = "reasoning_fidelity_settled"

REFERENCE = "main"
TREATMENT = "head"
ARMS = (REFERENCE, TREATMENT)

#: The corpus the 5.02x was measured on. A different corpus is not a smaller experiment, it is a
#: different one, so it is reported rather than silently tolerated.
EXPECTED_CORPUS = "23cd2464"

# ── the two controls ─────────────────────────────────────────────────────────────────────────
#: Idle is 61-62 fps at every rung on this host when the page is in a comparable state. The
#: failing repetitions this bar exists to catch read 18.6, 14.8 and 4.9.
IDLE_MIN_FPS = 45.0
#: and a multi-second frame INSIDE the idle window means the same thing even if the mean survived
IDLE_MAX_WORST_MS = 1000.0
#: the jam is 200 ms of spin every 250 ms; the documented resolution on this venue is ~61 -> ~17
CONTROL_MIN_RATE_DROP = 0.25
#: after both discards, a cell needs this many repetitions or it is not a cell
MIN_SURVIVING_REPS = 3

# ── the VOID rule's bar ──────────────────────────────────────────────────────────────────────
#: A rung counts as a venue only if the REFERENCE arm is this loaded on the SCORED gesture.
#: Anchored on main's measured r100K reading when the 5.02x was taken: 93% busy, 1,779 ms worst
#: frame. Both halves are required, and a null busy is MISSING rather than a failure of this test.
DEFECT_MIN_BUSY_PCT = 50.0
DEFECT_MIN_WORST_MS = 300.0

#: Below this the rate channel has no dynamic range left. 0.3 fps is one frame per 3.3 seconds,
#: and a ratio of numbers that small is a ratio of single-digit frame counts.
FLOOR_FPS = 0.3

#: A ratio must clear this AND the two arms' repetition ranges must not overlap before the
#: difference is called a move rather than noise.
MIN_RATIO = 1.15

#: What the PR was originally worth on this gesture at this rung, so the report can say by how
#: much the answer has changed rather than only what it is now.
ORIGINAL_RATIO_100K = 5.021
ORIGINAL_SPANS_100K = (74250, 10917)
#: What the BASE arm read when that ratio was taken: 5.1 effective fps at 93% busy on this gesture
#: at r100K. Carried so a rung that comes out VOID for want of a defect can be reported as the
#: ANSWER it is -- "the load this claim was measured against no longer reproduces" -- rather than
#: as a bare non-result a reader has to interpret.
ORIGINAL_BASE_100K_FPS = 5.1
ORIGINAL_BASE_100K_BUSY = 93.0


# ── accessors ────────────────────────────────────────────────────────────────────────────────
def _runs(obs: dict) -> list[dict]:
    return [r for r in (obs.get("runs") or []) if (r.get("payload") or {}).get("ok")]


def _sel(obs: dict, arm: str, rung: str) -> list[dict]:
    return [r for r in _runs(obs) if r.get("arm") == arm and r.get("rung") == rung]


def _action(payload: dict, name: str | None = None) -> dict:
    want = name or ACTION
    for a in payload.get("actions") or []:
        if a.get("name") == want and not a.get("not_applicable"):
            return a
    return {}


def _phase(payload: dict, name: str) -> dict:
    for p in payload.get("phases") or []:
        if p.get("phase") == name:
            return p
    return {}


def _eff_fps(payload: dict, name: str | None = None):
    """Effective frames over WALL TIME, recomputed rather than read.

    The scene writes `eff_fps` and this recomputes it from `raf.n` and `elapsed_ms`. They must
    agree; `_field_agreement` in `gates` checks that they do. Reading a field whose meaning
    changed underneath a criteria module is how a blind channel gets published as a headline.
    """
    a = _action(payload, name)
    n = (a.get("raf") or {}).get("n")
    el = a.get("elapsed_ms")
    return (1000.0 * n / el) if (n and el) else None


def _busy(payload: dict, name: str | None = None):
    """None means MISSING, and callers must not turn it into zero."""
    return ((_action(payload, name).get("busy")) or {}).get("busy_pct")


def _busy_reason(payload: dict, name: str | None = None):
    return ((_action(payload, name).get("busy")) or {}).get("busy_pct_reason")


def _blocked_per_frame(payload: dict, name: str | None = None):
    return _action(payload, name).get("blocked_ms_per_frame")


def _robust_per_frame(payload: dict, name: str | None = None):
    """The same window with its single worst tick dropped. Not a replacement for the mean: a
    window whose two figures disagree is a window that caught a stall, which is a fact about the
    window rather than about the arm."""
    return (_action(payload, name).get("robust") or {}).get("blocked_ms_per_frame")


def _worst(payload: dict, name: str | None = None):
    return (_action(payload, name).get("raf") or {}).get("max_ms")


def _fps_p50(payload: dict, name: str | None = None):
    """REPORTED, NEVER A HEADLINE. `1000/p50` read 62.5 jammed and unjammed alike on this venue."""
    return (_action(payload, name).get("raf") or {}).get("fps_p50")


def _spans(payload: dict):
    """Reasoning-scoped highlight spans from the SETTLED census, or None if it never settled.

    Every other census in the payload is a snapshot taken 2,500 ms after a click, and highlighting
    here is asynchronous: the same arm read 11,530 spans on one repetition and 11,094 on the next,
    and a blocked page read 7,259 because it had not finished highlighting when the snapshot was
    taken. A number that moves with how busy the page happens to be cannot price a fidelity change.
    """
    a = _action(payload, FIDELITY_ACTION)
    if not a.get("settled"):
        return None
    v = ((a.get("fence_census") or {}).get("reasoning") or {}).get("spans")
    return v if isinstance(v, (int, float)) else None


def _deferred_shells(payload: dict):
    a = _action(payload, FIDELITY_ACTION)
    if not a.get("settled"):
        return None
    v = (a.get("fence_census") or {}).get("reasoning_deferred_shells")
    return v if isinstance(v, (int, float)) else None


def _reasoning_fences(payload: dict):
    a = _action(payload, FIDELITY_ACTION)
    v = ((a.get("fence_census") or {}).get("reasoning") or {}).get("fences")
    return v if isinstance(v, (int, float)) else None


# ── per-repetition admissibility. This is where repetitions are DISCARDED. ────────────────────
def rep_state(run: dict) -> dict:
    """Is this one repetition in a comparable state? Two controls, both inside the session.

    Returns the reason it is not, by name, so the report can say which repetitions were dropped
    and why rather than presenting a median over whatever survived.
    """
    p = run.get("payload") or {}
    idle = _phase(p, "idle")
    live = p.get("liveness") or {}
    st = {"arm": run.get("arm"), "rung": run.get("rung"), "rep": run.get("rep"),
          "idle_fps": idle.get("eff_fps"),
          "idle_worst_ms": (idle.get("raf") or {}).get("max_ms"),
          "jam_clean_fps": live.get("clean_fps"), "jam_fps": live.get("jammed_fps"),
          "jam_drop": live.get("drop_fraction"),
          "jam_clean_p50": live.get("clean_fps_p50"), "jam_p50": live.get("jammed_fps_p50"),
          "ok": True, "why": []}

    if st["idle_fps"] is None:
        st["ok"] = False
        st["why"].append("no idle window was measured, so there is no idle control")
    else:
        if st["idle_fps"] < IDLE_MIN_FPS:
            st["ok"] = False
            st["why"].append(
                f"the IDLE window read {st['idle_fps']:.1f} fps, below the {IDLE_MIN_FPS:.0f} fps "
                f"bar: this page was already stalling before the gesture, so the arm was not in a "
                f"comparable state")
        if (st["idle_worst_ms"] or 0) > IDLE_MAX_WORST_MS:
            st["ok"] = False
            st["why"].append(
                f"the IDLE window contained a {st['idle_worst_ms']:,.0f} ms frame, over the "
                f"{IDLE_MAX_WORST_MS:,.0f} ms bar, so something was blocking the main thread with "
                f"nobody touching the page")

    # The jam is only asked to price a window the idle gate has already found quiet. That is what
    # keeps `1 - jammed/clean` from being handed a clean window slower than the jammed one, which
    # is the instrument defect that VOIDed run 33040070879.
    if st["ok"]:
        if st["jam_drop"] is None:
            st["ok"] = False
            st["why"].append("the jammed control produced no drop fraction")
        elif st["jam_drop"] < CONTROL_MIN_RATE_DROP:
            st["ok"] = False
            st["why"].append(
                f"the JAMMED control moved the rate by {100 * st['jam_drop']:+.0f}% "
                f"({st['jam_clean_fps']} -> {st['jam_fps']} fps), under the "
                f"{100 * CONTROL_MIN_RATE_DROP:.0f}% bar, so this session's frame channel was not "
                f"shown able to report a blocked main thread and none of its numbers mean "
                f"anything")
    return st


def surviving(obs: dict, arm: str, rung: str) -> list[dict]:
    return [r for r in _sel(obs, arm, rung) if rep_state(r)["ok"]]


def discarded(obs: dict, arm: str, rung: str) -> list[dict]:
    return [rep_state(r) for r in _sel(obs, arm, rung) if not rep_state(r)["ok"]]


# ── aggregation over the surviving repetitions ───────────────────────────────────────────────
def _vals(runs: list[dict], fn, name: str | None = None):
    out = []
    for r in runs:
        try:
            v = fn(r["payload"], name) if name is not None else fn(r["payload"])
        except TypeError:
            v = fn(r["payload"])
        if v is not None:
            out.append(v)
    return out


def _agg(xs: list) -> dict:
    """Median beside mean, always. Never one without the other on this host."""
    if not xs:
        return {"n": 0, "median": None, "mean": None, "min": None, "max": None}
    return {"n": len(xs), "median": statistics.median(xs), "mean": sum(xs) / len(xs),
            "min": min(xs), "max": max(xs)}


def cell(obs: dict, arm: str, rung: str) -> dict:
    """One (arm, rung) cell, over the repetitions that survived both controls."""
    runs = surviving(obs, arm, rung)
    drops = discarded(obs, arm, rung)
    busy_vals = _vals(runs, _busy)
    c = {
        "arm": arm, "rung": rung,
        "reps_run": len(_sel(obs, arm, rung)),
        "reps_surviving": len(runs), "reps_discarded": len(drops),
        "discards": drops,
        "fps": _agg(_vals(runs, _eff_fps)),
        "blocked_per_frame": _agg(_vals(runs, _blocked_per_frame)),
        "robust_per_frame": _agg(_vals(runs, _robust_per_frame)),
        "worst_ms": _agg(_vals(runs, _worst)),
        "fps_p50_blind": _agg(_vals(runs, _fps_p50)),
        "secondary_fps": _agg(_vals(runs, lambda p: _eff_fps(p, SECONDARY_ACTION))),
        # busy_pct: MISSING is counted, never averaged as zero.
        "busy": _agg(busy_vals),
        "busy_missing": len(runs) - len(busy_vals),
        "busy_missing_reasons": sorted({str(_busy_reason(r["payload"])) for r in runs
                                        if _busy(r["payload"]) is None}),
        "spans": _agg(_vals(runs, _spans)),
        "deferred_shells": _agg(_vals(runs, _deferred_shells)),
        "fences": _agg(_vals(runs, _reasoning_fences)),
        "settled_reps": sum(1 for r in runs if _spans(r["payload"]) is not None),
        "idle_fps": _agg([rep_state(r)["idle_fps"] for r in runs
                          if rep_state(r)["idle_fps"] is not None]),
        "jam_fps": _agg([rep_state(r)["jam_fps"] for r in runs
                         if rep_state(r)["jam_fps"] is not None]),
        "jam_drop": _agg([rep_state(r)["jam_drop"] for r in runs
                          if rep_state(r)["jam_drop"] is not None]),
    }
    return c


def rungs(obs: dict) -> list[str]:
    return [r for r in (obs.get("rungs") or []) if _sel(obs, REFERENCE, r)]


# ── rung admissibility: SCORED / VOID / INCONCLUSIVE, with the reason ────────────────────────
def rung_state(obs: dict, rung: str) -> dict:
    st: dict = {"rung": rung, "notes": []}
    base = cell(obs, REFERENCE, rung)
    head = cell(obs, TREATMENT, rung)
    st["base"], st["head"] = base, head

    thin = [c["arm"] for c in (base, head) if c["reps_surviving"] < MIN_SURVIVING_REPS]
    if thin:
        st["state"] = "INCONCLUSIVE"
        st["notes"].append(
            f"{' and '.join(thin)} kept fewer than {MIN_SURVIVING_REPS} repetitions at r{rung} "
            f"after the idle and jam controls "
            f"({base['reps_surviving']}/{base['reps_run']} main, "
            f"{head['reps_surviving']}/{head['reps_run']} head), so this rung has no cell to "
            f"compare rather than a cell that says something")
        return st

    # THE CLAMP, BEFORE THE DEFECT TEST. `None >= 50.0` is False, so folding a missing busy into
    # the defect test would report an unreadable instrument as "main was never slow".
    if base["busy"]["n"] == 0:
        st["state"] = "INCONCLUSIVE"
        st["notes"].append(
            f"main's busy percentage is unreadable at r{rung} in every surviving repetition "
            f"({'; '.join(base['busy_missing_reasons']) or 'no reason recorded'}), so the "
            f"setTimeout CLAMP did not calibrate and a frame rate cannot be qualified as a "
            f"collapse. This is a statement about the instrument and NOT about whether main is "
            f"slow here")
        return st

    st["base_busy"] = base["busy"]["mean"]
    st["base_worst"] = base["worst_ms"]["median"]
    st["base_fps"] = base["fps"]["median"]
    st["defect"] = bool(st["base_busy"] is not None and st["base_worst"] is not None
                        and st["base_busy"] >= DEFECT_MIN_BUSY_PCT
                        and st["base_worst"] >= DEFECT_MIN_WORST_MS)
    if not st["defect"]:
        st["notes"].append(
            f"the BASE arm does not exhibit the defect at r{rung}: {st['base_busy']:.1f}% busy "
            f"(bar {DEFECT_MIN_BUSY_PCT:.0f}%) with a {st['base_worst']:,.0f} ms worst frame (bar "
            f"{DEFECT_MIN_WORST_MS:,.0f} ms) on `{ACTION}`. A head arm that is fast against a base "
            f"that was never slow shows only that the harness ran, so this rung is VOID rather "
            f"than a pass")
        # AND WHAT THAT MEANS FOR THIS PARTICULAR QUESTION, at the rung the claim was made at. A
        # bare VOID reads as "we could not tell", and here it is very nearly the opposite: it says
        # the load PR 9695 was measured against is gone. `main` at this commit already ships
        # `code-fence-mode.ts` `SHIP_DEFAULT = "defer"`, so an unreached fence is a plain shell on
        # the BASE arm too, and PR 9695's remaining margin is only the per-fence deferral
        # bookkeeping inside reasoning panes.
        if rung == "100K" and st["base_fps"] is not None:
            st["notes"].append(
                f"and that is the answer to the question rather than a failure to answer it: main "
                f"now reads {st['base_fps']:.1f} fps at {st['base_busy']:.1f}% busy on this "
                f"gesture, against {ORIGINAL_BASE_100K_FPS:.1f} fps at "
                f"{ORIGINAL_BASE_100K_BUSY:.0f}% busy when the {ORIGINAL_RATIO_100K:.3f}x was "
                f"taken. The collapse PR 9695 removed is no longer there to remove, so there is "
                f"nothing left for it to win. VOID is the verdict because this run cannot price a "
                f"fix against a defect it did not observe, NOT because the measurement failed")

    st["at_floor"] = bool(st["base_fps"] is not None and st["base_fps"] < FLOOR_FPS)
    if st["at_floor"]:
        st["notes"].append(
            f"main sits at {st['base_fps']:.2f} fps, below the {FLOOR_FPS} fps instrument floor, "
            f"so a ratio here is a ratio of single-digit frame counts and is quoted as direction "
            f"only")

    st["state"] = "SCORED" if (st["defect"] and not st["at_floor"]) else "VOID"
    return st


def ratio(obs: dict, rung: str) -> dict:
    """head / main at this rung, with the uncertainty stated three ways rather than one."""
    base, head = cell(obs, REFERENCE, rung), cell(obs, TREATMENT, rung)
    b, h = base["fps"], head["fps"]
    out: dict = {"rung": rung, "point": None, "low": None, "high": None,
                 "separated": None, "base": b, "head": h}
    if not b["median"] or not h["median"]:
        return out
    out["point"] = h["median"] / b["median"]
    # The widest ratio the repetitions permit, which is the honest envelope for 3-5 readings:
    # nothing here is normal enough for a t interval and a spread of 5 is not a distribution.
    out["low"] = h["min"] / b["max"] if b["max"] else None
    out["high"] = h["max"] / b["min"] if b["min"] else None
    # Do the two arms' repetition ranges overlap? An overlap is not a small effect, it is an
    # effect this many repetitions cannot resolve, and it is reported as that.
    out["separated"] = bool(h["min"] > b["max"] or b["min"] > h["max"])
    return out


def _fmt(v, spec="{:.1f}", dash="n/a"):
    return dash if v is None else spec.format(v)


def _busy_cell(c: dict) -> str:
    """`MISSING` is a reading, `0` is a lie. A null busy is never folded into a mean, and when
    some repetitions have one and some do not, the report says how many did not."""
    if not c["busy"]["n"]:
        return "MISSING"
    s = _fmt(c["busy"]["mean"], "{:.1f}%")
    if c["busy_missing"]:
        s += f" ({c['busy_missing']} of {c['reps_surviving']} reps MISSING)"
    return s


def _cellrow(c: dict) -> str:
    return (f"| {c['arm']} | r{c['rung']} | "
            f"**{_fmt(c['fps']['median'], '{:.2f}')}** | {_fmt(c['fps']['mean'], '{:.2f}')} | "
            f"{_fmt(c['fps']['min'], '{:.2f}')}-{_fmt(c['fps']['max'], '{:.2f}')} | "
            f"{_fmt(c['blocked_per_frame']['median'])} | {_fmt(c['blocked_per_frame']['mean'])} | "
            f"{_fmt(c['robust_per_frame']['median'])} | "
            f"{_busy_cell(c)} | "
            f"{_fmt(c['worst_ms']['max'], '{:,.0f}')} | "
            f"{_fmt(c['spans']['median'], '{:,.0f}')} | "
            f"{c['reps_surviving']}/{c['reps_run']} |")


#: PRINTED IN THE VERDICT ARTIFACT ITSELF. A reader who knows this toolkit's documented workflow
#: will otherwise assume `lib/differential.py` ran and that ITS base/head enforcement is what
#: produced the verdict. Deviating is defensible; deviating silently is not.
PROVENANCE = """### Where the base/head enforcement lives in THIS run, and why it is not `differential.py`

`lib/differential.py` was NOT used here, and the VOID rule it exists to enforce is implemented in
this criteria module instead. The reason is mechanical rather than a preference.

`differential.py` runs the probe **once per state**: `for name, path in states["paths"].items():
obs[name] = run_probe(...)`. That is one probe invocation for base, then one for head, sequentially.
This measurement cannot be laid out that way. PR 9695 is not flag-gated -- its whole mechanism is
`reasoning.tsx` wrapping `<MarkdownText/>` in `<MarkdownCodeHighlightingContext.Provider
value="plain">`, a hardcoded literal that `markdown-text.tsx` reads with `useContext(...)`, with no
env var, no `__UNSLOTH_*__` global and no `VITE_*` switch -- so the two arms are two separate
FRONTEND BUILDS and cannot be toggled inside one running instance.

What is done instead: BOTH bundles are built in the non-GPU job from ONE checkout, and the
measuring probe alternates **which dist is served** per repetition, relaunching Studio each time on
a fresh port with a fresh Studio home and a freshly seeded thread, with the arm order swapped on
alternate repetitions. That is rep-granularity interleaving of the two UNMODIFIED shipping
artifacts, on one host, in one job, against one backend install. A relaunch per repetition is
wanted here anyway: the scored gesture is a settled-thread mount gesture, so every repetition
should start from a fresh mount.

Why it matters that the arms interleave: this host blocks the main thread for roughly 8.6 seconds
every 30 at r500K. Two sequential legs let such a stall sit entirely on one arm and be reported as
that arm's mechanism. Adjacency plus the alternating order is what keeps machine drift, and
"whichever arm went first", out of the arm difference.

No flag was added to PR 9695 to make interleaving possible. A runtime-flippable switch would be a
real context read that is not in the shipping artifact, which changes the thing being measured to
suit the instrument.

**The rule is kept verbatim:** if the `main` arm does not exhibit the defect at a rung, that rung
is VOID and scores nothing, however large the head-vs-main ratio. The per-rung jam control and the
per-repetition idle control are kept as well, and all three are pinned by
`amd_ci/selftest_r9695.py`, which is mutation-verified: each guard is shown going RED against a
deliberately broken copy of this module before the run is spent.
"""


def table(obs: dict) -> str:
    lines: list[str] = [PROVENANCE, ""]

    lines += ["### The gesture, per (arm, rung)", "",
              "Effective frame rate over WALL TIME on `action:reasoning_toggle_all`, over the "
              "repetitions that survived BOTH controls. Median and mean are both printed because "
              "this host blocks for about 8.6 s every 30 s at r500K and a mean over ~100 frames "
              "cannot survive one such frame. `stall-stripped` is the same window with its single "
              "worst tick dropped: a cell whose mean and stall-stripped figures disagree caught a "
              "stall, which is a fact about the window and not about the arm.", "",
              "| arm | rung | fps median | fps mean | fps range | blocked ms/frame p50 | "
              "blocked ms/frame mean | stall-stripped p50 | busy | worst frame ms | "
              "reasoning spans | reps kept |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for rung in rungs(obs):
        for arm in ARMS:
            lines.append(_cellrow(cell(obs, arm, rung)))
    lines.append("")

    lines += ["### head vs main, per rung", "",
              "| rung | rung state | head/main (medians) | envelope | ranges separated? | "
              "originally |", "|---|---|---|---|---|---|"]
    for rung in rungs(obs):
        st = rung_state(obs, rung)
        r = ratio(obs, rung)
        orig = f"{ORIGINAL_RATIO_100K:.3f}x" if rung == "100K" else "not measured"
        lines.append(
            f"| r{rung} | **{st['state']}** | "
            f"{_fmt(r['point'], '{:.3f}x')} | "
            f"{_fmt(r['low'], '{:.2f}')}-{_fmt(r['high'], '{:.2f}')}x | "
            f"{'yes' if r['separated'] else 'NO, they overlap'} | {orig} |")
    lines.append("")

    lines += ["### The two controls, per cell", "",
              "A cell whose controls did not resolve licenses nothing. `1000/p50` is printed "
              "beside the jam so the blind channel's non-response stays visible: on this venue it "
              "reads about 62.5 both jammed and clean.", "",
              "| arm | rung | idle fps (median) | jam fps (median) | jam drop | reps discarded | "
              "why |", "|---|---|---|---|---|---|---|"]
    for rung in rungs(obs):
        for arm in ARMS:
            c = cell(obs, arm, rung)
            why = "; ".join(f"rep {d['rep']}: {d['why'][0] if d['why'] else '?'}"
                            for d in c["discards"])[:400] or "none"
            lines.append(
                f"| {arm} | r{rung} | {_fmt(c['idle_fps']['median'], '{:.1f}')} | "
                f"{_fmt(c['jam_fps']['median'], '{:.1f}')} | "
                f"{_fmt(c['jam_drop']['median'], '{:.0%}')} | "
                f"{c['reps_discarded']}/{c['reps_run']} | {why} |")
    lines.append("")

    lines += ["### The fidelity trade, from the SETTLED census", "",
              f"Reasoning-scoped, because a thread-wide span count mixes the container the "
              f"gesture acts on with containers it does not touch. The original run recorded "
              f"{ORIGINAL_SPANS_100K[0]:,} -> {ORIGINAL_SPANS_100K[1]:,} spans at r100K.", "",
              "| arm | rung | reasoning fences | reasoning spans | deferred shells in reasoning | "
              "settled in |", "|---|---|---|---|---|---|"]
    for rung in rungs(obs):
        for arm in ARMS:
            c = cell(obs, arm, rung)
            lines.append(
                f"| {arm} | r{rung} | {_fmt(c['fences']['median'], '{:,.0f}')} | "
                f"{_fmt(c['spans']['median'], '{:,.0f}')} | "
                f"{_fmt(c['deferred_shells']['median'], '{:,.0f}')} | "
                f"{c['settled_reps']}/{c['reps_surviving']} reps |")
    lines.append("")

    lines += ["### The one-pane gesture, recorded and NEVER scored", "",
              "`reasoning_toggle` opens the FIRST pane only, so its possible effect is capped at "
              "one trace however large the thread is. It shares a name with the scored gesture "
              "and the two once read 1.9% and 1.913x for the same mechanism, both correctly.", "",
              "| arm | rung | fps median | fps mean |", "|---|---|---|---|"]
    for rung in rungs(obs):
        for arm in ARMS:
            c = cell(obs, arm, rung)
            lines.append(f"| {arm} | r{rung} | {_fmt(c['secondary_fps']['median'], '{:.2f}')} | "
                         f"{_fmt(c['secondary_fps']['mean'], '{:.2f}')} |")
    lines.append("")

    for rung in rungs(obs):
        st = rung_state(obs, rung)
        for n in st["notes"]:
            lines.append(f"- r{rung}: {n}")
    return "\n".join(lines)


# ── gates ────────────────────────────────────────────────────────────────────────────────────
def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out: list[tuple[str, bool, str]] = []
    st = obs.get("states") or {}
    build = obs.get("build") or {}
    ok = _runs(obs)

    out.append(("a display server was obtained",
                bool((obs.get("xserver") or {}).get("display")),
                obs.get("fatal") or str((obs.get("xserver") or {}).get("display"))))

    base_ref = obs.get("base_ref")
    landed = (build.get("clone") or {}).get("ref_landed")
    out.append(("the one checkout landed on the commit it was asked for",
                bool((build.get("clone") or {}).get("checkout_ok")),
                f"asked {str(base_ref)[:9]}, landed {str(landed)[:9]}: "
                f"{(build.get('clone') or {}).get('commit_line', '')[:90]}"))

    pf = build.get("preflight") or {}
    marker_ok = (pf.get("marker_absent_at_base") is True
                 and pf.get("marker_present_after_apply") is True
                 and pf.get("marker_after_revert") is False)
    out.append((f"r9695.patch flips `{obs.get('head_marker')}` on and off in the SOURCE",
                marker_ok,
                f"absent at base={pf.get('marker_absent_at_base')}, "
                f"present after apply={pf.get('marker_present_after_apply')}, "
                f"absent again after revert={pf.get('marker_after_revert')}"))

    out.append(("nothing outside studio/frontend differs between the arms, so ONE backend install "
                "can serve both", not (pf.get("changed_outside_frontend") or []),
                str(pf.get("changed_outside_frontend") or "nothing outside the frontend")))

    want = {REFERENCE: False, TREATMENT: True}
    got = {a: (st.get(a) or {}).get("head_marker_present") for a in ARMS}
    out.append(("each arm's built SOURCE carries the marker state it is labelled with",
                all(got.get(a) is want[a] for a in ARMS),
                "; ".join(f"{a}: marker={got.get(a)} (want {want[a]})" for a in ARMS)))

    out.append(("every arm's patches applied cleanly",
                all((st.get(a) or {}).get("patch_ok") for a in ARMS),
                "; ".join(f"{a}: {(st.get(a) or {}).get('patch_steps')}" for a in ARMS)[:400]))

    hashes = {a: (st.get(a) or {}).get("exported_bundle_hash") for a in ARMS}
    out.append(("the two arms are TWO bundles, not one",
                len({h for h in hashes.values() if h}) == len(ARMS),
                "; ".join(f"{a}={hashes.get(a)}" for a in ARMS)))

    out.append(("the bundle each arm was MEASURED with is the bundle it was BUILT as",
                all((st.get(a) or {}).get("bundle_hash_matches_build") for a in ARMS),
                "; ".join(f"{a}: built {(st.get(a) or {}).get('exported_bundle_hash')} "
                          f"measured {(st.get(a) or {}).get('bundle_hash_at_measure')}"
                          for a in ARMS)))

    out.append(("every arm served a PRODUCTION bundle",
                all(((st.get(a) or {}).get("dist") or {}).get("index_html")
                    and ((st.get(a) or {}).get("dist") or {}).get("asset_files", 0) > 0
                    for a in ARMS),
                "; ".join(f"{a}: {((st.get(a) or {}).get('dist') or {}).get('asset_files')} assets"
                          for a in ARMS)))

    planned = len(obs.get("plan") or [])
    out.append(("every planned session completed", bool(planned) and len(ok) == planned,
                f"{len(ok)}/{planned} sessions returned ok; "
                + "; ".join(f"{r.get('arm')}/{r.get('rung')}/rep{r.get('rep')}: "
                            f"{str((r.get('payload') or {}).get('error'))[:100]}"
                            for r in (obs.get("runs") or [])
                            if not (r.get("payload") or {}).get("ok"))[:400]))

    engines = [bool((r["payload"].get("engine_probe") or {}).get("has_webkit_message_handlers")
                    and (r["payload"].get("engine_probe") or {}).get("is_webkit_gtk_ua")
                    and not (r["payload"].get("engine_probe") or {}).get("has_chrome"))
               for r in ok]
    out.append(("every session really ran in WebKitGTK", bool(engines) and all(engines),
                f"{sum(engines)}/{len(engines)} sessions; "
                f"ua={str((ok[0]['payload'].get('ua') if ok else ''))[:80]}"))

    ihash = {(r["payload"].get("run_meta") or {}).get("instrument_hash") for r in ok}
    out.append(("ONE pinned instrument drove every session", len(ihash) == 1 and None not in ihash,
                f"studiobench hashes seen: {sorted(str(h) for h in ihash)}"))

    corp = {(r["payload"].get("run_meta") or {}).get("corpus_hash") for r in ok}
    out.append(("ONE corpus across every session, and it is the one the 5.02x was taken on",
                len(corp) == 1 and EXPECTED_CORPUS in str(next(iter(corp), "")),
                f"corpus hashes seen: {sorted(str(c) for c in corp)} "
                f"(expected {EXPECTED_CORPUS})"))

    mounted = [((r["payload"].get("mount") or {}).get("by") == "last_seeded_marker")
               for r in ok if r.get("rung") != "0K"]
    out.append(("the seeded thread really mounted, by the last seeded marker and not by the "
                "composer appearing", bool(mounted) and all(mounted),
                f"{sum(mounted)}/{len(mounted)} sessions reached the last seeded marker"))

    # The scene writes `eff_fps`; this module recomputes it. If those two ever disagree, one of
    # them means something other than what its name says, and a headline is being read off a
    # field whose definition moved.
    bad = []
    for r in ok:
        a = _action(r["payload"])
        w, got_v = a.get("eff_fps"), _eff_fps(r["payload"])
        if w is None or got_v is None or abs(w - got_v) > max(0.05, 0.01 * got_v):
            bad.append(f"{r.get('arm')}/{r.get('rung')}/rep{r.get('rep')}: scene {w} vs {got_v}")
    out.append(("the scene's eff_fps and this module's recomputation agree", not bad,
                "; ".join(bad)[:400] or f"agree on all {len(ok)} sessions"))

    cells = [(a, g) for g in rungs(obs) for a in ARMS]
    thin = [f"{a}/r{g}: {cell(obs, a, g)['reps_surviving']}/{cell(obs, a, g)['reps_run']}"
            for a, g in cells if cell(obs, a, g)["reps_surviving"] < MIN_SURVIVING_REPS]
    out.append((f"every cell kept at least {MIN_SURVIVING_REPS} repetitions after the idle and "
                f"jam controls", bool(cells) and not thin,
                "; ".join(thin) or f"all {len(cells)} cells kept enough"))

    out.append(("both rungs were measured, and both arms at each",
                sorted(rungs(obs)) == sorted(obs.get("rungs") or [])
                and all(_sel(obs, a, g) for a, g in cells),
                f"asked for {obs.get('rungs')}, measured {rungs(obs)}"))
    return out


# ── verdict ──────────────────────────────────────────────────────────────────────────────────
def _notes(st: dict) -> str:
    """One rung's notes as sentences. Without the full stops the verdict string reads as a single
    run-on clause and the rung boundary disappears exactly where a reader is deciding which rung a
    number belongs to."""
    return " ".join(n if n.endswith(".") else n + "." for n in st["notes"])


def verdict(obs: dict) -> tuple[str, str]:
    states = {g: rung_state(obs, g) for g in rungs(obs)}
    if not states:
        return "VOID", "no rung produced a usable pair of cells"

    inconc = [g for g, s in states.items() if s["state"] == "INCONCLUSIVE"]
    if inconc and len(inconc) == len(states):
        return "INCONCLUSIVE", " ".join(f"r{g}: {_notes(states[g])}" for g in inconc)

    scored = [g for g, s in states.items() if s["state"] == "SCORED"]
    if not scored:
        why = " ".join(f"r{g}: {_notes(states[g])}" for g in states)
        return "VOID", f"no rung was scoreable. {why}"

    verdicts: dict[str, str] = {}
    parts: list[str] = []
    for g in scored:
        r = ratio(obs, g)
        p = r["point"]
        if p is None:
            verdicts[g] = "UNREADABLE"
            parts.append(f"r{g}: no ratio could be formed")
            continue
        if not r["separated"] or (1 / MIN_RATIO) <= p <= MIN_RATIO:
            verdicts[g] = "ABSORBED"
            parts.append(
                f"r{g}: head/main = {p:.3f}x (envelope {r['low']:.2f}-{r['high']:.2f}x, "
                f"repetition ranges {'overlap' if not r['separated'] else 'are separated'}), so "
                f"PR 9695 buys nothing measurable here"
                + (f", against {ORIGINAL_RATIO_100K:.3f}x when it was first measured"
                   if g == "100K" else ""))
        elif p > MIN_RATIO:
            verdicts[g] = "SURVIVES"
            parts.append(
                f"r{g}: head/main = {p:.3f}x (envelope {r['low']:.2f}-{r['high']:.2f}x, ranges "
                f"separated), so PR 9695 still pays"
                + (f", against {ORIGINAL_RATIO_100K:.3f}x originally" if g == "100K" else ""))
        else:
            verdicts[g] = "SLOWER"
            parts.append(
                f"r{g}: head/main = {p:.3f}x (envelope {r['low']:.2f}-{r['high']:.2f}x, ranges "
                f"separated), so the patched arm is SLOWER than main here")

    for g, s in states.items():
        if s["state"] != "SCORED":
            parts.append(f"r{g} is {s['state']}: {_notes(s)}")

    # A full stop between the per-rung sentences. Without it the verdict string reads as one
    # run-on clause and the rung boundary disappears exactly where a reader is deciding which
    # rung a number belongs to.
    parts = [s if s.endswith(".") else s + "." for s in parts]

    kinds = set(verdicts.values())
    if kinds == {"ABSORBED"}:
        head = "WIN_ABSORBED"
    elif "SLOWER" in kinds:
        head = "HEAD_SLOWER"
    elif "SURVIVES" in kinds and "ABSORBED" in kinds:
        head = "WIN_SURVIVES_AT_SOME_RUNGS"
    elif kinds == {"SURVIVES"}:
        head = "WIN_SURVIVES"
    else:
        return "INCONCLUSIVE", " ".join(parts)
    return head, " ".join(parts)


def observed_capabilities(obs: dict) -> dict[str, bool]:
    ok = _runs(obs)
    st = obs.get("states") or {}
    return {
        "studio_production_bundle": all(
            ((st.get(a) or {}).get("dist") or {}).get("index_html") for a in ARMS),
        # Two or more rungs, each really seeded, each really mounted by the seeder's own marker.
        "studiobench_ladder": len(rungs(obs)) >= 2 and bool(ok) and all(
            (r["payload"].get("mount") or {}).get("by") == "last_seeded_marker" for r in ok),
        # A browser that ran and painted on this host. Compositing itself is established by
        # webkit_gpu_compositing, not here, so this stays False rather than being asserted.
        "gpu_browser_compositing": False,
        "egl_hardware_gl": False,
    }
