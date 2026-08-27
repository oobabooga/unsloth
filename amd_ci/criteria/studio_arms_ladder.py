#!/usr/bin/env python3
"""Criteria: does today's shipped `main` still collapse from 60 fps to 5 fps as a thread grows?

Judges. Observes nothing. probes/studio_arms_probe.py observes.

THE CLAIM UNDER TEST is the user's own words: "60 FPS downgrading to 5 FPS when the context
length grows from 0K to 100K to 500K". Two arms, both installed as they ship, nothing forced:

  pre   the last commit before the campaign started
  head  today's main

WHAT THIS MODULE REFUSES TO SCORE, and each refusal has cost a real run at least once:

  1. A SESSION WHOSE JAM CONTROL DID NOT RESOLVE. `GdkFrameClock::after-paint` reads 60.0 fps with
     the main thread 80% blocked, and `1000/p50` of rAF gaps reads 16.0 ms jammed and unjammed
     alike. Both are blind, and a blind instrument reports a tight repeat spread, which reads as
     precision. The scene therefore blocks the main thread on purpose for six seconds in EVERY
     session, at every rung, on both arms, and if that window does not read far below the clean
     one then nothing measured in that session means anything. Documented pass is about 61 -> 17.

  2. A BASE THAT DID NOT COLLAPSE. If `pre` does not exhibit the reported symptom then this run
     did not reproduce the problem and cannot speak to whether it is fixed. That is the toolkit's
     one rule and it is not negotiable here either.

  3. A HEAD THAT DID NOT BOOT INTO THE SHIPPED FIXED STATE. `math-block-mode.ts::gateOnEngine`
     turns containment OFF unless `CSS.supports("anchor-name: --unsloth-probe")` is true. If this
     runner's WebKitGTK fails that probe then the head arm ran WITHOUT the scroll fix, and the
     honest report of a small delta is "the fix was never on", not "the fix is worth little". That
     gets its own verdict string rather than a footnote.

WHAT IT REPORTS BESIDE THE FRAME RATE, because a frame rate alone hides the shape of the problem:
busy percentage, the stall-stripped mean (`robust.blocked_ms_per_frame`, the same window with its
single worst tick removed), the share of frames over 100 ms, and the worst frame. A window whose
mean and stall-stripped mean disagree is a window that caught a stall, and that is a fact about
the window rather than about the arm.

REGRESSIONS ARE REPORTED FIRST. A campaign reports its wins by default; this module computes the
`head` worse than `pre` cells before it computes anything else and puts them at the top.
"""

from __future__ import annotations

TITLE = ("Two shipped builds up the ladder on a real GPU: what frame rate does a user get at 0K, "
         "100K and 500K today, against before the campaign, with nothing forced on either arm?")
MODE = "capability"

NEEDS = [
    "linux", "drm_render_node", "egl_hardware_gl", "webkitgtk", "headless_display_server",
    "gpu_browser_compositing", "studio_production_bundle", "studiobench_ladder",
    # Declared because the CLAIM touches them, not because this host has them. An under-declared
    # NEEDS renders a report with no bounds at all.
    "discrete_gpu", "nvidia", "windows", "mlx", "xpu",
]

ARMS = ("pre", "head")
# `0K` is synthetic (an empty thread, constructed by the bench); the rest are studiobench's own
# RUNGS. `50K` was in this list and is not a rung the corpus can plan, and `1M` was missing.
RUNG_ORDER = ["0K", "1K", "10K", "100K", "500K", "1M"]
PHASES = ["idle", "idle_jammed", "scroll", "stream", "recover"]
SCORED_PHASES = ["idle", "scroll", "stream"]
ACTIONS = ["reasoning_toggle", "select_all_copy"]

# ── thresholds, all stated here rather than inline ──────────────────────────────────────────────

# The jam must cost at least this fraction of the clean effective frame rate, or the channel is
# blind and the session is unscoreable. The observed pass on this venue is ~0.72 (61 -> 17).
LIVENESS_MIN_DROP = 0.25

# What "the base exhibits the collapse" means, at the top rung. Both conditions, because a low
# frame rate on an idle page with nothing to draw is not a collapse: `idle:calibrate` reads 9.8 to
# 17.6 fps at 0 to 2% busy and means nothing at all.
BASE_COLLAPSE_MAX_FPS = 25.0
BASE_COLLAPSE_MIN_BUSY = 40.0

# What "head cleared it" means. Not "improved": the user's complaint is a number, so the bar is a
# number a user would call smooth again.
FIXED_MIN_FPS = 30.0
FIXED_MIN_RATIO = 1.5

# A head cell counts as a regression when it is this much worse than the same pre cell AND the
# repetitions agree about the direction.
REGRESSION_MIN_LOSS = 0.20
# Below this frame rate a proportional comparison is noise: at 3 fps a 20% loss is 0.6 fps. Set
# ABOVE the pre arm's own documented 500K scroll figure of 3.2 fps, because a floor that sits
# under the one cell it was written for is not a floor.
REGRESSION_MIN_ABS_FPS = 5.0

# The DOM has to actually grow up the ladder, or the rungs are labels on the same page.
MOUNT_MIN_GROWTH = 5.0

# A window whose mean is this many times its stall-stripped mean caught a stall. Not an error:
# reported, so a reader knows which of the two numbers to weigh.
MEAN_OVER_ROBUST_MAX = 2.0


# ── extraction ──────────────────────────────────────────────────────────────────────────────────

def _runs(obs: dict) -> list[dict]:
    return [r for r in (obs.get("runs") or []) if isinstance(r, dict)]


def _payload(run: dict) -> dict:
    p = run.get("payload")
    return p if isinstance(p, dict) else {}


def _ok(run: dict) -> bool:
    return bool(_payload(run).get("ok"))


def _sub(d, key: str) -> dict:
    """`d[key]` when it is a dict, else `{}`.

    `x.get("busy", {}).get("busy_pct")` raises when the key is PRESENT and null, which is a
    different thing from absent and is exactly what a degraded payload carries. The crash would
    land inside `gates()`, i.e. after an eight-hour run had already been paid for, and it would
    take the artifact's own re-scoring with it.
    """
    v = (d or {}).get(key)
    return v if isinstance(v, dict) else {}


def _phase(run: dict, name: str) -> dict:
    for ph in _payload(run).get("phases") or []:
        if isinstance(ph, dict) and ph.get("phase") == name:
            return ph
    return {}


def _action(run: dict, name: str) -> dict:
    for a in _payload(run).get("actions") or []:
        if isinstance(a, dict) and a.get("name") == name:
            return a
    return {}


def _rungs(obs: dict) -> list[str]:
    seen = {r.get("rung") for r in _runs(obs)}
    ordered = [r for r in RUNG_ORDER if r in seen]
    return ordered + sorted(x for x in seen if x and x not in ordered)


def _cells(obs: dict, arm: str, rung: str) -> list[dict]:
    return [r for r in _runs(obs)
            if r.get("arm") == arm and r.get("rung") == rung and _ok(r)]


def _census(run: dict, which: str) -> dict:
    """The mount census or the final one, whichever was asked for, never a None."""
    if which == "mount":
        return _sub(_sub(_payload(run), "mount"), "census")
    return _sub(_payload(run), which)


def _fps_values(obs: dict, arm: str, rung: str, phase: str) -> list[float]:
    out = []
    for r in _cells(obs, arm, rung):
        v = _phase(r, phase).get("eff_fps")
        if isinstance(v, (int, float)):
            out.append(float(v))
    return out


def _mean(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return sum(xs) / len(xs) if xs else None


def _fmt(v, nd = 1):
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def _reps_str(xs, nd = 1):
    if not xs:
        return "n/a"
    return ", ".join(f"{x:.{nd}f}" for x in xs)


# ── gates: the plumbing, checked before any number is shown ──────────────────────────────────────

def gates(obs: dict):
    arms = obs.get("arms") or {}
    runs = _runs(obs)

    # 0. THE PROBE PRODUCED OBSERVATIONS AT ALL. Without this, a probe that died on its own
    #    argument parsing reaches the gates with an empty dict, and every gate below then reports
    #    on nothing. Several of them would have said "yes" to that, which is the vacuity this
    #    toolkit exists to refuse. Observed for real: a mistyped flag exited the probe with rc=2
    #    and the mount gate passed on an empty census.
    yield ("the probe produced observations",
           not obs.get("_missing_output") and not obs.get("_parse_error")
           and not obs.get("fatal") and obs.get("_probe_rc") in (0, None) and bool(runs),
           f"probe rc {obs.get('_probe_rc')}, missing output "
           f"{bool(obs.get('_missing_output'))}, fatal {str(obs.get('fatal'))[:200]!r}, "
           f"{len(runs)} session(s) recorded")

    # 1. both arms installed
    installed = {a: (arms.get(a, {}).get("install", {}) or {}).get("rc") for a in ARMS}
    yield ("both arms installed",
           all(v == 0 for v in installed.values()),
           f"install rc {installed}")

    # 2. the two arms are two different commits, and the pre arm is the one that was asked for
    commits = {a: (arms.get(a, {}).get("clone", {}) or {}).get("commit") for a in ARMS}
    refs = obs.get("refs") or {}
    pre_ref = str(refs.get("pre") or "")
    pre_c = commits.get("pre") or ""
    head_c = commits.get("head") or ""
    yield ("the two arms are two different commits",
           bool(pre_c) and bool(head_c) and pre_c != head_c
           and (not pre_ref or pre_c.startswith(pre_ref) or pre_ref.startswith(pre_c[:9])),
           f"pre {pre_c[:9]} (asked for {pre_ref}), head {head_c[:9]}")

    # 3. a production bundle per arm, and NOT the same bytes twice. Assets cache on origin plus
    #    path and vite content-hashes on source change, so two arms reporting one bundle hash is
    #    the signature of a rebuilt bundle that kept its URL.
    dists = {a: (arms.get(a, {}).get("dist") or {}) for a in ARMS}
    hashes = {a: dists[a].get("bundle_hash") for a in ARMS}
    prod = all(dists[a].get("index_html") and (dists[a].get("asset_files") or 0) > 0
               for a in ARMS)
    yield ("a production bundle per arm, and the two differ",
           prod and hashes["pre"] is not None and hashes["pre"] != hashes["head"],
           f"pre {hashes['pre']}, head {hashes['head']}, "
           f"assets {dists['pre'].get('asset_files')}/{dists['head'].get('asset_files')}")

    # 4. one unsloth-zoo across both arms. A zoo mismatch makes this a two-variable experiment.
    zoo = {a: (arms.get(a, {}).get("zoo") or {}) for a in ARMS}
    zid = {a: (zoo[a].get("commit_id") or zoo[a].get("version")) for a in ARMS}
    yield ("one unsloth-zoo across both arms",
           bool(zid["pre"]) and zid["pre"] == zid["head"],
           f"pre {zid['pre']}, head {zid['head']}; "
           f"zoo main at start {(obs.get('zoo_main_at_start') or {}).get('sha')}")

    # 5. one instrument. `pre` has no studiobench, so both arms are driven by the head clone's
    #    pacer, seeder, corpus and scene. An arm measured with the harness that shipped alongside
    #    it differs by the change AND by the measuring device.
    sb_roots = {(_payload(r).get("run_meta") or {}).get("instrument_sb_root") for r in runs}
    corpora = {(_payload(r).get("run_meta") or {}).get("corpus_hash") for r in runs}
    sb_roots.discard(None)
    corpora.discard(None)
    yield ("one pinned instrument and one corpus for both arms",
           len(sb_roots) == 1 and len(corpora) == 1,
           f"sb_root {sorted(sb_roots)}, corpus {sorted(corpora)}")

    # 6. every planned session completed
    planned = obs.get("plan") or []
    done = [r for r in runs if _ok(r)]
    failed = [f"{r.get('arm')}/{r.get('rung')}/rep{r.get('rep')}"
              f"({str(_payload(r).get('error'))[:60]})" for r in runs if not _ok(r)]
    yield ("every planned session completed",
           bool(planned) and len(done) == len(planned) and not failed,
           f"{len(done)}/{len(planned)} ok" + (f"; failed {failed}" if failed else ""))

    # 7. every session really was WebKitGTK
    bad_engine = []
    for r in runs:
        ep = _payload(r).get("engine_probe") or {}
        if not (ep.get("is_webkit_gtk_ua") and ep.get("has_webkit_message_handlers")
                and not ep.get("has_chrome")):
            bad_engine.append(f"{r.get('arm')}/{r.get('rung')}/rep{r.get('rep')}")
    uas = {(_payload(r).get("ua") or "")[:70] for r in runs if _ok(r)}
    yield ("every session really was WebKitGTK",
           bool(runs) and not bad_engine,
           (f"UA {sorted(uas)}" if not bad_engine else f"not WebKitGTK: {bad_engine}"))

    # 8. one bundle per arm across every rung, and it is the bundle that arm built
    drift = []
    for r in runs:
        got = (_payload(r).get("run_meta") or {}).get("bundle_hash")
        want = r.get("expected_bundle_hash")
        if got and want and got != want:
            drift.append(f"{r.get('arm')}/{r.get('rung')}/rep{r.get('rep')}: {got} != {want}")
    per_arm = {a: {(_payload(r).get("run_meta") or {}).get("bundle_hash")
                   for r in runs if r.get("arm") == a} for a in ARMS}
    yield ("one bundle per arm across every rung",
           not drift and all(len(per_arm[a] - {None}) == 1 for a in ARMS),
           f"pre {sorted(per_arm['pre'])}, head {sorted(per_arm['head'])}"
           + (f"; drift {drift}" if drift else ""))

    # 9. the seeded thread really mounted: the DOM has to grow up the ladder
    # The verdict is only earned by comparisons that actually happened: an empty `growth` map is a
    # NO, not a yes. It used to be computed with a flag that started true, and an empty
    # observations dict then produced "yes, element growth {}".
    growth = {}
    for a in ARMS:
        base = _mean([_census(r, "final").get("elements") for r in _cells(obs, a, "0K")])
        for rung in _rungs(obs):
            if rung == "0K":
                continue
            top = _mean([_census(r, "final").get("elements") for r in _cells(obs, a, rung)])
            if base and top:
                g = top / base
                growth[f"{a}/{rung}"] = round(g, 1)
            else:
                growth[f"{a}/{rung}"] = None
    ok_growth = bool(growth) and all(
        isinstance(v, (int, float)) and v >= MOUNT_MIN_GROWTH for v in growth.values())
    yield (f"the seeded thread really mounted (DOM grows >= {MOUNT_MIN_GROWTH:g}x over 0K)",
           ok_growth, f"element growth {growth}")

    # 10. at least two repetitions per arm per rung, so a difference has a floor
    counts = {f"{a}/{rung}": len(_cells(obs, a, rung))
              for a in ARMS for rung in _rungs(obs)}
    yield ("every rung measured at least twice on each arm",
           bool(counts) and all(v >= 2 for v in counts.values()),
           f"sessions {counts}")

    # 11. the arm order really was rotated
    orders = {}
    for r in runs:
        orders.setdefault(str(r.get("rep")), []).append(r.get("arm"))
    firsts = {rep: (arms_[0] if arms_ else None) for rep, arms_ in orders.items()}
    # No escape hatch for "there was only one repetition". A single repetition cannot rotate, and
    # a gate that passes because it had nothing to check is the vacuity this toolkit exists to
    # refuse.
    yield ("arm order rotated across repetitions",
           len(firsts) >= 2 and len(set(firsts.values())) > 1,
           f"first arm per repetition {firsts}")


# ── the readback, and the two states it can be in ────────────────────────────────────────────────

def _readback(run: dict) -> dict:
    p = _payload(run)
    return p.get("readback_final") or p.get("readback_mounted") or {}


def _engine_gate_passes(obs: dict) -> tuple[bool, str]:
    """Does THIS runner's engine support the probe `gateOnEngine` keys containment on?"""
    vals = {}
    for r in _runs(obs):
        rb = _readback(r)
        if "css_supports_anchor_name" in rb:
            vals[f"{r.get('arm')}/{r.get('rung')}/rep{r.get('rep')}"] = \
                rb.get("css_supports_anchor_name")
    distinct = set(map(str, vals.values()))
    ok = bool(vals) and all(v is True for v in vals.values())
    return ok, f"CSS.supports(\"anchor-name: --unsloth-probe\") = {sorted(distinct)}"


def _head_is_fixed_state(obs: dict) -> tuple[bool, str]:
    """Did the head arm really boot into the shipped fixed state, with nothing forced?"""
    attrs, shells, forced = {}, {}, {}
    for r in _runs(obs):
        if r.get("arm") != "head" or not _ok(r):
            continue
        rb = _readback(r)
        key = f"{r.get('rung')}/rep{r.get('rep')}"
        attrs[key] = rb.get("math_block_attribute")
        shells[key] = rb.get("deferred_fence_shells")
        forced[key] = rb.get("math_runtime_global")
    on = bool(attrs) and all(v == "on" for v in attrs.values())
    # NOTHING FORCED is part of the claim: a runtime global set by anyone would mean the measured
    # configuration is not the shipped one.
    clean = all(v in (None, "threw") for v in forced.values())
    return on and clean, (f"data-math-block-containment {attrs}; deferred fence shells {shells}; "
                          f"runtime override {sorted(set(map(str, forced.values())))}")


def _pre_is_old_state(obs: dict) -> tuple[bool, str]:
    attrs, shells = {}, {}
    for r in _runs(obs):
        if r.get("arm") != "pre" or not _ok(r):
            continue
        rb = _readback(r)
        key = f"{r.get('rung')}/rep{r.get('rep')}"
        attrs[key] = rb.get("math_block_attribute")
        shells[key] = rb.get("deferred_fence_shells")
    # `bool(attrs)` is load-bearing: with no scored `pre` session both `all()` calls are true of
    # an empty map, and the function would certify the old state from no evidence at all.
    old = bool(attrs) and all(v is None for v in attrs.values()) \
        and all((v or 0) == 0 for v in shells.values())
    return old, f"data-math-block-containment {attrs}; deferred fence shells {shells}"


def _liveness(obs: dict) -> tuple[bool, list[str], dict]:
    """The jam control, per session. This is what licenses every other number in the run."""
    rows, bad = {}, []
    for r in _runs(obs):
        if not _ok(r):
            continue
        lv = _payload(r).get("liveness") or {}
        key = f"{r.get('arm')}/{r.get('rung')}/rep{r.get('rep')}"
        rows[key] = lv
        drop = lv.get("drop_fraction")
        if not isinstance(drop, (int, float)) or drop < LIVENESS_MIN_DROP:
            bad.append(f"{key}: clean {lv.get('clean_fps')} -> jammed {lv.get('jammed_fps')} "
                       f"(drop {drop})")
    return (bool(rows) and not bad), bad, rows


# ── the collapse itself ─────────────────────────────────────────────────────────────────────────

def _top_rung(obs: dict) -> str:
    rr = _rungs(obs)
    return rr[-1] if rr else ""


def _busy_mean(obs: dict, arm: str, rung: str, phase: str):
    return _mean([_sub(_phase(r, phase), "busy").get("busy_pct")
                  for r in _cells(obs, arm, rung)])


def _base_collapses(obs: dict) -> tuple[bool, str]:
    top = _top_rung(obs)
    fps = _fps_values(obs, "pre", top, "scroll")
    busy = _busy_mean(obs, "pre", top, "scroll")
    zero = _mean(_fps_values(obs, "pre", "0K", "idle"))
    m = _mean(fps)
    if m is None:
        return False, f"pre has no scored scroll window at {top}"
    if not isinstance(busy, (int, float)):
        # SAYING THE WRONG THING ABOUT THE SUBJECT IS WORSE THAN SAYING NOTHING. busy_pct is null
        # whenever the setTimeout clamp failed to calibrate, and the observed clamp on this venue
        # is 8 ms against a 10 ms ceiling, so this is two milliseconds away from happening. With
        # the old single condition, a pre arm reading 3.2 fps -- the collapse, verbatim -- was
        # reported as "the pre arm did NOT exhibit the reported collapse".
        return None, (f"pre scroll at {top} reads {_reps_str(fps)} fps but its busy percentage "
                      f"could not be read, because the setTimeout clamp did not calibrate. A "
                      f"frame rate cannot be qualified as a collapse without it. This is an "
                      f"INSTRUMENT failure, not a non-reproduction")
    ok = m <= BASE_COLLAPSE_MAX_FPS and busy >= BASE_COLLAPSE_MIN_BUSY
    return ok, (f"pre scroll at {top}: {_reps_str(fps)} fps (mean {m:.1f}), busy "
                f"{_fmt(busy)}%, against pre idle at 0K {_fmt(zero)} fps. Bar: at most "
                f"{BASE_COLLAPSE_MAX_FPS:g} fps AND at least {BASE_COLLAPSE_MIN_BUSY:g}% busy, "
                f"because a low frame rate on a page with nothing to draw is not a collapse")


def _regressions(obs: dict) -> list[str]:
    """Head worse than pre, computed BEFORE anything else and reported first."""
    out = []
    for rung in _rungs(obs):
        for phase in SCORED_PHASES:
            pre = _fps_values(obs, "pre", rung, phase)
            head = _fps_values(obs, "head", rung, phase)
            if len(pre) < 1 or len(head) < 1:
                continue
            mp, mh = _mean(pre), _mean(head)
            if mp is None or mh is None or mp < REGRESSION_MIN_ABS_FPS:
                continue
            loss = 1 - (mh / mp)
            if loss < REGRESSION_MIN_LOSS:
                continue
            # The repetitions must agree about the direction, or this is one slow draw.
            if not (max(head) < min(pre)):
                out.append(f"{rung} {phase}: head {_reps_str(head)} against pre {_reps_str(pre)} "
                           f"fps, {loss * 100:.0f}% worse on the mean but the repetitions "
                           f"OVERLAP, so this is not settled")
                continue
            out.append(f"{rung} {phase}: head {_reps_str(head)} against pre {_reps_str(pre)} fps, "
                       f"{loss * 100:.0f}% WORSE")
    for name in ACTIONS:
        for rung in _rungs(obs):
            pre = [_action(r, name).get("eff_fps") for r in _cells(obs, "pre", rung)]
            head = [_action(r, name).get("eff_fps") for r in _cells(obs, "head", rung)]
            pre = [x for x in pre if isinstance(x, (int, float))]
            head = [x for x in head if isinstance(x, (int, float))]
            mp, mh = _mean(pre), _mean(head)
            if mp is None or mh is None or mp < REGRESSION_MIN_ABS_FPS:
                continue
            if 1 - (mh / mp) >= REGRESSION_MIN_LOSS and max(head) < min(pre):
                out.append(f"{rung} action:{name}: head {_reps_str(head)} against pre "
                           f"{_reps_str(pre)} fps, {(1 - mh / mp) * 100:.0f}% WORSE")
    return out


# ── the report ──────────────────────────────────────────────────────────────────────────────────

def _cell_line(obs: dict, arm: str, rung: str, phase: str) -> str:
    cells = _cells(obs, arm, rung)
    fps = _fps_values(obs, arm, rung, phase)
    if not cells or not fps:
        return "n/a"
    busy = _busy_mean(obs, arm, rung, phase)
    robust = _mean([_sub(_phase(r, phase), "robust").get("blocked_ms_per_frame") for r in cells])
    mean_blocked = _mean([_phase(r, phase).get("blocked_ms_per_frame") for r in cells])
    over100 = _mean([_sub(_phase(r, phase), "raf").get("frames_over_100_pct") for r in cells])
    worst = max([_sub(_phase(r, phase), "raf").get("max_ms") or 0 for r in cells])
    # A window whose mean is far above its stall-stripped mean CAUGHT A STALL. That is a fact
    # about the window, not about the arm, and a reader who is not told which of the two numbers
    # to weigh will weigh the wrong one. Flagged rather than silently averaged away.
    stall = ""
    if isinstance(robust, float) and isinstance(mean_blocked, float) and robust > 0 \
            and mean_blocked / robust > MEAN_OVER_ROBUST_MAX:
        stall = (f" **caught a stall** (mean is {mean_blocked / robust:.1f}x the stall-stripped "
                 f"mean)")
    return (f"**{_reps_str(fps)}** fps / {_fmt(busy)}% busy / {_fmt(mean_blocked)} mean, "
            f"{_fmt(robust)} stall-stripped ms per frame / {_fmt(over100)}% of frames > 100 ms / "
            f"worst {_fmt(worst, 0)} ms{stall}")


def table(obs: dict) -> str:
    lines: list[str] = []

    regs = _regressions(obs)
    lines.append("### Regressions first")
    lines.append("")
    if regs:
        lines.append("`head` is worse than `pre` in these cells:")
        lines.append("")
        for r in regs:
            lines.append(f"- {r}")
    else:
        lines.append(f"No cell has `head` worse than `pre` by more than "
                     f"{REGRESSION_MIN_LOSS * 100:.0f}% with the repetitions agreeing, at any "
                     f"rung, in any of {', '.join(SCORED_PHASES)} or "
                     f"{', '.join('action:' + a for a in ACTIONS)}.")
    lines.append("")

    lines.append("### The ladder, per rung, per phase, `pre` against `head`")
    lines.append("")
    lines.append("Every cell reads: effective fps over WALL TIME per repetition / busy% / mean and "
                 "stall-stripped blocked ms per frame / share of frames over 100 ms / worst frame. "
                 "The frame rate is the headline because it is the only channel that both moves "
                 "under the jam control and means frames a user would have seen.")
    lines.append("")
    lines.append("| rung | phase | pre | head |")
    lines.append("|---|---|---|---|")
    for rung in _rungs(obs):
        for phase in SCORED_PHASES:
            lines.append(f"| {rung} | {phase} | {_cell_line(obs, 'pre', rung, phase)} "
                         f"| {_cell_line(obs, 'head', rung, phase)} |")
    lines.append("")

    lines.append("### The action windows that carried the original complaint")
    lines.append("")
    lines.append("| rung | action | pre | head |")
    lines.append("|---|---|---|---|")
    for rung in _rungs(obs):
        for name in ACTIONS:
            row = []
            for arm in ARMS:
                cells = _cells(obs, arm, rung)
                fps = [_action(r, name).get("eff_fps") for r in cells]
                fps = [x for x in fps if isinstance(x, (int, float))]
                na = [bool(_action(r, name).get("not_applicable")) for r in cells]
                if not fps:
                    row.append("not applicable at this rung" if any(na) else "n/a")
                    continue
                busy = _mean([_sub(_action(r, name), "busy").get("busy_pct") for r in cells])
                sync = _mean([_action(r, name).get("app_sync_ms") for r in cells])
                worst = max([_sub(_action(r, name), "raf").get("max_ms") or 0 for r in cells])
                row.append(f"**{_reps_str(fps)}** fps / {_fmt(busy)}% busy / "
                           f"{_fmt(sync)} ms in the handler / worst {_fmt(worst, 0)} ms")
            lines.append(f"| {rung} | {name} | {row[0]} | {row[1]} |")
    lines.append("")

    # ── IS THE HEAD ARM RENDERING THE SAME DOCUMENT? ──
    #
    # This section is not decoration and it can invalidate the one above. `head` carries
    # progressive message mounting, which `pre` does not have at all, so a faster head arm may be
    # a head arm that put less on the page. A frame rate is only comparable across two builds that
    # are showing comparable amounts of content, and the honest way to present that is to put the
    # census next to the frame rates rather than in a footnote.
    lines.append("### What each arm actually put on the page")
    lines.append("")
    lines.append("A faster arm that rendered less is not the same finding as a faster arm that "
                 "rendered the same thing. `head` carries progressive message mounting, which "
                 "`pre` does not have, so this is read before the table above is believed.")
    lines.append("")
    fields = [("elements", "elements"), ("messages", "messages"),
              ("assistant_chars", "assistant chars"), ("code_blocks", "code blocks"),
              ("highlight_spans", "pre span"), ("scroll_height", "scrollHeight")]
    lines.append("| rung | measure | pre | head | head / pre |")
    lines.append("|---|---|---|---|---|")
    for rung in _rungs(obs):
        for key, label in fields:
            vals = {}
            for arm in ARMS:
                vals[arm] = _mean([_census(r, "final").get(key)
                                   for r in _cells(obs, arm, rung)])
            ratio = (vals["head"] / vals["pre"]) if (vals["pre"] and vals["head"]) else None
            lines.append(f"| {rung} | {label} | {_fmt(vals['pre'], 0)} | "
                         f"{_fmt(vals['head'], 0)} | {_fmt(ratio, 2)} |")
    lines.append("")

    # ── the jam control, beside the table it licenses ──
    lines.append("### The jammed positive control, per rung, per arm, per repetition")
    lines.append("")
    lines.append("Six seconds of a deliberately blocked main thread inside every session. If a row "
                 "does not fall, nothing measured in that session means anything. `1000/p50` is "
                 "shown on the SAME series to make the point that it is blind: it barely moves.")
    lines.append("")
    lines.append("| arm | rung | rep | clean fps | jammed fps | drop | 1000/p50 clean -> jammed |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in _runs(obs):
        if not _ok(r):
            continue
        lv = _payload(r).get("liveness") or {}
        drop = lv.get("drop_fraction")
        flag = "" if isinstance(drop, (int, float)) and drop >= LIVENESS_MIN_DROP else " **DID NOT RESOLVE**"
        lines.append(f"| {r.get('arm')} | {r.get('rung')} | {r.get('rep')} | "
                     f"{_fmt(lv.get('clean_fps'))} | {_fmt(lv.get('jammed_fps'))} | "
                     f"{_fmt(drop, 2)}{flag} | "
                     f"{_fmt(lv.get('clean_fps_p50'))} -> {_fmt(lv.get('jammed_fps_p50'))} |")
    lines.append("")

    # ── the readback, verbatim, per rung ──
    lines.append("### What each arm actually was, read back out of the running page")
    lines.append("")
    lines.append("Nothing was forced on either arm. These are the shipped defaults as the page "
                 "reports them.")
    lines.append("")
    lines.append("| arm | rung | rep | anchor-name probe | data-math-block-containment | "
                 "content-visibility auto on .aui-math-block | deferred fence shells | "
                 "pre span |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in _runs(obs):
        if not _ok(r):
            continue
        rb = _readback(r)
        cv = rb.get("content_visibility_math_blocks") or {}
        lines.append(f"| {r.get('arm')} | {r.get('rung')} | {r.get('rep')} | "
                     f"`{rb.get('css_supports_anchor_name')}` | "
                     f"`{rb.get('math_block_attribute')}` | "
                     f"{cv.get('auto')}/{cv.get('sampled')} sampled of {cv.get('matched')} | "
                     f"{rb.get('deferred_fence_shells')} | {rb.get('highlight_spans')} |")
    lines.append("")

    # ── provenance ──
    arms = obs.get("arms") or {}
    lines.append("### Provenance")
    lines.append("")
    for a in ARMS:
        e = arms.get(a) or {}
        cl = e.get("clone") or {}
        zoo = e.get("zoo") or {}
        lines.append(f"- `{a}`: unsloth `{(cl.get('commit') or '')[:9]}` "
                     f"({cl.get('commit_line', '')[10:80]}), bundle "
                     f"`{(e.get('dist') or {}).get('bundle_hash')}`, unsloth-zoo "
                     f"`{zoo.get('commit_id') or zoo.get('version')}`")
    inst = obs.get("instrument") or {}
    corpora = sorted({(_payload(r).get("run_meta") or {}).get("corpus_hash") for r in _runs(obs)}
                     - {None})
    lines.append(f"- instrument: one pinned checkout for both arms ({inst.get('from_arm')}), "
                 f"corpus {corpora}; `pre` carries studiobench: "
                 f"{inst.get('pre_has_studiobench')}")
    lines.append(f"- zoo main when the run started: "
                 f"`{(obs.get('zoo_main_at_start') or {}).get('sha')}`")
    lines.append(f"- both arms installed with `install.sh "
                 f"{' '.join(obs.get('install_args') or ['--local'])}`")
    lines.append("")

    lines.append("### What this run did NOT measure")
    lines.append("")
    lines.append("- Chromium, WKWebView, WebView2 and the Tauri Desktop shell. This is WebKitGTK "
                 "under Xvfb on one integrated GPU, which is the engine Studio and Desktop use on "
                 "Linux and nothing else.")
    lines.append("- Any rung above the top one measured here. A 1M rung is not bounded by this "
                 "run and must not be inferred from it.")
    lines.append("- First paint and mount, which are discarded as warm-up. A user's very first "
                 "frame after opening a 500K thread is specifically not what these numbers are "
                 "about.")
    lines.append("- Any window not listed above: no find-in-page, no print, no thread switching, "
                 "no resize, no multi-window.")
    lines.append("- Correctness. This says nothing about whether either build renders the thread "
                 "the same way; it measures how fast it does whatever it does.")
    return "\n".join(lines)


def verdict(obs: dict) -> tuple[str, str]:
    # 1. THE INSTRUMENT. Nothing else can be read past a jam control that did not resolve.
    live_ok, live_bad, _rows = _liveness(obs)
    if not live_ok:
        return "VOID", (
            "the jammed positive control did not resolve in "
            f"{len(live_bad)} session(s), so the frame channel was not shown to be able to "
            "report a blocked main thread and no reading from this run means anything: "
            + "; ".join(live_bad[:6]))

    # 2. THE ENGINE GATE. If containment was never on, a small delta is not a small fix.
    gate_ok, gate_ev = _engine_gate_passes(obs)
    if not gate_ok:
        return "FIX_NOT_ENGAGED_ON_THIS_ENGINE", (
            "this runner's WebKitGTK FAILS the probe that `math-block-mode.ts::gateOnEngine` "
            "keys containment on, so the head arm ran WITHOUT the scroll fix and the numbers "
            f"below are what this engine gives with the fix disabled by its own gate. {gate_ev}. "
            "Read no conclusion about the fix from this run")

    head_ok, head_ev = _head_is_fixed_state(obs)
    if not head_ok:
        return "VOID", (
            "the head arm did not boot into the shipped fixed state, or something forced it, so "
            f"what was measured is not what a user gets: {head_ev}")

    pre_ok, pre_ev = _pre_is_old_state(obs)
    if not pre_ok:
        return "VOID", (
            "the pre arm shows the shipped fixed state's own markers, so the two arms are not "
            f"the two states this run claims to compare: {pre_ev}")

    # 3. THE ONE RULE. A base that did not collapse cannot license a claim about a fix.
    base_ok, base_ev = _base_collapses(obs)
    if base_ok is None:
        return "INCONCLUSIVE", (
            "the base arm's collapse could not be qualified because the instrument's own clamp "
            f"did not calibrate. {base_ev}")
    if not base_ok:
        return "VOID", (
            "the pre arm did NOT exhibit the reported collapse, so this run did not reproduce "
            f"the problem and cannot speak to whether today's main fixes it. {base_ev}")

    # 4. what actually happened
    top = _top_rung(obs)
    pre_fps = _mean(_fps_values(obs, "pre", top, "scroll"))
    head_fps = _mean(_fps_values(obs, "head", top, "scroll"))
    ratio = (head_fps / pre_fps) if (pre_fps and head_fps) else None
    regs = _regressions(obs)
    settled_regs = [r for r in regs if "OVERLAP" not in r]
    tail = ""
    if settled_regs:
        tail = (f" {len(settled_regs)} cell(s) are WORSE on head and are listed first in the "
                f"table: " + "; ".join(settled_regs[:4]))

    if head_fps is None or pre_fps is None:
        return "VOID", f"no scored scroll window at {top} on one of the arms"

    if head_fps >= FIXED_MIN_FPS and ratio and ratio >= FIXED_MIN_RATIO:
        return "CONFIRMED", (
            f"the collapse reproduces on `pre` and is absent on `head`: scroll at {top} goes "
            f"{pre_fps:.1f} -> {head_fps:.1f} fps ({ratio:.2f}x), clearing the "
            f"{FIXED_MIN_FPS:g} fps bar." + tail)
    if ratio and ratio >= FIXED_MIN_RATIO:
        return "FIX_INCOMPLETE", (
            f"`head` improves the collapse but does not clear it: scroll at {top} goes "
            f"{pre_fps:.1f} -> {head_fps:.1f} fps ({ratio:.2f}x), still under the "
            f"{FIXED_MIN_FPS:g} fps bar." + tail)
    return "NO_BENEFIT", (
        f"`pre` collapses and `head` does not materially recover it: scroll at {top} goes "
        f"{pre_fps:.1f} -> {head_fps:.1f} fps"
        + (f" ({ratio:.2f}x)" if ratio else "") + "." + tail)


def observed_capabilities(obs: dict) -> dict:
    runs = _runs(obs)
    arms = obs.get("arms") or {}
    prod = all(((arms.get(a) or {}).get("dist") or {}).get("index_html") for a in ARMS)
    webkit = bool(runs) and all((_payload(r).get("engine_probe") or {}).get("is_webkit_gtk_ua")
                                for r in runs if _ok(r))
    return {
        "webkitgtk": webkit,
        "headless_display_server": bool((obs.get("xserver") or {}).get("display")),
        "studio_production_bundle": bool(prod),
        "studiobench_ladder": bool([r for r in runs if _ok(r)]),
    }
