#!/usr/bin/env python3
"""Judge: does any INTERMEDIATE commit of PR 9477 beat the tip?

Judges only. `probes/studio_sweep_probe.py` observes.

THE QUESTION IS NON-MONOTONIC, so the verdict cannot be a ratio between two arms. The endpoint run
already established there is no NET regression across `33d65ce99..e48dab138`: the tip beats the
authored commit on every metric at both rungs. What two points cannot show is a PEAK -- a commit
that regressed something a later commit then partially repaired would leave BOTH endpoints below
some point in the middle, and an endpoint comparison would report "no regression" while concealing
exactly that. So the object judged here is a CURVE over eleven points, and the verdict is whether
any point strictly inside it stands above the tip by more than the channel can manufacture.

WHAT "MORE THAN THE CHANNEL CAN MANUFACTURE" MEANS, AND WHY IT IS MEASURED RATHER THAN ASSUMED.
This design has two FREE NULLS, and they are the whole reason a peak can be called at all:

    p6 vs p7   adf308cfb changes one Python TEST file; the two `studio/frontend` trees are
               identical by tree hash and must build the same bundle
    p8 vs p9   e48dab138 merges lint tooling only; identical again, so the TIP of the PR is
               bit-identical in the browser to its parent

Two pairs that CANNOT differ, measured through the entire pipeline exactly as every other point is.
The gap each pair shows is not an estimate of noise, it IS noise, arm-to-arm, at this rung and this
number of repetitions. The peak threshold is taken from them. A design without such a pair has to
import a threshold from somewhere else and can always be accused of choosing it to get an answer;
here the data sets it, and if the nulls come apart the run says so and declines to call a peak
rather than calling one against a floor it no longer trusts.

A candidate must clear THREE things, because any one alone has already produced a false positive
somewhere in this campaign:
  * the null floor, above;
  * its OWN repetition spread, since arm A of the endpoint run swung 44.9 to 26.4 fps between two
    reps and a mean alone would have drawn a peak out of that one arm;
  * the tip's repetition spread, for the same reason in the other direction.

WHY THE TIP AND NOT THE LEFT ENDPOINT IS THE REFERENCE. The user's question is "does any commit in
the middle beat the tip" -- that is, would we be better off having stopped somewhere. p0 is the
authored commit and is already known to be worse than the tip. Comparing a candidate to p0 would
answer a different and easier question.

p0 AND p0g ARE NOT CANDIDATES. They are the same commit as each other and they are the left
ANCHOR, outside the span `33d65ce99..e48dab138` (which is exclusive of its left end). They are
carried to price `GRID_COLLAPSE_REASONING_ENABLED`, which flips false -> true at p1 and would
otherwise confound the first step of the curve with the merge that carries it. p9 is not a
candidate either: it IS the tip.
"""

from __future__ import annotations

TITLE = ("PR 9477 commit sweep 33d65ce99..e48dab138: is there a peak in the middle? "
         "action:reasoning_toggle_all at r100K, real WebKitGTK / gfx1151")
MODE = "capability"

NEEDS = ["webkitgtk", "headless_display_server", "studio_production_bundle", "studiobench_ladder"]

ACTION = "reasoning_toggle_all"
RUNG = "100K"

#: The eleven points, in commit order. p0/p0g are the anchor, p1..p9 are the span.
ANCHOR = ("p0", "p0g")
SPAN = ("p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9")
ARMS = ANCHOR + SPAN
TIP = "p9"

#: Pairs that CANNOT differ in the browser. The peak threshold comes from what they nevertheless
#: measure. Keep in sync with the probe's PREDICTED_NULLS; the gates assert the source-level claim.
FREE_NULLS = (("p6", "p7"), ("p8", "p9"))

#: The jammed control must fall at least this far below the unjammed tip, or the frame channel
#: cannot report a blocked main thread and no point on the curve means anything.
CONTROL_MIN_DROP_PCT = 25.0

#: A floor below which a "null gap" is treated as this channel's irreducible resolution rather
#: than as evidence the nulls are healthy. Two arms that must be identical reading within 0.5 fps
#: of each other is as good as this instrument gets; claiming a sub-0.5 fps peak would be claiming
#: resolution the nulls have never demonstrated.
MIN_FLOOR_FPS = 0.5


# ── accessors over payload["actions"] ────────────────────────────────────────────────────────────
def _runs(obs: dict) -> list[dict]:
    return [r for r in (obs.get("runs") or []) if (r.get("payload") or {}).get("ok")]


def _sel(obs: dict, arm: str, rung: str = RUNG) -> list[dict]:
    return [r for r in _runs(obs) if r.get("arm") == arm and r.get("rung") == rung]


def _action(payload: dict) -> dict:
    for a in payload.get("actions") or []:
        if a.get("name") == ACTION and not a.get("not_applicable"):
            return a
    return {}


def _eff_fps(payload: dict):
    a = _action(payload)
    n = (a.get("raf") or {}).get("n")
    el = a.get("elapsed_ms")
    return (1000.0 * n / el) if (n and el) else None


def _busy(payload: dict):
    return ((_action(payload).get("busy")) or {}).get("busy_pct")


def _worst(payload: dict):
    return (_action(payload).get("raf") or {}).get("max_ms")


def _spans(payload: dict):
    return (_action(payload).get("census_open") or {}).get("highlight_spans")


def _reasoning_chars(payload: dict):
    return (_action(payload).get("census_open") or {}).get("reasoning_chars")


def _vals(obs: dict, arm: str, fn, rung: str = RUNG):
    return [v for v in (fn(r["payload"]) for r in _sel(obs, arm, rung)) if v is not None]


def _mean(xs):
    return (sum(xs) / len(xs)) if xs else None


def _spread(xs):
    return (max(xs) - min(xs)) if len(xs) >= 2 else 0.0


def _fmt(xs, p=1):
    return "[" + ", ".join(f"{x:.{p}f}" for x in xs) + "]"


def _floor(obs: dict) -> tuple[float, list[str], bool]:
    """The peak threshold, taken from the pairs that cannot differ.

    Returns (floor_fps, notes, trustworthy). `trustworthy` is False when a null pair is missing or
    its two arms did not build the same bundle, in which case the floor was never established and
    no peak may be called against it.
    """
    gaps, notes, ok = [], [], True
    st = obs.get("states") or {}
    for a, b in FREE_NULLS:
        ma, mb = _mean(_vals(obs, a, _eff_fps)), _mean(_vals(obs, b, _eff_fps))
        ha = (st.get(a) or {}).get("exported_bundle_hash")
        hb = (st.get(b) or {}).get("exported_bundle_hash")
        if ma is None or mb is None:
            notes.append(f"{a} vs {b}: not measured, so this null contributes no floor")
            ok = False
            continue
        if ha != hb:
            notes.append(f"{a} vs {b}: bundles differ ({ha} vs {hb}) though the source trees are "
                         f"identical, so the build is not reproducible and this null is void")
            ok = False
            continue
        gaps.append(abs(ma - mb))
        notes.append(f"{a} vs {b}: {ma:.1f} vs {mb:.1f} fps, gap {abs(ma - mb):.2f}, same bundle")
    floor = max(gaps) if gaps else 0.0
    return max(floor, MIN_FLOOR_FPS), notes, (ok and bool(gaps))


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out: list[tuple[str, bool, str]] = []
    st = obs.get("states") or {}
    runs, ok = obs.get("runs") or [], _runs(obs)
    built = obs.get("arms_built") or []

    out.append(("a display server was obtained", bool((obs.get("xserver") or {}).get("display")),
                obs.get("fatal") or str((obs.get("xserver") or {}).get("display"))))

    # ASKED versus LANDED, on every point. A shallow clone that cannot resolve a PR-only commit
    # leaves HEAD on the default branch tip and every later step succeeds against the wrong tree;
    # that has already turned two arms of this campaign silently into main.
    mism = [n for n in ARMS if not (st.get(n) or {}).get("checkout_ok")]
    out.append(("every point checked out the commit it was asked for", not mism,
                "; ".join(f"{n}: asked {str((st.get(n) or {}).get('ref'))[:9]} landed "
                          f"{str((st.get(n) or {}).get('ref_landed') or (st.get(n) or {}).get('commit'))[:9]}"
                          for n in ARMS)))

    # PAGINATION LIVE AT EVERY POINT, read out of the source by resolving `paginateReasoning={X}`
    # through the flag file. This is the condition that makes the eleven points one curve rather
    # than two curves spliced at p4, and the commit's position in the list is never trusted for it.
    pag = {n: (st.get(n) or {}).get("pagination_effective") for n in ARMS}
    out.append(("pagination is live at every point, established from the source",
                all(pag[n] == "true" for n in ARMS),
                ", ".join(f"{n}={pag[n]}" for n in ARMS)))

    # The second reasoning flag, recorded so the first step of the curve is interpretable.
    grid = {n: (st.get(n) or {}).get("grid_collapse") for n in ARMS}
    grid_ok = (grid.get("p0") == "false" and grid.get("p0g") == "true"
               and all(grid.get(n) == "true" for n in SPAN))
    out.append(("GRID_COLLAPSE_REASONING_ENABLED is false at p0, forced true at p0g, and true "
                "across the whole span", grid_ok, ", ".join(f"{n}={grid[n]}" for n in ARMS)))

    # p0 and p0g are the same commit and must differ ONLY by that literal: same source tree,
    # different bundle. A shared bundle means the edit never reached the compiler and the arm that
    # prices grid collapse is silently a duplicate of p0.
    g = obs.get("build", {}).get("grid_edit_took") or obs.get("grid_edit_took") or {}
    h0 = (st.get("p0") or {}).get("exported_bundle_hash")
    h0g = (st.get("p0g") or {}).get("exported_bundle_hash")
    out.append(("the grid-collapse edit reached the bundle (p0 and p0g are one commit, two "
                "bundles)", bool(h0) and bool(h0g) and h0 != h0g,
                f"p0={h0} p0g={h0g}" + (f" {g}" if g else "")))

    # THE FREE NULLS, as a SOURCE claim first: identical trees must give identical bundles, or
    # nothing small anywhere on this curve can be read.
    nn = []
    nulls_ok = True
    for a, b in FREE_NULLS:
        ta, tb = (st.get(a) or {}).get("frontend_tree_hash"), (st.get(b) or {}).get("frontend_tree_hash")
        ba, bb = (st.get(a) or {}).get("exported_bundle_hash"), (st.get(b) or {}).get("exported_bundle_hash")
        same_t, same_b = (ta is not None and ta == tb), (ba is not None and ba == bb)
        nn.append(f"{a}/{b}: source {'same' if same_t else 'DIFFER'}, bundle "
                  f"{'same' if same_b else 'DIFFER'}")
        if not (same_t and same_b):
            nulls_ok = False
    out.append(("the two commits that cannot change a pixel built identical bundles", nulls_ok,
                "; ".join(nn)))

    out.append(("every point installed a production bundle",
                all((st.get(n) or {}).get("dist", {}).get("index_html") for n in ARMS),
                "; ".join(f"{n}: assets={(st.get(n) or {}).get('dist', {}).get('asset_files')}"
                          for n in ARMS)))

    # The dist that was measured is the dist that was built.
    moved = [n for n in built if (st.get(n) or {}).get("bundle_hash_matches_build") is False]
    out.append(("every dist arrived from the build job unchanged", not moved,
                "; ".join(f"{n}: built {(st.get(n) or {}).get('exported_bundle_hash')} measured "
                          f"{(st.get(n) or {}).get('bundle_hash_at_measure')}" for n in moved)
                or f"{len(built)} dists hash-matched"))

    # ONE PINNED INSTRUMENT FOR EVERY POINT, by resolved path AND by content. An arm built from a
    # 21 August commit measured with that commit's own studiobench would make the ruler co-vary
    # with the subject, and p0's tree has no studiobench at all so it would not even fail loudly.
    files = {(r["payload"].get("run_meta") or {}).get("instrument_pacer_file") for r in ok}
    hashes = {(r["payload"].get("run_meta") or {}).get("instrument_hash") for r in ok}
    ipath = (obs.get("instrument") or {}).get("path") or ""
    inside = all(f and ipath and str(f).startswith(str(ipath)) for f in files)
    out.append(("every measurement used ONE pinned instrument, outside every point's own tree",
                len(files) == 1 and len(hashes) == 1 and inside,
                f"pacer files={sorted(str(f) for f in files)}, instrument hashes={sorted(hashes)}, "
                f"instrument clone={ipath}"))

    corpora = {(r["payload"].get("run_meta") or {}).get("corpus_hash") for r in ok}
    out.append(("every measurement used one frozen corpus", len(corpora) == 1,
                f"corpus hashes={sorted(str(c)[:16] for c in corpora)}"))

    out.append(("every measurement completed", bool(runs) and len(ok) == len(runs),
                f"{len(ok)}/{len(runs)} ok" + ("" if len(ok) == len(runs) else "; " + "; ".join(
                    f"{r.get('arm')}#{r.get('rep')}: "
                    f"{str((r.get('payload') or {}).get('error') or r.get('rc'))[:110]}"
                    for r in runs if r not in ok)[:600])))

    reps = {n: len(_sel(obs, n)) for n in ARMS}
    out.append(("every point was measured at least twice, so it has a spread and not just a mean",
                all(v >= 2 for v in reps.values()),
                ", ".join(f"{n}={v}" for n, v in reps.items())))

    # THE JAMMED POSITIVE CONTROL. If the channel does not fall a long way under a blocked main
    # thread it cannot report one, and no point on the curve means anything.
    jam, tip = _mean(_vals(obs, "JAM", _eff_fps)), _mean(_vals(obs, TIP, _eff_fps))
    drop = (100.0 * (tip - jam) / tip) if (jam is not None and tip) else None
    out.append((f"the jammed control fell at least {CONTROL_MIN_DROP_PCT:.0f}% below the tip",
                drop is not None and drop >= CONTROL_MIN_DROP_PCT,
                f"jam {jam:.2f} fps vs tip {tip:.2f} fps, a {drop:.1f}% fall"
                if drop is not None else "the control or the tip did not measure"))

    floor, notes, trust = _floor(obs)
    out.append(("the peak threshold was established by pairs that cannot differ", trust,
                f"floor {floor:.2f} fps from: " + "; ".join(notes)))
    return out


def _table(obs: dict) -> list[tuple]:
    rows = []
    for n in ARMS:
        f = _vals(obs, n, _eff_fps)
        rows.append((n, _mean(f), _spread(f), f,
                     _mean(_vals(obs, n, _busy)), _mean(_vals(obs, n, _worst)),
                     _mean(_vals(obs, n, _spans)), _mean(_vals(obs, n, _reasoning_chars))))
    return rows


def verdict(obs: dict) -> tuple[str, str]:
    if obs.get("fatal"):
        return "INCONCLUSIVE", str(obs["fatal"])[:400]
    if not _runs(obs):
        return "INCONCLUSIVE", "no measurement completed, so there is no curve to read"

    tip_f = _vals(obs, TIP, _eff_fps)
    tip_mean = _mean(tip_f)
    if tip_mean is None:
        return "INCONCLUSIVE", f"the tip ({TIP}) did not measure, so there is nothing to beat"

    floor, notes, trust = _floor(obs)
    if not trust:
        return "INCONCLUSIVE", ("the two commits that cannot change a pixel did not both measure "
                                "as identical builds, so this run never established what gap the "
                                "channel can manufacture and must not call a peak against a floor "
                                "it cannot justify: " + "; ".join(notes))

    tip_spread = _spread(tip_f)
    # CANDIDATES are the points strictly INSIDE the span: not the anchor, which is outside it and
    # already known to be worse, and not the tip, which is the reference.
    cands = []
    for n in SPAN:
        if n == TIP:
            continue
        f = _vals(obs, n, _eff_fps)
        m = _mean(f)
        if m is None:
            continue
        margin = m - tip_mean
        # Three hurdles, each of which alone has produced a false positive in this campaign.
        beats = margin > floor and margin > _spread(f) and margin > tip_spread
        cands.append((margin, n, m, _spread(f), f, beats))
    if not cands:
        return "INCONCLUSIVE", "no intermediate point measured, so the middle of the span is blank"

    cands.sort(reverse=True)
    winners = [c for c in cands if c[5]]
    best = cands[0]

    curve = "; ".join(f"{n}={m:.1f}" for _, n, m, _s, _f, _b in
                      sorted(cands, key=lambda c: SPAN.index(c[1])))
    anchor = "; ".join(f"{n}={_mean(_vals(obs, n, _eff_fps)):.1f}"
                       for n in ANCHOR if _mean(_vals(obs, n, _eff_fps)) is not None)
    base = (f"tip {TIP} = {tip_mean:.1f} fps (reps {_fmt(tip_f)}, spread {tip_spread:.2f}); "
            f"span {curve}; anchor {anchor}; noise floor {floor:.2f} fps from the two pairs that "
            f"cannot differ ({'; '.join(notes)})")

    if winners:
        margin, n, m, sp, f, _ = winners[0]
        return "PEAK", (f"{n} beats the tip: {m:.1f} vs {tip_mean:.1f} fps, +{margin:.1f}, which "
                        f"clears the {floor:.2f} fps null floor, its own rep spread ({sp:.2f}) and "
                        f"the tip's ({tip_spread:.2f}). Reps {_fmt(f)}. "
                        + (f"{len(winners)} points clear all three hurdles: "
                           f"{[w[1] for w in winners]}. " if len(winners) > 1 else "")
                        + base)

    margin, n, m, sp, f, _ = best
    return "NO_PEAK", (f"no intermediate commit beats the tip. The best of them, {n}, reads "
                       f"{m:.1f} fps against the tip's {tip_mean:.1f}, a margin of {margin:+.1f} "
                       f"fps, which does not clear the {floor:.2f} fps floor set by the pairs that "
                       f"cannot differ. The tip is the best point on this span, so the endpoint "
                       f"comparison was not hiding a peak. " + base)


def observed_capabilities(obs: dict) -> dict[str, bool]:
    st = obs.get("states") or {}
    return {
        "webkitgtk": bool(_runs(obs)),
        "headless_display_server": bool((obs.get("xserver") or {}).get("display")),
        "studio_production_bundle": all(
            (st.get(n) or {}).get("dist", {}).get("index_html") for n in ARMS),
        "studiobench_ladder": bool(_runs(obs)),
    }


def table(obs: dict) -> str:
    """The curve itself, which is the deliverable. Per-rep values, never means alone: the endpoint
    run had an arm swing 44.9 to 26.4 fps between two repetitions, and a table of means would have
    shown a clean peak that was one arm disagreeing with itself."""
    st = obs.get("states") or {}
    lines = [f"### The curve: {ACTION} at r{RUNG}", "",
             "| point | commit | what | eff fps | spread | per-rep | busy % | worst ms | "
             "`pre span` | reasoning chars | bundle |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    jf = _vals(obs, "JAM", _eff_fps)
    for n, m, sp, f, bu, wo, spn, rc in _table(obs):
        s = st.get(n) or {}
        lines.append(
            f"| {n} | `{str(s.get('commit'))[:9]}` | {str(s.get('what'))[:60]} | "
            f"{'-' if m is None else f'{m:.1f}'} | {sp:.2f} | {_fmt(f)} | "
            f"{'-' if bu is None else f'{bu:.0f}'} | {'-' if wo is None else f'{wo:,.0f}'} | "
            f"{'-' if spn is None else f'{spn:,.0f}'} | "
            f"{'-' if rc is None else f'{rc:,.0f}'} | {s.get('exported_bundle_hash')} |")
    lines += ["", f"Jammed positive control: {_mean(jf) if jf else '-'} fps, per-rep {_fmt(jf)}.",
              ""]
    floor, notes, trust = _floor(obs)
    lines += [f"Noise floor {floor:.2f} fps, trustworthy={trust}. " + "; ".join(notes), ""]
    groups: dict = {}
    for n in ARMS:
        groups.setdefault((st.get(n) or {}).get("exported_bundle_hash"), []).append(n)
    shared = "; ".join(f"{h}: {v}" for h, v in groups.items() if len(v) > 1)
    lines += ["Points sharing a bundle cannot differ in the browser: " + (shared or "none"), ""]
    return "\n".join(lines)
