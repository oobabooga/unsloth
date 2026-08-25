#!/usr/bin/env python3
"""Criteria: does PR 9477 piece 4, reasoning pagination, help `action:reasoning_toggle`?

Judges only. Observations come from probes/studio_p4_probe.py.

THE STATISTIC IS EFFECTIVE RATE OVER WALL TIME, `1000 * raf.n / elapsed_ms`, never `raf.fps_p50`.
That is not a preference. On this exact scene, at r100K, with the main thread blocked 200 ms in
every 250 ms, `1000/p50` read 62.5 fps -- IDENTICAL to the unjammed run -- while the effective rate
fell 57.9 -> 12.9 and busy went 25% -> 87%. A median cannot see mass that has moved into the tail.
`raf.fps_p50` is present in every payload and is quoted nowhere in this file.

THE ACTION WINDOW IS NOT A PHASE. `payload["phases"]` holds only idle/scroll/stream/recover;
`action:reasoning_toggle` exists as a mark and its numbers live in `payload["actions"]` under
`name == "reasoning_toggle"`. Every accessor here reads that array. Rows with `not_applicable` are
skipped rather than counted as zero: an empty thread has no reasoning pane, which is not a fast
measurement, it is no measurement.

THE VOID RULE, and it is why this run reports per rung. If the BASE arm does not exhibit the defect
at a rung, no comparison at that rung is evidence, and the rung is reported VOID rather than passed.
On this host the reasoning toggle is known to sit at ~25% busy at r100K and to load the main thread
only at r500K, so the rungs are expected to disagree and the report says which one carried the
verdict. The job's gate is that AT LEAST ONE rung qualified; a run where none did tells us nothing
and is INCONCLUSIVE, which announce.py fails.

THE TWO ONE-LINE ISOLATIONS ARE THE MEASUREMENT. C vs B and Aoff vs A each differ by a single
literal, so their ratio is pagination and cannot be anything else. baseT vs C is carried separately
because it prices the REST of 9477, which must not be attributed to pagination. A vs B is reported
because six of the branch's own commits sit between them and one is itself a perf change.

NON-VACUITY. Beyond the VOID rule: the arms must be different bundles, every arm must really have
been WebKitGTK, every arm must have loaded the SAME corpus, and -- the one specific to this piece --
PAGINATION MUST ACTUALLY HAVE FIRED. If the paginated arm mounted as much reasoning as the
unpaginated one, the flag did not reach the DOM and a flat result would be an artefact of the arm
rather than a fact about the mechanism. That is read out of the action row's own census.
"""

from __future__ import annotations

TITLE = "PR 9477 piece 4 (reasoning pagination) on action:reasoning_toggle, real WebKitGTK/gfx1151"
MODE = "capability"

NEEDS = [
    "linux", "drm_render_node", "egl_hardware_gl", "webkitgtk", "headless_display_server",
    "gpu_browser_compositing", "studio_production_bundle", "studiobench_ladder",
    "discrete_gpu", "nvidia", "windows", "mlx",
]

ACTION = "reasoning_toggle"
EXPECTED_CORPUS = "23cd2464"

#: A rung counts as exhibiting the defect only if the BASE arm's reasoning toggle is this loaded.
#: Anchored on the action window's own measured numbers, not on scroll's: locally this window reads
#: 25% busy / 128 ms worst at r100K and 37% busy / 267 ms worst at r500K, while the jammed control
#: reads 87% busy / 392 ms. A threshold below ~50% would let an unloaded rung masquerade as a venue.
DEFECT_MIN_BUSY_PCT = 50.0
DEFECT_MIN_WORST_MS = 300.0

#: The jammed control must fall at least this far below the unjammed base, or the channel is blind.
CONTROL_MIN_DROP_PCT = 25.0

#: A ratio has to clear this AND the arms' own rep spread before it is called a move.
MIN_RATIO = 1.15

#: Pagination has fired only if the paginated arm mounts materially less than its own pair.
PAGINATION_MIN_DROP_PCT = 20.0

ARMS = ("baseT", "C", "B", "Aoff", "A")
ISOLATIONS = (("C", "B"), ("Aoff", "A"))


# ── accessors over payload["actions"], NOT payload["phases"] ─────────────────────────────────
def _runs(obs: dict) -> list[dict]:
    return [r for r in (obs.get("runs") or []) if (r.get("payload") or {}).get("ok")]


def _sel(obs: dict, arm: str, rung: str) -> list[dict]:
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


def _blocked(payload: dict):
    return ((_action(payload).get("busy")) or {}).get("blocked_ms")


def _worst(payload: dict):
    return (_action(payload).get("raf") or {}).get("max_ms")


def _sync(payload: dict):
    return _action(payload).get("app_sync_ms")


def _mounted_reasoning(payload: dict):
    """How much text was in the DOM WHILE THE PANE WAS OPEN.

    `census_open`, not `census_after`: the gesture opens the pane and closes it again, so both
    the before and after snapshots are taken with the reasoning content unmounted on every arm.
    Scored on those, a paginated arm and an unpaginated one are identical and the "did the
    treatment do anything" gate would pass vacuously on a build where the flag never reached the
    DOM at all.
    """
    c = _action(payload).get("census_open") or {}
    v = c.get("assistant_chars")
    return v if isinstance(v, (int, float)) else None


def _vals(obs: dict, arm: str, rung: str, fn):
    return [v for v in (fn(r["payload"]) for r in _sel(obs, arm, rung)) if v is not None]


def _mean(xs):
    return (sum(xs) / len(xs)) if xs else None


def _spread(xs):
    return (max(xs) - min(xs)) if len(xs) >= 2 else None


def _bundles(obs: dict, arm: str) -> set:
    return {(r["payload"].get("run_meta") or {}).get("bundle_hash")
            for r in _runs(obs) if r.get("arm") == arm}


def _rungs(obs: dict) -> list[str]:
    return [r for r in (obs.get("rungs") or []) if any(_sel(obs, "baseT", r))]


def _qualifying(obs: dict) -> list[str]:
    """Rungs where the BASE arm exhibits the defect. Everything else is VOID at that rung."""
    out = []
    for rung in _rungs(obs):
        bu = _mean(_vals(obs, "baseT", rung, _busy))
        wo = _mean(_vals(obs, "baseT", rung, _worst))
        if bu is not None and wo is not None and (bu >= DEFECT_MIN_BUSY_PCT
                                                  and wo >= DEFECT_MIN_WORST_MS):
            out.append(rung)
    return out


def _control(obs: dict, rung: str):
    jam = _mean(_vals(obs, "JAM", rung, _eff_fps))
    clean = _mean(_vals(obs, "baseT", rung, _eff_fps))
    return jam, clean


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out: list[tuple[str, bool, str]] = []
    st = obs.get("states") or {}
    runs, ok = obs.get("runs") or [], _runs(obs)

    out.append(("a display server was obtained", bool((obs.get("xserver") or {}).get("display")),
                obs.get("fatal") or str((obs.get("xserver") or {}).get("display"))))

    bad = [n for n in ARMS if not (st.get(n) or {}).get("patch_ok")]
    out.append(("every arm's patches applied cleanly and completely", not bad,
                "; ".join(f"{n}: {(st.get(n) or {}).get('patch_steps')}" for n in bad)[:400]
                or "all five arms patched"))

    # THE FLAG, READ OUT OF THE SOURCE. The whole run turns on this one literal.
    want = {"baseT": None, "C": "false", "B": "true", "Aoff": "false", "A": "true"}
    got = {n: (st.get(n) or {}).get("pagination_literal") for n in ARMS}
    flags_ok = all(got[n] == want[n] for n in ARMS if want[n] is not None)
    out.append(("each arm carries the pagination literal it is supposed to", flags_ok,
                ", ".join(f"{n}={got[n]}" for n in ARMS)))

    out.append(("every arm installed a PRODUCTION bundle",
                all((st.get(n) or {}).get("dist", {}).get("index_html") for n in ARMS),
                "; ".join(f"{n}: rc={(st.get(n) or {}).get('install', {}).get('rc')} "
                          f"assets={(st.get(n) or {}).get('dist', {}).get('asset_files')}"
                          for n in ARMS)))

    out.append(("every measurement completed", bool(runs) and len(ok) == len(runs),
                f"{len(ok)}/{len(runs)} ok" + ("" if len(ok) == len(runs) else "; " + "; ".join(
                    f"{r.get('arm')}/{r.get('rung')}#{r.get('rep')}: "
                    f"{str((r.get('payload') or {}).get('error') or r.get('rc'))[:110]}"
                    for r in runs if r not in ok)[:600])))

    # The one-line pairs MUST be different bundles, or the comparison could find nothing.
    pair_notes, pairs_ok = [], True
    for lo, hi in ISOLATIONS:
        a, b = _bundles(obs, lo), _bundles(obs, hi)
        differ = bool(a) and bool(b) and not (a & b)
        pairs_ok = pairs_ok and differ
        pair_notes.append(f"{lo}={sorted(x or '?' for x in a)} vs {hi}={sorted(x or '?' for x in b)}")
    out.append(("each one-line isolation pair built DIFFERENT bundles", pairs_ok,
                "; ".join(pair_notes)))

    engines = {(r["payload"].get("engine_probe") or {}).get("is_webkit_gtk_ua") for r in ok}
    out.append(("every measurement really was WebKitGTK", bool(ok) and engines == {True},
                f"is_webkit_gtk_ua={engines}"))

    # ONE CORPUS. Each arm loads studiobench from its OWN checkout, so a corpus drift between the
    # arms would silently change the thread being measured while every other gate still passed.
    corpora = {str((r["payload"].get("run_meta") or {}).get("corpus_hash"))[:8] for r in ok}
    out.append((f"every arm loaded the same corpus ({EXPECTED_CORPUS})",
                corpora == {EXPECTED_CORPUS}, f"corpus hashes seen: {sorted(corpora)}"))

    # THE JAMMED POSITIVE CONTROL. Without it a flat result is indistinguishable from a blind
    # channel, and this campaign has published that mistake before.
    notes, ctrl_ok = [], False
    for rung in _rungs(obs):
        jam, clean = _control(obs, rung)
        if jam is None or clean is None or not clean:
            notes.append(f"{rung}: no control reading")
            continue
        drop = (1.0 - jam / clean) * 100.0
        notes.append(f"{rung}: {clean:.1f} -> {jam:.1f} fps ({drop:.0f}% fall)")
        ctrl_ok = ctrl_ok or drop >= CONTROL_MIN_DROP_PCT
    out.append((f"the jammed control falls at least {CONTROL_MIN_DROP_PCT:.0f}% "
                f"(the channel can report a blocked main thread)", ctrl_ok, "; ".join(notes)))

    # PAGINATION ACTUALLY FIRED on at least one paginated arm.
    fired, fnotes = False, []
    for rung in _rungs(obs):
        for off, on in ISOLATIONS:
            a = _mean(_vals(obs, off, rung, _mounted_reasoning))
            b = _mean(_vals(obs, on, rung, _mounted_reasoning))
            if a and b:
                drop = (1.0 - b / a) * 100.0
                fnotes.append(f"{rung} {off}->{on}: {a:,.0f} -> {b:,.0f} ({drop:+.0f}%)")
                fired = fired or drop >= PAGINATION_MIN_DROP_PCT
    out.append((f"pagination actually reduced what is mounted by >= {PAGINATION_MIN_DROP_PCT:.0f}%",
                fired, "; ".join(fnotes) or "no census pair available"))

    qual = _qualifying(obs)
    out.append((f"at least one rung's BASE arm exhibits the defect "
                f"(>= {DEFECT_MIN_BUSY_PCT:.0f}% busy AND >= {DEFECT_MIN_WORST_MS:.0f} ms worst "
                f"frame on {ACTION})", bool(qual),
                "; ".join(f"{r}: {_mean(_vals(obs, 'baseT', r, _busy)) or float('nan'):.0f}% busy, "
                          f"{_mean(_vals(obs, 'baseT', r, _worst)) or float('nan'):.0f} ms worst"
                          for r in _rungs(obs)) or "no base readings"))
    return out


def table(obs: dict) -> str:
    qual = set(_qualifying(obs))
    rows = [f"Window `action:{ACTION}`. Frame rate is EFFECTIVE rate over wall time "
            f"(`1000*raf.n/elapsed_ms`); `raf.fps_p50` is deliberately not quoted anywhere, "
            f"because it read 62.5 fps both jammed and unjammed on this scene.", ""]

    for rung in _rungs(obs):
        jam, clean = _control(obs, rung)
        mark = "SCORED" if rung in qual else "VOID at this rung: the base arm does not exhibit the defect"
        rows += [f"### r{rung} -- {mark}", "",
                 "| arm | pagination | eff fps | busy | worst frame | blocked | click handler |",
                 "|---|---|---|---|---|---|---|"]
        for arm in ARMS:
            f = _mean(_vals(obs, arm, rung, _eff_fps))
            bu = _mean(_vals(obs, arm, rung, _busy))
            wo = _vals(obs, arm, rung, _worst)
            bl = _mean(_vals(obs, arm, rung, _blocked))
            sy = _mean(_vals(obs, arm, rung, _sync))
            lit = ((obs.get("states") or {}).get(arm) or {}).get("pagination_literal")
            rows.append("| " + " | ".join([
                arm, str(lit),
                "-" if f is None else f"**{f:.1f}**",
                "-" if bu is None else f"{bu:.0f}%",
                "-" if not wo else f"{max(wo):,.0f} ms",
                "-" if bl is None else f"{bl:,.0f} ms",
                "-" if sy is None else f"{sy:.1f} ms",
            ]) + " |")
        if jam and clean:
            rows.append(f"| JAM (control) | n/a | **{jam:.1f}** | - | - | - | - |")
        rows += ["", "The one-line isolations at this rung, which are the measurement:", "",
                 "| pair | what differs | eff fps | worst frame | blocked | rep spread (fps) |",
                 "|---|---|---|---|---|---|"]
        for off, on in ISOLATIONS:
            a, b = _mean(_vals(obs, off, rung, _eff_fps)), _mean(_vals(obs, on, rung, _eff_fps))
            wa, wb = _mean(_vals(obs, off, rung, _worst)), _mean(_vals(obs, on, rung, _worst))
            ba, bb = _mean(_vals(obs, off, rung, _blocked)), _mean(_vals(obs, on, rung, _blocked))
            floor = max(_spread(_vals(obs, off, rung, _eff_fps)) or 0.0,
                        _spread(_vals(obs, on, rung, _eff_fps)) or 0.0)
            rows.append("| " + " | ".join([
                f"{off} -> {on}", "one literal",
                "-" if not (a and b) else f"**{b / a:.3f}x** ({a:.1f} -> {b:.1f})",
                "-" if not (wa and wb) else f"{wb / wa:.3f}x ({wa:,.0f} -> {wb:,.0f} ms)",
                "-" if not (ba and bb) else f"{bb / ba:.3f}x ({ba:,.0f} -> {bb:,.0f} ms)",
                f"{floor:.1f}",
            ]) + " |")
        base = _mean(_vals(obs, "baseT", rung, _eff_fps))
        c = _mean(_vals(obs, "C", rung, _eff_fps))
        if base and c:
            rows.append(f"| baseT -> C | ALL of 9477, flag off | **{c / base:.3f}x** "
                        f"({base:.1f} -> {c:.1f}) | - | - | - |")
        a_, b_ = _mean(_vals(obs, "A", rung, _eff_fps)), _mean(_vals(obs, "B", rung, _eff_fps))
        if a_ and b_:
            rows.append(f"| A -> B | six branch commits | **{b_ / a_:.3f}x** "
                        f"({a_:.1f} -> {b_:.1f}) | - | - | - |")
        rows += ["", "Per repetition, so a mean cannot hide a disagreement:", "",
                 "| arm | eff fps per rep | busy per rep |", "|---|---|---|"]
        for arm in ARMS:
            fs = _vals(obs, arm, rung, _eff_fps)
            bs = _vals(obs, arm, rung, _busy)
            rows.append(f"| {arm} | {', '.join(f'{x:.1f}' for x in fs) or '-'} | "
                        f"{', '.join(f'{x:.0f}%' for x in bs) or '-'} |")
        rows.append("")

    st = obs.get("states") or {}
    rows += ["Arms, all built as patches on upstream commits so nothing depends on a branch that "
             "can drift:", ""]
    for arm in ARMS:
        s = st.get(arm) or {}
        rows.append(f"- `{arm}` = `{str(s.get('ref'))[:9]}` + {s.get('patches') or 'nothing'}; "
                    f"pagination literal `{s.get('pagination_literal')}`; "
                    f"bundle {sorted(x or '?' for x in _bundles(obs, arm))}")
    rows += ["", "Arms were interleaved within each repetition, so drift over the job cannot land "
             "entirely on whichever arm ran last."]
    return "\n".join(rows)


def verdict(obs: dict) -> tuple[str, str]:
    qual = _qualifying(obs)
    if not qual:
        return "INCONCLUSIVE", ("no rung's base arm exhibited the defect on this host, so nothing "
                                "measured here speaks to whether pagination fixes it")
    best = None
    for rung in qual:
        for off, on in ISOLATIONS:
            a, b = _mean(_vals(obs, off, rung, _eff_fps)), _mean(_vals(obs, on, rung, _eff_fps))
            if not (a and b):
                continue
            floor = max(_spread(_vals(obs, off, rung, _eff_fps)) or 0.0,
                        _spread(_vals(obs, on, rung, _eff_fps)) or 0.0)
            ratio = b / a
            if best is None or ratio > best[0]:
                best = (ratio, rung, off, on, a, b, floor)
    if best is None:
        return "INCONCLUSIVE", "no isolation pair produced both arms at a qualifying rung"

    ratio, rung, off, on, a, b, floor = best
    moved = ratio >= MIN_RATIO and (b - a) > floor
    wa = _mean(_vals(obs, off, rung, _worst))
    wb = _mean(_vals(obs, on, rung, _worst))
    detail = (f"at r{rung}, the rung where the base arm exhibits the defect, flipping pagination "
              f"on ({off} -> {on}, one literal) moves the effective rate {a:.1f} -> {b:.1f} fps "
              f"({ratio:.3f}x) against an arm rep spread of {floor:.1f} fps"
              + (f", and the worst frame {wa:,.0f} -> {wb:,.0f} ms" if wa and wb else ""))
    if moved:
        return "HELPS", detail + (". Note this is the PERFORMANCE question only; whether it can "
                                  "ship is decided by the reachability of the unmounted pages")
    return "NO_BENEFIT", detail + (". That is a measured absence of benefit at the rung where the "
                                   "defect lives, not an absence of measurement")


def observed_capabilities(obs: dict) -> dict[str, bool]:
    st = obs.get("states") or {}
    return {
        "webkitgtk": bool(_runs(obs)),
        "headless_display_server": bool((obs.get("xserver") or {}).get("display")),
        "studio_production_bundle": all(
            (st.get(n) or {}).get("dist", {}).get("index_html") for n in ARMS),
        "studiobench_ladder": bool(_runs(obs)),
    }
