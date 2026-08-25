#!/usr/bin/env python3
"""Criteria: WHICH PIECE of PR 9477 owns the 4.981x on `action:reasoning_toggle_all`?

Judges only. Observations come from probes/studio_p123_probe.py.

WHAT IS BEING SPLIT. On the AMD venue, at r100K, on the gesture that opens EVERY reasoning pane,
upstream main (`90f85fdbf`) reads 5.1 effective fps at 93% busy, and arm `N111` -- all of 9477 with
`REASONING_PAGINATION_ENABLED = false` -- reads 25.3 fps at 60% busy. That 4.981x is one lump.
Piece 4 is excluded by construction because the flag is off in every arm here, so the lump belongs
to piece 1 (streamed content fidelity), piece 2 (the render schedule and text presentation) and/or
piece 3 (the streaming code policy). A span count pointed at piece 3, but a span count is an
inference and this file scores an ablation.

THIS IS RECONCILIATION, NOT DISCOVERY. The branch already carries an ablation, in `6adc583a2`: at
250K STREAMED reasoning characters, worst main-thread freeze went base 6,821 ms, null 6,878, the
streaming-render rewrite alone 869, plus the plain-code policy 211, plus pagination 79. On THAT
window piece 2 dominates. But that is a streaming window and this is a settled one, and this
campaign has already been caught by exactly that distinction once. So both orderings are reported
side by side and neither is assumed to carry the other.

PIECE 3 IS TWO MECHANISMS. Reading the source rather than the constant shows that `3a` is
`reasoning.tsx` passing `codeHighlighting="plain"`, which makes every fence in a reasoning pane
permanently plain AT ANY SIZE and is not a cap at all, while `3b` is a pair of size thresholds:
`isOversizedStreamingCode` at 4,096 characters and `shouldAutoHighlightStreamingCode` at 16,384.
On this gesture they act on different fences, so folding them into one arm would price whichever
dominates and then attribute it to both.

THE DESIGN IS A 2^3 FACTORIAL OVER (p2, p3a, p3b), and that is not extravagance. A purely
SUBTRACTIVE design cannot see redundancy: if two mechanisms are each independently sufficient,
removing either alone changes nothing and the honest reading of that design is "nothing matters",
which would be false. A purely ADDITIVE design cannot see a mechanism that only pays off in
company. All eight corners give each factor FOUR one-piece-at-a-time isolations, one per context
of the other two.

    arm     p1   p2   p3a  p3b   built as
    main    -    -    -    -     90f85fdbf: the reference the 4.981x was measured against
    Z       off  off  off  off   C_tip + all four off-patches       <- CLOSURE CONTROL
    A000    on   off  off  off   C_tip + p2_off + p3a_off + p3b_off
    A100    on   ON   off  off   C_tip + p3a_off + p3b_off
    A010    on   off  ON   off   C_tip + p2_off + p3b_off
    A001    on   off  off  ON    C_tip + p2_off + p3a_off
    A110    on   ON   ON   off   C_tip + p3b_off
    A101    on   ON   off  ON    C_tip + p3a_off
    A011    on   off  ON   ON    C_tip + p2_off
    A111    on   ON   ON   ON    C_tip alone: the 4.981x endpoint

Piece 1 sits outside the factorial and gets ONE isolation, `Z -> A000`. That is a budget decision
and it is stated rather than hidden: piece 1 is a stream-time mechanism and this window is a thread
that finished streaming before the measurement began, so it is the least likely of the four to
carry the lump and it is asked an existence question rather than a context sweep.

THE CLOSURE GATE IS THE ONE THAT CAN FALSIFY THE WHOLE SPLIT. `Z` has all four mechanisms
neutralised, so it should behave like `main`. If it does not, the neutralisations do not span the
change: something in 9477 that none of the four patches turns off is carrying part of the lump,
and every per-mechanism number below is a share of less than the whole. That is reported as a
number, not asserted away, and the verdict says so.

THE JAM CONTROL IS PER RUNG, AND THIS IS A FIX, NOT AN INHERITANCE. The p4 criteria computed
`ctrl_ok = ctrl_ok or drop >= 25` across all rungs, so a control that resolved at r100K licensed
r500K as well. At r500K the base arm sits at the instrument floor: 0.08 effective fps, and the
JAMMED arm read 0.14 -- the deliberately blocked page was FASTER than the clean one, because both
are past the point where the channel can resolve anything. Under the inherited gate r500K scored,
and it produced the largest ratio in the run and therefore the verdict. Here every rung is gated by
its OWN control, and a rung whose control does not fall is VOID however dramatic its ratios are.

WORST FRAME IS CARRIED SEPARATELY, and gated separately. At a rung where the rate channel is at
the floor, the worst frame can still be informative -- but only if the control shows THAT channel
resolving too. So there are two controls, one per channel, and a rung can be VOID for rate and
usable for worst frame. A rung that fails both is a rung this host cannot answer at, which is a
finding about the venue and is reported as one.

NEVER `raf.fps_p50`. On this exact scene `1000/p50` read 62.5 fps jammed and unjammed alike. Every
rate here is `1000 * raf.n / elapsed_ms`, effective frames over wall time.
"""

from __future__ import annotations

import math

TITLE = ("PR 9477 pieces 1/2/3 attribution on action:reasoning_toggle_all, "
         "real WebKitGTK/gfx1151")
MODE = "capability"

NEEDS = [
    "linux", "drm_render_node", "egl_hardware_gl", "webkitgtk", "headless_display_server",
    "gpu_browser_compositing", "studio_production_bundle", "studiobench_ladder",
    "discrete_gpu", "nvidia", "windows", "mlx",
]

#: THE GESTURE, quoted rather than named. `reasoning_toggle` names two different gestures in this
#: campaign: this scene's opens the FIRST pane, studiobench's opens EVERY pane. The one-pane
#: gesture caps out at a single 12,338-character trace and reads flat for every mechanism here, so
#: it is measured and printed but never scored.
ACTION = "reasoning_toggle_all"
SECONDARY_ACTION = "reasoning_toggle"
#: not a performance window at all: it exists to let the DOM settle so fidelity can be priced
FIDELITY_ACTION = "reasoning_fidelity_settled"

EXPECTED_CORPUS = "23cd2464"

#: The mechanisms, and what each is. Piece 3 is carried as TWO because the source shows it is two
#: and their user-visible costs are not comparable. `defer` is not one of 9477's mechanisms at all:
#: it is MAIN's fence-deferral machinery, which `code-fence-defer.tsx` shows is byte-identical
#: between the two trees, and which 9477 turns off only as a side effect of 3a.
FACTORS = ("p2", "p3a")
PIECES = ("p1", "p2", "p3a", "p3b")
PIECE_LABEL = {
    "p1": "streamed content fidelity (chat-adapter, parse-assistant-content)",
    "p2": "the streaming render rewrite (streaming-render-schedule, streaming-text-presentation)",
    "p3a": "reasoning panes render code plain at any size (reasoning.tsx codeHighlighting)",
    "p3b": "the code size thresholds (4,096 plain-first, 16,384 no-upgrade)",
    "defer": "MAIN's per-fence deferral machinery (observers, layout reads, flushSync jumps)",
}

REFERENCE = "main"
CLOSURE = "Z"
DECOUPLE = "M"
NULL_P1 = "N1"
NULL_P3B = "N3b"
#: mask -> arm name. bit i is FACTORS[i], 1 = ON.
MASKS = ((0, 0), (1, 0), (0, 1), (1, 1))


def arm_of(mask) -> str:
    return "A" + "".join(str(b) for b in mask)


FACTORIAL_ARMS = tuple(arm_of(m) for m in MASKS)
ARMS = (REFERENCE, CLOSURE, DECOUPLE) + FACTORIAL_ARMS + (NULL_P1, NULL_P3B)
WHOLE = arm_of((1, 1))        # = A11 = C_tip alone, the arm the 4.981x was measured on
NEUTRAL = CLOSURE
BASELINE = arm_of((0, 0))     # = A00, everything live turned off but piece 1 and 3b left on

#: THE DECOMPOSITION `M` BUYS. `A00 -> A01` is all that `codeHighlighting="plain"` has ever been
#: measured as; these two split it, and the split is the whole recommendation.
DECOUPLE_EDGES = (
    (BASELINE, DECOUPLE, "MAIN's fence-deferral machinery, colours held ON"),
    (DECOUPLE, arm_of((0, 1)), "the colours themselves, machinery held OFF"),
)

#: single-mechanism isolations that are not part of the 2^2 sweep. Each is one flip from `A11`.
EXTRA_EDGES = {
    "p1": (NULL_P1, WHOLE, "one flip from the whole"),
    "p3b": (NULL_P3B, WHOLE, "one flip from the whole"),
}

#: PREDICTED NULLS, asserted BEFORE the run by the probe and re-stated here so the report says
#: "confirmed inert" rather than "flat, cause unknown". A null that was predicted is a result; a
#: null that was not is an open question about whether the flip took at all.
PREDICTED_NULL_ARMS = (NULL_P1, NULL_P3B)
#: how close to `A11` a predicted-null arm has to land for the prediction to be called confirmed
NULL_TOLERANCE = 1.15

#: A rung counts as a venue only if the REFERENCE arm is this loaded on this gesture. Anchored on
#: the measured r100K reading of the reference arm: 93% busy, 1,779 ms worst frame.
DEFECT_MIN_BUSY_PCT = 50.0
DEFECT_MIN_WORST_MS = 300.0

#: The jammed control must move each channel by at least this much, AT THE RUNG BEING SCORED.
CONTROL_MIN_RATE_DROP_PCT = 25.0
CONTROL_MIN_WORST_RISE_PCT = 25.0

#: Below this the rate channel has no dynamic range left: 0.3 fps is one frame per 3.3 seconds,
#: and at r500K the reference arm presented TWO frames in 25 seconds. Ratios of numbers that
#: small are ratios of small integer frame counts, so they are quoted as direction and never as
#: a factor.
FLOOR_FPS = 0.3

#: A ratio has to clear this AND the arms' own rep spread before it is called a move.
MIN_RATIO = 1.15

#: `N000` should land within this of `main`, in either direction, or the three neutralisations do
#: not span 9477 and the split is of less than the whole.
CLOSURE_TOLERANCE = 1.25

#: piece -> the census quantity that piece ACTS ON, read from the SETTLED fidelity census.
#: A gate that keys on something the mechanism does not touch reports a real change as nothing:
#: this campaign has already published a gate on thread-wide `assistant_chars` that rendered a
#: 78% cut in REASONING characters as "+2%, did not fire".
ENGAGEMENT_MIN_PCT = 10.0

#: every mechanism that has at least one isolation edge, in the order the report walks them.
#: `defer` is last because it is not one of 9477's mechanisms and reads as a separate question.
SWEPT = FACTORS + ("p1", "p3b", "defer")


# ── accessors. Everything reads payload["actions"], never payload["phases"] ───────────────────
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


def _eff_fps(payload: dict, name: str | None = None):
    a = _action(payload, name)
    n = (a.get("raf") or {}).get("n")
    el = a.get("elapsed_ms")
    return (1000.0 * n / el) if (n and el) else None


def _busy(payload: dict, name: str | None = None):
    return ((_action(payload, name).get("busy")) or {}).get("busy_pct")


def _blocked(payload: dict, name: str | None = None):
    return ((_action(payload, name).get("busy")) or {}).get("blocked_ms")


def _worst(payload: dict, name: str | None = None):
    return (_action(payload, name).get("raf") or {}).get("max_ms")


def _reasoning_chars(payload: dict, name: str | None = None):
    c = _action(payload, name).get("census_open") or {}
    v = c.get("reasoning_chars")
    return v if isinstance(v, (int, float)) else None


def _fences(payload: dict, where: str = "reasoning") -> dict:
    """The SETTLED per-fence census, which is the only fidelity number worth quoting.

    Every other census in the payload is a snapshot taken 2,500 ms after a click, and highlighting
    on this app is asynchronous, so those counts move with how busy the page happened to be. The
    reference arm read 11,530 spans on one repetition and 11,094 on the next; the JAMMED arm read
    7,259, not because a jam changes what the app renders but because a blocked main thread had
    not finished rendering it. `reasoning_fidelity_settled` waits for the count to stop changing.
    """
    return ((_action(payload, FIDELITY_ACTION).get("fence_census") or {}).get(where)) or {}


def _fence_field(field: str, where: str = "reasoning"):
    def fn(payload: dict, name: str | None = None):
        v = _fences(payload, where).get(field)
        return v if isinstance(v, (int, float)) else None
    return fn


def _settled_ok(payload: dict) -> bool:
    return bool(_action(payload, FIDELITY_ACTION).get("settled"))


def _vals(obs: dict, arm: str, rung: str, fn, name: str | None = None):
    return [v for v in (fn(r["payload"], name) for r in _sel(obs, arm, rung)) if v is not None]


def _mean(xs):
    return (sum(xs) / len(xs)) if xs else None


def _spread(xs):
    return (max(xs) - min(xs)) if len(xs) >= 2 else None


def _bundles(obs: dict, arm: str) -> set:
    return {(r["payload"].get("run_meta") or {}).get("bundle_hash")
            for r in _runs(obs) if r.get("arm") == arm}


def _rungs(obs: dict) -> list[str]:
    return [r for r in (obs.get("rungs") or []) if any(_sel(obs, REFERENCE, r))]


# ── the isolation edges. This is the measurement. ────────────────────────────────────────────
def edges(piece: str):
    """Every pair of arms that differ ONLY in `piece`, as (off_arm, on_arm, context).

    Two per swept factor, one for each setting of the other, reported individually rather than
    averaged so a mechanism whose effect depends on its company says so. `p1` and `p3b` get one
    each, and both are PREDICTED NULLS, so what matters for them is not a ratio but whether the
    prediction held.
    """
    if piece in EXTRA_EDGES:
        return [EXTRA_EDGES[piece]]
    if piece == "defer":
        return [DECOUPLE_EDGES[0]]
    i = FACTORS.index(piece)
    out = []
    for mask in MASKS:
        if mask[i] != 0:
            continue
        on = tuple(1 if k == i else mask[k] for k in range(len(FACTORS)))
        others = [FACTORS[k] for k in range(len(FACTORS)) if k != i and mask[k] == 1]
        ctx = ("with " + " and ".join(others)) if others else "alone"
        out.append((arm_of(mask), arm_of(on), ctx))
    return out


# ── rung admissibility. THREE outcomes, not two. ─────────────────────────────────────────────
def _control(obs: dict, rung: str, fn, name: str | None = None):
    return _mean(_vals(obs, "JAM", rung, fn, name)), _mean(_vals(obs, REFERENCE, rung, fn, name))


def rung_state(obs: dict, rung: str) -> dict:
    """SCORED / WORST_FRAME_ONLY / VOID, each with the reason it got there.

    The p4 criteria took the control gate globally (`ctrl_ok or drop >= 25`), so r100K's working
    control licensed r500K, where the jammed arm was FASTER than the clean one. Here the control
    is evaluated at the rung it is protecting, per channel.
    """
    st = {"rung": rung, "notes": []}
    base_busy = _mean(_vals(obs, REFERENCE, rung, _busy))
    base_worst = _mean(_vals(obs, REFERENCE, rung, _worst))
    base_fps = _mean(_vals(obs, REFERENCE, rung, _eff_fps))
    jam_fps, _ = _control(obs, rung, _eff_fps)
    jam_worst, _ = _control(obs, rung, _worst)
    st.update(base_busy=base_busy, base_worst=base_worst, base_fps=base_fps,
              jam_fps=jam_fps, jam_worst=jam_worst)

    st["defect"] = bool(base_busy is not None and base_worst is not None
                        and base_busy >= DEFECT_MIN_BUSY_PCT
                        and base_worst >= DEFECT_MIN_WORST_MS)
    if not st["defect"]:
        st["notes"].append("the reference arm does not exhibit the defect at this rung")

    # channel 1: effective rate
    if jam_fps is None or not base_fps:
        st["rate_drop_pct"] = None
        st["rate_control_ok"] = False
        st["notes"].append("no rate control reading")
    else:
        d = (1.0 - jam_fps / base_fps) * 100.0
        st["rate_drop_pct"] = d
        st["rate_control_ok"] = d >= CONTROL_MIN_RATE_DROP_PCT
        if not st["rate_control_ok"]:
            st["notes"].append(
                f"jamming the main thread moved the rate by {d:+.0f}% "
                f"({base_fps:.2f} -> {jam_fps:.2f} fps), so the rate channel cannot report a "
                f"blocked main thread here")

    # channel 2: worst frame
    if jam_worst is None or not base_worst:
        st["worst_rise_pct"] = None
        st["worst_control_ok"] = False
        st["notes"].append("no worst-frame control reading")
    else:
        d = (jam_worst / base_worst - 1.0) * 100.0
        st["worst_rise_pct"] = d
        st["worst_control_ok"] = d >= CONTROL_MIN_WORST_RISE_PCT
        if not st["worst_control_ok"]:
            st["notes"].append(
                f"jamming the main thread moved the worst frame by {d:+.0f}% "
                f"({base_worst:,.0f} -> {jam_worst:,.0f} ms), so the worst-frame channel cannot "
                f"report a blocked main thread here either")

    st["at_floor"] = bool(base_fps is not None and base_fps < FLOOR_FPS)
    if st["at_floor"]:
        st["notes"].append(
            f"the reference arm sits at {base_fps:.2f} fps, below the {FLOOR_FPS} fps instrument "
            f"floor, so a rate RATIO here is a ratio of single-digit frame counts")

    if not st["defect"]:
        st["state"] = "VOID"
    elif st["rate_control_ok"] and not st["at_floor"]:
        st["state"] = "SCORED"
    elif st["worst_control_ok"]:
        st["state"] = "WORST_FRAME_ONLY"
    else:
        st["state"] = "VOID"
    return st


def _scored(obs: dict) -> list[str]:
    return [r for r in _rungs(obs) if rung_state(obs, r)["state"] == "SCORED"]


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out: list[tuple[str, bool, str]] = []
    st = obs.get("states") or {}
    runs, ok = obs.get("runs") or [], _runs(obs)

    out.append(("a display server was obtained", bool((obs.get("xserver") or {}).get("display")),
                obs.get("fatal") or str((obs.get("xserver") or {}).get("display"))))

    landed = "; ".join(
        f"{n}: asked {str((st.get(n) or {}).get('ref'))[:9]} "
        f"landed {str((st.get(n) or {}).get('ref_landed') or (st.get(n) or {}).get('commit'))[:9]}"
        + ("" if (st.get(n) or {}).get("checkout_ok", True) else " MISMATCH")
        for n in ARMS)
    out.append(("every arm checked out the commit it was asked for",
                all((st.get(n) or {}).get("checkout_ok") for n in ARMS), landed))

    bad = [n for n in ARMS if not (st.get(n) or {}).get("patch_ok")]
    out.append(("every arm's patches applied cleanly and completely", not bad,
                "; ".join(f"{n}: {(st.get(n) or {}).get('why') or (st.get(n) or {}).get('patch_steps')}"
                          for n in bad)[:600] or f"all {len(ARMS)} arms patched"))

    # THE PIECE MASK, READ BACK OUT OF EACH ARM'S SOURCE. The run turns on this: an arm whose
    # neutralisation patch applied to the wrong hunk still reports rc=0. Each `*_off.patch` plants
    # a sentinel the probe greps for, so "this arm has piece 2 off" is a fact about the tree that
    # was built and not a restatement of the patch list.
    # order is PIECES = (p1, p2, p3a, p3b). Piece 1 is ON at every factorial corner and off only
    # at the closure arm, which is the whole reason `Z -> A000` is piece 1's isolation.
    # order is PIECES = (p1, p2, p3a, p3b). The factorial corners leave p1 and p3b ON; only the
    # closure arm and the two null arms turn them off.
    want = {REFERENCE: None, CLOSURE: (False, False, False, False),
            DECOUPLE: (True, False, False, True),
            NULL_P1: (False, True, True, True),
            NULL_P3B: (True, True, True, False)}
    for m in MASKS:
        want[arm_of(m)] = (True, bool(m[0]), bool(m[1]), True)
    got = {n: (st.get(n) or {}).get("piece_mask") for n in ARMS}
    mask_ok = all(tuple(got[n] or ()) == want[n] for n in ARMS if want[n] is not None)
    out.append(("each arm's source carries the piece mask it is supposed to", mask_ok,
                "; ".join(f"{n}={''.join('1' if b else '0' for b in (got[n] or ())) or 'none'}"
                          for n in ARMS)))

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

    # EVERY ISOLATION EDGE MUST BE TWO DIFFERENT BUNDLES. Twelve of them. If a neutralisation
    # patch is a no-op at the bundle level, the pair it defines can only ever find nothing, and
    # that nothing would read as "this piece does not matter".
    bad_edges, seen = [], 0
    for piece in SWEPT:
        for lo, hi, _ in edges(piece):
            a, b = _bundles(obs, lo), _bundles(obs, hi)
            seen += 1
            if not (a and b and not (a & b)):
                bad_edges.append(f"{lo}/{hi}")
    out.append((f"all {seen} one-piece isolation edges are pairs of DIFFERENT bundles",
                not bad_edges, "identical or missing bundles on: " + ", ".join(bad_edges)
                if bad_edges else f"{seen}/{seen} edges differ"))

    # AND EVERY ARM MUST BE ITS OWN BUNDLE. Eight corners that collapse to three distinct bundles
    # would mean the patches overlap, and the factorial would be a fiction.
    per_arm = {n: sorted(x or "?" for x in _bundles(obs, n)) for n in ARMS}
    distinct = {tuple(v) for v in per_arm.values() if v}
    out.append(("the arms really are distinct trees (one bundle each, all different)",
                len(distinct) == len([v for v in per_arm.values() if v]),
                "; ".join(f"{n}={v}" for n, v in per_arm.items())[:700]))

    # THE SERVED BUNDLE, READ OUT OF THE BROWSER, NOT OFF THE DISK.
    #
    # This gate exists because of a silent-wrong-result mode the two-job design introduces.
    # `studio/backend/run.py::_resolve_frontend_path` tries `--frontend` first and, if that path
    # has no `index.html`, FALLS BACK to the dist inside the installed package. With one shared
    # backend install, that fallback exists and is the BASE bundle. So an arm whose dist failed to
    # arrive would not error: it would quietly serve main's frontend and be measured and reported
    # as that arm. A disk-side hash cannot see this, because the disk-side hash is of the
    # directory that was never served.
    #
    # `build.scripts` is the `<script src>` list the page actually loaded, and Vite emits
    # content-hashed filenames, so two arms that served different bundles cannot share it.
    served = {}
    for r in ok:
        served.setdefault(r.get("arm"), set()).add(
            tuple(sorted((r["payload"].get("build") or {}).get("scripts") or [])))
    collisions = []
    for a in ARMS:
        for b in ARMS:
            if a >= b or a not in served or b not in served:
                continue
            ha = (st.get(a) or {}).get("exported_bundle_hash")
            hb = (st.get(b) or {}).get("exported_bundle_hash")
            if ha and hb and ha != hb and served[a] & served[b]:
                collisions.append(f"{a}/{b}")
    every_arm_one = all(len(v) == 1 for v in served.values()) and bool(served)
    out.append(("every arm's BROWSER loaded that arm's own bundle, not the shared backend's "
                "fallback", not collisions and every_arm_one,
                (f"arms serving an identical script set despite different builds: "
                 f"{', '.join(collisions)}" if collisions else
                 f"{len({next(iter(v)) for v in served.values()})} distinct served script sets "
                 f"across {len(served)} arms")))

    # and the dist that was measured is byte-for-byte the dist that was built
    moved = [n for n in ARMS
             if (st.get(n) or {}).get("exported_bundle_hash")
             and (st.get(n) or {}).get("bundle_hash_matches_build") is False]
    out.append(("every arm's dist survived the trip from the build job unchanged", not moved,
                "; ".join(f"{n}: built {(st.get(n) or {}).get('exported_bundle_hash')} "
                          f"measured {(st.get(n) or {}).get('bundle_hash_at_measure')}"
                          for n in moved) or "every bundle hash matched the build job's"))

    engines = {(r["payload"].get("engine_probe") or {}).get("is_webkit_gtk_ua") for r in ok}
    out.append(("every measurement really was WebKitGTK", bool(ok) and engines == {True},
                f"is_webkit_gtk_ua={engines}"))

    corpora = {str((r["payload"].get("run_meta") or {}).get("corpus_hash"))[:8] for r in ok}
    out.append((f"every arm loaded the same corpus ({EXPECTED_CORPUS})",
                corpora == {EXPECTED_CORPUS}, f"corpus hashes seen: {sorted(corpora)}"))

    # ONE INSTRUMENT, BY RESOLVED PATH AND BY CONTENT HASH. The path alone says the import
    # resolved outside the arm's tree; the hash says the tree behind that path did not change
    # between the first arm and the ninth.
    pacers = {str((r["payload"].get("run_meta") or {}).get("instrument_pacer_file")) for r in ok}
    roots = {str((r["payload"].get("run_meta") or {}).get("instrument_sb_root")) for r in ok}
    hashes = {str((r["payload"].get("run_meta") or {}).get("instrument_hash")) for r in ok}
    one = (len(pacers) == 1 and len(roots) == 1 and "None" not in pacers
           and len(hashes) == 1 and "None" not in hashes)
    inside = all("/instrument/" in x for x in pacers) if one else False
    out.append(("every arm was driven by ONE instrument, outside every arm's own tree, "
                "identical by content hash", one and inside,
                f"pacer resolved to {sorted(pacers)}; sb-root {sorted(roots)}; "
                f"instrument hash {sorted(hashes)}"))

    # THE JAMMED CONTROL, PER RUNG AND PER CHANNEL. Not a global OR: that is precisely the bug
    # that let r500K score in the p4 run, where the jammed arm was faster than the clean one.
    notes, any_rung = [], False
    for rung in _rungs(obs):
        s = rung_state(obs, rung)
        notes.append(
            f"r{rung}: rate {s['base_fps'] or float('nan'):.2f} -> "
            f"{(s['jam_fps'] if s['jam_fps'] is not None else float('nan')):.2f} fps "
            f"({'ok' if s['rate_control_ok'] else 'BLIND'}), worst "
            f"{s['base_worst'] or float('nan'):,.0f} -> "
            f"{(s['jam_worst'] if s['jam_worst'] is not None else float('nan')):,.0f} ms "
            f"({'ok' if s['worst_control_ok'] else 'BLIND'}) -> {s['state']}")
        any_rung = any_rung or s["state"] != "VOID"
    out.append(("at least one rung has a jam control that resolves, AT THAT RUNG, on the channel "
                "it is used to score", any_rung, "; ".join(notes) or "no rungs measured"))

    # CLOSURE. The gate that can falsify the whole split.
    cnotes, closed = [], True
    for rung in _rungs(obs):
        a = _mean(_vals(obs, REFERENCE, rung, _eff_fps))
        b = _mean(_vals(obs, NEUTRAL, rung, _eff_fps))
        if not (a and b):
            continue
        ratio = b / a
        cnotes.append(f"r{rung}: {REFERENCE} {a:.1f} fps vs {NEUTRAL} {b:.1f} fps ({ratio:.2f}x)")
        if rung_state(obs, rung)["state"] == "SCORED":
            closed = closed and (1.0 / CLOSURE_TOLERANCE) <= ratio <= CLOSURE_TOLERANCE
    out.append((f"CLOSURE: with all four of 9477's mechanisms neutralised, {NEUTRAL} behaves like {REFERENCE} "
                f"(within {CLOSURE_TOLERANCE:.2f}x), so the four patches span the change",
                closed, "; ".join(cnotes) or "no closure reading"))

    # THE SETTLED CENSUS REALLY SETTLED, or the fidelity numbers are snapshots of a busy page.
    unsettled = sorted({f"{r['arm']}/{r['rung']}" for r in ok if not _settled_ok(r["payload"])})
    out.append(("the fidelity census reached quiescence on every arm", not unsettled,
                "did not settle within the bound: " + ", ".join(unsettled)
                if unsettled else "every arm's span count stopped changing before the census"))

    # ENGAGEMENT, PER PIECE, ON THE QUANTITY THAT PIECE ACTS ON. A piece that changed nothing
    # measurable cannot be credited or blamed, and a flat ratio for it means nothing.
    for piece, field, where, label in (
            ("p3a", "spans", "reasoning", "highlight spans inside the reasoning panes"),
            ("p2", "elements", None, "DOM elements while the panes are open")):
        fired, fnotes = False, []
        for rung in _rungs(obs):
            for lo, hi, ctx in edges(piece):
                if where:
                    fn = _fence_field(field, where)
                    a, b = _mean(_vals(obs, lo, rung, fn)), _mean(_vals(obs, hi, rung, fn))
                elif field == "elements":
                    def fn(p, n=None):
                        c = _action(p, FIDELITY_ACTION).get("census_open") or {}
                        return c.get("elements")
                    a, b = _mean(_vals(obs, lo, rung, fn)), _mean(_vals(obs, hi, rung, fn))
                else:
                    a, b = (_mean(_vals(obs, lo, rung, _reasoning_chars)),
                            _mean(_vals(obs, hi, rung, _reasoning_chars)))
                # `is None`, NOT truthiness. A quantity that fell to ZERO is the strongest
                # possible engagement -- p3a is expected to take the reasoning panes' highlight
                # spans to exactly 0 -- and `if not (a and b)` silently discards precisely that
                # case, reporting the mechanism as inert at the moment it worked completely.
                if a is None or b is None or not a:
                    continue
                ch = (b / a - 1.0) * 100.0
                if abs(ch) >= ENGAGEMENT_MIN_PCT:
                    fired = True
                    fnotes.append(f"r{rung} {lo}->{hi}: {a:,.0f} -> {b:,.0f} ({ch:+.0f}%)")
        out.append((f"{piece} engaged: turning it on changes {label} by >= "
                    f"{ENGAGEMENT_MIN_PCT:.0f}% on at least one edge", fired,
                    "; ".join(fnotes[:4]) or "no edge moved this quantity: this piece is either "
                    "inert on this gesture or its neutralisation did not take"))

    qual = _scored(obs)
    out.append((f"at least one rung is SCORED (reference arm loaded, control resolves, rate above "
                f"the {FLOOR_FPS} fps floor)", bool(qual),
                "; ".join(f"r{r}: {rung_state(obs, r)['state']}" for r in _rungs(obs))
                or "no reference readings"))
    return out


# ── attribution ──────────────────────────────────────────────────────────────────────────────
def _edge_ratios(obs: dict, piece: str, rung: str):
    out = []
    for lo, hi, ctx in edges(piece):
        a, b = _mean(_vals(obs, lo, rung, _eff_fps)), _mean(_vals(obs, hi, rung, _eff_fps))
        floor = max(_spread(_vals(obs, lo, rung, _eff_fps)) or 0.0,
                    _spread(_vals(obs, hi, rung, _eff_fps)) or 0.0)
        wa, wb = _mean(_vals(obs, lo, rung, _worst)), _mean(_vals(obs, hi, rung, _worst))
        out.append({"off": lo, "on": hi, "ctx": ctx, "a": a, "b": b,
                    "ratio": (b / a) if (a and b) else None, "spread": floor,
                    "wa": wa, "wb": wb,
                    "wratio": (wb / wa) if (wa and wb) else None})
    return out


def _main_effect(obs: dict, piece: str, rung: str):
    """Geometric mean of the four one-piece edges. Geometric because these are RATIOS: the
    arithmetic mean of 5.0x and 0.2x is 2.6x, which would call a piece that helps in one context
    exactly as much as it hurts in another a large win."""
    rs = [e["ratio"] for e in _edge_ratios(obs, piece, rung) if e["ratio"]]
    if not rs:
        return None, 0
    return math.exp(sum(math.log(r) for r in rs) / len(rs)), len(rs)


def table(obs: dict) -> str:
    rows = [
        f"Window `action:{ACTION}` -- the gesture that opens EVERY reasoning pane. The one-pane "
        f"gesture `action:{SECONDARY_ACTION}` is measured and printed but never scored: it caps "
        f"out at a single ~12,300-character trace and reads flat for every mechanism here.",
        "",
        "Frame rate is EFFECTIVE rate over wall time (`1000*raf.n/elapsed_ms`). `raf.fps_p50` is "
        "in every payload and quoted nowhere: on this scene it read 62.5 fps jammed and unjammed "
        "alike.",
        "",
        "Arms are the eight corners of a 2^3 factorial over `p2` (the streaming render rewrite), "
        "`p3a` (reasoning panes render code plain at any size) and `p3b` (the 4 KiB and 16 KiB "
        "code size thresholds), plus upstream `main` and a closure arm. `A111` is 9477 entire "
        "with pagination off, which is the arm the 4.981x was measured on; `Z` has all four "
        "mechanisms neutralised and should read like `main`.",
        "",
        "For comparison, the ablation already published on this branch in `6adc583a2`, measured "
        "on a STREAMING window at 250K streamed reasoning characters, worst main-thread freeze: "
        "base 6,821 ms, null 6,878, the streaming-render rewrite alone 869, plus the plain-code "
        "policy 211, plus pagination 79. That window is not this one, which is the point of "
        "running this.",
        "",
    ]

    for rung in _rungs(obs):
        s = rung_state(obs, rung)
        head = {"SCORED": "SCORED",
                "WORST_FRAME_ONLY": "WORST FRAME ONLY -- the rate channel is blind at this rung",
                "VOID": "VOID -- this rung answers nothing"}[s["state"]]
        rows += [f"### r{rung} -- {head}", ""]
        if s["notes"]:
            rows += ["Why: " + "; ".join(s["notes"]) + ".", ""]
        rows += [
            f"Control at THIS rung: rate {s['base_fps'] or float('nan'):.2f} -> "
            f"{(s['jam_fps'] if s['jam_fps'] is not None else float('nan')):.2f} fps, "
            f"worst frame {s['base_worst'] or float('nan'):,.0f} -> "
            f"{(s['jam_worst'] if s['jam_worst'] is not None else float('nan')):,.0f} ms, "
            f"with the page's main thread deliberately blocked 200 ms in every 250 ms.", ""]

        rows += ["| arm | " + " | ".join(PIECES) + " | eff fps | per rep | busy | worst frame "
                 "| blocked |",
                 "|---|" + "---|" * (len(PIECES) + 5)]
        for arm in ARMS:
            f = _mean(_vals(obs, arm, rung, _eff_fps))
            if f is None:
                continue
            fs = _vals(obs, arm, rung, _eff_fps)
            bu = _mean(_vals(obs, arm, rung, _busy))
            wo = _vals(obs, arm, rung, _worst)
            bl = _mean(_vals(obs, arm, rung, _blocked))
            m = (obs.get("states") or {}).get(arm, {}).get("piece_mask")
            cells = (["-"] * len(PIECES)) if not m else ["on" if b else "off" for b in m]
            rows.append("| " + " | ".join([
                f"`{arm}`", *cells, f"**{f:.1f}**",
                ", ".join(f"{x:.1f}" for x in fs) or "-",
                "-" if bu is None else f"{bu:.0f}%",
                "-" if not wo else f"{max(wo):,.0f} ms",
                "-" if bl is None else f"{bl:,.0f} ms"]) + " |")
        jf = _mean(_vals(obs, "JAM", rung, _eff_fps))
        jw = _mean(_vals(obs, "JAM", rung, _worst))
        if jf is not None:
            rows.append("| `JAM` (control) | " + " | ".join(["-"] * len(PIECES)) +
                        f" | **{jf:.1f}** | "
                        f"{', '.join(f'{x:.1f}' for x in _vals(obs, 'JAM', rung, _eff_fps))} | - | "
                        f"{jw or float('nan'):,.0f} ms | - |")
        rows.append("")

        rows += ["The lump being split, and whether the split closes:", "",
                 "| comparison | meaning | eff fps | worst frame |", "|---|---|---|---|"]
        for lo, hi, what in ((REFERENCE, WHOLE, "ALL of 9477, pagination off -- the 4.981x"),
                             (REFERENCE, NEUTRAL, "CLOSURE: all four of 9477's mechanisms neutralised")):
            a, b = _mean(_vals(obs, lo, rung, _eff_fps)), _mean(_vals(obs, hi, rung, _eff_fps))
            wa, wb = _mean(_vals(obs, lo, rung, _worst)), _mean(_vals(obs, hi, rung, _worst))
            rows.append("| " + " | ".join([
                f"`{lo}` -> `{hi}`", what,
                "-" if not (a and b) else f"**{b / a:.3f}x** ({a:.1f} -> {b:.1f})",
                "-" if not (wa and wb) else f"{wb / wa:.3f}x ({wa:,.0f} -> {wb:,.0f} ms)"]) + " |")
        rows.append("")

        rows += ["Every isolation edge. Each row differs from its partner by exactly one "
                 "mechanism:", "",
                 "| mechanism | off -> on | context | eff fps | worst frame | rep spread (fps) |",
                 "|---|---|---|---|---|---|"]
        for piece in SWEPT:
            for e in _edge_ratios(obs, piece, rung):
                rows.append("| " + " | ".join([
                    piece, f"`{e['off']}` -> `{e['on']}`", e["ctx"],
                    "-" if not e["ratio"] else f"**{e['ratio']:.3f}x** "
                                               f"({e['a']:.1f} -> {e['b']:.1f})",
                    "-" if not e["wratio"] else f"{e['wratio']:.3f}x "
                                                f"({e['wa']:,.0f} -> {e['wb']:,.0f} ms)",
                    f"{e['spread']:.1f}"]) + " |")
        rows.append("")

        # ── THE SPLIT THAT DECIDES WHAT SHIPS ────────────────────────────────────────────
        rows += ["Splitting `codeHighlighting=\"plain\"` into the two costs it removes together. "
                 "`code-fence-defer.tsx` is byte-identical between main and this branch, so the "
                 "deferral machinery is MAIN's own code and 9477 turns it off only as a side "
                 "effect. `M` keeps every colour and turns the machinery off directly:", "",
                 "| comparison | what it prices | eff fps | worst frame | rep spread (fps) |",
                 "|---|---|---|---|---|"]
        for lo, hi, what in DECOUPLE_EDGES:
            a, b = _mean(_vals(obs, lo, rung, _eff_fps)), _mean(_vals(obs, hi, rung, _eff_fps))
            wa, wb = _mean(_vals(obs, lo, rung, _worst)), _mean(_vals(obs, hi, rung, _worst))
            fl = max(_spread(_vals(obs, lo, rung, _eff_fps)) or 0.0,
                     _spread(_vals(obs, hi, rung, _eff_fps)) or 0.0)
            rows.append("| " + " | ".join([
                f"`{lo}` -> `{hi}`", what,
                "-" if not (a and b) else f"**{b / a:.3f}x** ({a:.1f} -> {b:.1f})",
                "-" if not (wa and wb) else f"{wb / wa:.3f}x ({wa:,.0f} -> {wb:,.0f} ms)",
                f"{fl:.1f}"]) + " |")
        a00 = _mean(_vals(obs, BASELINE, rung, _eff_fps))
        a01 = _mean(_vals(obs, arm_of((0, 1)), rung, _eff_fps))
        if a00 and a01:
            rows.append(f"| `{BASELINE}` -> `{arm_of((0, 1))}` | both together, which is all "
                        f"`codeHighlighting=\"plain\"` has ever been measured as | "
                        f"**{a01 / a00:.3f}x** ({a00:.1f} -> {a01:.1f}) | - | - |")
        rows.append("")

        # ── PREDICTED NULLS ──────────────────────────────────────────────────────────────
        rows += ["Predicted nulls, written down BEFORE the run so a flat reading is a result "
                 "rather than an unexplained flat reading:", "",
                 "| arm | mechanism turned off | predicted | measured | verdict |",
                 "|---|---|---|---|---|"]
        for arm, piece in ((NULL_P1, "p1"), (NULL_P3B, "p3b")):
            a = _mean(_vals(obs, WHOLE, rung, _eff_fps))
            b = _mean(_vals(obs, arm, rung, _eff_fps))
            if not (a and b):
                rows.append(f"| `{arm}` | {piece} | no change from `{WHOLE}` | - | not measured |")
                continue
            r = b / a
            held = (1.0 / NULL_TOLERANCE) <= r <= NULL_TOLERANCE
            rows.append(f"| `{arm}` | {piece} | no change from `{WHOLE}` | "
                        f"{r:.3f}x ({a:.1f} -> {b:.1f}) | "
                        f"{'CONFIRMED INERT' if held else 'PREDICTION FAILED'} |")
        rows.append("")

        rows += ["Main effect of each mechanism, the geometric mean of its edges (geometric "
                 "because these are ratios):", "",
                 "| mechanism | what it is | main effect | edges |", "|---|---|---|---|"]
        for piece in SWEPT:
            me, n = _main_effect(obs, piece, rung)
            rows.append(f"| {piece} | {PIECE_LABEL[piece]} | "
                        f"{'-' if me is None else f'**{me:.3f}x**'} | {n} |")
        rows.append("")

    # ── fidelity, from the settled census. Reported at every rung, gated by nothing, because
    #    it is a statement about what the arm RENDERS and not about how fast it did it.
    rows += ["### What each arm actually renders, once the DOM has stopped changing", "",
             "Taken by `action:reasoning_fidelity_settled`, which opens every pane and waits for "
             "the highlight-span count to stop changing before counting. This is a fidelity "
             "census, not a performance window, and its frame numbers are not scored anywhere.",
             ""]
    rows += ["9477 has TWO code-size thresholds, not one. `isOversizedStreamingCode` sends a "
             "fence down the plain-first path at 4,096 characters, and "
             "`shouldAutoHighlightStreamingCode` then refuses the upgrade back to a highlighted "
             "subtree above 16,384. Both are reported. 20,000 is main's `MAX_HIGHLIGHT_CHARS` and "
             "is carried for comparison ONLY: its gate has two callers and neither is the "
             "assistant thread, so it is not a precedent for either of these.", ""]
    for where, what in (("reasoning", "inside the reasoning panes, which is what this gesture "
                                      "opens"),
                        ("thread", "across the whole thread")):
        for rung in _rungs(obs):
            rows += [f"r{rung}, fenced code blocks {what}:", "",
                     "| arm | fences | highlighted | plain | spans | p50 chars | p99 chars | "
                     "max chars | >= 4,096 | > 16,384 | > 20,000 |",
                     "|---|---|---|---|---|---|---|---|---|---|---|"]
            for arm in ARMS:
                g = lambda f: _mean(_vals(obs, arm, rung, _fence_field(f, where)))  # noqa: E731
                n = g("fences")
                if n is None:
                    continue

                def s(x, fmt=",.0f"):
                    return "-" if x is None else format(x, fmt)
                rows.append("| " + " | ".join([
                    f"`{arm}`", s(n), s(g("highlighted_fences")), s(g("plain_fences")),
                    s(g("spans")), s(g("p50_chars")), s(g("p99_chars")), s(g("max_chars")),
                    s(g("over_p3_open")), s(g("over_p3_upgrade")),
                    s(g("over_main_cap"))]) + " |")
            rows.append("")

    st = obs.get("states") or {}
    rows += ["Arms, all built as patches on ONE upstream commit so nothing depends on a branch "
             "that can drift:", ""]
    for arm in ARMS:
        sarm = st.get(arm) or {}
        m = sarm.get("piece_mask")
        rows.append(f"- `{arm}` = `{str(sarm.get('ref'))[:9]}` + {sarm.get('patches') or 'nothing'}"
                    f"; pieces on = "
                    f"{[p for p, b in zip(PIECES, m or ()) if b] or 'none'}"
                    f"; bundle {sorted(x or '?' for x in _bundles(obs, arm))}")
    rows += ["", "Arms were interleaved within each repetition, so drift over the job cannot land "
             "entirely on whichever arm ran last."]
    return "\n".join(rows)


def verdict(obs: dict) -> tuple[str, str]:
    scored = _scored(obs)
    if not scored:
        states = "; ".join(f"r{r}={rung_state(obs, r)['state']}" for r in _rungs(obs))
        return "INCONCLUSIVE", (
            "no rung was scoreable: a rung is scored only if the reference arm exhibits the "
            "defect, the jammed control resolves AT THAT RUNG, and the reference rate is above "
            f"the {FLOOR_FPS} fps instrument floor. Rung states: {states}")

    rung = scored[0]
    base = _mean(_vals(obs, REFERENCE, rung, _eff_fps))
    whole = _mean(_vals(obs, WHOLE, rung, _eff_fps))
    neutral = _mean(_vals(obs, NEUTRAL, rung, _eff_fps))
    lump = (whole / base) if (base and whole) else None

    ranked = []
    for piece in SWEPT:
        me, n = _main_effect(obs, piece, rung)
        if me:
            ranked.append((me, piece, n))
    if not ranked:
        return "INCONCLUSIVE", "no piece produced a complete isolation edge at a scored rung"
    ranked.sort(reverse=True)
    top_ratio, top_piece, _ = ranked[0]

    edge_txt = "; ".join(
        f"{e['off']}->{e['on']} {e['ratio']:.2f}x (spread {e['spread']:.1f} fps)"
        for e in _edge_ratios(obs, top_piece, rung) if e["ratio"])
    others = ", ".join(f"{p} {m:.2f}x" for m, p, _ in ranked[1:])

    closure = ""
    if base and neutral:
        r = neutral / base
        if not ((1.0 / CLOSURE_TOLERANCE) <= r <= CLOSURE_TOLERANCE):
            closure = (f". CLOSURE FAILED: with all four of 9477's mechanisms neutralised {NEUTRAL} still reads "
                       f"{r:.2f}x the reference, so something in 9477 that none of these three "
                       f"patches turns off carries part of the lump and these shares are of less "
                       f"than the whole")
        else:
            closure = (f". Closure holds: {NEUTRAL} reads {r:.2f}x the reference, so the three "
                       f"neutralisations between them account for the change")

    detail = (f"at r{rung}, {REFERENCE} -> {WHOLE} (all of 9477, pagination off) is "
              f"{lump:.3f}x" if lump else f"at r{rung}")
    detail += (f". Splitting it by one-mechanism-at-a-time ablation over every context, "
               f"{top_piece} ({PIECE_LABEL[top_piece]}) has a main effect of {top_ratio:.3f}x "
               f"against {others or 'no other piece'}. Its edges: {edge_txt}{closure}")

    spreads = [e["spread"] for e in _edge_ratios(obs, top_piece, rung)
               if e["ratio"] and e["b"] is not None and e["a"] is not None]
    gaps = [e["b"] - e["a"] for e in _edge_ratios(obs, top_piece, rung) if e["ratio"]]
    beats_noise = bool(gaps and spreads and min(gaps) > max(spreads))

    if top_ratio >= MIN_RATIO and beats_noise:
        return "ATTRIBUTED", detail
    if top_ratio >= MIN_RATIO:
        return "ATTRIBUTED_WEAK", detail + (
            ". Flagged weak: at least one edge's gain does not clear the arms' own repetition "
            "spread, so the ranking is directional rather than settled")
    return "NO_SINGLE_OWNER", detail + (
        ". No piece clears the ratio threshold on its own, so the lump is either shared or "
        "carried by something these patches do not isolate")


def observed_capabilities(obs: dict) -> dict[str, bool]:
    st = obs.get("states") or {}
    return {
        "webkitgtk": bool(_runs(obs)),
        "headless_display_server": bool((obs.get("xserver") or {}).get("display")),
        "studio_production_bundle": all(
            (st.get(n) or {}).get("dist", {}).get("index_html") for n in ARMS),
        "studiobench_ladder": bool(_runs(obs)),
    }
