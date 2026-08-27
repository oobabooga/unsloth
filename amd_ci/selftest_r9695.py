#!/usr/bin/env python3
"""Self-test for criteria/studio_r9695_rebase.py, MUTATION-VERIFIED.

A self-test that passes proves nothing. Enforcement has been moved out of the shared, reviewed
`lib/differential.py` and into per-task criteria code, and per-task criteria code is exactly where
a check quietly stops checking. So every guard here is asserted twice:

  GREEN  the guard behaves on a synthetic payload whose right answer is known;
  RED    the same assertion is re-run against a DELIBERATELY BROKEN copy of the criteria module,
         and it must FAIL there. A guard that cannot be broken was never testing anything.

The mutations are not arbitrary. Each one is the specific edit that would turn this run into the
confident wrong answer it is most at risk of:

  base_never_collapsed  drop the VOID rule, so a fast head arm against a base that was never slow
                        scores as a win. This is the failure `differential.py` exists to prevent
                        and the one this module has taken responsibility for.
  busy_null_is_zero     treat an unreadable `busy_pct` as 0.0 instead of MISSING. On run
                        33040070879 the untreated version of this would have published
                        "the pre arm did NOT exhibit the reported collapse" about a page rendering
                        one frame every eight seconds.
  idle_control_off      stop discarding repetitions whose idle window was already stalling. This
                        is how a `plain` arm once read +50% at r500K with its idle control failing
                        in 3 repetitions of 5.
  jam_control_off       stop discarding repetitions whose frame channel could not see a
                        deliberately blocked main thread.
  ratio_ignores_overlap call a difference a move even when the two arms' repetition ranges overlap.

Usage:  python3 amd_ci/selftest_r9695.py
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

ROOT = Path(__file__).parent
CRIT = ROOT / "criteria" / "studio_r9695_rebase.py"

FAILED: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if cond else 'FAIL'} {name}" + (f"  -- {detail}" if not cond else ""))
    if not cond:
        FAILED.append(name)


def load(source: str | None = None):
    """Load the criteria module, optionally from mutated source."""
    text = source if source is not None else CRIT.read_text()
    mod = types.ModuleType("crit_under_test")
    mod.__file__ = str(CRIT)
    exec(compile(text, str(CRIT), "exec"), mod.__dict__)          # noqa: S102
    return mod


# ── synthetic payloads ───────────────────────────────────────────────────────────────────────
def action(name, fps, elapsed_ms=20000.0, busy=70.0, worst_ms=800.0, blocked=12.0,
           robust=None, p50_fps=62.5):
    """One action window whose `eff_fps` and `1000*raf.n/elapsed_ms` agree BY CONSTRUCTION.

    The criteria module recomputes the headline rather than reading it and gates on the two
    agreeing, so a fixture that fabricated them independently would fail a gate for a reason that
    has nothing to do with what the test is about.
    """
    n = max(1, round(fps * elapsed_ms / 1000.0))
    a = {
        "name": name, "ok": True, "not_applicable": False,
        "elapsed_ms": elapsed_ms, "frames": n,
        "eff_fps": round(1000.0 * n / elapsed_ms, 1),
        "blocked_ms_per_frame": blocked,
        "robust": {"blocked_ms_per_frame": robust if robust is not None else blocked,
                   "frames": n, "stall_frames_over_1s": 0, "worst_gap_ms": worst_ms},
        "raf": {"n": n, "max_ms": worst_ms, "fps_p50": p50_fps, "p50_ms": 1000.0 / p50_fps},
        "busy": ({"busy_pct": busy, "blocked_ms": blocked * n, "busy_pct_reason": None}
                 if busy is not None else
                 {"busy_pct": None, "blocked_ms": None,
                  "busy_pct_reason": "only 10 idle ticks, need 40"}),
        "census_before": {}, "census_after": {},
    }
    return a


def payload(*, arm, rung, fps, busy=70.0, worst_ms=800.0, spans=50000, fences=300,
            idle_fps=61.5, idle_worst=18.0, jam_fps=17.2, blocked=12.0, robust=None,
            settled=True, deferred=0):
    drop = None if not idle_fps else round(1 - jam_fps / idle_fps, 3)
    return {
        "ok": True, "arm": arm, "rung": rung, "skipped_send": True,
        "ua": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/605.1.15",
        "engine_probe": {"is_webkit_gtk_ua": True, "has_chrome": False,
                         "has_webkit_message_handlers": True, "vendor": "Apple Computer, Inc."},
        "mount": {"ms": 900, "by": "last_seeded_marker", "census": {}},
        "run_meta": {"instrument_hash": "deadbeefdeadbeef", "corpus_hash": "23cd2464",
                     "bundle_hash": f"bundle_{arm}", "rung": rung, "arm": arm},
        "clamp": {"clamp_ms": 8.0, "reason": None},
        "liveness": {"clean_fps": idle_fps, "jammed_fps": jam_fps, "drop_fraction": drop,
                     "clean_fps_p50": 62.5, "jammed_fps_p50": 62.5},
        "phases": [
            {"phase": "idle", "eff_fps": idle_fps, "elapsed_ms": 6000,
             "raf": {"n": int(idle_fps * 6), "max_ms": idle_worst, "fps_p50": 62.5},
             "busy": {"busy_pct": 2.0, "blocked_ms": 120.0, "busy_pct_reason": None}},
            {"phase": "idle_jammed", "eff_fps": jam_fps, "elapsed_ms": 6000, "jammed": True,
             "raf": {"n": int(jam_fps * 6), "max_ms": 210.0, "fps_p50": 62.5},
             "busy": {"busy_pct": 79.0, "blocked_ms": 4700.0, "busy_pct_reason": None}},
            {"phase": "scroll", "eff_fps": 40.0, "elapsed_ms": 8000,
             "raf": {"n": 320, "max_ms": 120.0, "fps_p50": 62.5},
             "busy": {"busy_pct": 30.0, "blocked_ms": 2400.0, "busy_pct_reason": None}},
        ],
        "actions": [
            action("reasoning_toggle", fps * 1.4, 6000.0, busy, worst_ms / 2, blocked / 2),
            action("reasoning_toggle_all", fps, 20000.0, busy, worst_ms, blocked, robust),
            {"name": "reasoning_fidelity_settled", "ok": True, "not_applicable": False,
             "settled": settled, "settle_polls": 6, "elapsed_ms": 9000, "frames": 300,
             "eff_fps": 33.3, "blocked_ms_per_frame": 5.0,
             "robust": {"blocked_ms_per_frame": 5.0},
             "raf": {"n": 300, "max_ms": 90.0, "fps_p50": 62.5},
             "busy": {"busy_pct": 20.0, "blocked_ms": 1800.0, "busy_pct_reason": None},
             "fence_census": {"reasoning": {"fences": fences, "spans": spans},
                              "thread": {"fences": fences + 40, "spans": spans + 5000},
                              "reasoning_deferred_shells": deferred}},
            action("select_all_copy", 60.0, 3000.0, 10.0, 40.0, 1.0),
        ],
        "final": {},
    }


def obs_for(cells: dict, rungs=("100K", "500K"), reps=5) -> dict:
    """`cells[(arm, rung)]` is a list of kwargs, one per repetition."""
    runs = []
    plan = []
    for rung in rungs:
        for arm in ("main", "head"):
            spec = cells.get((arm, rung))
            if spec is None:
                continue
            for i, kw in enumerate(spec, start=1):
                runs.append({"arm": arm, "rung": rung, "rep": str(i), "port": 5600 + len(runs),
                             "rc": 0, "payload": payload(arm=arm, rung=rung, **kw)})
                plan.append({"rung": rung, "rep": str(i), "arm": arm})
    st = {a: {"head_marker_present": a == "head", "patch_ok": True,
              "exported_bundle_hash": f"bundle_{a}", "bundle_hash_at_measure": f"bundle_{a}",
              "bundle_hash_matches_build": True,
              "dist": {"index_html": True, "asset_files": 500},
              "patch_steps": []} for a in ("main", "head")}
    return {
        "rungs": list(rungs), "reps": {r: reps for r in rungs},
        "base_ref": "0be140dbd458535f7f93dc1eaffe703611ff9acf",
        "head_marker": "MarkdownCodeHighlightingContext",
        "xserver": {"display": ":101"},
        "states": st,
        "build": {"clone": {"checkout_ok": True,
                            "ref_landed": "0be140dbd458535f7f93dc1eaffe703611ff9acf",
                            "commit_line": "0be140dbd 2026-08-26 fix(prompt storage)"},
                  "preflight": {"marker_absent_at_base": True,
                                "marker_present_after_apply": True,
                                "marker_after_revert": False,
                                "changed_outside_frontend": []}},
        "plan": plan, "runs": runs,
    }


def healthy(fps, **kw):
    return dict(fps=fps, **kw)


# ── the scenarios, each with its right answer known ──────────────────────────────────────────
def sc_win_survives() -> dict:
    """main slow and loaded, head five times faster, ranges nowhere near each other."""
    return obs_for({
        ("main", "100K"): [healthy(f, busy=93.0, worst_ms=1779.0, spans=74250)
                           for f in (5.0, 5.1, 5.2, 5.0, 5.1)],
        ("head", "100K"): [healthy(f, busy=60.0, worst_ms=420.0, spans=10917, deferred=300)
                           for f in (25.0, 25.4, 25.3, 25.1, 25.5)],
        ("main", "500K"): [healthy(f, busy=95.0, worst_ms=2600.0, spans=248000)
                           for f in (3.0, 3.1, 2.9, 3.0, 3.2)],
        ("head", "500K"): [healthy(f, busy=70.0, worst_ms=900.0, spans=11286, deferred=600)
                           for f in (12.0, 12.4, 12.1, 11.9, 12.3)],
    })


def sc_win_absorbed() -> dict:
    """main is still loaded -- the rung is a real venue -- and head simply does not help."""
    return obs_for({
        ("main", "100K"): [healthy(f, busy=88.0, worst_ms=1200.0, spans=12000)
                           for f in (24.0, 25.5, 23.8, 26.0, 24.6)],
        ("head", "100K"): [healthy(f, busy=86.0, worst_ms=1150.0, spans=9800, deferred=280)
                           for f in (25.0, 24.2, 26.1, 23.9, 25.4)],
        ("main", "500K"): [healthy(f, busy=94.0, worst_ms=2400.0, spans=13000)
                           for f in (11.0, 11.8, 10.6, 12.0, 11.2)],
        ("head", "500K"): [healthy(f, busy=93.0, worst_ms=2300.0, spans=11000, deferred=550)
                           for f in (11.4, 10.9, 11.9, 11.1, 11.6)],
    })


def sc_base_never_collapsed() -> dict:
    """THE ONE THAT MUST BE VOID. main is quick and idle-ish, head is quicker still."""
    return obs_for({
        ("main", "100K"): [healthy(f, busy=8.0, worst_ms=40.0) for f in (12.0, 12.2, 11.8, 12.1, 12.0)],
        ("head", "100K"): [healthy(f, busy=6.0, worst_ms=30.0, deferred=280)
                           for f in (58.0, 59.0, 58.5, 59.2, 58.8)],
        ("main", "500K"): [healthy(f, busy=9.0, worst_ms=50.0) for f in (12.0, 12.2, 11.8, 12.1, 12.0)],
        ("head", "500K"): [healthy(f, busy=7.0, worst_ms=35.0, deferred=550)
                           for f in (58.0, 59.0, 58.5, 59.2, 58.8)],
    })


def sc_busy_null() -> dict:
    """The clamp did not calibrate on main. Every frame rate is the collapse verbatim."""
    return obs_for({
        ("main", "100K"): [healthy(f, busy=None, worst_ms=8400.0) for f in (0.1, 0.3, 0.2, 0.1, 0.2)],
        ("head", "100K"): [healthy(f, busy=60.0, worst_ms=420.0, deferred=280)
                           for f in (25.0, 25.4, 25.3, 25.1, 25.5)],
    }, rungs=("100K",))


def sc_idle_stalling() -> dict:
    """The r500K `plain` arm from 2026-08-26: it looks like +50% and its idle control failed."""
    head = [healthy(4.16, busy=91.0, worst_ms=3000.0, idle_fps=18.6, idle_worst=9300.0),
            healthy(4.20, busy=91.0, worst_ms=3000.0, idle_fps=14.8, idle_worst=7900.0),
            healthy(4.10, busy=91.0, worst_ms=3000.0, idle_fps=4.9, idle_worst=8800.0),
            healthy(2.80, busy=91.0, worst_ms=3000.0),
            healthy(2.75, busy=91.0, worst_ms=3000.0)]
    return obs_for({
        ("main", "500K"): [healthy(f, busy=93.0, worst_ms=2900.0) for f in (2.77, 2.80, 2.70, 2.85, 2.75)],
        ("head", "500K"): head,
    }, rungs=("500K",))


def sc_idle_stalling_jam_resolves() -> dict:
    """The case where ONLY the idle control can save the run.

    In the real r500K episode the jam was priced against the same stalled idle window, so the jam
    control failed too and either guard would have caught it. That makes the idle guard look
    redundant, and a guard that is only ever exercised alongside another one is a guard nobody
    would notice losing. Here the page is stalling at rest AND the jam still resolves cleanly
    (18.6 -> 4.0 fps is a 78% drop, comfortably over the bar), so the jam control passes and the
    ONLY thing standing between the raw medians and a published +50% is the idle control.
    """
    head = [healthy(4.16, busy=91.0, worst_ms=3000.0, idle_fps=18.6, idle_worst=9300.0, jam_fps=4.0),
            healthy(4.20, busy=91.0, worst_ms=3000.0, idle_fps=14.8, idle_worst=7900.0, jam_fps=3.2),
            healthy(4.10, busy=91.0, worst_ms=3000.0, idle_fps=4.9, idle_worst=8800.0, jam_fps=1.1),
            healthy(2.80, busy=91.0, worst_ms=3000.0),
            healthy(2.75, busy=91.0, worst_ms=3000.0)]
    return obs_for({
        ("main", "500K"): [healthy(f, busy=93.0, worst_ms=2900.0)
                           for f in (2.77, 2.80, 2.70, 2.85, 2.75)],
        ("head", "500K"): head,
    }, rungs=("500K",))


def sc_jam_blind() -> dict:
    """The frame channel did not see a deliberately blocked main thread, in every repetition."""
    return obs_for({
        ("main", "100K"): [healthy(f, busy=93.0, worst_ms=1700.0, jam_fps=60.9)
                           for f in (5.0, 5.1, 5.2, 5.0, 5.1)],
        ("head", "100K"): [healthy(f, busy=60.0, worst_ms=420.0, jam_fps=60.9, deferred=280)
                           for f in (25.0, 25.4, 25.3, 25.1, 25.5)],
    }, rungs=("100K",))


def sc_overlapping() -> dict:
    """A 1.2x point ratio whose repetition ranges overlap. Five readings cannot resolve it."""
    return obs_for({
        ("main", "100K"): [healthy(f, busy=90.0, worst_ms=1400.0) for f in (5.0, 7.0, 4.5, 8.0, 5.5)],
        ("head", "100K"): [healthy(f, busy=88.0, worst_ms=1300.0, deferred=280)
                           for f in (6.0, 8.5, 5.2, 9.0, 6.6)],
    }, rungs=("100K",))


# ── the assertions, written so a mutant can be run through exactly the same ones ─────────────
def assertions(crit) -> list[tuple[str, bool, str]]:
    out: list[tuple[str, bool, str]] = []

    v, why = crit.verdict(sc_win_survives())
    out.append(("a real 5x with separated ranges reads WIN_SURVIVES", v == "WIN_SURVIVES", f"{v}: {why}"))

    v, why = crit.verdict(sc_win_absorbed())
    out.append(("a dead win reads WIN_ABSORBED and is a FINDING, not a failure",
                v == "WIN_ABSORBED", f"{v}: {why}"))
    out.append(("and WIN_ABSORBED keeps the job green (it is not in announce.py's NO_RESULT)",
                v not in ("VOID", "INCONCLUSIVE"), f"{v}"))

    o = sc_base_never_collapsed()
    v, why = crit.verdict(o)
    out.append(("base clean + head fast is VOID, NOT a pass", v == "VOID", f"{v}: {why}"))
    out.append(("and the reason names the base arm rather than the ratio",
                "does not exhibit the defect" in why, why))
    # A BARE VOID IS NOT AN ANSWER A READER CAN USE. At the rung the 5.02x was measured at, a base
    # that no longer collapses is very nearly the OPPOSITE of "we could not tell", and the report
    # has to say so or the run reads as a harness failure.
    out.append(("a VOID at r100K explains what it means for the question, not just the rule",
                "no longer there to remove" in why and "NOT because the measurement failed" in why,
                why))
    out.append(("and it quotes what main read when the 5.02x was taken, so the change is visible",
                "5.1 fps" in why and "93% busy" in why, why))
    out.append(("both rungs are VOID, so neither can license the other",
                all(crit.rung_state(o, g)["state"] == "VOID" for g in ("100K", "500K")),
                str({g: crit.rung_state(o, g)["state"] for g in ("100K", "500K")})))

    o = sc_busy_null()
    v, why = crit.verdict(o)
    out.append(("an unreadable busy_pct is INCONCLUSIVE naming the instrument",
                v == "INCONCLUSIVE", f"{v}: {why}"))
    out.append(("and it names the CLAMP", "CLAMP" in why.upper(), why))
    out.append(("and it does NOT say the base arm was never slow",
                "does not exhibit the defect" not in why and "did NOT exhibit" not in why, why))
    c = crit.cell(o, "main", "100K")
    out.append(("a null busy is MISSING, never averaged as zero",
                c["busy"]["n"] == 0 and c["busy"]["mean"] is None and c["busy_missing"] == 5,
                str(c["busy"]) + f" missing={c['busy_missing']}"))

    o = sc_idle_stalling()
    kept = crit.cell(o, "head", "500K")
    out.append(("the three repetitions whose idle window was stalling are DISCARDED",
                kept["reps_discarded"] == 3 and kept["reps_surviving"] == 2,
                f"kept {kept['reps_surviving']}, dropped {kept['reps_discarded']}"))
    out.append(("and the discard reason names the idle window",
                any("IDLE" in w for d in kept["discards"] for w in d["why"]),
                str([d["why"] for d in kept["discards"]])[:300]))
    v, why = crit.verdict(o)
    out.append(("a cell left with too few repetitions is INCONCLUSIVE, not a median over what "
                "survived", v == "INCONCLUSIVE", f"{v}: {why}"))
    out.append(("and the +50% that the raw medians would have shown is never published",
                "1.5" not in why and "+50" not in why, why))

    # THE IDLE CONTROL, ON ITS OWN. Above, the jam was priced against the same stalled window and
    # would have caught these repetitions too. Here it resolves cleanly, so the idle control is
    # the only thing between the raw medians and a published +50%.
    o = sc_idle_stalling_jam_resolves()
    kept = crit.cell(o, "head", "500K")
    out.append(("with the JAM control resolving, the idle control alone discards the three "
                "stalling repetitions",
                kept["reps_discarded"] == 3 and kept["reps_surviving"] == 2,
                f"kept {kept['reps_surviving']}, dropped {kept['reps_discarded']}; "
                + str([d["why"] for d in kept["discards"]])[:200]))
    out.append(("and every one of those discards is attributed to the IDLE window",
                bool(kept["discards"]) and all(any("IDLE" in w for w in d["why"])
                                               for d in kept["discards"]),
                str([d["why"] for d in kept["discards"]])[:300]))
    v, why = crit.verdict(o)
    out.append(("so the +50% those repetitions would have produced is never scored",
                v == "INCONCLUSIVE", f"{v}: {why}"))
    out.append(("and the surviving medians do not read as a win either",
                "1.5" not in why, why))

    o = sc_jam_blind()
    kept = crit.cell(o, "main", "100K")
    out.append(("a repetition whose JAMMED control did not resolve is discarded",
                kept["reps_surviving"] == 0, f"kept {kept['reps_surviving']}"))
    v, why = crit.verdict(o)
    out.append(("and a run with no resolving control concludes nothing",
                v == "INCONCLUSIVE", f"{v}: {why}"))

    o = sc_overlapping()
    r = crit.ratio(o, "100K")
    out.append(("overlapping repetition ranges are reported as overlapping",
                r["separated"] is False, str(r)))
    v, why = crit.verdict(o)
    out.append(("and an unseparated difference is not called a move",
                v == "WIN_ABSORBED", f"{v}: {why}"))

    # The gates, on a payload where everything is right, then with one thing wrong.
    o = sc_win_survives()
    g = {n: ok for n, ok, _ in crit.gates(o)}
    out.append(("every gate passes on a well-formed run", all(g.values()),
                str({k: v for k, v in g.items() if not v})))

    o2 = sc_win_survives()
    o2["states"]["head"]["exported_bundle_hash"] = "bundle_main"
    o2["states"]["head"]["bundle_hash_at_measure"] = "bundle_main"
    for r in o2["runs"]:
        r["payload"]["run_meta"]["bundle_hash"] = "bundle_main"
    g2 = {n: ok for n, ok, _ in crit.gates(o2)}
    out.append(("two arms that hash the same fail the two-bundles gate",
                not g2["the two arms are TWO bundles, not one"], str(g2)))

    o3 = sc_win_survives()
    o3["states"]["head"]["head_marker_present"] = False
    g3 = {n: ok for n, ok, _ in crit.gates(o3)}
    out.append(("an arm whose source does not carry the marker it is labelled with fails a gate",
                not g3["each arm's built SOURCE carries the marker state it is labelled with"],
                str(g3)))

    o4 = sc_win_survives()
    o4["runs"][0]["payload"]["engine_probe"]["has_chrome"] = True
    g4 = {n: ok for n, ok, _ in crit.gates(o4)}
    out.append(("a session that was not WebKitGTK fails a gate",
                not g4["every session really ran in WebKitGTK"], str(g4)))

    o5 = sc_win_survives()
    o5["runs"][0]["payload"]["run_meta"]["corpus_hash"] = "ffffffff"
    g5 = {n: ok for n, ok, _ in crit.gates(o5)}
    out.append(("a second corpus in the same run fails a gate",
                not g5["ONE corpus across every session, and it is the one the 5.02x was taken "
                       "on"], str(g5)))

    o6 = sc_win_survives()
    o6["runs"][0]["payload"]["actions"][1]["eff_fps"] = 999.0
    g6 = {n: ok for n, ok, _ in crit.gates(o6)}
    out.append(("a scene eff_fps that disagrees with the recomputation fails a gate",
                not g6["the scene's eff_fps and this module's recomputation agree"], str(g6)))

    # The verdict artifact has to SAY that enforcement moved out of differential.py.
    md = crit.table(sc_win_absorbed())
    out.append(("the verdict artifact states that lib/differential.py was not used",
                "differential.py" in md and "NOT used" in md, md[:200]))
    out.append(("and it says the VOID rule is kept verbatim",
                "VOID and scores nothing" in md, md[:200]))
    out.append(("the report prints a median beside every mean",
                "fps median" in md and "fps mean" in md
                and "blocked ms/frame p50" in md and "blocked ms/frame mean" in md, md[:400]))
    out.append(("and a MISSING busy is rendered as MISSING rather than as a number",
                "MISSING" in crit.table(sc_busy_null()), ""))
    return out


# ── the mutants ──────────────────────────────────────────────────────────────────────────────
#: name -> (what it breaks, [(from, to), ...])
MUTANTS = {
    "base_never_collapsed": (
        "the VOID rule: score every rung whether or not the base arm was ever slow",
        [('st["state"] = "SCORED" if (st["defect"] and not st["at_floor"]) else "VOID"',
          'st["state"] = "SCORED" if not st["at_floor"] else "VOID"')]),
    "busy_null_is_zero": (
        "treat an unreadable busy_pct as 0.0 rather than MISSING",
        [('    return ((_action(payload, name).get("busy")) or {}).get("busy_pct")',
          '    v = ((_action(payload, name).get("busy")) or {}).get("busy_pct")\n'
          '    return 0.0 if v is None else v')]),
    "idle_control_off": (
        "stop discarding repetitions whose idle window was already stalling",
        [("IDLE_MIN_FPS = 45.0", "IDLE_MIN_FPS = 0.0"),
         ("IDLE_MAX_WORST_MS = 1000.0", "IDLE_MAX_WORST_MS = 1e12")]),
    "jam_control_off": (
        "stop discarding repetitions whose channel could not see a deliberate jam",
        [("CONTROL_MIN_RATE_DROP = 0.25", "CONTROL_MIN_RATE_DROP = -1.0")]),
    "ratio_ignores_overlap": (
        "call a difference a move even when the arms' repetition ranges overlap",
        [('out["separated"] = bool(h["min"] > b["max"] or b["min"] > h["max"])',
          'out["separated"] = True')]),
    "thin_cell_accepted": (
        "score a cell that kept only one or two repetitions",
        [("MIN_SURVIVING_REPS = 3", "MIN_SURVIVING_REPS = 1")]),
}


def mutate(name: str) -> str:
    text = CRIT.read_text()
    _, edits = MUTANTS[name]
    for frm, to in edits:
        if frm not in text:
            raise SystemExit(
                f"mutation {name!r} does not apply: {frm!r} is not in the criteria module any "
                f"more. The mutation test is now vacuous, which is the exact failure it exists "
                f"to catch, so this is a hard error rather than a skip.")
        text = text.replace(frm, to, 1)
    return text


def main() -> int:
    print("== GREEN: the criteria module as shipped")
    crit = load()
    for name, cond, detail in assertions(crit):
        check(name, cond, detail)

    baseline = {name: cond for name, cond, _ in assertions(crit)}

    print("\n== RED: the same assertions against deliberately broken copies")
    print("   A mutant that changes NOTHING means the assertion above was not testing anything.")
    uncaught: list[str] = []
    for mname, (what, _) in MUTANTS.items():
        try:
            mutant = load(mutate(mname))
        except SystemExit as e:
            print(f"  MUTATION {mname}: {e}")
            FAILED.append(f"mutation {mname} does not apply")
            continue
        try:
            got = {name: cond for name, cond, _ in assertions(mutant)}
        except Exception as e:                                           # noqa: BLE001
            # A mutant that CRASHES is still caught: the assertion could not be satisfied.
            print(f"  caught  {mname:<24} ({what})")
            print(f"          the mutant raised {type(e).__name__}: {str(e)[:120]}")
            continue
        broke = [n for n, ok in baseline.items() if ok and not got.get(n, True)]
        if broke:
            print(f"  caught  {mname:<24} ({what})")
            for n in broke[:4]:
                print(f"          RED: {n}")
        else:
            print(f"  MISSED  {mname:<24} ({what})")
            print(f"          every assertion still passed against the broken module, so nothing "
                  f"here is testing this guard")
            uncaught.append(mname)
    for m in uncaught:
        FAILED.append(f"mutation {m} was not caught")

    print()
    if FAILED:
        print(f"{len(FAILED)} FAILED:")
        for f in FAILED:
            print(f"  - {f}")
        return 1
    print(f"all checks passed, and all {len(MUTANTS)} mutations were caught")
    return 0


if __name__ == "__main__":
    sys.exit(main())
