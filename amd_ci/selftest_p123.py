#!/usr/bin/env python3
"""Self-test for criteria/studio_p123_attribution.py.

The case this exists for is the r500K gate. The p4 criteria took the jammed control globally --
`ctrl_ok = ctrl_ok or drop >= CONTROL_MIN_DROP_PCT` -- so a control that resolved at r100K
licensed r500K too. At r500K, in the real run 32819830754, the reference arm read 0.08 effective
fps and the DELIBERATELY JAMMED arm read 0.14: the blocked page was faster than the clean one,
because both were far past the point where the channel could resolve anything. Under that gate
r500K scored, produced the largest ratio in the run, and carried the verdict.

Every number below is transcribed from that run's observations.json, so this is a replay of a
real failure and not an invented one.

Run: python3 amd_ci/selftest_p123.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


C = _load(HERE / "criteria" / "studio_p123_attribution.py", "p123crit")

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))
    if not cond:
        FAILURES.append(name)


def run(arm: str, rung: str, rep: str, *, n: int, elapsed: int, busy, worst: float,
        spans_reasoning: int = 1000, spans_thread: int = 7000, fences: int = 10,
        elements: int = 40000, reasoning_chars: int = 200000, settled: bool = True,
        bundle: str | None = None) -> dict:
    """One measurement, shaped exactly like a real payload."""
    act = {
        "name": C.ACTION, "ok": True, "elapsed_ms": elapsed,
        "raf": {"n": n, "max_ms": worst, "p99_ms": worst, "fps_p50": 62.5},
        "busy": {"busy_pct": busy, "blocked_ms": 1000},
        "census_open": {"reasoning_chars": reasoning_chars, "elements": elements,
                        "highlight_spans": spans_thread},
    }
    fid = {
        "name": C.FIDELITY_ACTION, "ok": True, "elapsed_ms": 8000,
        "raf": {"n": 100, "max_ms": 50}, "busy": {"busy_pct": 10, "blocked_ms": 10},
        "settled": settled,
        "census_open": {"reasoning_chars": reasoning_chars, "elements": elements,
                        "highlight_spans": spans_thread},
        "fence_census": {
            "reasoning": {"fences": fences, "spans": spans_reasoning,
                          "highlighted_fences": fences if spans_reasoning else 0,
                          "plain_fences": 0 if spans_reasoning else fences,
                          "p50_chars": 900, "p99_chars": 30000, "max_chars": 41000,
                          "over_p3_open": 3, "over_p3_upgrade": 1, "over_main_cap": 1},
            "thread": {"fences": fences * 2, "spans": spans_thread,
                       "highlighted_fences": fences, "plain_fences": fences,
                       "p50_chars": 800, "p99_chars": 28000, "max_chars": 41000,
                       "over_p3_open": 6, "over_p3_upgrade": 2, "over_main_cap": 2},
        },
    }
    return {"arm": arm, "rung": rung, "rep": rep, "payload": {
        "ok": True, "actions": [act, fid],
        "engine_probe": {"is_webkit_gtk_ua": True},
        "run_meta": {"bundle_hash": bundle or f"bundle_{arm}", "corpus_hash": C.EXPECTED_CORPUS,
                     "instrument_pacer_file": "/w/instrument/tests/studio/studiobench/pacer.py",
                     "instrument_sb_root": "/w/instrument", "instrument_hash": "cafebabe"},
    }}


def base_obs(runs: list[dict], rungs: list[str]) -> dict:
    states = {}
    for arm in C.ARMS:
        if arm == C.REFERENCE:
            mask = None
        elif arm == C.CLOSURE:
            mask = [False, False, False, False]
        else:
            mask = [True] + [c == "1" for c in arm[1:]]
        states[arm] = {"ref": C_REF, "ref_landed": C_REF, "commit": C_REF, "checkout_ok": True,
                       "patch_ok": True, "piece_mask": mask, "patches": [],
                       "dist": {"index_html": True, "asset_files": 607},
                       "install": {"rc": 0}}
    return {"rungs": rungs, "states": states, "runs": runs,
            "xserver": {"display": ":99"}}


C_REF = "90f85fdbf8fd6f9df1b99aff44f1e45da4c808d0"


# ── 1. THE REGRESSION THIS FILE EXISTS FOR ───────────────────────────────────────────────────
def test_r500k_control_is_per_rung() -> None:
    """r500K must be VOID: its own control reads backwards, whatever r100K's control did."""
    print("\n[1] the jammed control is evaluated at the rung it protects")
    runs = []
    # r100K: a control that works. reference 5.1 fps / 93% busy / 1,779 ms; JAM 1.3 fps / 1,816 ms.
    for rep in ("1", "2"):
        runs.append(run(C.REFERENCE, "100K", rep, n=27, elapsed=5106, busy=92.7, worst=1779))
        runs.append(run(C.WHOLE, "100K", rep, n=128, elapsed=5070, busy=60.6, worst=1771,
                        spans_reasoning=0))
    runs.append(run("JAM", "100K", "jam", n=7, elapsed=5406, busy=98.5, worst=1816))
    # r500K: the real numbers. The jammed arm is FASTER and its worst frame is no worse.
    for rep in ("1", "2"):
        runs.append(run(C.REFERENCE, "500K", rep, n=2, elapsed=24837, busy=100.3, worst=15078))
        runs.append(run(C.WHOLE, "500K", rep, n=2, elapsed=25565, busy=100.2, worst=15778,
                        spans_reasoning=0))
    runs.append(run("JAM", "500K", "jam", n=3, elapsed=21645, busy=95.8, worst=15228))

    obs = base_obs(runs, ["100K", "500K"])
    s100 = C.rung_state(obs, "100K")
    s500 = C.rung_state(obs, "500K")

    check("r100K scores", s100["state"] == "SCORED", s100["state"])
    check("r500K is VOID and not scored", s500["state"] == "VOID", s500["state"])
    check("r500K's reference arm DOES exhibit the defect, so VOID is not coming from that",
          s500["defect"] is True)
    check("r500K's rate control is reported as blind", s500["rate_control_ok"] is False,
          f"drop {s500['rate_drop_pct']:+.0f}%")
    check("r500K's worst-frame control is ALSO reported as blind",
          s500["worst_control_ok"] is False, f"rise {s500['worst_rise_pct']:+.0f}%")
    check("r500K names the reason in words", any("cannot report a blocked main thread" in n
                                                 for n in s500["notes"]))
    check("only r100K is offered for scoring", C._scored(obs) == ["100K"])

    # and the inherited bug is really gone: a global OR would have passed r500K here
    global_or = any(C.rung_state(obs, r)["rate_control_ok"] for r in ("100K", "500K"))
    check("a GLOBAL control would have passed r500K, which is the bug being fixed",
          global_or is True)


# ── 2. the instrument floor ──────────────────────────────────────────────────────────────────
def test_floor_downgrades_rather_than_scores() -> None:
    """A rung whose control resolves but whose reference is at the floor is worst-frame only."""
    print("\n[2] a reference arm at the instrument floor cannot carry a rate ratio")
    runs = []
    # control resolves on BOTH channels, but the reference presents 0.1 fps
    for rep in ("1", "2"):
        runs.append(run(C.REFERENCE, "500K", rep, n=2, elapsed=20000, busy=100.0, worst=8000))
        runs.append(run(C.WHOLE, "500K", rep, n=10, elapsed=5000, busy=90.0, worst=2000))
    runs.append(run("JAM", "500K", "jam", n=1, elapsed=30000, busy=99.0, worst=20000))
    obs = base_obs(runs, ["500K"])
    s = C.rung_state(obs, "500K")
    check("rate control resolves", s["rate_control_ok"] is True, f"{s['rate_drop_pct']:+.0f}%")
    check("worst-frame control resolves", s["worst_control_ok"] is True,
          f"{s['worst_rise_pct']:+.0f}%")
    check("reference is flagged as at the floor", s["at_floor"] is True, f"{s['base_fps']:.2f} fps")
    check("the rung is WORST_FRAME_ONLY, not SCORED", s["state"] == "WORST_FRAME_ONLY", s["state"])
    check("and it is therefore not offered for rate scoring", C._scored(obs) == [])


# ── 3. attribution actually attributes ───────────────────────────────────────────────────────
def _factorial_obs(fps_by_arm: dict, *, spans: dict | None = None) -> dict:
    runs = [run("JAM", "100K", "jam", n=7, elapsed=5406, busy=98.5, worst=1816)]
    for arm, fps in fps_by_arm.items():
        for rep in ("1", "2", "3"):
            n = max(1, int(round(fps * 5.0)))
            runs.append(run(arm, "100K", rep, n=n, elapsed=5000,
                            busy=max(5.0, 95.0 - fps * 1.6),
                            worst=max(50.0, 1800.0 - fps * 30),
                            spans_reasoning=(spans or {}).get(arm, 4271)))
    return base_obs(runs, ["100K"])


def test_attributes_to_the_owning_mechanism() -> None:
    """p3a carries the lump; the report must name p3a and not p2."""
    print("\n[3] the mechanism that owns the win is the one named")
    # p3a on -> ~5x, everything else flat. Bits are (p2, p3a, p3b).
    fps = {C.REFERENCE: 5.1, C.CLOSURE: 5.2,
           "A000": 5.2, "A100": 5.3, "A010": 25.0, "A001": 5.2,
           "A110": 25.4, "A101": 5.3, "A011": 25.2, "A111": 25.3}
    spans = {a: (0 if a in ("A010", "A110", "A011", "A111") else 4271) for a in fps}
    obs = _factorial_obs(fps, spans=spans)
    v, detail = C.verdict(obs)
    check("verdict attributes", v.startswith("ATTRIBUTED"), v)
    check("it names p3a", "p3a" in detail, detail[:150])
    me_p3a, n_p3a = C._main_effect(obs, "p3a", "100K")
    me_p2, _ = C._main_effect(obs, "p2", "100K")
    check("p3a has four isolation edges", n_p3a == 4, str(n_p3a))
    check("p3a's main effect is the large one", me_p3a > 4.0, f"{me_p3a:.2f}x")
    check("p2's main effect is flat", 0.9 < me_p2 < 1.1, f"{me_p2:.2f}x")
    g = dict((name, okv) for name, okv, _ in C.gates(obs))
    closure = [k for k in g if k.startswith("CLOSURE")][0]
    check("closure holds when Z matches main", g[closure] is True)
    p3a_gate = [k for k in g if k.startswith("p3a engaged")][0]
    check("p3a's engagement gate fires on highlight spans", g[p3a_gate] is True)


def test_redundancy_is_visible_not_hidden() -> None:
    """If p2 and p3a are each independently sufficient, a subtractive-only design would report
    nothing. The factorial must still show both as large in the contexts where they act."""
    print("\n[4] redundancy is visible rather than reported as 'nothing matters'")
    fast, slow = 25.0, 5.1
    fps = {C.REFERENCE: slow, C.CLOSURE: slow, "A000": slow}
    for m in ("100", "010", "001", "110", "101", "011", "111"):
        on_p2, on_p3a = m[0] == "1", m[1] == "1"
        fps["A" + m] = fast if (on_p2 or on_p3a) else slow
    obs = _factorial_obs(fps)
    me_p2, _ = C._main_effect(obs, "p2", "100K")
    me_p3a, _ = C._main_effect(obs, "p3a", "100K")
    check("p2 still reads above 1x despite being redundant with p3a", me_p2 > 1.3, f"{me_p2:.2f}x")
    check("p3a still reads above 1x too", me_p3a > 1.3, f"{me_p3a:.2f}x")
    e_p2 = C._edge_ratios(obs, "p2", "100K")
    alone = [e for e in e_p2 if e["off"] == "A000"][0]
    with_p3a = [e for e in e_p2 if e["off"] == "A010"][0]
    check("p2's edge is large where p3a is absent", alone["ratio"] > 4.0, f"{alone['ratio']:.2f}x")
    check("and flat where p3a already fired, which is the interaction being made visible",
          0.9 < with_p3a["ratio"] < 1.1, f"{with_p3a['ratio']:.2f}x")


def test_closure_failure_is_reported() -> None:
    """If Z does not match main, the split is of less than the whole and the verdict says so."""
    print("\n[5] a failed closure is reported rather than absorbed")
    fps = {C.REFERENCE: 5.1, C.CLOSURE: 15.0,
           "A000": 15.1, "A100": 15.2, "A010": 25.0, "A001": 15.1,
           "A110": 25.4, "A101": 15.3, "A011": 25.2, "A111": 25.3}
    obs = _factorial_obs(fps)
    g = dict((name, okv) for name, okv, _ in C.gates(obs))
    closure = [k for k in g if k.startswith("CLOSURE")][0]
    check("the closure gate fails", g[closure] is False)
    _, detail = C.verdict(obs)
    check("and the verdict says the shares are of less than the whole",
          "CLOSURE FAILED" in detail, detail[-160:])


def test_unsettled_fidelity_census_is_refused() -> None:
    print("\n[6] a fidelity census that never settled is not quoted as fidelity")
    runs = [run("JAM", "100K", "jam", n=7, elapsed=5406, busy=98.5, worst=1816)]
    for arm in (C.REFERENCE, C.WHOLE):
        runs.append(run(arm, "100K", "1", n=27, elapsed=5106, busy=92.7, worst=1779,
                        settled=(arm == C.REFERENCE)))
    obs = base_obs(runs, ["100K"])
    g = {name: (okv, ev) for name, okv, ev in C.gates(obs)}
    k = [x for x in g if x.startswith("the fidelity census")][0]
    check("the gate fails", g[k][0] is False)
    check("and it names the arm", C.WHOLE in g[k][1], g[k][1])


def test_never_quotes_p50() -> None:
    print("\n[7] fps_p50 is never the statistic")
    src = (HERE / "criteria" / "studio_p123_attribution.py").read_text()
    code = "\n".join(l for l in src.splitlines()
                     if not l.lstrip().startswith("#") and "fps_p50" not in l or "fps_p50" in l
                     and not l.lstrip().startswith("#"))
    uses = [l for l in src.splitlines()
            if "fps_p50" in l and not l.lstrip().startswith("#")
            and '"' not in l.split("fps_p50")[0][-2:]]
    # the only mentions must be inside prose, never as a dict lookup
    lookups = [l for l in src.splitlines() if 'get("fps_p50")' in l or '["fps_p50"]' in l]
    check("no code path reads fps_p50", not lookups, str(lookups[:2]))
    check("effective rate is computed from raf.n and elapsed_ms",
          "1000.0 * n / el" in src)
    del code, uses


def test_table_renders() -> None:
    print("\n[8] the report renders without exploding on a partial run")
    fps = {C.REFERENCE: 5.1, C.CLOSURE: 5.2, "A010": 25.0, "A111": 25.3}
    obs = _factorial_obs(fps)
    txt = C.table(obs)
    check("table is produced", len(txt) > 500, f"{len(txt)} chars")
    check("it names the published streaming-window ablation for comparison", "6,821 ms" in txt)
    check("it reports BOTH code thresholds", "4,096" in txt and "16,384" in txt)
    check("it says main's 20,000 is not a precedent", "not a precedent" in txt)
    gs = C.gates(obs)
    check("gates run on a partial arm set", len(gs) > 10, f"{len(gs)} gates")


def main() -> int:
    for fn in (test_r500k_control_is_per_rung, test_floor_downgrades_rather_than_scores,
               test_attributes_to_the_owning_mechanism, test_redundancy_is_visible_not_hidden,
               test_closure_failure_is_reported, test_unsettled_fidelity_census_is_refused,
               test_never_quotes_p50, test_table_renders):
        fn()
    print()
    if FAILURES:
        print(f"FAILED {len(FAILURES)}: {FAILURES}")
        return 1
    print("all p123 criteria self-tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
