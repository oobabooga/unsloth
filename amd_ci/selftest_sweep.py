#!/usr/bin/env python3
"""Self-test for studio_sweep_probe's flag machinery, against the REAL source of all eleven arms.

The sweep's whole claim is that pagination is live at every point. That claim is produced by
`prepare_flags` + `read_effective_pagination` running over eleven trees which carry the two flag
declarations at three different line numbers, in two different shapes, with different neighbours.
A static patch file could not do this, and the regex that replaces it is exactly the kind of thing
that silently matches nothing on six of eleven trees and reports success.

So it is exercised here, offline, against the actual bytes of each commit, before any GPU is held.
Run it from a checkout that can resolve the PR commits, or point --repo at one:

    python3 amd_ci/selftest_sweep.py --repo /path/to/a/clone/with/pr9477

Exit code 0 means every arm resolves to pagination=true and the grid-collapse column reads what
the sweep says it reads. Anything else means the sweep must not be launched.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "probes"))

from studio_sweep_probe import (  # noqa: E402
    ARM_GRID, ARM_ORDER, ARM_REF, FLAGS_REL, REASONING_REL, PREDICTED_NULLS,
    prepare_flags, read_bool_const, GRID_CONST,
)

#: What each arm MUST come out as. Written here independently of the probe's own tables, from the
#: source read at each commit, so that a change to the probe that breaks the sweep's premise fails
#: this file rather than passing because both sides moved together.
EXPECT = {
    #          pagination_effective, grid_collapse after prepare, flag const exists in tree
    "p0":  ("true", "false", False),
    "p0g": ("true", "true",  False),
    "p1":  ("true", "true",  False),
    "p2":  ("true", "true",  False),
    "p3":  ("true", "true",  False),
    "p4":  ("true", "true",  True),
    "p5":  ("true", "true",  True),
    "p6":  ("true", "true",  True),
    "p7":  ("true", "true",  True),
    "p8":  ("true", "true",  True),
    "p9":  ("true", "true",  True),
}


sys.path.insert(0, str(HERE / "criteria"))
import importlib.util as _ilu  # noqa: E402

_spec = _ilu.spec_from_file_location("sweep_crit", HERE / "criteria" / "studio_sweep_peak.py")
CRIT = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(CRIT)


def git(repo: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True, timeout=300)


# ── the decision logic, exercised on synthetic curves ────────────────────────────────────────────
#
# The verdict is the deliverable and it is a comparison of eleven noisy numbers against a floor
# derived from two of them. Every branch of it is cheap to exercise offline and expensive to get
# wrong in CI, where it would be discovered only after the exclusive GPU had already been held.

def _obs(fps: dict, bundles: dict | None = None, reps: int = 3) -> dict:
    """A minimal observations dict the criteria can read. `fps` maps arm -> list of per-rep fps."""
    bundles = bundles or {}
    default_b = {"p6": "H67", "p7": "H67", "p8": "H89", "p9": "H89"}
    states, runs = {}, []
    for arm in CRIT.ARMS:
        bh = bundles.get(arm, default_b.get(arm, f"B_{arm}"))
        states[arm] = {
            "commit": f"{arm}0000000", "what": arm, "checkout_ok": True,
            "pagination_effective": "true",
            "grid_collapse": "false" if arm == "p0" else "true",
            "exported_bundle_hash": bh, "bundle_hash_at_measure": bh,
            "bundle_hash_matches_build": True,
            "frontend_tree_hash": {"p6": "T67", "p7": "T67", "p8": "T89",
                                   "p9": "T89"}.get(arm, f"T_{arm}"),
            "dist": {"index_html": True, "asset_files": 600},
        }
    states["p0"]["exported_bundle_hash"] = "B_p0"
    states["p0g"]["exported_bundle_hash"] = "B_p0g"
    for arm, series in list(fps.items()):
        for i, v in enumerate(series, 1):
            runs.append({"arm": arm, "rung": CRIT.RUNG, "rep": str(i), "payload": {
                "ok": True,
                "run_meta": {"instrument_pacer_file": "/w/instrument/tests/studio/studiobench/pacer.py",
                             "instrument_hash": "IH", "corpus_hash": "23cd24646603bc54",
                             "bundle_hash": states.get(arm, {}).get("exported_bundle_hash")},
                "actions": [{"name": CRIT.ACTION, "elapsed_ms": 5000,
                             "raf": {"n": int(round(v * 5.0)), "max_ms": 300},
                             "busy": {"busy_pct": 30.0},
                             "census_open": {"highlight_spans": 7259,
                                             "reasoning_chars": 45747}}]}})
    return {"xserver": {"display": ":99"}, "instrument": {"path": "/w/instrument"},
            "states": states, "runs": runs, "arms_built": list(CRIT.ARMS),
            "rungs": [CRIT.RUNG], "reps": {CRIT.RUNG: reps}}


def _flat(v: float = 40.0) -> dict:
    out = {"JAM": [5.0, 5.1, 5.0]}
    for a in CRIT.ARMS:
        out[a] = [v, v + 0.1, v - 0.1]
    return out


def check_decision_logic() -> list[str]:
    bad: list[str] = []

    def run(label, fps, want, bundles=None, expect_in=None):
        obs = _obs(fps, bundles=bundles)
        try:
            gs = list(CRIT.gates(obs))
            v, why = CRIT.verdict(obs)
            CRIT.table(obs)
        except Exception as e:                                            # noqa: BLE001
            bad.append(f"{label}: criteria raised {type(e).__name__}: {e}")
            return
        failed = [n for n, ok, _ in gs if not ok]
        if want == "GATE_FAIL":
            if not failed:
                bad.append(f"{label}: expected a gate to fail, all {len(gs)} passed")
            else:
                print(f"  {label}: gate failed as expected -> {failed[0][:60]}")
            return
        if failed:
            bad.append(f"{label}: expected clean gates, these failed: {failed}")
        if v != want:
            bad.append(f"{label}: verdict {v!r}, expected {want!r} ({why[:160]})")
        elif expect_in and expect_in not in why:
            bad.append(f"{label}: verdict {v} but {expect_in!r} not named in: {why[:200]}")
        else:
            print(f"  {label}: {v}")

    # A flat curve is the expected result and must NOT be called a peak.
    run("flat curve", _flat(), "NO_PEAK")

    # A real peak: p4 stands 8 fps above a 40 fps curve, far beyond any spread here.
    f = _flat(); f["p4"] = [48.0, 48.1, 47.9]
    run("true peak at p4", f, "PEAK", expect_in="p4")

    # A peak smaller than the null floor must NOT be called. The nulls are made to disagree by
    # 2 fps, so a 1 fps bump is inside what the channel manufactures.
    f = _flat(); f["p7"] = [42.0, 42.1, 41.9]; f["p4"] = [41.0, 41.1, 40.9]
    run("bump below the null floor", f, "NO_PEAK")

    # One arm disagreeing with ITSELF is the arm-A failure mode: a mean 8 fps high, built out of
    # one good rep and one bad one. It must not be called a peak.
    f = _flat(); f["p4"] = [56.0, 24.0, 48.0]
    run("one arm disagreeing with itself", f, "NO_PEAK")

    # The tip being beaten by the ANCHOR is not a peak in the middle: p0/p0g are outside the span.
    f = _flat(); f["p0"] = [60.0, 60.1, 59.9]; f["p0g"] = [60.0, 60.1, 59.9]
    run("anchor above the tip is not a peak", f, "NO_PEAK")

    # If the two commits that cannot change a pixel built DIFFERENT bundles, the build is not
    # reproducible: the gate must catch it before any verdict is reached.
    run("free null built two bundles", _flat(), "GATE_FAIL",
        bundles={"p6": "H6", "p7": "H7"})

    # A blind frame channel: the jam control does not fall. No point on the curve means anything.
    f = _flat(); f["JAM"] = [39.5, 40.0, 40.2]
    run("jam control did not fall", f, "GATE_FAIL")

    # Pagination off at one point means that point is not on the curve being asked about.
    obs = _obs(_flat())
    obs["states"]["p5"]["pagination_effective"] = "false"
    gs = list(CRIT.gates(obs))
    if not any(not ok for n, ok, _ in gs if "pagination is live" in n):
        bad.append("a point with pagination off did not fail the pagination gate")
    else:
        print("  pagination off at one point: gate failed as expected")
    return bad


def materialise(repo: str, commit: str, root: Path) -> list[str]:
    """Write just the two flag-bearing files of `commit` into `root`, at their real paths."""
    got = []
    for rel in (FLAGS_REL, REASONING_REL):
        r = git(repo, "show", f"{commit}:{rel.as_posix()}")
        if r.returncode != 0:
            continue
        dest = root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(r.stdout, encoding="utf-8")
        got.append(rel.as_posix())
    return got


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=str(HERE.parent),
                    help="a clone that can resolve the PR 9477 commits")
    args = ap.parse_args()

    # Every arm's commit must resolve here, or the test is vacuous rather than passing.
    unresolved = []
    for arm in ARM_ORDER:
        r = git(args.repo, "rev-parse", "--verify", "--quiet", f"{ARM_REF[arm]}^{{commit}}")
        if r.returncode != 0 or not r.stdout.strip():
            unresolved.append((arm, ARM_REF[arm][:9]))
    if unresolved:
        print(f"FAIL: {args.repo} cannot resolve {unresolved}. Fetch refs/pull/9477/head first; "
              f"a self-test that skips the arms it cannot see would pass for the wrong reason.")
        return 2

    bad = []
    print(f"{'arm':4} {'commit':10} {'pagination':11} {'grid':6} {'flag?':6} how")
    for arm in ARM_ORDER:
        commit = ARM_REF[arm]
        with tempfile.TemporaryDirectory(dir=str(HERE.parent / "amd_ci")) as td:
            root = Path(td)
            files = materialise(args.repo, commit, root)
            if REASONING_REL.as_posix() not in files:
                bad.append(f"{arm}: {REASONING_REL} missing at {commit[:9]}")
                continue
            had_flag = read_bool_const(root / FLAGS_REL, "REASONING_PAGINATION_ENABLED") is not None
            res = prepare_flags(root, arm)
            pag, grid = res["pagination_effective"], res["grid_collapse"]
            want_pag, want_grid, want_flag = EXPECT[arm]
            print(f"{arm:4} {commit[:9]:10} {str(pag):11} {str(grid):6} {str(had_flag):6} "
                  f"{res['pagination_how']}")
            if pag != want_pag:
                bad.append(f"{arm}: pagination is {pag!r}, expected {want_pag!r} "
                           f"({res['pagination_how']})")
            if grid != want_grid:
                bad.append(f"{arm}: {GRID_CONST} is {grid!r}, expected {want_grid!r}")
            if had_flag != want_flag:
                bad.append(f"{arm}: pagination flag present={had_flag}, expected {want_flag}")
            # An edit that reports itself written must have matched EXACTLY one declaration.
            for e in res["edits"]:
                if e["existed"] and e["matched"] != 1:
                    bad.append(f"{arm}: edit of {e['const']} matched {e['matched']} declarations "
                               f"in {e['file']}; 0 changes nothing and >1 is ambiguous")

    # The free nulls are claims about the SOURCE and are checkable here, with no CI at all.
    print()
    for (a, b) in PREDICTED_NULLS:
        ha = git(args.repo, "rev-parse", f"{ARM_REF[a]}:studio/frontend").stdout.strip()
        hb = git(args.repo, "rev-parse", f"{ARM_REF[b]}:studio/frontend").stdout.strip()
        ok = bool(ha) and ha == hb
        print(f"predicted null {a} vs {b}: studio/frontend tree "
              f"{ha[:12] or '?'} vs {hb[:12] or '?'} -> {'IDENTICAL' if ok else 'DIFFERENT'}")
        if not ok:
            bad.append(f"predicted null {a} vs {b} is not a null: frontend trees differ. The "
                       f"sweep's noise estimate rests on this, so it must not be assumed.")

    # p0 and p0g are the same commit; the sweep needs them to differ only by the grid literal.
    h0 = git(args.repo, "rev-parse", f"{ARM_REF['p0']}:studio/frontend").stdout.strip()
    h0g = git(args.repo, "rev-parse", f"{ARM_REF['p0g']}:studio/frontend").stdout.strip()
    print(f"p0 and p0g are the same commit: {ARM_REF['p0'] == ARM_REF['p0g']}, "
          f"same source tree: {h0 == h0g}, grid forced on p0g: {ARM_GRID['p0g']}")
    if ARM_REF["p0"] != ARM_REF["p0g"] or ARM_GRID["p0g"] is not True:
        bad.append("p0/p0g are no longer the same commit with grid forced true on p0g")

    print()
    print("decision logic, on synthetic curves:")
    bad += check_decision_logic()

    print()
    if bad:
        print(f"FAIL ({len(bad)}):")
        for b in bad:
            print(f"  - {b}")
        return 1
    print(f"OK: all {len(ARM_ORDER)} arms resolve to pagination=true, the grid column reads as "
          f"declared, and both predicted nulls are identical source trees.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
