#!/usr/bin/env python3
"""Self-test for criteria/studio_arms_ladder.py, against payloads whose right answers are known.

A criteria module that can only produce ONE answer is the failure mode that has cost this campaign
the most: it looks like a passing test and it is a green-tick generator. So every branch is
exercised here from both directions, including every way THIS measurement can be wrong:

  * a jam control that did not resolve, on ONE session out of twelve
  * a base arm that did not collapse
  * an engine that fails the probe the containment gate keys on
  * a head arm that did not boot into the shipped state, and one that was forced
  * a pre arm carrying the head arm's own markers
  * head worse than pre, with the repetitions agreeing and with them overlapping
  * two arms that somehow report ONE bundle hash
  * two arms on two different unsloth-zoo commits
  * a rung measured once instead of twice

Run: python3 amd_ci/selftest_arms.py
"""

from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "arms_criteria", HERE / "criteria" / "studio_arms_ladder.py")
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)

FAILURES: list[str] = []


def check(name: str, got, want) -> None:
    if got != want:
        FAILURES.append(f"{name}: got {got!r}, want {want!r}")
        print(f"  FAIL {name}: got {got!r}, want {want!r}")
    else:
        print(f"  ok   {name}: {got!r}")


def gap_series(fps: float, seconds: float) -> list[float]:
    n = max(1, int(round(fps * seconds)))
    return [1000.0 / fps] * n


def window(phase: str, fps: float, busy: float, seconds: float = 6.0,
           worst_ms: float = 18.0, over100: float = 0.0, blocked_per_frame: float = 1.0,
           **extra) -> dict:
    frames = max(1, int(round(fps * seconds)))
    w = {
        "phase": phase,
        "elapsed_ms": int(seconds * 1000),
        "frames": frames,
        "eff_fps": round(fps, 1),
        "blocked_ms_per_frame": blocked_per_frame,
        "robust": {"blocked_ms": int(blocked_per_frame * frames), "frames": frames,
                   "blocked_ms_per_frame": blocked_per_frame,
                   "stall_frames_over_1s": 0, "worst_tick_ms": worst_ms,
                   "worst_gap_ms": worst_ms},
        "raf": {"n": frames, "p50_ms": 1000.0 / fps, "max_ms": worst_ms,
                "fps_p50": fps, "frames_over_100": 0, "frames_over_100_pct": over100},
        "raf_gaps_ms": gap_series(fps, seconds),
        "busy": {"busy_pct": busy, "blocked_ms": busy * seconds * 10, "ticks": 600,
                 "busy_pct_reason": None},
        "census": {"elements": 100},
    }
    w.update(extra)
    return w


def readback(arm: str, *, anchor = True, attr = None, shells = None, forced = None,
             spans = 300) -> dict:
    if attr is None:
        attr = "on" if arm == "head" else None
    if shells is None:
        shells = 40 if arm == "head" else 0
    return {
        "when": "final",
        "ua": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/605.1.15 Version/60.5",
        "css_supports_anchor_name": anchor,
        "css_supports_content_visibility": True,
        "engine_gate_probe": "anchor-name: --unsloth-probe",
        "math_block_attribute": attr,
        "math_runtime_global": forced,
        "aui_math_block": 595,
        "aui_math_display": 117,
        "katex_roots": 1027,
        "katex_display_roots": 117,
        "content_visibility_math_blocks": {"selector": ".aui-math-block", "matched": 595,
                                           "sampled": 60,
                                           "auto": 60 if arm == "head" else 0,
                                           "values": {}},
        "content_visibility_katex_display": {"selector": ".katex-display", "matched": 117,
                                             "sampled": 60, "auto": 0, "values": {}},
        "fence_runtime_global": None,
        "deferred_fence_shells": shells,
        "code_blocks": 51,
        "highlight_spans": spans,
    }


# The frame rates the fixture asserts against. `pre` is the reported collapse; `head` is a build
# that recovers the scroll and leaves idle alone.
FPS = {
    ("pre", "0K", "idle"): 61.0, ("pre", "0K", "scroll"): 61.0, ("pre", "0K", "stream"): 60.0,
    ("pre", "100K", "idle"): 61.0, ("pre", "100K", "scroll"): 26.0, ("pre", "100K", "stream"): 59.0,
    ("pre", "500K", "idle"): 60.0, ("pre", "500K", "scroll"): 3.2, ("pre", "500K", "stream"): 55.0,
    ("head", "0K", "idle"): 61.0, ("head", "0K", "scroll"): 61.0, ("head", "0K", "stream"): 60.0,
    ("head", "100K", "idle"): 61.0, ("head", "100K", "scroll"): 62.0,
    ("head", "100K", "stream"): 60.0,
    ("head", "500K", "idle"): 62.0, ("head", "500K", "scroll"): 38.0,
    ("head", "500K", "stream"): 58.0,
}
BUSY = {"idle": 2.0, "scroll": 90.0, "stream": 30.0}
ELEMENTS = {"0K": 900, "100K": 21000, "500K": 60000}


def session(arm: str, rung: str, rep: int, *, jam_drop = 0.72, rb = None,
            fps_override = None, ok = True) -> dict:
    fps = dict(FPS)
    if fps_override:
        fps.update(fps_override)
    idle_fps = fps[(arm, rung, "idle")]
    phases = [window("idle", idle_fps, BUSY["idle"])]
    phases.append(window("idle_jammed", round(idle_fps * (1 - jam_drop), 1), 82.0, jammed = True))
    for ph in ("scroll", "stream"):
        busy = BUSY[ph] if rung != "0K" else 5.0
        phases.append(window(ph, fps[(arm, rung, ph)], busy,
                             seconds = 20.0,
                             worst_ms = 3092.0 if fps[(arm, rung, ph)] < 10 else 60.0,
                             over100 = 80.0 if fps[(arm, rung, ph)] < 10 else 1.0,
                             blocked_per_frame = 287.0 if fps[(arm, rung, ph)] < 10 else 15.0))
    phases.append(window("recover", 61.0, 3.0))
    payload = {
        "ok": ok,
        "ua": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/605.1.15 Version/60.5",
        "rung": rung, "arm": arm,
        "engine_probe": {"is_webkit_gtk_ua": True, "vendor": "Apple Computer, Inc.",
                         "has_chrome": False, "has_webkit_message_handlers": True},
        "mount": {"ms": 4000, "by": "last_seeded_marker",
                  "census": {"elements": ELEMENTS[rung]}},
        "clamp": {"clamp_ms": 4.1, "reason": None},
        "liveness": {"clean_fps": idle_fps,
                     "jammed_fps": round(idle_fps * (1 - jam_drop), 1),
                     "drop_fraction": jam_drop,
                     "clean_fps_p50": 61.0, "jammed_fps_p50": 60.0,
                     "hog_ms": 200, "hog_period_ms": 250},
        "readback_final": rb if rb is not None else readback(arm),
        "actions": [
            {"name": "reasoning_toggle", "ok": True, "not_applicable": rung == "0K",
             "eff_fps": None if rung == "0K" else (9.0 if arm == "pre" else 45.0),
             "app_sync_ms": 890.0 if arm == "pre" else 120.0,
             "busy": {"busy_pct": 93.0 if arm == "pre" else 24.0},
             "raf": {"max_ms": 890.0 if arm == "pre" else 136.0}},
            {"name": "select_all_copy", "ok": True, "not_applicable": rung == "0K",
             "eff_fps": None if rung == "0K" else (12.0 if arm == "pre" else 40.0),
             "app_sync_ms": 300.0, "busy": {"busy_pct": 80.0},
             "raf": {"max_ms": 200.0}},
        ],
        "phases": phases,
        "final": {"elements": ELEMENTS[rung]},
        "run_meta": {"rung": rung, "rep": str(rep), "arm": arm,
                     "bundle_hash": f"{arm}hash(12 files)",
                     "instrument_sb_root": "/w/arms/repo_head",
                     "corpus_hash": "23cd2464"},
    }
    return {"rung": rung, "rep": str(rep), "arm": arm, "port": 5541,
            "expected_bundle_hash": f"{arm}hash(12 files)", "rc": 0,
            "payload": payload}


def fixture() -> dict:
    rungs = ["0K", "100K", "500K"]
    runs, plan = [], []
    for rep in (1, 2):
        order = ["pre", "head"] if rep % 2 == 1 else ["head", "pre"]
        for rung in rungs:
            for arm in order:
                runs.append(session(arm, rung, rep))
                plan.append({"rung": rung, "rep": str(rep), "arm": arm})
    return {
        "state": "host", "rungs_requested": "0K,100K,500K", "reps": 2,
        "refs": {"pre": "c87fe20e3", "head": "main"},
        "xserver": {"display": ":99"},
        "zoo_main_at_start": {"sha": "z" * 40},
        "arms": {
            "pre": {"arm": "pre", "ref": "c87fe20e3",
                    "clone": {"commit": "c87fe20e3" + "0" * 31,
                              "commit_line": "c87fe20e3 2026-08-16 pre", "has_studiobench": False},
                    "install": {"rc": 0, "seconds": 900.0},
                    "unsloth_bin": "/w/pre/bin/unsloth", "repo": "/w/arms/repo_pre",
                    "home": "/w/arms/studio_home_pre",
                    "dist": {"exists": True, "index_html": True, "asset_files": 20,
                             "bundle_hash": "prehash(12 files)"},
                    "zoo": {"commit_id": "z" * 40, "version": "2026.8.20"}},
            "head": {"arm": "head", "ref": "main",
                     "clone": {"commit": "40b4702cd" + "0" * 31,
                               "commit_line": "40b4702cd 2026-08-26 head", "has_studiobench": True},
                     "install": {"rc": 0, "seconds": 900.0},
                     "unsloth_bin": "/w/head/bin/unsloth", "repo": "/w/arms/repo_head",
                     "home": "/w/arms/studio_home_head",
                     "dist": {"exists": True, "index_html": True, "asset_files": 20,
                              "bundle_hash": "headhash(12 files)"},
                     "zoo": {"commit_id": "z" * 40, "version": "2026.8.20"}},
        },
        "instrument": {"sb_root": "/w/arms/repo_head", "from_arm": "head", "exists": True,
                       "pre_has_studiobench": False},
        "plan": plan,
        "runs": runs,
    }


def gate_ok(obs: dict, needle: str) -> bool:
    for name, ok, _ev in C.gates(obs):
        if needle in name:
            return ok
    raise AssertionError(f"no gate matching {needle!r}")


def all_gates_ok(obs: dict) -> bool:
    return all(ok for _n, ok, _e in C.gates(obs))


def main() -> int:
    print("== the healthy fixture: every gate passes and the collapse is confirmed")
    base = fixture()
    check("all gates pass", all_gates_ok(base), True)
    check("verdict", C.verdict(base)[0], "CONFIRMED")
    tbl = C.table(base)
    check("table names the top rung", "500K" in tbl, True)
    check("table shows the jam control", "jammed positive control" in tbl, True)
    check("table shows the readback", "read back out of the running page" in tbl, True)
    check("table bounds its own reach", "did NOT measure" in tbl, True)
    check("no regression claimed", "No cell has `head` worse" in tbl, True)

    print("== ONE session's jam control does not resolve: the whole run is VOID")
    o = fixture()
    o["runs"][7]["payload"]["liveness"]["drop_fraction"] = 0.02
    o["runs"][7]["payload"]["liveness"]["jammed_fps"] = 60.4
    v, why = C.verdict(o)
    check("verdict", v, "VOID")
    check("why names the control", "jammed positive control" in why, True)

    print("== a jam control that is missing entirely is not a pass")
    o = fixture()
    o["runs"][3]["payload"]["liveness"] = {}
    check("verdict", C.verdict(o)[0], "VOID")

    print("== the base does not collapse: VOID, and it is not a NO_BENEFIT")
    o = fixture()
    for r in o["runs"]:
        if r["arm"] == "pre" and r["rung"] == "500K":
            for ph in r["payload"]["phases"]:
                if ph["phase"] == "scroll":
                    ph["eff_fps"] = 59.0
                    ph["busy"]["busy_pct"] = 8.0
    v, why = C.verdict(o)
    check("verdict", v, "VOID")
    check("why names the base", "did NOT exhibit" in why, True)

    print("== a low base fps at LOW busy is not a collapse either")
    o = fixture()
    for r in o["runs"]:
        if r["arm"] == "pre" and r["rung"] == "500K":
            for ph in r["payload"]["phases"]:
                if ph["phase"] == "scroll":
                    ph["busy"]["busy_pct"] = 3.0
    check("verdict", C.verdict(o)[0], "VOID")

    print("== the engine fails the containment gate: said loudly, with its own verdict")
    o = fixture()
    for r in o["runs"]:
        r["payload"]["readback_final"]["css_supports_anchor_name"] = False
        if r["arm"] == "head":
            r["payload"]["readback_final"]["math_block_attribute"] = None
    v, why = C.verdict(o)
    check("verdict", v, "FIX_NOT_ENGAGED_ON_THIS_ENGINE")
    check("why says read no conclusion", "Read no conclusion" in why, True)

    print("== head did not boot into the shipped state: VOID")
    o = fixture()
    for r in o["runs"]:
        if r["arm"] == "head":
            r["payload"]["readback_final"]["math_block_attribute"] = None
    check("verdict", C.verdict(o)[0], "VOID")

    print("== head was FORCED by a runtime override: VOID, because that is not what ships")
    o = fixture()
    for r in o["runs"]:
        if r["arm"] == "head":
            r["payload"]["readback_final"]["math_runtime_global"] = "contain"
    v, why = C.verdict(o)
    check("verdict", v, "VOID")
    check("why names the override", "forced" in why, True)

    print("== the pre arm carries head's markers: VOID, the arms are not the two states")
    o = fixture()
    for r in o["runs"]:
        if r["arm"] == "pre":
            r["payload"]["readback_final"]["math_block_attribute"] = "on"
    check("verdict", C.verdict(o)[0], "VOID")

    print("== head recovers the scroll but not to the bar: FIX_INCOMPLETE")
    o = fixture()
    for r in o["runs"]:
        if r["arm"] == "head" and r["rung"] == "500K":
            for ph in r["payload"]["phases"]:
                if ph["phase"] == "scroll":
                    ph["eff_fps"] = 8.0
    check("verdict", C.verdict(o)[0], "FIX_INCOMPLETE")

    print("== head does not recover it at all: NO_BENEFIT")
    o = fixture()
    for r in o["runs"]:
        if r["arm"] == "head" and r["rung"] == "500K":
            for ph in r["payload"]["phases"]:
                if ph["phase"] == "scroll":
                    ph["eff_fps"] = 3.3
    check("verdict", C.verdict(o)[0], "NO_BENEFIT")

    print("== a REGRESSION is found, named, and reported first")
    o = fixture()
    for r in o["runs"]:
        if r["arm"] == "head" and r["rung"] == "100K":
            for ph in r["payload"]["phases"]:
                if ph["phase"] == "stream":
                    ph["eff_fps"] = 20.0
    regs = C._regressions(o)
    check("one settled regression", len([x for x in regs if "WORSE" in x and "OVERLAP" not in x]),
          1)
    check("it names the rung and phase", any("100K stream" in x for x in regs), True)
    tbl = C.table(o)
    check("regressions are the first section", tbl.startswith("### Regressions first"), True)
    check("verdict still reports the win", C.verdict(o)[0], "CONFIRMED_WITH_REGRESSION")
    check("verdict discloses the regression", "WORSE" in C.verdict(o)[1], True)

    # The cell below is run 33040070879's 100K scroll, verbatim: 58.5, 57.6 -> 54.5, 53.8 fps with
    # the worst frame 49 -> 440 ms in both repetitions. Under one 20% bar on the mean this printed
    # "no cell has head worse than pre" while carrying a nine times longer hitch.
    def _amd_100k_scroll(o: dict) -> None:
        vals = {("pre", "1"): (58.5, 49.0), ("pre", "2"): (57.6, 48.0),
                ("head", "1"): (54.5, 440.0), ("head", "2"): (53.8, 436.0)}
        for r in o["runs"]:
            if r["rung"] != "100K":
                continue
            fps, worst = vals[(r["arm"], r["rep"])]
            for ph in r["payload"]["phases"]:
                if ph["phase"] == "scroll":
                    ph["eff_fps"] = fps
                    ph["raf"]["max_ms"] = worst

    print("== a 9x WORSE WORST FRAME cannot pass because the mean barely moved")
    o = fixture()
    _amd_100k_scroll(o)
    regs = C._regressions(o)
    settled = [x for x in regs if "WORSE" in x and "OVERLAP" not in x]
    check("the cell is reported", any("100K scroll" in x for x in settled), True)
    check("the tail is named", any("in the tail" in x for x in settled), True)
    check("with the ratio", any("9.0x" in x for x in settled), True)
    check("the mean is named too", any("WORSE on the mean" in x for x in settled), True)
    check("the table does not claim the run is clean",
          "No cell has `head` worse" in C.table(o), False)
    check("the verdict carries it", C.verdict(o)[0], "CONFIRMED_WITH_REGRESSION")

    print("== the tail bar alone is enough: a flat mean with a hitch is still a regression")
    o = fixture()
    _amd_100k_scroll(o)
    for r in o["runs"]:
        if r["rung"] == "100K":
            for ph in r["payload"]["phases"]:
                if ph["phase"] == "scroll":
                    ph["eff_fps"] = 58.0          # identical on both arms: the mean bar cannot fire
    regs = [x for x in C._regressions(o) if "WORSE" in x and "OVERLAP" not in x]
    check("still reported", any("100K scroll" in x for x in regs), True)
    check("on the tail only", all("on the mean" not in x for x in regs if "100K scroll" in x),
          True)

    print("== the tail bar does NOT fire on a small absolute frame")
    o = fixture()
    for r in o["runs"]:
        if r["rung"] == "100K":
            for ph in r["payload"]["phases"]:
                if ph["phase"] == "scroll":
                    # 8 ms -> 24 ms is 3x and is not a hitch anybody can see.
                    ph["raf"]["max_ms"] = 24.0 if r["arm"] == "head" else 8.0
    check("nothing reported", [x for x in C._regressions(o) if "100K scroll" in x], [])

    print("== the tail bar does NOT fire when the repetitions overlap")
    o = fixture()
    _amd_100k_scroll(o)
    for r in o["runs"]:
        if r["rung"] == "100K" and r["arm"] == "head" and r["rep"] == "2":
            for ph in r["payload"]["phases"]:
                if ph["phase"] == "scroll":
                    ph["eff_fps"] = 60.0          # one rep better, one worse
                    ph["raf"]["max_ms"] = 40.0    # below pre's own worst
    regs = C._regressions(o)
    check("nothing settled at that cell",
          [x for x in regs if "100K scroll" in x and "OVERLAP" not in x], [])
    check("but it is still disclosed", any("100K scroll" in x and "OVERLAP" in x for x in regs),
          True)

    print("== an action window's tail is judged the same way")
    o = fixture()
    for r in o["runs"]:
        if r["rung"] == "100K":
            for a in r["payload"]["actions"]:
                if a["name"] == "select_all_copy" and r["arm"] == "head":
                    a["raf"]["max_ms"] = 900.0
    regs = [x for x in C._regressions(o) if "OVERLAP" not in x]
    check("the action cell is reported",
          any("action:select_all_copy" in x and "in the tail" in x for x in regs), True)

    print("== overlapping repetitions are NOT reported as a settled regression")
    o = fixture()
    seen = 0
    for r in o["runs"]:
        if r["arm"] == "head" and r["rung"] == "100K":
            seen += 1
            for ph in r["payload"]["phases"]:
                if ph["phase"] == "stream":
                    # rep 1 far below pre, rep 2 above it: the direction is not settled
                    ph["eff_fps"] = 20.0 if seen == 1 else 70.0
    regs = C._regressions(o)
    check("nothing settled", [x for x in regs if "WORSE" in x and "OVERLAP" not in x], [])
    check("but it is still disclosed", any("OVERLAP" in x for x in regs), True)

    print("== gates: two arms reporting ONE bundle hash")
    o = fixture()
    o["arms"]["head"]["dist"]["bundle_hash"] = "prehash(12 files)"
    check("bundle gate fires", gate_ok(o, "a production bundle per arm"), False)

    print("== gates: two different unsloth-zoo commits")
    o = fixture()
    o["arms"]["head"]["zoo"]["commit_id"] = "y" * 40
    check("zoo gate fires", gate_ok(o, "one unsloth-zoo"), False)
    check("and it passes when they match", gate_ok(fixture(), "one unsloth-zoo"), True)

    print("== gates: two instruments")
    o = fixture()
    o["runs"][0]["payload"]["run_meta"]["instrument_sb_root"] = "/w/arms/repo_pre"
    check("instrument gate fires", gate_ok(o, "one pinned instrument"), False)

    print("== gates: two corpora")
    o = fixture()
    o["runs"][0]["payload"]["run_meta"]["corpus_hash"] = "ac9d5d8e"
    check("corpus is covered by the same gate", gate_ok(o, "one pinned instrument"), False)

    print("== gates: a rung measured once")
    o = fixture()
    o["runs"] = [r for r in o["runs"] if not (r["arm"] == "head" and r["rung"] == "500K"
                                              and r["rep"] == "2")]
    check("repetition gate fires", gate_ok(o, "at least twice"), False)

    print("== gates: a session that did not complete")
    o = fixture()
    o["runs"][2]["payload"]["ok"] = False
    o["runs"][2]["payload"]["error"] = "timeout waiting for stream to start"
    check("completion gate fires", gate_ok(o, "every planned session completed"), False)

    print("== gates: a bundle that drifted mid-run")
    o = fixture()
    o["runs"][4]["payload"]["run_meta"]["bundle_hash"] = "otherhash(12 files)"
    check("bundle drift gate fires", gate_ok(o, "one bundle per arm"), False)

    print("== gates: the thread did not really mount")
    o = fixture()
    for r in o["runs"]:
        if r["rung"] == "500K":
            r["payload"]["final"]["elements"] = 1000
    check("mount gate fires", gate_ok(o, "really mounted"), False)

    print("== gates: a SETTLED census is what counts, not the first commit")
    # `head` carries progressive message mounting and `pre` does not, so head's census AT MOUNT is
    # taken before the thread has converged by design. Gating on that would fail the head arm for
    # doing the thing it was changed to do. The gate reads the settled census instead.
    o = fixture()
    for r in o["runs"]:
        if r["arm"] == "head" and r["rung"] in ("100K", "500K"):
            r["payload"]["mount"]["census"]["elements"] = 2000
    check("a small mount census does not fire the gate", gate_ok(o, "really mounted"), True)

    print("== gates: not WebKitGTK")
    o = fixture()
    o["runs"][1]["payload"]["engine_probe"]["has_chrome"] = True
    check("engine gate fires", gate_ok(o, "really was WebKitGTK"), False)

    print("== gates: the arm order was never rotated")
    o = fixture()
    for r in o["runs"]:
        r["rep"] = "1"
    check("rotation gate fires", gate_ok(o, "arm order rotated"), False)

    print("== gates: the pre arm is not the commit that was asked for")
    o = fixture()
    o["arms"]["pre"]["clone"]["commit"] = "deadbeef" + "0" * 32
    check("commit gate fires", gate_ok(o, "two different commits"), False)

    print("== gates: one install failed")
    o = fixture()
    o["arms"]["pre"]["install"]["rc"] = 1
    check("install gate fires", gate_ok(o, "both arms installed"), False)

    print("== AN EMPTY OBSERVATIONS DICT IS A NO ON EVERY GATE, and crashes nothing")
    # Observed for real: a mistyped probe flag exited with rc=2, `capability_run.py` recorded
    # `_missing_output`, and the gates reported on nothing. One of them said yes.
    empty = {"_state": "host", "_probe_rc": 2, "_missing_output": True}
    check("no gate passes vacuously", any(ok for _n, ok, _e in C.gates(empty)), False)
    check("the probe gate names the reason", gate_ok(empty, "produced observations"), False)
    check("the mount gate does not pass on an empty census", gate_ok(empty, "really mounted"),
          False)
    check("table does not raise", isinstance(C.table(empty), str), True)
    check("verdict does not raise", isinstance(C.verdict(empty)[0], str), True)

    print("== a probe that ran but recorded nothing is also a NO")
    hollow = {"_state": "host", "_probe_rc": 0, "arms": {}, "runs": [], "plan": []}
    check("no gate passes vacuously", any(ok for _n, ok, _e in C.gates(hollow)), False)
    check("table does not raise", isinstance(C.table(hollow), str), True)

    print("== a fatal from the probe is surfaced rather than scored")
    fatal = dict(fixture())
    fatal["fatal"] = "Studio did not install for pre"
    check("the probe gate fires", gate_ok(fatal, "produced observations"), False)

    print("== A CLAMP THAT DID NOT CALIBRATE IS AN INSTRUMENT FAILURE, NOT A NON-REPRODUCTION")
    # busy_pct is null whenever the setTimeout clamp could not be established, and the observed
    # clamp on this venue is 8 ms against a 10 ms ceiling. With the two conditions collapsed into
    # one, a pre arm reading 3.2 fps -- the collapse, verbatim -- was reported as "the pre arm did
    # NOT exhibit the reported collapse". Saying the wrong thing about the subject is worse than
    # saying nothing.
    o = fixture()
    for r in o["runs"]:
        for ph in r["payload"]["phases"]:
            ph["busy"] = {"busy_pct": None, "blocked_ms": None,
                          "busy_pct_reason": "only 12 idle ticks, need 40"}
    v, why = C.verdict(o)
    check("verdict", v, "INCONCLUSIVE")
    check("why blames the instrument", "clamp" in why, True)
    check("and does NOT claim the base failed to collapse", "did NOT exhibit" in why, False)

    print("== a key that is PRESENT AND NULL does not crash the scoring layer")
    # Different from absent, and it is what a degraded payload carries. A crash here lands after
    # an eight-hour run has been paid for, and takes the artifact's own re-scoring with it.
    for key in ("busy", "robust", "raf"):
        o = fixture()
        for r in o["runs"]:
            for ph in r["payload"]["phases"]:
                ph[key] = None
            for act in r["payload"]["actions"]:
                act[key] = None
        check(f"phases and actions with {key}=null: gates", 
              isinstance(list(C.gates(o)), list), True)
        check(f"phases and actions with {key}=null: table", isinstance(C.table(o), str), True)
        check(f"phases and actions with {key}=null: verdict",
              isinstance(C.verdict(o)[0], str), True)
    o = fixture()
    for r in o["runs"]:
        r["payload"]["mount"] = {"census": None}
        r["payload"]["final"] = None
    check("a null census does not crash gates", isinstance(list(C.gates(o)), list), True)
    check("and the mount gate says NO rather than raising", gate_ok(o, "really mounted"), False)
    check("a null census does not crash table", isinstance(C.table(o), str), True)

    print("== a window that caught a stall is FLAGGED rather than averaged away")
    o = fixture()
    for r in o["runs"]:
        if r["arm"] == "pre" and r["rung"] == "500K":
            for ph in r["payload"]["phases"]:
                if ph["phase"] == "scroll":
                    ph["blocked_ms_per_frame"] = 287.0
                    ph["robust"]["blocked_ms_per_frame"] = 19.0
    check("the table says which of the two numbers to weigh", "caught a stall" in C.table(o), True)
    check("and an ordinary window is not flagged", "caught a stall" in C.table(fixture()), False)

    print("== AN ACTION THAT DID NOT APPLY IS NOT A ZERO FRAME RATE")
    o = fixture()
    for r in o["runs"]:
        if r["rung"] == "0K":
            for act in r["payload"]["actions"]:
                act["not_applicable"] = True
                # What the scene really emits for a window that returned immediately: one
                # millisecond, no frames, and `1000 * 0 / 1` = 0.0.
                act["eff_fps"] = 0
                act["elapsed_ms"] = 1
                act["frames"] = 0
    tbl = C.table(o)
    row = [ln for ln in tbl.splitlines() if ln.startswith("| 0K | reasoning_toggle")]
    check("the 0K action row exists", len(row), 1)
    check("and says not applicable rather than 0.0 fps",
          "not applicable" in row[0] and "**0.0** fps" not in row[0], True, )

    print("== an untouched fixture is still clean after all of that")
    check("no shared mutation leaked", all_gates_ok(fixture()), True)
    check("deep copy is identical", C.verdict(copy.deepcopy(fixture()))[0], "CONFIRMED")

    print()
    if FAILURES:
        print(f"FAILED {len(FAILURES)}")
        for f in FAILURES:
            print("  -", f)
        return 1
    print("all criteria self-tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
