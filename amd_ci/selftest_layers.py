#!/usr/bin/env python3
"""Self-test for criteria/studio_layers_mechanism.py, run in the BUILD job before the GPU one.

It exists because the expensive failure in this campaign has never been the runner: it has been a
scoring layer that could only produce one answer. So each case below feeds a synthetic
observation whose right answer is known, and the ones that matter are the failures:

  * DRIFT: the local replication of the earlier ablation failed exactly here (baseline 17.5 fps
    against baseline_repeat 30.1) and every ratio in that run was meaningless. A drifted page must
    fail the gate, not be scored.
  * VACUOUS ARM: a declaration the engine DROPPED (three `overflow-anchor` arms in this campaign)
    must disqualify its arm, and must NOT read as a clean null.
  * PINNED GESTURE: an arm that collapses scrollHeight must be disqualified, because one such arm
    reported 111 fps at 184% busy and meant nothing.
  * BLIND CHANNEL: a jam the instrument cannot see must fail the liveness gate.
  * NEGATIVE CONTROL RECOVERING: if touching the DOM without removing anything "helps", every
    recovery in the run is an artefact.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "crit", HERE / "criteria" / "studio_layers_mechanism.py")
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)

FAIL = 0


def check(label: str, cond: bool, detail: str = "") -> None:
    global FAIL
    if cond:
        print(f"  ok   {label}")
    else:
        print(f"  FAIL {label}" + (f" -- {detail}" if detail else ""))
        FAIL += 1


def window(name, arm, bmpf, fps, *, fired = None, sh = 0.0, ok = True, frames = 80,
           travel = 1.0, mode = "pixel"):
    return {
        "name": name, "arm": arm, "ok": ok,
        "blocked_ms_per_frame": bmpf, "eff_fps": fps, "frames": frames,
        "busy": {"busy_pct": 90.0, "blocked_ms": (bmpf or 0) * frames},
        "raf": {"max_ms": 900, "n": frames},
        "fired": fired, "scroll_height_delta": sh,
        "census_before": {"scroller_scroll_height": 300000},
        "census_applied": {"scroller_scroll_height": 300000, "elements": 120000},
        "gesture": {"mode": mode, "steps": frames, "commanded_px": 0 if mode == "still" else frames,
                    "travelled_px": 0 if mode == "still" else frames * travel,
                    "travel_fraction": None if mode == "still" else travel,
                    "snapback_frames": 0, "stopped_by": "frames"},
    }


def fired_ok(sel, prop, want):
    return {"selector": sel, "prop": prop, "want": want, "total": 100000, "sampled": 400,
            "matching": 400, "fraction": 1.0, "fired": True, "non_matching_examples": []}


def fired_dropped(sel, prop, want, saw):
    return {"selector": sel, "prop": prop, "want": want, "total": 100000, "sampled": 400,
            "matching": 0, "fraction": 0.0, "fired": False, "non_matching_examples": [saw]}


def payload(*, base = 30.0, ps = 6.0, ks = 6.5, vh = 4.0, neg = 29.0, floor = 1.5, pos = 2.0,
            repeat = None, ps_fired = None, ps_sh = 0.0, jam_clean = 61.0, jam_jammed = 17.0):
    """One repetition, in SEQUENCE order. fps is derived so the two channels stay consistent."""
    repeat = base if repeat is None else repeat

    def f(b):  # a cheap monotone stand-in: less blocked time per frame, more frames per second
        return round(1000.0 / (16.0 + b), 1)

    arms = [
        window("baseline", "baseline", base, f(base)),
        window("noop_touch", "noop_touch", neg, f(neg)),
        window("baseline_2", "baseline", base, f(base)),
        window("position_static_all", "position_static_all", ps, f(ps),
               fired = ps_fired or fired_ok("[data-role] *", "position", "static"), sh = ps_sh),
        window("baseline_3", "baseline", base, f(base)),
        window("katex_static", "katex_static", ks, f(ks),
               fired = fired_ok(".katex *", "position", "static")),
        window("baseline_4", "baseline", base, f(base)),
        window("visibility_hidden_offscreen", "visibility_hidden_offscreen", vh, f(vh),
               fired = fired_ok(".katex", "visibility", "hidden")),
        window("baseline_repeat", "baseline", repeat, f(repeat)),
        window("still_no_scroll", "still_no_scroll", floor, f(floor), mode = "still"),
        window("detach_messages", "detach_messages", pos, f(pos), sh = -0.98),
    ]
    return {
        "ok": True, "arms": arms,
        "clamp": {"clamp_ms": 1.2, "samples": 400},
        "guard": {"scroll_behavior_before": "smooth", "scroll_behavior_after": "auto",
                  "ok": True, "scroller_tag": "DIV", "scroller_class": "aui-thread-viewport"},
        "liveness": {"clean_fps": jam_clean, "jammed_fps": jam_jammed,
                     "drop_fraction": round(1 - jam_jammed / jam_clean, 3),
                     "clean_blocked_ms_per_frame": 0.4,
                     "jammed_blocked_ms_per_frame": 40.0},
        "baseline_census": {"elements": 122222, "messages": 30, "katex_roots": 1027,
                            "katex_descendants": 101306, "code_blocks": 40,
                            "scroller_scroll_height": 316829},
        "positioned": {"message_descendants": {"total": 118000, "sampled": 1500,
                                               "non_static": 280, "estimate": 22026},
                       "katex_descendants": {"total": 101306, "sampled": 1500,
                                             "non_static": 250, "estimate": 16884}},
    }


def obs(reps = 2, **kw):
    return {
        "xserver": {"display": ":99"},
        "dist": {"index_html": True, "asset_files": 528},
        "install": {"rc": 0},
        "runs": [{"rep": i + 1, "rc": 0, "payload": payload(**kw)} for i in range(reps)],
    }


def gate(o, needle):
    for name, ok, ev in C.gates(o):
        if needle in name:
            return ok, ev
    return None, "gate not found"


print("case 1: a clean run where the ablation works")
o = obs()
bad = [n for n, ok, ev in C.gates(o) if not ok]
check("every gate passes", not bad, str(bad))
v, why = C.verdict(o)
check("the verdict is TRANSFERS", v == "TRANSFERS", f"{v}: {why}")
check("it names the arm with the largest saving, on blocked-ms-per-frame",
      "visibility_hidden_offscreen" in why, why)
check("and it compares against the local figure rather than asserting a direction",
      "llvmpipe" in why, why)
tbl = C.table(o)
check("the table leads with blocked ms per frame and says why busy% is not the metric",
      tbl.splitlines()[0].startswith("The primary metric is **blocked ms per frame**")
      and tbl.splitlines()[2].startswith("| window | blocked ms/frame |"), tbl.splitlines()[:4])
check("the table carries a per-repetition section", "Per repetition" in tbl)

print("case 2: the page DRIFTED (the failure that invalidated the local replication)")
o = obs(repeat = 30.0 * 0.55)
ok, ev = gate(o, "DRIFT")
check("the drift gate fails", ok is False, ev)
check("and it prints both baselines and every baseline in the run",
      "baseline_repeat" in ev and "all baselines" in ev, ev)

print("case 3: an arm whose declaration the engine DROPPED")
o = obs(ps_fired = fired_dropped("[data-role] *", "position", "static", "relative"))
bad = [n for n, ok, ev in C.gates(o) if not ok]
check("no INSTRUMENT gate fails: one bad arm must not throw away the others", not bad, str(bad))
dq = C._disqualified("position_static_all", C._scored(o)["position_static_all"])
check("the arm is disqualified", bool(dq), str(dq))
check("and the reason names the dropped declaration", "did not take" in (dq or ""), str(dq))
v, why = C.verdict(o)
check("the verdict falls back to a still-valid arm rather than the vacuous one",
      v == "TRANSFERS" and "position_static_all" not in why.split("Other")[0], why)
check("the table marks it DISQUALIFIED", "DISQUALIFIED" in C.table(o))

print("case 4: an arm that PINNED the gesture by collapsing scrollHeight")
o = obs(ps_sh = -0.44, ps = 0.5)
dq = C._disqualified("position_static_all", C._scored(o)["position_static_all"])
check("the arm is disqualified even though it looks like the biggest win ever",
      bool(dq) and "scrollHeight" in dq, str(dq))
v, why = C.verdict(o)
check("and it cannot be the headline", "position_static_all" not in why.split("Other")[0], why)

print("case 5: a channel that cannot see a jammed main thread")
o = obs(jam_clean = 60.0, jam_jammed = 60.0)
ok, ev = gate(o, "LIVENESS")
check("the liveness gate fails", ok is False, ev)

print("case 6: the NEGATIVE control recovers, so every recovery is an artefact")
o = obs(neg = 12.0)
ok, ev = gate(o, "NEGATIVE control")
check("the negative-control gate fails", ok is False, ev)

print("case 7: nothing was removed -- the honest null")
o = obs(ps = 29.0, ks = 29.5, vh = 28.8)
bad = [n for n, ok, ev in C.gates(o) if not ok]
check("the gates still pass, so this is a finding and not a broken run", not bad, str(bad))
v, why = C.verdict(o)
check("the verdict is DOES_NOT_TRANSFER", v == "DOES_NOT_TRANSFER", f"{v}: {why}")
check("and it says the cost is real rather than implying the harness failed",
      "controls behave" in why or "both controls" in why, why)

print("case 8: the positive control cannot show a win")
o = obs(pos = 29.5)
ok, ev = gate(o, "POSITIVE control")
check("the positive-control gate fails", ok is False, ev)

print("case 9: one repetition only")
o = obs(reps = 1)
ok, ev = gate(o, "at least two repetitions")
check("a single repetition fails the floor gate", ok is False, ev)

print("case 10: the gesture never moved the scroller")
o = obs()
for r in o["runs"]:
    for a in r["payload"]["arms"]:
        if a["arm"] == "baseline":
            a["gesture"]["travel_fraction"] = 0.02
ok, ev = gate(o, "actually moved the scroller")
check("an inert gesture fails its own gate", ok is False, ev)

print("\n" + (f"{FAIL} FAILED" if FAIL else "all layers criteria self-tests passed"))
sys.exit(1 if FAIL else 0)
