#!/usr/bin/env python3
"""Self-test for criteria/studio_cvkatex.py, run in the BUILD job before the GPU one.

It exists because the expensive failure in this campaign has never been the runner: it has been a
scoring layer that could only produce one answer. A criteria module that can only say "it works"
is worth less than no measurement at all, because it looks like evidence. So each case below feeds
a synthetic observation whose right answer is known, and the ones that matter are the failures:

  * THE VACUOUS ARM, which is the failure mode this particular fix invites. `content-visibility`
    needs SIZE CONTAINMENT, size containment does not apply to a non-atomic inline-level box, and
    `getComputedStyle(el).contentVisibility` returns the SPECIFIED value regardless -- so an arm
    can be `fired: true` on 400 sampled roots the engine is ignoring completely. Only the height
    probe can tell, and a run where `took_effect.fraction_changed` is 0.0 must disqualify the arm
    and must NOT read as a clean 0%.
  * DRIFT: the local replication of the earlier ablation failed exactly here (baseline 17.5 fps
    against baseline_repeat 30.1) and every ratio in that run was meaningless. And it must be
    caught PER RUNG, naming the rung, because a baseline_repeat at 100K says nothing about 500K.
  * PINNED GESTURE: `contain-intrinsic-size` moves scrollHeight BY DESIGN, so the gate is about
    magnitude. A fraction of a percent is expected; 7% pins the gesture and disqualifies the arm.
  * THE REFERENCE UPPER BOUND NOT REPRODUCING: `.katex{visibility:hidden}` measured +97% in run
    32869180652 (287.6 ms blocked per scroll event at 3.2 fps -> 10.0 ms at 61.0 fps). If it does
    not reproduce here, this session is not that experiment and no candidate can be read against
    that number.
  * THE ALL-SELECTOR ARM WINNING: the whole spec reading says adding `.katex` buys nothing,
    because 910 of the 1,027 roots are inline. If it buys 20 points, the reading is wrong and the
    module has to say so rather than report the shipped rule as a success.
  * A 0K RUNG WITH NO ARMS: an empty thread has nothing to scroll and no maths. That is a RESULT,
    and it must not crash the report, must not fail the window gates, and must not be turned into
    a win.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("crit", HERE / "criteria" / "studio_cvkatex.py")
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


PARK = {"500K": 153387, "100K": 30119, "10K": 4102, "0K": 0}


# ── synthetic payload builders ────────────────────────────────────────────────────────────────

def window(name, arm, bmpf, fps, *, fired = None, took = None, sh = 0.0, ok = True, frames = 80,
           travel = 1.0, mode = "pixel", mutations = 0, park = PARK["500K"]):
    return {
        "name": name, "arm": arm, "ok": ok,
        "blocked_ms_per_frame": bmpf, "eff_fps": fps, "frames": frames,
        "busy": {"busy_pct": 90.0, "blocked_ms": (bmpf or 0) * frames},
        "raf": {"max_ms": 900, "n": frames},
        "fired": fired, "took_effect": took, "scroll_height_delta": sh,
        "census_before": {"scroller_scroll_height": 316829},
        "census_applied": {"scroller_scroll_height": 316829, "elements": 164136},
        "mutations": {"added_elements": mutations, "added_nodes": mutations,
                      "removed_nodes": 0, "records": 1,
                      "buckets": {"data-slot:message-actions": mutations} if mutations else {}},
        "gesture": {"park": park, "mode": mode, "steps": frames,
                    "commanded_px": 0 if mode == "still" else frames,
                    "travelled_px": 0 if mode == "still" else frames * travel,
                    "travel_fraction": None if mode == "still" else travel,
                    "snapback_frames": 0, "stopped_by": "frames"},
    }


def fired_ok(sel, prop, want):
    return {"selector": sel, "prop": prop, "want": want, "total": 1027, "sampled": 400,
            "matching": 400, "fraction": 1.0, "fired": True, "non_matching_examples": []}


def fired_dropped(sel, prop, want, saw):
    return {"selector": sel, "prop": prop, "want": want, "total": 1027, "sampled": 400,
            "matching": 0, "fraction": 0.0, "fired": False, "non_matching_examples": [saw]}


def took(sel, changed, compared, *, before = 53.9, after = 56.0):
    """The height-delta probe: what the ENGINE did, as opposed to what it accepted."""
    return {"selector": sel, "total": 117, "compared": compared, "changed": changed,
            "shrank": changed,
            "fraction_changed": round(changed / compared, 3) if compared else None,
            "mean_height_before": before, "mean_height_after": after,
            "offscreen_before": compared, "offscreen_after": compared}


# The measured corpus, from run 32869180652: 1,027 `.katex` roots of which 117 are
# `.katex-display`, 101,306 descendants split 77,832 inline / 23,591 display.
def census_long(elements = 164136, *, deferred = 0, spans = 49857):
    return {"elements": elements, "messages": 30, "message_descendants": 118000,
            "katex_roots": 1027, "katex_descendants": 101306, "katex_display": 117,
            "katex_display_descendants": 23591, "math_blocks": 402, "code_blocks": 330,
            "highlight_spans": spans, "all_spans": 104000 + spans, "buttons": 90,
            "data_slots": 300, "fences_deferred": deferred, "code_block_bodies": 330,
            "scroller_scroll_height": 316829, "scroller_client_height": 1080}


def census_short(*, deferred = 0):
    """An empty thread: no maths, nothing to scroll, and the selector matches nothing."""
    return {"elements": 1204, "messages": 0, "message_descendants": 0,
            "katex_roots": 0, "katex_descendants": 0, "katex_display": 0,
            "katex_display_descendants": 0, "math_blocks": 0, "code_blocks": 0,
            "highlight_spans": 0, "all_spans": 140, "buttons": 22, "data_slots": 30,
            "fences_deferred": deferred, "code_block_bodies": 0,
            "scroller_scroll_height": 1080, "scroller_client_height": 1080}


def warmup(cen, *, quiesced = True, deltas = (2130, 2474, 12, 4), left_deferred = 0,
           latched_from = 318, park = PARK["500K"]):
    before = dict(cen, elements = cen["elements"] - 47516, fences_deferred = latched_from,
                  highlight_spans = max(0, cen.get("highlight_spans", 0) - 47516))
    after = dict(cen, fences_deferred = left_deferred)
    return {
        "park": park, "quiesced": quiesced, "elapsed_ms": 41000,
        "fence_latch": {"before": before, "after": after, "dispatched": True, "error": None,
                        "fences_latched": latched_from - left_deferred,
                        "elements_added": after["elements"] - before["elements"],
                        "highlight_spans_added": after.get("highlight_spans", 0)
                        - before.get("highlight_spans", 0)},
        "rounds": [{"i": i + 1, "element_delta": d, "frames": 60, "scroll_height": 316829,
                    "elements": before["elements"] + sum(deltas[:i + 1]),
                    "highlight_spans": after.get("highlight_spans", 0),
                    "mutations": {"added_elements": d, "buckets": {}}, "error": None}
                   for i, d in enumerate(deltas)],
        "mut_total": {"added_elements": sum(deltas), "added_nodes": sum(deltas),
                      "removed_nodes": 0, "records": 40,
                      "buckets": {"data-slot:message-actions": sum(deltas)}},
        "census_before": before, "census_after": after,
    }


def positioned_long():
    """~15,291 positioned boxes under inline maths against ~4,919 under display maths."""
    return {"message_descendants": {"selector": "[data-role] *", "total": 118000, "sampled": 1500,
                                    "non_static": 280, "estimate": 22026},
            "katex_descendants": {"selector": ".katex *", "total": 101306, "sampled": 1500,
                                  "non_static": 300, "estimate": 20261},
            "katex_display_roots": {"total": 117, "sampled": 117, "non_static": 40,
                                    "estimate": 40},
            "katex_inline_roots": {"total": 910, "sampled": 910, "non_static": 300,
                                   "estimate": 300},
            "katex_display_descendants": {"total": 23591, "sampled": 1500, "non_static": 313,
                                          "estimate": 4919},
            "katex_inline_descendants": {"total": 77832, "sampled": 1500, "non_static": 295,
                                         "estimate": 15291}}


def common(cen, *, jam_clean, jam_jammed, idle_fps, warm, park):
    return {
        "warmup": warm,
        "fence_latch": warm["fence_latch"],
        "clamp": {"clamp_ms": 1.2, "samples": 400},
        "guard": {"scroll_behavior_before": "smooth", "scroll_behavior_after": "auto",
                  "ok": True, "scroller_tag": "DIV", "scroller_class": "aui-thread-viewport"},
        "liveness": {"clean_fps": jam_clean, "jammed_fps": jam_jammed,
                     "drop_fraction": round(1 - jam_jammed / jam_clean, 3),
                     "clean_blocked_ms_per_frame": 0.4,
                     "jammed_blocked_ms_per_frame": 40.0},
        "idle": {"name": "idle", "eff_fps": idle_fps, "blocked_ms_per_frame": 0.4,
                 "elapsed_ms": 8000, "frames": int(idle_fps * 8)},
        "park": park,
        "math_blocks": {"count": cen["math_blocks"], "roots": cen["katex_roots"],
                        "median_height": 24 if cen["math_blocks"] else None},
        "baseline_census": cen,
        "positioned": positioned_long() if cen["katex_roots"] else {},
    }


def payload_long(*, rung = "500K", base = 287.6, cvd = 172.0, cva = 170.0, cvm = 60.0, vh = 10.0,
                 neg = 285.0, floor = 1.5, pos = 2.0, repeat = None,
                 cvd_fired = None, cvd_took = None, cva_took = None, cvd_sh = 0.0,
                 jam_clean = 61.0, jam_jammed = 17.0, idle_fps = 61.0, warm = None,
                 base_mutations = 0):
    """One repetition of a scored rung, in the scene's SEQUENCE order.

    fps is derived from blocked-ms-per-frame so the two channels stay consistent: the campaign's
    own arithmetic trap was reading busy%, a share of WALL time, as if it were work per event.
    """
    repeat = base if repeat is None else repeat
    park = PARK.get(rung, 153387)
    cen = census_long()

    def f(b):  # a cheap monotone stand-in: less blocked time per frame, more frames per second
        return round(1000.0 / (16.0 + b), 1)

    def w(name, arm, b, **kw):
        return window(name, arm, b, f(b), park = park, **kw)

    arms = [
        w("baseline", "baseline", base, mutations = base_mutations),
        w("noop_touch", "noop_touch", neg),
        w("baseline_2", "baseline", base),
        w("content_visibility_katex_display", "content_visibility_katex_display", cvd,
          fired = cvd_fired or fired_ok(".katex-display", "contentVisibility", "auto"),
          took = took(".katex-display", 90, 96) if cvd_took is None else cvd_took,
          sh = cvd_sh),
        w("baseline_3", "baseline", base),
        # Probed on the INLINE roots on purpose: those are the ones the claim is about, and on the
        # spec reading nothing there can change height at all.
        w("content_visibility_katex_all", "content_visibility_katex_all", cva,
          fired = fired_ok(".katex", "contentVisibility", "auto"),
          took = took(".katex:not(.katex-display *)", 2, 300, before = 18.2, after = 18.2)
          if cva_took is None else cva_took),
        w("baseline_4", "baseline", base),
        w("content_visibility_math_blocks", "content_visibility_math_blocks", cvm,
          fired = fired_ok(".cvk-mathblock", "contentVisibility", "auto"),
          took = took(".cvk-mathblock", 180, 200, before = 61.0, after = 24.0)),
        w("baseline_5", "baseline", base),
        w("visibility_hidden_offscreen", "visibility_hidden_offscreen", vh,
          fired = fired_ok(".katex", "visibility", "hidden")),
        w("baseline_repeat", "baseline", repeat),
        w("still_no_scroll", "still_no_scroll", floor, mode = "still"),
        w("detach_messages", "detach_messages", pos, sh = -0.98, mutations = 120000),
    ]
    p = {"ok": True, "rung": rung, "arms": arms, "no_scroll_range": False,
         "scrollable_px": 315749}
    p.update(common(cen, jam_clean = jam_clean, jam_jammed = jam_jammed, idle_fps = idle_fps,
                    warm = warm if warm is not None else warmup(cen, park = park), park = park))
    return p


def payload_short(*, rung = "0K", jam_clean = 61.2, jam_jammed = 17.2, idle_fps = 61.2,
                  warm = None):
    """The 0K rung: an empty thread. No arms, no scrollable range, and no maths to skip."""
    cen = census_short()
    p = {"ok": True, "rung": rung, "arms": [], "no_scroll_range": True, "scrollable_px": 0}
    p.update(common(cen, jam_clean = jam_clean, jam_jammed = jam_jammed, idle_fps = idle_fps,
                    warm = warm if warm is not None else warmup(
                        cen, deltas = (4, 0), left_deferred = 0, latched_from = 0, park = 0),
                    park = 0))
    return p


def obs(reps = 2, rungs = ("0K", "500K"), short_kw = None, per_rung = None, **kw):
    """rung OUTER, repetition INNER, exactly as the probe writes it."""
    runs = []
    for rung in rungs:
        for rep in range(1, reps + 1):
            if rung == "0K":
                pl = payload_short(rung = rung, **(short_kw or {}))
            else:
                over = dict(kw)
                over.update((per_rung or {}).get(rung, {}))
                pl = payload_long(rung = rung, **over)
            runs.append({"rung": rung, "rep": rep, "rc": 0, "payload": pl})
    return {"xserver": {"display": ":99"},
            "dist": {"index_html": True, "asset_files": 528},
            "install": {"rc": 0}, "rungs": list(rungs), "reps": reps, "runs": runs}


def gate(o, needle):
    for name, ok, ev in C.gates(o):
        if needle in name:
            return ok, ev
    return None, "gate not found"


def dq_of(o, name, rung = "500K"):
    return C._disqualified(name, C._scored(o, rung).get(name) or [])


# ── cases ─────────────────────────────────────────────────────────────────────────────────────

print("case 1: a clean multi-rung run where the SHIPPABLE arm wins, with 0K present")
o = obs()
bad = [n for n, ok, ev in C.gates(o) if not ok]
check("every gate passes", not bad, str(bad))
v, why = C.verdict(o)
check("the verdict is HELPS", v == "HELPS", f"{v}: {why}")
check("it names the shippable arm and its saving",
      "content_visibility_katex_display" in why and "40%" in why, why)
check("it reports the per-rep figures rather than a lone mean", "rep 1:" in why and "rep 2:" in why,
      why)
check("it says the 97% bound is a probe and not shippable",
      "97%" in why and "not shippable" in why and "visibility:hidden" in why, why)
check("it gives the structural reason the fix cannot reach the bound, from the positioned split",
      "24%" in why and "15,291" in why and "4,919" in why, why)
check("it states the measured scrollHeight change", "scrollHeight" in why, why)
check("it prices the exploratory arm as what a renderer-side hoist would be worth",
      "EXPLORATORY" in why and "content_visibility_math_blocks" in why, why)
tbl = C.table(o)
check("the table leads with blocked ms per frame and says why busy% is not the metric",
      tbl.splitlines()[0].startswith("The primary metric is **blocked ms per frame**"),
      tbl.splitlines()[:2])
check("the table is grouped by rung", "### Rung 500K" in tbl and "### Rung 0K" in tbl,
      [l for l in tbl.splitlines() if l.startswith("### ")])
check("the 0K rung gets its own short-context section stating what it establishes",
      "no scrollable range, and that is the short-context answer" in tbl
      and "MATCHES NOTHING" in tbl, tbl)
check("the short section reports roots, display roots, math blocks, elements and idle",
      "`.katex` roots: **0**" in tbl and "which is what the shipped rule selects: **0**" in tbl
      and "maths-bearing blocks" in tbl and "elements in the page: **1,204**" in tbl
      and "**61.2 fps**" in tbl and "**0.4 ms blocked per frame**" in tbl, tbl)
check("and it says the selector census is what makes the short-context claim, not a fast timing",
      "structural rather than statistical" in tbl, tbl)
check("the all-selector arm is labelled a probe rather than a fix",
      "**[SELECTOR PROBE, NOT A FIX]**" in tbl, tbl)
check("the shippable arm is labelled and the exploratory arm is labelled EXPLORATORY everywhere",
      "**[SHIPPABLE]**" in tbl and tbl.count("**[EXPLORATORY, NOT SHIPPABLE AS WRITTEN]**") >= 1,
      tbl)
check("the upper bound is labelled as a reference bound that cannot ship",
      "**[REFERENCE UPPER BOUND, NOT SHIPPABLE]**" in tbl, tbl)
check("the table carries a per-repetition section", "Per repetition" in tbl)
check("the table prints the measured scrollHeight for EVERY arm, not only failures",
      "scrollHeight, measured on every arm" in tbl
      and "- `content_visibility_katex_display` **[SHIPPABLE]**: +0.00%, +0.00%" in tbl, tbl)
check("the table carries the headline comparison with both height probes side by side",
      "Does adding `.katex` to the selector buy anything?" in tbl
      and "90/96" in tbl and "2/300" in tbl, tbl)
check("and the split the fix turns on, counted", "The split the whole fix turns on" in tbl
      and "~4,919" in tbl and "~15,291" in tbl, tbl)

print("case 2: the shippable arm was ACCEPTED and did nothing (the vacuous arm)")
o = obs(cvd = 282.0, cva = 281.0, cvd_took = took(".katex-display", 0, 96, after = 53.9))
bad = [n for n, ok, ev in C.gates(o) if not ok]
check("no INSTRUMENT gate fails: one vacuous arm must not throw away the run", not bad, str(bad))
dq = dq_of(o, "content_visibility_katex_display")
check("the arm is disqualified", bool(dq), str(dq))
check("and the reason says the declaration was accepted and did nothing",
      "ACCEPTED and did nothing" in (dq or ""), str(dq))
check("and it names the selector and gives changed/compared",
      ".katex-display" in (dq or "") and "0/96" in (dq or ""), str(dq))
check("and it names size containment on a non-atomic inline-level box as the reason",
      "SIZE CONTAINMENT" in (dq or "") and "inline-level box" in (dq or ""), str(dq))
v, why = C.verdict(o)
check("the verdict is NOT the winning one", v != "HELPS", f"{v}: {why}")
check("it is a measured absence of benefit rather than a broken run",
      v == "NO_BENEFIT" and "controls behave" in why, f"{v}: {why}")
check("the table marks it DISQUALIFIED", "DISQUALIFIED" in C.table(o))

print("case 3: the shippable arm's declaration was DROPPED by the engine")
o = obs(cvd = 282.0, cva = 281.0,
        cvd_fired = fired_dropped(".katex-display", "contentVisibility", "auto", "visible"))
bad = [n for n, ok, ev in C.gates(o) if not ok]
check("no INSTRUMENT gate fails", not bad, str(bad))
dq = dq_of(o, "content_visibility_katex_display")
check("the arm is disqualified", bool(dq), str(dq))
check("and the reason names the dropped declaration", "did not take" in (dq or ""), str(dq))
v, why = C.verdict(o)
check("the verdict is not the winning one", v == "NO_BENEFIT", f"{v}: {why}")

print("case 4: the shippable arm moved scrollHeight by 7%, which pins the gesture")
o = obs(cvd = 60.0, cvd_sh = 0.07)
dq = dq_of(o, "content_visibility_katex_display")
check("the arm is disqualified even though it looks like a large win",
      bool(dq) and "scrollHeight" in dq, str(dq))
check("and the message says a scrollHeight change is EXPECTED for this fix, and why",
      "EXPECTED" in (dq or "") and "contain-intrinsic-size" in (dq or ""), str(dq))
check("and that the gate is about magnitude, with the 300,000 px document named",
      "MAGNITUDE" in (dq or "") and "300,000 px" in (dq or ""), str(dq))
v, why = C.verdict(o)
check("and it cannot carry the headline", v == "NO_BENEFIT", f"{v}: {why}")
tbl = C.table(o)
check("a 0.00% arm is still printed in the scrollHeight list, labelled",
      "- `visibility_hidden_offscreen` **[REFERENCE UPPER BOUND, NOT SHIPPABLE]**: +0.00%" in tbl,
      tbl)
check("the disqualifying delta is printed as a number too, not only as a disqualification",
      "+7.00%  <- over the gate" in tbl, tbl)
check("the positive control is over the gate but marked exempt rather than as a fault",
      "exempt from it: this arm changes the thread on purpose" in tbl, tbl)

print("case 5: the page DRIFTED at the long rung")
o = obs(repeat = 287.6 * 1.4)
ok, ev = gate(o, "DRIFT")
check("the drift gate fails", ok is False, ev)
check("and it prints both baselines and every baseline in the run",
      "baseline_repeat" in ev and "all baselines" in ev, ev)
v, why = C.verdict(o)
check("the verdict is INCONCLUSIVE and names the failing gate",
      v == "INCONCLUSIVE" and "DRIFT" in why, f"{v}: {why}")
check("and it says that is not the same as the fix not working",
      "not the same as the fix not working" in why, why)

print("case 5b: only ONE rung drifted, and the gate has to name which")
o = obs(rungs = ("0K", "100K", "500K"), per_rung = {"100K": {"repeat": 287.6 * 1.4}})
ok, ev = gate(o, "DRIFT")
check("the drift gate fails", ok is False, ev)
check("and the evidence names the rung that failed", "100K:" in ev and "FAILS AT THIS RUNG" in ev,
      ev)
check("and it does not blame the rung that was clean",
      ev.split("500K:")[-1].find("FAILS AT THIS RUNG") < 0, ev)
check("and the 0K rung is excused rather than failed", "0K: no scrollable range" in ev, ev)

print("case 6: the jam control did not resolve, so the channel is blind")
o = obs(jam_clean = 60.0, jam_jammed = 60.0)
ok, ev = gate(o, "LIVENESS")
check("the liveness gate fails", ok is False, ev)
check("and it names the rung", "500K:" in ev, ev)
v, why = C.verdict(o)
check("the verdict is INCONCLUSIVE", v == "INCONCLUSIVE", f"{v}: {why}")

print("case 7: the NEGATIVE control recovers 40%, so every recovery is an artefact")
o = obs(neg = 287.6 * 0.6)
ok, ev = gate(o, "NEGATIVE control")
check("the negative-control gate fails", ok is False, ev)
v, why = C.verdict(o)
check("the verdict is INCONCLUSIVE", v == "INCONCLUSIVE", f"{v}: {why}")

print("case 8: the all-selector arm beats the display arm by 20 points")
o = obs(cvd = 172.6, cva = 115.0,
        cva_took = took(".katex:not(.katex-display *)", 260, 300, before = 18.2, after = 3.5))
ok, ev = gate(o, "adding `.katex` to the selector buys nothing")
check("the headline gate fails", ok is False, ev)
check("and it says the spec reading is wrong", "spec reading is WRONG" in ev, ev)
check("and that the shipped rule is leaving something on the table",
      "leaving something on the table" in ev, ev)
check("and it prints both savings and both height probes", "+40%" in ev and "+60%" in ev
      and "260/300" in ev, ev)

print("case 9: the reference upper bound only removes 10%")
o = obs(vh = 287.6 * 0.9)
ok, ev = gate(o, "the reference upper bound reproduces")
check("the upper-bound gate fails", ok is False, ev)
check("and it says this session is not the same experiment",
      "not the same experiment" in ev and "97%" in ev, ev)
check("and it prints the published figures it is being read against",
      "287.6" in ev and "10.0 ms" in ev, ev)
v, why = C.verdict(o)
check("the verdict is INCONCLUSIVE rather than a win on the candidate numbers",
      v == "INCONCLUSIVE", f"{v}: {why}")

print("case 10: a 0K-only run with no arms at all")
o = obs(rungs = ("0K",))
v, why = C.verdict(o)
check("it does not crash and does not claim a win", v != "HELPS", f"{v}: {why}")
check("the verdict is INCONCLUSIVE", v == "INCONCLUSIVE", f"{v}: {why}")
ok, ev = gate(o, "at least one rung carried a scrollable range")
check("the scoring-rung gate is the one that fails", ok is False, ev)
ok, ev = gate(o, "every repetition completed")
check("a no_scroll_range rung does NOT fail 'every repetition completed'", ok is True, ev)
for needle in ("DRIFT", "parked at the same fixed pixel", "actually moved the scroller"):
    ok, ev = gate(o, needle)
    check(f"the window gate '{needle}' excuses a rung with no windows", ok is True, ev)
tbl = C.table(o)
check("the table still renders the short-context section", "### Rung 0K" in tbl, tbl[:400])

print("case 11: the short rung is SLOWER at idle than the long rung")
o = obs(short_kw = {"idle_fps": 40.0})
ok, ev = gate(o, "idle frame rate at the short rung")
check("the short-context gate fails", ok is False, ev)
check("and it prints both idle readings and the census that makes the claim structural",
      "40.0 fps" in ev and "61.0 fps at 500K" in ev and "`.katex` roots" in ev, ev)

print("case 12: a SCORED window mounted content (the artefact that failed run 32865232787)")
o = obs(base_mutations = 2130)
ok, ev = gate(o, "no scored window mounted")
check("the quiescence gate fails", ok is False, ev)
check("and it names the window and where the nodes landed",
      "baseline" in ev and "data-slot" in ev, ev)
check("the positive control is exempt, since detaching messages is supposed to mutate",
      "detach_messages" not in ev, ev)

print("case 13: a window that parked somewhere else within its rung")
o = obs()
o["runs"][2]["payload"]["arms"][3]["gesture"]["park"] = 164500
ok, ev = gate(o, "parked at the same fixed pixel")
check("a moved park position fails the gate", ok is False, ev)

print("case 14: the warm-up never converged, and fences were left deferred")
cen = census_long()
o = obs(warm = warmup(cen, quiesced = False, deltas = (2130, 2474, 2381, 1898, 2322),
                      left_deferred = 41))
ok, ev = gate(o, "warm-up reached quiescence")
check("a warm-up that did not settle fails the gate", ok is False, ev)
check("and it prints the per-round element deltas rather than just saying no",
      "2474" in ev and "absorbed" in ev, ev)
ok, ev = gate(o, "deferred-fence reservoir was drained")
check("an undrained reservoir fails its own gate", ok is False, ev)
check("and it prints how many were left and how many latched",
      "41 left" in ev and "latched" in ev, ev)

print("case 15: one repetition of a rung only")
o = obs(reps = 1)
ok, ev = gate(o, "at least twice")
check("a single repetition per rung fails the floor gate", ok is False, ev)

print("case 16: the shippable arm wins but the exploratory arm wins bigger")
o = obs(cvm = 20.0)
v, why = C.verdict(o)
check("the verdict still rests on the shippable arm", v == "HELPS"
      and why.split("For scale")[0].count("content_visibility_math_blocks") == 0, why)
check("and the exploratory arm is reported as a bound on a renderer-side hoist, labelled",
      "EXPLORATORY, NOT SHIPPABLE AS WRITTEN" in why and "+93%" in why, why)

print("\n" + (f"{FAIL} FAILED" if FAIL else "all cvkatex criteria self-tests passed"))
sys.exit(1 if FAIL else 0)
