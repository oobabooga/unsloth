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
  * THE PRODUCT ARM MEASURED ON A BUNDLE THAT NEVER HAD THE FIX. `product_math_block_containment`
    sets ONE ATTRIBUTE and injects no CSS, so on the wrong build the toggle is inert and the arm
    reads as a clean null sitting next to the harness arm's win. That is the most dangerous shape
    in this file, because the plausible reading of it -- "the product implementation does not
    reproduce the harness arm" -- is the wrong one. The module must VOID the arm, say the bundle
    lacked the rule in words, and NOT void the run; and it must keep that answer distinct from a
    genuine null measured on a build where the precondition holds.

EVERY CASE ASSERTS ITS OWN FIXTURE FIRST (defect #51). A synthetic payload that drifts into no
longer exercising the property under test turns a passing case into a decoration, and the only
defence is for the case to fail loudly, naming itself, when its own inputs stop being what it
says they are.
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


def precondition(case: str, cond: bool, detail: str = "") -> None:
    """DEFECT #51: a case must first prove its own fixture still exercises what it claims.

    A fixture that drifts -- a builder default changed, a window renamed, a field dropped -- turns
    a green case into a decoration that asserts something about a payload nobody meant to build.
    This fails loudly and names the case rather than the assertion, because the fault is in the
    setup and looking at the assertion would waste the reader's time.
    """
    global FAIL
    if cond:
        print(f"  ok   [precondition] {case}")
    else:
        print(f"  FAIL [precondition] {case}: THE FIXTURE NO LONGER EXERCISES WHAT THIS CASE "
              f"TESTS, so its assertions below prove nothing"
              + (f" -- {detail}" if detail else ""))
        FAIL += 1


PARK = {"500K": 153387, "100K": 30119, "10K": 4102, "0K": 0}


# ── synthetic payload builders ────────────────────────────────────────────────────────────────

def window(name, arm, bmpf, fps, *, fired = None, took = None, sh = 0.0, ok = True, frames = 80,
           travel = 1.0, mode = "pixel", mutations = 0, park = PARK["500K"], selector = "none",
           robust_bmpf = None, stalls = 0, worst_gap = 900, p50 = None):
    """One measured window.

    `robust_bmpf` defaults to the mean, which is the healthy case: dropping the single worst frame
    changes nothing when no frame dominated. A window that caught one of this venue's 8.6 s stalls
    is built by passing a robust figure far below the mean, one frame over a second, and a worst
    gap in the thousands, which is what run 32876363634 looked like on the arm that read 112.9
    ms/frame at a p50 of 17 ms.

    `p50` defaults to `16 + bmpf`, the same monotone stand-in the fps channel uses, so a case that
    moves the mean moves the median with it unless it says otherwise. Cases about stalls pass the
    two apart on purpose, since that is the whole point of the statistic.
    """
    if robust_bmpf is None and bmpf is not None:
        robust_bmpf = round(bmpf * 0.98, 1)
    if p50 is None and bmpf is not None:
        p50 = round(16.0 + bmpf)
    return {
        "name": name, "arm": arm, "ok": ok,
        "blocked_ms_per_frame": bmpf, "eff_fps": fps, "frames": frames,
        "busy": {"busy_pct": 90.0, "blocked_ms": (bmpf or 0) * frames},
        "raf": {"max_ms": worst_gap, "n": frames, "p50_ms": p50,
                "p95_ms": None if p50 is None else round(p50 * 1.05),
                "over_100": stalls},
        "robust": {"blocked_ms": None if robust_bmpf is None else round(robust_bmpf * (frames - 1)),
                   "frames": frames - 1, "blocked_ms_per_frame": robust_bmpf,
                   "stall_frames_over_1s": stalls, "worst_tick_ms": worst_gap,
                   "worst_gap_ms": worst_gap},
        "selector": selector, "apply_detail": selector,
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


def stall(mean, robust, *, stalls = 1, gap = 8595):
    """Override one window: the MEAN it reported, and what it reads with that frame dropped.

    p50 follows the robust figure, not the mean, because that is what a median does: in run
    32876363634 the same arm read 10.0 and 112.9 ms/frame in two sessions at an identical p50 of
    17 ms. Pass `stalls = 0` and a small gap to build the FALSE POSITIVE from run 32882865551
    instead: `still_no_scroll` at r100K, 0.5 against 0.2, no frame over a second, worst gap 48 ms.
    """
    return {"mean": mean, "robust": robust, "stalls": stalls, "gap": gap}


def fired_ok(sel, prop, want):
    return {"selector": sel, "prop": prop, "want": want, "total": 1027, "sampled": 400,
            "matching": 400, "fraction": 1.0, "fired": True, "non_matching_examples": []}


def fired_dropped(sel, prop, want, saw):
    return {"selector": sel, "prop": prop, "want": want, "total": 1027, "sampled": 400,
            "matching": 0, "fraction": 0.0, "fired": False, "non_matching_examples": [saw]}


# The product branch's rule, frozen, exactly as `studio-math-block-containment` ships it. The
# scene quotes what it FOUND in the loaded stylesheets, so these two strings stand in for a real
# `CSSStyleRule.selectorText` and `.cssText` and are compared verbatim.
PRODUCT_SELECTOR_TEXT = ('html[data-math-block-containment="on"] .aui-thread-root '
                         ':is(.aui-math-block, .katex-display)')
PRODUCT_CSS_TEXT = (PRODUCT_SELECTOR_TEXT
                    + " { content-visibility: auto; contain-intrinsic-size: auto 7.5rem; }")


def product_rule(*, present = True, blocks = 402, katex_display = 117, moved = None,
                 sampled = 24, sheets_readable = 6, sheets_unreadable = 1):
    """The precondition the product arm cannot be read without, as the scene records it.

    Two independent halves, and the cases below need each of them to be able to say NO on its own:
    whether the RULE EXISTS in the bundle (`present`, from walking `document.styleSheets`), and
    whether it ACTS (`toggle_check`, from computed `content-visibility` off then on). A bundle
    built without the fix says no to both; a rule that shipped but does not reach the class says
    no only to the second, and those are different faults with different words.

    `blocks = 0` is the 0K rung: an empty thread has no maths, so there is nothing to toggle and
    that is the CORRECT answer there. The RULE must still be found, because one bundle serves
    every rung.
    """
    if blocks == 0:
        sampled, moved = 0, 0
    elif moved is None:
        moved = sampled if present else 0
    shown = min(sampled, 6)
    tc = {"selector": ".aui-math-block", "total": blocks, "sampled": sampled,
          "attribute_before": None, "attribute_after": None,
          "off_values": ["visible"] * shown,
          "on_values": [("auto" if i < min(moved, shown) else "visible") for i in range(shown)],
          "moved": moved, "auto_when_on": moved,
          "moved_fraction": (round(moved / sampled, 3) if sampled else None),
          "note": None if sampled else
                  ("no `.aui-math-block` element exists at this rung, so there is nothing to "
                   "toggle. At an empty thread that is the CORRECT answer and not a failure, but "
                   "the RULE must still be present in the bundle here")}
    return {"attribute": "data-math-block-containment", "block_class": "aui-math-block",
            "block_selector": ".aui-math-block",
            "sheets_total": sheets_readable + sheets_unreadable,
            "sheets_readable": sheets_readable, "sheets_unreadable": sheets_unreadable,
            "unreadable": ([{"sheet": "https://fonts.googleapis.com/css2", "why": "SecurityError"}]
                           if sheets_unreadable else []),
            "rules_scanned": 4821,
            "matches": ([{"selector_text": PRODUCT_SELECTOR_TEXT, "css_text": PRODUCT_CSS_TEXT,
                          "sheet": "(inline <style>)"}] if present else []),
            "present": present,
            "selector_text": PRODUCT_SELECTOR_TEXT if present else None,
            "css_text": PRODUCT_CSS_TEXT if present else None,
            "blocks": blocks, "katex_display": katex_display, "toggle_check": tc}


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
            "katex_display_descendants": 23591, "math_blocks": 402,
            "product_math_blocks": 402, "code_blocks": 330,
            "highlight_spans": spans, "all_spans": 104000 + spans, "buttons": 90,
            "data_slots": 300, "fences_deferred": deferred, "code_block_bodies": 330,
            "scroller_scroll_height": 316829, "scroller_client_height": 1080}


def census_short(*, deferred = 0):
    """An empty thread: no maths, nothing to scroll, and the selector matches nothing."""
    return {"elements": 1204, "messages": 0, "message_descendants": 0,
            "katex_roots": 0, "katex_descendants": 0, "katex_display": 0,
            "katex_display_descendants": 0, "math_blocks": 0,
            "product_math_blocks": 0, "code_blocks": 0,
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


def common(cen, *, jam_clean, jam_jammed, idle_fps, warm, park, prod = None):
    return {
        # PER RUNG AND PER REP, because a payload is one repetition of one rung. At 0K this is the
        # only thing the rung has to say about the product build.
        "product_rule": product_rule(blocks = cen["product_math_blocks"],
                                     katex_display = cen["katex_display"])
        if prod is None else prod,
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


SEL_VH = ".katex{visibility:hidden}"
SEL_CVD = ".katex-display{content-visibility:auto;contain-intrinsic-size:auto 3.5rem}"
SEL_CVA = ".katex,.katex-display{content-visibility:auto;contain-intrinsic-size:auto 3.5rem}"
SEL_CVM = ".cvk-mathblock{content-visibility:auto;contain-intrinsic-size:auto 24px} over 402 blocks"
# What the PRODUCT arm applied, in the shape the scene records it: the selector the PRODUCT
# stylesheet carries, read back out of the page, followed by the attribute this arm set and the
# populations it covers. The arm injects no CSS of its own, so this string is the only place a
# report can see what its window was actually about (defect #50).
SEL_PROD = (PRODUCT_SELECTOR_TEXT + ' <- set [data-math-block-containment="on"] on <html>, '
            "covering 402 .aui-math-block and 117 .katex-display")


def payload_long(*, rung = "500K", base = 287.6, cvd = 172.0, cva = 170.0, cvm = 60.0, vh = 10.0,
                 vh_late = None, neg = 285.0, floor = 1.5, pos = 2.0, repeat = None,
                 cvd_fired = None, cvd_took = None, cva_took = None, cvd_sh = 0.0,
                 jam_clean = 61.0, jam_jammed = 17.0, idle_fps = 61.0, warm = None,
                 base_mutations = 0, stalled = None, p50s = None,
                 prodv = 62.0, prod = None, prod_fired = None, prod_took = None, prod_sh = 0.0):
    """One repetition of a scored rung, in the scene's SEQUENCE order.

    fps is derived from blocked-ms-per-frame so the two channels stay consistent: the campaign's
    own arithmetic trap was reading busy%, a share of WALL time, as if it were work per event.

    `stalled` is {window name: stall(...)}, which overrides that window's mean, robust figure,
    stall count and worst gap together. `p50s` is {window name: median rAF gap} for cases about
    the median specifically.
    """
    repeat = base if repeat is None else repeat
    vh_late = vh if vh_late is None else vh_late
    stalled, p50s = stalled or {}, p50s or {}
    park = PARK.get(rung, 153387)
    cen = census_long()

    def f(b):  # a cheap monotone stand-in: less blocked time per frame, more frames per second
        return round(1000.0 / (16.0 + b), 1)

    def w(name, arm, b, **kw):
        o = stalled.get(name)
        if o:
            b = o["mean"]
            kw.setdefault("robust_bmpf", o["robust"])
            kw.setdefault("stalls", o["stalls"])
            kw.setdefault("worst_gap", o["gap"])
            # A median is not moved by the frame that moved the mean.
            kw.setdefault("p50", round(16.0 + o["robust"]))
        if name in p50s:
            kw["p50"] = p50s[name]
        return window(name, arm, b, f(b), park = park, **kw)

    # The scene's SEQUENCE, with the reference upper bound run TWICE: fourth, and again twelfth
    # after the three arms that skip and unskip large subtrees. Run 32876363634 read 55% from it
    # at r500K where run 32869180652 published 97%, and arm order was the only visible difference.
    arms = [
        w("baseline", "baseline", base, mutations = base_mutations),
        w("noop_touch", "noop_touch", neg, selector = "touched 30 messages"),
        w("baseline_2", "baseline", base),
        w("katex_root_visibility_hidden", "katex_root_visibility_hidden", vh, selector = SEL_VH,
          fired = fired_ok(".katex", "visibility", "hidden")),
        w("baseline_3", "baseline", base),
        w("content_visibility_katex_display", "content_visibility_katex_display", cvd,
          selector = SEL_CVD,
          fired = cvd_fired or fired_ok(".katex-display", "contentVisibility", "auto"),
          took = took(".katex-display", 90, 96) if cvd_took is None else cvd_took,
          sh = cvd_sh),
        w("baseline_4", "baseline", base),
        # Probed on the INLINE roots on purpose: those are the ones the claim is about, and on the
        # spec reading nothing there can change height at all.
        w("content_visibility_katex_all", "content_visibility_katex_all", cva, selector = SEL_CVA,
          fired = fired_ok(".katex", "contentVisibility", "auto"),
          took = took(".katex:not(.katex-display *)", 2, 300, before = 18.2, after = 18.2)
          if cva_took is None else cva_took),
        w("baseline_5", "baseline", base),
        w("content_visibility_math_blocks", "content_visibility_math_blocks", cvm,
          selector = SEL_CVM,
          fired = fired_ok(".cvk-mathblock", "contentVisibility", "auto"),
          took = took(".cvk-mathblock", 180, 200, before = 61.0, after = 24.0)),
        w("baseline_6", "baseline", base),
        # THE PRODUCT IMPLEMENTATION, one baseline after the harness arm it is compared with, so
        # the two are separated by a single window and nothing else.
        w("product_math_block_containment", "product_math_block_containment", prodv,
          selector = SEL_PROD,
          fired = prod_fired or fired_ok(".aui-math-block", "contentVisibility", "auto"),
          took = took(".aui-math-block", 180, 200, before = 61.0, after = 24.0)
          if prod_took is None else prod_took,
          sh = prod_sh),
        w("baseline_7", "baseline", base),
        w("katex_root_visibility_hidden_late", "katex_root_visibility_hidden", vh_late,
          selector = SEL_VH, fired = fired_ok(".katex", "visibility", "hidden")),
        w("baseline_repeat", "baseline", repeat),
        w("still_no_scroll", "still_no_scroll", floor, mode = "still",
          selector = "gesture mode: still"),
        w("detach_messages", "detach_messages", pos, sh = -0.98, mutations = 120000,
          selector = "removed 28 of 30 messages"),
    ]
    p = {"ok": True, "rung": rung, "arms": arms, "no_scroll_range": False,
         "scrollable_px": 315749}
    p.update(common(cen, jam_clean = jam_clean, jam_jammed = jam_jammed, idle_fps = idle_fps,
                    warm = warm if warm is not None else warmup(cen, park = park), park = park,
                    prod = prod))
    return p


def payload_short(*, rung = "0K", jam_clean = 61.2, jam_jammed = 17.2, idle_fps = 61.2,
                  warm = None, prod = None):
    """The 0K rung: an empty thread. No arms, no scrollable range, and no maths to skip."""
    cen = census_short()
    p = {"ok": True, "rung": rung, "arms": [], "no_scroll_range": True, "scrollable_px": 0}
    p.update(common(cen, jam_clean = jam_clean, jam_jammed = jam_jammed, idle_fps = idle_fps,
                    warm = warm if warm is not None else warmup(
                        cen, deltas = (4, 0), left_deferred = 0, latched_from = 0, park = 0),
                    park = 0, prod = prod))
    return p


# WHAT WAS BUILT. The product arm is a claim about one branch at one commit, so the observation
# has to name it or nothing measured can be attributed. The probe writes both of these.
SUBJECT = {"repo": "https://github.com/unslothai/unsloth",
           "ref": "studio-math-block-containment",
           "resolved_ref": "studio-math-block-containment",
           "commit": "3f9c1a2b7d4e5f60718293a4b5c6d7e8f9012345"}


def obs(reps = 2, rungs = ("0K", "500K"), short_kw = None, per_rung = None, prod_kw = None,
        subject = SUBJECT, **kw):
    """rung OUTER, repetition INNER, exactly as the probe writes it.

    `prod_kw` is applied to EVERY rung, because one bundle serves them all: a fixture that made
    the product rule present at 500K and absent at 0K would be describing something that cannot
    happen, and the precondition gate exists to catch exactly that shape.
    """
    runs = []
    for rung in rungs:
        for rep in range(1, reps + 1):
            if rung == "0K":
                sk = dict(short_kw or {})
                if prod_kw is not None:
                    sk.setdefault("prod", product_rule(
                        **{**{"blocks": 0, "katex_display": 0}, **prod_kw}))
                pl = payload_short(rung = rung, **sk)
            else:
                over = dict(kw)
                over.update((per_rung or {}).get(rung, {}))
                if prod_kw is not None:
                    over.setdefault("prod", product_rule(**prod_kw))
                pl = payload_long(rung = rung, **over)
            runs.append({"rung": rung, "rep": rep, "rc": 0, "payload": pl})
    s = dict(subject or {})
    return {"xserver": {"display": ":99"},
            "dist": {"index_html": True, "asset_files": 528},
            "install": {"rc": 0}, "rungs": list(rungs), "reps": reps, "runs": runs,
            "subject": s,
            "clone": {"url": s.get("repo"), "ref": s.get("ref"),
                      "resolved_ref": s.get("resolved_ref"), "commit": s.get("commit"),
                      "dest": "/tmp/studio_layers/cvkatex/repo"}}


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
      and "- `content_visibility_katex_display` **[SHIPPABLE]** (`.katex-display{content-"
          "visibility:auto;cont...`): +0.00%, +0.00%" in tbl, tbl)
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
check("a 0.00% arm is still printed in the scrollHeight list, labelled and with its selector",
      "- `katex_root_visibility_hidden` **[REFERENCE UPPER BOUND, NOT SHIPPABLE]** "
      "(`.katex{visibility:hidden}`): +0.00%" in tbl, tbl)
check("the disqualifying delta is printed as a number too, not only as a disqualification",
      "+7.00%  <- over the gate" in tbl, tbl)
check("the positive control is over the gate but marked exempt rather than as a fault",
      "exempt from it: this arm changes the thread on purpose" in tbl, tbl)

print("case 5: the page DRIFTED at the long rung")
o = obs(repeat = 287.6 * 1.4)
ok, ev = gate(o, "DRIFT")
check("the drift gate fails", ok is False, ev)
check("and it prints both baselines and every baseline in the run",
      "baseline_repeat" in ev and "all baseline p50" in ev, ev)
check("and it prints the raw means too, marked as not judged",
      "raw means, not judged" in ev, ev)
v, why = C.verdict(o)
check("the verdict is INCONCLUSIVE and names the failing gate",
      v == "INCONCLUSIVE" and "DRIFT" in why, f"{v}: {why}")
check("and it says that is not the same as the fix not working",
      "not the same as the fix not working" in why, why)

print("case 5a: the MEANS swing 40% but only because one baseline caught a host stall")
# Run 32902943628, at 500K. The eight baseline means came out
# [285.4, 253, 246.6, 174.5, 252.9, 184.3, 286.7, 172.2] while their p50 rAF gaps sat within one
# millisecond of each other at [249, 248, 247, 248, 250, 249, 249, 248]. The page had not drifted:
# this venue blocks its main thread for about 8.6 s once every 30 s, and one such frame spread
# over the ~94 frames of a window is worth ~91 ms/frame on the MEAN and nothing at all on the
# median. Judging drift on the mean made this gate a coin toss on the venue's own stall schedule.
o = obs(per_rung = {"500K": dict(
    base = 285.4, repeat = 172.2,
    stalled = {"baseline": stall(285.4, 196.3), "baseline_repeat": stall(172.2, 167.7, stalls = 0,
                                                                        gap = 331)},
    p50s = {"baseline": 249, "baseline_repeat": 248})})
ok, ev = gate(o, "DRIFT")
check("the drift gate PASSES, because the medians agree to within a millisecond", ok is True, ev)
check("and it shows the stall-stripped means agreeing too", "stall-stripped mean" in ev, ev)
check("and the raw 40% swing is still printed, so a reader can see what was discarded",
      "285.4" in ev and "172.2" in ev, ev)

print("case 5a2: a drift the median cannot see is still caught by the stall-stripped mean")
# The median is coarse: rAF gaps land on whole milliseconds, so at a cheap rung a real change can
# hide inside one bucket. That is why the gate judges two statistics rather than one.
o = obs(per_rung = {"500K": dict(
    base = 285.4, repeat = 400.0,
    stalled = {"baseline": stall(285.4, 280.0, stalls = 0, gap = 331),
               "baseline_repeat": stall(400.0, 395.0, stalls = 0, gap = 331)},
    p50s = {"baseline": 249, "baseline_repeat": 249})})
ok, ev = gate(o, "DRIFT")
check("the drift gate fails on the stall-stripped mean alone", ok is False, ev)
check("and the median it could not use is printed at +0%", "+0%" in ev, ev)

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

print("case 17: an 8.6 s stall inside a ~10 ms upper-bound window (run 32876363634's shape)")
o = obs(stalled = {"katex_root_visibility_hidden": stall(112.9, 18.8, gap = 8400)})
ok, ev = gate(o, "dominated by a single frame")
check("the stall gate fails", ok is False, ev)
check("and it names the rung, the window and both numbers",
      "500K:" in ev and "katex_root_visibility_hidden" in ev and "112.9" in ev and "18.8" in ev,
      ev)
check("and it prints the ratio, the stall count and the worst gap",
      "6.0x" in ev and "1 frame over 1 s" in ev and "8400 ms" in ev, ev)
v, why = C.verdict(o)
check("the verdict is INCONCLUSIVE, not a reading of the stalled window",
      v == "INCONCLUSIVE" and "dominated by a single frame" in why, f"{v}: {why}")

print("case 17b: a BASELINE caught one, which contaminates the arms on both sides of it")
# At a CHEAP rung, where one 8.6 s frame is worth more than everything else in the window.
CHEAP_100K = {"base": 12.0, "repeat": 12.0, "neg": 11.8, "cvd": 7.0, "cva": 7.0, "cvm": 3.0,
              "vh": 1.5, "floor": 0.5, "pos": 0.4}
o = obs(rungs = ("0K", "100K", "500K"),
        per_rung = {"100K": dict(CHEAP_100K, stalled = {"baseline_3": stall(110.0, 12.0)})})
ok, ev = gate(o, "dominated by a single frame")
check("the stall gate fails on a baseline too", ok is False, ev)
check("and it names the rung and the baseline window", "100K:" in ev and "baseline_3" in ev, ev)
check("and the clean rung is not blamed for it",
      ev.split("500K:")[-1].find("FAILS AT THIS RUNG") < 0, ev)
check("the table marks the baseline row disqualified",
      "`baseline_3` (baseline)  **[DISQUALIFIED: caught a stall]**" in C.table(o), C.table(o))

print("case 17b2: the same stall on a 500K baseline does NOT fire, and that is correct")
# A 287.6 ms/frame baseline that catches one 8.6 s frame reads ~378: a 1.3x ratio, because the
# window's real cost already dwarfs the stall. The gate fires where the stall DOMINATES, which is
# the claim it makes; a window whose reading moves by a third is a different problem from one
# whose reading is a single frame, and the drift and repeat gates are what cover that.
o = obs(stalled = {"baseline_3": stall(287.6 + 90.0, 287.6)})
ok, ev = gate(o, "dominated by a single frame")
check("a 1.3x inflation of an already-expensive window does not fire", ok is True, ev)
check("and the module says so rather than silently rounding it away",
      "survives dropping its worst frame" in ev, ev)

print("case 17c: a clean run, where the mean survives dropping its worst frame")
o = obs()
ok, ev = gate(o, "dominated by a single frame")
check("the stall gate passes", ok is True, ev)
check("and it says so rather than staying silent",
      "survives dropping its worst frame" in ev, ev)
check("the 0K rung is excused, having no windows", "0K: no scrollable range" in ev, ev)

print("case 17d: the FALSE POSITIVE from run 32882865551 -- 0.5 vs 0.2 ms, no frame over 1 s")
o = obs(stalled = {"still_no_scroll": stall(0.5, 0.2, stalls = 0, gap = 48)})
ok, ev = gate(o, "dominated by a single frame")
check("a 2.5x ratio between two near-zero numbers does NOT fire", ok is True, ev)
bad = [n for n, ok_, ev_ in C.gates(o) if not ok_]
check("and the run is not voided by it", not bad, str(bad))
check("the window is not disqualified either",
      not C._disqualified("still_no_scroll", C._scored(o, "500K")["still_no_scroll"]),
      str(C._disqualified("still_no_scroll", C._scored(o, "500K")["still_no_scroll"])))
check("both discriminators are what saved it: no frame over 1 s, and a mean under the floor",
      C._stall_dominated({"blocked_ms_per_frame": 0.5,
                          "robust": {"blocked_ms_per_frame": 0.2,
                                     "stall_frames_over_1s": 0, "worst_gap_ms": 48}}) is None
      and C._stall_dominated({"blocked_ms_per_frame": 0.5,
                              "robust": {"blocked_ms_per_frame": 0.2,
                                         "stall_frames_over_1s": 1, "worst_gap_ms": 48}}) is None)

print("case 17e: a stalled CONTROL disqualifies the control and does NOT void the run")
# 500K `still_no_scroll` in run 32882865551: mean 102.3, robust 10.3, one 8,595 ms frame.
o = obs(stalled = {"still_no_scroll": stall(102.3, 10.3, gap = 8595)})
ok, ev = gate(o, "dominated by a single frame")
check("the stall gate PASSES, because a control carries no conclusion", ok is True, ev)
check("and it still says the control caught one and was disqualified",
      "no conclusion-carrying window caught a stall" in ev and "still_no_scroll" in ev
      and "CONTROL, disqualified but not fatal" in ev, ev)
dq = C._disqualified("still_no_scroll", C._scored(o, "500K")["still_no_scroll"])
check("the control window is disqualified", bool(dq) and "caught one of this host" in dq, str(dq))
check("and the disqualification says its gate is re-read on the robust figure",
      "re-evaluated on the robust figure" in (dq or ""), str(dq))
ok, ev = gate(o, "exhibits the collapse")
check("the collapse gate passes on the robust floor", ok is True, ev)
check("and says so explicitly, and why", "read on its ROBUST figure" in ev
      and "voiding the whole run" in ev, ev)
ok, ev = gate(o, "POSITIVE control recovers")
check("the positive control's gate also passes", ok is True, ev)
bad = [n for n, ok_, ev_ in C.gates(o) if not ok_]
check("no gate fails, so a 26 minute GPU run is not thrown away", not bad, str(bad))
v, why = C.verdict(o)
check("and the run still produces an answer", v == "HELPS", f"{v}: {why}")

print("case 17f: a stalled CANDIDATE arm DOES void the run")
o = obs(stalled = {"content_visibility_katex_display": stall(112.9, 18.8)})
ok, ev = gate(o, "dominated by a single frame")
check("the stall gate fails", ok is False, ev)
check("and it names the candidate", "content_visibility_katex_display" in ev, ev)
v, why = C.verdict(o)
check("the verdict is INCONCLUSIVE", v == "INCONCLUSIVE", f"{v}: {why}")

print("case 17g: the reference upper bound's LATE window caught one -- also fatal")
o = obs(stalled = {"katex_root_visibility_hidden_late": stall(112.9, 18.8)})
ok, ev = gate(o, "dominated by a single frame")
check("the late upper-bound window carries a conclusion too", ok is False, ev)
check("and it is named", "katex_root_visibility_hidden_late" in ev, ev)

print("case 21: the positive control goes BELOW the floor, which is expected arithmetic")
# run 32882865551: detach_messages 1.9 ms against baseline 240.7 and floor 102.2 read 173%.
o = obs(base = 240.7, floor = 10.3, pos = 1.9, repeat = 240.7)
ok, ev = gate(o, "POSITIVE control recovers")
check("the gate passes", ok is True, ev)
check("the reported recovery is capped at 100%", "= 100%" in ev, ev)
check("and the raw ratio is printed beside it", "capped from a raw 104%" in ev, ev)
check("and it says why going below the floor is expected rather than an instrument fault",
      "expected rather than an instrument fault" in ev
      and "updateLayoutIgnorePendingStylesheets" in ev and "scales with DOM size" in ev, ev)
bad = [n for n, ok_, ev_ in C.gates(o) if not ok_]
check("nothing else fails because of it", not bad, str(bad))

print("case 22: the `.katex` comparison is decided on p50, not on the mean")
# run 32882865551 at r500K: baselines 246-250 ms, display 216-217, all 217-218.
same = {"baseline": 248, "baseline_2": 248, "baseline_3": 248, "baseline_4": 248,
        "baseline_5": 248, "baseline_6": 248, "baseline_repeat": 248,
        "content_visibility_katex_display": 216, "content_visibility_katex_all": 217}
o = obs(p50s = same)
ok, ev = gate(o, "adding `.katex` to the selector buys nothing")
check("216 against 217 buys nothing", ok is True, ev)
check("and the evidence says it was decided ON p50, with both medians",
      "ON p50" in ev and "216 ms median rAF gap against 248" in ev and "217 ms against 248" in ev,
      ev)
check("and it prints the mean beside it, as the metric a stall can move",
      "On the MEAN, which one stalled frame can move" in ev, ev)
o = obs(p50s = dict(same, content_visibility_katex_all = 150))
ok, ev = gate(o, "adding `.katex` to the selector buys nothing")
check("216 against 150 does not", ok is False, ev)
check("and it says the spec reading is wrong on the metric a stall cannot corrupt",
      "metric a stall cannot corrupt" in ev and "leaving something on the table" in ev, ev)

print("case 22b: p50 is reported per arm against its own neighbouring baselines")
o = obs(p50s = same)
tbl = C.table(o)
check("the report carries the p50 section",
      "p50, the one statistic a stall cannot move" in tbl, tbl)
check("with each arm's median against its neighbours'",
      "| 216 ms | 248 ms |" in tbl and "| 217 ms | 248 ms |" in tbl, tbl)
check("the per-rung table has a p50 column", "| p50 rAF gap | stalls > 1 s |" in tbl, tbl)
check("and the headline table shows p50 as the deciding column and the mean beside it",
      "cost removed on p50 (decides)" in tbl and "cost removed on the MEAN" in tbl, tbl)

print("case 18: the early and late readings of the reference upper bound disagree by 10x")
o = obs(vh = 10.0, vh_late = 100.0)
tbl = C.table(o)
check("the report carries the early/late comparison", "run TWICE at 500K" in tbl, tbl)
check("and it prints both readings and the factor between them",
      "early **10.0**" in tbl and "late **100.0**" in tbl and "10.0x between them" in tbl, tbl)
check("and it says they DISAGREE and that this is about the session, not the arm",
      "They DISAGREE" in tbl and "statement about this SESSION" in tbl, tbl)
check("and it tells the reader not to use the arm as a bound until they agree",
      "do not use this arm as a bound" in tbl, tbl)
v, why = C.verdict(o)
check("a disagreement is reported, not gated: the verdict still rests on the shippable arm",
      v == "HELPS", f"{v}: {why}")

print("case 18b: the early and late readings agree, killing the arm-order hypothesis")
o = obs()
tbl = C.table(o)
check("the report says they AGREE", "They AGREE" in tbl, tbl)
check("and names the two sessions the question came from",
      "32869180652" in tbl and "32876363634" in tbl, tbl)

print("case 19: the late upper-bound window is printed and is not treated as a candidate")
o = obs()
tbl = C.table(o)
check("the late window appears in the arm table",
      "`katex_root_visibility_hidden_late` **[REFERENCE UPPER BOUND, RE-RUN LATE, NOT SHIPPABLE]**"
      in tbl, tbl)
check("it is not in CANDIDATES", "katex_root_visibility_hidden_late" not in C.CANDIDATES,
      str(C.CANDIDATES))
v, why = C.verdict(o)
check("and it never carries the verdict", "katex_root_visibility_hidden_late" not in why, why)

print("case 20: the table quotes what each arm APPLIED, not only what it is called")
o = obs()
tbl = C.table(o)
check("the arm table has a selector column", "| window | selector | blocked ms/frame (MEAN, "
      "primary) |" in tbl, tbl.splitlines()[:20])
check("and it says the mean is primary and the robust figure sits alongside it",
      "stays the primary metric" in tbl and "never instead of it" in tbl, tbl)
check("the shipped rule's own declaration is quoted",
      "`.katex-display{content-visibility:auto;cont...`" in tbl, tbl)
check("the upper bound's declaration is quoted in full", "`.katex{visibility:hidden}`" in tbl, tbl)
check("the robust, p50 and stall columns are rendered, per rep rather than averaged",
      "| **172.0** | 168.6, 168.6 | 188, 188 | 0, 0 |" in tbl, tbl)
check("the per-rep table carries them too, since a stall lands in ONE repetition",
      "| window | selector | blocked ms/frame per rep | robust per rep | p50 per rep | "
      "stalls > 1 s per rep |" in tbl, tbl)
v, why = C.verdict(o)
check("and the verdict quotes the selector beside the arm name",
      "`content_visibility_katex_display` (`.katex-display{content-visibility:auto;cont...`)"
      in why, why)

print("case 23: THE PRODUCT ARM WINS BIG and is reported as a shippable candidate")
o = obs()
pw = [a for a in o["runs"][2]["payload"]["arms"] if a["name"] == "product_math_block_containment"]
pr0 = o["runs"][2]["payload"]["product_rule"]
precondition("case 23",
             len(pw) == 1 and pw[0]["selector"] == SEL_PROD and pr0["present"] is True
             and pr0["toggle_check"]["moved_fraction"] == 1.0
             and pw[0]["blocked_ms_per_frame"] < 0.5 * 287.6,
             f"window={[a['name'] for a in o['runs'][2]['payload']['arms']]}")
bad = [n for n, ok, ev in C.gates(o) if not ok]
check("every gate passes with the product arm present", not bad, str(bad))
check("the product arm is a CANDIDATE, so it is scored like every other candidate",
      "product_math_block_containment" in C.CANDIDATES, str(C.CANDIDATES))
check("and it is not disqualified", not dq_of(o, "product_math_block_containment"),
      str(dq_of(o, "product_math_block_containment")))
tbl = C.table(o)
check("the table labels it as the product implementation and a shippable candidate",
      "**[PRODUCT IMPLEMENTATION, SHIPPABLE CANDIDATE]**" in tbl, tbl[:200])
check("the product-versus-harness comparison is ONE table, next to the exploratory arm",
      "### The product implementation against the harness arm that bounds it (500K)" in tbl
      and tbl.split("### The product implementation")[1].split("###")[0]
          .count("content_visibility_math_blocks") >= 1, tbl)
check("and that table carries both the mean and p50 columns, since it is scored on both",
      "cost removed on the MEAN | cost removed on p50" in tbl, tbl)
check("the matched selectorText and cssText are quoted VERBATIM, not paraphrased",
      PRODUCT_SELECTOR_TEXT in tbl and PRODUCT_CSS_TEXT in tbl, tbl)
check("the precondition prints the toggle check's off and on samples",
      "attribute off -> ['visible'" in tbl and "attribute on -> ['auto'" in tbl, tbl)
check("and how many stylesheets could not be read, so 'absent' is not confused with 'unreadable'",
      "6 sheets readable, 1 unreadable" in tbl, tbl)
check("the arm is marked scored rather than void", "| scored |" in tbl, tbl)
v, why = C.verdict(o)
check("the verdict reports the product arm's saving on BOTH statistics",
      v == "HELPS" and "product_math_block_containment" in why
      and "on the mean and" in why and "on p50" in why, f"{v}: {why}")
check("and says its precondition holds", "precondition holds" in why, why)
check("and names the build it was measured on",
      "studio-math-block-containment" in why and "3f9c1a2b" in why, why)

print("case 24: product_rule.present FALSE -- the bundle was built without the fix")
o = obs(prod_kw = {"present": False}, prodv = 286.0,
        prod_fired = fired_dropped(".aui-math-block", "contentVisibility", "auto", "visible"),
        prod_took = took(".aui-math-block", 0, 200, before = 61.0, after = 61.0))
pr0 = o["runs"][2]["payload"]["product_rule"]
precondition("case 24",
             pr0["present"] is False and pr0["selector_text"] is None and pr0["matches"] == []
             and pr0["blocks"] > 0
             and o["runs"][0]["payload"]["product_rule"]["present"] is False,
             f"present={pr0['present']} blocks={pr0['blocks']}")
dq = dq_of(o, "product_math_block_containment")
check("the product arm is VOIDED", bool(dq), str(dq))
check("and the words say the BUNDLE was built without the rule",
      "BUILT WITHOUT THE PRODUCT RULE" in (dq or ""), str(dq))
check("and name the attribute it looked for and how many sheets it read",
      "data-math-block-containment" in (dq or "") and "readable" in (dq or "")
      and "could not be read" in (dq or ""), str(dq))
check("and say the toggle was INERT and the null is about the build, not the implementation",
      "INERT" in (dq or "") and "statement about the BUILD" in (dq or ""), str(dq))
check("the fault is named as the missing rule, NOT as a declaration the engine dropped",
      "did not take" not in (dq or ""), str(dq))
bad = [n for n, ok, ev in C.gates(o) if not ok]
check("the RUN is not voided: one wrongly built arm must not throw away the others", not bad,
      str(bad))
v, why = C.verdict(o)
check("the run still produces an answer from the arms that applied their own CSS", v == "HELPS",
      f"{v}: {why}")
check("and the verdict says IN WORDS that the product arm is void and why",
      "IS VOID" in why and "BUILT WITHOUT THE PRODUCT RULE" in why, why)
check("and does not report its null as a measurement", "READS A NULL" not in why, why)
tbl = C.table(o)
check("the table marks the arm VOID and repeats the reason under the comparison",
      "**VOID**" in tbl and "IS VOID AT 500K, AND ITS NUMBER MUST NOT BE READ AS A RESULT" in tbl,
      tbl)
ok, ev = gate(o, "`product_rule` precondition was recorded")
check("the precondition gate PASSES: it gates recording and agreement, never the answer",
      ok is True, ev)
check("and it still says the bundle was built without the fix",
      "NOT FOUND" in ev and "VOID" in ev, ev)

print("case 25: the rule shipped but the toggle moves no computed style at a maths-bearing rung")
o = obs(prod_kw = {"present": True, "moved": 0}, prodv = 285.0)
pr0 = o["runs"][2]["payload"]["product_rule"]
precondition("case 25",
             pr0["present"] is True and pr0["blocks"] > 0
             and pr0["toggle_check"]["moved"] == 0
             and pr0["toggle_check"]["moved_fraction"] == 0.0
             and o["runs"][0]["payload"]["product_rule"]["blocks"] == 0,
             f"present={pr0['present']} toggle={pr0['toggle_check']}")
dq = dq_of(o, "product_math_block_containment")
check("the product arm is VOIDED the same way", bool(dq), str(dq))
check("and the words say the rule is in the bundle but does not reach the blocks",
      "IS IN THE BUNDLE BUT DOES NOT REACH THE BLOCKS" in (dq or ""), str(dq))
check("and they print the off and on samples that prove it",
      "off ['visible'" in (dq or "") and "on ['visible'" in (dq or ""), str(dq))
check("and say this is NOT a missing build, so the two faults are never confused",
      "not a missing build" in (dq or "") and PRODUCT_SELECTOR_TEXT in (dq or ""), str(dq))
bad = [n for n, ok, ev in C.gates(o) if not ok]
check("the RUN is not voided by it", not bad, str(bad))
v, why = C.verdict(o)
check("and the verdict says the arm is void rather than reporting its number",
      "IS VOID" in why and "DOES NOT REACH THE BLOCKS" in why, why)

print("case 26: 0K, zero blocks and the rule PRESENT -- the correct answer, not a failure")
o = obs()
pr_short = o["runs"][0]["payload"]["product_rule"]
precondition("case 26",
             pr_short["blocks"] == 0 and pr_short["present"] is True
             and pr_short["toggle_check"]["sampled"] == 0
             and o["runs"][0]["payload"]["no_scroll_range"] is True,
             f"0K product_rule={pr_short}")
check("zero blocks at a rung with no maths does NOT void anything",
      C._product_void(pr_short) is None, str(C._product_void(pr_short)))
ok, ev = gate(o, "`product_rule` precondition was recorded")
check("the precondition gate passes at 0K", ok is True, ev)
check("and it says zero blocks there is the CORRECT answer rather than staying silent",
      "0K rep 1" in ev and "which is the CORRECT answer for a thread with no maths" in ev, ev)
check("while still reporting that the rule was found at 0K, since one bundle serves every rung",
      ev.split("500K")[0].count("FOUND") >= 1, ev)
bad = [n for n, ok_, ev_ in C.gates(o) if not ok_]
check("and no gate fails because of the empty rung", not bad, str(bad))
tbl = C.table(o)
check("the 0K section reports the product marker count and that the rule is still there",
      "the PRODUCT's own marker class" in tbl
      and "the product rule, which is the same bundle at every rung, was **FOUND** here" in tbl,
      tbl)
check("and says a missing rule there would be the bundle rather than the thread",
      "a missing RULE here would be the bundle, not the thread" in tbl, tbl)

print("case 27: the product arm reads a NULL on a build whose precondition holds")
o = obs(prodv = 280.0)
pr0 = o["runs"][2]["payload"]["product_rule"]
pnull = [a for a in o["runs"][2]["payload"]["arms"]
         if a["name"] == "product_math_block_containment"][0]
precondition("case 27",
             pr0["present"] is True and pr0["toggle_check"]["moved_fraction"] == 1.0
             and pnull["blocked_ms_per_frame"] > 0.9 * 287.6,
             f"present={pr0['present']} bmpf={pnull['blocked_ms_per_frame']}")
check("the arm is NOT disqualified: its precondition holds, so the null is a measurement",
      not dq_of(o, "product_math_block_containment"),
      str(dq_of(o, "product_math_block_containment")))
v, why = C.verdict(o)
check("the verdict reports it as a NULL", "READS A NULL" in why, why)
check("and says explicitly that it is not a win", "NOT a win" in why, why)
check("and distinguishes it from a bundle built without the fix",
      "not a bundle built without the fix" in why, why)
check("and does not call it void", "IS VOID" not in why, why)
check("the headline still rests on the shippable arm", v == "HELPS", f"{v}: {why}")
sv = C._saving(o, "500K", "product_math_block_containment")
check("its measured saving really is under the bar", sv is not None and sv < C.ARM_MIN_SAVING,
      str(sv))

print("case 28: the precondition was never recorded at one rung and repetition")
o = obs()
del o["runs"][2]["payload"]["product_rule"]
precondition("case 28",
             "product_rule" not in o["runs"][2]["payload"]
             and o["runs"][2]["rung"] == "500K" and o["runs"][2]["rep"] == 1
             and "product_rule" in o["runs"][3]["payload"],
             "the 500K rep 1 payload must be the only one missing the record")
ok, ev = gate(o, "`product_rule` precondition was recorded")
check("the gate FAILS, because an unasserted precondition is an instrument fault", ok is False, ev)
check("and it names the rung and the repetition", "500K rep 1: NO `product_rule` record" in ev, ev)
dq = dq_of(o, "product_math_block_containment")
check("and the arm itself is voided, naming the missing record",
      "NO `product_rule` PRECONDITION WAS RECORDED" in (dq or ""), str(dq))
v, why = C.verdict(o)
check("the verdict is INCONCLUSIVE", v == "INCONCLUSIVE", f"{v}: {why}")

print("case 29: the presence census DISAGREES between rungs, which cannot be a fact")
o = obs()
for i in (0, 1):
    o["runs"][i]["payload"]["product_rule"] = product_rule(present = False, blocks = 0,
                                                           katex_display = 0)
precondition("case 29",
             o["runs"][0]["payload"]["product_rule"]["present"] is False
             and o["runs"][2]["payload"]["product_rule"]["present"] is True,
             "0K must say absent while 500K says present")
ok, ev = gate(o, "`product_rule` precondition was recorded")
check("the gate fails", ok is False, ev)
check("and it says one bundle serves every rung, so this is a census fault",
      "DISAGREES BETWEEN RUNGS" in ev and "fault in the census" in ev, ev)
v, why = C.verdict(o)
check("the verdict is INCONCLUSIVE", v == "INCONCLUSIVE", f"{v}: {why}")

print("case 30: the run cannot say which build it measured")
o = obs()
ok, ev = gate(o, "the build under test is identified")
precondition("case 30", ok is True and "3f9c1a2b" in ev,
             "the clean fixture must identify its build before the missing case means anything")
check("a clean run names the repository, the ref and the commit",
      "studio-math-block-containment" in ev and "3f9c1a2b7d4e5f60718293a4b5c6d7e8f9012345" in ev,
      ev)
check("and the report prints the build at the top of the table",
      "**The build under test:" in C.table(o) and "studio-math-block-containment" in C.table(o),
      C.table(o)[:600])
o = obs(subject = {})
ok, ev = gate(o, "the build under test is identified")
check("a run with no clone record fails the gate", ok is False, ev)
check("and says why an unattributable number is not a measurement",
      "nothing measured here can be attributed to a build" in ev
      and "claim about a specific branch" in ev, ev)
v, why = C.verdict(o)
check("the verdict is INCONCLUSIVE", v == "INCONCLUSIVE", f"{v}: {why}")

print("\n" + (f"{FAIL} FAILED" if FAIL else "all cvkatex criteria self-tests passed"))
sys.exit(1 if FAIL else 0)
