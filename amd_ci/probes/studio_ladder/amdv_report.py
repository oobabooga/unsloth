#!/usr/bin/env python3
"""Score amdv_rung_bench payloads: presented frames per phase, per rung, with the null.

Two things this does that the payload cannot do for itself.

1. It cuts the PRESENTED-FRAME series per phase. wkgtk_drive.py records
   GdkFrameClock::after-paint for the whole run as one wall-clock series and splices it in as
   `widget_draws`, with no idea what the page was doing at the time. The scene records phase
   boundaries in wall clock for exactly this reason, so the cut is made here.

2. It scores with studiobench's own `scoring.frames.compute_frame_stats`, so a number from this
   harness and a number from studiobench mean the same thing: a MEASURED refresh budget rather
   than an assumed 60 Hz, `time_in_jank_pct` as a share of wall time rather than of frames, and
   `Measure` instrument floors instead of bare zeros.

The null is not a separate arm here. This is not an A/B: nothing is under test, the rung is the
independent variable, and the control for "did the rung do this" is the SAME rung measured
again. So repetitions of one rung give the spread, and a rung-to-rung difference means nothing
unless it is larger than that spread.

  python3 scripts/amdv_report.py outputs/amdv/*.json
"""
import argparse, json, os, statistics, sys
from pathlib import Path

WS = Path(os.environ.get("UNSLOTH_WORKSPACE", "/mnt/disks/unslothai/daniel3/workspace_37"))

RUNG_ORDER = ["0K", "1K", "10K", "100K", "500K", "1M"]


def draw_timestamps(payload):
    """Reconstruct the absolute after-paint instants from n/first/gaps."""
    wd = payload.get("widget_draws") or {}
    first, gaps = wd.get("first"), wd.get("gaps_ms") or []
    if first is None:
        return []
    ts = [float(first)]
    for g in gaps:
        ts.append(ts[-1] + float(g) / 1000.0)
    return ts


def phase_windows(payload):
    """[(name, start_wall_s, end_wall_s)] from the scene's own wall-clock marks."""
    marks = payload.get("marks") or []
    out = []
    for i, m in enumerate(marks):
        if i + 1 >= len(marks):
            break
        out.append((m["name"], m["wall_ms"] / 1000.0, marks[i + 1]["wall_ms"] / 1000.0))
    return out


def presented(payload):
    """Per-phase presented-frame deltas, in ms, plus the window length."""
    ts = draw_timestamps(payload)
    out = {}
    for name, a, b in phase_windows(payload):
        inside = [t for t in ts if a <= t <= b]
        deltas = [(y - x) * 1000.0 for x, y in zip(inside, inside[1:])]
        out[name] = {"deltas_ms": deltas, "window_ms": (b - a) * 1000.0,
                     "frames": len(inside)}
    return out


def score(deltas, window_ms, frames_mod):
    if frames_mod is None:
        return None
    return frames_mod.compute_frame_stats(deltas, window_ms)


def load_frames_module(sb_root: Path):
    sys.path.insert(0, str(sb_root))
    try:
        from tests.studio.studiobench.scoring import frames as frames_mod
        return frames_mod
    except Exception as e:  # noqa: BLE001
        print(f"note: studiobench scoring unavailable ({type(e).__name__}: {e}); "
              f"falling back to raw percentiles", file = sys.stderr)
        return None


def d(measure):
    return measure.display() if measure is not None and hasattr(measure, "display") else "n/a"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("payloads", nargs="+")
    ap.add_argument("--sb-root", default=str(WS / "wt_mainchk"))
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    frames_mod = load_frames_module(Path(args.sb_root))

    rows = []
    for p in args.payloads:
        payload = json.loads(Path(p).read_text())
        meta = payload.get("run_meta") or {}
        rung, rep = meta.get("rung", "?"), meta.get("rep", "?")
        if not payload.get("ok"):
            rows.append({"rung": rung, "rep": rep, "ok": False, "file": p,
                         "error": str(payload.get("error"))[:300],
                         "phase_reached": payload.get("phase")})
            continue
        pres = presented(payload)
        by_phase = {ph["phase"]: ph for ph in payload.get("phases") or []}
        row = {"rung": rung, "rep": rep, "ok": True, "file": p,
               "engine": payload.get("ua", "")[:60],
               "webkit_gtk": (payload.get("engine_probe") or {}).get("is_webkit_gtk_ua"),
               "bundle": meta.get("bundle_hash"),
               "seeded_messages": (meta.get("seeded") or {}).get("messages"),
               "seeded_chars": (meta.get("seeded") or {}).get("seeded_chars"),
               "read_back": meta.get("read_back_messages"),
               "mount_ms": (payload.get("mount") or {}).get("ms"),
               "mount_by": (payload.get("mount") or {}).get("by"),
               "elements": ((payload.get("final") or {}).get("elements")),
               "messages_dom": ((payload.get("final") or {}).get("messages")),
               "clamp": payload.get("clamp"),
               "first_token_ms": payload.get("first_token_ms"),
               "phases": {}}
        for name in ("idle", "scroll", "stream", "recover"):
            ph = by_phase.get(name) or {}
            pr = pres.get(name) or {"deltas_ms": [], "window_ms": 0.0, "frames": 0}
            st = score(pr["deltas_ms"], pr["window_ms"], frames_mod)
            raf = ph.get("raf") or {}
            rst = score(ph.get("raf_gaps_ms") or [], ph.get("elapsed_ms") or 0, frames_mod)
            row["phases"][name] = {
                "window_ms": round(pr["window_ms"]),
                "presented_frames": pr["frames"],
                "presented_fps": (None if not pr["window_ms"]
                                  else round(1000.0 * pr["frames"] / pr["window_ms"], 1)),
                "presented_stats": st,
                "raf_fps_p50": None if not raf.get("fps_p50") else round(raf["fps_p50"], 1),
                "raf_stats": rst,
                "busy": ph.get("busy"),
                "census": ph.get("census"),
            }
        rows.append(row)

    ok = [r for r in rows if r["ok"]]
    bad = [r for r in rows if not r["ok"]]

    print("\n# Studio in real WebKitGTK, by ladder rung\n")
    for r in rows:
        if not r["ok"]:
            print(f"  {r['rung']} rep {r['rep']}: FAILED in phase {r['phase_reached']!r} - "
                  f"{r['error']}")
    if bad:
        print()

    if ok:
        engines = {r["webkit_gtk"] for r in ok}
        bundles = {r["bundle"] for r in ok}
        print(f"engine is WebKitGTK on every run: {engines == {True}}")
        print(f"one bundle across every run: {len(bundles) == 1} {sorted(b or '' for b in bundles)}")
        print()

    print("| rung | rep | seeded msgs | DOM elements | mount ms | "
          "idle fps | scroll fps | stream fps | idle busy | scroll busy | stream busy |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in sorted(ok, key = lambda r: (RUNG_ORDER.index(r["rung"])
                                         if r["rung"] in RUNG_ORDER else 99, str(r["rep"]))):
        ph = r["phases"]
        def fps(n):
            return "-" if ph[n]["presented_fps"] is None else f"{ph[n]['presented_fps']:.1f}"
        def busy(n):
            b = ph[n]["busy"] or {}
            return "null" if b.get("busy_pct") is None else f"{b['busy_pct']:.1f}%"
        print(f"| {r['rung']} | {r['rep']} | {r['seeded_messages']} | {r['elements']:,} | "
              f"{r['mount_ms']:,} | {fps('idle')} | {fps('scroll')} | {fps('stream')} | "
              f"{busy('idle')} | {busy('scroll')} | {busy('stream')} |")

    print("\n## Presented-frame detail, scored by studiobench scoring/frames.py\n")
    print("| rung | rep | phase | frames | fps | p50 | p95 | max | time in jank | jank index |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in sorted(ok, key = lambda r: (RUNG_ORDER.index(r["rung"])
                                         if r["rung"] in RUNG_ORDER else 99, str(r["rep"]))):
        for name in ("idle", "scroll", "stream", "recover"):
            p = r["phases"][name]
            st = p["presented_stats"]
            if st is None:
                print(f"| {r['rung']} | {r['rep']} | {name} | {p['presented_frames']} | "
                      f"{p['presented_fps']} | - | - | - | - | - |")
                continue
            print(f"| {r['rung']} | {r['rep']} | {name} | {st.frames_total} | "
                  f"{d(st.effective_fps)} | {d(st.p50_frame_ms)} | {d(st.p95_frame_ms)} | "
                  f"{d(st.max_frame_ms)} | {d(st.time_in_jank_pct)} | {d(st.jank_index)} |")

    # ── the null: the same rung, measured again ──
    print("\n## The null: repeat spread at the same rung\n")
    by_rung = {}
    for r in ok:
        by_rung.setdefault(r["rung"], []).append(r)
    any_null = False
    for rung, rs in sorted(by_rung.items(),
                           key = lambda kv: RUNG_ORDER.index(kv[0]) if kv[0] in RUNG_ORDER else 99):
        if len(rs) < 2:
            continue
        any_null = True
        for name in ("idle", "scroll", "stream"):
            vals = [r["phases"][name]["presented_fps"] for r in rs
                    if r["phases"][name]["presented_fps"] is not None]
            if len(vals) < 2:
                continue
            spread = max(vals) - min(vals)
            print(f"  {rung} {name}: {', '.join(f'{v:.1f}' for v in vals)} fps  "
                  f"-> spread {spread:.1f} fps "
                  f"({100.0 * spread / max(vals):.1f}% of the best)")
    if not any_null:
        print("  NOT ESTABLISHED. No rung was measured twice, so no rung-to-rung difference "
              "in this report has a floor to be compared against, and none of them can be "
              "called real.")

    if args.json_out:
        def enc(o):
            return o.to_json() if hasattr(o, "to_json") else str(o)
        Path(args.json_out).write_text(json.dumps(rows, default = enc, indent = 1))
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
