#!/usr/bin/env python3
"""Judge helpers for the Windows items of probes/r8_win.py (imported by criteria/r8_measure.py). gates() proves each
listed item ran; table() carries the numbers and one verdict line per item against its pre-registered kill line."""
from __future__ import annotations

import statistics as st

RATE_TRAP_GBPS = 0.53  # rocm-systems #11692 hipMemsetAsync past the carve-out


def _ok(rs):
    return [r for r in rs or [] if r.get("rc") == 0 and r.get("verdict") in (None, "ok") and not r.get("missing")]


def gates(h: dict, items: list) -> list:
    out = []
    if "wsmoke" in items:
        out.append(("wsmoke gfx1151", bool(h.get("wsmoke_ok")), str([(r.get("tree"), r.get("arch"), r.get("gemm_tflops"))
                                                                         for r in h.get("wsmoke") or []])[:300]))
    if "sdma" in items:
        rs = _ok(h.get("sdma"))
        full = [r for r in rs if len(r.get("cells") or {}) >= 5 and all(c.get("iters") == 50 for c in r["cells"].values())]
        out.append(("sdma 4 processes x 5 cells x 50", len(full) == 4, f"{len(full)} complete of {len(h.get('sdma') or [])}"))
    if "sdma_renders" in items:
        skip = h.get("sdma_renders_skip")
        n = len([r for r in h.get("sdma_renders") or [] if r.get("verdict") == "ok"])
        out.append(("sdma_renders ran or gated off", bool(skip) or n >= 6, skip or f"{n} ok arms"))
    if "memset" in items:
        rs = [r for r in _ok(h.get("memset")) if (r.get("zero_") or {}).get("med_ms")]
        out.append(("memset 4 processes", len(rs) == 4, str(len(rs))))
    if "knee3" in items:
        n = sum(1 for r in h.get("knee3") or [] if r.get("gemm_tflops"))
        out.append(("knee3 points (of 108)", n >= 100, str(n)))
    if "knee_render" in items:
        skip = h.get("knee_render_skip")
        n = len([r for r in h.get("knee_render") or [] if r.get("verdict") == "ok"])
        out.append(("knee_render ran or gated off", bool(skip) or n >= 6, skip or f"{n} ok arms"))
    if "memprof" in items:
        m = h.get("memprof") or {}
        out.append(("memprof render recorded", m.get("verdict") in ("ok", "error") and bool(m.get("renders") or m.get("error")),
                    f"{m.get('verdict')} prof {'error' if (m.get('prof_whole') or {}).get('error') else 'ok' if m.get('prof_whole') else 'absent'}"))
    if "c153" in items:
        skip = h.get("c153_skip")
        n = len([r for r in h.get("c153") or [] if r.get("verdict") == "ok"])
        out.append(("c153 ran or not in this job", bool(skip) or n >= 6, skip or f"{n} ok arms"))
    return out


def _med(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return st.median(xs) if xs else None


def _render_s(r):
    """median timed render wall s of one arm record."""
    return _med([x.get("wall_s") for x in r.get("renders") or []])


def _abba(rows, a_name, b_name):
    a = [_render_s(r) for r in rows if r.get("arm") == a_name and r.get("verdict") == "ok"]
    b = [_render_s(r) for r in rows if r.get("arm") == b_name and r.get("verdict") == "ok"]
    a, b = [x for x in a if x], [x for x in b if x]
    if not a or not b:
        return None
    ma, mb = st.median(a), st.median(b)
    return {"a": a, "b": b, "med_a": round(ma, 3), "med_b": round(mb, 3), "delta": round(mb / ma - 1, 4),
            "aa_a": round((max(a) - min(a)) / ma, 4), "aa_b": round((max(b) - min(b)) / mb, 4)}


def table(h: dict) -> list:
    L = []
    for r in h.get("wsmoke") or []:
        L.append(f"wsmoke {r.get('tree')}: torch {r.get('torch')} hip {r.get('hip')} arch {r.get('arch')} total "
                 f"{r.get('total_gib')} GiB free0 {r.get('free0_gib')} free after 4x1GiB {r.get('free_4x1g_gib')} "
                 f"trusted {r.get('trusted_free_gib')} wddm used {r.get('wddm_used_gib')} ({r.get('wddm_s')} s) "
                 f"gemm {r.get('gemm_tflops')} TFLOP/s")
    if h.get("sdma"):
        tot = {"U": {}, "D": {}}
        for r in h["sdma"]:
            for cell, c in (r.get("cells") or {}).items():
                t = tot[r.get("arm", "?")].setdefault(cell, [0, 0])
                t[0] += int(c.get("wrong") or 0)
                t[1] += int(c.get("iters") or 0)
            L.append(f"sdma {r.get('i')} {r.get('arm')}: writer {r.get('writer')} behind {r.get('behind')} " + ", ".join(
                f"{k} {v.get('outcomes')}" for k, v in (r.get("cells") or {}).items()) + (f" ERROR {r.get('error')[:200]}" if r.get('error') else ""))
        u_bad = sum(v[0] for k, v in tot["U"].items() if not k.endswith("-norec"))
        L.append(f"sdma totals wrong/iters: U {tot['U']} D {tot['D']}")
        L.append(f"C6-W2 verdict: {'KILL (0 stale with PAL_DISABLE_SDMA unset: drop)' if u_bad == 0 else f'RACE REPRODUCED ({u_bad} wrong with SDMA on): cost renders decide env vs call-site fix'}")
    if h.get("sdma_renders_skip"):
        L.append(f"sdma_renders: {h['sdma_renders_skip']}")
    if h.get("sdma_renders"):
        ab = _abba(h["sdma_renders"], "U", "D")
        md5 = {r.get("arm"): sorted({x.get("md5") for x in r.get("renders") or []}) for r in h["sdma_renders"]}
        L.append(f"sdma_renders U vs D (PAL_DISABLE_SDMA=1): {ab}; md5 by arm {md5}")
        if ab:
            L.append(f"C6-W2 cost verdict: env costs {ab['delta'] * 100:.1f}% (A/A U {ab['aa_a'] * 100:.1f}%): "
                     + ("fix the call sites instead (over 2% and outside A/A)" if ab["delta"] > max(0.02, ab["aa_a"], ab["aa_b"])
                        else "env cost 2% or less, or inside A/A: env acceptable"))
    if h.get("memset"):
        for r in h["memset"]:
            L.append(f"memset ballast {r.get('ballast_gib')} GiB: zero_ {(r.get('zero_') or {}).get('gbps')} GB/s, "
                     f"hipMemsetAsync {(r.get('hipMemsetAsync') or {}).get('gbps')} GB/s rc {r.get('hipMemsetAsync_rc')}, "
                     f"D32 {(r.get('hipMemsetD32Async') or {}).get('gbps')} GB/s; lib {r.get('hip_lib')} "
                     + (f"ERROR {r.get('error')[:200]}" if r.get("error") else ""))
    ks = knee_summary(h.get("knee3"))
    if ks["points"]:
        for p, arms in ks["points"].items():
            L.append(f"knee point {p} {POINT_NAMES.get(p)}: " + ", ".join(
                f"{a} med {v['med']} [{v['min']}-{v['max']}]" for a, v in arms.items()))
        L.append(f"knee wins (median above base max and +5%): {ks['wins']}")
        L.append("C6-W1 / C15 #4 knee verdict: " + ("KILL (every env arm inside the base spread)" if not any(
            len(v) >= 3 for v in ks["wins"].values()) else "env arm beats base at 3+ points: render gate decides"))
        drops = [(r.get("point"), r.get("free_drop_gib"), r.get("ballast_gib"), r.get("counters_gib"))
                 for r in h["knee3"] if r.get("arm") == "base" and r.get("rep") == 0]
        L.append(f"knee base rep0 free drop vs held (C15-3 2x evidence): {drops}"[:3000])
    if h.get("knee_render_skip"):
        L.append(f"knee_render: {h['knee_render_skip']}")
    if h.get("knee_render"):
        ab = _abba(h["knee_render"], "B", "A")
        L.append(f"knee_render base vs {h['knee_render'][0].get('env_arm')}: {ab}")
        if ab:
            L.append(f"C6-W1 render verdict: {-ab['delta'] * 100:.1f}% faster; " + (
                "KILL (under 5% Qwen-2.1 render or inside A/A)" if -ab["delta"] < max(0.05, ab["aa_a"]) else "PASS"))
    m = h.get("memprof") or {}
    if m:
        pw = m.get("prof_whole") or {}
        warm = _render_s(m)
        cls = pw.get("classes") or {}
        ms_b = sum(v.get("bytes", 0) for k, v in cls.items() if k.startswith("memset"))
        ms_ms = sum(v.get("ms", 0) for k, v in cls.items() if k.startswith("memset:device"))
        ms_n = {k: v.get("n") for k, v in cls.items() if k.startswith("memset")}
        L.append(f"memprof Wan 5B: warm render {warm} s, profiled render {pw.get('wall_s')} s, memset events {ms_n}, "
                 f"device memset {round(ms_ms, 2)} ms, bytes {ms_b}; classes {cls}; names {pw.get('names')}"[:3000])
        if pw.get("error"):
            L.append(f"memprof profiler error: {pw['error'][-400:]}")
        if warm:
            share = (ms_b / (RATE_TRAP_GBPS * 1e9)) / warm if ms_b else None
            L.append("C15-2 verdict: " + (f"memset bytes / 0.53 GB/s = {share * 100:.2f}% of the render: "
                                          + ("KILL (under 2%)" if share < 0.02 else "OVER 2%: keep the torch-bump gate")
                                          if share is not None else
                                          f"no memset bytes in the trace (events {ms_n}): share by count only"))
    if h.get("c153_skip"):
        L.append(f"c153: {h['c153_skip']}")
    if h.get("c153"):
        ab = _abba(h["c153"], "M", "H")
        for r in h["c153"]:
            L.append(f"c153 {r.get('i')} {r.get('arm')}: {r.get('verdict')} load {r.get('load_s')} s status "
                     f"{r.get('status')} memlog_load {r.get('memlog_load')} renders "
                     f"{[(x.get('wall_s'), x.get('md5'), (x.get('memlog') or {}).get('driver_free_gib')) for x in r.get('renders') or []]} "
                     f"status_end {r.get('status_end')}"[:2500])
        L.append(f"c153 main vs head: {ab}")
    return L


POINT_NAMES = {0: "24G slab", 1: "24G x1GiB", 2: "24G x64MiB", 3: "24G x8MiB", 4: "40G slab", 5: "40G x1GiB",
               6: "40G x64MiB", 7: "40G x8MiB", 8: "40G x1GiB operands first"}


def knee_summary(rows) -> dict:
    by: dict = {}
    for r in rows or []:
        if r.get("gemm_tflops"):
            by.setdefault(r["point"], {}).setdefault(r["arm"], []).append(r["gemm_tflops"])
    summ, wins = {}, {a: [] for a in ("sub", "ipc", "both")}
    for p, arms in sorted(by.items()):
        summ[p] = {a: {"med": st.median(v), "min": min(v), "max": max(v), "n": len(v)} for a, v in arms.items()}
        b = summ[p].get("base")
        if not b:
            continue
        for a in wins:
            s = summ[p].get(a)
            if s and s["med"] > b["max"] and s["med"] >= 1.05 * b["med"]:
                wins[a].append(p)
    return {"points": summ, "wins": wins}
