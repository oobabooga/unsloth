#!/usr/bin/env python3
"""Judge for probes/r9_items.py (smoke, c63, m4, c61_knee, c61_krea). Gates prove each listed item ran (a gated-off
conditional item counts as ran when its reason is recorded); table() carries the numbers plus one VERDICT line per
item against its pre-registered kill line."""
from __future__ import annotations

import statistics as st

C63_SHARE_KILL = 0.01
KNEE_KILL = 29.0
KREA_GAIN_KILL = 0.10
M4_COST_KILL = 0.15
CRASH_RC_WIN = {3221225477: "0xC0000005 access violation", 3221226505: "0xC0000409 stack buffer overrun",
                3221225786: "0xC000013A", 134: "abort"}


def _f(x, d = 3):
    return "n/a" if x is None else f"{x:.{d}f}" if isinstance(x, float) else str(x)


def _crashed(r) -> bool:
    return r.get("rc") not in (0, None) or r.get("verdict") not in ("ok",)


def gates(h: dict, items: list) -> list:
    out = []
    if "smoke" in items:
        out.append(("smoke gfx1151", bool(h.get("smoke_ok")), str((h.get("smoke") or {}).get("arch"))))
    if "c63" in items:
        c = h.get("c63") or {}
        n = len(c.get("crash") or [])
        out.append(("c63 crash sweep ran (10 processes)", n == 10, f"{n} processes"))
        sh = [r for r in c.get("share") or [] if r.get("verdict") == "ok"]
        out.append(("c63 share measured (or recorded NOT RUN)", bool(sh) or bool(c.get("share_not_run")),
                    c.get("share_not_run") or f"{len(sh)} ok, hook calls {[r.get('hook_calls_last') for r in sh]}"))
    if "m4" in items:
        m = h.get("m4") or {}
        ok = [p for p in m.get("procs") or [] if p.get("verdict") == "ok"]
        eng = all(all(e.get("inside", {}).get("deterministic") == (p["arm"] == "D") for e in recs)
                  for p in ok for recs in (p.get("images") or {}).values())
        out.append(("m4 bf16 U and D processes ok", sum(1 for p in ok if p["dtype"] == "bf16") >= 4 or
                    bool(m.get("not_run")), m.get("not_run") or f"{len(ok)} ok"))
        out.append(("m4 engagement: deterministic flag in force inside D encodes only", eng, ""))
    if "c61_knee" in items:
        rows = h.get("c61_knee") or []
        ok = [r for r in rows if r.get("gemm_tflops")]
        out.append(("c61 knee arms measured (15)", len(ok) >= 13, f"{len(ok)} of {len(rows)}"))
        bal = [r.get("ballast_gib") for r in ok]
        out.append(("c61 ballast held 42 GiB in every arm", bool(bal) and min(bal) >= 41.9, f"min {min(bal or [0])}"))
    if "c63b" in items:
        c = h.get("c63b") or {}
        out.append(("c63b ran (30 conv processes, 3 encodes)", len(c.get("crash") or []) == 30 and
                    len(c.get("share") or []) == 3, f"{len(c.get('crash') or [])} / {len(c.get('share') or [])}"))
    if "c153" in items:
        c = h.get("c153") or {}
        ok = [r for r in c.get("arms") or [] if r.get("med_s")]
        out.append(("c153 decode arms measured (10)", len(ok) >= 8, f"{len(ok)} ok"))
        eng = [r.get("coverage") for r in ok if r.get("arm_kind") in ("T", "G")]
        out.append(("c153 Triton arms engaged (coverage > 0)", bool(eng) and all(x and x > 0 for x in eng), str(eng)))
        g = [(r.get("tables_facts") or {}).get("tables") for r in ok if r.get("arm_kind") == "G"]
        out.append(("c153 G arms ran with gfx1100 tables", bool(g) and all(x == "gfx1100" for x in g), str(g)))
    if "c61_krea" in items:
        out.append(("c61_krea ran or gated off", bool(h.get("c61_krea_skip")) or len(
            [r for r in h.get("c61_krea") or [] if r.get("renders")]) == 4, h.get("c61_krea_skip") or
            f"{len(h.get('c61_krea') or [])} arms"))
    return out


def _c63_lines(h):
    c = h.get("c63") or {}
    if not c:
        return []
    L = ["C6-3 patch-embed Conv3d (fp32, (N,3,2,16,16), k = stride):"]
    crashes = []
    for r in c.get("crash") or []:
        calls = r.get("calls") or []
        walls = [x.get("wall_ms") for x in calls if x.get("wall_ms") is not None]
        bad = _crashed(r) or len(walls) < 3 or not all(x.get("finite") for x in calls)
        if bad:
            crashes.append((r.get("n"), r.get("rc"), CRASH_RC_WIN.get(r.get("rc")), r.get("verdict"),
                            (r.get("error") or "")[-160:]))
        L.append(f"  sweep {r.get('sweep')} N {r.get('n')}: rc {r.get('rc')} verdict {r.get('verdict')} calls "
                 f"{len(walls)}/3 ms {walls} linear ms {[x.get('linear_ms') for x in calls]} max|conv-linear| "
                 f"{max([x.get('maxabs_vs_linear') or 0 for x in calls] or [None])} conv vs cpu64 "
                 f"{(calls[0] if calls else {}).get('maxabs_conv_vs_cpu64')}")
    shares = []
    for r in c.get("share") or []:
        if r.get("verdict") == "ok":
            shares.append(max(r["share_bf16"], r["share_fp32_conv_over_encode"]))
        L.append(f"  share N {r.get('n')} grid {r.get('grid')}: encode ms {r.get('encode_ms_med')} patch_embed ms "
                 f"{r.get('patch_embed_ms_med')} ({r.get('patch_embed_proj')}, hooks {r.get('hook_calls_last')}) share "
                 f"{r.get('share_bf16')}; fp32 conv3d alone {r.get('fp32_conv3d_ms')} ms = "
                 f"{r.get('share_fp32_conv_over_encode')} of encode; {r.get('verdict')} {(r.get('error') or '')[-200:]}")
    if c.get("share_not_run"):
        L.append(f"  share NOT RUN: {c['share_not_run']}")
    L.append(f"  crashes / wrong: {crashes or 'none'}")
    if crashes:
        v = "KEEP (crash or non-finite at N " + ",".join(str(x[0]) for x in crashes) + \
            "): geometry-gated F.linear swap, gated on allclose"
    elif not shares:
        v = "INCOMPLETE (no crash; share not measured)"
    elif max(shares) < C63_SHARE_KILL:
        v = f"KILL (no crash, patch-embed share max {max(shares):.4f} < 1% of encode)"
    else:
        v = f"KEEP for speed only (no crash, patch-embed share max {max(shares):.4f} >= 1%)"
    L.append(f"  VERDICT C6-3: {v}")
    return L


def _c63b_lines(h):
    c = h.get("c63b") or {}
    if not c:
        return []
    L = ["C6-3 follow-up: patch-embed Conv3d alone by dtype (1024 ch), then the vision encode at N=16384:"]
    bad = []
    for r in c.get("crash") or []:
        calls = r.get("calls") or []
        ok = not _crashed(r) and len(calls) == 3 and all(x.get("finite") for x in calls)
        if not ok:
            bad.append((r.get("dtype"), r.get("n"), r.get("rc"), (r.get("error") or "")[-160:].replace("\n", " ")))
        L.append(f"  sweep {r.get('sweep')} {r.get('dtype')} N {r.get('n')}: rc {r.get('rc')} {r.get('verdict')} "
                 f"ms {[x.get('wall_ms') for x in calls]} max|conv-linear| "
                 f"{max([x.get('maxabs_vs_linear') or 0 for x in calls] or [None])}")
    for r in c.get("share") or []:
        L.append(f"  encode N=16384 AMD_SERIALIZE_KERNEL={r.get('serialize')}: {r.get('verdict')} encode ms "
                 f"{r.get('encode_ms_med')} share {r.get('share_bf16')} err {(r.get('error') or '')[-300:]}")
    enc_bad = [r for r in c.get("share") or [] if r.get("verdict") != "ok"]
    L.append(f"  conv3d alone failures: {bad or 'none'}; encode failures {len(enc_bad)}/{len(c.get('share') or [])}")
    v = ("KEEP (correctness): " + ("conv3d alone fails at " + str(sorted({(b[0], b[1]) for b in bad})) if bad else
                                   "the conv3d alone passes but the encode fails at N=16384")) if bad or enc_bad else \
        "KILL (the Linux bf16 failure does not reproduce: every conv3d and encode ran clean)"
    L.append(f"  VERDICT C6-3 follow-up: {v}")
    return L


def _m4_lines(h):
    m = h.get("m4") or {}
    if not m:
        return []
    L = [f"M4 FLUX.3 klein ref vae.encode determinism (prefetch {m.get('prefetch', {}).get('wall_s')} s):"]
    if m.get("not_run"):
        return L + [f"  NOT RUN: {m['not_run']}", "  VERDICT M4: NOT RUN"]
    shas: dict = {}
    ms: dict = {}
    for p in m.get("procs") or []:
        if p.get("verdict") != "ok":
            L.append(f"  proc {p.get('i')} {p.get('dtype')} {p.get('arm')}: {p.get('verdict')} rc {p.get('rc')} "
                     f"{(p.get('error') or '')[-200:]}")
            continue
        for img, recs in (p.get("images") or {}).items():
            k = (p["dtype"], p["arm"], img)
            shas.setdefault(k, []).append([e["sha"] for e in recs])
            ms.setdefault((p["dtype"], p["arm"]), []).extend(e["ms"] for e in recs if not e["warm"])
            L.append(f"  proc {p['i']} {p['dtype']} {p['arm']} {img}: distinct sha in-process "
                     f"{len(set(e['sha'] for e in recs))}/{len(recs)} max|d| vs warm "
                     f"{max([e.get('maxabs_vs_warm') or 0 for e in recs])} ms {[e['ms'] for e in recs]}")
    verdicts = []
    for dt in ("bf16", "fp32"):
        u_all = {s for (d, a, _i), lst in shas.items() if d == dt and a == "U" for row in lst for s in row}
        u_imgs = {i for (d, a, i) in shas if d == dt and a == "U"}
        u_distinct = {i: len({s for row in shas.get((dt, "U", i), []) for s in row}) for i in u_imgs}
        d_distinct = {i: len({s for row in shas.get((dt, "D", i), []) for s in row}) for i in u_imgs}
        if not u_all:
            continue
        mu, md = st.median(ms.get((dt, "U")) or [0]) or None, st.median(ms.get((dt, "D")) or [0]) or None
        cost = (md / mu - 1) if mu and md else None
        su = ms.get((dt, "U")) or []
        aa = (max(su) - min(su)) / st.median(su) if su else None
        L.append(f"  {dt}: distinct latents across all U encodes {u_distinct}, across all D encodes {d_distinct}; "
                 f"encode ms median U {_f(mu)} D {_f(md)} cost {_f(cost)} (U spread {_f(aa)})")
        if all(v == 1 for v in u_distinct.values()):
            verdicts.append(f"{dt} KILL (U latents already bit-equal across encodes and processes)")
        elif cost is not None and cost > M4_COST_KILL:
            verdicts.append(f"{dt} KILL (deterministic scope costs {cost:.1%} > 15%)")
        elif not all(v == 1 for v in d_distinct.values()):
            verdicts.append(f"{dt} KILL (D still not bit-equal: the scope does not fix it)")
        else:
            verdicts.append(f"{dt} KEEP (U drifts, D bit-equal, cost {_f(cost)})")
    L.append(f"  VERDICT M4: {'; '.join(verdicts) or 'INCOMPLETE'}")
    return L


def knee_summary(rows) -> dict:
    by: dict = {}
    for r in rows or []:
        if r.get("gemm_tflops"):
            by.setdefault(r["arm"], []).append(r["gemm_tflops"])
    return {a: {"med": st.median(v), "min": min(v), "max": max(v), "n": len(v), "vals": v} for a, v in by.items()}


def _knee_lines(h):
    rows = h.get("c61_knee") or []
    if not rows:
        return []
    s = knee_summary(rows)
    L = ["C6-1 knee (42 GiB ballast in 1 GiB chunks, bf16 4096^3 GEMM TFLOP/s, 3 reps):"]
    for arm in ("base", "opfirst", "r2", "r4", "r8"):
        d = s.get(arm) or {}
        eng = [(r.get("in_reserve") or {}) for r in rows if r.get("arm") == arm]
        L.append(f"  {arm}: med {_f(d.get('med'))} vals {d.get('vals')} d2d GB/s "
                 f"{[r.get('d2d_gbps') for r in rows if r.get('arm') == arm]} in_reserve {eng if arm[0] == 'r' else '-'}"
                 f" ballast_in_reserve {[r.get('ballast_in_reserve') for r in rows if r.get('arm') == arm]}"
                 f" empty_cache calls {[r.get('empty_cache_calls') for r in rows if r.get('arm') == arm]}")
    errs = [(r.get("arm"), r.get("rep"), r.get("rc"), (r.get("error") or "")[-150:]) for r in rows
            if r.get("verdict") != "ok"]
    if errs:
        L.append(f"  errors: {errs}")
    passing = [a for a in ("r2", "r4", "r8") if (s.get(a) or {}).get("med", 0) >= KNEE_KILL]
    v = f"PASS to render gate (reserve {', '.join(passing)} at or above {KNEE_KILL})" if passing else \
        f"KILL (every reserve under {KNEE_KILL} TFLOP/s after the ballast)"
    L.append(f"  VERDICT C6-1 knee: {v}")
    return L


def _krea_lines(h):
    if "c61_krea_gate" not in h:
        return []
    if h.get("c61_krea_skip"):
        return [f"C6-1 Krea-2: {h['c61_krea_skip']}", "  VERDICT C6-1 Krea-2: NOT RUN (knee kill)"]
    rows = h.get("c61_krea") or []
    g = h.get("c61_krea_gate") or {}
    L = [f"C6-1 Krea-2-Turbo 1024^2, A = no reserve, B = R9_RESERVE_GIB {g.get('gib')} (empty_cache no-op):"]
    walls: dict = {}
    places = {}
    for r in rows:
        w = [x.get("wall_s") for x in r.get("renders") or [] if x.get("wall_s")]
        walls.setdefault(r["arm"], []).append(st.median(w) if w else None)
        stt = r.get("status") or {}
        places[r["arm"]] = (stt.get("memory_mode"), stt.get("offload_policy"), stt.get("speed_mode"))
        L.append(f"  {r['i']} {r['arm']}: load {r.get('load_s')} s render wall {w} verdict {r.get('verdict')} "
                 f"reserve {r.get('r9_reserve')} calls {r.get('r9_calls')} status {places[r['arm']]}")
    a, b = [x for x in walls.get("A", []) if x], [x for x in walls.get("B", []) if x]
    if len(a) < 2 or len(b) < 2:
        return L + ["  VERDICT C6-1 Krea-2: INCOMPLETE (fewer than 2 arms each)"]
    ma, mb = st.median(a), st.median(b)
    gain = ma / mb - 1
    aa = abs(a[0] - a[1]) / ma
    same = len(set(places.values())) == 1
    L.append(f"  median render A {ma:.2f} s B {mb:.2f} s: B faster by {gain:.1%} (A/A {aa:.1%}); same placement {same}")
    v = "VOID (placement differs between arms)" if not same else \
        f"KILL (under 10% faster: {gain:.1%})" if gain < KREA_GAIN_KILL else \
        f"KILL (gain {gain:.1%} inside A/A {aa:.1%})" if gain <= aa else f"KEEP ({gain:.1%} faster, A/A {aa:.1%})"
    L.append(f"  VERDICT C6-1 Krea-2: {v}")
    return L


def _c153_lines(h):
    c = h.get("c153") or {}
    if not c:
        return []
    L = ["C15-3 vendored aiter Triton conv3d, S = MIOpen, T = PR #6973 gfx1151 tables, G = gfx1100 tables:"]
    by: dict = {}
    for r in c.get("arms") or []:
        k = (r.get("mode"), r.get("arm_kind"))
        L.append(f"  {r.get('mode')} {r.get('tag')}: med_s {_f(r.get('med_s'))} s {r.get('s')} first {_f(r.get('first_s'))} "
                 f"peak {_f(r.get('peak_gib'))} GiB coverage {_f(r.get('coverage'))} counts {r.get('counts')} tables "
                 f"{(r.get('tables_facts') or {}).get('any_1x1x1_in_force')} err {(r.get('error') or r.get('not_run') or '')[:200]}")
        if r.get("med_s"):
            by.setdefault(k, []).append(r)
    lp = c.get("lpips") or {}
    for mode, x in lp.items():
        L.append(f"  LPIPS {mode} vs S1: {x.get('rows')} {x.get('verdict')} {(x.get('error') or '')[-200:]}")

    def med(mode, kind, key = "med_s"):
        v = [r[key] for r in by.get((mode, kind), []) if r.get(key) is not None]
        return st.median(v) if v else None

    def lp_upper(mode, kind):
        rows = (lp.get(mode) or {}).get("rows") or {}
        ups = [v["ci95"][1] for t, v in rows.items() if t.startswith(kind) and v.get("ci95")]
        return max(ups) if ups else None
    out = []
    for mode, name, kill, other in (("decode", "Wan 5B decode", 0.10, "G"), ("ltxdecode", "LTX-2.5 decode", 0.15, "S")):
        t, o, s_ = med(mode, "T"), med(mode, other), med(mode, "S")
        cov = [r.get("coverage") for r in by.get((mode, "T"), [])]
        if not t or not o:
            out.append(f"{name} INCOMPLETE")
            continue
        gain = o / t - 1
        aa = [r["med_s"] for r in by.get((mode, "T"), [])]
        aa = (max(aa) - min(aa)) / st.median(aa) if len(aa) > 1 else None
        floor = lp_upper(mode, "S") or 0.0
        up = lp_upper(mode, "T")
        dpeak = (med(mode, "T", "peak_gib") or 0) - (med(mode, other, "peak_gib") or 0)
        dpeak_s = (med(mode, "T", "peak_gib") or 0) - (med(mode, "S", "peak_gib") or 0)
        L.append(f"  {name}: T {t:.2f} s vs {other} {o:.2f} s = {gain:.1%} faster (T A/A {_f(aa)}); vs MIOpen "
                 f"{_f(s_ / t - 1 if s_ else None)}; T coverage {cov}; LPIPS upper T {_f(up)} vs S/S floor {_f(floor)}; "
                 f"peak T-{other} {dpeak:+.2f} GiB, T-S {dpeak_s:+.2f} GiB")
        why = []
        if not all(cov) or min(cov) <= 0:
            why.append("not engaged (coverage 0)")
        if gain < kill:
            why.append(f"under {kill:.0%} faster ({gain:.1%})")
        if aa is not None and abs(gain) <= aa:
            why.append("inside the T A/A spread")
        if up is not None and up - floor > 0.01:
            why.append(f"LPIPS upper +{up - floor:.4f} > 0.01")
        if up is None:
            why.append("LPIPS not measured")
        if dpeak > 2.0:
            why.append(f"peak +{dpeak:.2f} GiB > 2 GiB")
        out.append(f"{name} " + ("KILL (" + "; ".join(why) + ")" if why else f"KEEP ({gain:.1%} faster)"))
    L.append("  note: T is byte-for-byte the productisation ticket's Triton A (its vendored aiter conv3d already inlines "
             "the PR #6973 gfx1151 tables), so 'aiter + #6973 vs Triton A' is an A/A; the tables' own effect is T vs G, "
             "and the bar for the lever is the faster of S and T")
    L.append("  VERDICT C15-3: " + " | ".join(out))
    return L


def table(h: dict) -> list:
    L = [f"smoke {h.get('smoke')}"]
    for fn in (_c63_lines, _c63b_lines, _m4_lines, _knee_lines, _krea_lines, _c153_lines):
        try:
            L.extend(fn(h))
        except Exception as exc:  # noqa: BLE001
            L.append(f"{fn.__name__} error {exc!r}")
    return L
