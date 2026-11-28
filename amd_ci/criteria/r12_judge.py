#!/usr/bin/env python3
"""Judge for probes/r12_items.py. Gates prove each listed item ran on gfx1151 with engagement; table() carries the
numbers plus one VERDICT line per item against its pre-registered kill line:
  carbon12#6 (w8a8): KILL if the best M-bucketed table (aiter#7282 gfx1150, aiter#7270 gfx1151, or the per-M best of
       the candidate grid) is under 1.25x geomean over M <= 256, or any of its configs is not bit-identical to the
       current tile.
  carbon6#3 (conv): KILL if the aiter decode is under 1.08x MIOpen, or the VAE decode is under 10% of the render,
       or (for a win) the error is above 1.05x MIOpen's vs fp32.
  carbon16#2 (adaln): KILL if forced-on saves under 2% s/image, or any VERIFY mismatch / image md5 change; when the
       forced arm cannot install the table, the A/B comparison is VOID and the per-call modulation cost bounds the
       lever (KILL if that bound is under 2% of the render)."""
from __future__ import annotations

import math
import statistics as st

W8A8_GATE, CONV_GATE, CONV_SHARE, CONV_ERR_X, ADALN_GATE = 1.25, 1.08, 0.10, 1.05, 0.02


def _f(x, d = 3):
    return "n/a" if x is None else f"{x:.{d}f}" if isinstance(x, float) else str(x)


def _gm(xs):
    xs = [x for x in xs if isinstance(x, (int, float)) and x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else None


def _med(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return st.median(xs) if xs else None


def _spread(xs):
    xs = [x for x in xs if isinstance(x, (int, float)) and x > 0]
    if len(xs) < 2:
        return None
    m = st.median(xs)
    return (max(xs) - min(xs)) / m


# ---------------------------------------------------------------- w8a8
def _w8a8_rows(h):
    return [r for r in ((h.get("w8a8") or {}).get("rows") or []) if r.get("ms")]


def _w8a8_tables(rows):
    """{table: {row_index: speedup}} for p1150 / p1151 / oracle (per-M best grid config, bitwise-equal only)."""
    out = {"p1150": {}, "p1151": {}, "oracle": {}}
    bad = {"p1150": 0, "p1151": 0, "oracle": 0}
    for i, r in enumerate(rows):
        cur = r["ms"].get("cur")
        for t in ("p1150", "p1151"):
            if t in r["ms"] and cur:
                out[t][i] = cur / r["ms"][t]
                if not (r.get("bitwise_vs_cur") or {}).get(t, False):
                    bad[t] += 1
    by_m: dict = {}
    for i, r in enumerate(rows):
        by_m.setdefault(r["M"], []).append(i)
    picks = {}
    for m, idx in by_m.items():
        cands = set()
        for i in idx:
            cands |= {k for k in rows[i]["ms"] if k not in ("cur", "cur_aa")}
        best, bestg = None, 0.0
        for c in sorted(cands):
            sp = []
            ok = True
            for i in idx:
                r = rows[i]
                if c not in r["ms"] or not (r.get("bitwise_vs_cur") or {}).get(c, False):
                    ok = False
                    break
                sp.append(r["ms"]["cur"] / r["ms"][c])
            g = _gm(sp) if ok else None
            if g and g > bestg:
                best, bestg = c, g
        # never worse than keeping the current tile for this bucket
        if bestg < 1.0:
            best, bestg = "cur", 1.0
        picks[m] = (best, bestg)
        for i in idx:
            r = rows[i]
            out["oracle"][i] = 1.0 if best == "cur" else r["ms"]["cur"] / r["ms"][best]
    return out, bad, picks


def _w8a8_gates(h):
    w = h.get("w8a8") or {}
    rows = _w8a8_rows(h)
    pk = (w.get("profiler_kernels") or {}).get("by_arm") or {}
    eng = all(any("w8a8_mm" in n for n in pk.get(a, [])) for a in ("cur", "p1150", "p1151")) and bool(pk)
    return [("w8a8 bench ok", w.get("verdict") == "ok", str(w.get("error"))[-200:]),
            ("w8a8 Studio parity probe (prepare) passed", w.get("prepare") is True, str(w.get("prepare_state"))[:200]),
            ("w8a8 rows measured", len(rows) >= 100, f"{len(rows)} rows"),
            ("w8a8 'cur' arm == Studio _mm_launch bitwise (every row)", bool(rows) and all(r.get("cur_equals_production") for r in rows),
             str(sum(1 for r in rows if not r.get("cur_equals_production")))),
            ("w8a8 no fallback (_STATE broken None)", all(not r.get("broken") for r in rows), ""),
            ("w8a8 engagement: Studio w8a8_mm Triton kernel in cur / p1150 / p1151 profiles", eng, str(pk)[:300])]


def _w8a8_table(h):
    rows = _w8a8_rows(h)
    if not rows:
        return ["- carbon12#6: no rows"], "VOID (no rows)"
    L = []
    aa = [abs(r["ms"]["cur_aa"] / r["ms"]["cur"] - 1) for r in rows if r["ms"].get("cur_aa") and r["ms"].get("cur")]
    tabs, bad, picks = _w8a8_tables(rows)
    L.append(f"- carbon12#6 A/A (cur vs cur_aa, same config, interleaved): median {_f(_med(aa), 4)}, max {_f(max(aa) if aa else None, 4)}")
    L.append("")
    L.append("| M | rows | cur ms (geomean) | p1150 x | p1151 x | best grid cfg | grid x | bit-identical (p1150/p1151/grid) |")
    L.append("|---|---|---|---|---|---|---|---|")
    by_m: dict = {}
    for i, r in enumerate(rows):
        by_m.setdefault(r["M"], []).append(i)
    for m in sorted(by_m):
        idx = by_m[m]
        b1 = all((rows[i].get("bitwise_vs_cur") or {}).get("p1150", False) for i in idx)
        b2 = all((rows[i].get("bitwise_vs_cur") or {}).get("p1151", False) for i in idx)
        bg = all((rows[i].get("bitwise_vs_cur") or {}).get(picks[m][0], True) for i in idx)
        L.append(f"| {m} | {len(idx)} | {_f(_gm([rows[i]['ms']['cur'] for i in idx]), 4)} | "
                 f"{_f(_gm([tabs['p1150'].get(i) for i in idx]))} | {_f(_gm([tabs['p1151'].get(i) for i in idx]))} | "
                 f"{picks[m][0]} | {_f(picks[m][1])} | {b1}/{b2}/{bg} |")
    L.append("")
    L.append("| model.layer (N, K) | M=16 x p1150/p1151/grid | M=256 x | M=4096 x |")
    L.append("|---|---|---|---|")
    seen = []
    for r in rows:
        key = (r["model"], r["layer"])
        if key not in seen:
            seen.append(key)
    for key in seen:
        cells = []
        for m in (16, 256, 4096):
            i = next((j for j, r in enumerate(rows) if (r["model"], r["layer"]) == key and r["M"] == m), None)
            cells.append("n/a" if i is None else
                         f"{_f(tabs['p1150'].get(i), 2)}/{_f(tabs['p1151'].get(i), 2)}/{_f(tabs['oracle'].get(i), 2)}")
        r0 = next(r for r in rows if (r["model"], r["layer"]) == key)
        L.append(f"| {key[0]}.{key[1]} ({r0['N']}, {r0['K']}) | " + " | ".join(cells) + " |")
    small = [i for i, r in enumerate(rows) if r["M"] <= 256]
    sub1k = [i for i, r in enumerate(rows) if r["M"] < 1024]
    g = {t: (_gm([tabs[t].get(i) for i in small]), _gm([tabs[t].get(i) for i in sub1k]), _gm(list(tabs[t].values())))
         for t in tabs}
    for t, (a, b, c) in g.items():
        L.append(f"- carbon12#6 {t}: geomean M<=256 {_f(a)}x, M<1024 {_f(b)}x, all M {_f(c)}x; rows not bit-identical {bad.get(t, 0)}")
    pk = ((h.get("w8a8") or {}).get("profiler_kernels") or {})
    L.append(f"- carbon12#6 engagement profile ({pk.get('shape')}): {pk.get('by_arm')}")
    best_t = max(g, key = lambda t: g[t][0] or 0)
    bt = g[best_t][0] or 0
    bits_bad = sum(bad.values())
    if bt >= W8A8_GATE and bad.get(best_t, 0) == 0:
        v = f"SHIP-CANDIDATE ({best_t} {bt:.3f}x geomean over M <= 256, bit-identical)"
    else:
        why = [f"best table {best_t} {bt:.3f}x geomean over M <= 256 under {W8A8_GATE}x"] if bt < W8A8_GATE else []
        if bits_bad:
            why.append(f"{bits_bad} row(s) not bit-identical")
        v = "KILL (" + "; ".join(why or ["gate"]) + ")"
    return L, v


# ---------------------------------------------------------------- conv
def _conv_gates(h):
    c = h.get("conv") or {}
    rows = [r for r in c.get("rows") or [] if r.get("ms")]
    jit = set(c.get("aiter_jit_names") or [])
    eng_t = bool(rows) and bool(jit) and all(any(k[0] in jit or k[0].split("_0d")[0] in jit or any(k[0].startswith(j) for j in jit)
                                                 for k in (r.get("kernels") or {}).get("T", [])) for r in rows)
    eng_m = bool(rows) and all(not any(any(k[0].startswith(j) for j in jit) for k in (r.get("kernels") or {}).get("M", []))
                               for r in rows)
    dec = c.get("decode") or {}
    dec_ok = bool(dec) and all((d.get("routed_per_decode") or {}).get("T", 0) > 0 and (d.get("routed_per_decode") or {}).get("M", 1) == 0
                               for d in dec.values() if not d.get("error"))
    return [("conv bench ok", c.get("verdict") == "ok", str(c.get("error"))[-200:]),
            ("conv vendored aiter imported from the probe tree", "aiter_pkg" in str(c.get("aiter_file")), str(c.get("aiter_file"))),
            ("conv rows measured (all unique decoder shapes)", len(rows) >= 30, f"{len(rows)} rows"),
            ("conv engagement: every T arm profile has a vendored aiter @triton.jit kernel", eng_t, f"{len(jit)} jit names"),
            ("conv engagement: no aiter kernel in any MIOpen arm profile", eng_m, ""),
            ("conv decodes ran with routed calls under T only", dec_ok,
             str({k: (d.get("routed_per_decode"), d.get("error")) for k, d in dec.items()})[:300])]


def _conv_table(h):
    c = h.get("conv") or {}
    rows = [r for r in c.get("rows") or [] if r.get("ms")]
    if not rows:
        return ["- carbon6#3: no rows"], "VOID (no rows)"
    L = []
    aa = [abs(r["ms"]["M_aa"] / r["ms"]["M"] - 1) for r in rows if r["ms"].get("M_aa") and r["ms"].get("M")]
    L.append(f"- carbon6#3 env {c.get('env')} miopen db active {c.get('miopen_db_active')} ({c.get('miopen_db_eligible')})")
    L.append(f"- carbon6#3 A/A (M vs M_aa): median {_f(_med(aa), 4)}, max {_f(max(aa) if aa else None, 4)}")
    L.append("")
    L.append("| x (NCHW) | w | plane px | uses (vae: calls) | MIOpen ms | aiter ms | x | NHWC: MIOpen / aiter ms | x | err aiter/MIOpen | aiter kernel (top) | MIOpen kernel (top) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        ms = r["ms"]
        e = r.get("rel_l2") or {}
        sx = ms["M"] / ms["T"] if ms.get("T") else None
        sc = ms["M_cl"] / ms["T_nhwc"] if ms.get("T_nhwc") and ms.get("M_cl") else None
        er = e["T"] / e["M"] if e.get("T") is not None and e.get("M") else None
        kt = ((r.get("kernels") or {}).get("T") or [["n/a"]])[0][0]
        km = ((r.get("kernels") or {}).get("M") or [["n/a"]])[0][0]
        L.append(f"| {r['x']} | {r['w']} | {r['plane_px']} | {r['uses']} | {_f(ms.get('M'), 3)} | {_f(ms.get('T'), 3)} | "
                 f"{_f(sx, 2)} | {_f(ms.get('M_cl'), 3)} / {_f(ms.get('T_nhwc'), 3)} | {_f(sc, 2)} | {_f(er, 2)} | {kt} | {km} |")
    L.append("")
    small = [r["ms"]["M"] / r["ms"]["T"] for r in rows if r["plane_px"] <= 16384 and r["ms"].get("T")]
    L.append(f"- carbon6#3 small planes (<= 16384 px, comfy-kitchen#324 scope): aiter/MIOpen speedups {[round(x, 2) for x in small]}")
    per_vae = {}
    for vae in ("qwen21", "flux2", "flux1_zimage"):
        sm = st_ = 0.0
        for r in rows:
            n = (r.get("uses") or {}).get(vae)
            if n and r["ms"].get("T"):
                sm += n * r["ms"]["M"]
                st_ += n * r["ms"]["T"]
        per_vae[vae] = (sm, st_)
    dec = c.get("decode") or {}
    L.append("")
    L.append("| VAE (1024^2, bf16, random weights) | conv sum MIOpen ms | conv sum aiter ms | conv-sum x | decode MIOpen s | decode aiter s | decode x | PSNR aiter vs MIOpen dB | routed / decode |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for vae, (sm, st_) in per_vae.items():
        d = dec.get(vae) or {}
        s = d.get("s") or {}
        L.append(f"| {vae} | {_f(sm, 1)} | {_f(st_, 1)} | {_f(sm / st_ if st_ else None, 2)} | {_f(s.get('M'), 3)} | "
                 f"{_f(s.get('T'), 3)} | {_f(d.get('speedup'), 3)} | {_f(d.get('psnr_T_vs_M_db'), 1)} | "
                 f"{(d.get('routed_per_decode') or {}).get('T')} | ")
        if d.get("error"):
            L.append(f"- decode {vae} error: {d['error']}")
    # decode share of the render
    share = {}
    qa = [a for a in ((h.get("adaln") or {}).get("arms") or []) if a.get("verdict") == "ok"]
    qs = [(x["vae_s"][0] / x["wall_s"]) for a in qa for x in (a.get("renders") or [])[1:] if x.get("vae_s") and x.get("wall_s")]
    if qs:
        share["qwen21"] = _med(qs)
    cf = (h.get("convflux") or {}).get("arm") or {}
    fs = [(x["vae_s"][0] / x["wall_s"]) for x in (cf.get("renders") or [])[1:] if x.get("vae_s") and x.get("wall_s")]
    if fs:
        share["flux2"] = _med(fs)
    L.append(f"- carbon6#3 VAE decode share of the Studio render (measured renders): {share}; "
             f"convflux {(h.get('convflux') or {}).get('not_run') or 'ran'}")
    verdicts = []
    for vae in per_vae:
        d = dec.get(vae) or {}
        sp = d.get("speedup")
        errs = [r["rel_l2"]["T"] / r["rel_l2"]["M"] for r in rows if (r.get("uses") or {}).get(vae)
                and (r.get("rel_l2") or {}).get("T") is not None and (r.get("rel_l2") or {}).get("M")]
        emax = max(errs) if errs else None
        why = []
        if not isinstance(sp, (int, float)):
            verdicts.append(f"{vae}: VOID (no decode)")
            continue
        if sp < CONV_GATE:
            why.append(f"decode {sp:.3f}x under {CONV_GATE}x")
        if vae in share and share[vae] < CONV_SHARE:
            why.append(f"decode {share[vae] * 100:.1f}% of the render, under {CONV_SHARE * 100:.0f}%")
        if not why and emax is not None and emax > CONV_ERR_X:
            why.append(f"error up to {emax:.2f}x MIOpen's")
        if why:
            verdicts.append(f"{vae}: KILL ({'; '.join(why)})")
        elif vae not in share:
            verdicts.append(f"{vae}: decode {sp:.3f}x passes, decode share NOT MEASURED (render not run): SHIP-CANDIDATE only after a share >= 10%")
        else:
            verdicts.append(f"{vae}: SHIP-CANDIDATE (decode {sp:.3f}x, share {share[vae] * 100:.1f}%, err <= {_f(emax, 2)}x)")
    v = "KILL" if all("KILL" in x for x in verdicts) else ("SHIP-CANDIDATE" if any("SHIP-CANDIDATE (" in x for x in verdicts) else "see per-VAE")
    return L, f"{v} ({' | '.join(verdicts)})"


# ---------------------------------------------------------------- adaln
def _adaln_gates(h):
    res = h.get("adaln") or {}
    arms = res.get("arms") or []
    ok = [a for a in arms if a.get("verdict") == "ok"]
    A = [a for a in ok if a.get("arm") == "A"]
    B = [a for a in ok if a.get("arm") == "B"]
    out = [("adaln arms ok", bool(arms) and len(ok) == len(arms), f"{len(ok)} of {len(arms)}: " +
            "; ".join(str(a.get("error"))[-150:] for a in arms if a.get("verdict") != "ok")),
           ("adaln A (Auto) arms: install called and declined (Auto off on resident gfx1151)",
            bool(A) and all(a.get("adaln_install_calls") and not a.get("adaln_installed") for a in A),
            str([(c.get("decline_reason"), c.get("returned")) for a in A for c in a.get("adaln_install_calls") or []])[:300]),
           ("adaln B (forced) arms: install called, outcome recorded", bool(B) and all(a.get("adaln_install_calls") for a in B),
            str([(c.get("cls"), c.get("decline_reason"), c.get("returned"), c.get("adapter")) for a in B for c in a.get("adaln_install_calls") or []])[:300]),
           ("adaln B env reached the process", bool(B) and all((a.get("adaln_env") or {}).get("UNSLOTH_DIFFUSION_ADALN_PRECOMPUTE") == "1"
                                                            and a.get("adaln_mode") == "on" for a in B), str([a.get("adaln_env") for a in B])[:200]),
           ("adaln resident (no offload) in every arm", bool(ok) and all((a.get("status") or {}).get("offload_policy") in ("none", None) for a in ok),
            str([(a.get("status") or {}).get("offload_policy") for a in ok]))]
    return out


def _adaln_table(h):
    res = h.get("adaln") or {}
    arms = [a for a in res.get("arms") or [] if a.get("verdict") == "ok"]
    if not arms:
        return ["- carbon16#2: no arms"], "VOID (no arms)"
    L = ["", "| proc | arm | load s | s/image (measured) | s/step | steps | VAE s | peak GiB | md5 | installed | adaln counts | transformer |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for a in arms:
        rs = a.get("renders") or []
        m = rs[-1] if rs else {}
        cn = a.get("adaln_counts") or {}
        L.append(f"| {a['i']} | {a['arm']} | {a.get('load_s')} | {m.get('wall_s')} | {m.get('s_per_step')} | {m.get('n_steps')} | "
                 f"{(m.get('vae_s') or [None])[0]} | {m.get('peak_alloc_gib')} | {str(m.get('md5'))[:10]} | {a.get('adaln_installed')} | "
                 f"inst {cn.get('installs')} served {cn.get('served')} verify {cn.get('verify_checks')} | {a.get('transformer_class')} |")
    A = [a for a in arms if a["arm"] == "A"]
    B = [a for a in arms if a["arm"] == "B"]
    wa = [a["renders"][-1]["wall_s"] for a in A if a.get("renders")]
    wb = [a["renders"][-1]["wall_s"] for a in B if a.get("renders")]
    L.append(f"- carbon16#2 A s/image {wa} (spread {_f(_spread(wa), 4)}), B {wb} (spread {_f(_spread(wb), 4)})")
    calls = [(c.get("cls"), c.get("decline_reason"), c.get("returned")) for a in B for c in a.get("adaln_install_calls") or []]
    L.append(f"- carbon16#2 forced-arm install outcome: {calls[:2]}")
    mp = next((a.get("mod_probe") for a in A if a.get("mod_probe")), None)
    wall = _med(wa)
    bound = None
    if mp:
        bound = mp.get("total_s_per_render")
        L.append(f"- carbon16#2 per-call modulation Linears (the most a table could remove): {mp.get('sites')}; "
                 f"total {bound} s per render = {_f(bound / wall * 100 if bound is not None and wall else None, 3)}% of {wall} s")
    st0 = (A[0].get("status") if A else {}) or {}
    L.append(f"- carbon16#2 load status {st0}; params {A[0].get('params') if A else None}; vae {A[0].get('vae_class') if A else None} {A[0].get('vae_dtype') if A else None}; env {A[0].get('proc_env') if A else None}")
    L.append(f"- carbon16#2 rdna_mm launches per measured render (A): {[a['renders'][-1].get('rdna_mm') for a in A if a.get('renders')]}")
    engaged = bool(res.get("b_engaged"))
    if not engaged:
        if bound is not None and wall and bound / wall < ADALN_GATE:
            v = (f"KILL (forced arm cannot install: {calls[0][1] if calls else '?'}; so the A/B timing is VOID by construction. "
                 f"Ceiling = per-call modulation cost {bound} s/image = {bound / wall * 100:.3f}% < {ADALN_GATE * 100:.0f}%)")
        else:
            v = f"VOID (forced arm did not install; no usable ceiling: bound {bound}, wall {wall})"
        return L, v
    md5a = {a["renders"][-1].get("md5") for a in A if a.get("renders")}
    md5b = {a["renders"][-1].get("md5") for a in B if a.get("renders")}
    ver = [int((a.get("adaln_counts") or {}).get("verify_checks") or 0) for a in B]
    gain = (_med(wa) - _med(wb)) / _med(wa) if wa and wb else None
    L.append(f"- carbon16#2 gain {_f(gain * 100 if gain is not None else None, 2)}%; md5 A {md5a} B {md5b}; verify checks {ver}")
    if md5a != md5b:
        return L, f"KILL (image md5 differs A {md5a} vs B {md5b})"
    if gain is None or gain < ADALN_GATE:
        return L, f"KILL (gain {_f(gain * 100 if gain is not None else None, 2)}% under {ADALN_GATE * 100:.0f}%)"
    return L, f"SHIP-CANDIDATE (gain {gain * 100:.2f}%, bit-exact md5, verify checks {ver})"


def gates(h: dict, items: list) -> list:
    out = []
    if "smoke" in items:
        out.append(("smoke gfx1151", bool(h.get("smoke_ok")), str((h.get("smoke") or {}).get("arch"))))
        out.append(("smoke benches ok", (h.get("smoke_w8a8") or {}).get("verdict") == "ok" and (h.get("smoke_conv") or {}).get("verdict") == "ok",
                    str((h.get("smoke_w8a8") or {}).get("error")) + " / " + str((h.get("smoke_conv") or {}).get("error"))))
    if "w8a8" in items:
        out.extend(_w8a8_gates(h))
    if "conv" in items:
        out.extend(_conv_gates(h))
    if "adaln" in items:
        out.extend(_adaln_gates(h))
    if "convflux" in items:
        cf = h.get("convflux") or {}
        out.append(("convflux ran or recorded NOT RUN", bool(cf.get("not_run")) or (cf.get("arm") or {}).get("verdict") == "ok",
                    str(cf.get("not_run") or (cf.get("arm") or {}).get("error"))[:200]))
    return out


def table(h: dict) -> list:
    L = []
    if h.get("smoke"):
        L.append(f"smoke {h.get('smoke')}")
    for name, fn in (("carbon12#6", _w8a8_table), ("carbon6#3", _conv_table), ("carbon16#2", _adaln_table)):
        key = {"carbon12#6": "w8a8", "carbon6#3": "conv", "carbon16#2": "adaln"}[name]
        if not h.get(key):
            continue
        try:
            rows, v = fn(h)
        except Exception as exc:  # noqa: BLE001
            rows, v = [f"table error {exc!r}"], "VOID (judge error)"
        L.extend(rows)
        L.append(f"VERDICT {name}: {v}")
        L.append("")
    return L
