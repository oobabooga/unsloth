#!/usr/bin/env python3
"""Judge for probes/r9d_items.py (smoke, c62, c66). Gates prove each listed item ran with engagement (weight strides
per layout, GEMM kernels in each arm's profile, flags recorded); table() carries the numbers plus one VERDICT line per
item against its pre-registered kill line:
  C6-2 transposed_problem: KILL if under 5% block time over the best of contig / pitch, if it adds a copy kernel
       (more non-GEMM, non-attention launches per block than contig, or a new copy-class kernel), or on drift
       (non-finite, or rel err vs fp32 over 1.25x contig's).
  C6-6: KILL if Inductor glue under 10% of block device time; else KILL if the best flag is under 5% faster."""
from __future__ import annotations

import statistics as st

GAIN = 0.05
GLUE = 0.10
DRIFT_X = 1.25
LAYOUTS = ("contig", "pitch", "trans", "trans_pitch")


def _f(x, d = 3):
    return "n/a" if x is None else f"{x:.{d}f}" if isinstance(x, float) else str(x)


def _ok(rows):
    return [r for r in rows or [] if r.get("verdict") == "ok"]


def _stride_ok(r) -> bool:
    lay = r.get("layout")
    ws = r.get("weights") or []
    if not ws:
        return False
    for _, (n, k), (s0, s1) in ws:
        want = {"contig": (k, 1), "pitch": (k + 64, 1), "trans": (1, n), "trans_pitch": (1, n + 64)}[lay]
        if (s0, s1) != want:
            return False
    return True


def _med(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return st.median(xs) if xs else None


def _by(rows, key, val, sh, field = "ms"):
    return [(r.get("shapes") or {}).get(sh, {}).get(field) for r in rows if r.get(key) == val]


def _noncore(v):
    lp = v.get("launches_per_iter") or {}
    return sum(c for k, c in lp.items() if k not in ("gemm", "attention"))


def gates(h: dict, items: list) -> list:
    out = []
    if "smoke" in items:
        out.append(("smoke gfx1151", bool(h.get("smoke_ok")), str((h.get("smoke") or {}).get("arch"))))
        sb = h.get("smoke_block") or {}
        out.append(("smoke tiny compiled block ok", sb.get("verdict") == "ok", str(sb.get("error"))[-200:]))
    if "c62" in items:
        rows = h.get("c62") or []
        ok = _ok(rows)
        out.append(("c62 arms ok (9 processes)", len(ok) == 9, f"{len(ok)} of {len(rows)}: "
                    + "; ".join(f"{r.get('tag')} {str(r.get('error'))[-150:]}" for r in rows if r.get("verdict") != "ok")))
        out.append(("c62 engagement: weight strides match each layout", bool(ok) and all(_stride_ok(r) for r in ok),
                    ", ".join(f"{r['layout']}:{(r.get('weights') or [[0, 0, 0]])[0][2]}" for r in ok[:4])))
        gem = all((v.get("share") or {}).get("gemm", 0) > 0 for r in ok for v in (r.get("shapes") or {}).values())
        out.append(("c62 engagement: GEMM kernels in every arm's profile", bool(ok) and gem, ""))
    if "bufops" in items:
        b = h.get("bufops") or {}
        ok = [r for r in b.get("arms") or [] if r.get("verdict") == "ok" and len(r.get("renders") or []) >= 2]
        out.append(("bufops render arms ok (12 processes)", len(ok) >= 10, f"{len(ok)} of {len(b.get('arms') or [])}: "
                    + "; ".join(str(r.get("error"))[-150:] for r in b.get("arms") or [] if r.get("verdict") != "ok")))
        eng = [r for r in ok if (r.get("isa") or {}).get("amdgcn_files")]
        out.append(("bufops engagement: compiled ISA present in every arm's fresh Triton cache", len(eng) == len(ok) and bool(ok),
                    f"{len(eng)} of {len(ok)}"))
    if "c66" in items:
        c = h.get("c66") or {}
        ran = bool(c.get("not_run")) or len(_ok(c.get("arms"))) >= 6
        out.append(("c66 glue measured, flag arms ran or recorded NOT RUN", c.get("glue_median_max") is not None and ran,
                    c.get("not_run") or f"{len(_ok(c.get('arms')))} flag arms ok"))
    return out


def table(h: dict) -> list:
    L = []
    sb = h.get("smoke_block") or {}
    L.append(f"smoke {h.get('smoke', {}).get('arch')} gemm {h.get('smoke', {}).get('gemm_tflops')} TFLOP/s; tiny block "
             f"{sb.get('verdict')} env {sb.get('env')}")
    ok = _ok(h.get("c62"))
    shapes = sorted({sh for r in ok for sh in (r.get("shapes") or {})})
    c62_kill = []
    best_gain = {}
    noise = {}
    for sh in shapes:
        med = {lay: _med(_by(ok, "layout", lay, sh)) for lay in LAYOUTS}
        a = [x for x in _by(ok, "layout", "contig", sh) if x]
        aa = (max(a) - min(a)) / st.median(a) if len(a) >= 2 else None
        L.append(f"C6-2 {sh} (S {next(((r['shapes'][sh].get('S')) for r in ok if sh in r['shapes']), None)}) block ms "
                 + ", ".join(f"{k} {_f(v, 2)} {[_f(x, 2) for x in _by(ok, 'layout', k, sh)]}" for k, v in med.items())
                 + f"; contig A/A spread {_f(aa)}")
        noise[sh] = aa or 0.0
        inc = min(x for x in (med["contig"], med["pitch"]) if x) if med["contig"] or med["pitch"] else None
        ref_err = _med(_by(ok, "layout", "contig", sh, "rel_err_vs_fp32"))
        ref_nc = _med([_noncore(r["shapes"][sh]) for r in ok if r.get("layout") == "contig" and sh in r["shapes"]])
        cnames = {k[0] for r in ok if r.get("layout") == "contig" and sh in r["shapes"]
                  for k in r["shapes"][sh].get("kernels") or [] if k[3] not in ("gemm", "attention")}
        for lay in ("trans", "trans_pitch", "pitch"):
            if not med.get(lay) or not inc:
                continue
            gain = inc / med[lay] - 1 if lay != "pitch" else med["contig"] / med["pitch"] - 1
            err = _med(_by(ok, "layout", lay, sh, "rel_err_vs_fp32"))
            fin = all(_by(ok, "layout", lay, sh, "finite"))
            nc = _med([_noncore(r["shapes"][sh]) for r in ok if r.get("layout") == lay and sh in r["shapes"]])
            new = sorted({k[0][:90] for r in ok if r.get("layout") == lay and sh in r["shapes"]
                          for k in r["shapes"][sh].get("kernels") or [] if k[3] not in ("gemm", "attention")} - cnames)
            gemms = sorted({k[0][:110] for r in ok if r.get("layout") == lay and sh in r["shapes"]
                            for k in r["shapes"][sh].get("kernels") or [] if k[3] == "gemm"})
            share = _med([r["shapes"][sh].get("share", {}).get("gemm") for r in ok if r.get("layout") == lay
                          and sh in r["shapes"]])
            drift = (not fin) or (err is not None and ref_err and err > DRIFT_X * ref_err)
            extra = (nc is not None and ref_nc is not None and nc > ref_nc) or bool(new)
            L.append(f"  {sh} {lay}: gain {'vs contig' if lay == 'pitch' else 'vs best(contig, pitch)'} "
                     f"{_f(gain * 100, 1)}%; rel err {_f(err, 5)} (contig {_f(ref_err, 5)}) finite {fin}; non-GEMM "
                     f"launches/block {nc} (contig {ref_nc}); new non-GEMM kernels {new[:4]}; gemm share {_f(share)}; "
                     f"GEMM kernels {gemms[:3]}")
            if lay != "pitch":
                best_gain[(sh, lay)] = gain
                if drift:
                    c62_kill.append(f"drift {sh} {lay}")
                if extra:
                    c62_kill.append(f"extra copy/glue kernel {sh} {lay}")
    if ok:
        g = max(best_gain.values()) if best_gain else None
        reasons = list(c62_kill)
        if g is None or g < GAIN:
            reasons.insert(0, f"best transposed gain {_f(None if g is None else g * 100, 1)}% under 5%")
        clean = [k for k, v in best_gain.items() if v >= max(GAIN, noise.get(k[0], 0)) and not any(f"{k[0]} {k[1]}" in x for x in c62_kill)]
        L.append(f"VERDICT C6-2 transposed_problem: {'PASS ' + str(clean) if clean else 'KILL (' + '; '.join(reasons) + ')'}")
    else:
        L.append("VERDICT C6-2 transposed_problem: NOT MEASURED (no ok arm)")
    L.extend(_bufops_table(h.get("bufops") or {}))
    c = h.get("c66") or {}
    if c:
        L.append(f"C6-6 glue share by shape (contig arms, inductor+aten glue+copy) {c.get('glue_by_shape')}")
        for r in ok[:1]:
            for sh, v in (r.get("shapes") or {}).items():
                L.append(f"  {sh} share {v.get('share')} launches {v.get('launches_per_iter')}; top glue "
                         f"{[k[:3] for k in v.get('kernels') or [] if k[3] != 'gemm' and k[3] != 'attention'][:5]}")
        if c.get("not_run"):
            gm = c.get("glue_median_max")
            L.append(f"VERDICT C6-6: {'KILL' if gm is not None and gm < GLUE else 'NOT MEASURED'} ({c['not_run']})")
        else:
            arms = _ok(c.get("arms"))
            base_rows = [r for r in ok if r.get("layout") == "contig"] + [r for r in arms if r.get("flags") == "none"]
            best = None
            for sh in sorted({s for r in arms for s in r.get("shapes") or {}}):
                b = _med(_by(base_rows, "flags", "none", sh))
                bl = [x for x in _by(base_rows, "flags", "none", sh) if x]
                bn = (max(bl) - min(bl)) / st.median(bl) if len(bl) >= 2 else 0.0
                be = _med(_by(base_rows, "flags", "none", sh, "rel_err_vs_fp32"))
                for fl in ("cdt", "mag", "maxat"):
                    m = _med(_by(arms, "flags", fl, sh))
                    e = _med(_by(arms, "flags", fl, sh, "rel_err_vs_fp32"))
                    if not m or not b:
                        continue
                    gain = b / m - 1
                    drift = e is not None and be and e > DRIFT_X * be
                    gl = _med(_by(arms, "flags", fl, sh, "glue_share"))
                    L.append(f"  {sh} {fl}: ms {_f(m, 2)} {[_f(x, 2) for x in _by(arms, 'flags', fl, sh)]} vs none "
                             f"{_f(b, 2)} (A/A spread {_f(bn)}): gain {_f(gain * 100, 1)}%; glue {_f(gl)}; rel err {_f(e, 5)} (none {_f(be, 5)})")
                    if not drift and gain > bn and (best is None or gain > best[0]):
                        best = (gain, sh, fl)
            L.append("VERDICT C6-6 flags: " + ("PASS " + str(best) if best and best[0] >= GAIN else
                                               f"KILL (best flag gain {_f(best and best[0] * 100, 1)}% under 5%)"))
    return L


BUF_GAIN = 0.02


def _bufops_table(b: dict) -> list:
    if not b:
        return []
    L = [f"bufops Triton default {b.get('default')}"]
    ok = [r for r in b.get("arms") or [] if r.get("verdict") == "ok" and len(r.get("renders") or []) >= 2]
    if not ok:
        return L + ["VERDICT bufops render: NOT MEASURED (no ok arm)"]
    per = {"A": [], "B": []}
    sps = {"A": [], "B": []}
    md5 = {"A": set(), "B": set()}
    for r in ok:
        x = r["renders"][-1]
        per[r["arm"]].append(x.get("wall_s"))
        sps[r["arm"]].append(x.get("s_per_step"))
        md5[r["arm"]].add(x.get("md5"))
        isa = r.get("isa") or {}
        L.append(f"  {r['i']} {r['arm']} render1 {_f(x.get('wall_s'), 2)} s, {_f(x.get('s_per_step'), 4)} s/step, md5 "
                 f"{str(x.get('md5'))[:12]}, quant {(r.get('status') or {}).get('transformer_quant')}, rdna_mm "
                 f"{x.get('rdna_mm')}; ISA files {isa.get('amdgcn_files')} buffer_load {isa.get('buffer_load')} "
                 f"buffer_store {isa.get('buffer_store')} global_load {isa.get('global_load')}")
    a, bb = _med(per["A"]), _med(per["B"])
    al = [x for x in per["A"] if x]
    aa = (max(al) - min(al)) / st.median(al) if len(al) >= 2 else None
    gain = a / bb - 1 if a and bb else None
    bl_a = _med([(r.get("isa") or {}).get("buffer_load") for r in ok if r["arm"] == "A"])
    bl_b = _med([(r.get("isa") or {}).get("buffer_load") for r in ok if r["arm"] == "B"])
    engaged = bl_b is not None and bl_a is not None and bl_b > bl_a
    same = md5["A"] == md5["B"] and len(md5["A"]) == 1
    L.append(f"bufops render1 wall median A(=0) {_f(a, 2)} s {per['A']} vs B(=1) {_f(bb, 2)} s {per['B']}: gain "
             f"{_f(None if gain is None else gain * 100, 2)}%; A/A spread {_f(aa)}; s/step A {_f(_med(sps['A']), 4)} B "
             f"{_f(_med(sps['B']), 4)}; md5 A {sorted(m[:12] for m in md5['A'] if m)} B {sorted(m[:12] for m in md5['B'] if m)}; "
             f"ISA buffer_load median A {bl_a} B {bl_b} (engaged {engaged})")
    if not engaged:
        v = "NOT PROVEN (B compiled no more buffer_load than A: the variable did not change codegen)"
    elif not same:
        v = "KILL (md5 changed between A and B, or within an arm)"
    elif gain is None or gain < max(BUF_GAIN, aa or 0):
        v = f"KILL (render gain {_f(None if gain is None else gain * 100, 2)}% under 2% or inside A/A)"
    else:
        v = f"PASS ({gain * 100:.2f}% render, md5 identical)"
    return L + [f"VERDICT bufops render: {v}"]
