#!/usr/bin/env python3
"""Judge helpers for probes/r8_lnx_kern.py (imported by criteria/r8_measure.py). Gates prove each listed item ran on
gfx1151 (a NOT RUN with its reason recorded passes the gate; an unrecorded absence fails it). table() carries the
numbers and one verdict line per item against its kill line (triage_1 C9-3 / A1 fold-ins, triage_0 n6#7, triage_2
M10)."""
from __future__ import annotations

import math
import statistics as st

SHARE_KILL = 0.40
ROTQ_CLOSE = 0.08
K338_KILL = 1.15
PAD_KILL = 1.05
PAD_E2E_KILL = 0.02
BUF_KILL = 1.05
ATTN_SPEED_KILL = 1.15
ATTN_ERR_KILL = 1.15


def _gm(xs):
    xs = [x for x in xs if x and x > 0]
    return math.exp(sum(map(math.log, xs)) / len(xs)) if xs else None


def _f(x, d = 3):
    return "n/a" if x is None else f"{x:.{d}f}"


def _ok_proc(p):
    return p and p.get("rc") == 0 and not p.get("missing") and not p.get("error") and p.get("rows")


def _arch(h):
    return str(((h.get("kern_smoke") or {}).get("arch")) or "")


def _last_share(prof):
    per = [p for p in prof.get("per_render") or [] if p.get("n")]
    if per:
        return per[-1]["share"], f"rocprofv3, render {len(per) - 1} window ({prof.get('clock')})"
    tp = prof.get("torch_profile")
    if tp:
        return tp["share"], f"torch.profiler, render {tp.get('render')}"
    return None, "no kernel table"


def gates(h: dict, items: list) -> list:
    out = []
    sm = h.get("kern_smoke") or {}
    if "kern_smoke" in items:
        out.append(("kern_smoke gfx1151 + RDNA W8A8 ready", bool(sm.get("ok")),
                    f"arch {sm.get('arch')} prepare {sm.get('prepare')} mm_eq_ref {sm.get('mm_eq_reference_mm')}"))
    if "kern_build" in items:
        b = h.get("kern_build") or {}
        builds = b.get("builds") or {}
        rec = bool(builds) or bool(b.get("error"))
        out.append(("kern_build recorded", rec, "; ".join(f"{k}: ok={v.get('ok')} rc={(v.get('build') or {}).get('rc')}"
                                                          for k, v in builds.items()) or str(b.get("error"))[:200]))
    if "kern_prof" in items:
        p = h.get("kern_prof") or {}
        share, src = _last_share(p)
        arm = p.get("arm") or p.get("arm_torchprof") or {}
        engaged = any((r.get("rdna_mm") or {}).get("large") for r in arm.get("renders") or [])
        out.append(("kern_prof kernel shares of a W8A8 render", share is not None and engaged,
                    f"{src}; w8a8_mm launches seen {engaged}; skipped {p.get('skipped')}"))
    if "kern_a1" in items:
        a = h.get("kern_a1") or {}
        n = sum(1 for x in a.get("procs") or [] if _ok_proc(x))
        out.append(("kern_a1 processes", n >= 3, f"{n}/3 ok"))
    if "kern_bufops" in items:
        b = h.get("kern_bufops") or {}
        arms = [x.get("arm") for x in b.get("procs") or [] if _ok_proc(x)]
        out.append(("kern_bufops processes", arms.count("O") >= 2 and arms.count("N") >= 2, "".join(arms)))
    if "kern_k338" in items:
        k = h.get("kern_k338") or {}
        n = sum(1 for x in k.get("procs") or [] if _ok_proc(x) and not x.get("kitchen_error"))
        why = k.get("skipped") or k.get("not_run")
        out.append(("kern_k338 race or a recorded skip / NOT RUN", n >= 2 or bool(why), why or f"{n} processes ok"))
    if "kern_attn" in items:
        k = h.get("kern_attn") or {}
        n = sum(1 for x in k.get("procs") or [] if _ok_proc(x) and not x.get("kitchen_error")
                and all("kitchen" in (r.get("res") or {}) for r in x.get("rows") or []))
        why = k.get("not_run") or k.get("skipped")
        out.append(("kern_attn race or a recorded NOT RUN", n >= 2 or bool(why), why or f"{n} processes ok"))
    if items and any(i.startswith("kern_") for i in items) and "kern_smoke" in items:
        out.append(("kern items on gfx1151", _arch(h).startswith("gfx1151"), _arch(h)))
    return out


def _prof_lines(h):
    p = h.get("kern_prof") or {}
    L = []
    if not p:
        return L
    L.append(f"kern_prof: source {p.get('source')}; rocprofv3 {p.get('rocprofv3')}; tests "
             f"{[(t.get('tool'), t.get('rc'), t.get('kernel_rows')) for t in p.get('rocprofv3_tests') or []]}")
    arm = p.get("arm") or p.get("arm_torchprof") or {}
    L.append(f"  arm {arm.get('verdict')} load {arm.get('load_s')} s status {arm.get('status')}")
    for r in arm.get("renders") or []:
        L.append(f"  render {r.get('r')}: {r.get('wall_s')} s, {r.get('s_per_step')} s/step, speed {r.get('speed_mode')}"
                 f" {r.get('speed_optims')}, quant {r.get('transformer_quant')}, rdna_mm {r.get('rdna_mm')}")
    for i, w in enumerate(p.get("per_render") or []):
        L.append(f"  window {i}: {w.get('n')} kernels, {_f((w.get('dev_ns') or 0) / 1e9, 2)} s device, share {w.get('share')}")
    if p.get("torch_profile"):
        L.append(f"  torch.profiler: {p['torch_profile'].get('dev_ms')} ms device, share {p['torch_profile'].get('share')}")
    share, src = _last_share(p)
    g = (share or {}).get("w8a8_mm")
    r = (share or {}).get("rotq_rows")
    L.append(f"  VERDICT C9-3 step 0: INT8 GEMM (w8a8_mm) share {_f(g)} ({src}), all GEMM "
             f"{_f(p.get('all_gemm_share'))}: " + ("n/a" if g is None else
                                                  "KILL (under 40%: no #338 race)" if g < SHARE_KILL else
                                                  "passes the 40% gate (race runs)"))
    L.append(f"  VERDICT carbon9 #4 (rotq_rows): share {_f(r)}: " + ("n/a" if r is None else
                                                                  "CLOSE (8% or less)" if r <= ROTQ_CLOSE else
                                                                  "OPEN (over 8%: Triton butterfly next, kill under 1.3x rotq)"))
    return L


def _a1_lines(h):
    a = h.get("kern_a1") or {}
    L = []
    if not a:
        return L
    if a.get("skipped"):
        return [f"kern_a1: skipped ({a['skipped']})"]
    pads, gms, aa, exact = [], {}, [], True
    for p in a.get("procs") or []:
        for row in p.get("rows") or []:
            res = row.get("res") or {}
            st_ = res.get(f"gm{(row.get('studio_cfg') or [0, 0, 0, 8])[3]}") or {}
            if res.get("studio_aa", {}).get("med") and st_.get("med"):
                aa.append(res["studio_aa"]["med"] / st_["med"])
            for k, v in res.items():
                if isinstance(v, dict) and "med" in v and v.get("exact_vs_studio") is False:
                    exact = False
            if res.get("copy_contig", {}).get("med") and st_.get("med"):
                gms.setdefault("copy_contig/studio", []).append(res["copy_contig"]["med"] / st_["med"])
    for b in a.get("best") or []:
        for x in b:
            if x["m"] in (4096, 4352):
                pads.append(x["ratio"])
            gms.setdefault(f"best_gm={x['best_gm']}", []).append(1)
    L.append(f"kern_a1: best-GROUP_M / best-pitch ms ratio geomean (M 4096/4352) {_f(_gm(pads))} over {len(pads)} rows; "
             f"all ratios {[round(x, 3) for x in pads]}")
    L.append(f"  A/A (studio_aa / studio, in-process) min {_f(min(aa) if aa else None)} max {_f(max(aa) if aa else None)};"
             f" copy-of-kernel control {_f(_gm(gms.get('copy_contig/studio', [])))}; best GROUP_M counts "
             f"{ {k: len(v) for k, v in gms.items() if k.startswith('best_gm')} }; all arms torch.equal {exact}")
    e2e = a.get("e2e") or []
    if e2e:
        sps = {arm: [r2.get("s_per_step") for r in e2e if r.get("arm") == arm for r2 in (r.get("renders") or [])[1:]]
               for arm in "MP"}
        md5 = {arm: {r2.get("md5") for r in e2e if r.get("arm") == arm for r2 in r.get("renders") or []} for arm in "MP"}
        m, pp = st.median([x for x in sps["M"] if x] or [0]) or None, st.median([x for x in sps["P"] if x] or [0]) or None
        gain = None if not (m and pp) else (m - pp) / m
        L.append(f"  E2E s/step M {sps['M']} P {sps['P']} gain {_f(gain)}; pixels identical {md5['M'] == md5['P']}")
    g = _gm(pads)
    if g is None:
        v = "n/a"
    elif not exact:
        v = "KILL (torch.equal miss)"
    elif g < PAD_KILL:
        v = "KILL (under 1.05x over the best GROUP_M)"
    elif not e2e:
        v = "kernel passes; E2E missing"
    else:
        v = "KILL (under 2% E2E or pixels differ)" if (gain is None or gain < PAD_E2E_KILL or md5["M"] != md5["P"]) \
            else "PASS (pitch N+128: kernel and E2E clear; needs a copy-free consumer check before shipping)"
    L.append(f"  VERDICT A1 out_pad: {v}")
    return L


def _buf_lines(h):
    b = h.get("kern_bufops") or {}
    L = []
    if not b:
        return L
    if b.get("skipped"):
        return [f"kern_bufops: skipped ({b['skipped']})"]
    by: dict = {}
    shas: dict = {}
    for p in b.get("procs") or []:
        for r in p.get("rows") or []:
            key = (r["shape"], r["m"])
            med = ((r.get("res") or {}).get("studio") or {}).get("med")
            if med:
                by.setdefault(key, {}).setdefault(p.get("arm"), []).append(med)
            shas.setdefault(key, set()).add(r.get("sha"))
    ratios, aa = [], []
    for key, d in sorted(by.items()):
        o, n = d.get("O") or [], d.get("N") or []
        if o and n:
            ratios.append(st.median(o) / st.median(n))
            aa.append((max(o) - min(o)) / st.median(o))
            L.append(f"  {key}: off {[round(x, 3) for x in o]} on {[round(x, 3) for x in n]} ms -> {st.median(o) / st.median(n):.3f}x")
    exact = all(len(s) == 1 for s in shas.values())
    g = _gm(ratios)
    L.insert(0, f"kern_bufops: off/on geomean {_f(g)} over {len(ratios)} shapes; off-arm spread max {_f(max(aa) if aa else None)};"
                f" outputs identical across arms {exact}")
    v = "n/a" if g is None else "KILL (mismatch)" if not exact else \
        "KILL (under 5% kernel)" if g < BUF_KILL else "PASS (5% or more, bit-identical)"
    L.append(f"  VERDICT n6#7 AMDGCN_USE_BUFFER_OPS=1: {v}")
    return L


def _k338_lines(h):
    k = h.get("kern_k338") or {}
    L = []
    if not k:
        return L
    if k.get("skipped") or k.get("not_run"):
        return [f"kern_k338: {k.get('skipped') or k.get('not_run')}",
                f"  VERDICT C9-3 #338: {'KILL (GEMM share gate)' if k.get('skipped') else 'NOT RUN'}"]
    by: dict = {}
    exact_ref, exact_studio, same_as_main = True, True, True
    sha: dict = {}
    for p in k.get("procs") or []:
        for r in p.get("rows") or []:
            res = r.get("res") or {}
            s, kk = (res.get("studio") or {}).get("med"), (res.get("kitchen") or {}).get("med")
            key = (r["shape"], r["m"])
            if s and kk:
                by.setdefault(key, {}).setdefault(p.get("arm"), []).append(s / kk)
                by[key].setdefault("studio_ms", []).append(s)
            if r.get("kitchen_eq_reference_mm") is False:
                exact_ref = False
            if r.get("kitchen_eq_studio") is False:
                exact_studio = False
            sha.setdefault(key, {}).setdefault(p.get("arm"), set()).add(r.get("kitchen_sha"))
    for key, d in sha.items():
        if "K" in d and "M" in d and d["K"] != d["M"]:
            same_as_main = False
    k_r = [st.median(d["K"]) for d in by.values() if d.get("K")]
    m_r = [st.median(d["M"]) for d in by.values() if d.get("M")]
    aa = [(max(d["studio_ms"]) - min(d["studio_ms"])) / st.median(d["studio_ms"]) for d in by.values()
          if len(d.get("studio_ms", [])) > 1]
    g = _gm(k_r)
    L.append(f"kern_k338: Studio/#338 kernel ms geomean {_f(g)} over {len(k_r)} shapes (kitchen main {_f(_gm(m_r))});"
             f" Studio ms cross-process spread max {_f(max(aa) if aa else None)}")
    for key, d in sorted(by.items()):
        L.append(f"  {key}: #338 {[round(x, 3) for x in d.get('K', [])]} main {[round(x, 3) for x in d.get('M', [])]}")
    L.append(f"  #338 torch.equal reference_mm {exact_ref}, torch.equal Studio {exact_studio}, bit-identical to kitchen "
             f"main {same_as_main}")
    v = "n/a" if g is None else ("KILL (mismatch vs reference_mm" + (
        ", the same bytes as kitchen main: its epilogue order, not #338" if same_as_main else "") + ")") \
        if not exact_ref else \
        "KILL (under 1.15x kernel geomean over Studio)" if g < K338_KILL else "PASS (1.15x or more, exact)"
    L.append(f"  VERDICT C9-3 #338: {v}")
    L.extend(_tile_lines(k))
    return L


def _tile_lines(k):
    """Round-9 fold-in: Studio w8a8_mm at 256x128x64 / 128x256x64 vs the Studio cfg, per (shape, m), same race."""
    by, eq, errs = {}, {}, {}
    for p in k.get("procs") or []:
        for r in p.get("rows") or []:
            res = r.get("res") or {}
            s = (res.get("studio") or {}).get("med")
            for t in ("t256x128x64", "t128x256x64"):
                tm = (res.get(t) or {}).get("med")
                if s and tm:
                    by.setdefault(t, {}).setdefault((r["shape"], r["m"]), []).append(s / tm)
                if (r.get("tile_eq_reference_mm") or {}).get(t) is False:
                    eq[t] = False
                if (r.get("tile_error") or {}).get(t):
                    errs.setdefault(t, r["tile_error"][t][:120])
    L = []
    for t in ("t256x128x64", "t128x256x64"):
        d = by.get(t) or {}
        meds = [st.median(v) for v in d.values()]
        g = _gm(meds)
        exact = eq.get(t, True)
        L.append(f"  tile {t}: Studio-cfg ms / tile ms geomean {_f(g)} over {len(meds)} shapes, best "
                 f"{_f(max(meds) if meds else None)}; torch.equal reference_mm {exact}; compile error {errs.get(t)}")
        v = "NOT RUN" if g is None else "KILL (mismatch)" if not exact else \
            "KILL (no shape 5% or more faster)" if max(meds) < 1.05 else "KEEP (a shape 5% or more faster, exact)"
        L.append(f"  VERDICT C9-3 fold {t}: {v}")
    return L


def _attn_lines(h):
    k = h.get("kern_attn") or {}
    L = []
    if not k:
        return L
    if k.get("not_run") or k.get("skipped"):
        return [f"kern_attn: {k.get('not_run') or k.get('skipped')}", "  VERDICT M10: NOT RUN"]
    by: dict = {}
    for p in k.get("procs") or []:
        for r in p.get("rows") or []:
            res = r.get("res") or {}
            for arm in ("studio", "kitchen"):
                if (res.get(arm) or {}).get("med_ms"):
                    d = by.setdefault(r["S"], {}).setdefault(arm, {"ms": [], "err": []})
                    d["ms"].append(res[arm]["med_ms"])
                    d["err"].append(res[arm].get("rel_l2"))
            if r.get("kitchen_error"):
                by.setdefault(r["S"], {})["kitchen_error"] = r["kitchen_error"]
    worst_speed, worst_err = None, None
    for S, d in sorted(by.items()):
        s, kk = d.get("studio"), d.get("kitchen")
        if not (s and kk):
            L.append(f"  S {S}: incomplete {d.get('kitchen_error', '')[:200]}")
            continue
        sp = st.median(s["ms"]) / st.median(kk["ms"])
        er = st.median([x for x in kk["err"] if x is not None]) / max(1e-12, st.median([x for x in s["err"] if x is not None]))
        worst_speed = sp if worst_speed is None else min(worst_speed, sp)
        worst_err = er if worst_err is None else max(worst_err, er)
        L.append(f"  S {S}: studio ms {[round(x, 2) for x in s['ms']]} kitchen ms {[round(x, 2) for x in kk['ms']]} -> "
                 f"{sp:.3f}x; rel_l2 studio {st.median(s['err']):.5f} kitchen {st.median(kk['err']):.5f} ({er:.2f}x)")
    L.insert(0, f"kern_attn: worst speedup {_f(worst_speed)}, worst error ratio {_f(worst_err)} (kitchen / Studio)")
    v = "n/a" if worst_speed is None else "KILL (under 15% faster)" if worst_speed < ATTN_SPEED_KILL else \
        "KILL (error over 1.15x Studio)" if worst_err > ATTN_ERR_KILL else "PASS"
    L.append(f"  VERDICT M10 kitchen INT8 attention: {v} (per-shape: a shape under 1.15x counts as a kill)")
    return L


def table(h: dict) -> list:
    L = []
    sm = h.get("kern_smoke") or {}
    if sm:
        L.append(f"kern_smoke: arch {sm.get('arch')} torch {sm.get('torch')} hip {sm.get('hip')} triton {sm.get('triton')}"
                 f" prepare {sm.get('prepare')} mm==ref {sm.get('mm_eq_reference_mm')} roots "
                 f"{[(r.get('label'), r.get('hip_cmake'), bool(r.get('clang')), r.get('rocprofv3')) for r in sm.get('rocm_roots') or []]}")
    b = h.get("kern_build") or {}
    if b:
        L.append(f"kern_build: root {(b.get('rocm_root'))}; " + "; ".join(
            f"{k} ok={v.get('ok')} {((v.get('build') or {}).get('s'))} s" for k, v in (b.get("builds") or {}).items())
            + (f"; error {str(b.get('error'))[:200]}" if b.get("error") else ""))
    for fn in (_prof_lines, _a1_lines, _buf_lines, _k338_lines, _attn_lines):
        try:
            L += fn(h)
        except Exception as exc:  # noqa: BLE001 - a table bug must not hide the other items
            L.append(f"{fn.__name__}: table error {exc!r}")
    return L
