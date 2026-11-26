#!/usr/bin/env python3
"""Judge helpers for probes/r8_lnx_run.py (C15-6, M8 a-d, C18-6A). gates() proves each item ran on the device and
produced its numbers; table() renders the numbers and one verdict line per item against its pre-registered kill line.
Verdict words: KILL (kill line hit), PASS (kill line not hit: candidate survives), VOID (the setup cannot answer),
NOT RUN, INCOMPLETE."""
from __future__ import annotations

import math
import statistics as st

SPIN_KILL = 0.05


def _ok(r) -> bool:
    return bool(r) and not r.get("error") and r.get("rc") == 0 and not r.get("missing")


def _med(xs):
    xs = [x for x in xs if isinstance(x, (int, float)) and x > 0]
    return st.median(xs) if xs else None


def _spread(xs):
    xs = [x for x in xs if isinstance(x, (int, float)) and x > 0]
    if len(xs) < 2:
        return None
    m = st.median(xs)
    return (max(xs) - min(xs)) / m if m else None


def _f(x, nd = 3):
    if x is None:
        return "n/a"
    if isinstance(x, float) and math.isinf(x):
        return "inf"
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


# ------------------------------------------------------------------------------------------------------- verdicts
def v_c156(h):
    s = h.get("c156_spin") or {}
    rows = s.get("rows") or []
    a = [r.get("core_frac") for r in rows if r.get("arm") == "A" and r.get("ok")]
    b = [r.get("core_frac") for r in rows if r.get("arm") == "B" and r.get("ok")]
    if len(a) < 3:
        return "INCOMPLETE", f"{len(a)} unset reps ok"
    med_a, med_b = st.median(a), (st.median(b) if b else None)
    nums = f"idle core fraction unset {[round(x, 4) for x in a]} (median {med_a:.4f}), =1 {[round(x, 4) for x in b]}"
    if med_a < SPIN_KILL:
        return "KILL", f"{nums}: under 5% of a core without the variable (no spin on this ROCm pin)"
    rr = s.get("renders") or []
    wa = [_med([x.get("wall_s") for x in (r.get("renders") or [])]) for r in rr if r.get("arm") == "A"]
    wb = [_med([x.get("wall_s") for x in (r.get("renders") or [])]) for r in rr if r.get("arm") == "B"]
    prof = (s.get("rocprof") or {}).get("checks") or []
    pb = next((c for c in prof if c.get("arm") == "B"), {})
    return "PASS", (f"{nums}: spins; Wan render median s A {wa} B {wb}; rocprofv3 with the variable: rc {pb.get('rc')}"
                    f" kernel rows {pb.get('kernel_rows')}")


def v_m8b(h):
    s = h.get("m8b_mmap") or {}
    rows = s.get("micro") or []
    if not rows:
        return "NOT RUN", ""
    def frac(arm, mib = None):
        return [r.get("frac_Anonymous") for r in rows if r.get("arm") == arm and r.get("ok")
                and (mib is None or r.get("mib") == mib)]
    nums = (f"anon delta / mapped: copy 64M {frac('copy', 64)} read 64M {frac('read', 64)}; copy 1G {frac('copy', 1024)}"
            f" read 1G {frac('read', 1024)}")
    if not frac("copy"):
        return "INCOMPLETE", f"{nums}: no ACCESS_COPY micro arm finished"
    if s.get("void"):
        return "VOID", f"{nums}: ACCESS_COPY upload under the 1 GiB-equivalent on gfx1151 UMA (no CoW leak to fix)"
    loads = s.get("loads") or []
    c = [r.get("load_s") for r in loads if r.get("arm") == "copy" and r.get("verdict") == "ok"]
    rd = [r.get("load_s") for r in loads if r.get("arm") == "read" and r.get("verdict") == "ok"]
    calls = [((r.get("side") or {}).get("map_calls")) for r in loads]
    peaks = {r.get("arm"): (r.get("side") or {}).get("peak", {}).get("RssAnon") for r in loads}
    if not c or not rd:
        return "INCOMPLETE", f"{nums}; loads ok copy {len(c)} read {len(rd)}"
    if not any(calls):
        return "VOID", f"{nums}; the Qwen-2.1 load never mapped through _map_tensor (calls {calls})"
    mc, mr = st.median(c), st.median(rd)
    reg = (mr - mc) / mc
    line = (f"{nums}; load s copy {c} read {rd} (read {reg:+.1%}, A/A copy {_f(_spread(c))}); map calls {calls};"
            f" RssAnon peak (last per arm) {peaks}")
    return ("KILL" if reg > 0.02 else "PASS"), line


def v_m8a(h):
    s = h.get("m8a_wankitchen") or {}
    if s.get("skipped"):
        return "NOT RUN", s["skipped"]
    rows = s.get("rows") or []
    st_ = [r.get("med_s") for r in rows if r.get("arm") == "stock" and _ok(r)]
    ki = [r.get("med_s") for r in rows if r.get("arm") == "kitchen" and _ok(r)]
    if len(st_) < 2 or len(ki) < 2:
        return "INCOMPLETE", f"ok stock {len(st_)} kitchen {len(ki)}"
    eng = next((r.get("calls_per_decode") for r in rows if r.get("arm") == "kitchen" and _ok(r)), {})
    if not eng or not all(eng.get(k) for k in ("norm_silu", "d2s", "interleave_up")):
        return "VOID", f"kitchen ops not all engaged: {eng}"
    q = s.get("psnr") or {}
    p = (q.get("kitchen_vs_stock") or {}).get("psnr")
    gain = 1 - st.median(ki) / st.median(st_)
    aa = _spread(st_)
    line = (f"decode s stock {[round(x, 2) for x in st_]} kitchen {[round(x, 2) for x in ki]}: {gain:+.1%} "
            f"(A/A stock {_f(aa)}); engagement {eng}; PSNR kitchen vs stock {_f(p, 2)} dB, stock A/A "
            f"{_f((q.get('stock_aa') or {}).get('psnr'), 2)}, vs fp32: stock {_f((q.get('stock_vs_fp32') or {}).get('psnr'), 2)}"
            f" kitchen {_f((q.get('kitchen_vs_fp32') or {}).get('psnr'), 2)}")
    if p is None:
        return "INCOMPLETE", line + " (no PSNR)"
    kill = gain < 0.10 or (aa is not None and gain < aa + 0.05) or p < 55.0
    return ("KILL" if kill else "PASS"), line


def v_m8c(h):
    s = h.get("m8c_h3enc") or {}
    if s.get("skipped"):
        return "NOT RUN", s["skipped"]
    rows = s.get("rows") or []
    st_ = [r.get("med_s") for r in rows if r.get("arm") == "stock" and _ok(r)]
    lo = [r.get("med_s") for r in rows if r.get("arm") == "lowered" and _ok(r)]
    if len(st_) < 2 or len(lo) < 2:
        return "INCOMPLETE", f"ok stock {len(st_)} lowered {len(lo)}"
    eng = next((r.get("calls_per_encode") for r in rows if r.get("arm") == "lowered" and _ok(r)), {})
    if not (eng or {}).get("lowered"):
        return "VOID", f"no conv took the lowering: {eng}"
    q = s.get("psnr") or {}
    p_ref = (q.get("lowered_vs_fp32") or {}).get("psnr")
    p_st = (q.get("stock_vs_fp32") or {}).get("psnr")
    gain = 1 - st.median(lo) / st.median(st_)
    line = (f"encode s stock {[round(x, 3) for x in st_]} lowered {[round(x, 3) for x in lo]}: {gain:+.1%} (A/A "
            f"{_f(_spread(st_))}); calls {eng}; PSNR vs fp32: lowered {_f(p_ref, 2)} stock {_f(p_st, 2)} dB; "
            f"lowered vs stock {_f((q.get('lowered_vs_stock') or {}).get('psnr'), 2)}")
    if p_ref is None:
        return "INCOMPLETE", line + " (no fp32 reference)"
    return ("KILL" if gain < 0.15 or p_ref < 60.0 else "PASS"), line


def v_c186a(h):
    s = h.get("c186a_train") or {}
    if s.get("skipped"):
        return "NOT RUN", s["skipped"]
    rows = s.get("rows") or []

    def arm(prec):
        return [r for r in rows if r.get("precision") == prec and r.get("complete") and r.get("steady_s_per_step")]
    b, n, i = arm("bf16"), arm("nf4"), arm("int8")
    sps = {k: [round(r["steady_s_per_step"], 3) for r in v] for k, v in (("bf16", b), ("nf4", n), ("int8", i))}
    fails = {str(r.get("precision")) + str(r.get("seed")): (r.get("errors") or [r.get("raised") or r.get("error")])[:1]
             for r in rows if not r.get("complete")}
    if not i:
        return "INCOMPLETE", f"int8 arm did not finish: {fails}"
    if not any(r.get("w8a8_line") for r in i) or any(r.get("w8a8_fallback") for r in i):
        return "VOID", f"W8A8 not engaged on the int8 arm: {[(r.get('w8a8_line'), r.get('w8a8_fallback')) for r in i]}"
    meds = [_med([r["steady_s_per_step"] for r in v]) for v in (b, n) if v]
    best = min(meds) if meds else None
    if best is None:
        return "INCOMPLETE", f"no bf16 / nf4 arm finished: {fails}"
    mi = _med([r["steady_s_per_step"] for r in i])
    speed = best / mi

    def tail(r):
        ls = [x for x in (r.get("losses") or [])[10:] if isinstance(x, (int, float))]
        return sum(ls) / len(ls) if ls else None
    bt = [tail(r) for r in b if tail(r) is not None]
    it = [tail(r) for r in i if tail(r) is not None]
    env_ok, env_txt = None, "no 2-seed bf16 envelope"
    if len(bt) >= 2 and it:
        width = max(bt) - min(bt)
        lo, hi = min(bt) - width, max(bt) + width
        env_ok = all(lo <= x <= hi for x in it)
        env_txt = f"tail loss bf16 {[round(x, 4) for x in bt]} int8 {[round(x, 4) for x in it]} envelope [{lo:.4f}, {hi:.4f}]"
    peaks = {str(r.get("precision")) + str(r.get("seed")): r.get("max_allocated_gib") for r in rows}
    comp = {str(r.get("precision")) + str(r.get("seed")): r.get("compiled") for r in rows}
    line = (f"steady s/step {sps} (A/A spread bf16 {_f(_spread(sps['bf16']))} int8 {_f(_spread(sps['int8']))}); best "
            f"bf16/nf4 {best:.3f} vs int8 {mi:.3f} = {speed:.3f}x; {env_txt}; peak GiB {peaks}; compiled {comp}; "
            f"failed {fails}")
    if speed < 1.15 or env_ok is False:
        return "KILL", line
    return ("PASS" if env_ok else "INCOMPLETE"), line


def v_m8d(h):
    s = h.get("m8d_qvae") or {}
    if s.get("skipped"):
        return "NOT RUN", s["skipped"]
    rows = s.get("rows") or []
    by = {}
    for r in rows:
        by.setdefault(r.get("side"), []).append(r)
    done = {k: sum(1 for r in v if r.get("stage") == "done" and r.get("finite") and r.get("rc") == 0) for k, v in by.items()}
    hangs = [(r.get("side"), r.get("i")) for r in rows if r.get("rc") == "timeout"]
    secs = {k: [r.get("decode_s") for r in v] for k, v in by.items()}
    md5 = {k: len({r.get("md5") for r in v if r.get("md5")}) for k, v in by.items()}
    line = f"finished+finite {done} of 5 per side; hangs {hangs}; decode s {secs}; distinct md5 per side {md5}"
    if len(rows) < 10 and not s.get("stopped"):
        return "INCOMPLETE", line
    if done.get(1024) == 5 and done.get(1280) == 5:
        return "KILL", line + ": all 10 finished without a hang (no gfx1151 tiling guard needed)"
    if not hangs and not s.get("stopped"):
        # errors that are not hangs (download, import, OOM) answer nothing about the hang the kill line is about
        return "INCOMPLETE", line + ": runs failed without a hang (not the defect under test)"
    return "PASS", line + f": not all finished ({s.get('stopped') or 'hangs'}): guard needed"


VERDICTS = {"c156_spin": ("C15-6", v_c156), "m8b_mmap": ("M8(b)", v_m8b), "m8a_wankitchen": ("M8(a)", v_m8a),
            "m8c_h3enc": ("M8(c)", v_m8c), "c186a_train": ("C18-6A", v_c186a), "m8d_qvae": ("M8(d)", v_m8d)}


def gates(h: dict, items: list) -> list:
    out = []
    for item in items:
        if item not in VERDICTS:
            continue
        name, fn = VERDICTS[item]
        if f"{item}_error" in h:
            out.append((f"{name} ran", False, str(h[f"{item}_error"])[-300:]))
            continue
        if item not in h:
            out.append((f"{name} ran", False, "no observation"))
            continue
        try:
            v, _ = fn(h)
        except Exception as exc:  # noqa: BLE001
            out.append((f"{name} judged", False, repr(exc)[:200]))
            continue
        out.append((f"{name} produced a verdict", v in ("KILL", "PASS", "VOID", "NOT RUN"), v))
    return out


def table(h: dict) -> list:
    lines = []
    for item, (name, fn) in VERDICTS.items():
        if item not in h and f"{item}_error" not in h:
            continue
        try:
            v, txt = fn(h)
        except Exception as exc:  # noqa: BLE001
            v, txt = "INCOMPLETE", f"judge error {exc!r}"
        lines.append(f"{name} {v}: {txt}")
    return lines
