#!/usr/bin/env python3
"""Criteria for probes/rv28_probe.py (nov28 revisit of four gfx1151 levers). Arms differ by tree / env inside the head
step, so nothing is judged between states: the gates prove every requested item ran on gfx1151 with its arms and its
engagement counters; the per-item verdict lines (rules fixed before the run, below) are computed from the observations.

Rules:
  h3vae  SHIP iff median decode T <= S / 1.10, the S1/S2 and T1/T2 A/A spreads are both below the gain, PSNR(T, R) >= 50
         dB vs the fp32 decode, no non-finite output and the decoder fast path did not fall back.
  q21    report the end-to-end wall ratio B / K (3 runs each, A/A = spread of the B runs) and whether every K render is
         byte-identical to the same-index B render; kitchen calls > 0 or VOID.
  klein  end-to-end gain 1 - Q / B with fused calls > 0; only when >= 3%: paired d_i = LPIPS(Q_i, R_i) - LPIPS(B_i, R_i),
         PASS iff the bootstrap upper 95% of mean(d) <= max(0.01, upper 95% of the kernel-swap arm F's mean d).
  wan    speed P vs H (2 processes each, A/A = H1 vs H2), reserved memory delta; paired d_i = LPIPS(P_i, R_i) -
         LPIPS(H_i, R_i) at 8 clips, PASS iff upper 95% <= max(0.01, upper 95% of either kernel-swap arm, F TunableOp off
         or K rocBLAS); the A/A arm H2 must give d = 0 (deterministic) or the item is VOID.
"""

from __future__ import annotations

import random
import statistics as st

TITLE = "gfx1151 nov28 revisit: H3 VAE Triton, Wan weight pitch, C9-3 kitchen INT8 GEMM, FLUX.3 fused QK-norm+RoPE"
MODE = "regression"
NEEDS = ["gpu", "rocm", "exclusive_gpu", "windows", "nvidia", "multi_gpu"]


def _ok(rec):
    return bool(rec) and rec.get("verdict") == "ok" and rec.get("rc") == 0


def _med(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return round(st.median(xs), 3) if xs else None


def _walls(rec, key = "wall_s"):
    return [r.get(key) for r in (rec or {}).get("renders") or []]


def _boot_upper(ds, n = 20000, seed = 0):
    ds = [d for d in ds if isinstance(d, (int, float))]
    if len(ds) < 2:
        return None
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(ds) for _ in ds) / len(ds) for _ in range(n))
    return round(means[int(0.975 * n)], 5)


def _vs_ref(scores, fam, tag):
    for r in (scores or {}).get("vs_ref") or []:
        if r.get("a") == f"{fam}_{tag}":
            return {row["i"]: row.get("lpips") for row in r.get("rows") or []}
    return {}


def _paired(scores, fam, a, b):
    pa, pb = _vs_ref(scores, fam, a), _vs_ref(scores, fam, b)
    ds = [pa[i] - pb[i] for i in sorted(pa) if i in pb and pa[i] is not None and pb[i] is not None]
    return {"n": len(ds), "mean": round(sum(ds) / len(ds), 5) if ds else None, "upper95": _boot_upper(ds),
            "a_mean": round(st.mean([v for v in pa.values() if v is not None]), 5) if pa else None}


def gates(obs: dict) -> list:
    h = obs.get("head") or {}
    arch = (h.get("env") or {}).get("arch") or ""
    items = [x for x in ((h.get("args") or {}).get("items") or "").split(",") if x]
    out = [("gfx1151", arch.startswith("gfx1151"), arch)]
    for item in items:
        if item == "smoke":
            out.append(("smoke pytest", bool(h.get("smoke_ok")), f"rc {(h.get('smoke_pytest') or {}).get('rc')}"))
        elif item == "h3vae":
            rows = h.get("h3vae") or []
            ok = [r for r in rows if r.get("rc") == 0 and not r.get("error") and not r.get("missing")]
            out.append(("h3vae arms ok", len(ok) == len(rows) and len(rows) >= 3, f"{len(ok)} of {len(rows)}"))
            t = [r for r in ok if r.get("arm") == "fast"]
            out.append(("h3vae fast path engaged", bool(t) and all(r.get("fast_path_available") and "fused_decoder"
                        in (r.get("levers") or []) and not r.get("decoder_fast_failed") for r in t),
                        str([(r.get("tag"), r.get("levers"), r.get("decoder_fast_failed")) for r in t])))
        elif item in ("q21", "klein", "wan"):
            arms = h.get(item) or {}
            n = sum(1 for r in arms.values() if _ok(r))
            out.append((f"{item} arms ok", n == len(arms) and n >= 2, f"{n} of {len(arms)} {h.get(item + '_error', '')}"))
            if item == "q21":
                k = [r.get("kitchen_race") or {} for t, r in arms.items() if t.startswith("K")]
                calls = [x.get("kitchen_calls", 0) for x in k]
                out.append(("q21 kitchen kernel engaged", bool(calls) and all(c > 0 for c in calls), str(calls)))
            if item == "klein":
                q = [((r.get("flux3_qk") or {}).get("stats") or {}).get("fused", 0) for t, r in arms.items()
                     if t.startswith("Q")]
                out.append(("klein fused QK engaged", bool(q) and all(c > 0 for c in q), str(q)))
            if item == "wan":
                p = [((r.get("ab_counts") or {}).get("diffusion_rocm_weight_pitch") or {}) for t, r in arms.items()
                     if t.startswith("P")]
                out.append(("wan pitch counters read", bool(p), str(p)[:300]))
                ref = arms.get("R1") or {}
                out.append(("wan reference ran in fp32", bool(ref.get("fp32_denoiser")) and all(
                    d.get("dtypes") == ["torch.float32"] for d in ref.get("fp32_denoiser") or []),
                    str(ref.get("fp32_denoiser"))[:300]))
    return out


def head_is_worse(base: dict, head: dict) -> tuple:
    return False, "measurement only (arms differ by tree / env inside the head step)"


def _h3vae(h, L):
    rows = h.get("h3vae") or []
    by = {}
    for r in rows:
        by.setdefault(r.get("arm"), []).append(r)
    for r in rows:
        L.append(f"h3vae {r.get('tag')} {r.get('arm')}: s {r.get('s')} first {r.get('first_s')} peak {r.get('peak_gib')} "
                 f"levers {r.get('levers')} nonfinite {r.get('nonfinite')} err {str(r.get('error'))[:200]}")
    L.append(f"h3vae PSNR vs fp32 R {h.get('h3vae_psnr_vs_R')}; vs S1 {h.get('h3vae_psnr_vs_S1')}")
    s = [r.get("med_s") for r in by.get("stock", [])]
    t = [r.get("med_s") for r in by.get("fast", [])]
    if s and t and all(s) and all(t):
        sm, tm = st.median(s), st.median(t)
        aa_s = (max(s) - min(s)) / sm if len(s) > 1 else None
        aa_t = (max(t) - min(t)) / tm if len(t) > 1 else None
        sp = sm / tm
        ps = h.get("h3vae_psnr_vs_R") or {}
        pt = [v for k, v in ps.items() if k.startswith("T") and v is not None]
        fin = all((r.get("nonfinite") == 0) for r in rows if r.get("arm") in ("stock", "fast"))
        ok = (sp >= 1.10 and pt and min(pt) >= 50 and fin and aa_s is not None and aa_t is not None
              and max(aa_s, aa_t) < (sp - 1))
        L.append(f"VERDICT h3vae: decode S {sm:.2f} s vs T {tm:.2f} s = {sp:.3f}x; A/A S {aa_s} T {aa_t}; "
                 f"PSNR(T, fp32) min {min(pt) if pt else None}; finite {fin}: {'SHIP' if ok else 'KILL'}")


def _q21(h, L):
    arms = h.get("q21") or {}
    for t, r in arms.items():
        L.append(f"q21 {t}: walls {_walls(r)} denoise {_walls(r, 'denoise_s')} race "
                 f"{ {k: (v.get('pick'), v.get('equal'), v.get('studio_ms'), v.get('kitchen_ms')) for k, v in ((r.get('kitchen_race') or {}).get('keys') or {}).items()} } "
                 f"calls k/s {(r.get('kitchen_race') or {}).get('kitchen_calls')}/{(r.get('kitchen_race') or {}).get('studio_calls')} "
                 f"late {(r.get('kitchen_race') or {}).get('late_races')} err {str(r.get('error'))[:200]}")
    b = [_med(_walls(r)) for t, r in arms.items() if t.startswith("B") and _ok(r)]
    k = [_med(_walls(r)) for t, r in arms.items() if t.startswith("K") and _ok(r)]
    shas = h.get("q21_shas") or {}
    ident = None
    if "B1" in shas:
        ident = all(shas[t] == shas["B1"] for t in shas if t.startswith("K")) and bool(shas["B1"])
    if b and k and all(b) and all(k):
        bm, km = st.median(b), st.median(k)
        aa = (max(b) - min(b)) / bm if len(b) > 1 else None
        L.append(f"VERDICT q21 C9-3: render wall B {b} (median {bm:.2f}) vs K {k} (median {km:.2f}) = {bm / km:.3f}x "
                 f"end to end; A/A spread of B {aa}; outputs byte-identical to B {ident}")


def _klein(h, L):
    arms = h.get("klein") or {}
    for t, r in arms.items():
        L.append(f"klein {t}: walls {_walls(r)} denoise {_walls(r, 'denoise_s')} qk {r.get('flux3_qk', {}).get('stats')} "
                 f"lift {r.get('flux3_qk', {}).get('lift_diff')} speed {(r.get('status') or {}).get('speed_optims')} "
                 f"err {str(r.get('error'))[:200]}")
    b = [_med(_walls(r)) for t, r in arms.items() if t.startswith("B") and _ok(r)]
    q = [_med(_walls(r)) for t, r in arms.items() if t.startswith("Q") and _ok(r)]
    if b and q and all(b) and all(q):
        gain = 1 - st.median(q) / st.median(b)
        aa = (max(b) - min(b)) / st.median(b) if len(b) > 1 else None
        line = f"VERDICT klein FLUX.3 QK: wall B {b} Q {q} gain {gain * 100:.2f}% (A/A B {aa})"
        sc = h.get("klein_scores")
        if sc and gain >= 0.03:
            d = _paired(sc, "klein", "Q1", "B1")
            f = _paired(sc, "klein", "F1", "B1")
            bar = max(0.01, f["upper95"] or 0)
            line += f"; d(Q) {d}; floor d(F) {f}; {'PASS' if d['upper95'] is not None and d['upper95'] <= bar else 'FAIL'}"
        elif gain < 0.03:
            line += ": under the 3% end-to-end line, not pursued (KILL)"
        L.append(line)


def _wan(h, L):
    arms = h.get("wan") or {}
    for t, r in arms.items():
        L.append(f"wan {t}: walls {_walls(r)} denoise {_walls(r, 'denoise_s')} reserved {_med(_walls(r, 'reserved_gib'))} "
                 f"alloc {_med(_walls(r, 'alloc_gib'))} pitch {(r.get('ab_counts') or {}).get('diffusion_rocm_weight_pitch')} "
                 f"speed {(r.get('status') or {}).get('speed_optims')} err {str(r.get('error'))[:200]}")
    sc = h.get("wan_scores") or {}
    for r in sc.get("vs_ref") or []:
        L.append(f"wan LPIPS {r['a']} vs {r['b']}: mean {r.get('lpips_mean')} max {r.get('lpips_max')} "
                 f"psnr {r.get('psnr_mean')} n {r.get('n')}")
    for r in sc.get("pairs") or []:
        L.append(f"wan pair {r['a']} vs {r['b']}: lpips {r.get('lpips_mean')} identical "
                 f"{[x.get('identical') for x in r.get('rows') or []]}")
    hw = [_med(_walls(arms[t], "denoise_s")) for t in ("H1", "H2") if _ok(arms.get(t))]
    pw = [_med(_walls(arms[t], "denoise_s")) for t in ("P1", "P2") if _ok(arms.get(t))]
    hwall = [_med(_walls(arms[t])) for t in ("H1", "H2") if _ok(arms.get(t))]
    pwall = [_med(_walls(arms[t])) for t in ("P1", "P2") if _ok(arms.get(t))]
    if hw and pw and all(hw) and all(pw):
        aa = (max(hw) - min(hw)) / st.median(hw) if len(hw) > 1 else None
        res = [_med(_walls(arms[t], "reserved_gib")) for t in ("H1", "P1") if _ok(arms.get(t))]
        L.append(f"wan speed: denoise H {hw} P {pw} -> {(1 - st.median(pw) / st.median(hw)) * 100:.2f}% less; wall H "
                 f"{hwall} P {pwall}; A/A H {aa}; reserved H/P {res}")
    if sc.get("vs_ref"):
        d = _paired(sc, "wan", "P1", "H1")
        f = _paired(sc, "wan", "F1", "H1")
        k = _paired(sc, "wan", "K1", "H1")
        aa = _paired(sc, "wan", "H2", "H1")
        bar = max(0.01, f["upper95"] or 0, k["upper95"] or 0)
        void = aa["mean"] not in (0, 0.0, None) and abs(aa["mean"]) > 1e-4
        v = "VOID (A/A not deterministic)" if void else ("PASS" if d["upper95"] is not None and d["upper95"] <= bar
                                                         else "FAIL")
        L.append(f"VERDICT wan pitch quality: d(P) {d}; floors d(F TunableOp off) {f}, d(K rocBLAS) {k}; A/A {aa}; "
                 f"bar {bar}: {v}")


def table(obs: dict) -> str:
    h = obs.get("head") or {}
    L = [f"env {h.get('env')}", f"item seconds {h.get('item_s')}", f"smoke {h.get('smoke_ok')}",
         f"errors { {k: str(v)[-300:] for k, v in h.items() if k.endswith('_error') or k.endswith('_skipped')} }", ""]
    for fn in (_h3vae, _q21, _klein, _wan):
        try:
            fn(h, L)
        except Exception as exc:  # noqa: BLE001
            L.append(f"{fn.__name__} table error {exc!r}")
        L.append("")
    return "```\n" + "\n".join(L) + "\n```"
