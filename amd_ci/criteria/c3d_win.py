#!/usr/bin/env python3
"""Criteria for probes/c3d_probe.py (PR 19873, ROCm Triton conv3d for the Wan VAE, gfx1151).

Arms differ by tree / env inside the head step (A merge base, B old head, C new head, D new head + Winograd), so
the gates prove each item ran with engagement (routed > 0 for C / D, 0 for A); head_is_worse flags C when its
per-conv bf16 error exceeds 1.05x MIOpen's or its warm decode is slower than A beyond A's own spread."""

from __future__ import annotations

import statistics as st

TITLE = "PR 19873 WINDOWS: Triton conv3d Wan 2.2 VAE decode on gfx1151 (A base / C head defaults / D head + Winograd)"
MODE = "regression"
NEEDS = ["gpu", "rocm", "exclusive_gpu", "windows", "nvidia", "multi_gpu"]


def _h(obs):
    return obs.get("head") or {}


def _ok(rec):
    return rec.get("rc") == 0 and not rec.get("error") and not rec.get("missing")


def _warm(h, arm, res):
    return [r["res"][res] for t, r in (h.get("warm") or {}).items()
            if r.get("arm") == arm and _ok(r) and res in (r.get("res") or {})]


def _med(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return round(st.median(xs), 3) if xs else None


def _spread(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    if len(xs) < 2:
        return None
    m = st.median(xs)
    return round(100 * (max(xs) - min(xs)) / m, 2) if m else None


def _acc_rows(h):
    return [r for r in ((h.get("acc") or {}).get("rows") or []) if not r.get("error") and r.get("miopen")]


def gates(obs: dict) -> list:
    h = _h(obs)
    arch = (h.get("env") or {}).get("arch") or ""
    items = [x for x in ((h.get("args") or {}).get("items") or "").split(",") if x]
    out = [("gfx1151", arch.startswith("gfx1151"), arch),
           ("trees", bool((h.get("shas") or {}).get("base")) and bool(h.get("head_has_fix")), str(h.get("shas")))]
    if "smoke" in items:
        out.append(("smoke", bool(h.get("smoke_ok")), f"routed {h.get('smoke_routed')} fatal {h.get('fatal')}"))
    if "warm" in items:
        for res in ("480", "720"):
            n = {a: len(_warm(h, a, res)) for a in "ACD"}
            routed = {a: [w["counts"].get("routed", 0) for w in _warm(h, a, res)] for a in "ACD"}
            eng = all(x > 0 for x in routed["C"] + routed["D"]) and all(x == 0 for x in routed["A"])
            out.append((f"warm {res} runs", n["A"] >= 2 and n["C"] >= 2 and n["D"] >= 2, str(n)))
            out.append((f"warm {res} engagement", eng and bool(routed["C"]), str(routed)))
    if "acc" in items:
        rows = _acc_rows(h)
        out.append(("acc rows", len(rows) >= 5, f"{len(rows)} of {len(((h.get('acc') or {}).get('rows') or []))}"))
    if "cold" in items:
        ok = {t: _ok(r) and len(r.get("cold") or {}) == 2 for t, r in (h.get("cold") or {}).items()}
        out.append(("cold runs", any(ok[t] for t in ok if t[0] == "C") and any(ok[t] for t in ok if t[0] == "A"),
                    str(ok)))
    return out


def head_is_worse(base: dict, head: dict) -> tuple:
    h = head
    bad = []
    rows = _acc_rows(h)
    if rows:
        worst = max(r.get("ratio_direct") or 0 for r in rows)
        if worst > 1.05:
            bad.append(f"C per-conv error up to {worst:.3f}x MIOpen's (> 1.05x)")
    for arm in "CD":
        for res in ("480", "720"):
            for w in _warm(h, arm, res):
                fe = (w.get("counts") or {}).get("fallback_error", 0)
                if fe:
                    bad.append(f"{arm} warm {res}: fallback_error {fe}")
                if w.get("finite") is False:
                    bad.append(f"{arm} warm {res}: non-finite output")
    for t, p in (h.get("psnr720_vs_A1") or {}).items():
        if t != "A2" and (p is None or p < 40):
            bad.append(f"{t} 720 output PSNR vs A1 {p} dB (< 40)")
    for res in ("480", "720"):
        a = [w["s"] for w in _warm(h, "A", res)]
        c = [w["s"] for w in _warm(h, "C", res)]
        if a and c and _med(c) > max(a):
            bad.append(f"C warm {res} median {_med(c)} s above every A run (max {max(a)} s)")
    return (bool(bad), "; ".join(bad) or "C: per-conv error <= 1.05x MIOpen and warm decode not slower than A")


def table(obs: dict) -> str:
    h = _h(obs)
    L = [f"env {h.get('env')} shas {h.get('shas')} frames {(h.get('args') or {}).get('frames')} "
         f"vae {(h.get('vae_fetch') or {}).get('repo')} stored {(h.get('vae_fetch') or {}).get('stored_dtypes')}", ""]
    if h.get("smoke"):
        L.append("smoke: " + "; ".join(
            f"{a} rc {r.get('rc')} wrapped {r.get('wrapped')} routed {h.get('smoke_routed', {}).get(a)} "
            f"selfcheck {r.get('router_state')}" for a, r in h["smoke"].items()))
        L.append("")
    if h.get("warm"):
        L += ["| res | arm | n | decode s median | spread % | runs s | peak alloc GiB | routed / conv3d calls | routes |",
              "|---|---|---|---|---|---|---|---|---|"]
        for res in ("480", "720"):
            for arm in "ACD":
                ws = _warm(h, arm, res)
                if not ws:
                    continue
                s = [w["s"] for w in ws]
                L.append(f"| {res} | {arm} | {len(ws)} | {_med(s)} | {_spread(s)} | {s} | "
                         f"{_med([w.get('peak_alloc_gib') for w in ws])} | "
                         f"{ws[0]['counts'].get('routed')} / {ws[0].get('conv3d_calls')} {ws[0]['counts']} | "
                         f"{ws[0].get('routes')} |")
        L.append(f"psnr720 vs A1 {h.get('psnr720_vs_A1')}")
        L.append("")
    rows = _acc_rows(h)
    if rows:
        L += ["| w | x | miopen relL2 | C route | C/MIOpen | D route | D/MIOpen |", "|---|---|---|---|---|---|---|"]
        for r in rows:
            L.append(f"| {r['w']} | {r['x']} | {r['miopen']:.3e} | {r.get('route_direct')} | "
                     f"{r.get('ratio_direct') or 0:.3f} | {r.get('route_winograd')} | {r.get('ratio_winograd') or 0:.3f} |")
        L.append(f"C max ratio {max(r.get('ratio_direct') or 0 for r in rows):.3f}, "
                 f"D max ratio {max(r.get('ratio_winograd') or 0 for r in rows):.3f}")
        L.append("")
    for t, r in (h.get("cold") or {}).items():
        for res, c in (r.get("cold") or {}).items():
            L.append(f"cold {t} {res}: {c.get('s')} s variants_total {c.get('variants_total')} "
                     f"{c.get('variants_after')} cache {c.get('cache')}")
    return "\n".join(L)
