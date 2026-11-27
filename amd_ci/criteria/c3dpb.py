#!/usr/bin/env python3
"""Criteria for probes/c3dpb_probe.py (Studio ROCm conv3d prebuild, Wan 2.2 VAE decode, gfx1151).

A = main (Triton conv3d compiled inline on a cold cache), B = main + patch (cold calls on MIOpen until a prebuild
child built the variant), M = main with the router off (MIOpen). head_is_worse flags B when its first cold decode is
slower than M beyond M's own spread (+5 %), when a decode in B's main process compiled a kernel after the child
finished, when B's warm decode is slower than A's, or when B's warm output differs from A's (full sha256)."""

from __future__ import annotations

import statistics as st

TITLE = "Studio ROCm conv3d prebuild: Wan 2.2 VAE cold / warm decode on gfx1151 (A main / B branch / M MIOpen)"
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


def _flow(h, arm):
    r = (h.get("cold") or {}).get(arm) or {}
    return r.get("flow") or {} if _ok(r) else {}


def gates(obs: dict) -> list:
    h = _h(obs)
    arch = (h.get("env") or {}).get("arch") or ""
    items = [x for x in ((h.get("args") or {}).get("items") or "").split(",") if x]
    out = [("gfx1151", arch.startswith("gfx1151"), arch),
           ("trees", bool((h.get("shas") or {}).get("base")) and bool(h.get("head_has_prebuild")), str(h.get("shas")))]
    if "smoke" in items:
        out.append(("smoke", bool(h.get("smoke_ok")), f"routed {h.get('smoke_routed')} fatal {h.get('fatal')}"))
    if "cold" in items:
        fa, fb = _flow(h, "A"), _flow(h, "B")
        full = all(k in f for f in (fa, fb) for k in ("first480", "second480", "first720"))
        out.append(("cold runs", full, f"A {sorted(fa)} B {sorted(fb)}"))
        eng = full and fa["first480"]["counts"].get("routed", 0) > 0 and \
            fb["second480"]["counts"].get("routed", 0) == fb["second480"].get("conv3d_calls")
        out.append(("cold engagement (A routes inline, B fully routed after the child)", bool(eng),
                    str({k: (fb.get(k) or {}).get("counts") for k in ("first480", "second480", "first720")})))
    if "warm" in items:
        n = {a: len(_warm(h, a, "480")) for a in "AMB"}
        out.append(("warm runs", n["A"] >= 2 and n["B"] >= 2 and n["M"] >= 2, str(n)))
        routed = {a: [w["counts"].get("routed", 0) for w in _warm(h, a, "480")] for a in "AMB"}
        out.append(("warm engagement", all(x > 0 for x in routed["A"] + routed["B"]) and
                    all(x == 0 for x in routed["M"]) and bool(routed["B"]), str(routed)))
    return out


def head_is_worse(base: dict, head: dict) -> tuple:
    h = head
    bad = []
    m = [w["s"] for w in _warm(h, "M", "480")]
    fb = _flow(h, "B")
    if m and fb.get("first480"):
        lim = max(m) * 1.05
        if fb["first480"]["s"] > lim:
            bad.append(f"B first cold 480 decode {fb['first480']['s']} s > MIOpen max {max(m)} s + 5%")
    for k in ("second480", "first720"):
        r = fb.get(k) or {}
        if r and r.get("hsaco_after") != r.get("hsaco_before"):
            bad.append(f"B {k}: {r.get('hsaco_after', 0) - r.get('hsaco_before', 0)} kernels compiled in the decode")
        if r and (r.get("counts") or {}).get("fallback_cold"):
            bad.append(f"B {k}: fallback_cold {(r.get('counts') or {}).get('fallback_cold')} after the child")
    for res in ("480", "720"):
        a = [w["s"] for w in _warm(h, "A", res)]
        b = [w["s"] for w in _warm(h, "B", res)]
        if a and b and _med(b) > max(a) * 1.03:
            bad.append(f"B warm {res} median {_med(b)} s above A max {max(a)} s + 3%")
        sa = {w.get("sha256") for w in _warm(h, "A", res)}
        sb = {w.get("sha256") for w in _warm(h, "B", res)}
        if sa and sb and (len(sa | sb) != 1):
            bad.append(f"warm {res} output sha256 differs (A {sorted(sa)} B {sorted(sb)})")
        for w in _warm(h, "B", res):
            c = w.get("counts") or {}
            if c.get("fallback_error") or c.get("fallback_cold"):
                bad.append(f"B warm {res}: {c}")
    return (bool(bad), "; ".join(bad) or "B: cold first decode within MIOpen, no compile in the decode, "
                                         "warm decode and output equal to A")


def table(obs: dict) -> str:
    h = _h(obs)
    L = [f"os {h.get('os')} env {h.get('env')} shas {h.get('shas')} frames {(h.get('args') or {}).get('frames')}", ""]
    for a, r in (h.get("smoke") or {}).items():
        L.append(f"smoke {a}: rc {r.get('rc')} wrapped {r.get('wrapped')} routed {h.get('smoke_routed', {}).get(a)} "
                 f"queued {r.get('prebuild_queued')} child {r.get('child')}")
    L.append("")
    L += ["| arm | load s | first 480 s | since load s | counts | child wait s / done since load s | "
          "second 480 s | counts | hsaco +/- | first 720 s | counts | hsaco +/- |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for arm in "AB":
        r = (h.get("cold") or {}).get(arm) or {}
        f = r.get("flow") or {}
        c = r.get("child") or {}

        def g(k, x):
            return (f.get(k) or {}).get(x)

        def d(k):
            return (g(k, "hsaco_after") or 0) - (g(k, "hsaco_before") or 0)
        L.append(f"| {arm} | {r.get('load_s')} | {g('first480', 's')} | {g('first480', 'since_load_s')} | "
                 f"{g('first480', 'counts')} | {c.get('wait_s')} / {c.get('since_load_s')} | {g('second480', 's')} | "
                 f"{g('second480', 'counts')} | {d('second480')} | {g('first720', 's')} | {g('first720', 'counts')} | "
                 f"{d('first720')} |")
    L.append("")
    L += ["| res | arm | n | decode s median | runs s | warm-up s | routed / calls | sha256 |", "|---|---|---|---|---|---|---|---|"]
    for res in ("480", "720"):
        for arm in "MAB":
            ws = _warm(h, arm, res)
            if ws:
                s = [w["s"] for w in ws]
                L.append(f"| {res} | {arm} | {len(ws)} | {_med(s)} | {s} | {[w.get('warmup_s') for w in ws]} | "
                         f"{ws[0]['counts'].get('routed')} / {ws[0].get('conv3d_calls')} | "
                         f"{sorted({(w.get('sha256') or '')[:16] for w in ws})} |")
    return "\n".join(L)
