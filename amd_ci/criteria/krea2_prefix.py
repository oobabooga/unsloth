#!/usr/bin/env python3
"""Criteria for probes/krea2_prefix_probe.py (PR 19208), regression mode.

Head is worse when: its prefix path (mask-free block attention over the valid keys) moves the bf16 output past the
bf16 floor of the base's masked paths (vs one fp32 stock masked reference), its fp32 prefix output is not the fp32
masked output, Auto routes differently from the base and the routed output leaves the floor, or per-block compile
gains graph breaks / fails where the base compiled. Gates: ROCm torch, gfx1151 visible, each state imports its own
tree, the base text trim and the head prefix path provably ran.
"""
from __future__ import annotations

TITLE = "Krea-2 prefix K/V block attention on the real gfx1151 card (tiny random transformer), base versus head"
MODE = "regression"
NEEDS = ["rocm", "gpu", "windows"]
FP32_TOL = 1e-3
FLOOR_X = 1.5


def _states(obs):
    return {k: v for k, v in obs.items() if not k.startswith("_")}


def _floor(c, key):
    return max(c["masked16_vs_ref32"][key], c["trim16_vs_ref32"][key])


def gates(obs):
    out = []
    for name, o in _states(obs).items():
        err = o.get("error")
        out.append((f"{name}: probe child produced JSON", not err, str(err)[-300:] if err else f"rc={o.get('rc')}"))
        out.append((f"{name}: torch is a ROCm build", bool(o.get("hip")), f"{o.get('torch')} hip={o.get('hip')}"))
        out.append((f"{name}: gfx1151 visible", "gfx1151" in str(o.get("arch")), str(o.get("arch"))))
        f = str(o.get("trim_file"))
        out.append((f"{name}: checkout's own module", f"/{name}/" in f.replace("\\", "/") or name in f, f))
        cases = o.get("cases") or {}
        bad = [n for n, c in cases.items() if "error" in c]
        out.append((f"{name}: every case ran", bool(cases) and not bad,
                    "; ".join(f"{n}: {cases[n]['error'][-200:]}" for n in bad) or f"{len(cases)} cases"))
        ok = [c for c in cases.values() if "error" not in c]
        out.append((f"{name}: reference non-trivial", bool(ok) and all(c["finite_ref"] and c["ref32_absmean"] > 1e-3 for c in ok),
                    ", ".join(f"{c.get('ref32_absmean', 0):.3g}" for c in ok)))
        trimmed = [c for c in ok if c["trim16_calls"] and max(k for _, k in c["trim16_calls"]) < max(k for _, k in c["masked16_calls"])]
        out.append((f"{name}: text trim shortened the keys (base path engaged)", bool(trimmed), f"{len(trimmed)}/{len(ok)} cases"))
        if name in ("head", "merge"):
            out.append((f"{name}: carries diffusion_krea2_prefix_kv", bool(o.get("has_prefix")), str(o.get("has_prefix"))))
            eng = [c for c in ok if c.get("prefix16_stats", {}).get("prefix", 0) > 0 and c["prefix16_stats"]["masked"] == 0
                   and c["prefix16_calls"] and not any(m for m, _ in c["prefix16_calls"])]
            out.append((f"{name}: prefix path ran mask-free in every block", len(eng) == len(ok) and bool(ok),
                        f"{len(eng)}/{len(ok)} cases; stats {[c.get('prefix16_stats') for c in ok]}"))
        if name == "base":
            out.append(("base: has no prefix module (pre-PR tree)", not o.get("has_prefix"), str(o.get("has_prefix"))))
    return out


def table(obs):
    rows = ["| state | case | masked16 vs fp32 | trim16 vs fp32 | prefix16 vs fp32 | prefix16 vs masked16 | prefix32 vs fp32 | prefix calls |",
            "|---|---|---|---|---|---|---|---|"]
    for name, o in _states(obs).items():
        for n, c in (o.get("cases") or {}).items():
            if "error" in c:
                rows.append(f"| {name} | {n} | error | | | | | |")
                continue
            g = lambda k: f"{c[k]['max_abs']:.4g} / {c[k]['rel_l2']:.3g}" if k in c else "-"  # noqa: E731
            rows.append(f"| {name} | {n} | {g('masked16_vs_ref32')} | {g('trim16_vs_ref32')} | {g('prefix16_vs_ref32')} "
                        f"| {g('prefix16_vs_masked16')} | {g('prefix32_vs_ref32')} | {c.get('prefix16_stats', '-')} |")
    rows += ["", "| state | INT8 mode | routed modules | backends | route calls | vs fp32 max_abs | error |", "|---|---|---|---|---|---|---|"]
    for name, o in _states(obs).items():
        for m, r in (o.get("route") or {}).items():
            rows.append(f"| {name} | {m} | {r.get('routed')} | {r.get('backends')} | {r.get('route_calls')} "
                        f"| {(r.get('vs_ref32') or {}).get('max_abs')} | {str(r.get('error', ''))[-160:]} |")
    rows += ["", "| state | graph breaks | unique graphs | frames ok | error |", "|---|---|---|---|---|"]
    for name, o in _states(obs).items():
        c = o.get("compile") or {}
        rows.append(f"| {name} | {c.get('graph_breaks')} | {c.get('unique_graphs')} | {c.get('frames_ok')} | {str(c.get('error', ''))[-200:]} |")
    return "\n".join(rows)


def head_is_worse(base, head):
    bad, notes = [], []
    for n, c in (head.get("cases") or {}).items():
        if "error" in c or "prefix16_vs_ref32" not in c:
            bad.append(f"{n}: head case did not run the prefix path")
            continue
        for key in ("max_abs", "rel_l2"):
            fl = _floor(c, key)
            v = c["prefix16_vs_ref32"][key]
            if v > FLOOR_X * fl:
                bad.append(f"{n}: prefix16 {key} {v:.4g} > {FLOOR_X} x bf16 floor {fl:.4g}")
        if c["prefix32_vs_ref32"]["max_abs"] > FP32_TOL:
            bad.append(f"{n}: fp32 prefix vs fp32 masked max_abs {c['prefix32_vs_ref32']['max_abs']:.3g} > {FP32_TOL}")
    hc, bc = head.get("cases") or {}, base.get("cases") or {}
    for n, c in hc.items():
        b = bc.get(n) or {}
        if "masked16_vs_ref32" in c and "masked16_vs_ref32" in b and c["masked16_vs_ref32"] != b["masked16_vs_ref32"]:
            notes.append(f"{n}: stock masked path differs between arms ({b['masked16_vs_ref32']} vs {c['masked16_vs_ref32']})")
    for m in ("auto", "force"):
        hr, br = (head.get("route") or {}).get(m) or {}, (base.get("route") or {}).get(m) or {}
        if "error" in hr and "error" not in br:
            bad.append(f"INT8 {m}: head errored: {hr['error'][-200:]}")
        if hr.get("routed") != br.get("routed"):
            notes.append(f"INT8 {m}: routed {br.get('routed')} -> {hr.get('routed')}")
            c = hc.get("b2_v37_150") or {}
            if "masked16_vs_ref32" in c and (hr.get("vs_ref32") or {}).get("max_abs", 1e9) > FLOOR_X * _floor(c, "max_abs") * (4 if m == "force" else 1):
                bad.append(f"INT8 {m}: routed output max_abs {(hr.get('vs_ref32') or {}).get('max_abs')} past the floor")
    hcmp, bcmp = head.get("compile") or {}, base.get("compile") or {}
    if "error" in hcmp and "error" not in bcmp:
        bad.append(f"compile: head failed where base compiled: {hcmp['error'][-200:]}")
    elif "error" not in hcmp and "error" not in bcmp:
        if (hcmp.get("graph_breaks") or 0) > (bcmp.get("graph_breaks") or 0):
            bad.append(f"compile: graph breaks {bcmp.get('graph_breaks')} -> {hcmp.get('graph_breaks')}: {hcmp.get('graph_break_reasons')}")
        if (hcmp.get("unique_graphs") or 0) > (bcmp.get("unique_graphs") or 0):
            bad.append(f"compile: unique graphs {bcmp.get('unique_graphs')} -> {hcmp.get('unique_graphs')}")
    elif "error" in hcmp and "error" in bcmp:
        notes.append("compile failed at BOTH states (pre-existing, not judged)")
    if bad:
        return True, "; ".join(bad[:12])
    return False, ("prefix path within the bf16 floor of the masked paths in every case, fp32 prefix equals fp32 masked, "
                   "no new compile breaks; " + ("; ".join(notes) if notes else "Auto/forced routing unchanged"))
