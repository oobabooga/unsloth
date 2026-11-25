#!/usr/bin/env python3
"""Criteria for probes/csa_grad_probe.py, differential mode.

Defect (base): the checkout's saved-attention checkpoint around compiled blocks (arm comp_tree) moves the LoRA grads
away from the plain checkpoint (comp_off) by more than 10x the compile noise floor (comp_off vs comp_nock, min 0.02)
in some cell. Fixed (head): every cell's comp_tree vs comp_off within max(2x floor, 0.01).
Gates: ROCm torch on gfx1151, the job finished, compile engaged in every cell (unique_graphs > 0, 0 graph breaks),
the saved-attention path engaged (saved_calls > 0), each state ran its own module (base = main copy, head = fix copy).
"""
from __future__ import annotations

TITLE = "Studio saved-attention checkpoint + regional compile: one-step LoRA grads on gfx1151, base versus head"
MODE = "differential"
NEEDS = ["rocm", "gpu", "windows", "nvidia"]


def _states(obs):
    return {k: v for k, v in obs.items() if not k.startswith("_") and isinstance(v, dict)}


def _cells(o):
    for e in (o.get("grad_results") or {}).get("runs", []):
        for s in e["seeds"]:
            yield e, s


def _floor(s):
    return s["comp_off__vs__comp_nock"]["rel"]


def gates(obs):
    out = []
    for name, o in _states(obs).items():
        r = o.get("grad_results") or {}
        out.append((f"{name}: job finished", o.get("job_rc") == 0 and bool(r.get("runs")),
                    f"rc={o.get('job_rc')} {o.get('job_s')}s; {' | '.join((o.get('log_tail') or [])[-3:])[-300:]}"))
        out.append((f"{name}: ROCm torch on gfx1151", bool(r.get("hip")) and "gfx1151" in str(r.get("arch")),
                    f"{r.get('torch')} hip={r.get('hip')} arch={r.get('arch')} dev={r.get('device')}"))
        dyn = [(e["family"], e["targets"], e.get("dynamo", {})) for e in r.get("runs", [])]
        ok = bool(dyn) and all(d.get("unique_graphs", 0) > 0 and d.get("graph_breaks", 1) == 0 for _, _, d in dyn)
        out.append((f"{name}: compile engaged in every cell", ok,
                    "; ".join(f"{f}/{t}: graphs={d.get('unique_graphs')} breaks={d.get('graph_breaks')}" for f, t, d in dyn)))
        sv = [s.get("comp_tree", {}).get("saved_calls") for _, s in _cells(o)]
        out.append((f"{name}: saved-attention path engaged", bool(sv) and all((x or 0) > 0 for x in sv), str(sv)))
        want = "gcb_tree_equals_main_copy" if name == "base" else "gcb_tree_equals_fix_copy"
        out.append((f"{name}: checkout module is the expected tree", bool(o.get(want)), f"{want}={o.get(want)}"))
    return out


def table(obs):
    rows = ["| state | family | tset | seed | tree vs off | sac(main) vs off | fix vs off | sac vs sac_aa | off vs nock (floor) |",
            "|---|---|---|---|---|---|---|---|---|"]
    for name, o in _states(obs).items():
        for e, s in _cells(o):
            rows.append(f"| {name} | {e['family']} | {e['targets']} | {s['seed']} | {s['comp_tree__vs__comp_off']['rel']:.3e} | "
                        f"{s['comp_sac__vs__comp_off']['rel']:.3e} | {s['comp_fix__vs__comp_off']['rel']:.3e} | "
                        f"{s['comp_sac__vs__comp_sac_aa']['rel']:.3e} | {_floor(s):.3e} |")
    x = (obs.get("head") or {}).get("xproc") or {}
    if x:
        nb = sum(1 for v in x.values() if v.get("bitwise"))
        worst = max(x.items(), key = lambda kv: kv[1].get("max_abs", 0))
        rows += ["", f"Cross-process (base run vs head run, shared arms): {nb}/{len(x)} bitwise equal; "
                     f"worst {worst[0]} max_abs={worst[1].get('max_abs'):.3e} rel={worst[1].get('rel'):.3e}"]
    return "\n".join(rows)


def base_shows_defect(base):
    bad = [(e["family"], e["targets"], s["seed"], s["comp_tree__vs__comp_off"]["rel"]) for e, s in _cells(base)
           if s["comp_tree__vs__comp_off"]["rel"] > max(10 * _floor(s), 0.02)]
    return bool(bad), f"cells over 10x floor: {bad}"


def head_is_fixed(head):
    bad = [(e["family"], e["targets"], s["seed"], s["comp_tree__vs__comp_off"]["rel"]) for e, s in _cells(head)
           if s["comp_tree__vs__comp_off"]["rel"] > max(2 * _floor(s), 0.01)]
    return bool(list(_cells(head))) and not bad, f"cells over 2x floor: {bad}"
