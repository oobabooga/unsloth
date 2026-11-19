#!/usr/bin/env python3
"""Criteria for probes/aotriton_gate_17985_probe.py (#17985, PR 17997).

Defect at the base: on a card Studio reads as gfx1103 (an arch AOTriton lists as experimental, never verified with
Studio's process-wide TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1), the fused-SDPA probe children still run with the
experimental switch on (#17985: they hung a Radeon 780M).
Fixed (the head state's "fix" arm):
  as_gfx1103  every child ran with the switch "0";
  real        gfx1151 unchanged from the base: every child ran with the switch on, flash + mem_efficient confirmed,
              both switches on, and the parent dispatches the same aten SDPA ops with finite, accurate output.
The head state's "tree" arm (PR 17997 as submitted) is reported, not judged.

Gates: real ROCm torch on gfx1151, each arm imported its own tree, every scenario ran, base gfx1151 has a fused
kernel (else a regression cannot be seen).
"""

from __future__ import annotations

TITLE = "#17985: AOTriton experimental switch in the ROCm fused-SDPA probe children"
MODE = "differential"
NEEDS = ["gpu", "rocm"]

_SC = ("real", "as_gfx1103")
_FUSED = ("flash", "mem_efficient")


def _arm(o: dict, arm: str) -> dict:
    return ((o or {}).get("arms") or {}).get(arm) or {}


def _sc(o: dict, arm: str, sc: str) -> dict:
    return (_arm(o, arm).get("scenarios") or {}).get(sc) or {}


def _arms(name: str) -> tuple[str, ...]:
    return ("tree",) if name == "base" else ("tree", "fix")


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out = []
    for name in ("base", "head"):
        o = obs.get(name) or {}
        for arm in _arms(name):
            root = str(_arm(o, arm).get("root") or "")
            for sc in _SC:
                r = _sc(o, arm, sc)
                out.append((f"{name}/{arm}: {sc} ran", bool(r.get("ok")),
                            str(r.get("error") or r.get("stderr_tail") or "ok")[-300:].replace("\n", " ")
                            .replace("|", "/")))
                imp = str(r.get("imported_from") or "")
                out.append((f"{name}/{arm}: {sc} imported its own tree", bool(root) and imp.startswith(root), imp))
            r = _sc(o, arm, "real")
            out.append((f"{name}/{arm}: real ROCm torch on gfx1151",
                        bool(r.get("hip")) and str(r.get("arch") or "").startswith("gfx1151"),
                        f"torch {r.get('torch')} hip {r.get('hip')} arch {r.get('arch')}"))
    base_real = _sc(obs.get("base") or {}, "tree", "real")
    out.append(("base gfx1151 confirms a fused kernel", any(k in base_real.get("kernels", []) for k in _FUSED),
                str(base_real.get("kernels"))))
    fix = _sc(obs.get("head") or {}, "fix", "real")
    out.append(("fix arm carries the fix", bool(fix.get("has_fix")), str(fix.get("has_fix"))))
    return out


def _gates_used(r: dict) -> str:
    calls = r.get("child_calls") or []
    return ", ".join(f"{c.get('backend')}={c.get('gate')}" for c in calls) or "no children"


def _ops(r: dict) -> str:
    d = r.get("dispatch") or {}
    return "; ".join(f"{k}: {'/'.join(x.split('_scaled_dot_product_')[-1] for x in v.get('ops', []))} "
                     f"(rel {v.get('rel_l1', float('nan')):.1e})" for k, v in d.items())


def table(obs: dict) -> str:
    cols = [("base", "tree"), ("head", "tree"), ("head", "fix")]
    rows = ["| observation | base | PR 17997 as submitted | fix |", "|---|---|---|---|"]

    def row(label, f):
        cells = []
        for n, a in cols:
            try:
                cells.append(str(f(obs.get(n) or {}, a)).replace("|", "/"))
            except Exception as e:  # noqa: BLE001
                cells.append(f"error {e!r}")
        rows.append(f"| {label} | " + " | ".join(cells) + " |")

    row("torch / hip / arch", lambda o, a: f"{_sc(o, a, 'real').get('torch')} / {_sc(o, a, 'real').get('hip')} / "
        f"{_sc(o, a, 'real').get('arch')}")
    row("card reads as integrated (APU)", lambda o, a: _sc(o, a, "real").get("integrated"))
    for sc in _SC:
        row(f"{sc}: child experimental switch", lambda o, a, sc=sc: _gates_used(_sc(o, a, sc)))
        row(f"{sc}: kernels confirmed", lambda o, a, sc=sc: _sc(o, a, sc).get("kernels"))
        row(f"{sc}: switches after the guard", lambda o, a, sc=sc: _sc(o, a, sc).get("flags"))
        row(f"{sc}: math-only", lambda o, a, sc=sc: _sc(o, a, sc).get("math_only"))
        row(f"{sc}: parent SDPA op", lambda o, a, sc=sc: _ops(_sc(o, a, sc)))
    row("real: auto flash (flash_attn installed)", lambda o, a: f"{_sc(o, a, 'real').get('auto_flash')} "
        f"({_sc(o, a, 'real').get('flash_attn_installed')})")
    truth = (obs.get("base") or {}).get("truth") or {}
    rows.append("")
    rows.append("Stock child probe on this torch, experimental switch 0 / 1: " +
                ", ".join(f"{k}={v}" for k, v in sorted(truth.items())))
    return "\n".join(rows)


def _children_gates(r: dict) -> set:
    return {c.get("gate") for c in r.get("child_calls") or []}


def base_shows_defect(base: dict):
    r = _sc(base, "tree", "as_gfx1103")
    gates = _children_gates(r)
    bad = bool(r.get("child_calls")) and gates != {"0"}
    return (bad, f"gfx1103 children ran with the experimental switch {sorted(map(str, gates))}")


def head_is_fixed(head: dict):
    if "fix" not in ((head or {}).get("arms") or {}):
        # The merge state is PR 17997 as submitted merged into main: the candidate fix rides only in the head state.
        return (True, "not judged: this state carries no fix arm")
    bad = []
    g = _sc(head, "fix", "as_gfx1103")
    if not g.get("child_calls") or _children_gates(g) != {"0"}:
        bad.append(f"gfx1103 children switch {sorted(map(str, _children_gates(g)))}")
    r = _sc(head, "fix", "real")
    if _children_gates(r) != {"1"}:
        bad.append(f"gfx1151 children switch {sorted(map(str, _children_gates(r)))}")
    if not all(k in (r.get("kernels") or []) for k in _FUSED):
        bad.append(f"gfx1151 lost a fused kernel: {r.get('kernels')}")
    flags = r.get("flags") or {}
    if not (flags.get("flash") and flags.get("mem_efficient")):
        bad.append(f"gfx1151 switches {flags}")
    for k, v in (r.get("dispatch") or {}).items():
        if not v.get("finite") or not v.get("ops") or "attention_math" in "".join(v.get("ops")) \
                or v.get("rel_l1", 1) > 2e-2:
            bad.append(f"gfx1151 {k} dispatch {v}")
    return (not bad, "; ".join(bad) or "fixed: gfx1103 children run without the experimental kernels, gfx1151 unchanged")
