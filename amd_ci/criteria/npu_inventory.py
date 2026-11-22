"""Criteria: NPU stack inventory. Nothing regresses; the table is the deliverable."""
from __future__ import annotations
TITLE = "XDNA2 NPU software inventory"
MODE = "regression"
NEEDS: list[str] = []

def gates(obs):
    st = [k for k in obs if not k.startswith("_")]
    ok = all((obs[k].get("cmds") or {}) for k in st)
    return [("every state ran the inventory", ok and bool(st), ",".join(st))]

def head_is_worse(base, head):
    return False, "inventory only"

def table(obs):
    o = obs.get("head") or obs.get("base") or {}
    rows = [f"accel node: {o.get('has_accel_node')}, flm on PATH: {o.get('has_flm')}", ""]
    for k, v in (o.get("cmds") or {}).items():
        rows.append(f"### {k}\n```\n{v}\n```")
    return "\n".join(rows)
