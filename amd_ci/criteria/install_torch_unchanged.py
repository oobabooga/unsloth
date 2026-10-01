#!/usr/bin/env python3
"""Criteria: on ROCm, does the change leave the installed torch, the update path and training as
they were?

The torch 2.13 new-install route is CUDA 13 only, so on an AMD host the fresh install must pick the
same torch release at base and head, a second run must keep it, and a LoRA step must still train.

Pairs with probes/studio_install_torch_probe.py.
"""

from __future__ import annotations

TITLE = "Studio install on ROCm: torch, rerun and one LoRA step, base versus head"
MODE = "regression"
NEEDS: list[str] = ["rocm"]


def _torch(o: dict, phase: str) -> dict:
    return (o.get(phase) or {}).get("torch") or {}


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out = []
    for name in ("base", "head"):
        o = obs.get(name) or {}
        fresh = o.get("fresh") or {}
        out.append((f"{name} fresh install ran", fresh.get("rc") == 0,
                    o.get("error") or f"rc={fresh.get('rc')}, {fresh.get('seconds')}s"))
        info = _torch(o, "fresh")
        out.append((f"{name} installed a ROCm torch", bool(info.get("hip")),
                    f"torch {info.get('version')}, hip {info.get('hip')}, error {info.get('error')}"))
    return out


def table(obs: dict) -> str:
    rows = ["| state | fresh torch | rerun torch | rerun kept line | LoRA step loss |",
            "|---|---|---|---|---|"]
    for name in ("base", "head"):
        o = obs.get(name)
        if not o:
            continue
        sft = o.get("sft") or {}
        rows.append(f"| {name} | {_torch(o, 'fresh').get('version')} | {_torch(o, 'rerun').get('version')} "
                    f"| {(o.get('rerun') or {}).get('kept_line')} | {sft.get('loss', sft.get('error', '-'))} |")
    return "\n".join(rows)


def head_is_worse(base: dict, head: dict) -> tuple[bool, str]:
    problems = []
    b_ver, h_ver = _torch(base, "fresh").get("version"), _torch(head, "fresh").get("version")
    if b_ver != h_ver:
        problems.append(f"fresh install torch moved: base {b_ver}, head {h_ver}")
    if (head.get("rerun") or {}).get("rc") != 0:
        problems.append(f"head rerun failed rc={(head.get('rerun') or {}).get('rc')}")
    if _torch(head, "rerun").get("version") != h_ver:
        problems.append(f"head rerun changed torch {h_ver} -> {_torch(head, 'rerun').get('version')}")
    if (base.get("sft") or {}).get("finite") and not (head.get("sft") or {}).get("finite"):
        problems.append(f"LoRA step failed at head: {(head.get('sft') or {}).get('error')}")
    if problems:
        return True, "; ".join(problems)
    return False, (f"both states install torch {h_ver}, the rerun keeps it, "
                   f"and a LoRA step trains (head loss {(head.get('sft') or {}).get('loss')})")
