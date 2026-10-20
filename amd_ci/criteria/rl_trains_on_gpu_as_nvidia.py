#!/usr/bin/env python3
"""Criteria: as rl_trains_on_gpu, with the GPU presented as NVIDIA (lib/nvidia_identity.py).

Wiring only: NVIDIA branches run, the kernels are still the AMD GPU's.

The "defect" is the missing capability: the base checkout has no RL module. The head is fixed only
when GRPO and DPO each finished two optimizer steps on a ROCm device with finite losses and a non-zero gradient.

Pairs with probes/rl_train_probe.py.
"""

from __future__ import annotations

TITLE = "Studio RL training on the GPU presented as NVIDIA (Windows), base versus head"
MODE = "differential"
NEEDS: list[str] = ["gpu", "windows"]


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    head = obs.get("head") or {}
    return [
        ("base probe wrote its observation", bool(obs.get("base")) and "rl_available" in obs["base"],
         str((obs.get("base") or {}).get("rl_available"))),
        ("head saw an NVIDIA-presented GPU", bool(head.get("device")) and bool(head.get("cuda"))
         and not head.get("hip"),
         f"device={head.get('device')} cuda={head.get('cuda')} hip={head.get('hip')} {head.get('error', '')}"),
    ]


def table(obs: dict) -> str:
    rows = ["| state | RL module | objective | steps | losses | grad norms | error |",
            "|---|---|---|---|---|---|---|"]
    for name in ("base", "head", "merge"):
        o = obs.get(name)
        if not o:
            continue
        if not o.get("rl_available"):
            rows.append(f"| {name} | absent | - | - | - | - | - |")
            continue
        for objective in ("grpo", "dpo"):
            r = o.get(objective) or {}
            rows.append(
                f"| {name} | present | {objective} | {r.get('steps')} | {r.get('losses')} "
                f"| {r.get('grad_norms')} | {r.get('error', '')[:120]} |"
            )
    return "\n".join(rows)


def base_shows_defect(base: dict) -> bool:
    return not base.get("rl_available")


def head_is_fixed(head: dict) -> bool:
    return all(
        (head.get(o) or {}).get("steps", 0) >= 2
        and (head.get(o) or {}).get("finite")
        and (head.get(o) or {}).get("grad_positive")
        for o in ("grpo", "dpo")
    )
