#!/usr/bin/env python3
"""Criteria (unslothai/unsloth#17751): on ROCm with UNSLOTH_EXPERIMENTAL_HYBRID_PACKING=1 the hybrid varlen path
packs; base prints the ">2x faster" claim, head prints the opt-in note instead. Unset: both refuse packing alike."""

from __future__ import annotations

TITLE = "ROCm hybrid packing opt-in: packing message, base versus head"
MODE = "differential"
NEEDS = ["rocm", "gpu", "linux"]


def _states(obs: dict) -> dict:
    return {k: v for k, v in obs.items() if not k.startswith("_")}


def _run(o: dict, mode: str) -> dict:
    return (o.get("runs") or {}).get(mode) or {}


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out = []
    for name, o in _states(obs).items():
        r = _run(o, "opt_in")
        u = _run(o, "unset")
        out.append((f"{name}: torch is a ROCm build with a GPU", bool(r.get("hip")) and bool(r.get("gpu")),
                    f"hip={r.get('hip')} gpu={r.get('gpu')}"))
        out.append((f"{name}: unsloth imported from the checkout", "unsloth_file" in r, str(r.get("unsloth_file"))))
        out.append((f"{name}: opt-in trainer built", "stats" in r and not r.get("error"), str(r.get("error"))))
        out.append((f"{name}: opt-in varlen path patched", (r.get("stats") or {}).get("patched", 0) > 0,
                    str(r.get("stats"))))
        out.append((f"{name}: unset run refuses packing (control)", bool(u.get("packing_ignored")),
                    f"ignored={u.get('packing_ignored')} err={u.get('error')}"))
    return out


def table(obs: dict) -> str:
    rows = ["| state | mode | >2x claim | opt-in note | packing ignored | loss |", "|---|---|---|---|---|---|"]
    for name, o in _states(obs).items():
        for mode in ("opt_in", "unset"):
            r = _run(o, mode)
            rows.append(f"| {name} | {mode} | {r.get('old_claim')} | {r.get('opt_in_note')} | "
                        f"{r.get('packing_ignored')} | {r.get('train_loss', r.get('error', '-'))} |")
    return "\n".join(rows)


def base_shows_defect(base: dict) -> bool:
    r = _run(base, "opt_in")
    return bool(r.get("old_claim")) and not r.get("opt_in_note")


def head_is_fixed(head: dict) -> bool:
    r = _run(head, "opt_in")
    return bool(r.get("opt_in_note")) and not r.get("old_claim") and bool(_run(head, "unset").get("packing_ignored"))
