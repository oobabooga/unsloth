#!/usr/bin/env python3
"""Criteria: LFM3-Omni generate through Unsloth on ROCm (unslothai/unsloth#16096, issue #16054).

Defect: Unsloth's generate wrapper forces cache_implementation / compile_config, which LFM3's own generate
rejects ("does not implement ['cache_implementation', ...]"). Fixed when both generates succeed and the
3 LoRA losses match base's (the PR does not touch training).
"""

from __future__ import annotations

TITLE = "LFM3-Omni generate through Unsloth on ROCm"
MODE = "differential"
NEEDS = ["rocm", "gpu", "integrated_gpu", "linux"]


def _states(obs: dict):
    return {n: v for n, v in obs.items() if not n.startswith("_")}


def _step(v: dict, name: str) -> dict:
    return (v.get("steps") or {}).get(name) or {}


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    st = _states(obs)
    return [
        ("every state imported its own checkout", bool(st) and all(v.get("imports_own_checkout") for v in st.values()),
         ", ".join(f"{n}: {v.get('unsloth_file') or v.get('error', '')[:120]}" for n, v in st.items())),
        ("ROCm device used", bool(st) and all(v.get("hip") and v.get("device") for v in st.values()),
         ", ".join(f"{n}: hip={v.get('hip')} device={v.get('device')}" for n, v in st.items())),
        ("transformers 5.22.0 in every state", bool(st) and all(v.get("transformers") == "5.22.0" for v in st.values()),
         ", ".join(f"{n}={v.get('transformers')}" for n, v in st.items())),
        ("training ran in every state", bool(st) and all(_step(v, "lora_train_3_steps").get("ok") for v in st.values()),
         ", ".join(f"{n}={_step(v, 'lora_train_3_steps').get('detail')}" for n, v in st.items())),
    ]


def table(obs: dict) -> str:
    rows = ["| state | generate | LoRA losses | generate after LoRA |", "|---|---|---|---|"]
    for n, v in _states(obs).items():
        g, t, g2 = (_step(v, k) for k in ("generate_base_model", "lora_train_3_steps", "generate_lora_model"))
        rows.append(f"| {n} | {g.get('detail')} | {t.get('detail')} | {g2.get('detail')} |")
    return "\n".join(rows)


def base_shows_defect(base: dict) -> bool:
    d = str(_step(base, "generate_base_model").get("detail"))
    return not _step(base, "generate_base_model").get("ok") and "cache_implementation" in d


def head_is_fixed(head: dict) -> bool:
    return all(_step(head, k).get("ok") for k in ("generate_base_model", "lora_train_3_steps", "generate_lora_model"))
