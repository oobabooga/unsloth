#!/usr/bin/env python3
"""Criteria for unsloth-zoo PR 1363: the Windows ROCm torchao stub must read as absent.

The defect: zoo's stub answers every attribute, dunders included, with a sentinel class, so
``inspect.getmodule`` (reached from torch.library while transformers imports) calls ``.endswith``
on a class and raises "endswith() takes no arguments", and transformers 5 parses the sentinel
``__version__`` and raises InvalidVersion. The base must show one of those, with zoo's stub
actually installed, or the run is VOID. The head is fixed when transformers.modeling_utils
imports with the stub in place and the Unsloth arm does not fail with either signature.
"""

from __future__ import annotations

TITLE = "unsloth-zoo PR 1363: Windows ROCm torchao stub reads as absent"
MODE = "differential"

NEEDS = [
    "rocm", "gpu", "windows",
    "amd_fp8_matrix_cores", "nvidia", "mig", "gpu_partitions",
    "windows_docker", "multi_gpu", "multi_gpu_amd", "discrete_gpu", "xpu", "mlx",
]

SIGNATURES = ("endswith() takes no arguments", "Invalid version")


def _arm(state: dict, name: str) -> dict:
    return (state.get("arms") or {}).get(name) or {}


def _hit(arm: dict) -> bool:
    text = f"{arm.get('error', '')} {arm.get('tb', '')} {arm.get('stderr_tail', '')}"
    return any(s in text for s in SIGNATURES)


def gates(obs: dict):
    head, base = obs.get("head") or {}, obs.get("base") or {}
    return [
        ("ROCm torch on an AMD GPU", bool(head.get("hip")) and str(head.get("arch") or "").startswith("gfx"),
         f"{head.get('device')} {head.get('arch')} hip={head.get('hip')}"),
        ("zoo installed its stub at base", bool(_arm(base, "zoo_import").get("torchao_is_zoo_stub")),
         str(_arm(base, "zoo_import").get("torchao_type") or _arm(base, "zoo_import").get("error", ""))[:120]),
        ("zoo installed its stub at head", bool(_arm(head, "zoo_import").get("torchao_is_zoo_stub")),
         str(_arm(head, "zoo_import").get("torchao_type") or _arm(head, "zoo_import").get("error", ""))[:120]),
    ]


def base_shows_defect(base: dict):
    zi, ur = _arm(base, "zoo_import"), _arm(base, "unsloth_run")
    shown = _hit(zi) or _hit(ur)
    return shown, (zi.get("error") or ur.get("error") or "no failure")[:200]


def head_is_fixed(head: dict):
    zi, ur = _arm(head, "zoo_import"), _arm(head, "unsloth_run")
    if not zi.get("ok"):
        return False, f"zoo import arm failed: {zi.get('error', '')[:200]}"
    if _hit(ur):
        return False, f"unsloth arm still hits the stub: {ur.get('error', '')[:200]}"
    detail = "modeling_utils imports with the stub in place"
    detail += f"; unsloth arm {'ran' if ur.get('ok') else 'failed for another reason: ' + str(ur.get('error', ''))[:120]}"
    return True, detail


def table(obs: dict) -> str:
    lines = ["| state | stub | torchao.__version__ | modeling_utils | unsloth load+generate+5 LoRA steps | loss | text | error |",
             "|---|---|---|---|---|---|---|---|"]
    for state in ("base", "head", "merge"):
        s = obs.get(state) or {}
        if not s:
            continue
        zi, ur = _arm(s, "zoo_import"), _arm(s, "unsloth_run")
        err = (zi.get("error") or ur.get("error") or "").replace("|", "/").replace("\n", " ")[:150]
        lines.append(
            f"| {state} | {zi.get('torchao_is_zoo_stub')} | {zi.get('torchao_version', '')} | "
            f"{zi.get('modeling_utils', 'FAILED')} | {'ok' if ur.get('ok') else 'FAILED'} | {ur.get('loss', '')} | "
            f"{str(ur.get('text', '')).replace('|', '/')[:60]} | {err} |"
        )
    return "\n".join(lines)
