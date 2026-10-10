#!/usr/bin/env python3
"""Criteria: does Studio install vLLM on an AMD GPU and generate with it (PR #12785)?

Base: Studio refuses vLLM on this AMD host (any non-empty support verdict, no install).
Head: the managed install succeeds on the ROCm lock, and greedy generation through
ManagedEngine answers "Paris" at the model's default precision.
"""

from __future__ import annotations

TITLE = "Managed vLLM on AMD gfx1151: install, load and generate"
MODE = "differential"
NEEDS = ["gpu", "rocm", "windows", "multi_gpu", "nvidia", "discrete_gpu", "mig", "xpu", "mlx"]


def _states(obs):
    return {n: v for n, v in obs.items() if not n.startswith("_")}


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    states = _states(obs)
    out = []
    imported = all("import_error" not in v for v in states.values())
    out.append(("every state imported its engine installer", imported,
                ", ".join(f"{n}: {v.get('import_error')}" for n, v in states.items() if "import_error" in v)))
    arch = {n: (v.get("host") or {}).get("arch") for n, v in states.items()}
    out.append(("the GPU is gfx1151 under ROCm torch",
                all(a and a.startswith("gfx1151") for a in arch.values()), str(arch)))
    return out


def _gen(state: dict, precision: str) -> dict:
    return next((g for g in state.get("generations") or [] if g.get("precision") == precision), {})


def table(obs: dict) -> str:
    rows = ["| state | gpu_platform | vLLM verdict | SGLang verdict | install | env on disk | auto answer | fp8 answer | int4 |",
            "|---|---|---|---|---|---|---|---|---|"]
    for n, v in _states(obs).items():
        env = v.get("env_bytes")
        cell = lambda g: (repr((g.get("text") or "")[:40]) + f" ({g.get('load_s')}s load)") if g.get("text") is not None else (g.get("error") or "-")[:120]
        rows.append(
            f"| {n} | {v.get('gpu_platform')} | {str(v.get('vllm_reason'))[:140]} | {str(v.get('sglang_reason'))[:80]} | "
            f"{v.get('install_ok', '-')} {v.get('install_s', '')}s | {f'{env / 2**30:.2f} GiB' if env else '-'} | "
            f"{cell(_gen(v, 'auto'))} | {cell(_gen(v, 'fp8'))} | {str(v.get('int4_refusal'))[:100]} |"
        )
    head = _states(obs).get("head") or {}
    if head.get("checkpoints"):
        rows += ["", "Quantized checkpoints on head (informational, not part of the verdict):", "",
                 "| checkpoint | load | answer or error |", "|---|---|---|"]
        for g in head["checkpoints"]:
            ans = repr((g.get("text") or "")[:40]) if g.get("text") is not None else (g.get("error") or "")[:200]
            rows.append(f"| {g.get('model')} | {g.get('load_s', '-')}s | {ans} |")
    if head.get("install_error"):
        rows += ["", "head install error:", "```", head["install_error"][-2000:], "```"]
    for g in head.get("generations") or []:
        if g.get("error"):
            rows += ["", f"head {g['precision']} error:", "```", (g.get("traceback") or g["error"])[-1500:],
                     "\n".join(g.get("engine_tail") or [])[-2500:], "```"]
    return "\n".join(rows)


def base_shows_defect(base: dict) -> bool:
    return bool(base.get("vllm_reason")) and not base.get("install_ok")


def head_is_fixed(head: dict) -> bool:
    auto = _gen(head, "auto")
    return (
        head.get("vllm_reason") is None
        and head.get("install_ok") is True
        and (head.get("installed_info") or {}).get("platform") == "rocm"
        and "paris" in (auto.get("text") or "").lower()
    )
