#!/usr/bin/env python3
"""Criteria: does REAP expert pruning work on the GPU, matching the CPU's choice of experts?

Base has no REAP (the PR adds it), so the "defect" is its absence. Head must, for every model:
prune to half the experts on CPU fp32, GPU fp32 and GPU bf16, leave every expert tensor at the
new count, give finite logits, and keep >= 75% of the CPU's experts at the same dtype (GPU fp32 vs CPU
fp32, GPU bf16 vs CPU bf16: same weights, so only the kernels differ). Near ties flip between
kernels (a CUDA B200 run kept 47/48 in fp32), so exact equality is not the bar.

Pairs with probes/reap_probe.py.
"""

from __future__ import annotations

TITLE = "REAP expert pruning on gfx1151 (CPU fp32 vs GPU fp32 vs GPU bf16)"
MODE = "differential"
NEEDS = ["gpu", "nvidia", "multi_gpu", "mig", "xpu", "mlx", "windows_rocm_wddm"]
MODELS = ("qwen3_moe", "olmoe")
MIN_OVERLAP = 0.75


def _overlap(a: dict, b: dict) -> float:
    total = sum(len(v) for v in b.values())
    same = sum(len(set(a.get(k) or []) & set(v)) for k, v in b.items())
    return same / total if total else 0.0


def _model(state: dict, mode: str, name: str) -> dict:
    return ((state.get("modes") or {}).get(mode) or {}).get("models", {}).get(name) or {}


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out = []
    for name in ("base", "head"):
        modes = (obs.get(name) or {}).get("modes") or {}
        ran = [m for m, v in modes.items() if v.get("torch")]
        out.append((f"{name}: every mode started torch", len(ran) == 4, ", ".join(ran) or "none"))
    head_gpu = ((obs.get("head") or {}).get("modes") or {}).get("gpu_fp32") or {}
    out.append(("head GPU is ROCm", bool(head_gpu.get("hip")), f"hip={head_gpu.get('hip')} arch={head_gpu.get('arch')}"))
    return out


def table(obs: dict) -> str:
    rows = ["| state | mode | model | result | experts after | finite | layers equal to CPU fp32 | kept overlap |",
            "|---|---|---|---|---|---|---|---|"]
    for state in ("base", "head"):
        o = obs.get(state) or {}
        for mode, v in (o.get("modes") or {}).items():
            if v.get("error") and not v.get("models"):
                rows.append(f"| {state} | {mode} | - | {v['error'][:120]} | | | | |")
                continue
            for name in MODELS:
                r = _model(o, mode, name)
                ref = _model(o, "cpu_" + mode.split("_")[1], name).get("kept") or {}
                kept = r.get("kept") or {}
                same = sum(kept.get(k) == ref.get(k) for k in ref)
                res = r.get("error", "ok")[:120]
                rows.append(f"| {state} | {mode} | {name} | {res} | {r.get('expert_rows')} | "
                            f"{r.get('logits_finite')} | {same}/{len(ref)} | {_overlap(kept, ref):.1%} |")
    return "\n".join(rows)


def base_shows_defect(base: dict) -> bool:
    return all("reap.py" in (v.get("error") or "") for v in (base.get("modes") or {}).values())


def _ok(r: dict) -> bool:
    return (not r.get("error") and r.get("expert_rows") == [8] and r.get("num_experts_after") == 8
            and r.get("logits_finite") is True and len(r.get("kept") or {}) == 3)


def head_is_fixed(head: dict) -> bool:
    for name in MODELS:
        for dtype in ("fp32", "bf16"):
            cpu, gpu = _model(head, "cpu_" + dtype, name), _model(head, "gpu_" + dtype, name)
            if not (_ok(cpu) and _ok(gpu)):
                return False
            if _overlap(gpu.get("kept") or {}, cpu.get("kept") or {}) < MIN_OVERLAP:
                return False
    return True
