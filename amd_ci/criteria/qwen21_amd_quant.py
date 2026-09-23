#!/usr/bin/env python3
"""Criteria: judge the Qwen-Image-2.1 FP8 / INT8 measurement on gfx1151.

**A measurement run, not a differential.** The states straddle PR 11604 (the Qwen-Image-2.1 GGUF
single-file fix) only so the harness has a head checkout with the pinned diffusers build; nothing
in the quant selector moves between them. The question is "which FP8 / INT8 paths render
Qwen-Image-2.1 correctly on this AMD part, and what do they cost", so MODE is "regression" and the
verdict line says in words that no comparison of states took place.

Non-vacuity gates make a run that measured nothing go red instead of green: torch must import on a
ROCm build, the part must be an AMD GPU, the bf16 reference must have rendered, and at least one
quantised arm must have been attempted. A table with a missing reference is not a result.
"""

from __future__ import annotations

TITLE = "Qwen-Image-2.1 FP8 / INT8 on gfx1151: what runs, how fast, how close to bf16"
MODE = "regression"

NEEDS = [
    "rocm", "gpu",
    "amd_fp8_matrix_cores", "amd_fp4_matrix_cores",
    "nvidia", "mig", "gpu_partitions",
    "windows", "windows_rocm_wddm", "windows_docker",
    "multi_gpu", "multi_gpu_amd", "discrete_gpu",
    "xpu", "mlx",
]

ORDER = (
    "bf16", "studio_int8", "fp8_layerwise", "studio_fp8",
    "int8_weight", "fp8_weight", "te_fp8", "gguf_q4km",
)


def _val(rec):
    if isinstance(rec, dict) and rec.get("ok"):
        return rec.get("value")
    return None


def _head(obs: dict) -> dict:
    return obs.get("head") or {}


def gates(obs: dict):
    head = _head(obs)
    dev = _val(head.get("device")) or {}
    arms = head.get("arms") or {}
    ref = _val(arms.get("bf16")) or {}
    out = [
        ("torch imports at head", bool((head.get("torch_import") or {}).get("ok")),
         (head.get("packages") or {}).get("torch")),
        ("ROCm torch", bool(dev.get("hip")), f"hip={dev.get('hip')}"),
        ("AMD GPU visible", str(dev.get("arch") or "").startswith("gfx"),
         f"{dev.get('name')} {dev.get('arch')}"),
        ("bf16 reference rendered", bool(ref.get("images")),
         f"{len(ref.get('images') or [])} image(s)"),
        ("a quantised arm was attempted", any(a != "bf16" for a in arms),
         ", ".join(a for a in arms if a != "bf16") or "none"),
    ]
    return out


def _mean(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return round(sum(xs) / len(xs), 4) if xs else None


def table(obs: dict) -> str:
    head = _head(obs)
    arms = head.get("arms") or {}
    dev = _val(head.get("device")) or {}
    args = head.get("args") or {}
    ref = _val(arms.get("bf16")) or {}
    ref_step = _mean([i.get("median_step_s") for i in ref.get("images") or []])
    lines = [
        f"Device: {dev.get('name')} ({dev.get('arch')}), torch {dev.get('torch')}, "
        f"{dev.get('total_gib')} GiB; {args.get('size')}px, {args.get('steps')} steps, "
        f"cfg {args.get('cfg')}, {args.get('prompts')} prompt(s). LPIPS / PSNR against the bf16 arm, "
        "same prompt and seed.",
        "",
        "| arm | runs | transformer GiB | median s/step | vs bf16 | peak GiB | LPIPS | PSNR | error |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for arm in [a for a in ORDER if a in arms] + [a for a in arms if a not in ORDER]:
        rec = arms[arm]
        v = _val(rec) or {}
        imgs = v.get("images") or []
        step = _mean([i.get("median_step_s") for i in imgs])
        speed = f"{ref_step / step:.2f}x" if (step and ref_step) else ""
        e = rec.get("error") or {}
        emsg = rec.get("skipped") or (f"{e.get('type')}: {e.get('message', '')[:140]}" if e else "")
        emsg = emsg.replace("|", "/").replace("\n", " ")
        lines.append(
            f"| {arm} | {'yes' if rec.get('ok') else 'NO'} | {v.get('transformer_weight_gib', '')} | "
            f"{step if step is not None else ''} | {speed} | "
            f"{_mean([i.get('peak_gib') for i in imgs]) or ''} | "
            f"{_mean([i.get('lpips') for i in imgs]) if arm != 'bf16' else 'ref'} | "
            f"{_mean([i.get('psnr') for i in imgs]) if arm != 'bf16' else 'ref'} | {emsg} |"
        )
    sel = _val(head.get("selector")) or {}
    if sel:
        lines += ["", "Studio today on this box (head checkout):", ""]
        for k, rec in sel.items():
            val = _val(rec) if isinstance(rec, dict) and "ok" in rec else rec
            if isinstance(rec, dict) and rec.get("ok") is False:
                val = f"raised {rec['error']['type']}"
            lines.append(f"- `{k}` -> `{val}`")
    return "\n".join(lines)


def head_is_worse(base: dict, head: dict):
    return False, ("measurement run, not a differential: the states exist only to supply a "
                   "checkout, and the table above is the result")
