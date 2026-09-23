#!/usr/bin/env python3
"""Criteria for PR 11631: an explicit INT8 / FP8 transformer on an AMD GPU.

The defect: Studio refuses an explicit ``transformer_quant`` of int8 or fp8 on every ROCm GPU, so the
user's only choices are bf16 or a GGUF. The base must show that refusal for BOTH schemes, or the run is
VOID. The head is fixed when both load through Studio's own ``DiffusionBackend``, report the scheme as
applied, hold the transformer at no more than 60% of the bf16 load's bytes, and render every prompt
within LPIPS 0.35 of the bf16 render of the same seed (gfx1151 measured 0.08 to 0.15 for these arms;
0.35 still rejects a broken render, which scores 0.6 and up).
"""

from __future__ import annotations

TITLE = "PR 11631: explicit INT8 / FP8 weight-only on AMD (Qwen-Image-2.1 via Studio's DiffusionBackend)"
MODE = "differential"

NEEDS = [
    "rocm", "gpu",
    "amd_fp8_matrix_cores", "nvidia", "mig", "gpu_partitions",
    "windows", "windows_rocm_wddm", "windows_docker",
    "multi_gpu", "multi_gpu_amd", "discrete_gpu", "xpu", "mlx",
]

SCHEMES = ("int8", "fp8")
LPIPS_MAX = 0.35
BYTES_MAX = 0.60


def _arms(state: dict) -> dict:
    return state.get("arms") or {}


def gates(obs: dict):
    head, base = obs.get("head") or {}, obs.get("base") or {}
    ref = _arms(head).get("off") or {}
    return [
        ("ROCm torch on an AMD GPU", bool(head.get("hip")) and str(head.get("arch") or "").startswith("gfx"),
         f"{head.get('device')} {head.get('arch')} hip={head.get('hip')}"),
        ("bf16 reference loaded and rendered at head", bool(ref.get("loaded")) and bool(ref.get("images")),
         f"{len(ref.get('images') or [])} image(s)"),
        ("base probed both schemes", all(s in _arms(base) for s in SCHEMES), ", ".join(_arms(base))),
    ]


def base_shows_defect(base: dict):
    arms = _arms(base)
    refused = [s for s in SCHEMES if (arms.get(s) or {}).get("loaded") is False]
    return len(refused) == len(SCHEMES), f"refused at base: {refused}"


def head_is_fixed(head: dict):
    arms = _arms(head)
    ref = arms.get("off") or {}
    ref_gib = ref.get("transformer_gib") or 0
    problems = []
    for s in SCHEMES:
        rec = arms.get(s) or {}
        if not rec.get("loaded"):
            problems.append(f"{s} did not load: {(rec.get('error') or {}).get('message', '')[:120]}")
            continue
        resolved = rec.get("resolved") or {}
        if rec.get("transformer_quant") != s or resolved.get("status") != "applied":
            problems.append(f"{s} reported {rec.get('transformer_quant')!r} / {resolved.get('status')!r}")
        if ref_gib and (rec.get("transformer_gib") or ref_gib) > BYTES_MAX * ref_gib:
            problems.append(f"{s} transformer {rec.get('transformer_gib')} GiB vs bf16 {ref_gib}")
        imgs = rec.get("images") or []
        if len(imgs) != len(ref.get("images") or []) or any("error" in i for i in imgs):
            problems.append(f"{s} did not render every prompt")
        worst = max((i.get("lpips", 0) for i in imgs), default = 0)
        if worst > LPIPS_MAX:
            problems.append(f"{s} LPIPS {worst} > {LPIPS_MAX}")
    return not problems, "; ".join(problems) or "both schemes load, report applied, halve the weights and render"


def table(obs: dict) -> str:
    lines = [
        "| state | scheme | loaded | reported | transformer GiB | s/image | peak GiB | LPIPS | PSNR | note |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for state in ("base", "head"):
        for scheme, rec in _arms(obs.get(state) or {}).items():
            imgs = rec.get("images") or []
            ok = [i for i in imgs if "error" not in i]

            def mean(key):
                vals = [i[key] for i in ok if isinstance(i.get(key), (int, float))]
                return round(sum(vals) / len(vals), 3) if vals else ""

            note = ((rec.get("error") or {}).get("message") or (rec.get("resolved") or {}).get("reason") or "")
            note = str(note).replace("|", "/").replace("\n", " ")[:150]
            lines.append(
                f"| {state} | {scheme} | {'yes' if rec.get('loaded') else 'NO'} | "
                f"{rec.get('transformer_quant', '')} | {rec.get('transformer_gib', '')} | {mean('seconds')} | "
                f"{mean('peak_gib')} | {mean('lpips') if scheme != 'off' else 'ref'} | "
                f"{mean('psnr') if scheme != 'off' else 'ref'} | {note} |"
            )
    return "\n".join(lines)
