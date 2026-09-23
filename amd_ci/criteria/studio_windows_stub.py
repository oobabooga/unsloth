#!/usr/bin/env python3
"""Criteria for PR 11640: a bf16 Qwen-Image-2.1 load through Studio on Windows ROCm.

The defect: Studio's torchao stub has no ``__version__``, transformers 5 parses "N/A" for it, and
``transformers.modeling_utils`` fails to import, so every model load dies with InvalidVersion. The
base must show exactly that, or the run is VOID. The head is fixed when the same load succeeds and
renders a non-degenerate image (mean luma between 5 and 250, so black or blown-out frames fail).
"""

from __future__ import annotations

TITLE = "PR 11640: Studio loads a diffusers model on Windows ROCm (torchao stub version)"
MODE = "differential"

NEEDS = [
    "rocm", "gpu", "windows",
    "amd_fp8_matrix_cores", "nvidia", "mig", "gpu_partitions",
    "windows_docker", "multi_gpu", "multi_gpu_amd", "discrete_gpu", "xpu", "mlx",
]


def _ref(state: dict) -> dict:
    return (state.get("arms") or {}).get("off") or {}


def gates(obs: dict):
    head, base = obs.get("head") or {}, obs.get("base") or {}
    return [
        ("ROCm torch on an AMD GPU", bool(head.get("hip")) and str(head.get("arch") or "").startswith("gfx"),
         f"{head.get('device')} {head.get('arch')} hip={head.get('hip')}"),
        ("base attempted the bf16 load", "off" in (base.get("arms") or {}), ", ".join(base.get("arms") or {})),
        ("head attempted the bf16 load", "off" in (head.get("arms") or {}), ", ".join(head.get("arms") or {})),
    ]


def base_shows_defect(base: dict):
    ref = _ref(base)
    message = (ref.get("error") or {}).get("message", "")
    return (ref.get("loaded") is False and "Invalid version" in message), message[:200] or "loaded"


def head_is_fixed(head: dict):
    ref = _ref(head)
    if not ref.get("loaded"):
        return False, f"did not load: {(ref.get('error') or {}).get('message', '')[:200]}"
    imgs = [i for i in ref.get("images") or [] if "error" not in i]
    if not imgs:
        return False, "loaded but rendered nothing"
    luma = [i.get("mean_luma") for i in imgs]
    ok = all(isinstance(v, (int, float)) and 5 < v < 250 for v in luma)
    return ok, f"loaded and rendered {len(imgs)} image(s), mean luma {luma}"


def table(obs: dict) -> str:
    lines = ["| state | loaded | s/image | peak GiB | mean luma | error |", "|---|---|---|---|---|---|"]
    for state in ("base", "head", "merge"):
        ref = _ref(obs.get(state) or {})
        if not ref:
            continue
        imgs = [i for i in ref.get("images") or [] if "error" not in i]
        err = ((ref.get("error") or {}).get("message") or "").replace("|", "/").replace("\n", " ")[-150:]
        lines.append(
            f"| {state} | {'yes' if ref.get('loaded') else 'NO'} | "
            f"{imgs[0].get('seconds') if imgs else ''} | {imgs[0].get('peak_gib') if imgs else ''} | "
            f"{imgs[0].get('mean_luma') if imgs else ''} | {err} |"
        )
    return "\n".join(lines)
