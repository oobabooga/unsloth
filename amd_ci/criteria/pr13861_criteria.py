#!/usr/bin/env python3
"""Criteria for PR 13861 on gfx1151 (merge commit vs its parent).

Regression mode. The head is worse if, on this real ROCm torch, Studio stops asking for
causal-conv1d for Qwen3.5, setup starts installing an SSM kernel, or the LoRA losses move.
"""

from __future__ import annotations

import math

TITLE = "PR 13861 on gfx1151: Qwen3.5 causal-conv1d decision, setup SSM step, LoRA losses"
MODE = "regression"
NEEDS: list[str] = ["rocm", "gpu", "linux", "nvidia"]

_LOSS_RTOL = 1e-3


def _d(o: dict, part: str) -> dict:
    return ((o or {}).get(part) or {}).get("data") or {}


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out = []
    for name in ("base", "head"):
        o = obs.get(name) or {}
        t = _d(o, "train")
        losses = t.get("losses") or []
        ok = len(losses) == 4 and all(math.isfinite(x) for x in losses)
        out.append((f"{name} Qwen3.5 LoRA ran 4 finite steps", ok,
                    f"rc={(o.get('train') or {}).get('rc')} losses={losses} "
                    f"tail={(o.get('train') or {}).get('tail', '')[-300:] if not ok else ''}"))
        out.append((f"{name} ran on ROCm gfx1151", bool(t.get("hip")) and "gfx1151" in str(t.get("arch")),
                    f"torch={t.get('torch')} hip={t.get('hip')} arch={t.get('arch')} device={t.get('device')}"))
        out.append((f"{name} model is qwen3_5", t.get("model_type") in ("qwen3_5", "qwen3_5_text"),
                    f"model_type={t.get('model_type')}"))
        d = _d(o, "decision")
        out.append((f"{name} Studio decision probe ran", bool(d),
                    f"rc={(o.get('decision') or {}).get('rc')}"))
    h = _d(obs.get("head") or {}, "decision")
    out.append(("head exposes zoo_triton_causal_conv1d_usable", bool(h.get("has_zoo_predicate")),
                f"has={h.get('has_zoo_predicate')}"))
    tb = _d(obs.get("base") or {}, "train").get("unsloth_file", "")
    th = _d(obs.get("head") or {}, "train").get("unsloth_file", "")
    out.append(("each state imported its own unsloth tree",
                "/states/base/" in tb and "/states/head/" in th, f"base={tb} head={th}"))
    return out


def table(obs: dict) -> str:
    rows = ["| state | commit | setup _ensure_ssm_kernels() calls | zoo_triton usable | "
            "wants causal_conv1d (name) | resolved (config) | SSM wheel URLs | causal_conv1d importable | "
            "conv kernels (step 1) | losses | peak GB |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name in ("base", "head"):
        o = obs.get(name) or {}
        d, t = _d(o, "decision"), _d(o, "train")
        rows.append(
            f"| {name} | {str(o.get('commit'))[:9]} | {o.get('setup_calls_ensure_ssm_kernels')} | "
            f"{d.get('zoo_triton_causal_conv1d_usable', 'n/a (absent)')} | {d.get('model_wants_causal_conv1d')} | "
            f"{d.get('resolved_model_wants_causal_conv1d')} | {d.get('resolve_wheel_url')} | "
            f"{t.get('causal_conv1d_importable')} | {', '.join(t.get('conv_kernels') or []) or 'none'} | "
            f"{t.get('losses')} | {t.get('peak_alloc_gb')} |")
    h = _d(obs.get("head") or {}, "decision")
    rows.append("")
    rows.append(f"torch dist version: `{h.get('torch_dist_version')}`; installed SSM dists in the venv: "
                f"`{h.get('installed_dists')}`; wheel env: `{h.get('wheel_env')}`")
    return "\n".join(rows)


def head_is_worse(base: dict, head: dict) -> tuple[bool, str]:
    problems = []
    dh = _d(head, "decision")
    if dh.get("zoo_triton_causal_conv1d_usable") is not False:
        problems.append(f"zoo_triton_causal_conv1d_usable is {dh.get('zoo_triton_causal_conv1d_usable')} on ROCm")
    if dh.get("model_wants_causal_conv1d") is not True:
        problems.append("head no longer asks for causal-conv1d for Qwen3.5 by name on ROCm")
    db = _d(base, "decision")
    if db.get("resolved_model_wants_causal_conv1d") is True and dh.get("resolved_model_wants_causal_conv1d") is not True:
        problems.append("head's config-based decision dropped causal-conv1d on ROCm")
    if (head.get("setup_calls_ensure_ssm_kernels") or 0) > (base.get("setup_calls_ensure_ssm_kernels") or 0):
        problems.append("head setup calls _ensure_ssm_kernels more often than base")
    for n, u in (dh.get("resolve_wheel_url") or {}).items():
        if u:
            problems.append(f"head resolves an SSM wheel ({n}) on ROCm: {u}")
    lb, lh = _d(base, "train").get("losses") or [], _d(head, "train").get("losses") or []
    diffs = [abs(a - b) / max(abs(a), 1e-9) for a, b in zip(lb, lh)]
    if len(lb) != len(lh) or any(x > _LOSS_RTOL for x in diffs):
        problems.append(f"losses differ (rel diffs {diffs})")
    if problems:
        return True, "; ".join(problems)
    return False, (f"ROCm keeps the causal-conv1d request for Qwen3.5, setup's SSM step is gone "
                   f"(base calls {base.get('setup_calls_ensure_ssm_kernels')}, head "
                   f"{head.get('setup_calls_ensure_ssm_kernels')}), no SSM wheel resolves for ROCm torch, "
                   f"and losses match (max rel diff {max(diffs) if diffs else 'n/a'})")
