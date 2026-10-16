#!/usr/bin/env python3
"""Criteria for PR 13971 on gfx1151 (PR head vs its merge base with main).

Regression mode. The head is worse if, on this real ROCm torch, Studio stops asking for
causal-conv1d for Qwen3.5, the flash-attn setup decision changes, an SSM wheel resolves for
ROCm, or the LoRA losses / peak memory move.
"""

from __future__ import annotations

import math

TITLE = "PR 13971 on gfx1151: Qwen3.5 causal-conv1d decision, flash-attn sm80 gate on ROCm, LoRA losses"
MODE = "regression"
NEEDS: list[str] = ["rocm", "gpu", "linux", "nvidia"]

_LOSS_RTOL = 1e-3
_MEM_RTOL = 0.02


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
        out.append((f"{name} Studio decision probe ran", bool(_d(o, "decision")),
                    f"rc={(o.get('decision') or {}).get('rc')}"))
        f = _d(o, "flash")
        out.append((f"{name} flash-attn setup probe ran", bool(f) and not str(f.get("flash_attn", "")).startswith("ERR"),
                    f"rc={(o.get('flash') or {}).get('rc')} result={f.get('flash_attn')} tail={(o.get('flash') or {}).get('tail', '')[-300:] if not f else ''}"))
    h = _d(obs.get("head") or {}, "decision")
    out.append(("head exposes gpu_below_flash_attn_floor", bool(h.get("has_flash_floor")), f"has={h.get('has_flash_floor')}"))
    out.append(("head exposes resolve_ssm_kernels", isinstance(h.get("resolve_ssm_kernels"), dict), f"{h.get('resolve_ssm_kernels')}"))
    tb = _d(obs.get("base") or {}, "train").get("unsloth_file", "")
    th = _d(obs.get("head") or {}, "train").get("unsloth_file", "")
    out.append(("each state imported its own unsloth tree",
                "/states/base/" in tb and "/states/head/" in th, f"base={tb} head={th}"))
    return out


def table(obs: dict) -> str:
    rows = ["| state | commit | zoo_triton usable | wants causal_conv1d (name) | resolved (config) | "
            "resolve_ssm_kernels | flash-attn below floor | flash-attn setup | SSM wheel URLs | "
            "causal_conv1d importable | conv kernels (step 1) | losses | peak GB |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name in ("base", "head"):
        o = obs.get(name) or {}
        d, t, f = _d(o, "decision"), _d(o, "train"), _d(o, "flash")
        rows.append(
            f"| {name} | {str(o.get('commit'))[:9]} | {d.get('zoo_triton_causal_conv1d_usable', 'n/a (absent)')} | "
            f"{d.get('model_wants_causal_conv1d')} | {d.get('resolved_model_wants_causal_conv1d')} | "
            f"{d.get('resolve_ssm_kernels', 'n/a')} | {d.get('gpu_below_flash_attn_floor', 'n/a (absent)')} | "
            f"{f.get('flash_attn')} {f.get('url')} {f.get('notes')} | {d.get('resolve_wheel_url')} | "
            f"{t.get('causal_conv1d_importable')} | {', '.join(t.get('conv_kernels') or []) or 'none'} | "
            f"{t.get('losses')} | {t.get('peak_alloc_gb')} |")
    h = _d(obs.get("head") or {}, "decision")
    rows.append("")
    rows.append(f"torch dist version: `{h.get('torch_dist_version')}`; installed SSM dists: "
                f"`{h.get('installed_dists')}`; wheel env: `{h.get('wheel_env')}`")
    return "\n".join(rows)


def head_is_worse(base: dict, head: dict) -> tuple[bool, str]:
    problems = []
    dh, db = _d(head, "decision"), _d(base, "decision")
    if dh.get("model_wants_causal_conv1d") is not True:
        problems.append("head does not ask for causal-conv1d for Qwen3.5 by name on ROCm")
    if dh.get("resolved_model_wants_causal_conv1d") is not True:
        problems.append("head's config-based decision does not ask for causal-conv1d on ROCm")
    rk = dh.get("resolve_ssm_kernels")
    if not (isinstance(rk, dict) and rk.get("causal_conv1d") is True and rk.get("mamba_ssm") is False):
        problems.append(f"head resolve_ssm_kernels unexpected: {rk}")
    if dh.get("gpu_below_flash_attn_floor") is not False:
        problems.append(f"gpu_below_flash_attn_floor is {dh.get('gpu_below_flash_attn_floor')} on ROCm gfx1151")
    fb, fh = _d(base, "flash"), _d(head, "flash")
    if (fb.get("flash_attn"), fb.get("url")) != (fh.get("flash_attn"), fh.get("url")):
        problems.append(f"flash-attn setup decision changed on ROCm: base {fb.get('flash_attn')} {fb.get('url')} vs head {fh.get('flash_attn')} {fh.get('url')}")
    for n, u in (dh.get("resolve_wheel_url") or {}).items():
        if u:
            problems.append(f"head resolves an SSM wheel ({n}) on ROCm: {u}")
    tb, th = _d(base, "train"), _d(head, "train")
    lb, lh = tb.get("losses") or [], th.get("losses") or []
    diffs = [abs(a - b) / max(abs(a), 1e-9) for a, b in zip(lb, lh)]
    if len(lb) != len(lh) or any(x > _LOSS_RTOL for x in diffs):
        problems.append(f"losses differ (rel diffs {diffs})")
    pb, ph = tb.get("peak_alloc_gb"), th.get("peak_alloc_gb")
    if pb and ph and (ph - pb) / pb > _MEM_RTOL:
        problems.append(f"peak memory rose {pb} -> {ph} GB")
    if problems:
        return True, "; ".join(problems)
    return False, (f"ROCm still asks for causal-conv1d for Qwen3.5 (no ROCm wheel resolves), the flash-attn "
                   f"setup decision is unchanged ({fh.get('flash_attn')}), losses match (max rel diff "
                   f"{max(diffs) if diffs else 'n/a'}) and peak memory {pb} -> {ph} GB")
