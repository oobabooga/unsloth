#!/usr/bin/env python3
"""Criteria (unsloth-zoo MoE PRs 2477 / 2478 / 2479): tiny Qwen3-MoE and gpt-oss LoRA / QLoRA training with
Unsloth gradient checkpointing on gfx1151 ROCm, base vs head.

Regression mode. Head is WORSE than base when any of:
  * a case that finished at the base crashes, or gives non-finite loss / grads, at the head
  * a case's per-step losses differ from the base's by more than 2e-2 relative (the PRs change kernels, glue and
    checkpoint routing replay, never the objective), or its step-0 LoRA grad norm by more than 5e-2 relative
  * a test that ran at the base fails at the head, or a test only the head has fails there

Gates (non-vacuity): each state imported unsloth_zoo from its own checkout and ran on a GPU; every base case
finished with finite losses. gpt-oss uses hf-internal-testing/tiny-random-GptOssForCausalLM (bf16): the trl tiny
checkpoint is MXFP4, which this ROCm stack does not dequantize (it fails at base and head alike).
"""

from __future__ import annotations

TITLE = "unsloth-zoo MoE PR: tiny MoE LoRA / QLoRA training on gfx1151 ROCm, base vs head"
MODE = "regression"
NEEDS = ["rocm", "gpu"]
_LOSS_TOL, _GRAD_TOL = 2e-2, 5e-2


def _cases(o: dict) -> dict:
    return (o or {}).get("train") or {}


def _done(a: dict) -> bool:
    return a.get("stage") == "done" and a.get("ok") is True and bool(a.get("finite"))


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out = []
    for name in ("base", "head"):
        o = obs.get(name) or {}
        ck = str(o.get("checkout") or "<none>")
        cases = _cases(o)
        out.append((f"{name} ran training cases", bool(cases), f"{len(cases)} cases"))
        for case, a in cases.items():
            f = str(a.get("unsloth_zoo_file") or "")
            out.append((f"{name}/{case} imported the state's unsloth_zoo", bool(f) and f.startswith(ck),
                        f or f"not imported (stage={a.get('stage')}, rc={a.get('rc')}): {str(a.get('error'))[:200]}"))
            out.append((f"{name}/{case} ran on a GPU", bool(a.get("device")), str(a.get("device"))))
    for case, a in _cases(obs.get("base")).items():
        out.append((f"base/{case} finished with finite losses", _done(a),
                    a.get("error") or f"stage={a.get('stage')} losses={a.get('losses')}"))
    return out


def _rel(a: float, b: float) -> float:
    return abs(a - b) / (abs(b) + 1e-12)


def table(obs: dict) -> str:
    rows = ["| state | case | result | losses | step-0 grad norm | counters |", "|---|---|---|---|---|---|"]
    vers = None
    for name in ("base", "head", "merge"):
        for case, a in _cases(obs.get(name)).items():
            vers = vers or a.get("versions")
            res = "ok" if _done(a) else f"FAIL: {str(a.get('error') or a.get('stage'))[:120]}"
            ls = "[" + ", ".join(f"{x:.5f}" for x in a.get("losses") or []) + "]"
            rows.append(f"| {name} | {case} | {res} | {ls} | {a.get('grad_norm_step0')} | {a.get('counters')} |")
    for name in ("base", "head", "merge"):
        p = (obs.get(name) or {}).get("pytest")
        if p and p.get("selected"):
            rows.append(f"\n{name} pytest: passed {p.get('n_passed')}, failed {p.get('n_failed')}, skipped {p.get('n_skipped')}")
    if vers:
        rows += ["", "Versions: " + ", ".join(f"{k}={v}" for k, v in vers.items())]
    return "\n".join(rows)


def head_is_worse(base: dict, head: dict) -> tuple[bool, str]:
    problems, notes = [], []
    hc, bc = _cases(head), _cases(base)
    for case, b in bc.items():
        h = hc.get(case) or {}
        if not _done(b):
            continue
        if not _done(h):
            problems.append(f"{case} finished at the base but not at the head: {h.get('error') or h.get('stage')}")
            continue
        worst = max(_rel(x, y) for x, y in zip(h["losses"], b["losses"]))
        g = _rel(h.get("grad_norm_step0") or 0.0, b.get("grad_norm_step0") or 0.0)
        if worst > _LOSS_TOL or g > _GRAD_TOL:
            problems.append(f"{case}: loss max rel {worst:.2e} (tol {_LOSS_TOL}), step-0 grad norm rel {g:.2e} (tol {_GRAD_TOL})")
        else:
            notes.append(f"{case}: loss max rel {worst:.2e}, grad norm rel {g:.2e}")
    bp, hp = base.get("pytest") or {}, head.get("pytest") or {}
    bf = set(bp.get("failed", [])) | set(bp.get("errors", []))
    hf = set(hp.get("failed", [])) | set(hp.get("errors", []))
    new = sorted(hf - bf)
    if new:
        problems.append("failing at the head (not at the base): " + ", ".join(f"`{n.split('::')[-1]}`" for n in new))
    if bf & hf:
        notes.append("failing at BOTH states (pre-existing): " + ", ".join(f"`{n.split('::')[-1]}`" for n in sorted(bf & hf)))
    if hp.get("selected"):
        notes.append(f"head pytest: {hp.get('n_passed')} passed, {hp.get('n_failed')} failed, {hp.get('n_skipped')} skipped")
    if problems:
        return True, "; ".join(problems) + (". Also: " + "; ".join(notes) if notes else "")
    return False, "; ".join(notes)
