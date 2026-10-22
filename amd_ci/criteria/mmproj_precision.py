#!/usr/bin/env python3
"""Criteria: on ROCm gfx1151, does the head's projector choice read images at least as well as
the base's? Pairs with probes/mmproj_precision_probe.py (unslothai/unsloth#14679).

Regression mode: a model whose images the base reads correctly must still be read correctly at
the head (with BF16 instead of F16). Gates make it non-vacuous: every model served on a ROCm
device, the base picked F16 and the head picked BF16 for every model.
"""

from __future__ import annotations

TITLE = "Vision projector precision on ROCm: base (F16 first) vs head (BF16 first)"
MODE = "regression"
NEEDS = ["rocm", "gpu", "integrated_gpu", "linux"]


def _models(state: dict) -> dict:
    return state.get("models") or {}


def _all_ok(row: dict) -> bool:
    ok = row.get("ok") or []
    return len(ok) == 3 and all(ok)


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out = []
    for name in ("base", "head"):
        s = obs.get(name) or {}
        rows = _models(s)
        out.append((f"{name}: selection ran and every model was served", bool(rows) and not s.get("selection_error")
                    and all(r.get("ready") for r in rows.values()),
                    s.get("selection_error") or ", ".join(f"{m}: ready={r.get('ready')} err={r.get('error')}"
                                                           for m, r in rows.items())))
        out.append((f"{name}: every model's weights were on a ROCm device",
                    bool(rows) and all(r.get("rocm_model_buffer") for r in rows.values()),
                    ", ".join(f"{m}={r.get('rocm_model_buffer')}" for m, r in rows.items())))
    base_p = {m: r.get("mmproj") for m, r in _models(obs.get("base") or {}).items()}
    head_p = {m: r.get("mmproj") for m, r in _models(obs.get("head") or {}).items()}
    out.append(("base picked F16 and head picked BF16 for every model",
                bool(base_p) and all(v == "mmproj-F16.gguf" for v in base_p.values())
                and bool(head_p) and all(v == "mmproj-BF16.gguf" for v in head_p.values()),
                f"base={base_p} head={head_p}"))
    return out


def _med(xs):
    xs = sorted(xs or [])
    return xs[len(xs) // 2] if xs else None


def table(obs: dict) -> str:
    rows = ["| model | base projector | base answers | base encode ms | head projector | head answers | head encode ms |",
            "|---|---|---|---|---|---|---|"]
    b, h = _models(obs.get("base") or {}), _models(obs.get("head") or {})
    for m in sorted(set(b) | set(h)):
        rb, rh = b.get(m, {}), h.get(m, {})
        rows.append(f"| {m} | {rb.get('mmproj')} | {sum(rb.get('ok') or [])}/3 | {rb.get('encode_ms')} | "
                    f"{rh.get('mmproj')} | {sum(rh.get('ok') or [])}/3 | {rh.get('encode_ms')} |")
    return "\n".join(rows)


def head_is_worse(base: dict, head: dict) -> tuple[bool, str]:
    b, h = _models(base), _models(head)
    worse = [m for m in b if _all_ok(b[m]) and not _all_ok(h.get(m, {}))]
    if worse:
        return True, "read correctly at the base, not at the head: " + ", ".join(worse)
    fixed = [m for m in b if not _all_ok(b[m]) and _all_ok(h.get(m, {}))]
    both = [m for m in b if not _all_ok(b[m]) and not _all_ok(h.get(m, {}))]
    detail = "every model the base reads correctly is read correctly at the head"
    if fixed:
        detail += "; newly correct at the head: " + ", ".join(fixed)
    if both:
        detail += "; wrong at both: " + ", ".join(both)
    return False, detail
