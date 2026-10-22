#!/usr/bin/env python3
"""Criteria: on gfx1151, does the head's projector choice read images at least as well as the
base's, per llama.cpp bundle (ROCm, Vulkan)? Pairs with probes/mmproj_precision_probe.py
(unslothai/unsloth#14679).

Regression mode: a model whose images the base reads correctly must still be read correctly at the
head. Gates make it non-vacuous: every model served with its weights on a GPU device, the base
picked F16 everywhere, and the head picked what its rule says (BF16 for MiMo-VL on every backend,
BF16 for every model on ROCm, F16 for other models elsewhere).
"""

from __future__ import annotations

TITLE = "Vision projector precision on gfx1151: base (F16 first) vs head (per-device rule)"
MODE = "regression"
NEEDS = ["rocm", "gpu", "integrated_gpu", "linux", "windows"]


def _configs(state: dict) -> dict:
    return (state or {}).get("configs") or {}


def _all_ok(row: dict) -> bool:
    ok = row.get("ok") or []
    return len(ok) == 3 and all(ok)


def _expected_head(backend: str, model: str) -> str:
    return "mmproj-BF16.gguf" if model.startswith("mimo") or backend == "rocm" else "mmproj-F16.gguf"


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out = []
    for name in ("base", "head"):
        cfgs = _configs(obs.get(name))
        rows = [(a, m, r) for a, c in cfgs.items() for m, r in (c.get("models") or {}).items()]
        errs = [f"{a}: {c['selection_error']}" for a, c in cfgs.items() if c.get("selection_error")]
        out.append((f"{name}: selection ran and every model was served", bool(rows) and not errs
                    and all(r.get("ready") for _, _, r in rows),
                    "; ".join(errs) or ", ".join(f"{a}/{m}: ready={r.get('ready')} err={r.get('error')}"
                                                for a, m, r in rows)))
        out.append((f"{name}: every model's weights were on a GPU device",
                    bool(rows) and all(r.get("gpu_model_buffer") for _, _, r in rows),
                    ", ".join(f"{a}/{m}={r.get('gpu_model_buffer')}" for a, m, r in rows)))
    bad = []
    for a, c in _configs(obs.get("base")).items():
        bad += [f"base {a}/{m}={r.get('mmproj')}" for m, r in (c.get("models") or {}).items()
                if r.get("mmproj") != "mmproj-F16.gguf"]
    for a, c in _configs(obs.get("head")).items():
        bad += [f"head {a}/{m}={r.get('mmproj')}" for m, r in (c.get("models") or {}).items()
                if r.get("mmproj") != _expected_head(c.get("backend"), m)]
    out.append(("base picked F16 everywhere; head picked what its rule says", not bad and bool(_configs(obs.get("head"))),
                ", ".join(bad) or "all as expected"))
    return out


def table(obs: dict) -> str:
    rows = ["| bundle | model | base projector | base answers | head projector | head answers | head sample |",
            "|---|---|---|---|---|---|---|"]
    b, h = _configs(obs.get("base")), _configs(obs.get("head"))
    for a in sorted(set(b) | set(h)):
        bm, hm = (b.get(a) or {}).get("models") or {}, (h.get(a) or {}).get("models") or {}
        for m in sorted(set(bm) | set(hm)):
            rb, rh = bm.get(m, {}), hm.get(m, {})
            sample = ((rh.get("answers") or [""])[0] or "").replace("|", "/").replace("\n", " ")[-50:]
            rows.append(f"| {a} | {m} | {rb.get('mmproj')} | {sum(rb.get('ok') or [])}/3 | "
                        f"{rh.get('mmproj')} | {sum(rh.get('ok') or [])}/3 | {sample} |")
    return "\n".join(rows)


def head_is_worse(base: dict, head: dict) -> tuple[bool, str]:
    worse, fixed, both = [], [], []
    for a, c in _configs(base).items():
        hm = (_configs(head).get(a) or {}).get("models") or {}
        for m, r in (c.get("models") or {}).items():
            hb, hh = _all_ok(r), _all_ok(hm.get(m, {}))
            if hb and not hh:
                worse.append(f"{a}/{m}")
            elif not hb and hh:
                fixed.append(f"{a}/{m}")
            elif not hb and not hh:
                both.append(f"{a}/{m}")
    if worse:
        return True, "read correctly at the base, not at the head: " + ", ".join(worse)
    detail = "every model the base reads correctly is read correctly at the head"
    if fixed:
        detail += "; newly correct at the head: " + ", ".join(fixed)
    if both:
        detail += "; wrong at both: " + ", ".join(both)
    return False, detail
