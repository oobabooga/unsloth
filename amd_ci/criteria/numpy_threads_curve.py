#!/usr/bin/env python3
"""Criteria: measurement only. numpy's OpenBLAS memory and speed per OPENBLAS_NUM_THREADS on Windows (#12374 default)."""

from __future__ import annotations

import statistics

TITLE = "numpy OpenBLAS threads vs committed memory and speed on Windows"
MODE = "regression"
NEEDS = ["windows"]
KEYS = ["mm2048_f64_ms", "mm2048_f32_ms", "mm1024_f64_ms", "proj_20000x768_ms", "svd800_ms", "solve800_ms"]


def _states(obs: dict) -> dict:
    return {k: v for k, v in obs.items() if not k.startswith("_")}


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    st = _states(obs)
    errs = {k: v.get("probe_error") for k, v in st.items() if v.get("probe_error")}
    rows_err = {k: [r.get("error") for r in v.get("rows", []) if r.get("error")][:2] for k, v in st.items()}
    return [("every state ran the curve", not errs, str(errs)[:800]),
            ("every cell ran", not any(rows_err.values()), str(rows_err)[:800])]


def table(obs: dict) -> str:
    out = []
    for k, v in _states(obs).items():
        by: dict = {}
        for r in v.get("rows", []):
            if not r.get("error"):
                by.setdefault(r.get("env"), []).append(r)
        out += [f"### {k}: numpy {v.get('numpy')}, logical {v.get('logical')}, physical {v.get('physical')}", "",
                "| OPENBLAS_NUM_THREADS | threads after import | committed MB after import | committed MB after work | "
                + " | ".join(KEYS) + " |", "|---" * (4 + len(KEYS)) + "|"]
        for env, rs in by.items():
            def m(f):
                return statistics.median([f(r) for r in rs])
            out.append(f"| {env or 'unset'} | {m(lambda r: r['import']['threads'])} | {m(lambda r: r['import']['private_mb'])} | "
                       f"{m(lambda r: r['after']['private_mb'])} | " + " | ".join(str(m(lambda r, k=k: r[k])) for k in KEYS) + " |")
        out.append("")
    return "\n".join(out)


def head_is_worse(base: dict, head: dict) -> tuple[bool, str]:
    return False, "measurement only"
