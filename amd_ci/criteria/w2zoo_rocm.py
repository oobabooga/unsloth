#!/usr/bin/env python3
"""Criteria (W2 zoo fixes for Qwen3.8-Flash-Next) on gfx1151 ROCm, base (zoo main) vs head, by test ID.

Regression mode. Head is WORSE when a test that passed at the base fails (or errors) at the head, or a test that only
the head has (the fixes' flip tests) fails there. Gates: the venv built with a GPU visible, each state imported
unsloth_zoo from its own checkout, and both suites ran and collected tests.
"""

from __future__ import annotations

TITLE = "unsloth-zoo W2 (Qwen3.8-Flash-Next) fixes on gfx1151 ROCm, base vs head"
MODE = "regression"
NEEDS = ["rocm", "gpu"]


def _py(o):
    return (o or {}).get("pytest") or {}


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out = []
    for name in ("base", "head"):
        o = obs.get(name) or {}
        env = ((o.get("venv") or {}).get("env") or {})
        out.append((f"{name} venv sees a ROCm GPU", bool(env.get("cuda_available")) and bool(env.get("hip")),
                    json.dumps(env)[:300] if env else str((o.get("venv") or {}).get("error"))[:300]))
        f = str(((o.get("zoo_import") or {}).get("file")) or "")
        ck = str(o.get("checkout") or "<none>")
        out.append((f"{name} imported its own unsloth_zoo", bool(f) and f.startswith(ck), f or "import failed"))
        p = _py(o)
        n = (p.get("n_passed") or 0) + (p.get("n_failed") or 0)
        out.append((f"{name} suite ran tests", p.get("rc") in (0, 1) and n > 0, f"rc={p.get('rc')} passed={p.get('n_passed')} "
                    f"failed={p.get('n_failed')} skipped={p.get('n_skipped')}"))
    return out


def head_is_worse(base: dict, head: dict) -> tuple[bool, str]:
    b, h = _py(base), _py(head)
    bf = set(b.get("failed") or []) | set(b.get("errors") or [])
    hf = set(h.get("failed") or []) | set(h.get("errors") or [])
    new = sorted(hf - bf)
    if new:
        return True, "failing at the head only: " + ", ".join(f"`{t.split('::')[-1]}`" for t in new)
    head_only = sorted(set(h.get("passed") or []) - set(b.get("passed") or []) - bf)
    return False, (f"no new failures; {len(head_only)} tests pass at the head only; failing at both: "
                   + (", ".join(f"`{t.split('::')[-1]}`" for t in sorted(hf & bf)) or "none"))


def table(obs: dict) -> str:
    rows = ["| state | passed | failed | skipped |", "|---|---|---|---|"]
    for name in ("base", "head"):
        p = _py(obs.get(name))
        rows.append(f"| {name} | {p.get('n_passed')} | {p.get('n_failed')} | {p.get('n_skipped')} |")
    return "\n".join(rows)


import json  # noqa: E402
