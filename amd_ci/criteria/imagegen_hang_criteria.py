#!/usr/bin/env python3
"""Criteria (issue #9404): a default Z-Image-Turbo GGUF Q4_K_M render on gfx1151 Linux.

Defect: the first render does not produce an image within 300 s of clicking Generate
(the reporter watched step 0 for about 4.5 minutes with nothing). Fixed: the first
render finishes within 300 s and returns a real PNG.
"""

from __future__ import annotations

TITLE = "Issue 9404: Z-Image-Turbo GGUF Q4_K_M default render, v0.1.800-beta vs head"
MODE = "differential"
NEEDS = ["rocm", "gpu"]
LIMIT_S = 300


def _first(o: dict) -> dict:
    r = o.get("renders") or []
    return r[0] if r else {}


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out = []
    for name in [n for n in ("base", "head", "main") if n in obs]:
        o = obs.get(name) or {}
        out.append((f"{name} installed and loaded the model", o.get("load_status") == 200,
                    o.get("error") or f"install_rc={o.get('install_rc')} load={o.get('load_status')} "
                    f"torch={o.get('torch')}"))
        r = _first(o)
        out.append((f"{name} render was started and observed", bool(r.get("samples")),
                    f"{len(r.get('samples') or [])} progress samples"))
        t = o.get("torch") or {}
        arch = t.get("arch") if isinstance(t, dict) else None
        out.append((f"{name} ran on gfx1151 ROCm", bool(arch and "gfx1151" in arch and t.get("hip")),
                    f"arch={arch} hip={t.get('hip') if isinstance(t, dict) else None}"))
    return out


def _fmt(o: dict) -> str:
    r = _first(o)
    w = (o.get("renders") or [{}, {}])[1] if len(o.get("renders") or []) > 1 else {}
    return (f"| {o.get('state')} `{(o.get('commit') or '')[:9]}` | {o.get('load_s')} | "
            f"{r.get('first_step_s')} | {r.get('elapsed_s')} | {r.get('finished')} | "
            f"{w.get('elapsed_s')} | {r.get('error') or ''} |")


def table(obs: dict) -> str:
    rows = ["| state | load s | first step s | first render s | finished | warm render s | error |",
            "|---|---|---|---|---|---|---|"]
    for n in ("base", "head", "main"):
        if obs.get(n):
            rows.append(_fmt(obs[n]))
    return "\n".join(rows)


def _finished_fast(o: dict) -> bool:
    r = _first(o)
    return bool(r.get("finished")) and (r.get("elapsed_s") or 1e9) <= LIMIT_S and r.get("png_ok") is not False


def base_shows_defect(base: dict) -> tuple[bool, str]:
    r = _first(base)
    return (not _finished_fast(base)), f"first render finished={r.get('finished')} in {r.get('elapsed_s')} s"


def head_is_fixed(head: dict) -> tuple[bool, str]:
    r = _first(head)
    return _finished_fast(head), f"first render finished={r.get('finished')} in {r.get('elapsed_s')} s"
