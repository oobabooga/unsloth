#!/usr/bin/env python3
"""Criteria: does the head pick the same PyTorch index as the base on this real AMD host?

A messaging-only change to get_torch_index_url must leave its stdout alone. Gated on the
host actually being read as ROCm, or the comparison is cpu vs cpu and proves nothing.
"""

from __future__ import annotations

TITLE = "PyTorch index chosen on real gfx1151, base versus head"
MODE = "regression"
NEEDS: list[str] = ["gpu"]


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out = []
    for name in ("base", "head"):
        o = obs.get(name) or {}
        url = o.get("index_url") or ""
        out.append((f"{name} produced an index", url.startswith("http") and o.get("fn_rc") == 0,
                    o.get("error") or f"rc={o.get('fn_rc')} url={url or '-'}; {str(o.get('driver_tail', ''))[-200:]}"))
    h = obs.get("head") or {}
    out.append(("host ROCm read by amd-smi", bool(h.get("host_rocm")), str(h.get("host_rocm"))))
    out.append(("head took a ROCm index (not cpu)", "/rocm" in (h.get("index_url") or "") or "/gfx" in (h.get("index_url") or ""),
                h.get("index_url") or "-"))
    return out


def table(obs: dict) -> str:
    rows = ["| state | host ROCm | index | cap note | radeon rc |", "|---|---|---|---|---|"]
    for name in ("base", "head"):
        o = obs.get(name) or {}
        note = next((ln for ln in (o.get("stderr") or "").splitlines() if "No validated PyTorch" in ln), "none")
        rows.append(f"| {name} | {o.get('host_rocm')} | {o.get('index_url')} | {note} | {o.get('radeon_curl_rc')} |")
    return "\n".join(rows)


def head_is_worse(base: dict, head: dict) -> tuple[bool, str]:
    b, h = base.get("index_url"), head.get("index_url")
    if b != h:
        return True, f"index moved: base {b} -> head {h}"
    return False, f"same index at both states: {h}"
