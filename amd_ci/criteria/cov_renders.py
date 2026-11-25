#!/usr/bin/env python3
"""Criteria for probes/cov_probe.py: gfx1151 coverage renders through Studio (head tree only). Nothing is judged
between states; the gates prove every arm ran to a recorded end (a render, or a load / generate error with its
message: a clean refusal is a result, a missing record or a timeout is not). Numbers are read from the table."""

from __future__ import annotations

TITLE = "gfx1151: Studio coverage renders (H3-Image GGUF, MiniMax-H3 video, Wan2.2 A14B)"
MODE = "regression"
NEEDS = ["gpu", "rocm", "exclusive_gpu", "nvidia", "multi_gpu"]


def gates(obs: dict) -> list:
    h = obs.get("head") or {}
    arms = h.get("arms") or {}
    out = [("arms recorded", bool(arms), str(list(arms)))]
    for tag, r in arms.items():
        out.append((f"{tag} ended with a record", r.get("verdict") in ("ok", "error") and r.get("rc") != "timeout",
                    f"verdict {r.get('verdict')} rc {r.get('rc')}"))
    return out


def head_is_worse(base: dict, head: dict) -> tuple:
    return False, "measurement only (head tree)"


def table(obs: dict) -> str:
    h = obs.get("head") or {}
    L = ["| arm | model | verdict | load s | render wall s | s/step | peak alloc GiB | error |", "|---|---|---|---|---|---|---|---|"]
    for tag, r in (h.get("arms") or {}).items():
        rs = r.get("renders") or []
        L.append(f"| {tag} | {r.get('model')} | {r.get('verdict')} | {r.get('load_s')} | "
                 f"{[x.get('wall_s') for x in rs]} | {[x.get('s_per_step') for x in rs]} | "
                 f"{[x.get('peak_alloc_gib') for x in rs]} | {str(r.get('error') or '')[:200]} |")
    return "\n".join(L)
