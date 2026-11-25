#!/usr/bin/env python3
"""Criteria for probes/r6win_probe.py (R6-A n0#1, GPU_RESOURCE_CACHE_SIZE=64 on Windows gfx1151). Measurement only:
the gates prove the arms ran on gfx1151 with five timed renders each; KEEP / KILL is read from the observations."""

from __future__ import annotations

import statistics as st

TITLE = "gfx1151 Windows: GPU_RESOURCE_CACHE_SIZE=64 vs unset, FLUX.3 klein, 5 mixed-size jobs per process"
MODE = "regression"
NEEDS = ["gpu", "rocm", "windows", "exclusive_gpu", "nvidia", "multi_gpu"]


def _ok(h):
    return [r for r in h.get("arms") or [] if r.get("rc") == 0 and len(r.get("renders") or []) >= 5]


def gates(obs: dict) -> list:
    h = obs.get("head") or {}
    arch = h.get("arch") or ""
    ok = _ok(h)
    return [("gfx1151", str(arch).startswith("gfx1151"), str(arch)),
            ("off arms", sum(r["setting"] == "off" for r in ok) >= 4, str(sum(r["setting"] == "off" for r in ok))),
            ("on arms", sum(r["setting"] == "on" for r in ok) >= 4, str(sum(r["setting"] == "on" for r in ok)))]


def head_is_worse(base: dict, head: dict) -> tuple:
    return False, "measurement only (arms differ by one env var inside the head step)"


def table(obs: dict) -> str:
    h = obs.get("head") or {}
    L = [f"torch {h.get('torch')} hip {h.get('hip')} gpu {h.get('gpu')}"]
    for r in h.get("arms") or []:
        ws = [x.get("wall_s") for x in r.get("renders") or []]
        L.append(f"{r.get('idx')} {r.get('setting')} rc={r.get('rc')} load {r.get('load_s')} jobs {ws}")
    for s in ("off", "on"):
        rs = [r for r in _ok(h) if r["setting"] == s]
        if rs:
            j1 = [r["renders"][0]["wall_s"] for r in rs]
            rest = [st.median([x["wall_s"] for x in r["renders"][1:]]) for r in rs]
            L.append(f"{s}: job1 median {st.median(j1):.2f} {j1}; jobs2-5 medians {rest}")
    return "\n".join(L)
