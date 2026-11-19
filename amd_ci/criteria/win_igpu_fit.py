#!/usr/bin/env python3
"""Criteria for probes/win_igpu_fit_probe.py: discovery run. Gates only prove llama-server loaded on
the iGPU; the finding is read from the observations (fit's free figure vs RAM, RAM drop per load)."""

from __future__ import annotations

TITLE = "Windows iGPU: llama.cpp fit view vs physical memory"
MODE = "regression"
NEEDS = ["gpu", "windows"]


def _states(obs: dict) -> dict:
    return {k: v for k, v in obs.items() if not k.startswith("_") and isinstance(v, dict)}


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out = []
    for name, o in _states(obs).items():
        runs = o.get("runs") or []
        out.append((f"{name}: probe ran", not o.get("error"), str(o.get("error") or "ok")))
        out.append((f"{name}: small load ready", bool(runs) and runs[0].get("status") == "ready",
                    str(runs[0].get("status") if runs else None)))
    return out


def table(obs: dict) -> str:
    rows = ["| state | run | status | avail before | min avail | min commit avail | private loaded |",
            "|---|---|---|---|---|---|---|"]
    for name, o in _states(obs).items():
        for r in o.get("runs") or []:
            rows.append(f"| {name} | {r.get('name')} | {r.get('status')} | "
                        f"{(r.get('before') or {}).get('avail_phys_mib')} | {r.get('min_avail_phys_mib')} | "
                        f"{r.get('min_commit_avail_mib')} | {(r.get('loaded_proc') or {}).get('private_mib')} |")
    return "\n".join(rows)


def head_is_worse(base: dict, head: dict):
    return False, "discovery run: no comparison"
