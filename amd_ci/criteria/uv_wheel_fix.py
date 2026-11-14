#!/usr/bin/env python3
"""Criteria for probes/uv_wheel_probe.py. Defect: in a pip-less uv venv with uv off PATH the CUDA
runtime wheel step fails (base). Fixed: the same test passes and no other test in the file fails."""

from __future__ import annotations

TITLE = "CUDA runtime wheel step in a pip-less uv venv with uv off PATH (Windows)"
MODE = "differential"
NEEDS = ["windows"]
REPRO = "test_a_pipless_uv_venv_with_uv_off_path_gets_the_wheels"


def _states(obs):
    return {n: v for n, v in obs.items() if not n.startswith("_")}


def gates(obs):
    states = _states(obs)
    out = [(f"{n}: the repro test ran (not skipped)", (v.get("results") or {}).get(REPRO) in ("passed", "failed"),
            str((v.get("results") or {}).get(REPRO))) for n, v in states.items()]
    out.append(("uv found on the runner", all(v.get("uv") for v in states.values()),
                ", ".join(f"{n}={v.get('uv')}" for n, v in states.items())))
    base = states.get("base") or {}
    out.append(("uv lookup absent at base, present at head",
                base.get("installer_has_uv_lookup") is False
                and all(v.get("installer_has_uv_lookup") for n, v in states.items() if n != "base"),
                ", ".join(f"{n}={v.get('installer_has_uv_lookup')}" for n, v in states.items())))
    return out


def table(obs):
    rows = ["| state | rc | " + REPRO + " | failed tests |", "|---|---|---|---|"]
    for n, v in _states(obs).items():
        r = v.get("results") or {}
        rows.append(f"| {n} | {v.get('rc')} | {r.get(REPRO)} | {', '.join(k for k, s in r.items() if s == 'failed') or 'none'} |")
    return "\n".join(rows)


def base_shows_defect(base):
    return (base.get("results") or {}).get(REPRO) == "failed"


def head_is_fixed(head):
    r = head.get("results") or {}
    return r.get(REPRO) == "passed" and not any(s == "failed" for s in r.values())
