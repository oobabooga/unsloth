#!/usr/bin/env python3
"""Criteria: NPU stack survey + AMD Ryzen AI NPU diffusion on the Linux runner.

This is an exploration, not a PR differential: the probe's heavy part ignores
the checkout, so base vs head can never differ and the verdict word is
NO_REGRESSION by construction. The payload is the observation JSON and the
table below. The gates only prove the probe actually ran at the head (survey
present, heavy part reached a decision), so an empty run reads INCONCLUSIVE.

Pairs with probes/npu_diffusion_bench_probe.py.
"""

from __future__ import annotations

TITLE = "Ryzen AI NPU diffusion on the gfx1151 Linux runner (exploration)"
MODE = "regression"
NEEDS: list[str] = []


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    h = obs.get("head") or {}
    sv = h.get("survey") or {}
    heavy = h.get("heavy")
    decided = isinstance(heavy, dict) and bool(heavy.get("stop") or heavy.get("sandbox") or heavy.get("error") or heavy.get("rai"))
    return [
        ("head survey ran", bool(sv.get("uname")), sv.get("uname", "missing")[:120] if sv else "missing"),
        ("head heavy part reached a decision", decided,
         (heavy.get("stop") or heavy.get("error") or "ran") if isinstance(heavy, dict) else str(heavy)),
    ]


def _cell(r: dict) -> str:
    if not r:
        return "-"
    if r.get("skipped"):
        return f"skipped: {r['skipped']}"
    c = r.get("cold") or {}
    p = r.get("profile") or {}
    s = f"cold rc={c.get('rc')} wall {c.get('wall_s')} s, images {len(r.get('images') or [])}"
    if p:
        s += f"; profile rc={p.get('rc')} wall {p.get('wall_s')} s"
    return s


def table(obs: dict) -> str:
    h = obs.get("head") or {}
    sv = h.get("survey") or {}
    rows = ["| item | value |", "|---|---|"]
    for k in ("os_release", "lspci_npu", "dev_accel", "accel_rw", "lsmod_amdxdna", "amdxdna_params",
              "flm_path", "xrt_system", "ryzen_ai_dirs", "python312", "disk_work"):
        v = str(sv.get(k, "-")).replace("\n", " / ").replace("|", "/")[:200]
        rows.append(f"| {k} | {v} |")
    heavy = h.get("heavy") or {}
    if isinstance(heavy, dict):
        rows.append(f"| heavy stop | {heavy.get('stop') or heavy.get('error') or '-'} |")
        q = ((heavy.get("rai") or {}).get("quicktest") or {})
        rows.append(f"| quicktest rc | {q.get('rc', '-')} |")
        for tag, r in ((heavy.get("sandbox") or {}).get("runs") or {}).items():
            rows.append(f"| {tag} | {_cell(r)} |")
    return "\n".join(rows)


def head_is_worse(base: dict, head: dict) -> tuple[bool, str]:
    return False, ("exploration probe: the measured part ignores the checkout, so base and head "
                   "cannot differ; read the table and the observation JSON")
