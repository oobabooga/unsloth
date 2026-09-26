#!/usr/bin/env python3
"""Criteria: the PR's Windows-only change leaves the Linux inventory unchanged, ZLUDA or not.

On Linux ZLUDA is reached through LD_LIBRARY_PATH, which the installer's bare libcuda.so.1 /
libnvidia-ml.so.1 names honour on both states. Regression mode: every scenario must read the
same at base and head. Whether ZLUDA fools the Linux probe is recorded, not judged here.

Pairs with probes/zluda_inventory_probe.py.
"""

from __future__ import annotations

TITLE = "NVIDIA inventory with ZLUDA on LD_LIBRARY_PATH (Linux), base versus head"
MODE = "regression"
NEEDS = ["linux", "nvidia"]


def _inv(row: dict, kind: str):
    return ((row or {}).get(kind) or {}).get("inventory")


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out = []
    for name in ("base", "head"):
        o = obs.get(name) or {}
        sc = o.get("scenarios") or {}
        out.append((f"{name}: at least one ZLUDA build probed", len(sc) > 1, ", ".join(sc)))
        out.append((f"{name}: every Python read ran", bool(sc) and all((r.get("py") or {}).get("ran") for r in sc.values()),
                    "; ".join(f"{k}: rc={(r.get('py') or {}).get('rc')}" for k, r in sc.items())))
    return out


def table(obs: dict) -> str:
    rows = ["| state | scenario | setup.ps1 under pwsh | nvidia_probe.py |", "|---|---|---|---|"]
    for name in ("base", "head", "merge"):
        o = obs.get(name)
        if not o:
            continue
        for k, r in (o.get("scenarios") or {}).items():
            ps = _inv(r, "ps") if (r.get("ps") or {}).get("ran") else "(no pwsh)"
            rows.append(f"| {name} | {k} | `{ps}` | `{_inv(r, 'py')}` |")
    return "\n".join(rows)


def head_is_worse(base: dict, head: dict) -> tuple[bool, str]:
    bs, hs = base.get("scenarios") or {}, head.get("scenarios") or {}
    moved = [k for k in hs if (_inv(bs.get(k), "py"), _inv(bs.get(k), "ps")) != (_inv(hs[k], "py"), _inv(hs[k], "ps"))]
    fooled = [k for k, r in bs.items() if k != "control" and (_inv(r, "py") or _inv(r, "ps"))]
    note = ("; ZLUDA on LD_LIBRARY_PATH reads as NVIDIA on BOTH states for: " + ", ".join(fooled)) if fooled else \
           "; no ZLUDA build read as NVIDIA on either state"
    if moved:
        return True, "scenarios whose reading moved: " + ", ".join(moved) + note
    return False, "every scenario reads the same on base and head" + note
