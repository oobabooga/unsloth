#!/usr/bin/env python3
"""Criteria: ZLUDA on PATH must not read as an NVIDIA driver on an AMD Windows host.

The defect: the installer's driver-library probe bound nvcuda.dll / nvml.dll by bare name, so
the loader walked PATH and found ZLUDA's, which answer like a CUDA driver. Base should report an
inventory with a ZLUDA build on PATH; head should report none. The control (nothing added) must
be empty on both: a host whose real driver answers cannot show this defect. The Python twin
(studio/nvidia_probe.py) is unchanged by the PR, so it must read the same on both states.

Pairs with probes/zluda_inventory_probe.py.
"""

from __future__ import annotations

TITLE = "Installer NVIDIA inventory with ZLUDA on PATH (Windows)"
MODE = "differential"
NEEDS = ["windows", "nvidia"]


def _zluda(state: dict) -> dict:
    return {k: v for k, v in (state.get("scenarios") or {}).items() if k != "control"}


def _inv(row: dict, kind: str):
    return ((row or {}).get(kind) or {}).get("inventory")


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out = []
    for name in ("base", "head"):
        o = obs.get(name) or {}
        sc = o.get("scenarios") or {}
        out.append((f"{name}: at least one ZLUDA build probed", len(_zluda(o)) > 0,
                    ", ".join(_zluda(o)) or "none"))
        ran = all((r.get("ps") or {}).get("ran") for r in sc.values())
        out.append((f"{name}: every PowerShell read ran", bool(sc) and ran,
                    "; ".join(f"{k}: {(r.get('ps') or {}).get('stderr', '')[:120]}"
                              for k, r in sc.items() if not (r.get("ps") or {}).get("ran")) or "ok"))
        ctl = sc.get("control") or {}
        out.append((f"{name}: control reads no NVIDIA driver", _inv(ctl, "ps") is None and _inv(ctl, "py") is None,
                    f"ps={_inv(ctl, 'ps')} py={_inv(ctl, 'py')} system32 nvcuda={o.get('system32_nvcuda')}"))
    b, h = obs.get("base") or {}, obs.get("head") or {}
    same = all(_inv((b.get("scenarios") or {}).get(k), "py") == _inv(r, "py")
               for k, r in (h.get("scenarios") or {}).items())
    out.append(("Python twin (unchanged file) reads the same on both states", same, ""))
    return out


def table(obs: dict) -> str:
    rows = ["| state | scenario | installer (setup.ps1, PS 5.1) | Studio twin (nvidia_probe.py) |",
            "|---|---|---|---|"]
    for name in ("base", "head", "merge"):
        o = obs.get(name)
        if not o:
            continue
        for k, r in (o.get("scenarios") or {}).items():
            rows.append(f"| {name} | {k} | `{_inv(r, 'ps')}` | `{_inv(r, 'py')}` |")
    return "\n".join(rows)


def base_shows_defect(base: dict) -> bool:
    return any(_inv(r, "ps") is not None for r in _zluda(base).values())


def head_is_fixed(head: dict) -> bool:
    z = _zluda(head)
    return bool(z) and all(_inv(r, "ps") is None for r in z.values())
