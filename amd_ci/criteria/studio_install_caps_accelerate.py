#!/usr/bin/env python3
"""Criteria: does a fresh Windows install end on an accelerate that can train?

The question the PR thread asked and nobody had answered: run one full install honouring
the new file and confirm 1.14 survives every later with-deps step. This judges exactly
that, on the installer handoff path (SKIP_STUDIO_BASE=1) rather than on the update path,
because the handoff is the one the constraints file alone never reaches.

`base_shows_defect` requires the base to finish the install and STILL be on 1.15. A base
whose installer crashed is not a demonstration of the defect, it is a broken harness, and
it would otherwise read as a confirmation because the version did indeed stay at 1.15.

Pairs with probes/studio_install_accelerate_probe.py.
"""

from __future__ import annotations

TITLE = "A full installer run with SKIP_STUDIO_BASE=1, base versus head"
MODE = "differential"
NEEDS = ["gpu", "rocm", "windows", "windows_rocm_wddm",
         "multi_gpu", "nvidia", "mig", "xpu", "mlx"]


def _major_minor(v: str | None) -> tuple[int, int] | None:
    if not v:
        return None
    parts = v.split(".")
    try:
        return int(parts[0]), int("".join(c for c in parts[1] if c.isdigit()) or 0)
    except (IndexError, ValueError):
        return None


def _states(obs: dict) -> dict:
    return {k: v for k, v in obs.items() if not k.startswith("_")}


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out: list[tuple[str, bool, str]] = []
    for name, v in _states(obs).items():
        out.append((f"{name}: venv seeded as install.ps1 leaves one",
                    v.get("seeded_accelerate") == "1.15.0" and bool(v.get("seeded_torch")),
                    f"accelerate={v.get('seeded_accelerate')} torch={v.get('seeded_torch')}"
                    f" {v.get('setup_error', '')}".strip()))
        out.append((f"{name}: installer completed",
                    v.get("installer_rc") == 0,
                    f"exit {v.get('installer_rc')}" if v.get("installer_rc") is not None
                    else v.get("setup_error") or v.get("probe_error") or "never ran"))
        # The ROCm torch must still be there. An installer that "fixed" accelerate by
        # dragging a PyPI torch over the ROCm build has broken the thing it protects.
        seeded, final = v.get("seeded_torch"), v.get("final_torch")
        out.append((f"{name}: ROCm torch untouched", bool(final) and final == seeded,
                    f"{seeded} -> {final}"))
    return out


def table(obs: dict) -> str:
    rows = ["| state | seeded accelerate | after a full install | torch | installer |",
            "|---|---|---|---|---|"]
    for name, v in _states(obs).items():
        rows.append(
            f"| {name} | {v.get('seeded_accelerate') or '-'} | "
            f"**{v.get('final_accelerate') or '-'}** | {v.get('final_torch') or '-'} | "
            f"exit {v.get('installer_rc')} |"
        )
    return "\n".join(rows)


def base_shows_defect(base: dict) -> bool:
    if base.get("installer_rc") != 0:
        return False
    mm = _major_minor(base.get("final_accelerate"))
    return mm is not None and mm >= (1, 15)


def head_is_fixed(head: dict) -> bool:
    if head.get("installer_rc") != 0:
        return False
    mm = _major_minor(head.get("final_accelerate"))
    return mm is not None and mm < (1, 15)
