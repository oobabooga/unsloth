#!/usr/bin/env python3
"""Criteria: on an integrated GPU, the keep-cpu offload path must not page-lock host weights.

Base pins the kept host weights (the defect: on an APU the "VRAM" is system RAM,
so the pinned copy only takes RAM from the GPU). Head classifies the card as
integrated and pins nothing. The gates make sure the runner really is an
integrated GPU and the probe really reached the pin site, so neither verdict
can pass by the probe failing.
"""

from __future__ import annotations

TITLE = "No pinned host weights on an integrated GPU"
MODE = "differential"
NEEDS = ["gpu", "windows_rocm_wddm", "multi_gpu", "nvidia", "mig", "xpu", "mlx"]


def _states(obs: dict) -> dict:
    return {n: v for n, v in obs.items() if not n.startswith("_")}


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    states = _states(obs)
    flags = {n: v.get("is_integrated") for n, v in states.items()}
    out = [("runner reports an integrated GPU", all(bool(f) for f in flags.values()),
            ", ".join(f"{k}={v}" for k, v in flags.items()))]
    reached = {n: ("pinned_bytes" in v and "pin_error" not in v) for n, v in states.items()}
    out.append(("probe reached _pin_host_weights in every state", all(reached.values()),
                ", ".join(f"{k}={states[k].get('pin_error', 'ok')}" for k in states)))
    return out


def table(obs: dict) -> str:
    rows = ["| state | is_integrated | arch | helper | integrated | pinning allowed | pinned bytes | any pinned |",
            "|---|---|---|---|---|---|---|---|"]
    for name, v in _states(obs).items():
        rows.append(
            f"| {name} | {v.get('is_integrated')} | {v.get('arch')} | {v.get('helper_present')} | "
            f"{v.get('device_is_integrated')} | {v.get('host_pinning_allowed')} | {v.get('pinned_bytes')} | "
            f"{v.get('any_param_pinned')} |"
        )
    return "\n".join(rows)


def base_shows_defect(base: dict) -> bool:
    return bool(base.get("pinned_bytes")) and bool(base.get("any_param_pinned"))


def head_is_fixed(head: dict) -> bool:
    return (
        head.get("device_is_integrated") is True
        and head.get("host_pinning_allowed") is False
        and head.get("pinned_bytes") == 0
        and head.get("any_param_pinned") is False
    )
