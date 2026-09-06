#!/usr/bin/env python3
"""Criteria: does this change move the bundle choice on a host it does not target?

The Vulkan widening is inside `if is_linux`, and `install.ps1` is untouched, so on
a Windows AMD box the claim being checked is a negative one: same host reading,
same first bundle, at both states. `MODE = "regression"` is the right shape for a
negative -- differential mode would demand the base exhibit a defect, and here the
whole point is that it must not.

A regression check on an unchanged path is easy to write vacuously, so the gates
insist the reading is real: the probe imported the module, resolved a host, and
produced at least one candidate bundle. "Both states returned nothing" is not
agreement.

Pairs with probes/llama_backend_routing_probe.py.
"""

from __future__ import annotations

TITLE = "llama.cpp bundle selection, unchanged off the targeted path"
NEEDS = [
    "windows", "gpu", "vulkan", "rocm", "windows_rocm_wddm", "windows_docker",
    "gfx1033_vangogh", "glibc_pre_228", "xpu", "nvidia", "mlx", "discrete_gpu",
    "amdvlk", "multi_gpu",
]
MODE = "regression"

# The fields whose agreement across states IS the claim. `has_amd_gpu_without_rocm`
# is deliberately excluded: it does not exist at the base, so comparing it would
# report the new field itself as the difference and every run would go red.
_COMPARED = (
    "is_linux", "is_windows", "is_macos", "is_x86_64", "is_arm64",
    "has_physical_nvidia", "has_usable_nvidia", "has_rocm", "has_intel_gpu",
    "rocm_gfx_target",
)


def _reading(o: dict) -> dict:
    h = o.get("host") or {}
    return {k: h.get(k) for k in _COMPARED}


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out: list[tuple[str, bool, str]] = []
    for name in ("base", "head"):
        o = obs.get(name) or {}
        err = o.get("error")
        out.append((f"{name} probe produced a reading", not err and bool(o.get("host")),
                    err or "ok"))
        # Without a candidate there is no selection to compare, and "no bundle at
        # either state" would otherwise read as a clean no-change result.
        out.append((f"{name} selected at least one bundle",
                    bool(o.get("attempts")),
                    f"first={o.get('first_install_kind')}, "
                    f"plan_error={o.get('plan_error')}"))
    return out


def table(obs: dict) -> str:
    rows = ["| state | platform | has_rocm | has_intel_gpu | has_physical_nvidia "
            "| first bundle | attempts |",
            "|---|---|---|---|---|---|---|"]
    for name in ("base", "head", "merge"):
        o = obs.get(name)
        if not o:
            continue
        h = o.get("host") or {}
        attempts = ", ".join(f"`{a['install_kind']}`" for a in o.get("attempts", [])) or "none"
        rows.append(
            f"| {name} | {o.get('platform')} | {h.get('has_rocm')} | {h.get('has_intel_gpu')} "
            f"| {h.get('has_physical_nvidia')} | `{o.get('first_install_kind')}` | {attempts} |"
        )
    return "\n".join(rows)


def head_is_worse(base: dict, head: dict) -> tuple[bool, str]:
    b, h = _reading(base), _reading(head)
    drifted = sorted(k for k in _COMPARED if b.get(k) != h.get(k))
    if drifted:
        return True, ("the host reading moved: "
                      + ", ".join(f"`{k}` {b.get(k)} -> {h.get(k)}" for k in drifted))
    bb = [a["install_kind"] for a in base.get("attempts", [])]
    hh = [a["install_kind"] for a in head.get("attempts", [])]
    if bb != hh:
        return True, f"the candidate order moved: {bb} -> {hh}"
    return False, (f"identical at both states: host reading unchanged and the same "
                   f"candidate order {bb}")
