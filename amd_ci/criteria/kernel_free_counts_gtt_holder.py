#!/usr/bin/env python3
"""Criteria: trusted_mem_get_info counts GTT held outside HIP (Vulkan-style GEM buffers) on Linux ROCm."""

from __future__ import annotations

TITLE = "trusted_mem_get_info against a non-HIP GTT holder"
MODE = "differential"
NEEDS = ["gpu", "multi_gpu", "nvidia", "xpu", "mlx"]
_CTX: dict = {"held": 0.0}


def _states(obs):
    return {n: v for n, v in obs.items() if not n.startswith("_")}


def gates(obs):
    held = float((obs.get("_fixture") or {}).get("allocated_gib") or 0.0)
    _CTX["held"] = held
    st = _states(obs)
    hip = [v.get("hip_free_gib") for v in st.values()]
    out = [("holder raised kernel GTT use >= 2 GiB", held >= 2.0, f"{held:.2f} GiB")]
    out.append(("every state read trusted_mem_get_info",
                all(v.get("trusted_free_gib") is not None for v in st.values()),
                "; ".join(f"{n}: {v.get('error', 'ok')}" for n, v in st.items())))
    out.append(("HIP free blind to the holder (the defect's premise)",
                all(h is not None and v.get("sysfs_gtt_free_gib") is not None
                    and h - v["sysfs_gtt_free_gib"] >= 0.5 * held for h, v in zip(hip, st.values())),
                ", ".join(f"{n}: hip {v.get('hip_free_gib')} sysfs {v.get('sysfs_gtt_free_gib')}"
                          for n, v in st.items())))
    out.append(("observer stayed a bystander",
                all((v.get("observer_allocated_gib") or 0) < 0.5 for v in st.values()), ""))
    return out


def table(obs):
    rows = ["| state | HIP free | trusted free | kernel helper | sysfs GTT free | HIP total |",
            "|---|---|---|---|---|---|"]
    f = lambda x: "-" if x is None else f"{x:.2f}"
    for n, v in _states(obs).items():
        rows.append(f"| {n} | {f(v.get('hip_free_gib'))} | {f(v.get('trusted_free_gib'))} | "
                    f"{f(v.get('kernel_free_gib'))} | {f(v.get('sysfs_gtt_free_gib'))} | {f(v.get('hip_total_gib'))} |")
    rows += ["", f"Holder: {_CTX['held']:.2f} GiB of GTT via amdgpu GEM_CREATE (GiB)."]
    return "\n".join(rows)


def base_shows_defect(base):
    t, s = base.get("trusted_free_gib"), base.get("sysfs_gtt_free_gib")
    return t is not None and s is not None and t - s >= 0.5 * _CTX["held"]


def head_is_fixed(head):
    t, s = head.get("trusted_free_gib"), head.get("sysfs_gtt_free_gib")
    return t is not None and s is not None and abs(t - s) <= 0.5
