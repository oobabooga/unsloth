# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

TITLE = "Diffusion memory sampling on real ROCm"
MODE = "regression"
NEEDS = ["gpu", "rocm", "nvidia", "discrete_gpu", "windows_rocm_wddm", "multi_gpu"]


def gates(observations):
    rows = []
    for arm in ("base", "head"):
        sample = observations.get(arm) or {}
        rows.append(
            (arm + " ran on ROCm", bool(sample.get("hip")), str(sample.get("hip")))
        )
        rows.append(
            (arm + " completed", not sample.get("error"), str(sample.get("error")))
        )
        rows.append(
            (
                arm + " allocated and read GPU data",
                sample.get("allocated_delta", 0) >= 64 * 2**20
                and sample.get("sum") == 64 * 2**20,
                str(sample.get("allocated_delta")),
            )
        )
        snap = sample.get("snapshot") or {}
        rows.append(
            (
                arm + " read known memory",
                isinstance(snap.get("free_mib"), int)
                and isinstance(snap.get("total_mib"), int)
                and snap.get("total_mib", 0) > 0,
                str(snap),
            )
        )
    rows.append(
        (
            "head reuse probe ran",
            isinstance((observations.get("head") or {}).get("reuse_probe"), bool),
            str((observations.get("head") or {}).get("reuse_probe")),
        )
    )
    return rows


def head_is_worse(base, head):
    worse = (
        head.get("sum") != base.get("sum")
        or head.get("allocated_delta", 0) < base.get("allocated_delta", 0)
        or (head.get("snapshot") or {}).get("total_mib")
        != (base.get("snapshot") or {}).get("total_mib")
    )
    return (
        worse,
        "Compared GPU data, allocated bytes and total capacity; free VRAM may vary with co-tenants.",
    )


def table(observations):
    lines = ["| state | GPU | memory snapshot | reuse probe |", "|---|---|---|---|"]
    for arm in ("base", "head"):
        sample = observations.get(arm) or {}
        lines.append(
            f"| {arm} | {sample.get('gpu')} | {sample.get('snapshot')} | {sample.get('reuse_probe', 'not present')} |"
        )
    return "\n".join(lines)
