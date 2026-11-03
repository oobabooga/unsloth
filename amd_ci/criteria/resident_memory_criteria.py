# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

TITLE = "Diffusion sampler regression on real ROCm"
MODE = "regression"
NEEDS = ["gpu", "rocm", "nvidia", "discrete_gpu", "windows_rocm_wddm", "multi_gpu"]


def gates(observations):
    rows = []
    for arm in ("base", "head"):
        sample = observations.get(arm) or {}
        rows.extend(
            [
                (arm + " ran on ROCm", bool(sample.get("hip")), str(sample.get("hip"))),
                (arm + " completed", not sample.get("error"), str(sample.get("error"))),
                (
                    arm + " imported its checkout",
                    sample.get("imports_checkout") is True,
                    str(sample.get("source")),
                ),
                (
                    arm + " allocated and read GPU data",
                    sample.get("allocated_delta", 0) >= 64 * 2**20
                    and sample.get("sum") == 64 * 2**20,
                    str(sample.get("allocated_delta")),
                ),
            ]
        )
        for field in ("snapshot", "after"):
            snap = sample.get(field) or {}
            rows.append(
                (
                    arm + " read known " + field + " memory",
                    isinstance(snap.get("free_mib"), int)
                    and isinstance(snap.get("total_mib"), int)
                    and 0 <= snap["free_mib"] <= snap["total_mib"]
                    and snap["total_mib"] > 0,
                    str(snap),
                )
            )
    return rows


def head_is_worse(base, head):
    worse = (
        head.get("sum") != base.get("sum")
        or head.get("allocated_delta", 0) < base.get("allocated_delta", 0)
        or any(
            (head.get(field) or {}).get("total_mib")
            != (base.get(field) or {}).get("total_mib")
            for field in ("snapshot", "after")
        )
    )
    return (
        worse,
        "Compared GPU data, allocated bytes and capacity; free memory may vary with co-tenants.",
    )


def table(observations):
    lines = [
        "| state | GPU | memory snapshot | callback API present |",
        "|---|---|---|---|",
    ]
    for arm in ("base", "head"):
        sample = observations.get(arm) or {}
        lines.append(
            f"| {arm} | {sample.get('gpu')} | {sample.get('snapshot')} | "
            f"{sample.get('sampler_accepts_can_stop')} |"
        )
    lines.extend(
        [
            "",
            "Scope: default sampler on real ROCm, with a live 64 MiB GPU allocation and readback.",
            "CUDA resident-fastpath engagement is NOT tested: the production caller excludes ROCm.",
            "No callback is supplied directly; doing so would bypass the caller's backend eligibility.",
            "No physical load-time, inference, or NVIDIA performance claim follows from this run.",
        ]
    )
    return "\n".join(lines)
