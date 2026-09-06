#!/usr/bin/env python3
"""Criteria: is the host-pinned VRAM discount withheld on a unified-memory APU?

PR 9931 removes host-pinned embeddings from a discrete device's VRAM budget. On
gfx1151 the device pool IS system RAM, so removing them would promise VRAM the
load then takes: the under-count direction. The PR's own contract is that a
shared or unclassifiable device gets no discount. This asks whether that holds
on the real silicon, through the same seams load_model composes.

Regression mode: the base has no discount at all, so the base cannot be "broken"
in the differential sense. The question is whether the HEAD starts discounting
on a host where nothing was discounted before, which is what head_is_worse asks.

Non-vacuity gate: with shared_memory forced False and a patched 1 GiB reading,
the discount must come back as 1 GiB. Otherwise "applied 0" is silence about a
path that never ran.

Pairs with probes/host_pinned_discount_probe.py.
"""

from __future__ import annotations

TITLE = "Host-pinned VRAM discount on a unified-memory APU (gfx1151)"
MODE = "regression"
# Declared for the change: the discount reasons about every backend's memory
# topology, so every platform it could be wrong on belongs here.
NEEDS = ["gpu", "rocm", "windows_rocm_wddm", "vulkan", "multi_gpu", "nvidia", "mig", "xpu", "mlx"]

GIB = 1024 ** 3


def _head(obs: dict) -> dict:
    return obs.get("head") or {}


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    head = _head(obs)
    out: list[tuple[str, bool, str]] = []
    arch = head.get("arch")
    out.append(("the host really is gfx1151",
                bool(arch and "gfx1151" in str(arch)),
                f"arch={arch} is_integrated={head.get('is_integrated')} "
                f"torch={head.get('torch_version')} hip={head.get('torch_hip')}"))
    out.append(("the head exposes the discount and the classification seam",
                bool(head.get("has_discount")) and bool(head.get("has_classification_seam")),
                f"has_discount={head.get('has_discount')} "
                f"has_classification_seam={head.get('has_classification_seam')} "
                f"error={head.get('backend_error') or head.get('import_error') or '-'}"))
    cand = int(head.get("discount_candidate") or 0)
    out.append(("the discount path is live (patched 1 GiB reading returns 1 GiB when discrete)",
                cand == GIB,
                f"discount_candidate={cand}"))
    return out


def table(obs: dict) -> str:
    rows = ["| state | arch | is_integrated | classify unified | classification known | apu detector | shared | candidate | applied |",
            "|---|---|---|---|---|---|---|---|---|"]
    for name, v in obs.items():
        if name.startswith("_"):
            continue
        rc = v.get("rocm_classify") or {}
        rows.append(
            f"| {name} | {v.get('arch')} | {v.get('is_integrated')} | {rc.get('unified', '-')} | "
            f"{v.get('classification_known', 'absent')} | {v.get('apu_wants_unified', '-')} | "
            f"{v.get('shared_memory', '-')} | {v.get('discount_candidate', 'absent')} | "
            f"{v.get('discount_applied', 'absent')} |")
    return "\n".join(rows)


def head_is_worse(base: dict, head: dict) -> tuple[bool, str]:
    rc = head.get("rocm_classify") or {}
    if not rc.get("unified"):
        return True, (f"_rocm_classify_unified_memory called this gfx1151 NOT unified "
                      f"(arch={rc.get('arch')}, is_integrated={head.get('is_integrated')}); "
                      "the per-device classifier is wrong on the real APU")
    if head.get("classification_known") is False:
        return True, ("the classification seam reported the APU as unclassifiable; the "
                      "budget still fails safe (shared), but the seam did not answer")
    if not head.get("shared_memory"):
        return True, ("load_model's composed shared_memory came out False on a unified "
                      "pool, so the discount WOULD be applied to memory the CPU shares")
    applied = int(head.get("discount_applied") or 0)
    if applied != 0:
        return True, f"a {applied}-byte host-pinned discount was applied on a unified-memory host"
    return False, ("the discount is withheld on the real gfx1151 unified pool (applied 0) "
                   "while the same call with shared_memory forced False returns the patched "
                   f"{head.get('discount_candidate')} bytes, so the zero is attributable to the "
                   "detected topology and not to a dead path; classifier says unified, "
                   f"classification_known={head.get('classification_known')}, "
                   f"apu_detector={head.get('apu_wants_unified')}")
