#!/usr/bin/env python3
"""Criteria for PR 9829, on the real GPU: does the reroute disturb a healthy host?

The change adds a repair pass that reinstalls Torch from AMD's per-architecture
index when the generic ROCm wheel carries no kernels for the runtime target. This
machine is not such a host: gfx1151 is in the generic wheel's architecture list,
so the correct behaviour here is that nothing changes. That is the claim this
module judges, and it is the only one about the missing-kernel routing that real
AMD hardware can settle in this pool.

Regression mode, deliberately. There is no defect to reproduce on a gfx1151, so
asking for one would be a differential the base cannot fail honestly.

Judges, never observes. Pairs with probes/rocm_route_host_probe.py.
"""

from __future__ import annotations

TITLE = "PR 9829: what each state decides about this real gfx1151 host"
MODE = "regression"

# The change routes by architecture across Windows, mixed discrete + integrated
# hosts, and multi-GPU visible-device masks. This host is one integrated AMD GPU
# on Linux, so the rest is declared and reported as untested rather than implied.
NEEDS = [
    "linux", "rocm", "gpu", "integrated_gpu", "amd_smi",
    "windows", "multi_gpu", "multi_gpu_amd", "discrete_gpu",
]

# Answers that must not move between the states on a host the change should not
# be touching at all.
COMPARED = ("target_gfx", "gfx_codes", "rocm_version", "kfd_targets",
            "amd_arch_index_url", "strix_needs_amd_arch_index")


def _norm(v):
    if isinstance(v, (list, tuple)):
        return [_norm(x) for x in v]
    return v


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out: list[tuple[str, bool, str]] = []
    base = obs.get("base") or {}
    head = obs.get("head") or {}

    for name, o in (("base", base), ("head", head)):
        out.append((f"{name} installer loaded", not o.get("error") and bool(o.get("symbols")),
                    o.get("error") or f"{sum(bool(v) for v in (o.get('symbols') or {}).values())} "
                                      f"of the queried symbols present"))

    # Non-vacuity: "the two states agree" is worth nothing if neither of them saw
    # the GPU. The agreement has to be about a real detection.
    for name, o in (("base", base), ("head", head)):
        out.append((f"{name} detected this host's GPU on real hardware",
                    bool(o.get("gfx_codes")), f"gfx_codes={o.get('gfx_codes')}"))

    out.append(("the installed wheel's architecture list was read",
                bool(head.get("wheel_arch_list")),
                f"torch={head.get('torch_version')} hip={head.get('torch_hip')} "
                f"{len(head.get('wheel_arch_list') or [])} arches"))
    return out


def table(obs: dict) -> str:
    rows = ["| question | base | head |", "|---|---|---|"]
    base, head = obs.get("base") or {}, obs.get("head") or {}
    for key in COMPARED:
        b, h = base.get(key, "n/a"), head.get(key, "n/a")
        rows.append(f"| `{key}` | `{b}` | `{h}` |")

    extra = [""]
    arches = head.get("wheel_arch_list") or []
    target = head.get("target_gfx")
    if arches:
        extra.append(f"Installed wheel: `{head.get('torch_version')}` "
                     f"(hip `{head.get('torch_hip')}`), architectures "
                     f"`{', '.join(arches)}`.")
        extra.append("")
        extra.append(f"This host's target `{target}` is "
                     f"{'PRESENT in' if target in arches else '**ABSENT from**'} that list, and "
                     f"the head declares the generic wheel set as "
                     f"`{', '.join(head.get('declared_generic_wheel_gfx') or []) or 'n/a'}`.")
    extra.append("")
    extra.append("gfx1103 and gfx1031-gfx1036, the architectures this change repairs, are not "
                 "present in this pool. Nothing here executes a kernel on one.")
    return "\n".join(rows + extra)


def head_is_worse(base: dict, head: dict) -> tuple[bool, str]:
    if head.get("error") and not base.get("error"):
        return True, f"the head's installer failed to answer on this host: {head['error']}"

    moved = [k for k in COMPARED
             if k in base and k in head and _norm(base[k]) != _norm(head[k])]
    if moved:
        return True, ("the head changes what this healthy gfx1151 host resolves: "
                      + "; ".join(f"`{k}` {base[k]!r} -> {head[k]!r}" for k in moved))

    # A head-only answer is an addition, not a change, but it must not claim this
    # host needs the repair the change adds.
    if head.get("generic_lacks_kernels_for_target"):
        return True, ("the head reports that the generic wheel lacks kernels for this host's "
                      f"target {head.get('target_gfx')!r}, which contradicts the architecture "
                      f"list read from the installed wheel")

    arches = head.get("wheel_arch_list") or []
    target = head.get("target_gfx")
    if arches and target and target not in arches:
        return True, (f"the wheel installed here carries no kernels for {target}, so this host "
                      f"is itself in the state the change repairs and the comparison above is "
                      f"not about a healthy host")

    detail = (f"both states resolve `{target}` on this machine and route it identically; "
              f"the head adds no repair for it")
    only_head = [k for k in COMPARED if k in head and k not in base]
    if only_head:
        detail += ". New answers at the head, with no base counterpart to compare: " + \
                  ", ".join(f"`{k}`={head[k]!r}" for k in only_head)
    return False, detail
