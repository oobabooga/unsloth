#!/usr/bin/env python3
"""Does this host's ROCm APU report a window that the panel calls dedicated VRAM?

The defect: `hardware.py` sets `shared_memory` only on Windows, so a Linux ROCm
APU arrives as `unified_memory: true, shared_memory: false`, and
`gpuMemoryTotalsGb` splits on `shared_memory` alone. The GTT window is then
counted as VRAM standing beside system RAM, and the System panel prints it as
the card's capacity.

Judged on the RENDERED capacity label of each state's own component, not on
whether a helper exists. The two states differ by one predicate inside
`gpuMemoryTotalsGb`, so a criteria module asking "was the file changed" would
pass on any branch that touched it.

Non-vacuity, each failing rather than skipping:

  * the host actually reported a unified-memory device. Without one this run
    never reaches the changed branch, and says so rather than ticking green for
    a path it could not enter.
  * both states rendered.
  * the Vulkan inventory is identical at both states. This change is about the
    torch inventory; the Vulkan rows are the nearest thing on this host it must
    not have touched, and two open PRs are working in them.
"""

from __future__ import annotations

import json
import re

TITLE = "Unified-memory APU totals on gfx1151, base versus head"
MODE = "differential"

NEEDS: list[str] = [
    "gpu", "rocm", "integrated_gpu", "discrete_gpu", "vulkan",
    "multi_gpu", "nvidia", "xpu", "mlx", "windows", "windows_docker",
]

ROCM_CASE = "measured_rocm_only"
# The old label: a bare capacity with no account of where the memory lives.
# The new one carries the shared split the same way a Windows APU always has.
SHARED_MARKER = "shared"

_CTX: dict = {}


def _case(state: dict, name: str, surface: str = "resources") -> dict:
    render = state.get("render") or {}
    case = ((render.get("cases") or {}).get(name) or {}).get(surface) or {}
    return case if isinstance(case, dict) else {}


def _unified_devices(state: dict) -> list[dict]:
    devices = ((state.get("backend_gpu") or {}).get("devices")) or []
    return [d for d in devices
            if d.get("unified_memory") is True and d.get("shared_memory") is not True]


def _capacity_label(state: dict) -> str:
    """The VRAM tile's capacity reading, which is what the split renders into."""
    text = _case(state, ROCM_CASE).get("text") or ""
    match = re.search(r"\|\s*VRAM\s*\|[^|]*\|\s*([^|]+)\|", text)
    return (match.group(1) if match else "").strip()


def _vulkan_view(state: dict) -> str:
    return json.dumps(state.get("vulkan_gpu"), sort_keys = True, default = str)


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    base, head = obs.get("base") or {}, obs.get("head") or {}
    _CTX["base"], _CTX["head"] = base, head
    out: list[tuple[str, bool, str]] = []

    for name in ("base", "head"):
        state = obs.get(name) or {}
        unified = _unified_devices(state)
        out.append((
            f"{name}: the host reports a unified-memory device the flag does not cover",
            bool(unified),
            "; ".join(
                f"{d.get('name')}: unified_memory={d.get('unified_memory')}, "
                f"shared_memory={d.get('shared_memory')}, "
                f"memory_total_gb={d.get('memory_total_gb')}"
                for d in unified
            ) or "none, so this host never reaches the changed branch",
        ))
        label = _capacity_label(state)
        out.append((
            f"{name}: the Resources tab rendered a capacity",
            bool(label),
            label or (_case(state, ROCM_CASE).get("error")
                      or (state.get("render") or {}).get("error")
                      or "no VRAM tile in the render"),
        ))

    same_vulkan = _vulkan_view(base) == _vulkan_view(head)
    out.append((
        "the Vulkan inventory is identical at both states",
        same_vulkan,
        "byte-identical" if same_vulkan else
        "the Vulkan rows differ between the states, so this run cannot say the "
        "change left the inference inventory alone",
    ))
    return out


def base_shows_defect(base: dict) -> tuple[bool, str]:
    label = _capacity_label(base)
    if not label:
        return False, "the base render shows no capacity at all"
    if SHARED_MARKER in label.lower():
        return False, (f"the base already reports the window as shared: {label!r}, "
                       f"so there is nothing here to fix")
    unified = _unified_devices(base)
    total = unified[0].get("memory_total_gb") if unified else None
    return True, (f"base prints {label!r} as the card's capacity while the device "
                  f"reports unified_memory with memory_total_gb={total}")


def head_is_fixed(state: dict) -> tuple[bool, str]:
    label = _capacity_label(state)
    if not label:
        return False, "the head render shows no capacity at all"
    if SHARED_MARKER not in label.lower():
        return False, f"the head still prints the window as plain capacity: {label!r}"
    unified = _unified_devices(state)
    total = unified[0].get("memory_total_gb") if unified else None
    # The aggregate must survive the split: a fit verdict is measured against it.
    if total is not None and str(total).rstrip("0").rstrip(".") not in label.replace(",", ""):
        return False, (f"the shared figure in {label!r} does not carry the device's "
                       f"reported {total} GiB")
    return True, f"head reports the window as shared host memory: {label!r}"


def table(obs: dict) -> str:
    rows = ["| state | VRAM tile capacity |", "|---|---|"]
    for name in ("base", "head"):
        rows.append(f"| {name} | `{_capacity_label(obs.get(name) or {}) or '(not rendered)'}` |")
    unified = _unified_devices(obs.get("base") or {})
    if unified:
        d = unified[0]
        rows += ["", "Measured ROCm row: "
                 f"`name={d.get('name')}`, `index_kind={d.get('index_kind')}`, "
                 f"`shared_memory={d.get('shared_memory')}`, "
                 f"`unified_memory={d.get('unified_memory')}`, "
                 f"`memory_total_gb={d.get('memory_total_gb')}`"]
    return "\n".join(rows)
