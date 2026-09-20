#!/usr/bin/env python3
"""Does this host's integrated GPU reach the broken display, and does the head fix it?

The defect: a Vulkan integrated GPU reports allocation headroom with no usage
figure, and the System panel renders that as `Unknown / <headroom>` with an empty
usage bar, which reads as an undetected card.

The judgement is made on the RENDERED text of each state's own components, not on
whether the head's new hook imports. A criteria module that asked "does
gpu-memory-display.ts exist" would pass on any branch that added the file.

Three things have to hold before the comparison means anything, and each fails
rather than skips:

  * ggml classified a device on this host as integrated. Without that the host
    never reaches the changed branch and the run says so, rather than reporting a
    green tick for a code path it could not enter.
  * both states rendered. A state that threw is not a state that showed anything.
  * the ROCm rows are identical between the states. The change claims to be
    presentation for Vulkan shared memory only; the APU's ROCm view is the
    nearest thing on this host that it must not have touched.
"""

from __future__ import annotations

import json

TITLE = "Vulkan integrated GPU memory display on gfx1151, base versus head"
MODE = "differential"

# Declared from what the CHANGE touches, not from what this host has. capability.py
# subtracts the host and prints the remainder; an under-declared list would let the
# report imply a reach it does not have.
NEEDS: list[str] = [
    "gpu", "vulkan", "integrated_gpu", "rocm", "discrete_gpu",
    "multi_gpu", "nvidia", "xpu", "mlx", "windows", "windows_docker",
]

SHARED_CASE = "measured_vulkan_only"
# The old sentence. `Unknown /` is the used-over-capacity reading the PR replaces.
DEFECT_MARKERS = ("Unknown /",)
# The new ones, both required: the estimate without the "shared" sentence would be
# a bare number with no account of where the memory comes from.
FIXED_MARKERS = ("Estimated available:", "Shared with system RAM")

_CTX: dict = {}


def _case(state: dict, name: str, surface: str = "resources") -> dict:
    render = state.get("render") or {}
    case = ((render.get("cases") or {}).get(name) or {}).get(surface) or {}
    return case if isinstance(case, dict) else {}


def _igpu_rows(state: dict) -> list[dict]:
    rows = ((state.get("ggml_devices") or {}).get("rows")) or []
    return [r for r in rows if r.get("is_igpu")]


def _shared_devices(state: dict) -> list[dict]:
    devices = ((state.get("vulkan_gpu") or {}).get("devices")) or []
    return [d for d in devices
            if d.get("index_kind") == "vulkan" and d.get("shared_memory") is True]


def _rocm_view(state: dict) -> str:
    """The ROCm rows, as a stable string, so the two states can be compared."""
    return json.dumps(state.get("backend_gpu"), sort_keys = True, default = str)


def _quoted_estimate(text: str) -> float | None:
    """The number the estimate sentence actually shows.

    Compared against the reading with a rounding tolerance rather than rebuilt with
    a copy of the formatter: there are two different formatGiB implementations in
    the frontend (a local one in resources-tab.tsx, and lib/memory/format.ts) and a
    criteria module that hard-codes either one fails on presentation, not on the
    thing being judged.
    """
    import re
    match = re.search(r"Estimated available:\s*([0-9]+(?:\.[0-9]+)?)", text)
    return float(match.group(1)) if match else None


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    base, head = obs.get("base") or {}, obs.get("head") or {}
    _CTX["base"], _CTX["head"] = base, head
    out: list[tuple[str, bool, str]] = []

    for name in ("base", "head"):
        state = obs.get(name) or {}
        igpu = _igpu_rows(state)
        names = ", ".join(r.get("name") or f"index {r['index']}" for r in igpu) or "none"
        out.append((
            f"{name}: ggml reports an integrated device",
            bool(igpu),
            f"{len(igpu)} of {len(((state.get('ggml_devices') or {}).get('rows')) or [])} "
            f"device(s): {names}. "
            f"{(state.get('ggml_devices') or {}).get('error', '')}".strip(),
        ))
        shared = _shared_devices(state)
        out.append((
            f"{name}: the inventory carries a vulkan shared row",
            bool(shared),
            f"{len(shared)} row(s) with index_kind=vulkan and shared_memory=true; "
            f"free_gb={[d.get('vram_free_gb') for d in shared]}, "
            f"used_gb={[d.get('vram_used_gb') for d in shared]}",
        ))
        case = _case(state, SHARED_CASE)
        out.append((
            f"{name}: the Resources tab rendered",
            bool(case.get("text")),
            case.get("error") or (state.get("render") or {}).get("error")
            or f"{len(case.get('text') or '')} characters of visible text",
        ))

    same_rocm = _rocm_view(base) == _rocm_view(head)
    out.append((
        "the ROCm view of the same APU is identical at both states",
        same_rocm,
        "byte-identical" if same_rocm else
        "the ROCm inventory differs between the states, so this run cannot say the "
        "change left it alone",
    ))
    return out


def base_shows_defect(base: dict) -> tuple[bool, str]:
    case = _case(base, SHARED_CASE)
    text = case.get("text") or ""
    hit = [m for m in DEFECT_MARKERS if m in text]
    if not hit:
        return False, f"no unknown-usage reading in the base render: {text[:200]}"
    # A bar drawn for a device with no usage figure is the other half of the symptom.
    bars = [b for b in (case.get("bars") or []) if not b.startswith(("CPU=", "RAM=", "Disk="))]
    return True, (f"base renders {hit} with usage bars {bars}")


def head_is_fixed(state: dict) -> tuple[bool, str]:
    case = _case(state, SHARED_CASE)
    text = case.get("text") or ""
    missing = [m for m in FIXED_MARKERS if m not in text]
    if missing:
        return False, f"the head render is missing {missing}: {text[:200]}"
    if any(m in text for m in DEFECT_MARKERS):
        return False, f"the head render still carries an unknown-usage reading: {text[:200]}"
    bars = [b for b in (case.get("bars") or []) if not b.startswith(("CPU=", "RAM=", "Disk="))]
    if bars:
        return False, f"a GPU usage bar survives for a device with no usage figure: {bars}"
    shared = _shared_devices(state)
    if shared:
        measured = max((d.get("vram_free_gb") or 0) for d in shared)
        quoted = _quoted_estimate(text)
        if quoted is None:
            return False, f"the estimate quotes no number at all: {text[:200]}"
        # 0.51 covers the coarsest rounding either formatter applies (to the
        # nearest whole GiB above 10); anything further off is a different figure.
        if abs(quoted - measured) > 0.51:
            return False, (f"the estimate shows {quoted} GiB where the device reported "
                           f"{measured} GiB of headroom")
        return True, (f"renders {quoted} GiB against a measured {measured} GiB of "
                      f"headroom, with no usage bar")
    return True, "renders the shared estimate with no usage bar"


def table(obs: dict) -> str:
    rows = ["| surface | base | head |", "|---|---|---|"]
    for surface in ("resources", "monitor"):
        b = _case(obs.get("base") or {}, SHARED_CASE, surface).get("text") or "(not rendered)"
        h = _case(obs.get("head") or {}, SHARED_CASE, surface).get("text") or "(not rendered)"
        rows.append(f"| {surface} | {_trim(b)} | {_trim(h)} |")
    base_shared = _shared_devices(obs.get("base") or {})
    if base_shared:
        d = base_shared[0]
        rows += ["", "Measured Vulkan row: "
                 f"`index_kind={d.get('index_kind')}`, "
                 f"`shared_memory={d.get('shared_memory')}`, "
                 f"`memory_total_gb={d.get('memory_total_gb')}`, "
                 f"`vram_used_gb={d.get('vram_used_gb')}`, "
                 f"`vram_free_gb={d.get('vram_free_gb')}`"]
    return "\n".join(rows)


def _trim(text: str, limit: int = 220) -> str:
    text = text.replace("|", "\\|").replace("\n", " ")
    # The GPU part is what the table is about; the CPU/RAM/disk prefix is not.
    for marker in ("VRAM", "GPU memory", "Shared with system RAM"):
        index = text.find(marker)
        if index > 0:
            text = text[index:]
            break
    return text[:limit] + ("..." if len(text) > limit else "")
