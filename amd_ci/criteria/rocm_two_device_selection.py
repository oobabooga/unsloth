#!/usr/bin/env python3
"""Criteria: does PR 8791 drop an uncovered device when there are two to choose from?

This is the half the wheel-swap run could not reach. There, one device was genuinely
uncovered, so the fail-open rule kept it and `selected` was `[0]` in both states: real
detection, but no selection to observe. Here there are two devices, so the gate has a
choice to make, and the base/head difference is visible in `selected` itself.

The cost is that the second device is fabricated. Read the honesty gates below before
the verdict table, because they are what separates this from a unit test in a costume.

WHAT IS REAL. `get_physical_gpu_count`, `_get_parent_visible_gpu_spec`,
`get_visible_gpu_utilization`, `_gpu_entries`, the VRAM ranking, the gate, all four
selector exits, and the metadata -- every line executed is the shipped code, running on
a real ROCm host against a real GPU.

WHAT IS FAKE, EXACTLY TWO THINGS.
  1. The `amd-smi` binary, stubbed on PATH. The stub SHELLS OUT TO THE REAL amd-smi and
     duplicates its GPU entry, so the JSON is the real schema of the real tool. This is
     the same boundary already accepted for the install.sh probes.
  2. `torch.cuda.device_count()` -> 2 and `get_device_properties(1)` -> gfx1036.

WHAT THIS THEREFORE CANNOT SAY. Device 1 is a SELECTION INPUT, not a compute device.
torch's C++ runtime is bound to one HIP device, so no tensor can be placed on it. This
run says nothing about kernel dispatch, nothing about `device_map="balanced"` really
sharding, and nothing about the `hipErrorInvalidKernelFile` crash itself -- that half
was established separately by the uncovered-wheel run, on real silicon.

So the claim is narrow and should be quoted narrowly: **the wiring is right**. That is
worth measuring precisely because the earlier VOID showed the wiring is where the risk
lives -- the first attempt mocked `device_count` alone and the selector never reached
the gate at all, because enumeration goes through amd-smi and not through torch.
"""

from __future__ import annotations

TITLE = "Two-device selection with the uncovered device fabricated (PR 8791, wiring only)"
MODE = "differential"

NEEDS = ["rocm", "gpu", "integrated_gpu", "discrete_gpu",
         "multi_gpu", "multi_gpu_amd", "nvidia", "windows", "xpu", "mlx"]


def _t(state: dict) -> dict:
    return state.get("two_device") or {}


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out: list[tuple[str, bool, str]] = []
    states = {n: v for n, v in obs.items() if not n.startswith("_")}

    real = {n: v.get("real_amd_smi") for n, v in states.items()}
    out.append(("a real amd-smi was found for the stub to wrap",
                all(bool(v) for v in real.values()),
                ", ".join(f"{k}={v}" for k, v in real.items())))

    # If this is not 1, the host changed under the run and the whole premise -- that a
    # second device is fabricated on a one-GPU box -- is not what was measured.
    rc = {n: _t(v).get("real_device_count") for n, v in states.items()}
    out.append(("the host really has one torch device",
                all(v == 1 for v in rc.values()),
                ", ".join(f"{k}={v}" for k, v in rc.items())))

    out.append(("the run is on a ROCm host",
                all(_t(v).get("is_rocm") is True for v in states.values()),
                ", ".join(f"{k}={_t(v).get('is_rocm')}" for k, v in states.items())))

    # The defect is a free-VRAM-only ranking, so device 1 has to actually WIN that
    # ranking. If it does not, the base picks the good card by tie-break and the
    # differential is measuring luck.
    def _outranks(v):
        rows = _t(v).get("util_vram") or []
        by = {r.get("index"): r.get("free_gb") for r in rows}
        f0, f1 = by.get(0), by.get(1)
        return isinstance(f0, (int, float)) and isinstance(f1, (int, float)) and f1 > f0
    out.append(("device 1 reports more free VRAM than device 0, so it wins the ranking",
                all(_outranks(v) for v in states.values()),
                ", ".join(f"{k}={_t(v).get('util_vram')}" for k, v in states.items())))

    # The whole point: the stub has to carry the count through the REAL spec function.
    # If this is 1, amd-smi was not stubbed successfully and nothing below is two-device.
    pc = {n: _t(v).get("physical_gpu_count") for n, v in states.items()}
    out.append(("the stub carried a count of 2 through get_physical_gpu_count()",
                all(v == 2 for v in pc.values()),
                ", ".join(f"{k}={v}" for k, v in pc.items())))

    nid = {n: _t(v).get("numeric_ids") for n, v in states.items()}
    out.append(("the real spec function yielded two ids, with no mask set",
                all(v == [0, 1] for v in nid.values()),
                ", ".join(f"{k}={v}" for k, v in nid.items())))

    # And the enumeration has to survive all the way to the device list, or the
    # selector is ranking one device regardless of what the count said.
    ud = {n: _t(v).get("util_devices") for n, v in states.items()}
    out.append(("both devices reached the utilization list",
                all(v == [0, 1] for v in ud.values()),
                ", ".join(f"{k}={v}" for k, v in ud.items())))

    pa = {n: _t(v).get("presented_archs") for n, v in states.items()}
    out.append(("device 0 is the real arch and device 1 is the uncovered one",
                all(isinstance(v, list) and len(v) == 2 and v[0] != v[1]
                    for v in pa.values()),
                ", ".join(f"{k}={v}" for k, v in pa.items())))

    b, h = obs.get("base") or {}, obs.get("head") or {}
    out.append(("base predates the gate and head carries it",
                b.get("has_gate_source") is False and h.get("has_gate_source") is True,
                f"base={b.get('has_gate_source')} head={h.get('has_gate_source')}"))
    return out


def table(obs: dict) -> str:
    rows = ["| state | physical count | numeric ids | util devices | archs presented | "
            "free VRAM |", "|---|---|---|---|---|---|"]
    for name, v in obs.items():
        if name.startswith("_"):
            continue
        t = _t(v)
        rows.append(f"| {name} | {t.get('physical_gpu_count')} | {t.get('numeric_ids')} | "
                    f"{t.get('util_devices')} | `{t.get('presented_archs')}` | "
                    f"{t.get('util_vram')} |")

    rows += ["", "| state | gate present | uncovered ids | selected | selection mode |",
             "|---|---|---|---|---|"]
    for name, v in obs.items():
        if name.startswith("_"):
            continue
        t = _t(v)
        rows.append(f"| {name} | {t.get('gate_present')} | {t.get('uncovered_ids')} | "
                    f"{t.get('selected')} | {t.get('selection_mode')} |")

    errs = []
    for name, v in obs.items():
        if name.startswith("_"):
            continue
        t = _t(v)
        for key in ("error", "hardware_error", "spec_error", "util_error",
                    "gate_error", "select_error", "physical_count_error"):
            if t.get(key):
                errs.append(f"| {name} | {key} | {str(t[key])[:160]} |")
    if errs:
        rows += ["", "| state | where | error |", "|---|---|---|"] + errs

    rows += [
        "",
        "**Read the top table before the verdict.** Device 1 does not exist. The count "
        "reaches the selector through the real `get_physical_gpu_count` -> `amd-smi` "
        "path, with the amd-smi binary stubbed to duplicate its own real output; only "
        "that binary and `torch.cuda`'s device metadata are fabricated. Every function "
        "between them is the shipped code on a real ROCm host.",
        "",
        "**This measures wiring, not compute.** torch is bound to one HIP device, so no "
        "tensor can live on device 1. Nothing here speaks to kernel dispatch or to "
        "`device_map=\"balanced\"` actually sharding. The crash itself was reproduced "
        "separately, unspoofed, by the uncovered-wheel run.",
        "",
        "Device 1 is shaped like the iGPU in issue 8792: twice the total VRAM and "
        "almost none of it used, because an APU reports a large slice of shared system "
        "memory. That is what makes it WIN a free-VRAM-only ranking. A first run cloned "
        "device 0 exactly, the two tied, and the base picked the good card by tie-break "
        "-- which looked like a clean base and was really the defect being hidden.",
        "",
        "It is still worth having: the previous attempt mocked `torch.cuda.device_count` "
        "alone and returned VOID, because enumeration runs through amd-smi rather than "
        "torch and the selector never reached the gate. Wiring is exactly where that "
        "failure was.",
    ]
    return "\n".join(rows)


def base_shows_defect(base: dict) -> bool:
    """Two devices, one uncovered, and the base hands both over anyway."""
    t = _t(base)
    if t.get("physical_gpu_count") != 2 or t.get("util_devices") != [0, 1]:
        return False
    if t.get("gate_present"):
        return False
    sel = t.get("selected")
    return isinstance(sel, list) and 1 in sel


def head_is_fixed(head: dict) -> bool:
    """The head identifies device 1 as uncovered and leaves it out of the selection.

    Both halves are required. Naming the device without acting on it is the old
    behaviour with a log line, and dropping it without the gate present would mean
    something else moved the answer.
    """
    t = _t(head)
    if t.get("physical_gpu_count") != 2 or t.get("util_devices") != [0, 1]:
        return False
    if not t.get("gate_present"):
        return False
    if t.get("uncovered_ids") != [1]:
        return False
    sel = t.get("selected")
    # Device 0 is real and covered, so it must survive: a fix that strands the host
    # on CPU is not a fix.
    return isinstance(sel, list) and 1 not in sel and 0 in sel
