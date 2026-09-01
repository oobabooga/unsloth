#!/usr/bin/env python3
"""Criteria: PR 8791 against two devices torch really has.

Supersedes the earlier two-device run, which fabricated three things: the amd-smi
binary, `torch.cuda.device_count()` and `get_device_properties()`. The HIP device
multiplier removes two. Here torch enumerates two devices because the C++ runtime does,
and both hold real tensors and run real kernels; the probe proves that before using
them, so a shim that silently failed makes the run INCONCLUSIVE rather than a pass.

ONE Python patch remains: the arch string of device 1. The gate compares the device arch
against `get_arch_list()`, so without it both devices report the real gfx1151, neither
is uncovered, and there is nothing for the gate to find. The amd-smi stub returns for
the selection leg only, because `hardware.py` enumerates through amd-smi rather than
torch and the shim does not reach it.

UNCHANGED LIMIT. The phantom device is device 0 wearing another number: shared memory,
shared compute, serialised. This still says nothing about a model really sharding across
two GPUs. What it now says is that the gate reaches its decision on devices torch is not
merely pretending to have.
"""

from __future__ import annotations

TITLE = "PR 8791's gate and selector on two real torch devices (HIP device multiplier)"
MODE = "differential"

NEEDS = ["rocm", "gpu", "integrated_gpu", "discrete_gpu",
         "multi_gpu", "multi_gpu_amd", "nvidia", "windows", "xpu", "mlx"]


def _g(s: dict) -> dict:
    return s.get("gate_leg") or {}


def _s(s: dict) -> dict:
    return s.get("selection_leg") or {}


def _devices_real(leg: dict) -> bool:
    """Both devices took a real tensor and ran a real kernel."""
    d = leg.get("devices_are_real") or {}
    return bool(d) and all(str(v).startswith("matmul ok") for v in d.values())


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    states = {n: v for n, v in obs.items() if not n.startswith("_")}
    out: list[tuple[str, bool, str]] = []

    ctl = {n: (v.get("control_no_shim") or {}).get("device_count")
           for n, v in states.items()}
    out.append(("unshimmed, this host really has one device",
                all(v == 1 for v in ctl.values()),
                ", ".join(f"{k}={v}" for k, v in ctl.items())))

    cnt = {n: _g(v).get("device_count") for n, v in states.items()}
    out.append(("under the multiplier torch enumerates two",
                all(v == 2 for v in cnt.values()),
                ", ".join(f"{k}={v}" for k, v in cnt.items())))

    cpp = {n: _g(v).get("cpp_device_count") for n, v in states.items()}
    out.append(("and the C++ runtime agrees, so it is not a Python patch",
                all(v == 2 for v in cpp.values()),
                ", ".join(f"{k}={v}" for k, v in cpp.items())))

    # The gate that stops this being the old mocked run wearing a better badge.
    out.append(("both devices hold real tensors and run real kernels",
                all(_devices_real(_g(v)) for v in states.values()),
                ", ".join(f"{k}={_g(v).get('devices_are_real')}" for k, v in states.items())))

    arch = {n: _g(v).get("presented_archs") for n, v in states.items()}
    out.append(("device 1 presents the uncovered arch",
                all(isinstance(v, list) and len(v) == 2 and v[0] != v[1]
                    for v in arch.values()),
                ", ".join(f"{k}={v}" for k, v in arch.items())))

    # A tie on free VRAM means the base picks the good card by luck and the
    # differential measures nothing. Voided a run once; now it is a stated gate.
    def _ranked(v):
        s_ = _s(v)
        return s_.get("util_devices") == [0, 1] and s_.get("physical_gpu_count") == 2
    out.append(("the selector saw both devices",
                all(_ranked(v) for v in states.values()),
                ", ".join(f"{k}={_s(v).get('util_devices')}" for k, v in states.items())))

    b, h = obs.get("base") or {}, obs.get("head") or {}
    out.append(("base predates the gate and head carries it",
                b.get("has_gate_source") is False and h.get("has_gate_source") is True,
                f"base={b.get('has_gate_source')} head={h.get('has_gate_source')}"))
    return out


def table(obs: dict) -> str:
    rows = ["### The devices", "",
            "| state | unshimmed count | shimmed count | C++ count | real tensors + kernels | archs |",
            "|---|---|---|---|---|---|"]
    for n, v in obs.items():
        if n.startswith("_"):
            continue
        g = _g(v)
        rows.append(f"| {n} | {(v.get('control_no_shim') or {}).get('device_count')} | "
                    f"{g.get('device_count')} | {g.get('cpp_device_count')} | "
                    f"{g.get('devices_are_real')} | `{g.get('presented_archs')}` |")

    rows += ["", "### Gate leg (no amd-smi stub)", "",
             "| state | gate present | uncovered ids |", "|---|---|---|"]
    for n, v in obs.items():
        if n.startswith("_"):
            continue
        g = _g(v)
        rows.append(f"| {n} | {g.get('gate_present')} | "
                    f"{g.get('uncovered_ids', g.get('gate_error', '-'))} |")

    rows += ["", "### Selection leg (amd-smi stubbed so hardware.py can enumerate)", "",
             "| state | physical count | util devices | uncovered ids | selected | mode |",
             "|---|---|---|---|---|---|"]
    for n, v in obs.items():
        if n.startswith("_"):
            continue
        s = _s(v)
        rows.append(f"| {n} | {s.get('physical_gpu_count')} | {s.get('util_devices')} | "
                    f"{s.get('uncovered_ids')} | {s.get('selected')} | "
                    f"{s.get('selection_mode')} |")

    rows += [
        "",
        "**What is real here that was not before.** The earlier two-device run on this PR "
        "fabricated three things: the amd-smi binary, `torch.cuda.device_count()` and "
        "`get_device_properties()`. Under the HIP device multiplier torch enumerates two "
        "devices because the C++ runtime does, and the table above shows both taking real "
        "tensors and running real kernels before the gate is asked anything.",
        "",
        "**What is still fabricated.** One Python patch: the arch string of device 1, "
        "without which both devices report the real gfx1151 and the gate has nothing to "
        "find. Plus the amd-smi stub on the selection leg only, because `hardware.py` "
        "enumerates through amd-smi rather than torch. The gate leg needs neither.",
        "",
        "**Unchanged limit.** The phantom device is device 0 wearing another number. "
        "Nothing here shows a model really sharding across two GPUs; that still needs two "
        "physical AMD cards of different architectures.",
    ]
    return "\n".join(rows)


def base_shows_defect(base: dict) -> bool:
    """Two real devices, one uncovered, and the base has no mechanism that notices."""
    g = _g(base)
    if g.get("device_count") != 2 or not _devices_real(g):
        return False
    if g.get("gate_present"):
        return False
    sel = _s(base).get("selected")
    # Either the base cannot tell at all, or it hands the uncovered device over.
    return sel is None or (isinstance(sel, list) and 1 in sel)


def head_is_fixed(head: dict) -> bool:
    """The head names device 1 uncovered and keeps the real device."""
    g = _g(head)
    if g.get("device_count") != 2 or not _devices_real(g):
        return False
    if not g.get("gate_present") or g.get("uncovered_ids") != [1]:
        return False
    sel = _s(head).get("selected")
    if sel is None:
        return False
    return 1 not in sel and 0 in sel
