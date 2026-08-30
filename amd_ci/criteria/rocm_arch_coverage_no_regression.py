#!/usr/bin/env python3
"""Criteria: does PR 8791 leave a covered single-GPU ROCm host exactly as it was?

The defect in issue 8792 needs two AMD GPUs of different archs on one board. This
runner has one gfx1151 APU, so the defect is UNREACHABLE and this is deliberately a
regression check, not a differential one. Claiming CONFIRMED here would be claiming
something the hardware cannot support.

What a regression looks like, and each is a real way this PR could hurt a working host:

  - the gate drops the only GPU, sending a working machine to CPU;
  - `apply_gpu_ids` writes a different visibility mask than it used to, under any of
    the masks the probe drives;
  - torch stops seeing the device under a mask it used to survive.

The table also carries the measurement this run exists for: the shipped wheel's real
`torch.cuda.get_arch_list()`. The gate rejects that list whole if any token is
non-concrete, so if this wheel ships a generic code object the gate is inert on modern
ROCm and the PR does not fix the host it was written for. That is a finding about
REACH, not a regression, so it is reported in the table and not scored.
"""

from __future__ import annotations

import re

TITLE = "GPU selection on a covered single-GPU ROCm host (PR 8791)"
MODE = "regression"

# The change reaches every ROCm host and rewrites visibility handling for all of them,
# so the shapes it could hurt are declared even though this host is none of them.
NEEDS = [
    "rocm", "gpu", "integrated_gpu", "discrete_gpu",
    "multi_gpu", "multi_gpu_amd", "nvidia", "mig", "gpu_partitions",
    "windows", "xpu",
]

# Same rule the PR uses, restated here so the criteria can say whether the wheel's
# tokens would have been accepted, without importing the checkout.
_CONCRETE = re.compile(r"^gfx[0-9][0-9a-f]{2,4}$")

_CTX: dict = {"note": ""}


def _masks(state: dict) -> dict:
    return state.get("masks") or {}


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out: list[tuple[str, bool, str]] = []
    states = {n: v for n, v in obs.items() if not n.startswith("_")}

    readable = []
    for name, st in states.items():
        for label, r in _masks(st).items():
            if r.get("hardware_file") is None:
                readable.append(f"{name}/{label}: "
                                f"{r.get('hardware_error') or r.get('child_error') or 'no reading'}")
    out.append(("every mask produced a hardware reading",
                not readable, "; ".join(readable) or "all readings present"))

    hip = {n: (_masks(v).get("none") or {}).get("torch_hip") for n, v in states.items()}
    out.append(("torch is a ROCm build", all(bool(v) for v in hip.values()),
                ", ".join(f"{k}=hip {v}" for k, v in hip.items())))

    seen = {n: (_masks(v).get("none") or {}).get("device_count") for n, v in states.items()}
    out.append(("a GPU was visible unmasked", all(bool(v) for v in seen.values()),
                ", ".join(f"{k}={v}" for k, v in seen.items())))

    b, h = obs.get("base") or {}, obs.get("head") or {}
    shape = (b.get("has_gate_source") is False and h.get("has_gate_source") is True)
    out.append(("base predates the arch gate and head carries it", shape,
                f"base={b.get('has_gate_source')} head={h.get('has_gate_source')}"))
    return out


def table(obs: dict) -> str:
    rows = ["| state | mask | devices | uncovered ids | CUDA_VISIBLE | HIP_VISIBLE | ROCR_VISIBLE |",
            "|---|---|---|---|---|---|---|"]
    for name, v in obs.items():
        if name.startswith("_"):
            continue
        for label, r in _masks(v).items():
            ap = r.get("applied") or {}
            unc = r.get("uncovered_ids")
            unc = "n/a (no gate)" if r.get("gate_present") is False else unc
            rows.append(
                f"| {name} | {label} | {r.get('device_count')} | {unc} | "
                f"{ap.get('CUDA_VISIBLE_DEVICES')} | {ap.get('HIP_VISIBLE_DEVICES')} | "
                f"{ap.get('ROCR_VISIBLE_DEVICES')} |"
            )

    head = obs.get("head") or {}
    arch_list = ((_masks(head).get("none") or {}).get("arch_list")) or []
    archs = ((_masks(head).get("none") or {}).get("device_archs")) or []
    non_concrete = [t for t in (str(a).split(":")[0].strip().lower() for a in arch_list)
                    if t and not _CONCRETE.match(t)]

    rows += ["", "### What the shipped ROCm wheel reports", "",
             f"`torch.cuda.get_arch_list()` = `{arch_list}`",
             f"device `gcnArchName` = `{archs}`", ""]
    if not arch_list:
        rows.append("The arch list was unreadable, so this run says nothing about the "
                    "gate's reach.")
    elif non_concrete:
        rows.append(
            f"**The gate is inert on this wheel.** Non-concrete tokens {non_concrete} "
            "are present, and the PR rejects the arch list whole when any token is "
            "non-concrete, so `rocm_gpu_ids_without_torch_kernels()` returns the empty "
            "set no matter what is plugged in. On a wheel shaped like this the PR "
            "cannot fix issue 8792. This is a statement about reach, not a regression, "
            "and it is not scored below."
        )
    else:
        rows.append(
            "Every token is concrete, so the gate would evaluate rather than fail open "
            "on this wheel class. That is the precondition for the PR fixing issue "
            "8792; the fix itself still needs two AMD GPUs to observe."
        )
    rows += ["", "This host is one covered gfx1151 APU. The two-GPU shape the PR exists "
             "for was not reachable, so nothing below should be read as confirming the fix."]
    if _CTX.get("note"):
        rows += ["", _CTX["note"]]
    return "\n".join(rows)


def head_is_worse(base: dict, head: dict) -> tuple[bool, str]:
    problems: list[str] = []

    bm, hm = _masks(base), _masks(head)
    for label in sorted(set(bm) | set(hm)):
        b, h = bm.get(label) or {}, hm.get(label) or {}

        # A working host must not lose its GPU to the new gate.
        unc = h.get("uncovered_ids")
        if unc:
            problems.append(
                f"{label}: the gate excluded {unc} on a host whose wheel covers it "
                f"(arch {h.get('device_archs')}, list {h.get('arch_list')})"
            )

        # Visibility must come out the same. This is where the apply_gpu_ids rewrite
        # would show up on a host that never needed translating.
        ba, ha = b.get("applied") or {}, h.get("applied") or {}
        if ba and ha and ba != ha:
            problems.append(f"{label}: apply_gpu_ids wrote {ha}, base wrote {ba}")

        # And the device must still be there afterwards.
        if b.get("device_count") and not h.get("device_count"):
            problems.append(f"{label}: torch saw {b.get('device_count')} device(s) at "
                            f"base and {h.get('device_count')} at head")

        if b.get("parent_visible_ids") is not None and \
           h.get("parent_visible_ids") is not None and \
           b["parent_visible_ids"] != h["parent_visible_ids"]:
            problems.append(f"{label}: visible ids {h['parent_visible_ids']} vs base "
                            f"{b['parent_visible_ids']}")

    if problems:
        return True, "; ".join(problems)
    return False, ("a covered single-GPU ROCm host selects and masks identically at "
                   "base and head across every mask driven")
