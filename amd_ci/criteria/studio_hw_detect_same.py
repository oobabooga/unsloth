#!/usr/bin/env python3
"""Criteria: does this checkout's Studio hardware detection conclude the same thing
on a real gfx1151 ROCm host, base versus head, and does the head keep the GPU when
one torch probe raises?

Regression mode. Pairs with probes/studio_hw_detect_probe.py.

REGRESSION if any of:
  * on the `unmasked` or `hip0` cell, any stable field differs base vs head:
    detect_hardware() result, IS_ROCM, CHAT_ONLY, CHAT_ONLY_REASON, CHAT_ONLY_DETAIL,
    CHAT_ONLY_MISMATCH_VENDORS, whether detection raised, the "Hardware detected"
    banner, torch device count, the physical inventory,
    `_devices_that_can_establish_a_mismatch(inventory)` and the getters
    (physical / visible counts, backend-visible info, visible utilization, summary)
    with volatile readings (utilization, temperature, power, used / free memory)
    removed and remaining floats compared at 2% / 0.5 absolute tolerance, because
    a unified-memory APU derives some totals from live host memory;
  * on any injection cell, the head's detection raised, or its device / IS_ROCM /
    CHAT_ONLY / CHAT_ONLY_REASON differ from the head's own unmasked cell.

The base's injection-cell outcome is REPORTED, never required: the base may
legitimately lose the GPU (or raise) there; that is what the PR fixes.

Gates (non-vacuity): both states' probes ran every cell and imported the
checkout's module; every injection was actually applied; the BASE unmasked cell
really measured the ROCm route (device cuda, IS_ROCM true, torch.version.hip set,
arch gfx1151), so an identical-because-both-fell-back-to-CPU comparison cannot
pass; when the device multiplier is active, torch really reported the extra
devices.
"""

from __future__ import annotations

import os
import re

TITLE = "Studio hardware detection on real gfx1151, base versus head (PR 11944)"
MODE = "regression"
# Everything the change touches: guarded XPU probes, the CUDA / ROCm branch, Intel
# discrete detection by PCI bus, the Linux Intel pin hint, Windows and WSL
# nvidia-smi discovery, device-count bookkeeping.
NEEDS = ["linux", "rocm", "gpu", "xpu", "nvidia", "windows", "discrete_gpu", "multi_gpu"]

EXPECT_ARCH = "gfx1151"
COMPARED_CELLS = ("unmasked", "hip0")
INJECTION_CELLS = ("inject_xpu_is_available_raises",
                   "inject_cuda_get_device_properties_raises",
                   "inject_cuda_is_available_raises_after_first")
DETECT_KEYS = ("device", "IS_ROCM", "CHAT_ONLY", "CHAT_ONLY_REASON", "CHAT_ONLY_DETAIL",
               "CHAT_ONLY_MISMATCH_VENDORS")
VERDICT_KEYS = ("device", "IS_ROCM", "CHAT_ONLY", "CHAT_ONLY_REASON")
VOLATILE = re.compile(r"(util|temp|used|free|power|busy|clock|fan|seconds|percent|_pct$)",
                      re.IGNORECASE)


# ---------------------------------------------------------------- helpers

def cells(o: dict) -> dict:
    return (o or {}).get("cells") or {}


def cell(o: dict, name: str) -> dict:
    return cells(o).get(name) or {}


def det(c: dict) -> dict:
    return c.get("detect") or {}


def stable(v):
    if isinstance(v, dict):
        return {k: stable(x) for k, x in v.items() if not VOLATILE.search(str(k))}
    if isinstance(v, list):
        return [stable(x) for x in v]
    return v


def fields(c: dict) -> dict:
    """The comparable projection of one cell."""
    d = det(c)
    out = {f"detect {k}": d.get(k) for k in DETECT_KEYS}
    out["detect raised"] = bool(d.get("raised"))
    out["detect banner"] = d.get("banner")
    out["torch device_count"] = (c.get("raw_torch") or {}).get("device_count")
    out["inventory"] = stable(c.get("inventory", c.get("inventory_error")))
    out["mismatch devices"] = stable(c.get("mismatch_devices", c.get("mismatch_devices_error")))
    for name, g in sorted((c.get("getters") or {}).items()):
        out[name] = stable(g)
    return out


def close(a, b) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if isinstance(a, int) and isinstance(b, int):
            return a == b
        return abs(a - b) <= max(0.5, 0.02 * max(abs(a), abs(b)))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(close(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(close(x, y) for x, y in zip(a, b))
    return a == b


def _short(v, n: int = 140) -> str:
    return str(v).replace("|", "\\|").replace("\n", " ")[:n]


def _verdict_of(c: dict) -> str:
    d = det(c)
    if d.get("raised"):
        return f"RAISED {d['raised']}"
    return ", ".join(f"{k}={d.get(k)}" for k in VERDICT_KEYS)


# ---------------------------------------------------------------- gates

def probe_gates(obs: dict) -> list[tuple[str, bool, str]]:
    out = []
    for name in ("base", "head"):
        o = obs.get(name) or {}
        ok = (o.get("_probe_rc") == 0 and not o.get("_missing_output")
              and not o.get("_parse_error") and bool(cells(o)))
        out.append((f"{name} probe wrote observations", ok,
                    f"rc={o.get('_probe_rc')}, cells={list(cells(o))}"))
        bad = []
        for cn in COMPARED_CELLS + INJECTION_CELLS:
            c = cell(o, cn)
            if (not c or c.get("_rc") != 0 or c.get("_missing_output") or c.get("_parse_error")
                    or c.get("import_error") or "detect" not in c):
                bad.append(f"{cn}: rc={c.get('_rc')} "
                           f"{c.get('import_error') or c.get('_parse_error') or ''}"
                           f"{' missing output' if c.get('_missing_output') else ''}"
                           f"{' timeout' if c.get('_timeout') else ''}".strip())
        out.append((f"{name} every cell imported the checkout and ran detect_hardware()",
                    not bad, "; ".join(bad) or "ok"))
        not_applied = [cn for cn in INJECTION_CELLS
                       if not (cell(o, cn).get("injection") or {}).get("applied")]
        out.append((f"{name} every injection was applied", not not_applied,
                    ", ".join(f"{cn}: {(cell(o, cn).get('injection') or {}).get('why_not')}"
                              for cn in not_applied) or "ok"))
    # Spoofed devices: the multiplier must really have reached torch, or the spoof
    # job measured the one-device host twice.
    spoof = (cell(obs.get("base") or {}, "unmasked").get("env") or {}).get("AMD_CI_SPOOFED_DEVICES")
    if spoof or os.environ.get("AMD_CI_SPOOFED_DEVICES"):
        for name in ("base", "head"):
            n = (cell(obs.get(name) or {}, "unmasked").get("raw_torch") or {}).get("device_count")
            out.append((f"{name} torch saw the spoofed devices (AMD_CI_SPOOFED_DEVICES={spoof})",
                        isinstance(n, int) and n >= 2, f"device_count={n}"))
    return out


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out = probe_gates(obs)
    b = cell(obs.get("base") or {}, "unmasked")
    d, rt = det(b), (b.get("raw_torch") or {})
    archs = [x.get("arch") for x in rt.get("devices") or []]
    out.append(("base unmasked measured the ROCm route (cuda, IS_ROCM, hip, gfx1151)",
                d.get("device") == "cuda" and d.get("IS_ROCM") is True and bool(rt.get("hip"))
                and any(str(a or "").startswith(EXPECT_ARCH) for a in archs),
                f"device={d.get('device')} IS_ROCM={d.get('IS_ROCM')} hip={rt.get('hip')} "
                f"torch={rt.get('version')} archs={archs}"))
    return out


# ---------------------------------------------------------------- judgement

def table(obs: dict) -> str:
    b, h = obs.get("base") or {}, obs.get("head") or {}
    rows = ["| cell | field | base | head | same |", "|---|---|---|---|---|"]
    for cn in COMPARED_CELLS:
        fb, fh = fields(cell(b, cn)), fields(cell(h, cn))
        for k in fb:
            same = close(fb[k], fh.get(k))
            rows.append(f"| {cn} | {k} | `{_short(fb[k])}` | `{_short(fh.get(k))}` | "
                        f"{'yes' if same else 'NO'} |")
    rows += ["", "| injection cell | base verdict | head verdict | head matches head unmasked |",
             "|---|---|---|---|"]
    hu = det(cell(h, "unmasked"))
    for cn in INJECTION_CELLS:
        hc = det(cell(h, cn))
        keep = not hc.get("raised") and all(hc.get(k) == hu.get(k) for k in VERDICT_KEYS)
        rows.append(f"| {cn} | `{_short(_verdict_of(cell(b, cn)))}` | "
                    f"`{_short(_verdict_of(cell(h, cn)))}` | {'yes' if keep else 'NO'} |")
    return "\n".join(rows)


def head_is_worse(base: dict, head: dict) -> tuple[bool, str]:
    problems = []
    for cn in COMPARED_CELLS:
        fb, fh = fields(cell(base, cn)), fields(cell(head, cn))
        moved = [k for k in fb if not close(fb[k], fh.get(k))]
        if moved:
            problems.append(f"{cn}: " + ", ".join(f"`{k}`" for k in moved))
    hu = det(cell(head, "unmasked"))
    lost = []
    for cn in INJECTION_CELLS:
        hc = det(cell(head, cn))
        if hc.get("raised") or any(hc.get(k) != hu.get(k) for k in VERDICT_KEYS):
            lost.append(f"{cn} ({_verdict_of(cell(head, cn))})")
    if lost:
        problems.append("head lost its own unmasked verdict on: " + "; ".join(lost))
    if problems:
        return True, "; ".join(problems)
    base_inj = "; ".join(f"{cn}: {_verdict_of(cell(base, cn))}" for cn in INJECTION_CELLS)
    return False, (f"all stable fields identical base vs head on {', '.join(COMPARED_CELLS)} "
                   f"({_verdict_of(cell(head, 'unmasked'))}); the head keeps that verdict on "
                   f"all {len(INJECTION_CELLS)} injection cells. Base on the injection cells, "
                   f"reported only: {base_inj}")
