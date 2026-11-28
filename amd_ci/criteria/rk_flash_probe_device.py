#!/usr/bin/env python3
"""Criteria: Studio's ROCm flash-attn probe on a non-current card (unsloth #16767 follow-up; judges
probes/rk_flash_probe_device_probe.py under `scaffold.py --spoof-devices 1`; base = origin/main, head = base + fix).

Wiring only: cuda:1 is the real gfx1151 under another ordinal, so this shows which device the probe launches on, not
how a second physical card behaves; multi_gpu stays a declared gap.

base_shows_defect: flash-attn runs on cuda:1 inside torch.cuda.device(1), refuses it from current device 0 with its
device-mismatch assertion, and the base's probe calls cuda:1 unusable while calling cuda:0 usable.
head_is_fixed: the probe calls both cuda:0 and cuda:1 usable and leaves the current device at 0.
"""

from __future__ import annotations

TITLE = "gfx1151 Linux (spoofed 2nd ordinal): ROCm flash-attn probe enters the probed card"
MODE = "differential"
NEEDS = ["gpu", "rocm", "linux", "multi_gpu", "multi_gpu_amd", "exclusive_gpu", "windows", "discrete_gpu"]


def _one(x, n):
    return str(x or "").replace("\n", " ").replace("|", "/")[:n]


def _p(o, dev):
    return ((o or {}).get("probe") or {}).get(dev) or {}


def gates(obs):
    b, h = obs.get("base") or {}, obs.get("head") or {}
    out = []
    for name, o in (("base", b), ("head", h)):
        out.append((f"{name}: probe ran", not o.get("error") and o.get("rc") == 0,
                    f"rc {o.get('rc')}, {_one(o.get('error'), 200)}"))
        out.append((f"{name}: two ordinals (spoofed), gfx1151, flash_attn imported",
                    o.get("device_count") == 2 and bool(o.get("spoofed")) and bool(o.get("hip"))
                    and all(str(a).startswith("gfx1151") for a in (o.get("archs") or [None]))
                    and bool((o.get("flash_attn") or {}).get("file")),
                    f"count {o.get('device_count')}, spoofed {o.get('spoofed')}, archs {o.get('archs')}, "
                    f"flash_attn {(o.get('flash_attn') or {}).get('version')}"))
        out.append((f"{name}: flash-attn works on cuda:1 when it is current",
                    ((o.get("raw") or {}).get("inside_device_1") or {}).get("ok") is True,
                    _one((o.get("raw") or {}).get("inside_device_1"), 200)))
        out.append((f"{name}: probe module from this checkout", bool(o.get("module_from_checkout")), ""))
    out.append(("head carries the fix, base does not",
                bool(h.get("has_device_scope")) and not b.get("has_device_scope"), ""))
    return out


def base_shows_defect(base):
    raw = (base.get("raw") or {}).get("from_current_0") or {}
    mismatch = raw.get("ok") is False and ("cuda:1" in str(raw.get("error")) or "evice" in str(raw.get("error")))
    p0, p1 = _p(base, "cuda:0"), _p(base, "cuda:1")
    return (mismatch and p0.get("verdict") is True and p1.get("verdict") is False,
            f"raw from current 0: {_one(raw.get('error') or 'ok', 250)}; probe cuda:0 {p0.get('verdict')}, "
            f"cuda:1 {p1.get('verdict')}")


def head_is_fixed(head, base = None):
    p0, p1 = _p(head, "cuda:0"), _p(head, "cuda:1")
    ok = p0.get("verdict") is True and p1.get("verdict") is True and p1.get("current_after") == 0
    return ok, f"probe cuda:0 {p0.get('verdict')}, cuda:1 {p1.get('verdict')}, current after {p1.get('current_after')}"


def table(obs):
    lines = ["| state | raw cuda:1 from current 0 | raw inside device(1) | probe cuda:0 | probe cuda:1 | current after |",
             "|---|---|---|---|---|---|"]
    for st in ("base", "head"):
        o = obs.get(st) or {}
        raw = o.get("raw") or {}
        cell = lambda r: "ok" if (r or {}).get("ok") else _one((r or {}).get("error"), 140)
        lines.append(f"| {st} | {cell(raw.get('from_current_0'))} | {cell(raw.get('inside_device_1'))} | "
                     f"{_p(o, 'cuda:0').get('verdict', _p(o, 'cuda:0').get('error'))} | "
                     f"{_p(o, 'cuda:1').get('verdict', _p(o, 'cuda:1').get('error'))} | "
                     f"{_p(o, 'cuda:1').get('current_after')} |")
    return "\n".join(lines)
