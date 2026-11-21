#!/usr/bin/env python3
"""Criteria: unsloth-zoo PR 2644 on gfx1151 (Strix Halo, unified memory), base vs head. Pairs with
probes/pr2644_probe.py. Differential mode.
Defect at the base: smart gradient checkpointing copies every checkpointed layer input from the device into host
RAM (reentrant leases > 0, non-reentrant pack moves the tensor to the CPU), which on this APU is the same DRAM.
Fixed at the head: the detector selects device 0 (keep_on_device_indices == [0]), no reentrant lease is taken, the
non-reentrant pack keeps the tensor, and the gradients are bit-identical to the base's.
Gates: HIP venv on gfx1151, each state imported its own checkout, every cell ran on the GPU."""
from __future__ import annotations

TITLE = "unsloth-zoo PR 2644: GC keeps activations on device on gfx1151, base vs head"
MODE = "differential"
NEEDS = ["rocm", "gpu"]


def _cells(o):
    return [c for c in (o or {}).get("cells") or []]


def gates(obs):
    out = []
    for name in ("base", "head"):
        o = obs.get(name) or {}
        v = o.get("venv") or {}
        env = v.get("env") or {}
        out.append((f"{name} venv on HIP gfx1151", not v.get("error") and bool(env.get("hip"))
                    and str(env.get("arch") or "").startswith("gfx1151"), f"{v.get('error') or ''} {env}"))
        cells = _cells(o)
        ck = str(o.get("checkout") or "<none>")
        out.append((f"{name} cells imported their own checkout",
                    len(cells) == 2 and all(str(c.get("gc_file") or "").startswith(ck) for c in cells),
                    str([c.get("gc_file") or c.get("log_tail", "")[-300:] for c in cells])))
        out.append((f"{name} cells ran on the GPU",
                    len(cells) == 2 and all(c.get("cuda_available") and c.get("reentrant_leases_per_step") for c in cells),
                    str([(c.get("rc"), (c.get("props") or {}).get("gcnArchName")) for c in cells])))
    return out


def table(obs):
    rows = ["| state | cell | arch | integrated | pool | keep | leases/step | nonreentrant kept | step s | peak MiB | grad |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name in ("base", "head", "merge"):
        for i, c in enumerate(_cells(obs.get(name))):
            p = c.get("props") or {}
            rows.append(f"| {name} | {i} | {p.get('gcnArchName')} | {p.get('is_integrated')} | "
                        f"{(c.get('memory_pool') or {}).get('pool')} | {c.get('keep_on_device_indices')} | "
                        f"{c.get('reentrant_leases_per_step')} | {c.get('nonreentrant_pack_device_is_none')} | "
                        f"{[round(t, 4) for t in c.get('reentrant_step_s') or []]} | "
                        f"{round((c.get('peak_allocated') or 0) / 2**20)} | {str(c.get('grad_sha256'))[:12]} |")
    for name in ("base", "head"):
        pt = (obs.get(name) or {}).get("pytest")
        if pt:
            rows.append(f"\n{name} pytest {TEST_NAME}: passed {pt.get('n_passed')} failed {pt.get('n_failed')} "
                        f"skipped {pt.get('n_skipped')} failing {pt.get('failed')}")
    return "\n".join(rows)


TEST_NAME = "tests/test_gc_unified_memory_keep_on_device.py"


def base_shows_defect(base):
    cells = _cells(base)
    offl = all(sum(c.get("reentrant_leases_per_step") or [0]) > 0 for c in cells) and \
        all(c.get("nonreentrant_pack_device_is_none") is False for c in cells)
    return bool(cells) and offl, f"base leases {[c.get('reentrant_leases_per_step') for c in cells]}"


def head_is_fixed(head):
    cells = _cells(head)
    ok = bool(cells) and all(c.get("keep_on_device_indices") == [0] and sum(c.get("reentrant_leases_per_step") or [1]) == 0
                             and c.get("nonreentrant_pack_device_is_none") is True for c in cells)
    pt = head.get("pytest") or {}
    ok = ok and pt.get("n_failed", 1) == 0 and (pt.get("n_passed") or 0) > 0
    return ok, (f"head keep {[c.get('keep_on_device_indices') for c in cells]} leases "
                f"{[c.get('reentrant_leases_per_step') for c in cells]} pytest passed {pt.get('n_passed')} failed {pt.get('n_failed')}")
