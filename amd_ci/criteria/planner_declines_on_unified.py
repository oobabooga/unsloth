#!/usr/bin/env python3
"""Criteria: does the planner keep its hands off a unified-memory APU?

On an integrated GPU the "VRAM" the planner would spill out of IS system RAM, so
an -ot spill moves bytes between two names for one pool: it frees no device
memory and buys only the CPU backend's slower read path. The planner has an
explicit abstain for that, and this asks whether the abstain actually fires on
real gfx1151 silicon rather than on a stubbed HostProfile.

Regression mode, not differential, because the defect shape here is not "the
base is broken". Before this change there is no planner at all, so the base
cannot spill. The question is whether the HEAD starts spilling on a host where
nothing was spilled before, which is exactly what head_is_worse asks.

The non-vacuity gate is the discrete reading. A layout that fits anyway would
"abstain" on any host and prove nothing, so the same layout and budget with
unified_memory forced False has to produce a real spill before the unified
reading's silence counts as evidence.

Pairs with probes/planner_unified_probe.py.
"""

from __future__ import annotations

TITLE = "Offload planner on a unified-memory APU (gfx1151)"
MODE = "regression"
# Declared for the change, not for the host: the planner reasons about device
# placement generally, so every platform it could be wrong on belongs here.
NEEDS = ["gpu", "rocm", "windows_rocm_wddm", "multi_gpu", "nvidia", "mig", "xpu", "mlx"]


def _head(obs: dict) -> dict:
    return obs.get("head") or {}


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    head = _head(obs)
    out: list[tuple[str, bool, str]] = []

    arch = head.get("arch")
    out.append(("the host really is an integrated gfx1151",
                bool(arch and "gfx1151" in str(arch)) and bool(head.get("is_integrated")),
                f"arch={arch} integrated={head.get('is_integrated')}"))

    out.append(("the head exposes the unified-memory detector",
                bool(head.get("has_detector")),
                f"has_detector={head.get('has_detector')} "
                f"error={head.get('detector_error', '-')}"))

    out.append(("the head exposes the planner",
                bool(head.get("has_planner")),
                f"has_planner={head.get('has_planner')} "
                f"error={head.get('planner_error', '-')}"))

    # Non-vacuity: the same layout and budget MUST spill when the flag is off,
    # or the unified reading is silence about nothing.
    disc = head.get("discrete") or {}
    spilled = int(disc.get("n_spilled_blocks") or 0)
    out.append(("the same layout really does spill on a discrete host",
                spilled > 0,
                f"discrete spilled {spilled} blocks"))
    return out


def table(obs: dict) -> str:
    rows = ["| state | detector | unified ids | flag on: spilled | flag off: spilled |",
            "|---|---|---|---|---|"]
    for name, v in obs.items():
        if name.startswith("_"):
            continue
        if not v.get("has_planner"):
            rows.append(f"| {name} | {v.get('detector_says_unified', '-')} | "
                        f"{v.get('unified_gpu_ids', '-')} | no planner | no planner |")
            continue
        u = v.get("unified") or {}
        d = v.get("discrete") or {}
        rows.append(f"| {name} | {v.get('detector_says_unified')} | "
                    f"{v.get('unified_gpu_ids', '-')} | "
                    f"{u.get('n_spilled_blocks')} | {d.get('n_spilled_blocks')} |")
    head = _head(obs)
    reason = ((head.get("unified") or {}).get("reason") or "").strip()
    if reason:
        rows += ["", f"Head's stated reason under the unified flag: {reason}"]
    return "\n".join(rows)


def head_is_worse(base: dict, head: dict) -> tuple[bool, str]:
    if not head.get("detector_says_unified"):
        # Not a pass and not a failure of the change: the host did not present
        # as unified, so the abstain was never asked the question.
        return True, ("the detector did NOT report this host as unified memory "
                      f"(ids={head.get('unified_gpu_ids')}), so the abstain was never "
                      "exercised; treat this as no result rather than a pass")

    u = head.get("unified") or {}
    spilled = int(u.get("n_spilled_blocks") or 0)
    if spilled or u.get("spilled_lm_head") or u.get("changed"):
        return True, (f"the planner emitted a plan on a unified-memory host: "
                      f"{spilled} blocks, lm_head={u.get('spilled_lm_head')}, "
                      f"changed={u.get('changed')}")

    d = head.get("discrete") or {}
    return False, ("the planner declined on the real gfx1151 unified pool while spilling "
                   f"{d.get('n_spilled_blocks')} blocks for the identical layout and budget "
                   "with the flag off, so the abstain is attributable to the detected "
                   "hardware and not to a layout that fitted anyway")
