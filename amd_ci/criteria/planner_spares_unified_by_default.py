#!/usr/bin/env python3
"""Criteria: now that the planner is default-on, does a unified APU still escape it?

The stakes changed with the default-on flip. While the planner was opt-in, a
Strix Halo owner reached the unified-memory abstain only by asking for it; now
every launch does, so that abstain is the only thing between this hardware and
an -ot spill that moves bytes inside a single pool -- freeing no device memory
and buying only the CPU backend's slower read path.

Regression mode. The question is not "is the base broken": at the base the gate
is off, so nothing is planned and nothing can be wrong. The question is whether
the HEAD, which plans by default, starts spilling on a host where nothing was
spilled before. That is exactly head_is_worse.

Two non-vacuity gates, and the second is the one that matters:

  1. the same layout must really spill when the APU answer is forced False,
     or "abstained" is just a load that fitted anyway;
  2. the gate must be ON by default at the head. Silence from a gate that is
     still off looks identical to silence from the APU abstain, and would let
     a build that never shipped the flip pass as though it had.

Pairs with probes/planner_default_on_probe.py.
"""

from __future__ import annotations

TITLE = "Default-on offload planner vs a unified-memory APU (gfx1151)"
MODE = "regression"
# Declared for the CHANGE, not the host: making the planner default-on affects
# every platform it can place a load on.
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

    out.append(("the head exposes the launch seam",
                bool(head.get("has_seam")),
                f"has_seam={head.get('has_seam')} error={head.get('seam_error', '-')}"))

    # The gate has to be ON, or the silence below is about the gate and not the
    # hardware. This is the gate that would catch a build without the flip.
    out.append(("the planner is ON by default at the head",
                head.get("gate_default") is True,
                f"smart_offload_enabled({{}}) = {head.get('gate_default')}"))

    disc = (head.get("discrete") or {})
    spilled = int(disc.get("n_spilled_blocks") or 0)
    out.append(("the same load really does spill on a discrete host",
                spilled > 0,
                f"discrete spilled {spilled} blocks"))
    return out


def table(obs: dict) -> str:
    rows = ["| state | gate default | detector | unified ids | default: spilled | forced discrete: spilled |",
            "|---|---|---|---|---|---|"]
    for name, v in obs.items():
        if name.startswith("_"):
            continue
        if not v.get("has_seam"):
            rows.append(f"| {name} | {v.get('gate_default', '-')} | "
                        f"{v.get('detector_says_unified', '-')} | "
                        f"{v.get('unified_gpu_ids', '-')} | no seam | no seam |")
            continue
        d = v.get("default") or {}
        x = v.get("discrete") or {}
        rows.append(f"| {name} | {v.get('gate_default')} | "
                    f"{v.get('detector_says_unified')} | {v.get('unified_gpu_ids', '-')} | "
                    f"{'-' if not d.get('planned') else d.get('n_spilled_blocks')} | "
                    f"{'-' if not x.get('planned') else x.get('n_spilled_blocks')} |")
    head = _head(obs)
    reason = ((head.get("default") or {}).get("reason") or "").strip()
    if reason:
        rows += ["", f"Head's plan on the default path: {reason}"]
    return "\n".join(rows)


def head_is_worse(base: dict, head: dict) -> tuple[bool, str]:
    if not head.get("detector_says_unified"):
        # Not a pass and not a failure of the change: the host did not present
        # as unified, so the abstain was never asked the question.
        return True, ("the detector did NOT report this host as unified memory "
                      f"(ids={head.get('unified_gpu_ids')}), so the abstain was never "
                      "exercised; treat this as no result rather than a pass")

    d = head.get("default") or {}
    spilled = int(d.get("n_spilled_blocks") or 0)
    if d.get("planned") and (spilled or d.get("spilled_lm_head")):
        return True, (f"with nothing set in the environment the planner spilled on a "
                      f"unified-memory host: {spilled} blocks, "
                      f"lm_head={d.get('spilled_lm_head')}, reason={d.get('reason')!r}")

    x = head.get("discrete") or {}
    return False, ("with the planner default-ON and nothing set in the environment, the seam "
                   "declined on the real gfx1151 unified pool while spilling "
                   f"{x.get('n_spilled_blocks')} blocks for the identical load with the APU "
                   "answer forced False, so the abstain is attributable to the detected "
                   "hardware and not to the gate or to a load that fitted anyway")
