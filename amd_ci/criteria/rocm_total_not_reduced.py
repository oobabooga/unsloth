#!/usr/bin/env python3
"""Criteria: does the change reduce a total, or wake the Windows path up on Linux?

Regression mode ON PURPOSE. PR 9314 fixes a Windows carve-out, and this host is
Linux, so the base cannot exhibit that defect and a differential would be VOID by
the one rule. What this host CAN answer is whether the change is inert where it
promises to be inert, and whether the correction that is live on Linux
(_torch_get_device_inventory) ever reports a device as smaller than it was.

Two ways to be worse:

  * a total shrank. An understated total hides models the device can hold, which
    is the failure the carve-out classifier exists to prevent in the first place.
  * the Windows-only path returned something on a host that is not Windows. That
    would break the cross-platform claim the whole review rests on.

Pairs with probes/rocm_total_correction_probe.py.
"""

from __future__ import annotations

TITLE = "ROCm total correction on a real gfx1151 APU, base versus head"
MODE = "regression"
# Declare what the CHANGE needs, not merely what this probe touches. The gaps are
# what get rendered as "Not tested here", so anything omitted here is a claim the
# report will silently appear to cover. PR 9314 is a Windows carve-out fix whose
# worst regression needed an APU beside a discrete card, and this host is one
# integrated GPU on Linux, so it can answer none of that.
# The context measurement below is single-GPU on an APU the classifier calls
# unified, so it cannot show the case the concern is really about: a DISCRETE card
# on an unsettled runtime, where the classifier fails open and the probe would
# attach a context that buys nothing. discrete_gpu and multi_gpu already carry it.
NEEDS: list[str] = ["rocm", "windows", "windows_rocm_wddm", "discrete_gpu", "multi_gpu"]

TOL = 0.01  # GiB; totals are rounded to 2 dp before we see them


def _totals(o: dict) -> dict:
    """device index -> total_gb, as the inventory path reports it."""
    return {d.get("index"): d.get("total_gb") for d in (o.get("inventory") or [])}


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out = []
    for name in ("base", "head"):
        o = obs.get(name) or {}
        ran = not o.get("hardware_error") and o.get("inventory") is not None
        out.append((f"{name} probe read the hardware", ran,
                    o.get("hardware_error") or o.get("torch_error")
                    or f"{len(o.get('inventory') or [])} device(s)"))
    head = obs.get("head") or {}
    # Non-vacuity: if the classifier never fires here, the live-on-Linux half of
    # this measurement is not actually exercising the changed branch, and the run
    # should say so rather than quietly reporting a pass.
    out.append(("the carve-out classifier fires on this host",
                bool(head.get("classifier_says_carve_out")),
                f"classifier_says_carve_out={head.get('classifier_says_carve_out')}"))
    return out


def table(obs: dict) -> str:
    rows = ["| state | props total | driver total | inventory total | windows path "
            "| classifier | poll attaches context |",
            "|---|---|---|---|---|---|---|"]
    for name in ("base", "head", "merge"):
        o = obs.get(name)
        if not o:
            continue
        totals = ", ".join(f"{v} GiB" for v in _totals(o).values()) or "none"
        inert = o.get("windows_path_inert")
        rows.append(
            f"| {name} | {o.get('props_total_gib', 0):.2f} GiB "
            f"| {o.get('driver_total_gib', 0):.2f} GiB | {totals} "
            f"| {'inert' if inert else 'RETURNED DATA'} "
            f"| {'carve-out' if o.get('classifier_says_carve_out') else 'settled'} "
            f"| {_ctx(o)} |")
    head = obs.get("head") or {}
    note = (
        "\n\nThis host is a single unified-memory APU on Linux, where "
        f"`props.total_memory` and the driver total agree "
        f"(`driver_exceeds_props={head.get('driver_exceeds_props')}`), so there is no "
        "carve-out to widen and nothing here reproduces the Windows defect. That is "
        "the point of the measurement, not a shortcoming of it.")
    return "\n".join(rows) + note


def _ctx(o: dict) -> bool | None:
    """Did one telemetry poll leave a primary HIP context attached?"""
    c = o.get("context_cost") or {}
    if c.get("error") or "initialized_after" not in c:
        return None
    return bool(c["initialized_after"])


def head_is_worse(base: dict, head: dict) -> tuple[bool, str]:
    # A context the base did not attach is a permanent ~612 MiB off the budget on
    # every poll, which is the cost the poll path exists to avoid.
    b_ctx, h_ctx = _ctx(base), _ctx(head)
    if b_ctx is False and h_ctx is True:
        return True, ("one telemetry poll attaches a primary HIP context at the head "
                      "that the base did not attach")

    if not head.get("windows_path_inert", False):
        return True, ("`_rocm_windows_per_device_vram` returned data on a "
                      f"{head.get('platform')} host; it must be inert off Windows")

    b_tot, h_tot = _totals(base), _totals(head)
    shrunk = [
        f"device {i}: {b_tot[i]} -> {h_tot.get(i)} GiB"
        for i in b_tot
        if i in h_tot and b_tot[i] is not None and h_tot[i] is not None
        and h_tot[i] < b_tot[i] - TOL
    ]
    if shrunk:
        return True, "a reported total shrank at the head: " + "; ".join(shrunk)

    dropped = sorted(set(b_tot) - set(h_tot))
    if dropped:
        return True, f"device(s) {dropped} present at the base and missing at the head"

    return False, ("the Windows path stayed inert, every device the base reported is "
                   "still reported, no total shrank, and the poll attaches no "
                   f"context the base did not (base={b_ctx}, head={h_ctx})")
