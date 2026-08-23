#!/usr/bin/env python3
"""Turn the 9362 observations into a verdict, with the non-vacuity gates first.

Order matters here. Before comparing anything, this checks that the holder
really held memory, that the observer was really a bystander, and that the idle
baseline was quiet. A differential measured against a drifting baseline is
inconclusive, and widening the tolerance until it passes would be the easiest
way to fake this result.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

GIB = 1024 ** 3


def jload(p: Path) -> dict | None:
    try:
        return json.loads(p.read_text())
    except Exception:  # noqa: BLE001
        return None


def main() -> int:
    d = Path(sys.argv[1])
    lines: list[str] = ["## PR 9362, driver-level free VRAM", ""]

    idles = [jload(d / f"idle_{i}.json") for i in (1, 2, 3)]
    idles = [x for x in idles if x]
    holder_raw = (d / "holder.log").read_text() if (d / "holder.log").is_file() else ""
    holder = None
    for line in holder_raw.splitlines():
        if line.strip().startswith("{"):
            try:
                j = json.loads(line)
                if j.get("status") == "READY":
                    holder = j
            except Exception:  # noqa: BLE001
                pass

    # ---- gates
    gates: list[tuple[str, bool, str]] = []

    idle_free = [x.get("raw_driver_free_gib") for x in idles if x.get("raw_driver_free_gib")]
    quiet = bool(idle_free) and (max(idle_free) - min(idle_free)) <= 0.25
    gates.append(("idle baseline quiet within 256 MiB", quiet,
                  f"spread {max(idle_free) - min(idle_free):.3f} GiB" if idle_free else "no samples"))

    held = (holder or {}).get("allocated_gib", 0)
    gates.append(("holder really held >= 2 GiB", held >= 2.0, f"{held:.2f} GiB"))

    busy = {p.stem.replace("busy_", ""): jload(p) for p in sorted(d.glob("busy_*.json"))}
    busy = {k: v for k, v in busy.items() if v}
    bystander = all((v.get("observer_allocated_gib") or 0) < 0.5 for v in busy.values())
    gates.append(("observer stayed a bystander", bystander,
                  ", ".join(f"{k}={v.get('observer_allocated_gib', 0):.2f}" for k, v in busy.items())))

    lines += ["| gate | ok | evidence |", "|---|---|---|"]
    for name, ok, ev in gates:
        lines.append(f"| {name} | {'yes' if ok else 'NO'} | {ev} |")
    lines.append("")

    if not all(ok for _, ok, _ in gates):
        lines.append("**INCONCLUSIVE** - a non-vacuity gate failed. The comparison below is "
                     "not trustworthy and the tolerances must not be widened to rescue it.")
        print("\n".join(lines))
        return 0

    # ---- the differential
    lines += ["| state | summary free | driver free | delta | summary total | props total |",
              "|---|---|---|---|---|---|"]
    deltas: dict[str, float] = {}
    for name, v in busy.items():
        s = v.get("summary") or {}
        fr, raw = s.get("vram_free_gb"), v.get("raw_driver_free_gib")
        if fr is None or raw is None:
            lines.append(f"| {name} | error: {v.get('summary_error', 'no reading')} | | | | |")
            continue
        deltas[name] = fr - raw
        lines.append(f"| {name} | {fr:.2f} | {raw:.2f} | {fr - raw:+.2f} | "
                     f"{s.get('vram_total_gb')} | {v.get('props_total_gib', 0):.2f} |")
    lines.append("")

    base_d = deltas.get("base62")
    head_d = deltas.get("head62")
    if base_d is None or head_d is None:
        lines.append("**VOID** - missing a base or head reading.")
        print("\n".join(lines))
        return 0

    # Pre-fix should over-report by most of what the holder took; post-fix
    # should track the driver.
    total = next((v.get("raw_driver_total_gib", 0) for v in busy.values()), 0)
    # Two ceilings, and the tighter one wins. A percentage of a large unified
    # pool alone is loose enough to let a half-fix through: on a 178 GiB card
    # 2% is 3.6 GiB, nearly the whole holder. Tying it to what the holder
    # actually took keeps the test honest whatever the card's size.
    tol = min(max(0.5, 0.02 * total), max(0.5, 0.25 * held))
    base_over = base_d >= 0.5 * held
    head_tracks = abs(head_d) <= tol

    lines.append(f"- holder took {held:.2f} GiB; tolerance {tol:.2f} GiB "
                 f"(min of 2% of total and 25% of the holder)")
    lines.append(f"- base over-reported free by {base_d:+.2f} GiB: **{base_over}**")
    lines.append(f"- head tracked the driver within tolerance: **{head_tracks}**")

    apu = next((v for v in busy.values() if v.get("is_integrated")), None)
    if apu:
        s = apu.get("summary") or {}
        lines.append(f"- APU path: props total {apu.get('props_total_gib', 0):.2f} GiB vs "
                     f"summary total {s.get('vram_total_gb')} GiB "
                     f"(driver {apu.get('raw_driver_total_gib', 0):.2f} GiB)")
    lines.append("")

    if not base_over:
        lines.append("**VOID** - the base leg did not over-report, so the defect did not "
                     "reproduce and the head result proves nothing.")
    elif head_tracks:
        lines.append("**CONFIRMED** - the pre-fix summary was blind to another process's VRAM "
                     "and the post-fix summary tracks the driver.")
    else:
        lines.append("**FIX INCOMPLETE** - base over-reported but head did not track the driver.")

    rec = jload(d / "recovered.json")
    if rec and idle_free:
        r = rec.get("raw_driver_free_gib", 0)
        lines.append("")
        lines.append(f"- free after the holder exited: {r:.2f} GiB vs idle median "
                     f"{statistics.median(idle_free):.2f} GiB")

    lines += ["", "### Not tested here", "",
              "- Windows ROCm/WDDM free cap: `rocm_windows_free_is_untrusted()` is false on "
              "Linux, so `trusted_mem_get_info` was a pure passthrough in every reading above.",
              "- `nvidia.py` UUID masks, MIG, multi-GPU reordering: NVIDIA-only, and this is a "
              "single-GPU AMD host.",
              "- XPU and MLX paths: need Intel / Apple hardware."]
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
