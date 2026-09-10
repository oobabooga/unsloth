#!/usr/bin/env python3
"""Criteria: can the GPU summary see VRAM held by a different process?

The defect shape is a summary computed as `total - this_process_allocated`,
which is blind to other processes by construction. With a holder fixture
resident, the base state should over-report free by roughly what the holder
took, and the head should track the driver.

Tolerance is the TIGHTER of two ceilings on purpose. A flat percentage of total
is too loose on a large unified pool: 2% of 178 GiB is 3.6 GiB, nearly the whole
holder, which would let a half-fix pass. Tying it to the holder keeps the test
honest whatever the card's size.

Two gates bound the claim to the part it is about. The comparison is against the
DRIVER's free figure, so it is worth nothing on a host where that figure does not
itself respond to a resident allocation: on an integrated part the pool is system
RAM, and whether `hipMemGetInfo` free tracks physical residency there is exactly
what is in doubt (ROCm/ROCm#5595, ggml-org/llama.cpp#18159). The fixture reads
free before and after committing, so "the driver's own free moved" is checkable
rather than assumed, and a host that is not the integrated gfx1151 this is a
claim about is INCONCLUSIVE rather than a quiet pass on different silicon.
"""

from __future__ import annotations

GIB = 1024 ** 3
TITLE = "GPU summary against driver-reported free VRAM"
MODE = "differential"
NEEDS = ["gpu", "rocm", "integrated_gpu", "windows", "windows_rocm_wddm",
         "multi_gpu", "nvidia", "mig", "xpu", "mlx"]

# The part this is a claim about. Anything else is INCONCLUSIVE: a summary that
# tracks the driver on a discrete card says nothing about a unified pool, and
# reporting it as a pass would be a result about the wrong hardware.
EXPECT_ARCH = "gfx1151"


def _held(obs: dict) -> float:
    return float((obs.get("_fixture") or {}).get("allocated_gib") or 0.0)


def _delta(state: dict) -> float | None:
    s = state.get("summary") or {}
    fr, raw = s.get("vram_free_gb"), state.get("raw_driver_free_gib")
    if fr is None or raw is None:
        return None
    return fr - raw


# The criteria API passes a single state to base_shows_defect/head_is_fixed, so
# the holder size travels on the state via this module-level stash, set by
# gates() which does see the whole observation set.
_CTX: dict = {"held": 0.0, "tol": 0.5}


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    held = _held(obs)
    total = 0.0
    for name, v in obs.items():
        if not name.startswith("_") and v.get("raw_driver_total_gib"):
            total = v["raw_driver_total_gib"]
            break
    _CTX["held"] = held
    _CTX["tol"] = min(max(0.5, 0.02 * total), max(0.5, 0.25 * held)) if held else 0.5

    fixture = obs.get("_fixture") or {}
    states = {n: v for n, v in obs.items() if not n.startswith("_")}

    out = [("holder really held >= 2 GiB", held >= 2.0, f"{held:.2f} GiB")]

    # Is this the hardware the claim is about? Read off the head where possible,
    # since every state runs on the same box.
    ref = states.get("head") or (next(iter(states.values())) if states else {})
    arch = str(ref.get("arch") or "")
    integrated = ref.get("is_integrated")
    out.append((f"the host is an integrated {EXPECT_ARCH} part",
                arch.startswith(EXPECT_ARCH) and bool(integrated),
                f"arch={arch or '-'} is_integrated={integrated} "
                f"torch={ref.get('torch_version', '-')} hip={ref.get('torch_hip', '-')}"))

    # The driver figure is the yardstick, so it has to be a yardstick. If free
    # does not move when 8 GiB is committed and touched, every delta below is a
    # comparison against a constant.
    before = fixture.get("driver_free_before_gib")
    after = fixture.get("driver_free_gib")
    drop = fixture.get("driver_free_drop_gib")
    moved = (drop is not None and held > 0 and drop >= 0.5 * held)
    out.append(("the driver's own free tracks the held allocation",
                bool(moved),
                f"free {before if before is None else f'{before:.2f}'} -> "
                f"{after if after is None else f'{after:.2f}'} GiB, "
                f"drop {drop if drop is None else f'{drop:.2f}'} GiB against "
                f"{held:.2f} GiB held"))
    bystanders = {n: v.get("observer_allocated_gib", 0)
                  for n, v in obs.items() if not n.startswith("_")}
    out.append(("observer stayed a bystander",
                all((v or 0) < 0.5 for v in bystanders.values()),
                ", ".join(f"{k}={v:.2f}" for k, v in bystanders.items())))
    readable = all((v.get("summary") or {}).get("vram_free_gb") is not None
                   for n, v in obs.items() if not n.startswith("_"))
    out.append(("every state produced a summary", readable,
                "" if readable else "a state failed to read the summary"))
    return out


def table(obs: dict) -> str:
    rows = ["| state | summary free | driver free | delta | summary total | props total |",
            "|---|---|---|---|---|---|"]
    for name, v in obs.items():
        if name.startswith("_"):
            continue
        s = v.get("summary") or {}
        fr, raw = s.get("vram_free_gb"), v.get("raw_driver_free_gib")
        if fr is None or raw is None:
            rows.append(f"| {name} | error: {v.get('summary_error', 'no reading')} | | | | |")
            continue
        rows.append(f"| {name} | {fr:.2f} | {raw:.2f} | {fr - raw:+.2f} | "
                    f"{s.get('vram_total_gb')} | {v.get('props_total_gib', 0):.2f} |")
    rows.append("")
    rows.append(f"Holder took {_CTX['held']:.2f} GiB; tolerance {_CTX['tol']:.2f} GiB "
                f"(the tighter of 2% of total and 25% of the holder).")

    kernel = next((v.get("kernel") for n, v in obs.items()
                   if not n.startswith("_") and v.get("kernel")), None)
    if kernel:
        rows += ["", "Kernel's own view, for the record. Which of these the driver's"
                     " reported total actually is cannot be read off `mem_get_info`:",
                 "", "| source | GiB |", "|---|---|"]
        limit = kernel.get("ttm_pages_limit_gib")
        rows.append(f"| ttm pages_limit | {'-' if limit is None else f'{limit:.2f}'} |")
        for card, row in (kernel.get("cards") or {}).items():
            for name in ("mem_info_gtt_total", "mem_info_gtt_used",
                         "mem_info_vram_total", "mem_info_vram_used"):
                value = row.get(f"{name}_gib")
                if value is not None:
                    rows.append(f"| {card} {name} | {value:.2f} |")
    return "\n".join(rows)


def base_shows_defect(base: dict) -> bool:
    d = _delta(base)
    return d is not None and d >= 0.5 * _CTX["held"]


def head_is_fixed(head: dict) -> bool:
    d = _delta(head)
    return d is not None and abs(d) <= _CTX["tol"]
