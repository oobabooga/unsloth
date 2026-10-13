#!/usr/bin/env python3
"""Criteria: does the H3 native fit read the real amdgpu GTT pool and keep an offloaded render inside it?

Base has no GTT-aware fit, so an offloaded --offload-to-cpu render pins its whole set whatever the pool (the reported
radv "Not enough memory for command submission"). Head must read the same free GTT sysfs reports, keep a render
1.4x the pool out of it (encoder to the CPU or a refusal) and leave a render at half the pool unchanged."""

from __future__ import annotations

GIB = 1024**3
TITLE = "MiniMax-H3 native fit against the real amdgpu GTT pool"
MODE = "differential"
NEEDS = ["gpu", "vulkan", "windows"]
AUTO = ["--offload-to-cpu", "--diffusion-fa", "--max-vram", "-1"]
TOL = 512 * 1024**2


def _amd(obs_state: dict) -> list:
    return [c for c in obs_state.get("raw_cards", []) if c.get("vendor") == "0x1002" and c.get("gtt_total")]


def gates(obs: dict) -> list:
    out = []
    for name in ("base", "head"):
        s = obs.get(name) or {}
        out.append((f"{name}: an amdgpu card reports mem_info_gtt_total", bool(_amd(s)), str(_amd(s))[:300]))
        out.append((f"{name}: checkout imported", "import_error" not in s, s.get("import_error", "ok")))
    return out


def table(obs: dict) -> str:
    rows = ["| state | GTT total (sysfs) | GTT used (sysfs) | fit reads free | over-pool flags / refusal | under-pool flags |",
            "|---|---|---|---|---|---|"]
    for name in ("base", "head"):
        s = obs.get(name) or {}
        cards = _amd(s)
        c = cards[0] if cards else {}
        r = (s.get("reading_any") or {}).get("free")
        f = s.get("fits") or {}
        over = f.get("over_pool") or {}
        under = f.get("under_pool") or {}
        g = lambda v: f"{v / GIB:.2f} GiB" if isinstance(v, int) else "n/a"  # noqa: E731
        rows.append(
            f"| {name} | {g(c.get('gtt_total'))} | {g(c.get('gtt_used'))} | {g(r)} | "
            f"{' '.join(over.get('flags', [])) or 'no fit (absent)'} / {over.get('refusal') or '-'} | "
            f"{' '.join(under.get('flags', [])) or 'no fit (absent)'} |"
        )
    head = obs.get("head") or {}
    rows.append("")
    rows.append(f"head host_budgets={head.get('host_budgets')} rocm_missing_sonames={head.get('rocm_missing_sonames')}")
    rows.append(f"head over-pool note: {((head.get('fits') or {}).get('over_pool') or {}).get('note')}")
    return "\n".join(rows)


def base_shows_defect(base: dict) -> bool:
    # No GTT-aware fit at the base: an over-pool offloaded render is launched with its flags unchanged.
    return bool(_amd(base)) and not base.get("has_gtt_fit")


def head_is_fixed(head: dict) -> bool:
    cards = _amd(head)
    if not cards or not head.get("has_gtt_fit"):
        return False
    c = cards[0]
    if c.get("gtt_used") is None:
        return False
    raw_free = c["gtt_total"] - c["gtt_used"]
    read = (head.get("reading_per_slot") or {}).get(c["pci_slot"]) or [None, None]
    if read[0] is None or abs(read[0] - raw_free) > TOL or read[1] is not True:
        return False
    fits = head.get("fits") or {}
    over, under = fits.get("over_pool") or {}, fits.get("under_pool") or {}
    over_kept_out = over.get("refusal") is not None or "--clip-on-cpu" in over.get("flags", [])
    under_unchanged = under.get("refusal") is None and under.get("flags") == AUTO
    return over_kept_out and under_unchanged
