#!/usr/bin/env python3
"""Criteria: MiniMax-H3 GGUF, Vulkan build, integrated AMD (UMA), GTT held to ~22.5 GiB free.

Base reproduces when its Auto 864x480x124 render, launched with the base tree's own flags, fails. Head is fixed when
the head tree reads the device as UMA, launches the same render with a --max-vram reserve (a value below -1) and the
render completes with a non-empty video, and when the 1344x768x124 render does not crash (a refusal before launch or a
completed render both count)."""

from __future__ import annotations

TITLE = "MiniMax-H3 Vulkan render on gfx1151 with GTT held to 22.5 GiB free (PR 13786)"
MODE = "differential"
NEEDS = ["gpu", "vulkan", "windows"]
SMALL, LARGE = "864x480", "1344x768"


def _max_vram(flags):
    if not flags or "--max-vram" not in flags:
        return None
    try:
        return float(flags[flags.index("--max-vram") + 1])
    except (IndexError, ValueError):
        return None


def _entry(s: dict, size: str) -> dict:
    return ((s.get("auto") or {}).get(size)) or {}


def gates(obs: dict) -> list:
    fx = obs.get("_fixture") or {}
    out = [("fixture held GTT at 22.5 GiB free (sysfs)", fx.get("ready") is True and abs(float(fx.get("gtt_free_gib", 0)) - 22.5) <= 0.5,
            f"gtt_free={fx.get('gtt_free_gib')} total={fx.get('gtt_total_gib')}")]
    for name in ("base", "head"):
        s = obs.get(name) or {}
        bad = s.get("import_error") or s.get("probe_error")
        out.append((f"{name}: tree imported and probe completed", not bad, bad or s.get("imported_from", "")))
        out.append((f"{name}: Studio reads a Vulkan AMD device (GTT check on)", s.get("listed_class") == "vulkan" and s.get("check_gtt") is True,
                    f"{s.get('listed_class')} / {s.get('listed_desc')}"))
        e = _entry(s, SMALL).get("launch") or {}
        out.append((f"{name}: Auto picked the offloaded path at {SMALL} (the path under test)", e.get("resident") is False,
                    f"resident={e.get('resident')} card_free={e.get('card_free_bytes')}"))
        out.append((f"{name}: {SMALL} launch produced flags and a render ran", bool(_entry(s, SMALL).get("renders")),
                    f"refusal={e.get('refusal')}"))
    h = obs.get("head") or {}
    out.append(("head: the new UMA read engaged (sd_cpp_vulkan_device_is_uma -> True)", h.get("uma_reading") is True, str(h.get("uma_reading"))))
    return out


def table(obs: dict) -> str:
    rows = ["| state | size | --max-vram | launch refusal | render | secs | peak GTT used | sd.cpp budget line | user-visible error advice |",
            "|---|---|---|---|---|---|---|---|---|"]
    for name in ("base", "head"):
        s = obs.get(name) or {}
        for size in (SMALL, LARGE):
            e = _entry(s, size)
            la = e.get("launch") or {}
            r = (e.get("renders") or [{}])[0]
            budget = next((x for x in r.get("log_highlights", []) if "auto-detected" in x), "-")
            rend = "not run" if not e.get("renders") else ("OK, %s bytes" % r.get("output_bytes") if r.get("ok") else "FAILED: " + (r.get("error_head") or "")[:120].replace("\n", " "))
            rows.append(f"| {name} | {size} | {_max_vram(la.get('flags'))} | {(la.get('refusal') or '-')[:160]} | {rend} | {r.get('secs', '-')} | "
                        f"{r.get('peak_gtt_used_gib', '-')} | {budget[:160]} | {(r.get('user_advice') or '-')[:120]} |")
    for name in ("base", "head"):
        s = obs.get(name) or {}
        ex = (s.get("explicit") or {}).get("low_vram") or {}
        la = ex.get("launch_864x480") or {}
        rows.append(f"\n{name} Low VRAM (explicit, fit only) 864x480 flags: {la.get('flags')} refusal: {la.get('refusal')}")
        rows.append(f"{name} load fit (auto, smallest clip {s.get('small_clip')}): {s.get('auto_load_fit')}")
        for size in (SMALL, LARGE):
            for r in _entry(s, size).get("renders", []):
                rows.append(f"{name} {size} capacity_warnings={r.get('capacity_warnings')} ok={r.get('ok')} "
                            f"capacity_failure={r.get('capacity_failure')} accelerator_failure={r.get('accelerator_failure')} highlights={r.get('log_highlights')[:6]}")
    return "\n".join(rows)


def base_shows_defect(base: dict) -> bool:
    renders = _entry(base, SMALL).get("renders") or []
    return bool(renders) and not renders[0].get("ok")


def head_is_fixed(head: dict) -> bool:
    small = _entry(head, SMALL)
    la = small.get("launch") or {}
    mv = _max_vram(la.get("flags"))
    renders = small.get("renders") or []
    if head.get("uma_reading") is not True or mv is None or mv >= -1 or not renders:
        return False
    if not (renders[0].get("ok") and renders[0].get("output_bytes", 0) > 0):
        return False
    large = _entry(head, LARGE)
    lr = large.get("renders") or []
    refused = (large.get("launch") or {}).get("refusal") is not None
    return refused or bool(lr and lr[0].get("ok"))
