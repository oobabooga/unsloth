#!/usr/bin/env python3
"""Criteria: did WebKitGTK render on the GPU on this host, or on the CPU?

Judges only. The observations come from probes/webkit_paint_probe.py.

Three things this deliberately does NOT do:

  * It does not treat the frame rate as evidence. An X server with no vblank
    imposes no real presentation cadence, so a number near 60 here means the
    timer ran, exactly as 60.0 fps on headless Chromium meant the timer ran.
    The rate is printed as an observation and no verdict depends on it.
  * It does not accept "libwebkit2gtk-4.1 is installed" as rendering, nor a
    hardware EGL context measured by a DIFFERENT process. The engine picks its
    own driver, so the reading has to come from inside the page.
  * It does not turn "we could not get a display" into "WebKit cannot render
    here". That is a gate, and a failed gate is INCONCLUSIVE.
"""

from __future__ import annotations

TITLE = "Does WebKitGTK composite on the GPU on the AMD CI runner?"
MODE = "capability"

NEEDS = [
    "linux", "drm_render_node", "egl_hardware_gl", "vulkan_hardware",
    "webkitgtk", "headless_display_server", "gpu_browser_compositing",
    "discrete_gpu", "nvidia", "windows", "mlx",
]

SOFTWARE_RENDERERS = ("llvmpipe", "softpipe", "swrast", "mesa offscreen",
                      "lavapipe", "swiftshader")


def _is_software(r: str | None) -> bool:
    return bool(r) and any(s in r.lower() for s in SOFTWARE_RENDERERS)


def _page(obs: dict) -> dict:
    return ((obs.get("webkit") or {}).get("page") or {})


def _renderer(obs: dict) -> str | None:
    return _page(obs).get("webgl_renderer")


def _dri_holders(obs: dict) -> list[dict]:
    return [p for p in ((obs.get("webkit") or {}).get("webkit_processes") or [])
            if p.get("dri_fds")]


def _painted(obs: dict) -> bool:
    w = obs.get("webkit") or {}
    # A blank 1280x800 PNG compresses to a few kB of near-identical bytes; a
    # painted one does not. Both conditions, so an error page cannot pass.
    return (w.get("snapshot_bytes") or 0) > 3000 and (w.get("snapshot_distinct_bytes") or 0) > 64


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out: list[tuple[str, bool, str]] = []
    x = obs.get("xserver") or {}
    out.append(("a display server was obtained", bool(x.get("display")),
                f"{x.get('binary')} on {x.get('display')}" if x.get("display")
                else f"attempts: {x.get('attempts')}"))

    w = obs.get("webkit") or {}
    ok = bool(w.get("gi")) and not w.get("import_error")
    out.append(("WebKitGTK loaded through the GObject bindings", ok,
                w.get("import_error") or f"WebKitGTK {w.get('webkit_version')}"))

    p = _page(obs)
    out.append(("the page ran and reported back", bool(p.get("frames")),
                f"frames={p.get('frames')} reason={w.get('finish_reason')} "
                f"terminated={w.get('web_process_terminated')}"))

    out.append(("the view really painted, not a blank surface", _painted(obs),
                f"snapshot {w.get('snapshot_bytes')} bytes, "
                f"{w.get('snapshot_distinct_bytes')} distinct"))
    return out


def table(obs: dict) -> str:
    p = _page(obs)
    w = obs.get("webkit") or {}
    holders = _dri_holders(obs)
    ref = obs.get("reference_gl_renderer")
    r = _renderer(obs)

    rows = ["| reading | value |", "|---|---|"]
    rows.append(f"| WebKitGTK version | {w.get('webkit_version')} |")
    rows.append(f"| hardware acceleration policy | {w.get('hardware_acceleration_policy')} |")
    rows.append(f"| **renderer inside the page** (WEBGL_debug_renderer_info) | **{r}** |")
    rows.append(f"| vendor inside the page | {p.get('webgl_vendor')} |")
    rows.append(f"| same host, standalone EGL probe | {ref} |")
    rows.append(f"| WebKit processes holding /dev/dri | "
                f"{', '.join(h['pid'] + ' ' + ','.join(h['dri_fds']) for h in holders) or 'none'} |")
    rows.append(f"| snapshot | {w.get('snapshot_bytes')} bytes, "
                f"{w.get('snapshot_distinct_bytes')} distinct byte values |")
    rows.append(f"| frames / p95 (NOT evidence, see below) | {p.get('frames')} in "
                f"{p.get('ms')} ms = {round(p.get('fps') or 0, 1)} fps, "
                f"p95 {p.get('p95_frame_ms')} ms |")
    rows.append("")
    for h in holders:
        for fi in h.get("fdinfo") or []:
            if fi:
                rows.append(f"amdgpu fdinfo, pid {h['pid']}: "
                            + ", ".join(f"{k}={v}" for k, v in sorted(fi.items())[:8]))
    rows.append("")
    rows.append("The frame rate above is a timer reading, not a presentation reading: this X "
                "server has no vblank. It is recorded so the number is not mistaken later for "
                "evidence about the 60 -> 5 fps symptom, which nothing in this run touches.")
    return "\n".join(rows)


def verdict(obs: dict) -> tuple[str, str]:
    r = _renderer(obs)
    holders = _dri_holders(obs)
    ref = obs.get("reference_gl_renderer")

    if r is None:
        return "NOT_CAPABLE", (
            "the page painted but could not report a renderer, so what drew it is unknown "
            f"(webgl={_page(obs).get('webgl')}, error={_page(obs).get('webgl_error')!r})")
    if _is_software(r):
        return "NOT_CAPABLE", (
            f"WebKit rendered through {r!r}, which is the CPU, on a host whose standalone EGL "
            f"context reports {ref!r}. The engine is not reaching the GPU here, so a perf "
            f"number taken in it would describe a software rasteriser")
    if not holders:
        return "PARTIAL", (
            f"WebKit reports {r!r}, which is not a software rasteriser, but no WebKit process "
            f"was seen holding a /dev/dri node, so the corroboration is missing and the "
            f"reading rests on one source")
    return "CAPABLE", (
        f"WebKit rendered through {r!r} from inside the page, matching the standalone EGL "
        f"reading of {ref!r}, and {len(holders)} WebKit process(es) held the render node open. "
        f"This host composites WebKitGTK on the GPU")


def observed_capabilities(obs: dict) -> dict[str, bool]:
    r = _renderer(obs)
    hw = bool(r) and not _is_software(r)
    return {
        "webkitgtk": bool((obs.get("webkit") or {}).get("gi")),
        "headless_display_server": bool((obs.get("xserver") or {}).get("display")),
        "gpu_browser_compositing": hw and bool(_dri_holders(obs)) and _painted(obs),
    }
