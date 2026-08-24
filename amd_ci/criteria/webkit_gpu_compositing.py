#!/usr/bin/env python3
"""Criteria: did WebKitGTK render on the GPU on this host, or on the CPU?

Judges only. The observations come from probes/webkit_paint_probe.py.

The first version of this file asked the page for its renderer through
`WEBGL_debug_renderer_info` and treated any string that was not a known software
rasteriser as hardware. On the runner that returned **"Apple GPU" / "Apple Inc."**
from WebKitGTK on Linux: every WebKit port masks that extension for
fingerprinting reasons, and there is no runtime switch to unmask it. The reading
was therefore not a reading at all, and "not llvmpipe" quietly passed it. That is
the exact failure mode this toolkit exists to prevent, so the string is now
recognised as MASKED and cannot support a verdict on its own.

What replaced it is evidence the engine does not author:

  * amdgpu's per-fd counters (`drm-engine-gfx`, `drm-memory-vram`) for the render
    node held open by WebKit's own process, read from /proc by the kernel driver;
  * which mesa driver that process mapped into its address space
    (`radeonsi_dri.so` against `swrast_dri.so`).

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

# What WebKit reports instead of the device. On a Linux/AMD host this is a mask,
# not a Mac.
MASKED_RENDERERS = ("apple gpu", "apple inc")

SOFTWARE_DRIVER_LIBS = ("swrast_dri.so", "llvmpipe", "softpipe")
AMD_DRIVER_LIBS = ("radeonsi_dri.so", "radeonsi")


def _is_software(r: str | None) -> bool:
    return bool(r) and any(s in r.lower() for s in SOFTWARE_RENDERERS)


def _is_masked(r: str | None) -> bool:
    return bool(r) and any(s in r.lower() for s in MASKED_RENDERERS)


def _page(obs: dict) -> dict:
    return ((obs.get("webkit") or {}).get("page") or {})


def _renderer(obs: dict) -> str | None:
    return _page(obs).get("webgl_renderer")


def _dri_holders(obs: dict) -> list[dict]:
    return [p for p in ((obs.get("webkit") or {}).get("webkit_processes") or [])
            if p.get("dri_fds")]


def _procs(obs: dict) -> list[dict]:
    return ((obs.get("webkit") or {}).get("webkit_processes") or [])


def _gfx_engine_ns(obs: dict) -> int:
    """GPU engine time the kernel attributes to WebKit's own process."""
    best = 0
    for p in _procs(obs):
        for fi in p.get("fdinfo") or []:
            v = (fi or {}).get("drm-engine-gfx") or ""
            digits = "".join(ch for ch in v if ch.isdigit())
            if digits:
                best = max(best, int(digits))
    return best


def _vram_kib(obs: dict) -> int:
    best = 0
    for p in _procs(obs):
        for fi in p.get("fdinfo") or []:
            v = (fi or {}).get("drm-memory-vram") or ""
            digits = "".join(ch for ch in v if ch.isdigit())
            if digits:
                best = max(best, int(digits))
    return best


def _mapped(obs: dict) -> set[str]:
    out: set[str] = set()
    for p in _procs(obs):
        out |= set(p.get("mapped_drivers") or [])
    return out


def _amd_driver_mapped(obs: dict) -> bool:
    return any(any(t in m for t in AMD_DRIVER_LIBS) for m in _mapped(obs))


def _software_driver_mapped(obs: dict) -> bool:
    return any(any(t in m for t in SOFTWARE_DRIVER_LIBS) for m in _mapped(obs))


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
    rows.append(f"| renderer inside the page (WEBGL_debug_renderer_info) | {r}"
                f"{' - MASKED by WebKit, carries no device information' if _is_masked(r) else ''} |")
    rows.append(f"| vendor inside the page | {p.get('webgl_vendor')} |")
    rows.append(f"| same host, standalone EGL probe | {ref} |")
    rows.append(f"| **mesa driver mapped into WebKit's process** | "
                f"**{', '.join(sorted(_mapped(obs))) or 'none'}** |")
    rows.append(f"| **amdgpu GFX engine time attributed to WebKit** | "
                f"**{_gfx_engine_ns(obs):,} ns** over {p.get('ms')} ms of animation |")
    rows.append(f"| amdgpu VRAM attributed to WebKit | {_vram_kib(obs):,} KiB |")
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


def _gpu_composited(obs: dict) -> bool:
    """All three, from sources the engine does not author."""
    return (_amd_driver_mapped(obs) and not _software_driver_mapped(obs)
            and _gfx_engine_ns(obs) > 0 and bool(_dri_holders(obs)))


def verdict(obs: dict) -> tuple[str, str]:
    r = _renderer(obs)
    holders = _dri_holders(obs)
    ns, mapped = _gfx_engine_ns(obs), _mapped(obs)

    # An in-page string that names a software rasteriser is still decisive
    # against, because nothing masks its way INTO llvmpipe.
    if _is_software(r):
        return "NOT_CAPABLE", (
            f"WebKit rendered through {r!r}, which is the CPU, so a perf number taken in it "
            f"would describe a software rasteriser")
    if _software_driver_mapped(obs) and not _amd_driver_mapped(obs):
        return "NOT_CAPABLE", (
            f"WebKit's process mapped {sorted(mapped)}, a software rasteriser, and no AMD "
            f"driver, so the engine is not reaching the GPU here")
    if not holders:
        return "NOT_CAPABLE", (
            "no WebKit process held a /dev/dri node open while an animating page was "
            "compositing, so nothing it drew went through the GPU")
    if not _amd_driver_mapped(obs):
        return "PARTIAL", (
            f"a WebKit process held the render node, but no AMD driver was found mapped into "
            f"it (mapped: {sorted(mapped) or 'nothing recognised'}), so what did the drawing "
            f"is not established")
    if ns <= 0:
        return "PARTIAL", (
            f"WebKit mapped the AMD driver and held the render node, but amdgpu attributed no "
            f"GFX engine time to it, so the GPU path was opened and possibly never used")
    return "CAPABLE", (
        f"amdgpu attributed {ns:,} ns of GFX engine time and {_vram_kib(obs):,} KiB of VRAM to "
        f"WebKit's own process, which mapped {sorted(m for m in mapped if 'radeonsi' in m)} and "
        f"held {holders[0]['dri_fds'][0]} open while an animating page painted. The in-page "
        f"renderer string is {r!r}, which WebKit masks on every port and which is therefore "
        f"not part of this finding. This host composites WebKitGTK on the GPU")


def observed_capabilities(obs: dict) -> dict[str, bool]:
    return {
        "webkitgtk": bool((obs.get("webkit") or {}).get("gi")),
        "headless_display_server": bool((obs.get("xserver") or {}).get("display")),
        "gpu_browser_compositing": _gpu_composited(obs) and _painted(obs),
    }
