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

The second version replaced it with amdgpu's per-fd counters plus a NAME
WHITELIST over the engine's mapped libraries, and required an AMD-sounding
library name. That returned PARTIAL on a host where amdgpu had attributed 121 ms
of GFX engine time to WebKit's own web process, because the whitelist had no
entry that recent mesa actually uses. A whitelist cannot tell "the driver is not
loaded" apart from "the driver is not on my list", so a library NAME is now
corroboration only and never the deciding gate.

What decides, in order of how hard it is to fake:

  1. A NEGATIVE CONTROL. The same engine, page, X server and probe, run twice,
     with `LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe` the only difference.
     If the GPU counters collapse in the control leg, the counters in the real
     leg were caused by the browser's rendering and not by something incidental.
     This is the base-vs-head discipline of the rest of the toolkit applied to a
     capability question: one leg has to show the absence.
  2. amdgpu's per-fd counters (`drm-engine-gfx`, `drm-memory-vram`) for the
     render node held open by WebKit's own process, sampled early and again at
     the end so the figure is time accrued WHILE the page animated, written by
     the kernel driver and not by anything in userspace.
  3. Which shared objects that process mapped, reported unfiltered.

Four things this deliberately does NOT do:

  * It does not treat the frame rate as evidence. An X server with no vblank
    imposes no real presentation cadence, so a number near 60 here means the
    timer ran, exactly as 60.0 fps on headless Chromium meant the timer ran.
    The rate is printed as an observation and no verdict depends on it.
  * It does not accept "libwebkit2gtk-4.1 is installed" as rendering, nor a
    hardware EGL context measured by a DIFFERENT process, nor an open device
    node, nor a ROCm agent enumeration. Compositing a frame is the claim.
  * It does not accept a mapped `libgallium-*.so` as an AMD driver. That single
    megadriver contains the software rasterisers too, so its presence does not
    name a device; only `radeonsi`/`amdgpu`/`radeon` do.
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

SOFTWARE_DRIVER_LIBS = ("swrast_dri.so", "llvmpipe", "softpipe", "kms_swrast")
# Names that identify the DEVICE. `libgallium-*.so` is deliberately absent: it
# is a megadriver that also contains llvmpipe, so it names nothing.
AMD_DRIVER_LIBS = ("radeonsi", "libvulkan_radeon", "libdrm_amdgpu", "amdgpu")

# Below this share of the real leg's engine time, the control leg counts as
# having collapsed. Not zero: an X server handshake can bill a few microseconds.
CONTROL_COLLAPSE_FRACTION = 0.10


def _is_software(r: str | None) -> bool:
    return bool(r) and any(s in r.lower() for s in SOFTWARE_RENDERERS)


def _is_masked(r: str | None) -> bool:
    return bool(r) and any(s in r.lower() for s in MASKED_RENDERERS)


def _leg(obs: dict, which: str = "webkit") -> dict:
    return obs.get(which) or {}


def _page(obs: dict, which: str = "webkit") -> dict:
    return _leg(obs, which).get("page") or {}


def _renderer(obs: dict) -> str | None:
    return _page(obs).get("webgl_renderer")


def _procs(obs: dict, which: str = "webkit", key: str = "webkit_processes") -> list[dict]:
    return _leg(obs, which).get(key) or []


def _dri_holders(obs: dict, which: str = "webkit") -> list[dict]:
    return [p for p in _procs(obs, which) if p.get("dri_fds")]


def _ns(v: str | None) -> int:
    digits = "".join(ch for ch in (v or "") if ch.isdigit())
    return int(digits) if digits else 0


def _by_client(procs: list[dict], field: str) -> dict[str, int]:
    """field per DRM client id, so two samples can be differenced safely."""
    out: dict[str, int] = {}
    for p in procs:
        for fi in p.get("fdinfo") or []:
            cid = (fi or {}).get("drm-client-id") or p.get("pid") or "?"
            out[cid] = max(out.get(cid, 0), _ns((fi or {}).get(field)))
    return out


def _gfx_engine_ns(obs: dict, which: str = "webkit") -> int:
    """Cumulative GPU engine time the kernel attributes to WebKit's process."""
    v = _by_client(_procs(obs, which), "drm-engine-gfx")
    return max(v.values()) if v else 0


def _gfx_delta_ns(obs: dict, which: str = "webkit") -> int | None:
    """GPU engine time accrued between the early sample and the end.

    None when no early sample exists, which is a missing reading and not a zero.
    """
    t0 = _procs(obs, which, "webkit_processes_t0")
    if not t0:
        return None
    a, b = _by_client(t0, "drm-engine-gfx"), _by_client(_procs(obs, which), "drm-engine-gfx")
    if not b:
        return 0
    return max(max(0, ns - a.get(cid, 0)) for cid, ns in b.items())


def _vram_kib(obs: dict, which: str = "webkit") -> int:
    v = _by_client(_procs(obs, which), "drm-memory-vram")
    return max(v.values()) if v else 0


def _mapped(obs: dict, which: str = "webkit") -> set[str]:
    out: set[str] = set()
    for p in _procs(obs, which):
        out |= set(p.get("mapped_drivers") or [])
    return out


def _mapped_all(obs: dict, which: str = "webkit") -> set[str]:
    out: set[str] = set()
    for p in _procs(obs, which):
        out |= set(p.get("mapped_all") or [])
    return out or _mapped(obs, which)


def _amd_driver_mapped(obs: dict, which: str = "webkit") -> bool:
    pool = _mapped(obs, which) | _mapped_all(obs, which)
    return any(any(t in m for t in AMD_DRIVER_LIBS) for m in pool)


def _software_driver_mapped(obs: dict, which: str = "webkit") -> bool:
    pool = _mapped(obs, which) | _mapped_all(obs, which)
    return any(any(t in m for t in SOFTWARE_DRIVER_LIBS) for m in pool)


def _painted(obs: dict, which: str = "webkit") -> bool:
    w = _leg(obs, which)
    # A blank 1280x800 PNG compresses to a few kB of near-identical bytes; a
    # painted one does not. Both conditions, so an error page cannot pass.
    return (w.get("snapshot_bytes") or 0) > 3000 and (w.get("snapshot_distinct_bytes") or 0) > 64


def _control_ran(obs: dict) -> bool:
    c = _leg(obs, "control")
    return bool(c) and bool(c.get("gi")) and bool(_page(obs, "control").get("frames"))


def _control_collapsed(obs: dict) -> bool | None:
    """Did forcing mesa to software remove the GPU work? None if unmeasured."""
    if not _control_ran(obs):
        return None
    real, ctrl = _gfx_engine_ns(obs), _gfx_engine_ns(obs, "control")
    if real <= 0:
        return None
    return ctrl <= real * CONTROL_COLLAPSE_FRACTION


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out: list[tuple[str, bool, str]] = []
    x = obs.get("xserver") or {}
    out.append(("a display server was obtained", bool(x.get("display")),
                f"{x.get('binary')} on {x.get('display')}" if x.get("display")
                else f"attempts: {x.get('attempts')}"))

    w = _leg(obs)
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
    w = _leg(obs)
    holders = _dri_holders(obs)
    ref = obs.get("reference_gl_renderer")
    r = _renderer(obs)
    delta = _gfx_delta_ns(obs)
    amd_names = sorted(m for m in (_mapped(obs) | _mapped_all(obs))
                       if any(t in m for t in AMD_DRIVER_LIBS))

    rows = ["| reading | value |", "|---|---|"]
    rows.append(f"| WebKitGTK version | {w.get('webkit_version')} |")
    rows.append(f"| hardware acceleration policy | {w.get('hardware_acceleration_policy')} |")
    rows.append(f"| renderer inside the page (WEBGL_debug_renderer_info) | {r}"
                f"{' - MASKED by WebKit, carries no device information' if _is_masked(r) else ''} |")
    rows.append(f"| vendor inside the page | {p.get('webgl_vendor')} |")
    rows.append(f"| unmasked GL RENDERER inside the page | {p.get('webgl_renderer_plain')} |")
    rows.append(f"| WebGPU adapter info inside the page | {p.get('webgpu')} |")
    rows.append(f"| same host, standalone EGL probe (a DIFFERENT process) | {ref} |")
    rows.append(f"| **amdgpu GFX engine time attributed to WebKit** | "
                f"**{_gfx_engine_ns(obs):,} ns** cumulative; "
                f"**{'unmeasured' if delta is None else format(delta, ',') + ' ns'}** accrued "
                f"during the animation |")
    rows.append(f"| amdgpu VRAM attributed to WebKit | {_vram_kib(obs):,} KiB |")
    rows.append(f"| WebKit processes holding /dev/dri | "
                f"{', '.join(h['pid'] + ' ' + ','.join(sorted(set(h['dri_fds']))) for h in holders) or 'none'} |")
    rows.append(f"| device-naming libraries mapped into WebKit | "
                f"{', '.join(amd_names) or 'none recognised (corroboration only)'} |")
    rows.append(f"| software rasteriser mapped into WebKit | "
                f"{'yes' if _software_driver_mapped(obs) else 'no'} |")
    rows.append(f"| snapshot | {w.get('snapshot_bytes')} bytes, "
                f"{w.get('snapshot_distinct_bytes')} distinct byte values |")
    rows.append(f"| frames / p95 (NOT evidence, see below) | {p.get('frames')} in "
                f"{p.get('ms')} ms = {round(p.get('fps') or 0, 1)} fps, "
                f"p95 {p.get('p95_frame_ms')} ms |")

    rows.append("")
    rows.append("**Negative control: the same page with `LIBGL_ALWAYS_SOFTWARE=1 "
                "GALLIUM_DRIVER=llvmpipe`, everything else identical.**")
    rows.append("")
    rows.append("| reading | real leg | forced-software leg |")
    rows.append("|---|---|---|")
    cp = _page(obs, "control")
    rows.append(f"| ran | {bool(p.get('frames'))} | {_control_ran(obs)} |")
    rows.append(f"| amdgpu GFX engine time | {_gfx_engine_ns(obs):,} ns | "
                f"{_gfx_engine_ns(obs, 'control'):,} ns |")
    rows.append(f"| amdgpu VRAM | {_vram_kib(obs):,} KiB | {_vram_kib(obs, 'control'):,} KiB |")
    rows.append(f"| processes holding /dev/dri | {len(holders)} | "
                f"{len(_dri_holders(obs, 'control'))} |")
    rows.append(f"| software rasteriser mapped | {_software_driver_mapped(obs)} | "
                f"{_software_driver_mapped(obs, 'control')} |")
    rows.append(f"| in-page renderer string | {r} | {cp.get('webgl_renderer')} |")
    rows.append(f"| painted | {_painted(obs)} | {_painted(obs, 'control')} |")

    rows.append("")
    for h in holders:
        for fi in h.get("fdinfo") or []:
            if fi:
                rows.append(f"amdgpu fdinfo, pid {h['pid']}: "
                            + ", ".join(f"{k}={v}" for k, v in sorted(fi.items())[:8]))
                break
    rows.append("")
    rows.append("The frame rate above is a timer reading, not a presentation reading: this X "
                "server has no vblank. It is recorded so the number is not mistaken later for "
                "evidence about the 60 -> 5 fps symptom, which nothing in this run touches.")
    return "\n".join(rows)


def _gpu_composited(obs: dict) -> bool:
    """The chain, from sources the engine does not author."""
    if _is_software(_renderer(obs)):
        return False
    if not _dri_holders(obs) or _gfx_engine_ns(obs) <= 0:
        return False
    if _software_driver_mapped(obs) and not _amd_driver_mapped(obs):
        return False
    return _control_collapsed(obs) is True or _amd_driver_mapped(obs)


def verdict(obs: dict) -> tuple[str, str]:
    r = _renderer(obs)
    holders = _dri_holders(obs)
    ns = _gfx_engine_ns(obs)
    delta = _gfx_delta_ns(obs)
    mapped = _mapped(obs) | _mapped_all(obs)
    collapsed = _control_collapsed(obs)

    # An in-page string that names a software rasteriser is still decisive
    # against, because nothing masks its way INTO llvmpipe.
    if _is_software(r):
        return "NOT_CAPABLE", (
            f"WebKit rendered through {r!r}, which is the CPU, so a perf number taken in it "
            f"would describe a software rasteriser")
    if _software_driver_mapped(obs) and not _amd_driver_mapped(obs) and ns <= 0:
        return "NOT_CAPABLE", (
            f"WebKit's process mapped a software rasteriser and no AMD driver, and amdgpu "
            f"attributed no engine time to it, so the engine is not reaching the GPU here")
    if not holders:
        return "NOT_CAPABLE", (
            "no WebKit process held a /dev/dri node open while an animating page was "
            "compositing, so nothing it drew went through the GPU")
    if ns <= 0:
        return "PARTIAL", (
            f"WebKit held the render node open, but amdgpu attributed no GFX engine time to "
            f"it, so the GPU path was opened and possibly never used")

    accrued = ("" if delta is None else
               f", {delta:,} ns of it while the page was animating")
    if collapsed is True:
        return "CAPABLE", (
            f"amdgpu attributed {ns:,} ns of GFX engine time{accrued} and {_vram_kib(obs):,} KiB "
            f"of VRAM to WebKit's own web process, which held "
            f"{sorted(set(holders[0]['dri_fds']))[0]} open while an animating page painted. The "
            f"same page in the same engine with mesa forced to software billed only "
            f"{_gfx_engine_ns(obs, 'control'):,} ns, so that GPU work was caused by the "
            f"browser's rendering and not by something incidental. The in-page renderer string "
            f"is {r!r}, which WebKit masks on every port and which is therefore not part of "
            f"this finding. This host composites WebKitGTK on the GPU")
    if _amd_driver_mapped(obs):
        return "CAPABLE", (
            f"amdgpu attributed {ns:,} ns of GFX engine time{accrued} and {_vram_kib(obs):,} KiB "
            f"of VRAM to WebKit's own process, which mapped "
            f"{sorted(m for m in mapped if any(t in m for t in AMD_DRIVER_LIBS))} and held "
            f"{sorted(set(holders[0]['dri_fds']))[0]} open while an animating page painted. The "
            f"in-page renderer string is {r!r}, which WebKit masks on every port and which is "
            f"therefore not part of this finding. This host composites WebKitGTK on the GPU")
    if collapsed is False:
        return "PARTIAL", (
            f"amdgpu attributed {ns:,} ns of GFX engine time to WebKit, but the forced-software "
            f"control leg billed a comparable {_gfx_engine_ns(obs, 'control'):,} ns, so the "
            f"counter is not tracking what the browser drew with and the reading does not "
            f"support a claim either way")
    return "PARTIAL", (
        f"amdgpu attributed {ns:,} ns of GFX engine time to WebKit's own process, which is real "
        f"GPU work, but no device-naming library was found mapped into it (mapped: "
        f"{sorted(mapped)[:12] or 'nothing recognised'}) and the negative control did not run, "
        f"so what did the drawing is corroborated by only one source")


def observed_capabilities(obs: dict) -> dict[str, bool]:
    return {
        "webkitgtk": bool(_leg(obs).get("gi")),
        "headless_display_server": bool((obs.get("xserver") or {}).get("display")),
        "gpu_browser_compositing": _gpu_composited(obs) and _painted(obs),
    }
