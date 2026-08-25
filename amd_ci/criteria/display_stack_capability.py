#!/usr/bin/env python3
"""Criteria: can this host be a REAL GPU-rendered browser venue?

Judges only. Every fact it uses comes from probes/display_stack_probe.py.

This is a CAPABILITY question, not a differential one: there is no defect and no
base/head pair, so the VOID rule in lib/differential.py does not apply and must
not be simulated by pointing two states at the same tree. What does carry over
is the discipline that makes a verdict worth reading, so:

  * A NO is a finding and keeps the job green. Only a failed gate, meaning we
    could not tell, is a non-result.
  * The decisive fact is the GL renderer string from a context that was made
    current and then PAINTED, never the presence of a driver, an ICD file or a
    ROCm banner. `system_info` naming ROCm is exactly the signal that has
    already made a detector call working Vulkan runs "cpu-only", and the mirror
    of that mistake here would be calling llvmpipe a GPU because mesa is
    installed and /dev/dri exists.
  * Software rasterisation is the answer this criteria is most at risk of
    mislabelling, so the software renderer names are matched explicitly and
    selftest.py feeds a llvmpipe string through to prove the answer flips.
"""

from __future__ import annotations

TITLE = "Can the AMD CI runner host a GPU-rendered WebKitGTK venue?"
MODE = "capability"

# What STANDING UP THE VENUE touches, not what this host happens to have. The
# last four are here because the reported 60 -> 5 fps collapse is a user-desktop
# symptom, and a Linux APU under RADV cannot speak for a Windows WebView2 build,
# an NVIDIA laptop, a discrete card or a Mac.
NEEDS = [
    "linux", "drm_render_node", "egl_hardware_gl", "vulkan_hardware",
    "webkitgtk", "headless_display_server", "gpu_browser_compositing",
    "discrete_gpu", "nvidia", "windows", "mlx",
]

# Renderer strings that mean the CPU drew it. Matched case-insensitively.
SOFTWARE_RENDERERS = ("llvmpipe", "softpipe", "swrast", "mesa offscreen",
                      "lavapipe", "swiftshader", "zink on llvmpipe")

PAINT_EXPECTED = [0, 255, 0, 255]


# --------------------------------------------------------------------------
# reading the observations


def _is_software(renderer: str | None) -> bool:
    return bool(renderer) and any(s in renderer.lower() for s in SOFTWARE_RENDERERS)


def _egl_entries(obs: dict) -> dict:
    return (obs.get("egl") or {})


def _painted(e: dict) -> bool:
    return e.get("painted_pixel_rgba") == PAINT_EXPECTED and e.get("gl_error_after_paint") == 0


def _hardware_gl(e: dict) -> bool:
    """A context that was made current, named a non-software renderer, and drew."""
    return bool(e.get("made_current")) and bool(e.get("gl_renderer")) \
        and not _is_software(e.get("gl_renderer")) and _painted(e)


def _best_gl(obs: dict) -> tuple[str | None, dict]:
    """Prefer a context pinned to a render node over one that chose for itself."""
    entries = _egl_entries(obs)
    for key in sorted(entries):
        if key.startswith("gbm:") and _hardware_gl(entries[key]):
            return key, entries[key]
    for key in sorted(entries):
        if _hardware_gl(entries[key]):
            return key, entries[key]
    for key in sorted(entries):
        if entries[key].get("gl_renderer"):
            return key, entries[key]
    return (None, {})


def _openable_render_nodes(obs: dict) -> list[str]:
    return [n["path"] for n in ((obs.get("dri") or {}).get("nodes") or [])
            if n.get("open_rdwr") and "renderD" in n.get("path", "")]


def _vulkan_hardware(obs: dict) -> list[dict]:
    devs = (obs.get("vulkan") or {}).get("devices") or []
    return [d for d in devs
            if d.get("type") in ("INTEGRATED_GPU", "DISCRETE_GPU")
            and not _is_software(d.get("name"))]


def _webkit_present(obs: dict) -> list[str]:
    return (obs.get("webkit") or {}).get("libwebkit2gtk_4_1") or []


def _apt_uris(obs: dict) -> int:
    r = (obs.get("webkit") or {}).get("apt_print_uris") or {}
    if r.get("rc") != 0:
        return 0
    return sum(1 for ln in (r.get("stdout") or "").splitlines() if ln.startswith("'"))


def _display_server(obs: dict) -> list[str]:
    d = obs.get("display_servers") or {}
    return [t for t in ("Xvfb", "Xwayland", "weston", "cage", "sway", "wayfire", "labwc")
            if d.get(t)]


def _amd_evidence(obs: dict) -> str:
    bits: list[str] = []
    rocm = (obs.get("rocminfo") or {}).get("stdout") or ""
    for line in rocm.splitlines():
        if "gfx" in line and "Name:" in line:
            bits.append(line.strip())
            break
    for line in (obs.get("pci") or {}).get("display_controllers") or []:
        if any(v in line for v in ("AMD", "ATI", "Advanced Micro")):
            bits.append(line.strip())
            break
    for e in (obs.get("drm_sysfs") or {}).get("entries") or []:
        if e.get("vendor") == "0x1002":
            bits.append(f"{e['name']} vendor 0x1002 driver={e.get('driver')}")
            break
    return "; ".join(bits[:3])


# --------------------------------------------------------------------------
# gates: a failed gate means we could not tell, which is not the same as no


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out: list[tuple[str, bool, str]] = []

    ran = obs.get("_probe_rc") == 0 and isinstance(obs.get("dri"), dict)
    out.append(("the probe completed and produced observations", ran,
                f"rc={obs.get('_probe_rc')}"))

    amd = _amd_evidence(obs)
    out.append(("this is the AMD host the question is about", bool(amd),
                amd or "no gfx target, no AMD display controller, no 0x1002 DRM device"))

    entries = _egl_entries(obs)
    completed = [k for k, v in entries.items() if v.get("child_rc") == 0]
    out.append(("at least one EGL attempt ran to completion", bool(completed),
                f"{len(completed)}/{len(entries)} attempts; "
                f"crashed: {[k for k, v in entries.items() if v.get('crashed_with_signal')]}"))
    return out


# --------------------------------------------------------------------------


def table(obs: dict) -> str:
    nodes = (obs.get("dri") or {}).get("nodes") or []
    open_nodes = _openable_render_nodes(obs)
    key, best = _best_gl(obs)
    vk_hw = _vulkan_hardware(obs)
    vk_all = (obs.get("vulkan") or {}).get("devices") or []
    wk = _webkit_present(obs)
    srv = _display_server(obs)

    rows = ["| # | question | observation |", "|---|---|---|"]
    rows.append(f"| 1 | /dev/dri nodes, openable by the runner user | "
                f"{len([n for n in nodes if 'renderD' in n.get('path','') or 'card' in n.get('path','')])} "
                f"node(s); openable render node(s): {open_nodes or 'NONE'} |")
    rows.append(f"| 2 | display controller on the PCI bus | "
                f"{'; '.join((obs.get('pci') or {}).get('display_controllers') or []) or 'none reported'} |")
    rows.append(f"| 3 | EGL context without X or Wayland | "
                f"{'made current via ' + key if best.get('made_current') else 'no context'} |")
    vk_names = ", ".join("{} ({})".format(d.get("name"), d.get("type")) for d in vk_all)
    rows.append(f"| 4 | Vulkan physical devices | "
                f"{vk_names or (obs.get('vulkan') or {}).get('error', 'none')} |")
    rows.append(f"| 5 | **GL renderer string** | "
                f"**{best.get('gl_renderer') or 'none'}** "
                f"({'SOFTWARE' if _is_software(best.get('gl_renderer')) else 'hardware' if best.get('gl_renderer') else 'n/a'}); "
                f"paint {best.get('painted_pixel_rgba')} |")
    rows.append(f"| 6 | libwebkit2gtk-4.1 | "
                f"{wk[0] if wk else 'absent'}; apt download URIs available: {_apt_uris(obs)} |")
    rows.append(f"| 7 | something that paints | "
                f"GL clear+readback {'succeeded' if _painted(best) else 'did not succeed'}; "
                f"headless display servers: {srv or 'none'} |")
    rows.append("")
    rows.append(f"dmabuf import extensions (WebKit's accelerated compositing path): "
                f"`EGL_EXT_image_dma_buf_import`={best.get('has_image_dmabuf_import')}, "
                f"modifiers={best.get('has_image_dmabuf_import_modifiers')}")
    rows.append("")
    rows.append(f"Vulkan devices that are not software: {[d['name'] for d in vk_hw] or 'none'}. "
                f"Read from an enumerated physical device, not from an ICD filename and not "
                f"from a ROCm banner.")
    return "\n".join(rows)


def verdict(obs: dict) -> tuple[str, str]:
    key, best = _best_gl(obs)
    gl_hw = _hardware_gl(best)
    renderer = best.get("gl_renderer")
    wk = bool(_webkit_present(obs))
    wk_installable = _apt_uris(obs) > 0 and bool((obs.get("webkit") or {}).get("have_dpkg_deb"))

    if not gl_hw:
        if renderer and _is_software(renderer):
            return "NOT_CAPABLE", (
                f"the only GL context obtainable here renders through {renderer!r}, which is "
                f"the CPU. A browser hosted on this host would be measured against a software "
                f"rasteriser, so it cannot speak to a GPU compositing symptom")
        if not _openable_render_nodes(obs):
            return "NOT_CAPABLE", (
                "no /dev/dri render node can be opened by the runner user, so no GPU context "
                "exists to render into")
        return "NOT_CAPABLE", (
            f"no GL context was both made current and able to paint "
            f"(renderer={renderer!r}, error={best.get('error')!r})")

    if wk or wk_installable:
        return "CAPABLE", (
            f"a GPU context on {key} renders through {renderer!r} and painted correctly, and "
            f"WebKitGTK is {'installed' if wk else 'fetchable without root'}. Standing the "
            f"venue up is now an engineering task, not a hardware question")
    return "PARTIAL", (
        f"the GPU half is real: {key} renders through {renderer!r} and painted. The browser "
        f"half is not: libwebkit2gtk-4.1 is absent and could not be fetched without root, so "
        f"nothing on this host reproduces the WebKitGTK path Studio actually uses")


def observed_capabilities(obs: dict) -> dict[str, bool]:
    """Facts this probe ESTABLISHED, overlaid on the static host profile.

    lib/capability.py cannot detect these by looking: whether a GL context is
    hardware is only knowable by making one. Without this overlay the report
    would list `egl_hardware_gl` as a gap on a host that just proved it.
    """
    _, best = _best_gl(obs)
    return {
        "egl_hardware_gl": _hardware_gl(best),
        "vulkan_hardware": bool(_vulkan_hardware(obs)),
        "drm_render_node": bool(_openable_render_nodes(obs)),
        "webkitgtk": bool(_webkit_present(obs)),
        "headless_display_server": bool(_display_server(obs)),
        # Only a probe that actually composites a page may set this. Nothing in
        # this run does, so it stays False and is reported as an open gap.
        "gpu_browser_compositing": False,
    }
