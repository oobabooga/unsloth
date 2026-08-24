#!/usr/bin/env python3
"""What this host can and cannot answer.

Every report this toolkit produces ends with a "Not tested here" section, built
from here rather than from whoever wrote the probe remembering. Two of four PR
comments in the session that motivated this toolkit would have overstated their
reach without one, because the interesting paths were Windows-only or NVIDIA-only
and nothing in the run says so on its own.

A capability is a claim about the HOST, not about a probe. Probes declare which
capabilities they need; anything they need that the host lacks is reported as
untested instead of silently passing.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class Capability:
    name: str
    present: bool
    detail: str = ""


# Things a probe may ask for. The reason strings are what end up in a report, so
# they say why the host cannot answer, not merely that it cannot.
KNOWN_GAPS = {
    "windows": "needs a Windows host; this runner is Linux",
    "windows_rocm_wddm": (
        "needs Windows ROCm. `rocm_windows_free_is_untrusted()` is "
        "`sys.platform == 'win32' and IS_ROCM`, so on Linux the WDDM free cap is "
        "never applied and `trusted_mem_get_info` is a pure passthrough"
    ),
    "multi_gpu": "needs at least two GPUs; this runner has one",
    "multi_gpu_amd": "needs at least two AMD GPUs for HIP to amd-smi remapping",
    "nvidia": "needs an NVIDIA GPU; this runner is AMD",
    "mig": "needs a MIG-capable NVIDIA GPU",
    "gpu_partitions": "needs a partitionable accelerator (MI300 class)",
    "xpu": "needs an Intel GPU",
    "mlx": "needs Apple Silicon",
    "discrete_gpu": "needs a discrete GPU; this runner is a unified-memory APU",
    "amdvlk": "needs the proprietary AMD Vulkan driver; this runner has RADV (mesa)",
    "exfat_or_smb": "needs a filesystem that refuses symlinks",
    "no_symlink_privilege": "needs Windows outside developer mode",
    # Display / render. A compute GPU is not a rendering one: an APU can be
    # present for HIP and still expose no openable render node, no compositor
    # and no GPU-accelerated browser stack on a headless CI host.
    "drm_render_node": (
        "needs a /dev/dri/renderD* the runner user can open; ROCm reaching the GPU "
        "through /dev/kfd says nothing about this"
    ),
    "egl_hardware_gl": (
        "needs an EGL context whose GL renderer is a real device rather than llvmpipe. "
        "Established by making a context and painting, never by the presence of mesa"
    ),
    "vulkan_hardware": (
        "needs a Vulkan physical device that is not lavapipe. Read from an enumerated "
        "device, not from an ICD filename and not from a ROCm banner"
    ),
    "webkitgtk": (
        "needs libwebkit2gtk-4.1, the engine Unsloth Studio and Desktop actually render "
        "with; headless Chromium is a different compositor and a different main thread"
    ),
    "headless_display_server": (
        "needs Xvfb, Xwayland or a headless Wayland compositor to host a GTK window"
    ),
    "gpu_browser_compositing": (
        "needs a browser engine observed compositing on the GPU; library presence and a "
        "hardware GL context are both necessary and neither is sufficient"
    ),
}


# Capabilities no amount of looking can settle: whether a GL context is hardware
# is only knowable by making one. They default False and a probe overlays what it
# established, via `detect(observed = ...)`. Defaulting them True would let a
# report claim rendering it never did.
PROBE_ESTABLISHED = frozenset({
    "egl_hardware_gl", "vulkan_hardware", "gpu_browser_compositing",
})


# Capabilities that are read out of torch. If torch is missing they are all
# False, which is indistinguishable from the hardware genuinely lacking them.
TORCH_DERIVED = frozenset({
    "rocm", "nvidia", "gpu", "multi_gpu", "multi_gpu_amd", "integrated_gpu",
    "discrete_gpu", "windows_rocm_wddm", "mig", "gpu_partitions", "xpu",
})


@dataclass
class HostProfile:
    system: str = ""
    machine: str = ""
    python: str = ""
    torch: str | None = None
    hip: str | None = None
    cuda: str | None = None
    gpu_count: int = 0
    gpu_names: list[str] = field(default_factory = list)
    gpu_archs: list[str] = field(default_factory = list)
    is_integrated: list[Any] = field(default_factory = list)
    vulkan_icds: list[str] = field(default_factory = list)
    render_nodes: list[str] = field(default_factory = list)
    capabilities: dict[str, bool] = field(default_factory = dict)
    # Capabilities a probe established rather than detection guessed, so a reader
    # can tell a measured True from a default one.
    observed_overlay: list[str] = field(default_factory = list)
    # False when torch could not be imported, so every GPU-derived capability
    # below is a detection failure and not an observation about the hardware.
    torch_detection: bool = True

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent = 2)


def detect(require_torch: bool = True, observed: dict[str, bool] | None = None) -> HostProfile:
    p = HostProfile(
        system = platform.system(),
        machine = platform.machine(),
        python = platform.python_version(),
    )
    try:
        import torch
        p.torch = torch.__version__
        p.hip = torch.version.hip
        p.cuda = torch.version.cuda
        if torch.cuda.is_available():
            p.gpu_count = torch.cuda.device_count()
            for i in range(p.gpu_count):
                props = torch.cuda.get_device_properties(i)
                p.gpu_names.append(props.name)
                p.gpu_archs.append(getattr(props, "gcnArchName", "") or "")
                p.is_integrated.append(getattr(props, "is_integrated", None))
    except Exception as e:  # noqa: BLE001
        p.torch_detection = False
        if require_torch:
            p.torch = f"unavailable: {type(e).__name__}: {e}"

    try:
        import glob
        p.vulkan_icds = sorted(
            f.rsplit("/", 1)[-1] for f in glob.glob("/usr/share/vulkan/icd.d/*.json")
        )
    except Exception:  # noqa: BLE001
        pass

    p.render_nodes = _openable_render_nodes()

    rocm = bool(p.hip)
    nvidia = bool(p.cuda) and not rocm
    integrated = any(bool(x) for x in p.is_integrated)
    p.capabilities = {
        "linux": p.system == "Linux",
        "windows": p.system == "Windows",
        "rocm": rocm,
        "nvidia": nvidia,
        "gpu": p.gpu_count > 0,
        "multi_gpu": p.gpu_count > 1,
        "multi_gpu_amd": rocm and p.gpu_count > 1,
        "integrated_gpu": integrated,
        "discrete_gpu": p.gpu_count > 0 and not integrated,
        "windows_rocm_wddm": rocm and p.system == "Windows",
        "vulkan": any("radeon" in i or "amd" in i for i in p.vulkan_icds),
        "amdvlk": False,
        "mig": False,
        "gpu_partitions": False,
        "xpu": _has("xpu"),
        "mlx": p.system == "Darwin" and p.machine == "arm64",
        "exfat_or_smb": False,
        "no_symlink_privilege": False,
        "amd_smi": shutil.which("amd-smi") is not None,
        "rocm_smi": shutil.which("rocm-smi") is not None,
        # Display / render, detected statically. Each is necessary for a browser
        # venue and none is sufficient; the sufficient ones are in
        # PROBE_ESTABLISHED and stay False until a probe says otherwise.
        "drm_render_node": bool(p.render_nodes),
        "webkitgtk": bool(_find_lib("libwebkit2gtk-4.1.so*")),
        "headless_display_server": (
            bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
            or any(shutil.which(x) for x in ("Xvfb", "Xwayland", "weston", "cage"))
        ),
        "egl_hardware_gl": False,
        "vulkan_hardware": False,
        "gpu_browser_compositing": False,
    }

    if observed:
        # A probe outranks detection, because it measured rather than inferred.
        unknown = sorted(k for k in observed if k not in p.capabilities)
        if unknown:
            raise ValueError(f"observed capabilities not known to capability.py: {unknown}. "
                             f"Add them with a KNOWN_GAPS reason, or the report will state a "
                             f"gap it cannot explain.")
        p.capabilities.update({k: bool(v) for k, v in observed.items()})
        p.observed_overlay = sorted(observed)
    return p


_LIB_DIRS = ("/usr/lib/x86_64-linux-gnu", "/usr/lib64", "/usr/lib", "/lib/x86_64-linux-gnu",
             "/usr/local/lib/x86_64-linux-gnu", "/usr/local/lib")


def _find_lib(pattern: str) -> list[str]:
    import glob
    hits: list[str] = []
    for d in _LIB_DIRS:
        hits += glob.glob(f"{d}/{pattern}")
    return sorted(set(hits))


def _openable_render_nodes() -> list[str]:
    """Nodes the runner user can really open, not ones that merely exist.

    `os.access` and a mode bit both said yes on a host where the open failed,
    because the user was not in the owning group.
    """
    import glob
    ok: list[str] = []
    for path in sorted(glob.glob("/dev/dri/renderD*")):
        try:
            fd = os.open(path, os.O_RDWR)
            os.close(fd)
            ok.append(path)
        except OSError:
            continue
    return ok


def _has(attr: str) -> bool:
    try:
        import torch
        mod = getattr(torch, attr, None)
        return bool(mod and mod.is_available())
    except Exception:  # noqa: BLE001
        return False


def untested_section(profile: HostProfile, needed: list[str]) -> str:
    """Render the gaps between what a probe wanted and what the host is.

    Never returns empty. `needed` is authored by the criteria module, so an
    under-declared NEEDS produces no gaps, and a section that simply vanished
    read as "this run was unbounded" on exactly the reports that were most
    bounded. The empty case is therefore rendered as the claim it actually is,
    with NEEDS shown so a reader can see what was asserted and disagree.
    """
    missing = [n for n in needed if not profile.capabilities.get(n, False)]
    if missing:
        lines = ["### Not tested here", ""]
        for name in missing:
            if name in TORCH_DERIVED and not profile.torch_detection:
                # Saying "this host has no ROCm" about a machine that plainly does,
                # because torch failed to import, is a false claim rather than a gap.
                lines.append(f"- **{name}**: UNDETERMINED. torch could not be imported in the "
                             f"reporting environment, so this was never measured. Not a "
                             f"statement about the hardware.")
            else:
                lines.append(f"- **{name}**: {KNOWN_GAPS.get(name, 'not available on this host')}")
        return "\n".join(lines)

    if not needed:
        return ("### Not tested here\n\n"
                "- The criteria module declares no `NEEDS`, so no bounds could be computed. "
                "This is not the same as a result that holds everywhere: treat its reach as "
                "unknown until `NEEDS` is filled in.")
    return ("### Not tested here\n\n"
            f"- Nothing. This host satisfied every capability the criteria declared "
            f"(`NEEDS = {needed}`). That is a claim about the DECLARATION, not proof the "
            f"change was exercised everywhere it matters; check `NEEDS` covers the change.")


if __name__ == "__main__":
    print(detect().to_json())
