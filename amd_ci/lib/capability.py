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
    "multi_gpu_spoofed": (
        "the real GPU wearing other numbers. Shared memory, shared compute, no "
        "parallelism. Wiring only, never sharding, throughput or collectives"
    ),
    "nvidia": "needs an NVIDIA GPU; this runner is AMD",
    "mig": "needs a MIG-capable NVIDIA GPU",
    "gpu_partitions": "needs a partitionable accelerator (MI300 class)",
    "xpu": "needs an Intel GPU",
    "mlx": "needs Apple Silicon",
    "discrete_gpu": "needs a discrete GPU; this runner is a unified-memory APU",
    "amdvlk": "needs the proprietary AMD Vulkan driver; this runner has RADV (mesa)",
    "exfat_or_smb": "needs a filesystem that refuses symlinks",
    "no_symlink_privilege": "needs Windows outside developer mode",
}


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
    capabilities: dict[str, bool] = field(default_factory = dict)
    # False when torch could not be imported, so every GPU-derived capability
    # below is a detection failure and not an observation about the hardware.
    torch_detection: bool = True
    # How many of gpu_count are fabricated by the HIP device multiplier. Non-zero
    # means torch's device list is not the hardware, and multi_gpu is forced False
    # below no matter what torch reported.
    spoofed_devices: int = 0

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent = 2)


def detect(require_torch: bool = True) -> HostProfile:
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

    p.spoofed_devices = detect_spoofed_devices()
    p.capabilities = capabilities_for(p)
    return p


def detect_spoofed_devices() -> int:
    """How many of torch's devices are fabricated by the HIP device multiplier."""
    try:
        from . import device_multiplier as _dm
    except Exception:  # noqa: BLE001
        try:
            import device_multiplier as _dm  # type: ignore[no-redef]
        except Exception:  # noqa: BLE001
            return 0
    return _dm.spoofed_count()


def capabilities_for(p: "HostProfile") -> dict[str, bool]:
    """The capability rules, as a pure function of an observed profile.

    Split out of detect() so the spoof rule below is exercised directly by selftest.
    A test that rebuilds this dict by hand tests its own arithmetic and passes
    happily while the shipped rule is broken; that happened once.
    """
    # A spoofed device must never satisfy multi_gpu. detect() reads
    # torch.cuda.device_count(), and under the HIP device multiplier that is 2 on a
    # one-GPU box -- which would drop the multi-GPU line from "Not tested here" and
    # turn a wiring run into something that reads as hardware validation. The real
    # count is what the capabilities are computed from; see lib/device_multiplier.py.
    real_gpu_count = max(0, p.gpu_count - p.spoofed_devices)

    rocm = bool(p.hip)
    nvidia = bool(p.cuda) and not rocm
    integrated = any(bool(x) for x in p.is_integrated)
    return {
        "linux": p.system == "Linux",
        "windows": p.system == "Windows",
        "rocm": rocm,
        "nvidia": nvidia,
        "gpu": real_gpu_count > 0,
        "multi_gpu": real_gpu_count > 1,
        "multi_gpu_amd": rocm and real_gpu_count > 1,
        "integrated_gpu": integrated,
        "discrete_gpu": real_gpu_count > 0 and not integrated,
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
    }


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

    # Stated unconditionally when active: a reader must not have to notice that
    # multi_gpu happens to be in NEEDS to learn the device list was fabricated.
    spoof: list[str] = []
    if profile.spoofed_devices:
        n = profile.spoofed_devices
        spoof = ["", f"- **{n} of the {profile.gpu_count} devices torch reported "
                     f"{'is' if n == 1 else 'are'} FABRICATED** by the HIP device "
                     f"multiplier: {KNOWN_GAPS['multi_gpu_spoofed']}."]

    if missing or spoof:
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
        return "\n".join(lines + spoof)

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
