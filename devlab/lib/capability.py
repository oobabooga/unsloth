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
        if require_torch:
            p.torch = f"unavailable: {type(e).__name__}: {e}"

    try:
        import glob
        p.vulkan_icds = sorted(
            f.rsplit("/", 1)[-1] for f in glob.glob("/usr/share/vulkan/icd.d/*.json")
        )
    except Exception:  # noqa: BLE001
        pass

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
    }
    return p


def _has(attr: str) -> bool:
    try:
        import torch
        mod = getattr(torch, attr, None)
        return bool(mod and mod.is_available())
    except Exception:  # noqa: BLE001
        return False


def untested_section(profile: HostProfile, needed: list[str]) -> str:
    """Render the gaps between what a probe wanted and what the host is."""
    missing = [n for n in needed if not profile.capabilities.get(n, False)]
    if not missing:
        return ""
    lines = ["### Not tested here", ""]
    for name in missing:
        lines.append(f"- **{name}**: {KNOWN_GAPS.get(name, 'not available on this host')}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(detect().to_json())
