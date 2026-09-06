#!/usr/bin/env python3
"""Criteria: does an AMD GPU with no usable ROCm still get handed the CPU bundle?

The defect is a routing one. Every AMD branch in the llama.cpp selector was gated
on `has_rocm`, and the Vulkan branch additionally required `has_intel_gpu`, so an
AMD GPU whose ROCm is missing or unusable was indistinguishable from a headless
CPU box and took `bin-ubuntu-x64` -- the CPU bundle -- on a machine with a working
Vulkan device sitting idle.

So: base picks `linux-cpu`, head picks `linux-vulkan`, on a host that really is
AMD and really has no ROCm.

That host is reached with a container rather than an environment variable, and the
distinction is the whole reason this reading means anything. `HSA_OVERRIDE_GFX_VERSION`
and `UNSLOTH_ROCM_GFX_ARCH` are documented as things the probe must ignore, and
`ROCR_VISIBLE_DEVICES` is documented as SUPPRESSING the new route, so every variable
to hand either does nothing or fakes the answer in the wrong direction. Inside a
plain image the GPU and its `/sys/class/drm/card*/device/vendor` are the runner's
real silicon while rocminfo, amd-smi and /opt/rocm are genuinely absent.

Pairs with probes/llama_backend_routing_probe.py.
"""

from __future__ import annotations

TITLE = "llama.cpp bundle selection on an AMD GPU with no usable ROCm"
MODE = "differential"

# Authored, not derived: every capability this CHANGE touches, so the report says
# what it does not cover. The change also alters the ROCm torch index for gfx1033,
# the Intel Vulkan branch, the NVIDIA precedence rules, the macOS leaf and a
# glibc-floored CPU constraint; none of those are answerable from one gfx1151 box.
NEEDS = [
    "linux", "gpu", "vulkan", "rocm", "gfx1033_vangogh", "glibc_pre_228",
    "windows", "windows_docker", "xpu", "nvidia", "mlx", "discrete_gpu",
    "amdvlk", "multi_gpu",
]

_AMD_VENDOR = "0x1002"


def _is_amd(o: dict) -> bool:
    return any(v.endswith(_AMD_VENDOR) for v in o.get("drm_vendors", []))


def _host(o: dict) -> dict:
    return o.get("host") or {}


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out: list[tuple[str, bool, str]] = []
    for name in ("base", "head"):
        o = obs.get(name) or {}
        err = o.get("error") or o.get("plan_error")
        out.append((f"{name} probe produced a reading", not err and bool(o.get("host")),
                    err or "ok"))

        # An AMD vendor id in sysfs is the non-vacuity gate. Without it the whole
        # comparison is about a machine that is not the one the defect is about,
        # and both states would agree on the CPU bundle for the honest reason.
        out.append((f"{name} host really is AMD (sysfs {_AMD_VENDOR})", _is_amd(o),
                    ", ".join(o.get("drm_vendors", [])) or "no card*/device/vendor readable"))

        h = _host(o)
        tools = o.get("rocm_tools") or {}
        rocm_absent = (h.get("has_rocm") is False
                       and not tools.get("rocminfo_on_path")
                       and not tools.get("amd_smi_on_path")
                       and not tools.get("opt_rocm_rocminfo"))
        out.append((f"{name} ROCm genuinely absent, not masked", rocm_absent,
                    f"has_rocm={h.get('has_rocm')}, tools={tools}"))

        # Vulkan routing requires no PHYSICAL NVIDIA, so an NVIDIA card here would
        # make both states pick CPU for a reason that has nothing to do with AMD.
        out.append((f"{name} no NVIDIA in the way", h.get("has_physical_nvidia") is False,
                    f"has_physical_nvidia={h.get('has_physical_nvidia')}"))

        # And no Intel iGPU: the base already routes Intel to Vulkan, so an Intel
        # host would show the fix at BOTH states and the differential would be VOID
        # for the right reason but with a misleading table.
        out.append((f"{name} no Intel iGPU claiming the old branch", h.get("has_intel_gpu") is False,
                    f"has_intel_gpu={h.get('has_intel_gpu')}"))
    return out


def table(obs: dict) -> str:
    rows = ["| state | containerised | AMD in sysfs | has_rocm | has_intel_gpu "
            "| has_amd_gpu_without_rocm | first bundle | attempts |",
            "|---|---|---|---|---|---|---|---|"]
    for name in ("base", "head", "merge"):
        o = obs.get(name)
        if not o:
            continue
        h = _host(o)
        attempts = ", ".join(f"`{a['install_kind']}`" for a in o.get("attempts", [])) or "none"
        field = h.get("has_amd_gpu_without_rocm")
        shown = "field absent" if not o.get("has_amd_gpu_without_rocm_field_exists") else str(field)
        rows.append(
            f"| {name} | {o.get('containerised')} | {_is_amd(o)} | {h.get('has_rocm')} "
            f"| {h.get('has_intel_gpu')} | {shown} | `{o.get('first_install_kind')}` | {attempts} |"
        )
    return "\n".join(rows)


def base_shows_defect(base: dict) -> bool:
    # The CPU bundle on a machine with an AMD GPU and a Vulkan device is the defect.
    return base.get("first_install_kind") == "linux-cpu" and _is_amd(base)


def head_is_fixed(head: dict) -> bool:
    return head.get("first_install_kind") == "linux-vulkan" and _is_amd(head)
