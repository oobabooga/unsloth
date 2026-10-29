#!/usr/bin/env python3
"""Criteria for probes/hip_sdk_not_on_path_probe.py (PR 15853).

Defect (base): a Windows AMD host with a HIP SDK 7.x installed whose bin dir is not on PATH
(HIP_PATH set) gets the Vulkan sd.cpp build for a ROCm request, and a forced ROCm request does not
end on a ROCm build that starts.

Fixed (head): the same host picks ROCm, installs the win-rocm build, and that build STARTS through
the head's runtime_env and lists the ROCm0 device; a forced request likewise; the SDK is also found
by the %ProgramFiles%\\AMD\\ROCm scan; with no SDK, or only a 6.x one, ROCm requests still start at
Vulkan.
"""

from __future__ import annotations

TITLE = "sd.cpp on Windows AMD with the HIP SDK installed but not on PATH (real prebuilts, real loader)"
MODE = "differential"
NEEDS = ["windows", "amd_smi"]


def _c(o: dict, case: str) -> dict:
    return ((o or {}).get("cases") or {}).get(case) or {}


def _ok_start(s) -> bool:
    return bool(s and s.get("exists") and s.get("rc") == 0 and not s.get("dll_not_found"))


def _rocm_runs(c: dict) -> bool:
    return (c.get("preferred_for_rocm") == "rocm" and c.get("record_accelerator") == "rocm"
            and _ok_start(c.get("start_runtime_env_version"))
            and bool((c.get("start_runtime_env_list_devices") or {}).get("rocm_device_listed")))


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out = []
    for name in ("base", "head"):
        o = obs.get(name) or {}
        sdk = o.get("sdk") or {}
        out.append((f"{name}: probe ran every case", not o.get("probe_error") and
                    all(not _c(o, k).get("probe_error") and "preferred_for_rocm" in _c(o, k)
                        for k in ("hip_path", "hip_path_force", "program_files", "no_sdk", "sdk6_only")),
                    o.get("probe_error") or "; ".join(f"{k}: {v.get('probe_error')}" for k, v in
                                                      (o.get("cases") or {}).items() if v.get("probe_error")) or "ok"))
        out.append((f"{name}: runs on Windows", o.get("platform") == "win32", str(o.get("platform"))))
        has = sdk.get("bin_has") or {}
        out.append((f"{name}: a HIP SDK 7.x bin dir exists ({sdk.get('kind')})",
                    bool(has.get("hipblas.dll") and has.get("amdhip64_7.dll") and has.get("rocblas.dll")),
                    f"{sdk.get('root')}: {has}"))
        ns0 = _c(o, "no_sdk")
        out.append((f"{name}: hipblas.dll unreachable via PATH / System32 (no-SDK case reports it missing)",
                    "hipblas.dll" in (ns0.get("hip_missing_dlls") or []),
                    f"missing={ns0.get('hip_missing_dlls')}, hidden {o.get('hidden_path_entries')}"))
        ns = _c(o, "no_sdk")
        out.append((f"{name}: control, no SDK: Vulkan build installed and starts",
                    ns.get("record_accelerator") == "vulkan" and _ok_start(ns.get("start_runtime_env_version")),
                    f"record={ns.get('record_accelerator')}, start={(ns.get('start_runtime_env_version') or {}).get('rc_hex')}"))
    return out


def table(obs: dict) -> str:
    rows = ["| state | case | HIP_PATH | missing DLLs | preferred(rocm) | installed | --version via runtime_env "
            "| --list-devices ROCm0 | --version PATH only | ROCm failure recorded |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for name in ("base", "head"):
        o = obs.get(name) or {}
        for case, c in (o.get("cases") or {}).items():
            rows.append(
                f"| {name} | {case} | `{(c.get('env') or {}).get('HIP_PATH')}` | {c.get('hip_missing_dlls')} "
                f"| {c.get('preferred_for_rocm')} | {c.get('record_accelerator')} "
                f"| {(c.get('start_runtime_env_version') or {}).get('rc_hex')} "
                f"| {(c.get('start_runtime_env_list_devices') or {}).get('rocm_device_listed')} "
                f"| {(c.get('start_path_only_version') or {}).get('rc_hex')} | {c.get('rocm_failure_recorded')} |")
    h = obs.get("head") or {}
    rows.append("")
    rows.append(f"SDK: {(h.get('sdk') or {}).get('kind')} at `{(h.get('sdk') or {}).get('root')}`; "
                f"host facts: `{h.get('host')}`")
    for case in ("hip_path", "hip_path_force"):
        c = _c(h, case)
        rows.append(f"head {case}: bundle DLLs {c.get('bundle_dlls')}, runtime_env PATH tail "
                    f"`{c.get('runtime_env_path_tail')}`, listed {c.get('listed_device')}, list-devices tail "
                    f"`{((c.get('start_runtime_env_list_devices') or {}).get('output_tail') or '')[-300:]}`")
    return "\n".join(rows)


def base_shows_defect(base: dict) -> tuple[bool, str]:
    hp, hf = _c(base, "hip_path"), _c(base, "hip_path_force")
    vulkan = hp.get("preferred_for_rocm") == "vulkan" and hp.get("record_accelerator") == "vulkan"
    forced_fails = not _rocm_runs(hf)
    return vulkan and forced_fails, (
        f"HIP_PATH only: preferred={hp.get('preferred_for_rocm')}, installed={hp.get('record_accelerator')}; "
        f"forced: installed={hf.get('record_accelerator')}, ROCm failure recorded={hf.get('rocm_failure_recorded')}, "
        f"rocm build runs={_rocm_runs(hf)}")


def head_is_fixed(head: dict) -> tuple[bool, str]:
    hp, hf = _c(head, "hip_path"), _c(head, "hip_path_force")
    pf, ns, s6 = _c(head, "program_files"), _c(head, "no_sdk"), _c(head, "sdk6_only")
    control = bool((hp.get("start_path_only_version") or {}).get("dll_not_found"))
    checks = {
        "HIP_PATH: ROCm chosen, installed, starts, lists ROCm0": _rocm_runs(hp),
        "runtime_env carries the SDK bin": bool(hp.get("runtime_env_has_sdk_bin")),
        "same build without SDK dirs is refused 0xC0000135 (non-vacuity)": control,
        "forced: ROCm runs": _rocm_runs(hf),
        "ProgramFiles scan: rocm": pf.get("preferred_for_rocm") == "rocm",
        "no SDK: vulkan": ns.get("preferred_for_rocm") == "vulkan" and ns.get("record_accelerator") == "vulkan",
        "6.x-only HIP_PATH: vulkan": s6.get("preferred_for_rocm") == "vulkan",
    }
    return all(checks.values()), ", ".join(f"{k}={v}" for k, v in checks.items())
