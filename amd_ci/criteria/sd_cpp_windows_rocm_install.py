#!/usr/bin/env python3
"""Criteria for probes/sd_cpp_windows_rocm_install_probe.py.

Defect (base): on a Windows AMD host without the HIP SDK runtime, a ROCm request installs the
upstream win-rocm bundle over a working Vulkan build, sweeps the Vulkan binaries, and leaves only
binaries the loader refuses (exit 0xC0000135); a fresh ROCm ensure hands back such a binary.

Fixed (head): the ROCm install is refused before it touches the tree, the Vulkan build survives and
starts, a ROCm request on this host starts at Vulkan, and a fresh ROCm ensure ends on a Vulkan
server that starts.
"""

from __future__ import annotations

TITLE = "sd.cpp ROCm install on Windows AMD without the HIP SDK runtime (real prebuilts, real loader)"
MODE = "differential"
NEEDS = ["windows", "amd_smi"]


def _starts(entry) -> bool:
    return bool(entry and entry.get("exists") and entry.get("rc") is not None
                and not entry.get("dll_not_found"))


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out = []
    for name in ("base", "head"):
        o = obs.get(name) or {}
        out.append((f"{name}: probe ran to the end", not o.get("probe_error") and "ensure_server" in o,
                    o.get("probe_error") or "ok"))
        out.append((f"{name}: runs on Windows", o.get("platform") == "win32", str(o.get("platform"))))
        host = o.get("host") or {}
        out.append((f"{name}: hipblas.dll not reachable (HIP SDK absent or hidden)",
                    not host.get("hipblas_on_path_after_hide") and not host.get("hipblas_in_system32"),
                    f"hidden {host.get('hidden_path_entries')}, HIP_PATH={host.get('hip_path_env')}"))
        seeds = o.get("seed_starts") or []
        out.append((f"{name}: the seeded Vulkan build installed and starts",
                    not o.get("seed_error") and bool(seeds) and all(_starts(s) for s in seeds),
                    o.get("seed_error") or ", ".join(f"{s.get('path')}: {s.get('rc_hex')}" for s in seeds)))
    return out


def table(obs: dict) -> str:
    rows = ["| state | rocm install raised | Vulkan binaries left | binaries in tree (exit of --help) "
            "| preferred_accelerator(rocm) | fresh ensure(rocm) server | its exit |",
            "|---|---|---|---|---|---|---|"]
    for name in ("base", "head"):
        o = obs.get(name)
        if not o:
            continue
        ri = o.get("rocm_install") or {}
        left = "<br>".join(f"`{s.get('path', '').rsplit(chr(92), 1)[-1]}` {s.get('rc_hex')}"
                           for s in o.get("after_rocm_starts") or []) or "none"
        es = o.get("ensure_server_start") or {}
        rows.append(
            f"| {name} | {ri.get('raised') or 'no (installed)'} | {len(o.get('after_rocm_seed_binaries_left') or [])}"
            f"/{len(o.get('seed_binaries') or [])} | {left} | {o.get('preferred_for_rocm')} "
            f"| `{o.get('ensure_server')}` ({(o.get('ensure_record') or {}).get('accelerator')}) | {es.get('rc_hex')} |")
    head = obs.get("head") or {}
    msg = (head.get("rocm_install") or {}).get("message")
    if msg:
        rows.append("")
        rows.append(f"Head refusal: `{msg[:300]}`")
    return "\n".join(rows)


def base_shows_defect(base: dict) -> tuple[bool, str]:
    swept = not (base.get("after_rocm_seed_binaries_left") or [])
    raised = (base.get("rocm_install") or {}).get("raised")
    starts = base.get("after_rocm_starts") or []
    nothing_runs = bool(starts) and not any(_starts(s) for s in starts)
    refused = any(s.get("dll_not_found") for s in starts)
    ensure_bad = not _starts(base.get("ensure_server_start"))
    ok = raised is None and swept and nothing_runs and refused and ensure_bad
    return ok, (f"rocm install raised={raised}, Vulkan swept={swept}, nothing left starts={nothing_runs}, "
                f"0xC0000135 seen={refused}, fresh ensure server does not start={ensure_bad}")


def head_is_fixed(head: dict) -> tuple[bool, str]:
    raised = (head.get("rocm_install") or {}).get("raised")
    kept = head.get("after_rocm_seed_binaries_left") or []
    intact = bool(kept) and len(kept) == len(head.get("seed_binaries") or [])
    starts = head.get("after_rocm_starts") or []
    all_start = bool(starts) and all(_starts(s) for s in starts)
    pref = head.get("preferred_for_rocm")
    ensure_ok = _starts(head.get("ensure_server_start")) and \
        (head.get("ensure_record") or {}).get("accelerator") == "vulkan"
    ok = raised == "UnrunnableBuildError" and intact and all_start and pref == "vulkan" and ensure_ok
    return ok, (f"rocm install raised={raised}, Vulkan kept={intact}, every binary starts={all_start}, "
                f"preferred={pref}, fresh ensure on a starting Vulkan server={ensure_ok}")
