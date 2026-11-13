#!/usr/bin/env python3
"""Criteria: does the head report the NPU's load source on a real XDNA 2 host, where the base has none?

Windows: the NPU's DEVPKEY_GPU_LUID must be a real adapter LUID (listed under GPU Adapter Memory),
the head must resolve that same LUID, and npu_load_percent must answer (a number, or None only when
Windows lists no engine instance for an idle NPU) without blocking past the PowerShell timeouts.
Linux: no counter exists, so the head must answer None quickly. Pairs with probes/npu_monitor_probe.py.
"""

from __future__ import annotations

TITLE = "AMD NPU live monitor: detection, LUID, load counter"
MODE = "differential"
NEEDS = ["windows"]


def _supported(o: dict) -> bool:
    return bool((o.get("detect") or {}).get("supported"))


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out = []
    for name in ("base", "head"):
        o = obs.get(name) or {}
        out.append((f"{name} probe imported the checkout",
                    "hardware_file" in o and "detect" in o,
                    o.get("hardware_error") or o.get("detect_error") or o.get("hardware_file", "")))
        out.append((f"{name} host has a supported XDNA 2 NPU", _supported(o),
                    str(o.get("detect"))))
    head = obs.get("head") or {}
    if head.get("os") == "Windows":
        luid = head.get("pnp_luid")
        out.append(("NPU PnP node carries DEVPKEY_GPU_LUID", luid is not None,
                    f"instance={head.get('pnp_instance')} status={head.get('pnp_status')} "
                    f"luid={luid} raw={head.get('pnp')}"))
        out.append(("that LUID is a real adapter Windows lists",
                    luid is not None and luid in (head.get("adapter_luids") or []),
                    f"adapters={head.get('adapter_luids')}"))
    return out


def table(obs: dict) -> str:
    rows = ["| state | os | NPU | has npu_load_percent | calls (value, s) | resolved LUID | PnP LUID | engine instances |",
            "|---|---|---|---|---|---|---|---|"]
    for name in ("base", "head"):
        o = obs.get(name)
        if not o:
            continue
        d = o.get("detect") or {}
        calls = ", ".join(f"({c['value']}, {c['seconds']})" for c in o.get("npu_load_calls", [])) or "-"
        rows.append(f"| {name} | {o.get('os')} | {d.get('name')} {d.get('driver_version') or ''} "
                    f"| {o.get('has_npu_load_percent')} | {calls} | {o.get('npu_luid_cached')} "
                    f"| {o.get('pnp_luid', '-')} | {len(o.get('npu_engine_instances') or [])} |")
    return "\n".join(rows)


def base_shows_defect(base: dict) -> bool:
    return base.get("has_npu_load_percent") is False


def head_is_fixed(head: dict) -> bool:
    calls = head.get("npu_load_calls") or []
    if len(calls) != 3 or any(c["seconds"] > 12 for c in calls):
        return False
    # Later calls inside the TTL are served from the cache.
    if any(c["seconds"] > 0.5 for c in calls[1:]):
        return False
    values = [c["value"] for c in calls]
    if head.get("os") != "Windows":
        return all(v is None for v in values)
    luid = head.get("pnp_luid")
    if luid is None or head.get("npu_luid_cached") != luid:
        return False
    if head.get("npu_engine_instances"):
        return all(isinstance(v, float) and 0.0 <= v <= 100.0 for v in values)
    return all(v is None for v in values)
