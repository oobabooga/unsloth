# SPDX-License-Identifier: AGPL-3.0-only
import re

TITLE = "Real Vulkan decode after an injected fitter-process failure"
MODE = "differential"
NEEDS = ["windows", "vulkan", "nvidia", "rocm", "multi_gpu", "mlx"]


def gpu_decode(value):
    lines = "\n".join(value.get("gpu_evidence") or [])
    offloaded = any(int(n) > 0 for n in re.findall(r"offloaded (\d+)/", lines))
    return (value.get("ready") and offloaded
            and (value.get("completion") or {}).get("tokens_predicted") == 8)


def recovered(value):
    data = value.get("backend") or {}
    launches = data.get("launches") or []
    return bool(data.get("loaded") and not data.get("cpu_fallback_reason")
                and data.get("fault_injected")
                and (data.get("completion") or {}).get("tokens_predicted") == 8
                and len(launches) >= 2 and "off" in launches[1])


def gates(observations):
    checks = []
    for name, value in observations.items():
        if name.startswith("_"):
            continue
        native = value.get("native") or {}
        checks.append((f"{name}: actual AMD Vulkan device", "Vulkan" in value.get("devices", "")
                       and "AMD" in value.get("devices", ""), value.get("devices", "")[:500]))
        for mode in ("on", "off"):
            checks.append((f"{name}: native fit-{mode} GPU decode", bool(gpu_decode(native.get(mode) or {})),
                           str(native.get(mode, {}))[:600]))
        launches = (value.get("backend") or {}).get("launches") or []
        checks.append((f"{name}: injected failing launch reached",
                       bool((value.get("backend") or {}).get("fault_injected")), str(launches[:1])[:600]))
    return checks


def base_shows_defect(value):
    return bool((value.get("backend") or {}).get("launches")) and not recovered(value)


def head_is_fixed(value):
    return recovered(value)


def table(observations):
    lines = ["| state | native fit-on GPU decode | native fit-off GPU decode | recovery GPU decode |",
             "|---|---|---|---|"]
    for name, value in observations.items():
        if not name.startswith("_"):
            native = value.get("native") or {}
            lines.append(f"| {name} | {bool(gpu_decode(native.get('on') or {}))} | "
                         f"{bool(gpu_decode(native.get('off') or {}))} | {recovered(value)} |")
    lines.append("\nThe first child failure is injected. Native RX 5700 XT driver failure is not reproduced by this check.")
    return "\n".join(lines)
