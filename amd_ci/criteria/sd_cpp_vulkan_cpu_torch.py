#!/usr/bin/env python3
"""Criteria: with CPU-only torch beside a Radeon, the image engine installs a Vulkan sd.cpp build.

Differential (#13666). The defect: the base resolves the plain CPU build ("auto"/"cpu"), whose
sd-cli lists no Vulkan device. Fixed: the head resolves "vulkan", installs a -vulkan asset, and
its sd-cli lists a Vulkan device on the real Radeon and renders one image there.

Gates: both probes ran; torch is CPU-only and the diffusion backend is "cpu" at both states
(otherwise the route under test is not reached); the real inventory lists an AMD card; both
installs produced an sd-cli whose --list-devices ran.

Pairs with probes/sd_cpp_vulkan_route_probe.py.
"""

from __future__ import annotations

TITLE = "sd.cpp build chosen with CPU-only torch on a real Radeon 8060S, base versus head"
MODE = "differential"
NEEDS = ["gpu", "rocm"]


def _vulkan_rows(o: dict) -> list[str]:
    text = (o or {}).get("list_devices") or ""
    return [l.strip() for l in text.splitlines() if l.strip().lower().startswith("vulkan")]


def _asset(o: dict) -> str:
    return str((o or {}).get("asset"))


def _record_accel(o: dict) -> str:
    return str(((o or {}).get("install_record") or {}).get("accelerator"))


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out = []
    for name in ("base", "head"):
        o = obs.get(name) or {}
        ok = o.get("_probe_rc") == 0 and not o.get("_missing_output") and not o.get("error")
        out.append((f"{name} probe wrote observations", ok,
                    f"rc={o.get('_probe_rc')} {(o.get('error') or '')[-300:]}"))
        out.append((f"{name} torch is CPU-only and backend is cpu",
                    o.get("torch_gpu") is False and o.get("backend") == "cpu",
                    f"torch {o.get('torch')} gpu={o.get('torch_gpu')} backend={o.get('backend')}"))
        devs = ((o.get("inventory") or {}).get("devices")) or []
        out.append((f"{name} inventory lists an AMD card", any(d.get("vendor") == "amd" for d in devs),
                    str(devs)[:300]))
        out.append((f"{name} sd-cli installed and --list-devices ran",
                    bool(o.get("sd_cli")) and o.get("list_devices_rc") == 0,
                    (o.get("install_error") or "")[-300:] or f"rc={o.get('list_devices_rc')}"))
    return out


def table(obs: dict) -> str:
    b, h = obs.get("base") or {}, obs.get("head") or {}
    rows = ["| observation | base | head |", "|---|---|---|"]
    rows.append(f"| image_install_accelerator(cpu) | `{b.get('image_install_accelerator')}` | "
                f"`{h.get('image_install_accelerator')}` |")
    rows.append(f"| asset | `{_asset(b)}` | `{_asset(h)}` |")
    rows.append(f"| install record accelerator | `{_record_accel(b)}` | `{_record_accel(h)}` |")
    rows.append(f"| sd-cli Vulkan devices | `{_vulkan_rows(b)}` | `{_vulkan_rows(h)}` |")
    rows.append(f"| 512x512 image, 1 step: rc / PNG bytes | `{b.get('generate_rc')}` / "
                f"`{b.get('generate_png_bytes')}` | `{h.get('generate_rc')}` / "
                f"`{h.get('generate_png_bytes')}` |")
    rows.append(f"| image wall time (s) | `{b.get('generate_s')}` | `{h.get('generate_s')}` |")
    rows.append(f"| sd-cli Vulkan log lines | `{b.get('generate_vulkan_lines')}` | "
                f"`{h.get('generate_vulkan_lines')}` |")
    return "\n".join(rows)


def base_shows_defect(base: dict) -> bool:
    return base.get("image_install_accelerator") in ("auto", "cpu") and not _vulkan_rows(base)


def head_is_fixed(head: dict) -> bool:
    return (head.get("image_install_accelerator") == "vulkan"
            and "vulkan" in _asset(head).lower() and _record_accel(head) == "vulkan"
            and bool(_vulkan_rows(head))
            and head.get("generate_rc") == 0 and (head.get("generate_png_bytes") or 0) > 0)
