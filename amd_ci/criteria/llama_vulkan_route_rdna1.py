#!/usr/bin/env python3
"""Criteria: an RDNA 1 host with ROCm userspace installed gets the Vulkan llama.cpp prebuilt.

Differential. The defect: with rocminfo reporting gfx1010, the base resolves no Vulkan bundle
(no linux-rocm bundle covers gfx1010, so setup source-builds HIP). Fixed: the head resolves
linux-vulkan, and the bundle it chose lists a Vulkan device on the real Radeon and offloads a
tiny model to it.

Gates keep the comparison honest:
  * every probe ran and both arms of every state parsed a resolver answer;
  * the fake rocminfo was in force and an AMD Vulkan ICD exists on the host;
  * the control arm (HSA_OVERRIDE_GFX_VERSION set, the documented opt-out) is identical at
    base and head, so the change moved only the case it targets.

Pairs with probes/llama_vulkan_route_probe.py.
"""

from __future__ import annotations

import re

TITLE = "RDNA 1 (faked rocminfo gfx1010) llama.cpp prebuilt choice on real gfx1151, base versus head"
MODE = "differential"
NEEDS = ["linux", "rocm", "gpu", "discrete_gpu"]


def _payload(o: dict, arm: str) -> dict:
    return (((o or {}).get(arm) or {}).get("payload")) or {}


def _kind(o: dict, arm: str) -> str:
    p = _payload(o, arm)
    if not p:
        return "none"
    return p.get("install_kind") or ("no prebuilt (source build)" if not p.get("prebuilt_available") else "?")


def _vulkan_rows(o: dict) -> list[str]:
    text = ((o or {}).get("chosen_run") or {}).get("list_devices") or ""
    return [l.strip() for l in text.splitlines() if l.strip().lower().startswith("vulkan")]


def _offload_evidence(o: dict) -> list[str]:
    run = (o or {}).get("chosen_run") or {}
    rows = list(run.get("load_tensors") or []) + list(run.get("device_lines") or [])
    out = []
    for r in rows:
        low = r.lower()
        m = re.search(r"offloaded (\d+)/(\d+) layers to gpu", low)
        if (m and int(m.group(1)) > 0) or ("vulkan" in low and "buffer size" in low):
            out.append(r)
    return out


def _offloaded(o: dict) -> bool:
    return bool(_offload_evidence(o)) and ((o or {}).get("chosen_run") or {}).get("completion_rc") == 0


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out = []
    for name in ("base", "head"):
        o = obs.get(name) or {}
        ok = o.get("_probe_rc") == 0 and not o.get("_missing_output") and not o.get("error")
        out.append((f"{name} probe wrote observations", ok,
                    f"rc={o.get('_probe_rc')} {(o.get('error') or '')[-200:]}"))
        for arm in ("default", "control"):
            a = o.get(arm) or {}
            out.append((f"{name} {arm} resolver answered", a.get("rc") == 0 and bool(a.get("payload")),
                        f"rc={a.get('rc')} payload={a.get('payload')}"))
        out.append((f"{name} fake rocminfo in force", bool(o.get("fake_rocminfo_ok")),
                    f"real rocminfo on PATH: {o.get('real_rocminfo')}"))
    icds = (obs.get("head") or {}).get("vulkan_icds") or []
    out.append(("an AMD Vulkan ICD is installed", any("radeon" in i or "amd" in i for i in icds),
                ", ".join(icds) or "none"))
    b, h = _payload(obs.get("base"), "control"), _payload(obs.get("head"), "control")
    out.append(("control arm (HSA_OVERRIDE_GFX_VERSION) identical at base and head",
                bool(b) and b == h, f"base `{_kind(obs.get('base'), 'control')}` head "
                f"`{_kind(obs.get('head'), 'control')}`"))
    return out


def table(obs: dict) -> str:
    rows = ["| observation | base | head |", "|---|---|---|"]
    b, h = obs.get("base") or {}, obs.get("head") or {}
    rows.append(f"| default arm install kind | `{_kind(b, 'default')}` | `{_kind(h, 'default')}` |")
    rows.append(f"| default arm asset | `{_payload(b, 'default').get('asset')}` | "
                f"`{_payload(h, 'default').get('asset')}` |")
    rows.append(f"| control arm install kind | `{_kind(b, 'control')}` | `{_kind(h, 'control')}` |")
    rows.append(f"| chosen bundle Vulkan devices | `{_vulkan_rows(b)}` | `{_vulkan_rows(h)}` |")
    hr = h.get("chosen_run") or {}
    rows.append(f"| tiny GGUF -ngl 99 offload evidence | - | `{_offload_evidence(h)[:4]}` |")
    rows.append(f"| tiny GGUF completion (--device Vulkan0 -ngl 99) rc / s / text | - | "
                f"`{hr.get('completion_rc')}` / `{hr.get('completion_s')}` / "
                f"`{(hr.get('completion_text') or '').strip()[-80:]}` |")
    return "\n".join(rows)


def base_shows_defect(base: dict) -> bool:
    return not _kind(base, "default").endswith("vulkan")


def head_is_fixed(head: dict) -> bool:
    return (_kind(head, "default") == "linux-vulkan" and bool(_vulkan_rows(head))
            and _offloaded(head))
