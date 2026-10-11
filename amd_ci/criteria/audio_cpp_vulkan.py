#!/usr/bin/env python3
"""Criteria for unslothai/unsloth#13298 / issue #13042: judge, never observe.

Defect: on a host with an AMD GPU and no NVIDIA driver, the audio.cpp installer
picks the CPU bundle. Fixed: it picks the Vulkan bundle, which enumerates a
Vulkan GPU, re-runs idempotently, and actually serves the MOSS ASR model.

Pairs with probes/audio_cpp_probe.py.
"""

from __future__ import annotations

import platform

TITLE = "audio.cpp prebuilt on an AMD GPU host (#13298 / #13042), base versus head"
MODE = "differential"
NEEDS: list[str] = ["vulkan", "discrete_gpu", "nvidia",
                    "windows" if platform.system() == "Windows" else "linux"]


def _rec(o: dict) -> dict:
    return o.get("record") or {}


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out = []
    for name in ("base", "head"):
        o = obs.get(name) or {}
        inst = o.get("install") or {}
        out.append((f"{name} installer ran to completion", inst.get("rc") == 0,
                    f"rc={inst.get('rc')}; {(inst.get('stderr') or '').strip()[-200:]}"))
        out.append((f"{name} install record written", bool(_rec(o).get("asset")),
                    f"asset={_rec(o).get('asset')}"))
        out.append((f"{name} installed server found", bool(o.get("server")), str(o.get("server"))))
    return out


def base_shows_defect(o: dict) -> tuple[bool, str]:
    r = _rec(o)
    asset = (r.get("asset") or "").lower()
    cpu = (r.get("backend") or "").lower() == "cpu" or ("cpu" in asset and "vulkan" not in asset)
    not_vk = o.get("detected_direct") != "vulkan"
    return cpu and not_vk, f"asset={r.get('asset')} backend={r.get('backend')} detected={o.get('detected_direct')}"


def head_is_fixed(o: dict) -> tuple[bool, str]:
    r = _rec(o)
    checks = {
        "backend vulkan": (r.get("backend") or "").lower() == "vulkan",
        "vulkan asset": "vulkan" in (r.get("asset") or "").lower(),
        "detected vulkan": o.get("detected_direct") == "vulkan",
        # ggml-vulkan tags an integrated GPU `[IGPU]`, a discrete one `[GPU]`; the PR accepts any `Vulkan:<n>` row.
        "lists a Vulkan GPU": any(("[GPU]" in row or "[IGPU]" in row) for row in o.get("vulkan_rows") or []),
        "idempotent re-run": bool(o.get("rerun_already_matches")),
    }
    if o.get("state") == "head":
        bench = o.get("bench") or []
        checks["MOSS served on vulkan (all repeats ok)"] = bool(bench) and all(b.get("ok") for b in bench)
    bad = [k for k, v in checks.items() if not v]
    return not bad, "failed: " + ", ".join(bad) if bad else "all checks hold"


def table(obs: dict) -> str:
    rows = ["| state | asset | backend | detected (record / direct) | Vulkan rows | re-run | "
            "MOSS request_s | load_s | transcript == base |", "|---|---|---|---|---|---|---|---|---|"]
    base_t = [b.get("transcript") for b in (obs.get("base") or {}).get("bench") or []]
    ref = base_t[0] if base_t else None
    for name in ("base", "head", "merge"):
        o = obs.get(name)
        if not o:
            continue
        r = _rec(o)
        bench = o.get("bench") or []
        req = ", ".join(str(b.get("request_s")) if b.get("ok") else f"FAIL({b.get('error', '')[:60]})"
                        for b in bench) or "-"
        load = ", ".join(str(b.get("load_s")) for b in bench) or "-"
        match = ", ".join(str(b.get("transcript") == ref) if ref else "n/a" for b in bench) or "-"
        vk = "<br>".join(o.get("vulkan_rows") or []) or "none"
        rows.append(f"| {name} | `{r.get('asset')}` | {r.get('backend')} | "
                    f"{r.get('detected_accelerator')} / {o.get('detected_direct')} | {vk} | "
                    f"{'already matches' if o.get('rerun_already_matches') else 'NO'} | {req} | {load} | {match} |")
    host = (obs.get("head") or {}).get("host") or {}
    return "\n".join(rows) + f"\n\nHost: `{host.get('node')}` {host.get('system')} {host.get('release')}, " \
                             f"nvidia-smi: {host.get('nvidia_smi')}"
