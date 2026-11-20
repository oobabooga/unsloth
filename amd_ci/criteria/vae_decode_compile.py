#!/usr/bin/env python3
"""Criteria (Windows ROCm): judges probes/vae_decode_compile_probe.py.

base_shows_defect: the narrow-head (64) VAE is refused decode compile on a U-Net pipe (main's platform-wide exclusion).
head_is_fixed: the narrow VAE is allowed on the U-Net pipe and when the env forces it, its compiled decode on the GPU
is finite and within 1% relative L2 of eager, and the wide-head (512) VAE is still refused in both cases.
"""

from __future__ import annotations

TITLE = "gfx1151 Windows: VAE decode compile kept for narrow-head VAEs, still refused for head dim 512"
MODE = "differential"
NEEDS = ["gpu", "rocm", "windows", "windows_rocm_wddm", "linux", "nvidia", "discrete_gpu"]


def _a(o, name, key):
    return ((o.get("allowed") or {}).get(name) or {}).get(key)


def gates(obs):
    out = []
    for name in ("base", "head"):
        o = obs.get(name) or {}
        out.append((f"{name}: gate observed", "allowed" in o,
                    f"{o.get('error') or ''} {o.get('torch')} {o.get('arch')} {o.get('platform')}"[:300]))
        out.append((f"{name}: on the GPU", bool(o.get("cuda")), str(o.get("arch"))))
        out.append((f"{name}: Windows ROCm", o.get("platform") == "win32" and bool(o.get("hip")), str(o.get("hip"))))
    return out


def base_shows_defect(base):
    return (_a(base, "narrow", "unet") is False, f"narrow allowed {base.get('allowed', {}).get('narrow')}")


def head_is_fixed(head):
    ok = (
        _a(head, "narrow", "unet") is True
        and _a(head, "narrow", "dit_env_forced") is True
        and _a(head, "wide", "unet") is False
        and _a(head, "wide", "dit_env_forced") is False
        and head.get("compiled_finite") is True
        and (head.get("compiled_rel_l2") is not None and head["compiled_rel_l2"] <= 0.01)
    )
    return (bool(ok), f"allowed {head.get('allowed')}, compiled finite {head.get('compiled_finite')}, "
                      f"rel L2 {head.get('compiled_rel_l2')}")


def table(obs):
    rows = ["| state | platform | arch | narrow unet / forced | wide unet / forced | compiled finite | rel L2 |",
            "|---|---|---|---|---|---|---|"]
    for name in ("base", "head"):
        o = obs.get(name) or {}
        rows.append(f"| {name} | {o.get('platform')} | {o.get('arch')} | {_a(o, 'narrow', 'unet')} / "
                    f"{_a(o, 'narrow', 'dit_env_forced')} | {_a(o, 'wide', 'unet')} / {_a(o, 'wide', 'dit_env_forced')} | "
                    f"{o.get('compiled_finite')} | {o.get('compiled_rel_l2')} |")
    return "\n".join(rows)
