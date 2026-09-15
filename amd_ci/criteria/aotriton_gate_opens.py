#!/usr/bin/env python3
"""Criteria: does importing unsloth open the ROCm AOTriton SDPA gate, and do the
kernels it unlocks actually run and agree with the MATH reference?

Defect shape (base): the variable stays unset after `import unsloth`, torch then
refuses the flash / mem-efficient backends ("No available kernel") and the default
dispatch is bit-identical to MATH, which is what a silent fallback looks like.

Fixed shape (head): the variable reads "1" after the import, both backends run,
stay finite, and agree with MATH within fp16 tolerance.

VOID if the base does not show the defect on this torch/arch: that means the gate
is not what stands between this card and the kernels, and a green head would say
nothing about the PR. Pairs with probes/aotriton_gate_probe.py.
"""

from __future__ import annotations

TITLE = "ROCm AOTriton gate after `import unsloth`"
MODE = "differential"
# Everything the CHANGE touches, not what the host has: the gate is set on every
# platform, and only a ROCm build reads it.
NEEDS = ["rocm", "gpu", "windows", "nvidia", "xpu", "mlx", "multi_gpu"]

TOL = 2e-2


def _states(obs: dict):
    return [(n, v) for n, v in obs.items() if not n.startswith("_")]


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out = []
    archs = {n: v.get("arch") for n, v in _states(obs)}
    torchs = {n: v.get("torch") for n, v in _states(obs)}
    out.append(("every state reached torch on a device",
                all(v.get("cuda_available") for _, v in _states(obs)),
                ", ".join(f"{n}: torch={v.get('torch')} hip={v.get('hip')} err={v.get('torch_error', '')[:80]}"
                          for n, v in _states(obs))))
    out.append(("same GPU and torch build at every state",
                len(set(archs.values())) == 1 and len(set(torchs.values())) == 1,
                f"arch {archs}, torch {torchs}"))
    out.append(("probe scrubbed the variable before importing",
                all(v.get("gate_before_import") is None for _, v in _states(obs)),
                str({n: v.get("gate_before_import") for n, v in _states(obs)})))
    out.append(("torch reports a ROCm (HIP) build",
                all(v.get("hip") for _, v in _states(obs)),
                str({n: v.get("hip") for n, v in _states(obs)})))
    return out


def table(obs: dict) -> str:
    keys = [("gate_after_import", "gate after import"), ("can_use_flash", "can_use_flash"),
            ("can_use_efficient", "can_use_efficient"), ("flash_ran", "flash ran"),
            ("efficient_ran", "efficient ran"),
            ("flash_max_abs_diff_vs_math", "flash |diff| vs math"),
            ("efficient_max_abs_diff_vs_math", "efficient |diff| vs math"),
            ("math_peak_gib", "math peak GiB"), ("default_peak_gib", "default peak GiB"),
            ("default_matches_math_exactly", "default == math bit-exact")]
    names = [n for n, _ in _states(obs)]
    rows = ["| observation | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for k, label in keys:
        vals = []
        for _, v in _states(obs):
            x = v.get(k)
            if isinstance(x, float):
                vals.append(f"{x:.4g}")
            elif x is None:
                vals.append("n/a")
            else:
                vals.append(str(x).lower())
        rows.append(f"| {label} | " + " | ".join(vals) + " |")
    extra = []
    for n, v in _states(obs):
        if v.get("flash_error"):
            extra.append(f"{n} flash error: `{v['flash_error'][:120]}`")
        if v.get("stderr_gate_warnings"):
            extra.append(f"{n} torch stderr: `{v['stderr_gate_warnings'][0][:160]}`")
        if v.get("unsloth_import_error"):
            extra.append(f"{n} import unsloth raised: `{v['unsloth_import_error'][:160]}`")
    hdr = ""
    first = _states(obs)[0][1] if _states(obs) else {}
    if first:
        hdr = (f"GPU `{first.get('device_name')}` ({first.get('arch')}), torch `{first.get('torch')}`, "
               f"hip `{first.get('hip')}`.\n\n")
    return hdr + "\n".join(rows) + ("\n\n" + "\n".join(f"- {e}" for e in extra) if extra else "")


def base_shows_defect(base: dict) -> bool:
    gate_unset = base.get("gate_after_import") is None
    refused = (base.get("flash_ran") is False) or (base.get("efficient_ran") is False)
    silent = base.get("default_matches_math_exactly") is True
    return gate_unset and (refused or silent)


def head_is_fixed(head: dict) -> bool:
    if head.get("gate_after_import") != "1":
        return False
    for name in ("flash", "efficient"):
        if not head.get(f"{name}_ran") or not head.get(f"{name}_finite"):
            return False
        d = head.get(f"{name}_max_abs_diff_vs_math")
        if d is None or d > TOL:
            return False
    return True
