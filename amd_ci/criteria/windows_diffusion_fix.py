#!/usr/bin/env python3
"""Criteria: the two Windows ROCm diffusion fixes (judges probes/diffusion_lora_train_probe.py with --suite winfix
--infer ... --no-lora-render; base = --patch-base, head = base + the fix).

base_shows_defect (either is the defect):
  - FLUX.3 Klein Base ``auto`` training (resolves to bf16, compile on) ends in an error event naming the missing C++
    compiler ("cl is not found"), or
  - a plain Studio render differs from the same render with every fused SDPA backend off (math only, measured exact)
    by LPIPS > 0.3: the VAE's head-dim-512 attention ran a fused kernel that miscomputes it.
head_is_fixed (all of):
  - FLUX.3 Klein Base ``auto`` trained every step with finite losses, compiled False, and a warning naming the
    missing compiler;
  - every plain render is within LPIPS 0.1 of its math-only twin (the DiT's own head-128 attention stays fused, so
    the two are not bit-identical; a broken VAE scores 0.5+).
The Qwen-Image-2.1 training cell (head only) and the renders' cross-OS LPIPS against Linux are tabulated, not judged.
"""

from __future__ import annotations

import math

TITLE = "gfx1151 Windows: diffusion training without a C++ compiler, and VAE attention at head dim 512"
MODE = "differential"
NEEDS = ["gpu", "rocm", "windows", "windows_rocm_wddm", "linux", "exclusive_gpu", "nvidia", "discrete_gpu"]
RENDER_BROKEN = 0.3
RENDER_OK = 0.1


def _target_auto(state: dict) -> dict:
    return ((((state or {}).get("families") or {}).get("target") or {}).get("cells") or {}).get("auto") or {}


def _cl_death(cell: dict) -> bool:
    return any("is not found" in str(e) for e in cell.get("errors") or [])


def _eager_with_warning(cell: dict, steps) -> tuple[bool, str]:
    c = cell.get("complete") or {}
    losses = cell.get("losses") or []
    if cell.get("errors"):
        return False, f"error: {str(cell['errors'][0])[:200]}"
    if not c or c.get("stopped") or c.get("steps_run") != steps:
        return False, f"incomplete: {c}"
    if len(losses) != steps or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in losses):
        return False, "losses missing or non-finite"
    if cell.get("compiled") is not False:
        return False, f"compiled={cell.get('compiled')}"
    if not any("C++ compiler" in str(w) for w in cell.get("warnings") or []):
        return False, f"no compiler warning in {cell.get('warnings')}"
    return True, "trained eager with the warning"


def _renders(state: dict) -> dict:
    return {m: r for m, r in ((state or {}).get("infer") or {}).items()}


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out = []
    for name in ("base", "head"):
        o = obs.get(name) or {}
        h = o.get("host") or {}
        out.append((f"{name}: probe ran", o.get("_probe_rc") == 0 and not o.get("_missing_output"),
                    f"rc={o.get('_probe_rc')}"))
        out.append((f"{name}: Windows ROCm gfx1151", bool(h.get("hip")) and str(h.get("arch", "")).startswith("gfx1151"),
                    f"{h.get('torch')} {h.get('arch')}"))
        cell = _target_auto(o)
        out.append((f"{name}: FLUX.3 Klein Base auto cell ran", bool(cell) and not cell.get("stage_error"),
                    f"rc={cell.get('_rc')} {str(cell.get('stage_error') or '')[:150]}"))
        rs = _renders(o)
        ok = bool(rs) and all(r.get("lpips_vs_math") is not None for r in rs.values())
        out.append((f"{name}: every plain render and its math twin scored", ok,
                    "; ".join(f"{m}: {r.get('lpips_vs_math')} {str(r.get('skipped') or r.get('stage_error') or '')[:80]}"
                              for m, r in rs.items()) or "none"))
    return out


def base_shows_defect(base: dict):
    cl = _cl_death(_target_auto(base))
    broken = [m for m, r in _renders(base).items() if (r.get("lpips_vs_math") or 0) > RENDER_BROKEN]
    return (cl or bool(broken), f"cl death {cl}; renders off their math twin: {broken}")


def head_is_fixed(head: dict):
    ok, why = _eager_with_warning(_target_auto(head), head.get("steps"))
    if not ok:
        return (False, f"FLUX.3 auto: {why}")
    bad = {m: r.get("lpips_vs_math") for m, r in _renders(head).items()
           if r.get("lpips_vs_math") is None or r["lpips_vs_math"] > RENDER_OK}
    if bad:
        return (False, f"renders off their math twin: {bad}")
    return (True, "auto trains eager with the warning; every render matches its math-only twin")


def table(obs: dict) -> str:
    rows = ["| state | cell | result | loss q1 -> q4 | samples/s | peak GB | compiled | warnings / error |",
            "|---|---|---|---|---|---|---|---|"]
    for name in ("base", "head"):
        o = obs.get(name) or {}
        for role, fam in ((o.get("families") or {}).items()):
            for p, c in (fam.get("cells") or {}).items():
                losses = [x for x in c.get("losses") or [] if isinstance(x, (int, float))]
                k = max(1, len(losses) // 4)
                trend = f"{sum(losses[:k]) / k:.3f} -> {sum(losses[-k:]) / k:.3f}" if len(losses) >= 4 else "-"
                done = (c.get("complete") or {}).get("steps_run")
                res = "error" if c.get("errors") else (f"trained {done}" if done else "no result")
                note = (c.get("errors") or c.get("warnings") or [""])[0]
                rows.append(f"| {name} | {fam.get('resolved_family')} {p} | {res} | {trend} | "
                            f"{c.get('samples_per_second')} | {c.get('trainer_peak_memory_gb')} | {c.get('compiled')} | "
                            f"{str(note)[:160]} |")
    rows += ["", "| state | render | LPIPS vs math-only twin | MAD | studio s | math-only s | load |",
             "|---|---|---|---|---|---|---|"]
    for name in ("base", "head"):
        for m, r in _renders(obs.get(name) or {}).items():
            rows.append(f"| {name} | {m} @ {r.get('res')} | {r.get('lpips_vs_math')} | {r.get('mad_vs_math')} | "
                        f"{(r.get('studio') or {}).get('s')} | {(r.get('math_only') or {}).get('s')} | "
                        f"{r.get('load_status')} {str(r.get('skipped') or r.get('stage_error') or '')[:120]} |")
    return "\n".join(rows)
