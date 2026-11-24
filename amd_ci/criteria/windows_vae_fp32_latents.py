#!/usr/bin/env python3
"""Criteria: #17953 on Windows ROCm (judges probes/diffusion_lora_train_probe.py with --suite winfix --latents
--infer flux3-klein --infer-res 512 --precisions bf16-eager --pytest ...; base = origin/main, head = main + #17953).

base_shows_defect (any): the FLUX.3 Klein Base training latents (trainer's own fp32 encode) differ from their math
twin by relative L2 > 0.05; the plain 512 render is LPIPS > 0.3 from its math twin; a selected test fails.
head_is_fixed (all): latents within relative L2 1e-4 of the math twin (the same math, chunked or not) and finite; the
512 render within LPIPS 0.1 of its math twin; FLUX.3 Klein Base bf16 (compile off) trains every step with finite
losses; every selected test passes and at least one ran. Loss against the Linux run is reported offline.
"""

from __future__ import annotations

import math

TITLE = "gfx1151 Windows: #17953 training latents, VAE render and tests (base main vs head)"
MODE = "differential"
NEEDS = ["gpu", "rocm", "windows", "windows_rocm_wddm", "linux", "exclusive_gpu", "nvidia", "discrete_gpu"]


def _lat(o):
    return (o or {}).get("latents") or {}


def _render(o):
    return ((o or {}).get("infer") or {}).get("flux3-klein") or {}


def _cell(o):
    return (((((o or {}).get("families") or {}).get("target") or {}).get("cells")) or {}).get("bf16-eager") or {}


def _pt(o):
    return (o or {}).get("pytest") or {}


def _bad_tests(o):
    p = _pt(o)
    return sorted(set(p.get("failed") or []) | set(p.get("errors") or []))


def gates(obs):
    out = []
    for name in ("base", "head"):
        o = obs.get(name) or {}
        lat, r, p = _lat(o), _render(o), _pt(o)
        out.append((f"{name}: latents measured", bool(lat.get("rows")), str(lat.get("stage_error") or "")[:200]))
        out.append((f"{name}: render and math twin scored", r.get("lpips_vs_math") is not None,
                    f"{r.get('lpips_vs_math')} {str(r.get('stage_error') or r.get('lpips_error') or '')[:150]}"))
        n = sum(p.get(k, 0) or 0 for k in ("n_passed", "n_failed", "n_errors", "n_skipped"))
        out.append((f"{name}: tests collected", n > 0, f"{n} ({str(p.get('error') or '')[:150]})"))
    out.append(("head: trainer VAE routing present", bool(_lat(obs.get("head")).get("guard_present")),
                f"patched {_lat(obs.get('head')).get('guard_patched')}"))
    return out


def base_shows_defect(base):
    lat = (_lat(base).get("max_rel_l2") or 0) > 0.05
    ren = (_render(base).get("lpips_vs_math") or 0) > 0.3
    tst = bool(_bad_tests(base))
    return (lat or ren or tst, f"latents wrong {lat}, render wrong {ren}, failing tests {len(_bad_tests(base))}")


def head_is_fixed(head):
    lat = _lat(head)
    if lat.get("max_rel_l2") is None or lat["max_rel_l2"] > 1e-4 or not all(r.get("finite") for r in lat["rows"]):
        return (False, f"latents off math: {lat.get('max_rel_l2')}")
    if (_render(head).get("lpips_vs_math") or 1) > 0.1:
        return (False, f"render off math: {_render(head).get('lpips_vs_math')}")
    c = _cell(head)
    losses = c.get("losses") or []
    if (c.get("complete") or {}).get("steps_run") != head.get("steps") or not losses or not all(
            isinstance(x, (int, float)) and math.isfinite(x) for x in losses):
        return (False, f"bf16 training: {c.get('errors') or c.get('complete')}")
    if _bad_tests(head) or not (_pt(head).get("n_passed") or 0):
        return (False, f"tests: {_bad_tests(head)}")
    return (True, "latents and render match math; bf16 trains; every test passes")


def table(obs):
    rows = ["| state | latents max rel L2 vs math | max abs | studio mean / std | math mean / std | render LPIPS vs math "
            "| bf16 loss q1 -> q4 (steps) | tests passed / failed |", "|---|---|---|---|---|---|---|---|"]
    for name in ("base", "head"):
        o = obs.get(name) or {}
        lat = _lat(o)
        r0 = (lat.get("rows") or [{}])[0]
        mx = max((r.get("max_abs_diff") or 0) for r in lat.get("rows") or [{}])
        c = _cell(o)
        L = [x for x in c.get("losses") or [] if isinstance(x, (int, float))]
        k = max(1, len(L) // 4)
        trend = f"{sum(L[:k]) / k:.3f} -> {sum(L[-k:]) / k:.3f} ({len(L)})" if L else str(c.get("errors") or "-")[:80]
        p = _pt(o)
        rows.append(f"| {name} | {lat.get('max_rel_l2')} | {mx} | {r0.get('studio_mean')} / {r0.get('studio_std')} | "
                    f"{r0.get('math_mean')} / {r0.get('math_std')} | {_render(o).get('lpips_vs_math')} | {trend} | "
                    f"{p.get('n_passed')} / {len(_bad_tests(o))} |")
    for name in ("base", "head"):
        bad = _bad_tests(obs.get(name))
        if bad:
            rows.append(f"\n**{name} failing tests**: " + ", ".join(f"`{t.split('::')[-1]}`" for t in bad[:12]))
    return "\n".join(rows)
