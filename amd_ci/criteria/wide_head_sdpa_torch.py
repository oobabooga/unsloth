#!/usr/bin/env python3
"""Criteria: Windows ROCm wide-head SDPA guard (#17976) on the torch build this job installed (judges
probes/wide_head_sdpa_torch_probe.py).

base_shows_defect: with stock dispatch (base tree, no guard) some head-512 case comes out wrong (non-finite or
relative L2 error above REL_MAX against the float32 math reference), or Gemma 4's sdpa forward disagrees with eager.
head_is_fixed: the guard is installed, every head-512 case runs math only and is correct, every head-256 case runs
the same op as base, and Gemma 4's sdpa forward matches eager.
A torch whose kernels are already correct at 512 reads VOID: the defect is gone there, nothing for the guard to fix.
"""

from __future__ import annotations

TITLE = "gfx1151 Windows: SDPA head 512 with and without the wide-head guard, per torch build"
MODE = "differential"
NEEDS = ["gpu", "rocm", "windows", "windows_rocm_wddm", "nvidia", "discrete_gpu"]
REL_MAX = 0.02  # bf16 kernels vs float32 math land near 2e-3; a broken kernel is far above this
TOP1_MIN = 0.98
NLL_GAP_MAX = 0.1


def _task(o, name):
    return ((o or {}).get("tasks") or {}).get(name) or {}


def _rows(o):
    return _task(o, "kernels").get("kernels") or []


def _bad(e):
    return not e or "refused" in e or not e.get("finite") or e.get("rel_l2", 1) > REL_MAX


def _fwd(o):
    return _task(o, "hf_fwd").get("hf_fwd") or {}


def _fwd_wrong(f):
    if "top1_agree" not in f:
        return None
    return (not f.get("finite") or f["top1_agree"] < TOP1_MIN
            or (f.get("nll_sdpa", 0) - f.get("nll_eager", 0)) > NLL_GAP_MAX)


def gates(obs):
    b, h = obs.get("base") or {}, obs.get("head") or {}
    out = []
    for name, o in (("base", b), ("head", h)):
        k = _task(o, "kernels")
        out.append((f"{name}: kernel task ran on the GPU", bool(_rows(o)) and not k.get("error"),
                    f"rc {k.get('rc')}, rows {len(_rows(o))}, error {k.get('error')}"))
        errs = [r for r in _rows(o) if r.get("error")]
        out.append((f"{name}: every case measured", not errs, f"{len(errs)} errored, e.g. {errs[:1]}"))
    out.append(("both on Windows ROCm gfx1151",
                all(o.get("platform") == "win32" and o.get("hip") and str(o.get("arch", "")).startswith("gfx1151")
                    for o in (b, h)),
                f"{b.get('platform')} {b.get('hip')} {b.get('arch')} / {h.get('platform')} {h.get('arch')}"))
    out.append(("same torch build in both states", b.get("torch") == h.get("torch") and bool(b.get("torch")),
                f"{b.get('torch')} / {h.get('torch')}"))
    out.append(("guard: head installed, base has none",
                bool(h.get("guard_installed")) and not b.get("guard_installed"),
                f"head {h.get('guard_installed')} ({(h.get('guard') or {}).get('fix_returned')}), "
                f"base {b.get('guard_installed')}"))
    return out


def base_shows_defect(base):
    wide = [r for r in _rows(base) if r.get("d") == 512]
    bad = [r for r in wide if _bad(r.get("default"))]
    fw = _fwd_wrong(_fwd(base))
    worst = max((r.get("default") or {}).get("rel_l2", 0) or 0 for r in wide) if wide else None
    return (bool(bad) or bool(fw),
            f"{len(bad)}/{len(wide)} head-512 cases wrong on default dispatch (worst rel L2 {worst}, ops "
            f"{sorted({o for r in wide for o in r.get('default_ops', [])})}); Gemma 4 forward wrong {fw}")


def head_is_fixed(head, base = None):
    wide = [r for r in _rows(head) if r.get("d") == 512]
    narrow = [r for r in _rows(head) if r.get("d") == 256]
    bad = [r for r in wide if _bad(r.get("default"))]
    non_math = [r for r in wide if set(r.get("default_ops") or []) != {"_scaled_dot_product_attention_math"}]
    fw = _fwd_wrong(_fwd(head))
    ok = bool(wide) and not bad and not non_math and fw is not True and bool(head.get("guard_installed"))
    return (ok, f"{len(bad)}/{len(wide)} head-512 wrong, {len(non_math)} not math-only; head-256 ops "
                f"{sorted({o for r in narrow for o in r.get('default_ops', [])})}; Gemma 4 forward wrong {fw}")


def table(obs):
    lines = ["| state | torch | d | cases | default wrong | worst rel L2 | ops | flash wrong | efficient wrong | ms @2048 |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for st in ("base", "head"):
        o = obs.get(st) or {}
        for d in (256, 512):
            rows = [r for r in _rows(o) if r.get("d") == d]
            if not rows:
                continue
            worst = max(((r.get("default") or {}).get("rel_l2") or 0) for r in rows)
            ops = sorted({x for r in rows for x in r.get("default_ops", [])})
            fl = sum(1 for r in rows if "refused" not in (r.get("flash") or {}) and _bad(r.get("flash")))
            ef = sum(1 for r in rows if "refused" not in (r.get("efficient") or {}) and _bad(r.get("efficient")))
            ms = next((r.get("default_ms_median") for r in rows if r.get("default_ms_median")), "")
            lines.append(f"| {st} | {o.get('torch')} | {d} | {len(rows)} | "
                         f"{sum(1 for r in rows if _bad(r.get('default')))} | {worst} | {', '.join(ops)} | "
                         f"{fl} | {ef} | {ms} |")
        f = _fwd(o)
        if f:
            lines.append(f"| {st} | Gemma 4 E2B fwd | | | top1 {f.get('top1_agree')} | nll sdpa {f.get('nll_sdpa')}"
                         f" vs eager {f.get('nll_eager')} | {f.get('sdpa_ops')} | | | |")
    return "\n".join(lines)
