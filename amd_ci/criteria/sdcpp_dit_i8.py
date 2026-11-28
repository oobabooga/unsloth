#!/usr/bin/env python3
"""Criteria: upstream sd.cpp `--linear-type dit=i8` vs `--linear-type raw` on one binary, gfx1151 (timing).
Pairs with probes/sdcpp_dit_i8_probe.py. Regression mode, where "head" is read as the i8 arm and "base" as raw: both
states carry the same shared measurement (the arms are flags on one binary, not two trees).

Per model: median s/step per arm over ok runs, A/A noise = median per-prompt |raw2 - raw| / raw (floored at 1%),
speedup = raw / i8, paired LPIPS / PSNR i8-vs-raw and raw2-vs-raw, and engagement (i8 PNG differs from raw on every
prompt while raw2 is byte-identical to raw, or raw2 is closer to raw than i8 is).
Gates (non-vacuity, Qwen-Image-2.1 only; Krea is reported, not required): binary listed a GPU; every Qwen arm rendered
on every prompt; i8 engaged on every prompt.
head_is_worse: the i8 arm is slower than raw by more than the A/A noise.
"""

from __future__ import annotations

import statistics

TITLE = "sd.cpp --linear-type dit=i8 vs raw (upstream master-1147-3b0426f) on gfx1151"
MODE = "regression"
NEEDS = ["gpu", "linux", "exclusive_gpu", "nvidia", "windows"]
NOISE_FLOOR = 0.01
REQUIRED = "qwen"


def _sh(obs: dict) -> dict:
    for name in ("base", "head"):
        s = (obs.get(name) or {}).get("shared")
        if s:
            return s
    return (obs or {}).get("shared") or {}


def _ok(r) -> bool:
    return isinstance(r, dict) and r.get("rc") == 0 and isinstance(r.get("s_per_step"), (int, float)) and r.get("sha256")


def _scores(sh: dict, model: str) -> dict:
    out: dict = {}
    for p in ((sh.get("scores") or {}).get("pairs") or []):
        if p.get("model") == model:
            out.setdefault(p["pair"], {})[p["prompt"]] = p
    return out


def summary(sh: dict, model: str) -> dict:
    res = (sh.get("models") or {}).get(model) or {}
    runs = res.get("runs") or {}
    s: dict = {"model": model, "skipped": res.get("skipped"), "n_prompts": len(runs)}
    if not runs:
        return s
    for arm in ("raw", "i8", "raw2"):
        ok = [a[arm] for a in runs.values() if _ok(a.get(arm))]
        s[arm] = {"n": len(ok), "n_all": len(runs), "s_step": statistics.median([r["s_per_step"] for r in ok]) if ok else None,
                  "samp": statistics.median([r["sampling_s"] for r in ok]) if ok else None,
                  "wall": statistics.median([r["wall_s"] for r in ok]) if ok else None,
                  "fallback": sorted({ln for r in ok for ln in r.get("fallback_lines") or []})[:8],
                  "rcs": [str((a.get(arm) or {}).get("rc")) for a in runs.values()]}
    full = [a for a in runs.values() if all(_ok(a.get(k)) for k in ("raw", "i8", "raw2"))]
    s["n_full"] = len(full)
    s["i8_differs"] = sum(a["i8"]["sha256"] != a["raw"]["sha256"] for a in full)
    s["aa_identical"] = sum(a["raw2"]["sha256"] == a["raw"]["sha256"] for a in full)
    aa = [abs(a["raw2"]["s_per_step"] - a["raw"]["s_per_step"]) / a["raw"]["s_per_step"] for a in full]
    s["aa_noise"] = max(NOISE_FLOOR, statistics.median(aa)) if aa else None
    s["aa_noise_max"] = max(aa) if aa else None
    paired = [a["raw"]["s_per_step"] / a["i8"]["s_per_step"] for a in full]
    s["speedup_paired_median"] = statistics.median(paired) if paired else None
    if s["raw"]["s_step"] and s["i8"]["s_step"]:
        s["speedup"] = s["raw"]["s_step"] / s["i8"]["s_step"]
    sc = _scores(sh, model)
    for pair in ("i8_vs_raw", "raw2_vs_raw"):
        rows = list((sc.get(pair) or {}).values())
        lp = [r["lpips"] for r in rows if isinstance(r.get("lpips"), (int, float))]
        ps = [r["psnr"] for r in rows if isinstance(r.get("psnr"), (int, float))]
        s[pair] = {"n": len(rows), "lpips_mean": statistics.mean(lp) if lp else None, "lpips_n": len(lp),
                   "lpips_max": max(lp) if lp else None, "psnr_mean": statistics.mean(ps) if ps else None,
                   "psnr_inf": sum(r.get("psnr") == "inf" for r in rows),
                   "max_abs": max((r.get("max_abs") or 0) for r in rows) if rows else None}
    # engagement: i8 differs everywhere, and the A/A is tighter (identical, or closer than i8)
    i8l, aal = s["i8_vs_raw"]["lpips_mean"], s["raw2_vs_raw"]["lpips_mean"]
    aa_tighter = s["aa_identical"] == s["n_full"] or (i8l is not None and aal is not None and aal < i8l)
    s["engaged"] = bool(s["n_full"]) and s["i8_differs"] == s["n_full"] and aa_tighter
    return s


def gates(obs: dict) -> list:
    sh = _sh(obs)
    b = sh.get("bin") or {}
    ld = (b.get("list_devices_final") or {})
    q = summary(sh, REQUIRED)
    return [
        ("sd-cli obtained and --list-devices lists a GPU", bool(b.get("binary")) and bool(ld.get("gpu_listed")),
         f"{b.get('source')} backend {b.get('backend')}: {str(ld.get('out', ''))[-250:]!r}".replace("|", "/")),
        ("Qwen-Image-2.1: every arm rendered on every prompt", q.get("n_full", 0) == len(sh.get("prompts") or [1] * 4)
         and q.get("n_full", 0) >= 4, f"full prompts {q.get('n_full')} of {q.get('n_prompts')}; rcs "
         + "; ".join(f"{k}: {','.join((q.get(k) or {}).get('rcs') or [])}" for k in ('raw', 'i8', 'raw2'))
         + f"; skipped {q.get('skipped')}"),
        ("Qwen-Image-2.1: i8 path engaged (i8 PNG differs from raw on every prompt, A/A tighter)",
         bool(q.get("engaged")), f"i8 differs {q.get('i8_differs')}/{q.get('n_full')}, raw2 identical "
         f"{q.get('aa_identical')}/{q.get('n_full')}, LPIPS i8 {_f((q.get('i8_vs_raw') or {}).get('lpips_mean'), 4)} "
         f"vs A/A {_f((q.get('raw2_vs_raw') or {}).get('lpips_mean'), 4)}"),
    ]


def _f(x, nd = 3):
    return "-" if x is None else f"{x:.{nd}f}"


def table(obs: dict) -> str:
    sh = _sh(obs)
    b = sh.get("bin") or {}
    sc = sh.get("scores") or {}
    rows = [f"Binary: {b.get('source')} (backend {b.get('backend')}), scoring: {'; '.join(sc.get('method') or [])}"
            + (f" (LPIPS unavailable: {sc.get('lpips_error')})" if sc.get("lpips_error") else ""), "",
            "| model | arm | n ok | median s/step | median sampling s | median wall s | fallback lines |",
            "|---|---|---|---|---|---|---|"]
    sums = [summary(sh, m) for m in (sh.get("models") or {})]
    for s in sums:
        if s.get("skipped") and not s.get("raw"):
            rows.append(f"| {s['model']} | - | 0 | - | - | - | skipped: {s['skipped']} |")
            continue
        for arm in ("raw", "i8", "raw2"):
            c = s.get(arm) or {}
            rows.append(f"| {s['model']} | {arm} | {c.get('n')}/{c.get('n_all')} | {_f(c.get('s_step'))} | "
                        f"{_f(c.get('samp'), 2)} | {_f(c.get('wall'), 1)} | "
                        f"{'; '.join(c.get('fallback') or []).replace('|', '/')[:300] or '-'} |")
    rows += ["", "| model | speedup raw/i8 (medians) | paired median speedup | A/A noise (median, max) | "
             "i8 differs | raw2 identical | LPIPS i8 vs raw (mean, n, max) | LPIPS raw2 vs raw (mean, n) | "
             "PSNR i8 / A/A | engaged |", "|---|---|---|---|---|---|---|---|---|---|"]
    for s in sums:
        if not s.get("raw"):
            continue
        i, a = s.get("i8_vs_raw") or {}, s.get("raw2_vs_raw") or {}
        rows.append(f"| {s['model']} | {_f(s.get('speedup'))} | {_f(s.get('speedup_paired_median'))} | "
                    f"{_f(s.get('aa_noise'), 4)}, {_f(s.get('aa_noise_max'), 4)} | {s.get('i8_differs')}/{s.get('n_full')} | "
                    f"{s.get('aa_identical')}/{s.get('n_full')} | {_f(i.get('lpips_mean'), 4)}, {i.get('lpips_n')}, "
                    f"{_f(i.get('lpips_max'), 4)} | {_f(a.get('lpips_mean'), 4)}, {a.get('lpips_n')} | "
                    f"{_f(i.get('psnr_mean'), 2)} / {_f(a.get('psnr_mean'), 2) if a.get('psnr_mean') is not None else ''}"
                    f"{' (inf x' + str(a.get('psnr_inf')) + ')' if a.get('psnr_inf') else ''} | {s.get('engaged')} |")
    return "\n".join(rows)


def head_is_worse(base: dict, head: dict):
    sh = (head or {}).get("shared") or (base or {}).get("shared") or {}
    q = summary(sh, REQUIRED)
    raw, i8 = (q.get("raw") or {}).get("s_step"), (q.get("i8") or {}).get("s_step")
    if not raw or not i8:
        return True, "Qwen-Image-2.1 i8 or raw arm produced no timing"
    noise = q.get("aa_noise") or NOISE_FLOOR
    detail = (f"Qwen-Image-2.1 median s/step i8 {i8:.3f} vs raw {raw:.3f} (speedup {raw / i8:.3f}x, A/A noise "
              f"{noise:.4f}); LPIPS i8 vs raw {_f((q.get('i8_vs_raw') or {}).get('lpips_mean'), 4)}")
    k = summary(sh, "krea")
    if k.get("speedup"):
        detail += f"; Krea-2 Turbo speedup {k['speedup']:.3f}x"
    if i8 > raw * (1 + noise):
        return True, "i8 slower than raw: " + detail
    return False, detail
