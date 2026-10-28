#!/usr/bin/env python3
"""Criteria: does the head pair (unsloth, unsloth_zoo) regress the standard jobs against the base pair?

Per job, medians over the probe's repeats. REGRESSION when a base-passing job fails at the head, the
per-step loss leaves atol 1e-3 + rtol 5% of the base, steady step time grows more than 20%, or peak
memory grows more than 15% and 0.25 GB (jobs/compare.py's WARN thresholds, gated here).
Pairs with probes/cum_jobs_probe.py.
"""

from __future__ import annotations

import statistics

TITLE = "Standard jobs, cumulative base pair versus head pair"
MODE = "regression"
NEEDS = ["gpu", "rocm", "nvidia", "multi_gpu"]


def _med(runs, key):
    vals = [r.get(key) for r in runs if isinstance(r.get(key), (int, float))]
    return statistics.median(vals) if vals else None


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out = []
    for name in ("base", "head"):
        o = obs.get(name) or {}
        env = o.get("env") or {}
        out.append((f"{name} installed both repos", not o.get("error"), o.get("error") or "ok"))
        out.append((f"{name} ran on the ROCm GPU", bool(env.get("hip")) and bool(env.get("arch")), str(env)[:200]))
        jobs = o.get("jobs") or {}
        ran = {k: any(r.get("passed") for r in v) for k, v in jobs.items()}
        out.append((f"{name} jobs produced passing metrics", bool(jobs) and all(ran.values()),
                    ", ".join(f"{k}={'ok' if v else 'FAIL'}" for k, v in ran.items()) or "no jobs"))
    return out


def table(obs: dict) -> str:
    rows = ["| job | state | passed | final loss | steady step ms | peak GB |", "|---|---|---|---|---|---|"]
    for name in ("base", "head"):
        for job, runs in ((obs.get(name) or {}).get("jobs") or {}).items():
            fl = [r["losses"][-1] for r in runs if r.get("losses")]
            rows.append(f"| `{job}` | {name} | {sum(bool(r.get('passed')) for r in runs)}/{len(runs)} "
                        f"| {fl[0] if fl else '-'} | {_med(runs, 'steady_step_ms')} | {_med(runs, 'peak_mem_gb')} |")
    return "\n".join(rows)


def head_is_worse(base: dict, head: dict) -> tuple[bool, str]:
    bad = []
    for job, bruns in (base.get("jobs") or {}).items():
        hruns = (head.get("jobs") or {}).get(job) or []
        if not any(r.get("passed") for r in bruns):
            continue
        if not any(r.get("passed") for r in hruns):
            bad.append(f"`{job}` fails at the head")
            continue
        bl = next((r["losses"] for r in bruns if r.get("losses")), [])
        hl = next((r["losses"] for r in hruns if r.get("losses")), [])
        for i, (x, y) in enumerate(zip(bl, hl)):
            if x is not None and y is not None and abs(y - x) > 1e-3 + 0.05 * abs(x):
                bad.append(f"`{job}` step {i + 1} loss {y:.4f} vs {x:.4f}")
                break
        bt, ht = _med(bruns, "steady_step_ms"), _med(hruns, "steady_step_ms")
        if bt and ht and ht > 1.2 * bt:
            bad.append(f"`{job}` steady step {ht:.1f} ms vs {bt:.1f} ms")
        bm, hm = _med(bruns, "peak_mem_gb"), _med(hruns, "peak_mem_gb")
        if bm and hm and hm > 1.15 * bm and hm - bm > 0.25:
            bad.append(f"`{job}` peak memory {hm:.2f} GB vs {bm:.2f} GB")
    if bad:
        return True, "; ".join(bad)
    return False, "every base-passing job passes at the head within loss, step-time and memory tolerances"
