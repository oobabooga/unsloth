"""Criteria: Segmind Vega 1024px, NPU (Ryzen AI 1.9 Linux) vs gfx1151 iGPU (diffusers ROCm).

Not a PR check: the deliverable is the table. Gates require both arms to have produced
timed runs, so a broken arm reads INCONCLUSIVE instead of a silent one-sided number.
"""
from __future__ import annotations

import statistics

TITLE = "XDNA2 NPU vs gfx1151 iGPU: Segmind Vega 1024px 20 steps timing"
MODE = "regression"
NEEDS: list[str] = ["gpu"]


def _npu_runs(b):
    return [r for r in (b.get("npu") or {}).get("runs", []) if "denoise_s" in r]


def _igpu_runs(b):
    return ((b.get("igpu") or {}).get("result") or {}).get("runs", [])


def gates(obs):
    b = obs.get("base") or {}
    n, g = _npu_runs(b), _igpu_runs(b)
    nsteps = {k: v.get("rc") for k, v in ((b.get("npu") or {}).get("steps") or {}).items()}
    return [("NPU arm produced timed runs", len(n) >= 2, f"{len(n)} runs; steps {nsteps}"),
            ("iGPU arm produced timed runs", len(g) >= 3, f"{len(g)} runs")]


def head_is_worse(base, head):
    return False, "measurement only"


def _med(xs):
    xs = [x for x in xs if x is not None]
    return f"{statistics.median(xs):.2f}" if xs else "-"


def table(obs):
    b = obs.get("base") or {}
    n, g = _npu_runs(b), _igpu_runs(b)
    n_w, g_w = n[1:] or n, g[1:] or g  # drop the first (warmup) run
    rows = ["| arm | runs | denoise s (median) | VAE s | total s | it/s |", "|---|---|---|---|---|---|"]
    if n_w:
        d = statistics.median(r["denoise_s"] for r in n_w)
        rows.append(f"| NPU | {len(n_w)} | {d:.2f} | {_med([r.get('vae_s') for r in n_w])} | "
                    f"{_med([r.get('total_s') - r.get('load_s', 0) for r in n_w])} (excl. load) | {20 / d:.2f} |")
    if g_w:
        d = statistics.median(r["denoise_s"] for r in g_w)
        rows.append(f"| iGPU | {len(g_w)} | {d:.2f} | {_med([r.get('vae_s') for r in g_w])} | "
                    f"{_med([r.get('total_s') for r in g_w])} | {20 / d:.2f} |")
    rows.append("")
    rows.append(f"idle power {b.get('idle_power')} / after {b.get('idle_power_after')}")
    rows.append("NPU per-run power: " + "; ".join(str(r.get("power")) for r in n))
    rows.append(f"iGPU whole-script power: {(b.get('igpu') or {}).get('power_whole_script')}")
    rows.append("NPU steps: " + str({k: v for k, v in ((b.get("npu") or {}).get("steps") or {}).items()}))
    for r in (b.get("npu") or {}).get("runs", []):
        rows.append("```\n" + str(r.get("tail", ""))[-2500:] + "\n```")
    if not n:
        rows.append("NPU log tail:\n```\n" + str((b.get("npu") or {}).get("log_tail", ""))[-4000:] + "\n```")
    rows.append("iGPU tail:\n```\n" + str((b.get("igpu") or {}).get("tail", ""))[-2500:] + "\n```")
    return "\n".join(rows)
