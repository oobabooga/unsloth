#!/usr/bin/env python3
"""Criteria: tiny Unsloth LoRA SFT + DPO, unsloth-zoo base vs head (same unsloth commit), on this GPU.

Pairs with probes/zoo_train_probe.py. Per job (sft, dpo), repeat 0 of each state:
  REGRESSION if at any logged step |head_loss - base_loss| > REL_TOL * max(|base_loss|, 1e-6),
  or if the head's peak allocated memory (median over repeats) > MEM_TOL * base's.
Step time (steady median, first step = compile) is REPORTED with the base A/A spread, never judged: one tiny run on a
shared pool is not a timing measurement.
Gates (INCONCLUSIVE when any fails): every base/head run finished and passed its own job checks with max_steps logged
steps; ran on a HIP GPU; each state imported unsloth_zoo from its own checkout at the expected SHA; unsloth is the same
file + revision in both states; the base A/A losses agree within REL_TOL (else the tolerance is below the noise); the
head engaged the new fused_linear path (CALLS ce / logprobs > 0) with no device left in _TRITON_BROKEN.
"""

from __future__ import annotations

import statistics

TITLE = "Tiny Unsloth LoRA SFT + DPO on gfx1151: zoo base vs head, same unsloth"
MODE = "regression"
NEEDS = ["rocm", "gpu", "discrete_gpu", "windows", "nvidia", "multi_gpu", "mig", "xpu", "mlx"]
REL_TOL = 1e-2
MEM_TOL = 1.05
# Which fused_linear counter proves the head's new path ran, per job.
ENGAGE = {"sft": ("ce",), "dpo": ("logprobs",)}


def _runs(state: dict, job: str) -> list:
    return ((state or {}).get("runs") or {}).get(job) or []


def _losses(run: dict) -> list:
    return [s.get("loss") for s in ((run.get("metrics") or {}).get("steps") or []) if s.get("loss") is not None]


def _peak(runs: list):
    v = [((r.get("metrics") or {}).get("summary") or {}).get("peak_mem_gb") for r in runs]
    v = [x for x in v if isinstance(x, (int, float))]
    return statistics.median(v) if v else None


def _steady(runs: list) -> list:
    v = [((r.get("metrics") or {}).get("summary") or {}).get("steady_step_ms") for r in runs]
    return [x for x in v if isinstance(x, (int, float))]


def _first(runs: list) -> list:
    v = [((r.get("metrics") or {}).get("summary") or {}).get("first_step_ms") for r in runs]
    return [x for x in v if isinstance(x, (int, float))]


def _jobs(obs: dict) -> list:
    return sorted(set((obs.get("base") or {}).get("runs") or {}) | set((obs.get("head") or {}).get("runs") or {}))


def _maxrel(a: list, b: list):
    if not a or len(a) != len(b):
        return None
    return max(abs(x - y) / max(abs(y), 1e-6) for x, y in zip(a, b))


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out = []
    jobs = _jobs(obs)
    out.append(("jobs present at base and head", bool(jobs), str(jobs)))
    uns = {}
    for name in ("base", "head"):
        st = obs.get(name) or {}
        co, sha = st.get("checkout") or "?", st.get("zoo_sha") or "?"
        for job in jobs:
            runs = _runs(st, job)
            bad = []
            for r in runs:
                m, c = r.get("metrics") or {}, r.get("calls") or {}
                n = len(_losses(r))
                steps = len((m.get("steps") or []))
                failed = [k for k, v in (m.get("checks") or {}).items() if not v.get("ok")]
                if r.get("rc") != 0 or not m.get("passed") or not m.get("finished") or failed:
                    bad.append(f"r{r.get('rep')} rc={r.get('rc')} passed={m.get('passed')} failed_checks={failed} "
                               f"err={str(m.get('error') or r.get('error') or r.get('stderr_tail'))[-300:]}")
                if n == 0 or n != steps:
                    bad.append(f"r{r.get('rep')} {n} losses over {steps} steps")
                t = c.get("torch") or {}
                if not (t.get("is_available") and t.get("hip") and m.get("device") == "rocm"):
                    bad.append(f"r{r.get('rep')} not on a HIP GPU: {t} device={m.get('device')}")
                zf = c.get("unsloth_zoo_file") or ""
                zrev = (m.get("revisions") or {}).get("unsloth_zoo")
                if not zf.startswith(co) or (zrev or "").removesuffix("-dirty") != sha:
                    bad.append(f"r{r.get('rep')} zoo {zf} rev {zrev} != checkout {co} @ {sha[:9]}")
                uns.setdefault((c.get("unsloth_file"), (m.get("revisions") or {}).get("unsloth"),
                                (m.get("versions") or {}).get("unsloth")), []).append(f"{name}/{job}/r{r.get('rep')}")
            out.append((f"{name}/{job}: {len(runs)} runs finished, passed, on HIP, own zoo", bool(runs) and not bad,
                        "; ".join(bad) or f"zoo {sha[:9]}, {len(runs)} runs"))
    out.append(("same unsloth in every run", len(uns) == 1, "; ".join(f"{k}: {len(v)} runs" for k, v in uns.items())))
    base, head = obs.get("base") or {}, obs.get("head") or {}
    for job in jobs:
        br = _runs(base, job)
        if len(br) >= 2:
            aa = _maxrel(_losses(br[1]), _losses(br[0]))
            out.append((f"base/{job}: A/A losses agree within {REL_TOL:g}", aa is not None and aa <= REL_TOL,
                        f"max rel diff {aa}"))
        hr = _runs(head, job)
        need = ENGAGE.get(job, ("ce", "logprobs"))
        engaged = [sum(((r.get("calls") or {}).get("calls") or {}).get(k, 0) for k in need) for r in hr]
        broken = [(r.get("calls") or {}).get("triton_broken") for r in hr]
        out.append((f"head/{job}: new fused_linear path engaged ({'+'.join(need)})",
                    bool(hr) and all(e > 0 for e in engaged) and all(b == [] for b in broken),
                    f"CALLS per run {[(r.get('calls') or {}).get('calls') for r in hr]}; _TRITON_BROKEN {broken}"))
    return out


def table(obs: dict) -> str:
    base, head = obs.get("base") or {}, obs.get("head") or {}
    lines = ["| job | base steady ms (reps) | head steady ms (reps) | base first ms | head first ms | base peak GiB | "
             "head peak GiB | max rel loss diff (head r0 vs base r0) | base A/A |", "|---|---|---|---|---|---|---|---|---|"]
    steps = []
    for job in _jobs(obs):
        br, hr = _runs(base, job), _runs(head, job)
        aa = _maxrel(_losses(br[1]), _losses(br[0])) if len(br) >= 2 else None
        d = _maxrel(_losses(hr[0]), _losses(br[0])) if br and hr else None
        lines.append(f"| {job} | {_steady(br)} | {_steady(hr)} | {_first(br)} | {_first(hr)} | {_peak(br)} | "
                     f"{_peak(hr)} | {d} | {aa} |")
        if br and hr:
            for i, (b, h) in enumerate(zip(_losses(br[0]), _losses(hr[0])), 1):
                steps.append(f"| {job} | {i} | {b:.6f} | {h:.6f} | {abs(h - b) / max(abs(b), 1e-6):.2e} |")
    return "\n".join(lines + ["", "| job | step | base loss | head loss | rel diff |", "|---|---|---|---|---|"] + steps)


def head_is_worse(base: dict, head: dict) -> tuple[bool, str]:
    bad, notes = [], []
    for job in sorted(set(base.get("runs") or {}) | set(head.get("runs") or {})):
        br, hr = _runs(base, job), _runs(head, job)
        if not br or not hr:
            bad.append(f"{job}: missing runs")
            continue
        bl, hl = _losses(br[0]), _losses(hr[0])
        if len(bl) != len(hl) or not bl:
            bad.append(f"{job}: {len(bl)} base vs {len(hl)} head losses")
            continue
        over = [(i + 1, b, h) for i, (b, h) in enumerate(zip(bl, hl)) if abs(h - b) > REL_TOL * max(abs(b), 1e-6)]
        if over:
            bad.append(f"{job}: loss off by > {REL_TOL:g} rel at steps " + ", ".join(f"{i} ({b:.5f} -> {h:.5f})" for i, b, h in over[:5]))
        bm, hm = _peak(br), _peak(hr)
        if bm is None or hm is None:
            bad.append(f"{job}: peak memory missing")
        elif hm > MEM_TOL * bm:
            bad.append(f"{job}: peak memory {bm} -> {hm} GiB (> {MEM_TOL}x)")
        notes.append(f"{job}: max rel loss diff {_maxrel(hl, bl):.2e}, peak {bm} -> {hm} GiB, "
                     f"steady {_steady(br)} -> {_steady(hr)} ms")
    return bool(bad), "; ".join(bad or notes)
