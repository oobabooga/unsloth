#!/usr/bin/env python3
"""Criteria: does the head's ROCm torch wheel leave CPU threads spinning where the base's does not?

Pairs with probes/idle_cpu_probe.py (issue #12942). Run against merged PR #12670, whose parent
pins rocm7.14.1 and whose merge commit pins rocm7.14.0, so base vs head differ by the wheel only.

An arm SPINS when its median idle busy-cores is >= SPIN_CORES: one idle process holding two
whole cores is not a sampling artefact. The head is worse when an arm spins there, not at the
base, and the gap clears both a 1-core floor and 3x the larger repeat spread (A/A noise).
"""

from __future__ import annotations

import statistics

TITLE = "Idle CPU spin of the Windows ROCm torch wheel (#12942)"
MODE = "regression"
NEEDS = ["windows", "rocm", "gpu", "windows_rocm_wddm", "discrete_gpu"]
SPIN_CORES = 2.0
FLOOR_CORES = 0.3


def _states(obs: dict) -> dict:
    return {k: v for k, v in obs.items() if not k.startswith("_")}


def _busy(state: dict, arm: str) -> list[float]:
    return [r["busy_cores"] for r in (state.get("arms") or {}).get(arm, []) if "busy_cores" in r]


def _stat(vals: list[float]) -> tuple[float, float]:
    if not vals:
        return float("nan"), float("nan")
    return statistics.median(vals), max(vals) - min(vals)


def _first_child(state: dict, arm: str) -> dict:
    for r in (state.get("arms") or {}).get(arm, []):
        if r.get("child"):
            return r["child"]
    return {}


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    st = _states(obs)
    out = []
    errs = {k: v.get("probe_error") for k, v in st.items() if v.get("probe_error")}
    out.append(("every state probed without error", not errs, str(errs) if errs else ""))
    tags = {k: v.get("tag") for k, v in st.items()}
    out.append(("base and head pin different wheels",
                len(set(tags.values())) == len(tags) and None not in tags.values(), str(tags)))
    hip = {k: _first_child(v, "import").get("hip") for k, v in st.items()}
    out.append(("torch is a ROCm build in every state", all(hip.values()), str(hip)))
    gpu = {k: _first_child(v, "gpu_init").get("cuda_available") for k, v in st.items()}
    out.append(("the GPU is visible to torch in every state", all(gpu.values()), str(gpu)))
    floor = {k: _stat(_busy(v, "noop"))[0] for k, v in st.items()}
    out.append(("a bare python.exe idles below the floor",
                all(f == f and f < FLOOR_CORES for f in floor.values()),
                ", ".join(f"{k}={f:.2f}" for k, f in floor.items())))
    missing = [f"{k}:{a}" for k, v in st.items() for a in (v.get("arms") or {})
               if len(_busy(v, a)) < len((v.get("arms") or {})[a])]
    out.append(("every arm produced a reading", not missing and all(v.get("arms") for v in st.values()),
                ", ".join(missing)))
    mm = {k: _first_child(v, "matmul").get("matmul_ok") for k, v in st.items()}
    out.append(("the CPU matmul ran and is finite", all(mm.values()), str(mm)))
    return out


def table(obs: dict) -> str:
    st = _states(obs)
    names = list(st)
    arms = list((st[names[0]].get("arms") or {}).keys()) if names else []
    head = "| arm | " + " | ".join(f"{n} ({st[n].get('tag')}) busy cores, median [min-max] / threads / hot" for n in names) + " |"
    rows = [head, "|---|" + "---|" * len(names)]
    for a in arms:
        cells = []
        for n in names:
            vals = _busy(st[n], a)
            med, _ = _stat(vals)
            recs = [r for r in st[n]["arms"].get(a, []) if "busy_cores" in r]
            thr = max((r["threads"] for r in recs), default = 0)
            hot = max((r["threads_over_half_core"] for r in recs), default = 0)
            rng = f"[{min(vals):.2f}-{max(vals):.2f}]" if vals else "[-]"
            cells.append(f"{med:.2f} {rng} / {thr} / {hot}")
        rows.append(f"| {a} | " + " | ".join(cells) + " |")
    rows.append("")
    for n in names:
        ob = _first_child(st[n], "matmul").get("openblas") or []
        for rec in ob:
            rows.append(f"- {n}: `{rec.get('path')}` config=`{rec.get('config')}` "
                        f"parallel={rec.get('openblas_get_parallel')} "
                        f"num_threads={rec.get('openblas_get_num_threads')} {rec.get('error', '')}")
        env1 = _first_child(st[n], "matmul_env1").get("openblas") or []
        for rec in env1:
            rows.append(f"- {n} with OPENBLAS_NUM_THREADS=1: num_threads={rec.get('openblas_get_num_threads')}")
        rows.append(f"- {n}: torch {_first_child(st[n], 'import').get('torch')}, "
                    f"hip {_first_child(st[n], 'import').get('hip')}, cpu_count {st[n].get('cpu_count')}")
    rows.append("")
    rows.append(f"An arm spins at median >= {SPIN_CORES} busy cores; noise floor gate < {FLOOR_CORES}.")
    return "\n".join(rows)


def head_is_worse(base: dict, head: dict) -> tuple[bool, str]:
    worse = []
    for arm in (head.get("arms") or {}):
        bm, bs = _stat(_busy(base, arm))
        hm, hs = _stat(_busy(head, arm))
        if hm != hm or bm != bm:
            continue
        margin = max(1.0, 3 * max(bs, hs))
        if hm >= SPIN_CORES and hm - bm > margin:
            worse.append(f"{arm}: {bm:.2f} -> {hm:.2f} cores")
    if worse:
        return True, "head spins idle where base does not: " + "; ".join(worse)
    spinning_both = [a for a in (head.get("arms") or {})
                     if _stat(_busy(head, a))[0] >= SPIN_CORES and _stat(_busy(base, a))[0] >= SPIN_CORES]
    if spinning_both:
        return False, "both wheels spin in: " + ", ".join(spinning_both) + " (wheel change is not the differentiator)"
    return False, "no arm spins on either wheel"
