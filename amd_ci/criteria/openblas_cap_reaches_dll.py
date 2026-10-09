#!/usr/bin/env python3
"""Criteria: does the configured thread count reach rocm-openblas.dll on Windows ROCm? (#12942, PR #13048)

Pairs with probes/openblas_cap_probe.py. Base shows the defect when a user's OPENBLAS_NUM_THREADS=1 is ignored:
the DLL reports more than one thread and sustained CPU BLAS fans out over >= FANOUT_CORES cores. The head is
fixed when that user value gives one thread and at most CAPPED_CORES busy cores, Studio's own default gives the
DLL torch's thread count (backend and a Desktop-spawned worker), and the real backend boots healthy and idles
under IDLE_CORES while polled. The perf rows (plain numpy import, torch CPU matmul per DLL thread count) are
reported, not gated.
"""

from __future__ import annotations

import statistics

TITLE = "OpenBLAS thread cap reaches rocm-openblas.dll on Windows ROCm (PR #13048)"
MODE = "differential"
NEEDS = ["windows", "rocm", "gpu", "windows_rocm_wddm", "discrete_gpu"]
FANOUT_CORES = 4.0
CAPPED_CORES = 1.5
IDLE_CORES = 0.3


def _states(obs: dict) -> dict:
    return {k: v for k, v in obs.items() if not k.startswith("_")}


def _med(vals):
    vals = [v for v in vals if v is not None]
    return statistics.median(vals) if vals else None


def _threads(state: dict, key: str = "cap"):
    return _med([(r.get("child") or {}).get("openblas_threads") for r in state.get(key) or []])


def _torch_threads(state: dict, key: str = "cap"):
    return _med([(r.get("child") or {}).get("torch_threads") for r in state.get(key) or []])


def _busy(state: dict, key: str = "cap"):
    return _med([r.get("busy_cores_under_blas") for r in state.get(key) or []])


def _runs(v: dict) -> list:
    return list(v.get("cap") or []) + list(v.get("cap_user1") or [])


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    st = _states(obs)
    out = []
    errs = {k: v.get("probe_error") for k, v in st.items() if v.get("probe_error")}
    out.append(("every state built and probed", not errs, str(errs)[:600] if errs else ""))
    hip = {k: [(r.get("child") or {}).get("hip") for r in _runs(v)] for k, v in st.items()}
    out.append(("every cap run imported a ROCm torch", all(h and all(h) for h in hip.values()), str(hip)))
    loaded = {k: [(r.get("child") or {}).get("dll_loaded") for r in _runs(v)] for k, v in st.items()}
    out.append(("rocm-openblas.dll was loaded in every cap run", all(x and all(x) for x in loaded.values()),
                str(loaded)))
    env1 = {k: [(r.get("child") or {}).get("env_openblas") for r in _runs(v)] for k, v in st.items()}
    out.append(("configure_cpu_threads() set OPENBLAS_NUM_THREADS=1 in every state",
                all(x and all(e == "1" for e in x) for x in env1.values()), str(env1)))
    mm = {k: [(r.get("child") or {}).get("matmul_ok") for r in _runs(v)] for k, v in st.items()}
    out.append(("the CPU matmul ran and is finite", all(x and all(x) for x in mm.values()), str(mm)))
    n = {k: (len(v.get("cap") or []), len(v.get("cap_user1") or [])) for k, v in st.items()}
    out.append(("every state ran the default and the user OPENBLAS_NUM_THREADS=1 arms",
                all(a > 0 and b > 0 for a, b in n.values()), str(n)))
    wk = {k: (v.get("worker") or {}) for k, v in st.items()}
    out.append(("the worker arm imported core.training.worker and loaded rocm-openblas.dll",
                all(w.get("worker_import") == "ok" and w.get("dll_loaded") for w in wk.values()),
                str({k: (w.get("worker_import"), w.get("dll_loaded"), w.get("error")) for k, w in wk.items()})[:600]))
    return out


def table(obs: dict) -> str:
    rows = ["| state | wheel | user OPENBLAS_NUM_THREADS=1: DLL threads | busy cores (runs) | Studio default: DLL threads "
            "/ torch threads | busy cores (runs) | Desktop-spawned worker: DLL / torch threads | --api-only healthy s "
            "| idle busy cores |",
            "|---|---|---|---|---|---|---|---|---|"]
    for k, v in _states(obs).items():
        s = v.get("studio") or {}
        w = v.get("worker") or {}
        rows.append(f"| {k} | {v.get('wheel')} | {_threads(v, 'cap_user1')} | {_busy(v, 'cap_user1')} "
                    f"{[r.get('busy_cores_under_blas') for r in v.get('cap_user1') or []]} | {_threads(v)} / "
                    f"{_torch_threads(v)} | {_busy(v)} {[r.get('busy_cores_under_blas') for r in v.get('cap') or []]} | "
                    f"{w.get('openblas_threads')} / {w.get('torch_threads')} (marker {w.get('env_marker')}) | "
                    f"{s.get('healthy_s')} | {s.get('busy_last60')} |")
    rows.append("")
    rows.append(f"Defect: a user's OPENBLAS_NUM_THREADS=1 leaves > 1 DLL thread and >= {FANOUT_CORES} busy cores. "
                f"Fixed: that gives 1 thread and <= {CAPPED_CORES} busy cores, Studio's default gives the DLL torch's "
                f"thread count in the backend and the worker, backend healthy and idle < {IDLE_CORES} cores.")
    for k, v in _states(obs).items():
        perf = v.get("perf") or []
        if not perf:
            continue
        rows += ["", f"Perf and memory, one fresh process per cell (venv of `{k}`; numbers are medians over 3 reps):", "",
                 "| cell | OPENBLAS_NUM_THREADS | threads after import | private MB after import | threads after BLAS "
                 "| private MB after BLAS | 2048 matmul ms | small op ms |",
                 "|---|---|---|---|---|---|---|---|"]
        cells = {}
        for r in perf:
            name = r.get("kind") if r.get("kind") != "numpy" else f"numpy ({r.get('numpy')})"
            if r.get("kind") in ("one", "torch", "logical"):
                name = f"torch, DLL at {r.get('dll_threads')} ({r.get('kind')})"
            cells.setdefault((name, r.get("env")), []).append(r)
        for (name, env), rs in cells.items():
            def m(f):
                return _med([f(r) for r in rs])
            rows.append(f"| {name} | {env} | {m(lambda r: (r.get('after_import') or {}).get('threads'))} | "
                        f"{m(lambda r: (r.get('after_import') or {}).get('private_mb'))} | "
                        f"{m(lambda r: (r.get('after_blas') or {}).get('threads'))} | "
                        f"{m(lambda r: (r.get('after_blas') or {}).get('private_mb'))} | {m(lambda r: r.get('matmul_2048_ms'))} | "
                        f"{m(lambda r: r.get('cosine_1000x1024_ms') or r.get('linear_64x1500x384_ms'))} |")
        errs = [r for r in perf if r.get("error")]
        if errs:
            rows.append(f"perf errors: {str(errs)[:1500]}")
        rows.append(f"small op: numpy cosine of 1000 x 1024 against one vector; torch Linear(384, 1536) on 64 x 1500 x 384. "
                    f"logical CPUs {perf[0].get('logical')}, physical {perf[0].get('physical')}.")
    return "\n".join(rows)


def base_shows_defect(base: dict) -> bool:
    t, b = _threads(base, "cap_user1"), _busy(base, "cap_user1")
    return t is not None and b is not None and t > 1 and b >= FANOUT_CORES


def head_is_fixed(head: dict) -> bool:
    t, b = _threads(head, "cap_user1"), _busy(head, "cap_user1")
    s = head.get("studio") or {}
    w = head.get("worker") or {}
    idle = s.get("busy_last60")
    default_ok = _threads(head) is not None and _threads(head) == _torch_threads(head)
    worker_ok = w.get("openblas_threads") is not None and w.get("openblas_threads") == w.get("torch_threads")
    return (t == 1 and b is not None and b <= CAPPED_CORES and default_ok and worker_ok
            and s.get("healthy_s") is not None
            and not s.get("error") and idle is not None and idle < IDLE_CORES)
