#!/usr/bin/env python3
"""Criteria: after run.py's configure_cpu_threads(), is rocm-openblas.dll capped on Windows ROCm? (#12942, PR #13048)

Pairs with probes/openblas_cap_probe.py. Base shows the defect when the DLL still reports more than one thread
after configure_cpu_threads() and sustained CPU BLAS fans out over >= FANOUT_CORES cores; the head is fixed when
the DLL reports one thread, the same BLAS load stays at or under CAPPED_CORES, and the real backend boots healthy
and idles under IDLE_CORES while polled.
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


def _threads(state: dict):
    return _med([(r.get("child") or {}).get("openblas_threads") for r in state.get("cap", [])])


def _busy(state: dict):
    return _med([r.get("busy_cores_under_blas") for r in state.get("cap", [])])


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    st = _states(obs)
    out = []
    errs = {k: v.get("probe_error") for k, v in st.items() if v.get("probe_error")}
    out.append(("every state built and probed", not errs, str(errs)[:600] if errs else ""))
    hip = {k: [(r.get("child") or {}).get("hip") for r in v.get("cap", [])] for k, v in st.items()}
    out.append(("every cap run imported a ROCm torch", all(h and all(h) for h in hip.values()), str(hip)))
    loaded = {k: [(r.get("child") or {}).get("dll_loaded") for r in v.get("cap", [])] for k, v in st.items()}
    out.append(("rocm-openblas.dll was loaded in every cap run", all(x and all(x) for x in loaded.values()),
                str(loaded)))
    env1 = {k: [(r.get("child") or {}).get("env_openblas") for r in v.get("cap", [])] for k, v in st.items()}
    out.append(("configure_cpu_threads() set OPENBLAS_NUM_THREADS=1 in every state",
                all(x and all(e == "1" for e in x) for x in env1.values()), str(env1)))
    mm = {k: [(r.get("child") or {}).get("matmul_ok") for r in v.get("cap", [])] for k, v in st.items()}
    out.append(("the CPU matmul ran and is finite", all(x and all(x) for x in mm.values()), str(mm)))
    return out


def table(obs: dict) -> str:
    rows = ["| state | wheel | DLL threads after configure_cpu_threads() | busy cores under sustained CPU BLAS (median, runs) "
            "| Studio --api-only healthy s | idle busy cores (last 60 s, polled) |",
            "|---|---|---|---|---|---|"]
    for k, v in _states(obs).items():
        runs = [r.get("busy_cores_under_blas") for r in v.get("cap", [])]
        s = v.get("studio") or {}
        rows.append(f"| {k} | {v.get('wheel')} | {_threads(v)} | {_busy(v)} {runs} | {s.get('healthy_s')} | "
                    f"{s.get('busy_last60')} |")
    rows.append("")
    rows.append(f"Defect: > 1 DLL thread and >= {FANOUT_CORES} busy cores. Fixed: 1 thread, <= {CAPPED_CORES} busy "
                f"cores, backend healthy and idle < {IDLE_CORES} cores.")
    return "\n".join(rows)


def base_shows_defect(base: dict) -> bool:
    t, b = _threads(base), _busy(base)
    return t is not None and b is not None and t > 1 and b >= FANOUT_CORES


def head_is_fixed(head: dict) -> bool:
    t, b = _threads(head), _busy(head)
    s = head.get("studio") or {}
    idle = s.get("busy_last60")
    return (t == 1 and b is not None and b <= CAPPED_CORES and s.get("healthy_s") is not None
            and not s.get("error") and idle is not None and idle < IDLE_CORES)
