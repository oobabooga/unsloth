#!/usr/bin/env python3
"""Criteria: Studio's OpenBLAS default under a Windows job memory cap (#12374, PR #13048).

Pairs with probes/openblas_memcap_probe.py. base = main (default 1), head = the memory-aware default. Differential on
the `studio` arm: the base shows the defect when a capped start dies (OpenBLAS cannot get its buffers); the head is
fixed when it survives every cap and still picks 8 numpy threads uncapped. The unset and fixed8 arms show what a fixed default would do under the same caps.
"""

from __future__ import annotations

TITLE = "OpenBLAS default under a Windows job memory cap (PR #13048)"
MODE = "differential"
NEEDS = ["windows"]


def _states(obs: dict) -> dict:
    return {k: v for k, v in obs.items() if not k.startswith("_")}


def _cells(state: dict, arm: str, cap: int) -> list:
    return [c for c in state.get("cells", []) if c.get("arm") == arm and c.get("cap_mb") == cap]


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    st = _states(obs)
    errs = {k: v.get("probe_error") for k, v in st.items() if v.get("probe_error")}
    jobs = {k: sorted({(c.get("job_set"), c.get("job_assigned")) for c in v.get("cells", []) if c.get("cap_mb")})
            for k, v in st.items()}
    return [("every state built and probed", not errs, str(errs)[:800]),
            ("the job memory cap was set and the child assigned to it", all(j == [(True, True)] for j in jobs.values()),
             str(jobs))]


def table(obs: dict) -> str:
    rows = ["| state | cap MB | arm | survived (runs) | OPENBLAS_NUM_THREADS | rocm-openblas threads | headroom MB | committed MB after numpy | after torch | died at |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for k, v in _states(obs).items():
        for cap in (0, 5000, 3500, 2500):
            for arm in ("studio", "unset", "fixed8"):
                cs = _cells(v, arm, cap)
                if not cs:
                    continue
                died = sorted({c.get("stage") for c in cs if not c.get("ok")})
                rows.append(f"| {k} | {cap or 'none'} | {arm} | {sum(c['ok'] for c in cs)}/{len(cs)} | "
                            f"{cs[0].get('openblas_env')} | {cs[0].get('dll_threads')} | {cs[0].get('headroom_mb')} | {cs[0].get('numpy_private_mb')} | "
                            f"{cs[0].get('torch_private_mb')} | {', '.join(d or '' for d in died)} |")
    return "\n".join(rows)


def base_shows_defect(base: dict) -> bool:
    capped = [c for c in base.get("cells", []) if c.get("arm") == "studio" and c.get("cap_mb")]
    return any(not c.get("ok") for c in capped)


def head_is_fixed(head: dict) -> bool:
    studio = [c for c in head.get("cells", []) if c.get("arm") == "studio"]
    uncapped = _cells(head, "studio", 0)
    return bool(studio) and all(c.get("ok") for c in studio) and all(c.get("openblas_env") == "8" for c in uncapped)
