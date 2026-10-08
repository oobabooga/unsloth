#!/usr/bin/env python3
"""Criteria: does an idle 903 backend burn CPU on Windows ROCm where 902 does not? (#12942)

Pairs with probes/studio_idle_probe.py. base = v0.1.902-beta, head = v0.1.903-beta. An arm is
HOT when its mean busy cores over the last 60 s of the idle window is >= HOT_CORES; the head is
worse when its default arm is hot and the base's is not (gap >= 1 core).
"""

from __future__ import annotations

TITLE = "Idle Studio backend CPU on Windows ROCm, 902 vs 903 (#12942)"
MODE = "regression"
NEEDS = ["windows", "rocm", "gpu", "windows_rocm_wddm", "discrete_gpu"]
HOT_CORES = 2.0


def _states(obs: dict) -> dict:
    return {k: v for k, v in obs.items() if not k.startswith("_")}


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    st = _states(obs)
    out = []
    errs = {k: v.get("probe_error") for k, v in st.items() if v.get("probe_error")}
    out.append(("every state built and probed", not errs, str(errs)[:600] if errs else ""))
    arm_err = {f"{k}:{a}": r.get("error") for k, v in st.items() for a, r in (v.get("arms") or {}).items()
               if r.get("error")}
    out.append(("every arm reached healthy and stayed up", not arm_err and all(v.get("arms") for v in st.values()),
                str(arm_err)[:600]))
    polled = {f"{k}:{a}": (r.get("poll_status") or {}).get("/api/inference/status", {}).get("200", 0)
              for k, v in st.items() for a, r in (v.get("arms") or {}).items()}
    out.append(("the Desktop-style poller was authenticated (status 200s)", all(n > 0 for n in polled.values()),
                str(polled)))
    sampled = {f"{k}:{a}": r.get("timeline", [{}])[-1].get("threads") if r.get("timeline") else None
               for k, v in st.items() for a, r in (v.get("arms") or {}).items()}
    out.append(("the sampled process is the interpreter (> 1 thread)",
                all((t or 0) > 1 for t in sampled.values()), str(sampled)))
    return out


def table(obs: dict) -> str:
    st = _states(obs)
    rows = ["| state | tag | arm | healthy s | busy cores last 60 s | max | threads | threads > 0.5 core |",
            "|---|---|---|---|---|---|---|---|"]
    for k, v in st.items():
        for a, r in (v.get("arms") or {}).items():
            thr = r["timeline"][-1]["threads"] if r.get("timeline") else None
            rows.append(f"| {k} | {v.get('tag')} | {a} | {r.get('healthy_s')} | {r.get('busy_last60')} | "
                        f"{r.get('busy_max')} | {thr} | {r.get('threads_over_half_core')} |")
    return "\n".join(rows)


def _busy(state: dict, arm: str = "default") -> float | None:
    return ((state.get("arms") or {}).get(arm) or {}).get("busy_last60")


def head_is_worse(base: dict, head: dict) -> tuple[bool, str]:
    b, h = _busy(base), _busy(head)
    if b is None or h is None:
        return False, "no reading"
    if h >= HOT_CORES and h - b >= 1.0:
        return True, f"idle 903 backend burns {h:.2f} cores vs {b:.2f} on 902"
    return False, f"idle busy cores 902={b:.2f} 903={h:.2f}"
