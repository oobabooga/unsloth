#!/usr/bin/env python3
"""Criteria: a returning process must reuse the reduction configs a first process tuned.

Base (no pin): the compile workers save each tuned config under torch's default cache_key_tag while the tagged
process looks it up under its own, so the returning start benchmarks again (bench > 0). Head pins the lookup key and
benchmarks nothing on the returning start. The gates make sure the runner is ROCm, both processes compiled, the
first start really autotuned something and the compile-worker pool was in use, so neither verdict can pass by the
probe doing nothing.
"""

from __future__ import annotations

TITLE = "Returning process reuses tuned Triton reduction configs"
MODE = "differential"
NEEDS = ["gpu", "windows_rocm_wddm", "multi_gpu", "nvidia", "mig", "xpu", "mlx"]


def _states(obs: dict) -> dict:
    return {n: v for n, v in obs.items() if not n.startswith("_")}


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    states = _states(obs)
    out = []
    hip = {n: (v.get("first") or {}).get("hip") for n, v in states.items()}
    out.append(("ROCm torch on a GPU", all(h for h in hip.values()), ", ".join(f"{k}={v}" for k, v in hip.items())))
    ok = {n: all((v.get(p) or {}).get("ok") for p in ("first", "returning")) for n, v in states.items()}
    out.append(("both processes compiled and ran in every state", all(ok.values()),
                ", ".join(f"{k}={(v.get('first') or {}).get('error') or (v.get('returning') or {}).get('error') or 'ok'}"
                          for k, v in states.items())))
    tuned = {n: (v.get("first") or {}).get("bench") or 0 for n, v in states.items()}
    out.append(("first start autotuned reductions (bench > 0)", all(t > 0 for t in tuned.values()),
                ", ".join(f"{k}={v}" for k, v in tuned.items())))
    threads = {n: (v.get("first") or {}).get("compile_threads") or 0 for n, v in states.items()}
    out.append(("compile-worker pool in use (compile_threads > 1)", all(t > 1 for t in threads.values()),
                ", ".join(f"{k}={v}" for k, v in threads.items())))
    return out


def table(obs: dict) -> str:
    rows = ["| state | device | torch | helper | pinned | first bench | first s | returning bench | returning s |",
            "|---|---|---|---|---|---|---|---|---|"]
    for name, v in _states(obs).items():
        f, r = v.get("first") or {}, v.get("returning") or {}
        rows.append(f"| {name} | {f.get('device')} {f.get('arch')} | {f.get('torch')} | {r.get('helper_present')} | "
                    f"{r.get('pinned')} | {f.get('bench')} | {f.get('seconds')} | {r.get('bench')} | {r.get('seconds')} |")
    return "\n".join(rows)


def base_shows_defect(base: dict) -> bool:
    return ((base.get("returning") or {}).get("bench") or 0) > 0


def head_is_fixed(head: dict) -> bool:
    r = head.get("returning") or {}
    return r.get("pinned") is True and r.get("bench") == 0
