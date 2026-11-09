#!/usr/bin/env python3
"""Criteria: Mega-cache on Windows gfx1151, zoo main (base) against a zoo PR (head).

Judges the observations of probes/megacache_probe.py. The question this run exists for is
descriptive (does the Mega-cache save, load and hit on Windows; does anything compile at all), so
the table carries the evidence and the verdict only answers "is head worse than base":

- a run (workload, model, arm, run) that completed on base raises on head;
- a warm run with the cache dir wiped hits fewer FX / AOTAutograd entries on head than on base;
- a compile_cache test that passes (or does not exist) on base fails on head, including head's
  own new tests, since a test the PR adds that fails on Windows is a Windows regression.

Gates: both states ran the probe and imported torch, and torch saw the GPU (a ROCm torch that
does not is an install fault, not a finding).
"""

from __future__ import annotations

TITLE = "Mega-cache on Windows gfx1151: zoo main vs zoo PR"
MODE = "regression"
NEEDS = ["gpu", "rocm", "windows", "windows_rocm_wddm", "exclusive_gpu", "nvidia", "multi_gpu", "linux"]


def _states(obs):
    return [(n, obs[n]) for n in ("base", "head", "merge") if n in obs]


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out = []
    for name in ("base", "head"):
        o = obs.get(name) or {}
        runs = o.get("runs") or []
        env = o.get("env") or {}
        out.append((f"{name} probe ran child processes", bool(runs),
                    f"{len(runs)} runs" + (f"; error {o.get('error')}" if o.get("error") else "")))
        out.append((f"{name} imported torch", bool(env.get("torch")),
                    str(env.get("torch") or env.get("torch_error") or env.get("env_error"))[:200]))
        out.append((f"{name} torch saw the GPU", bool(env.get("cuda_available")),
                    f"hip={env.get('hip')} device={env.get('device_name')} arch={env.get('gcn_arch')}"))
    return out


def _key(r):
    return (r.get("workload"), r.get("model"), r.get("arm"), r.get("run"))


def _hits(r):
    return ((r.get("inductor") or {}).get("fxgraph_cache_hit", 0) or 0,
            (r.get("aot") or {}).get("autograd_cache_hit", 0) or 0)


def head_is_worse(base: dict, head: dict):
    problems = []
    b_runs = {_key(r): r for r in base.get("runs") or []}
    for r in head.get("runs") or []:
        b = b_runs.get(_key(r))
        if b is None:
            continue
        if not b.get("error") and r.get("error"):
            problems.append(f"{'/'.join(str(x) for x in _key(r))}: base ok, head raised {r['error'][:160]}")
        if r.get("run") == "wiped" and _hits(r) < _hits(b):
            problems.append(f"{'/'.join(str(x) for x in _key(r))}: warm hits FX/AOT {_hits(b)} on base, "
                            f"{_hits(r)} on head")
    b_fail = set((base.get("pytest") or {}).get("failed") or [])
    h_fail = set((head.get("pytest") or {}).get("failed") or [])
    new = sorted(h_fail - b_fail)
    if new:
        problems.append("compile_cache tests failing on head only: " + ", ".join(t.split("::")[-1] for t in new[:12]))
    if problems:
        return True, "; ".join(problems)
    return False, "no run raised on head that passed on base, warm hit counts held, no new test failures"


def _fmt_run(r):
    i, a = r.get("inductor") or {}, r.get("aot") or {}
    mc = r.get("megacache") or {}
    frames = (r.get("frames") or {}).get("total", 0)
    err = (r.get("error") or "")[:90].replace("|", "/")
    log = (r.get("megacache_log") or [""])[-1].replace("Unsloth: Mega-cache ", "")[:70].replace("|", "/")
    return (f"| {r.get('workload')} | {str(r.get('model') or '')[-22:]} | {r.get('arm')} | {r.get('run')} | "
            f"{r.get('device', '')} | {frames} | {i.get('fxgraph_cache_hit', 0)}/{i.get('fxgraph_cache_miss', 0)}/"
            f"{i.get('fxgraph_cache_bypass', 0)} | {a.get('autograd_cache_hit', 0)}/{a.get('autograd_cache_miss', 0)} | "
            f"{mc.get('armed')}/{mc.get('hit')} | {(r.get('bundle') or {}).get('bundle_bytes', 0)} | "
            f"{r.get('first_step_s')} | {r.get('process_wall_s')} | {log} | {err} |")


def table(obs: dict) -> str:
    rows = []
    for name, o in _states(obs):
        env = o.get("env") or {}
        trust = o.get("trust") or {}
        rows.append(f"**{name}** zoo `{str(o.get('zoo_sha'))[:9]}`: torch {env.get('torch')} hip {env.get('hip')} "
                    f"device {env.get('device_name')} ({env.get('gcn_arch')}), triton "
                    f"{env.get('triton') or env.get('triton_error')}, compile smoke gpu "
                    f"{env.get('compile_smoke_cuda')}, cpu {env.get('compile_smoke_cpu')}, MSVC {o.get('msvc')}")
        rows.append("")
        rows.append(f"trust: default on={trust.get('enabled_by_default')}, default root trusted="
                    f"{trust.get('default_root_trusted')}, fresh dir={trust.get('fresh_trusted_existing')}, "
                    f"Everyone-writable dir={(trust.get('everyone_write') or {}).get('dir_trusted')}, "
                    f"link={(trust.get('link') or {}).get('trusted')}; fsops {o.get('fsops')}")
        p = o.get("pytest") or {}
        rows.append(f"compile_cache tests: {len(p.get('passed') or [])} passed, {len(p.get('failed') or [])} failed, "
                    f"{len(p.get('skipped') or [])} skipped ({', '.join(p.get('files') or [])})")
        rows.append("")
        rows.append("| workload | model | arm | run | dev | frames | FX h/m/b | AOT h/m | armed/hit | bundle B | "
                    "first step s | process s | mega-cache log | error |")
        rows.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        rows += [_fmt_run(r) for r in o.get("runs") or []]
        rows.append("")
    return "\n".join(rows)
