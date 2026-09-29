#!/usr/bin/env python3
"""Criteria: is Studio's Decision API worse at the head than at the base on this machine?

Worse = more changed answers against laya's fp32 CPU forward, a probability error past both 5e-3 and
1.5x the base, peak device or host memory more than 10% (and 64 MiB) above the base, or a workload
more than 25% slower. Pairs with probes/laya_decision_probe.py.
"""

from __future__ import annotations

TITLE = "Decision API (Laya): accuracy, memory and latency, base versus head"
DEVICES = [d for d in ("auto", "cpu") if d in __import__("os").environ.get("LAYA_PROBE_DEVICES", "auto,cpu").split(",")]
MODE = "regression"
NEEDS: list[str] = ["rocm", "gpu"]


def _run(obs, device):
    return ((obs or {}).get("runs") or {}).get(device) or {}


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out = []
    for name in ("base", "head"):
        for device in DEVICES:
            r = _run(obs.get(name), device)
            ok = r.get("rc") == 0 and (r.get("accuracy") or {}).get("answers", 0) > 0
            out.append((f"{name} {device} run answered requests", ok,
                        f"rc={r.get('rc')} answers={(r.get('accuracy') or {}).get('answers')} "
                        f"device={r.get('agent_device')} {str(r.get('stderr_tail', ''))[-200:] if not ok else ''}"))
        r = _run(obs.get(name), "auto")
        out.append((f"{name} ran on the ROCm GPU", bool(r.get("hip")) and r.get("agent_device") == "cuda",
                    f"hip={r.get('hip')} agent_device={r.get('agent_device')} gpu={r.get('gpu_name')}"))
    return out


def table(obs: dict) -> str:
    rows = ["| state | device | dtype | load s | peak RSS MiB | peak torch MiB (worst) | device used over idle MiB | "
            "guardrail ms | briefing ms | worst ms | max dprob | changed | graphs |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name in ("base", "head"):
        for device in DEVICES:
            r = _run(obs.get(name), device)
            w = r.get("workloads") or {}
            ms = lambda k: next((v.get("median_ms") for n, v in w.items() if n.startswith(k)), None)
            worst = next((v.get("peak_allocated_mib") for n, v in w.items() if n.startswith("worst")), None)
            acc = r.get("accuracy") or {}
            rows.append(f"| {name} | {r.get('agent_device')} | {r.get('dtype')} | {r.get('load_s')} | {r.get('peak_rss_mib')} "
                        f"| {worst} | {r.get('peak_device_used_over_idle_mib')} | {ms('guardrail')} | {ms('briefing')} "
                        f"| {ms('worst')} | {acc.get('max_prob_diff_vs_fp32_cpu')} | {acc.get('changed_answers')} "
                        f"| {r.get('cuda_graphs')} |")
    return "\n".join(rows)


def head_is_worse(base: dict, head: dict) -> tuple[bool, str]:
    problems, notes = [], []
    for device in DEVICES:
        b, h = _run(base, device), _run(head, device)
        ba, ha = b.get("accuracy") or {}, h.get("accuracy") or {}
        if ha.get("changed_answers", 0) > ba.get("changed_answers", 0):
            problems.append(f"{device}: {ha.get('changed_answers')} changed answers vs {ba.get('changed_answers')}")
        bd, hd = ba.get("max_prob_diff_vs_fp32_cpu", 0), ha.get("max_prob_diff_vs_fp32_cpu", 0)
        if hd > max(5e-3, 1.5 * bd):
            problems.append(f"{device}: max prob diff {hd} vs {bd}")
        for key in ("peak_rss_mib", "peak_device_used_over_idle_mib"):
            bv, hv = b.get(key), h.get(key)
            if bv and hv and hv > bv * 1.10 + 64:
                problems.append(f"{device}: {key} {hv} vs {bv}")
            elif bv and hv:
                notes.append(f"{device} {key} {bv} -> {hv}")
        for wname, bw in (b.get("workloads") or {}).items():
            hw = (h.get("workloads") or {}).get(wname) or {}
            if hw.get("median_ms") and bw.get("median_ms"):
                ratio = bw["median_ms"] / hw["median_ms"]
                notes.append(f"{device} {wname.split(':')[0]} {bw['median_ms']} -> {hw['median_ms']} ms ({ratio:.2f}x)")
                if hw["median_ms"] > bw["median_ms"] * 1.25:
                    problems.append(f"{device}: {wname} {bw['median_ms']} -> {hw['median_ms']} ms")
    if problems:
        return True, "; ".join(problems)
    return False, "no accuracy, memory or latency regression; " + "; ".join(notes)
