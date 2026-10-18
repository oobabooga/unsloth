#!/usr/bin/env python3
"""Criteria: do LoRA / QLoRA SFT and 16-bit / 4-bit inference behave the same at head as at base?

Per job, by name: head must not fail a job base passes, per-step loss within 1e-3 + 5% of base
(jobs/compare.py's tolerance), peak memory not above base by > 15% and > 0.25 GB. Step time and
decode tok/s are reported (one sample per state on a shared pool: advisory, not gated). A job that
fails on both states is pre-existing and reported, not a regression. REAP, when the head has it,
must pass its GPU check.

Pairs with probes/train_infer_probe.py.
"""

from __future__ import annotations

TITLE = "Unsloth LoRA / QLoRA training and inference, base versus head (+ REAP on head)"
MODE = "regression"
NEEDS = ["gpu", "nvidia", "multi_gpu", "mig", "xpu", "mlx", "windows_rocm_wddm"]
JOBS = ("sft16", "sft4", "infer16", "infer4")


def _job(state: dict, name: str) -> dict:
    return ((state or {}).get("jobs") or {}).get(name) or {}


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out = []
    for name in ("base", "head"):
        j = _job(obs.get(name), "sft16")
        out.append((f"{name}: LoRA SFT produced metrics", j.get("passed") is not None, str(j.get("error") or j.get("rc"))[:200]))
    return out


def table(obs: dict) -> str:
    rows = ["| job | state | passed | loss first / last | median step ms | peak GB | decode tok/s | graph breaks | failed checks / error |",
            "|---|---|---|---|---|---|---|---|---|"]
    for job in JOBS:
        for state in ("base", "head"):
            j = _job(obs.get(state), job)
            loss = j.get("loss") or []
            ll = f"{loss[0]:.4f} / {loss[-1]:.4f}" if loss and None not in (loss[0], loss[-1]) else "-"
            brk = (j.get("compile") or {}).get("graph_breaks")
            err = ", ".join(j.get("checks_failed") or []) or (j.get("error") or "")[:120]
            rows.append(f"| {job} | {state} | {j.get('passed')} | {ll} | {j.get('median_step_ms')} | {j.get('peak_mem_gb')} | "
                        f"{(j.get('summary') or {}).get('decode_tokens_per_s')} | {brk} | {err} |")
    reap = (obs.get("head") or {}).get("reap")
    if reap is not None:
        models = {k.split('/')[-1]: v.get("ok") for k, v in (reap.get("models") or {}).items()}
        rows.append("")
        rows.append(f"REAP on head: ok={reap.get('ok')} {models} pytest={(reap.get('pytest') or {}).get('tail')} gpu={reap.get('gpu')} {reap.get('versions')}")
        if not reap.get("ok"):
            rows.append(f"REAP error: {str(reap.get('error') or reap.get('import_error') or '')[-600:]}")
    return "\n".join(rows)


def head_is_worse(base: dict, head: dict) -> tuple[bool, str]:
    worse, notes = [], []
    for job in JOBS:
        b, h = _job(base, job), _job(head, job)
        if b.get("passed") and not h.get("passed"):
            worse.append(f"{job}: passes on base, fails on head ({h.get('checks_failed') or h.get('error')})")
            continue
        if not b.get("passed") and not h.get("passed"):
            notes.append(f"{job}: fails on both (pre-existing)")
            continue
        bl, hl = b.get("loss") or [], h.get("loss") or []
        if len(bl) != len(hl):
            worse.append(f"{job}: {len(bl)} base steps vs {len(hl)} head steps")
        for i, (x, y) in enumerate(zip(bl, hl)):
            if x is not None and y is not None and abs(x - y) > 1e-3 + 0.05 * abs(x):
                worse.append(f"{job}: step {i + 1} loss {x:.4f} -> {y:.4f}")
                break
        bm, hm = b.get("peak_mem_gb"), h.get("peak_mem_gb")
        if bm and hm and hm > bm * 1.15 and hm - bm > 0.25:
            worse.append(f"{job}: peak memory {bm:.2f} -> {hm:.2f} GB")
        bt, ht = b.get("median_step_ms"), h.get("median_step_ms")
        if bt and ht:
            notes.append(f"{job}: step time x{ht / bt:.3f}")
        bd, hd = (b.get("summary") or {}).get("decode_tokens_per_s"), (h.get("summary") or {}).get("decode_tokens_per_s")
        if bd and hd:
            notes.append(f"{job}: decode tok/s x{hd / bd:.3f}")
    reap = (head or {}).get("reap")
    if reap is not None and not reap.get("ok"):
        worse.append("REAP GPU check failed on head")
    return bool(worse), "; ".join(worse + notes)
