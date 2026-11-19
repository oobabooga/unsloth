#!/usr/bin/env python3
"""Criteria for probes/win_mmap_release_probe.py (unslothai/llama.cpp#376).

Defect: with every layer on the GPU, --load-mode mmap leaves the GGUF's pages in llama-server's
working set, so mmap costs at least half the weights more than --load-mode none.
Fixed: the patched llama.dll's mmap working set is within 10% of the weights of none, with the
same greedy text as every other run."""

from __future__ import annotations

import re
import statistics

TITLE = "Windows gfx1151: llama-server working set, --load-mode mmap vs none (llama.cpp#376)"
MODE = "differential"
NEEDS = ["gpu", "windows"]


def _states(obs: dict) -> dict:
    return {k: v for k, v in obs.items() if not k.startswith("_") and isinstance(v, dict)}


def _ws(o: dict, arm: str, mode: str) -> float | None:
    vals = [r["proc"]["ws_mib"] for r in o.get("runs") or []
            if r["arm"] == arm and r["mode"] == mode and r.get("proc")]
    return statistics.median(vals) if vals else None


def _gpu_shared_mib(r: dict) -> int:
    m = [int(x) for ln in r.get("gpu") or [] if "shared usage" in ln
         for x in re.findall(r"=(\d+)MiB", ln)]
    return max(m or [0])


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out = []
    texts = set()
    for name, o in _states(obs).items():
        runs = o.get("runs") or []
        out.append((f"{name}: probe ran", not o.get("error") and len(runs) == 7, str(o.get("error") or len(runs))))
        bad = [f"{r['arm']}/{r['mode']}={r['status']}" for r in runs if r.get("status") != "ready"]
        out.append((f"{name}: every load ready", not bad, ", ".join(bad) or "7/7"))
        model = o.get("model_mib") or 0
        gpu = [_gpu_shared_mib(r) for r in runs]
        out.append((f"{name}: weights on the GPU in every run", bool(gpu) and min(gpu) >= 0.8 * model,
                    f"GPU shared usage {gpu} MiB vs model {model} MiB"))
        sizes = o.get("dll_sizes") or {}
        out.append((f"{name}: three llama.dll builds in place", len(set(sizes.values())) >= 2, str(sizes)))
        texts |= {r.get("text") for r in runs}
    out.append(("greedy output identical across all runs", len(texts) == 1, f"{len(texts)} distinct"))
    return out


def base_shows_defect(base: dict):
    model = base.get("model_mib") or 0
    m, n, orig = _ws(base, "base", "mmap"), _ws(base, "base", "none"), _ws(base, "orig", "mmap")
    if None in (m, n, orig):
        return False, "missing readings"
    ok = m - n >= 0.5 * model and abs(orig - m) <= 0.1 * model
    return ok, f"rebuilt base mmap {m} vs none {n} MiB (+{m - n}); bundle's own llama.dll mmap {orig} MiB; weights {model} MiB"


def head_is_fixed(head: dict):
    model = head.get("model_mib") or 0
    m, n = _ws(head, "head", "mmap"), _ws(head, "head", "none")
    if None in (m, n):
        return False, "missing readings"
    warned = [w for r in head.get("runs") or [] if r["arm"] == "head" for w in r.get("warnings") or []
              if "VirtualUnlock" in w]
    return (m - n <= 0.1 * model and not warned,
            f"head mmap {m} vs none {n} MiB ({m - n:+}); VirtualUnlock warnings: {len(warned)}")


def table(obs: dict) -> str:
    rows = ["| state | # | llama.dll | load mode | status | working set MiB | private WS MiB | GPU shared MiB | text |",
            "|---|---|---|---|---|---|---|---|---|"]
    for name, o in _states(obs).items():
        for i, r in enumerate(o.get("runs") or []):
            p = r.get("proc") or {}
            rows.append(f"| {name} | {i} | {r['arm']} | {r['mode']} | {r['status']} | {p.get('ws_mib')} | "
                        f"{p.get('private_ws_mib')} | {_gpu_shared_mib(r)} | {str(r.get('text'))[:30]!r} |")
    return "\n".join(rows)
