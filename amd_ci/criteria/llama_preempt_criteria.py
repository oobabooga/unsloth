#!/usr/bin/env python3
"""Criteria: four chats that do not fit the shared KV pool together.

Base (upstream b11612) must fail them with a context error; the head (#269 with
--preempt-ram) must finish all four by parking, on the GPU, and a request that is
never preempted must produce the same text as the base. Pairs with
llama_preempt_probe.py.
"""

from __future__ import annotations

TITLE = "llama.cpp #269 KV preemption on gfx1151, upstream b11612 versus head"
MODE = "differential"
NEEDS = ["rocm", "gpu"]


def _on_gpu(o: dict) -> bool:
    lines = " ".join((o.get("single") or {}).get("default", {}).get("device_lines") or [])
    return any(k in lines for k in ("ROCm", "Vulkan", "HIP"))


def _texts(o: dict, arm: str) -> list:
    return [r.get("content") for r in ((o.get("single") or {}).get(arm) or {}).get("runs") or []]


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out = []
    for name in ("base", "head"):
        o = obs.get(name) or {}
        built = bool(o.get("backend")) and not any(o.get(k) for k in ("build_error", "configure_error",
                                                                     "clone_error", "error"))
        out.append((f"{name} built llama-server ({o.get('backend')}, {o.get('llama_sha', '')[:9]})", built,
                    o.get("error") or (o.get("build_tail") or "")[-200:].replace("\n", " ")))
        out.append((f"{name} model is on the GPU", _on_gpu(o),
                    "; ".join((o.get("single") or {}).get("default", {}).get("device_lines") or [])[:300]))
        over = o.get("over") or {}
        out.append((f"{name} oversubscribed run finished", "results" in over, over.get("error", "")))
    b, h = obs.get("base") or {}, obs.get("head") or {}
    bt, ht = _texts(b, "default"), _texts(h, "default")
    out.append(("head without --preempt-ram writes what the base writes", bool(bt) and bt == ht,
                f"{len(bt)} base runs, {len(ht)} head runs, equal={bt == ht}"))
    out.append(("head with --preempt-ram writes the same when nothing is preempted",
                bool(ht) and _texts(h, "preempt_ram") == ht, ""))
    return out


def _tps(o: dict, arm: str) -> str:
    runs = ((o.get("single") or {}).get(arm) or {}).get("runs") or []
    p = sorted((r.get("timings") or {}).get("predicted_per_second") or 0 for r in runs)
    q = sorted((r.get("timings") or {}).get("prompt_per_second") or 0 for r in runs)
    return f"{p[len(p) // 2]:.1f} / {q[len(q) // 2]:.1f}" if p else "-"


def table(obs: dict) -> str:
    rows = ["| state | backend | over: completed | ctx errors | parks | makespan s | gen / prompt tok/s (default) | gen / prompt tok/s (--preempt-ram) |",
            "|---|---|---|---|---|---|---|---|"]
    for name in ("base", "head", "merge"):
        o = obs.get(name)
        if not o:
            continue
        ov = o.get("over") or {}
        parks = sum(((r.get("preempt") or {}).get("parks") or 0) for r in ov.get("results") or [])
        rows.append(f"| {name} | {o.get('backend')} | {ov.get('completed')}/4 | {ov.get('ctx_exceeded')} "
                    f"| {parks} | {ov.get('makespan')} | {_tps(o, 'default')} | {_tps(o, 'preempt_ram')} |")
    return "\n".join(rows)


def base_shows_defect(base: dict) -> tuple[bool, str]:
    ov = base.get("over") or {}
    bad = (ov.get("completed") or 0) < 4 or (ov.get("ctx_exceeded") or 0) > 0
    return bad, f"completed {ov.get('completed')}/4, context errors {ov.get('ctx_exceeded')}"


def head_is_fixed(head: dict) -> tuple[bool, str]:
    ov = head.get("over") or {}
    parks = sum(((r.get("preempt") or {}).get("parks") or 0) for r in ov.get("results") or [])
    ok = ov.get("completed") == 4 and not ov.get("ctx_exceeded") and parks > 0 and head.get("help_has_preempt_ram")
    return ok, f"completed {ov.get('completed')}/4, context errors {ov.get('ctx_exceeded')}, parks {parks}"
