#!/usr/bin/env python3
"""Criteria: does the head's dependency window train on gfx1151 as well as the base's did?

Pairs with probes/sft_stack_probe.py. Regression mode: the head is worse when a model that
trained at the base errors, goes non-finite, or gains torch.compile graph breaks at the head.
Gated on both states having installed, run on the ROCm GPU, and the head having resolved a
different transformers than the base, else the comparison says nothing about the change.
"""

from __future__ import annotations

TITLE = "LoRA SFT on each state's own dependency window, base versus head"
MODE = "regression"
NEEDS = ["gpu", "rocm", "nvidia", "windows", "windows_rocm_wddm", "multi_gpu", "discrete_gpu", "mlx"]


def _ver(o: dict, name: str) -> str:
    return (o.get("installed") or {}).get(name, "?")


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out = []
    for name in ("base", "head"):
        o = obs.get(name) or {}
        runs = o.get("runs") or {}
        out.append((f"{name} installed its window", not o.get("error") and o.get("install_rc") == 0,
                    o.get("error") or f"transformers {_ver(o, 'transformers')}, trl {_ver(o, 'trl')}"))
        ran = [k for k, r in runs.items() if r.get("hip")]
        out.append((f"{name} trained on the ROCm GPU", len(ran) > 0,
                    ", ".join(f"{k} on {runs[k].get('device')} (hip {runs[k].get('hip')})" for k in ran) or "no run reached the GPU"))
    b, h = obs.get("base") or {}, obs.get("head") or {}
    out.append(("head resolved a different transformers than base",
                _ver(b, "transformers") != _ver(h, "transformers"),
                f"base {_ver(b, 'transformers')} vs head {_ver(h, 'transformers')}"))
    return out


def table(obs: dict) -> str:
    rows = ["| state | transformers | trl | model | final loss | finite | graph breaks | graphs | steady ms | error |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for name in ("base", "head"):
        o = obs.get(name) or {}
        for model, r in sorted((o.get("runs") or {}).items()):
            losses = r.get("losses") or []
            rows.append(f"| {name} | {_ver(o, 'transformers')} | {_ver(o, 'trl')} | {model} | "
                        f"{losses[-1] if losses else '-'} | {r.get('finite', '-')} | {r.get('graph_breaks', '-')} | "
                        f"{r.get('unique_graphs', '-')} | {r.get('steady_step_ms', '-')} | "
                        f"{(r.get('error') or '')[:120].replace('|', '/')} |")
    return "\n".join(rows)


def head_is_worse(base: dict, head: dict) -> tuple[bool, str]:
    problems = []
    for model, b in (base.get("runs") or {}).items():
        h = (head.get("runs") or {}).get(model) or {"error": "missing at head"}
        if not b.get("error") and h.get("error"):
            problems.append(f"{model} errors at the head: {h['error'][:200]}")
        elif b.get("finite") and not h.get("finite", False):
            problems.append(f"{model} loss is not finite at the head")
        elif (h.get("graph_breaks") or 0) > (b.get("graph_breaks") or 0):
            problems.append(f"{model} graph breaks {b.get('graph_breaks')} -> {h.get('graph_breaks')}")
    if problems:
        return True, "; ".join(problems)
    return False, "every model that trained at the base trains at the head, finite, with no new graph breaks"
