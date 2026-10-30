#!/usr/bin/env python3
"""Criteria (T7h.6): with triton-windows 3.9 (no block pointers) and torch 2.11, GPT-OSS flex prefill after a generate()
materialises full attention scores at zoo main (flex_decoding fails to compile, Dynamo skips flex_attention for the rest
of the process); zoo head sends one-token decode to the main flex kernel, so prefill stays flat. Pairs with
probes/t7h3_flexmem_probe.py run on base and head.

Defect: a flex prefill step at T >= 2048 peaks above 50% of one layer's dense bf16 scores (32 heads x T x T x 2 bytes),
or any step errors.
"""

from __future__ import annotations

import os

TITLE = "T7h.6: GPT-OSS flex prefill memory after generate, Triton without block pointers (base zoo main vs head guard)"
MODE = "differential"
NEEDS = ["rocm", "gpu", "discrete_gpu", "nvidia", "multi_gpu", "mig", "xpu", "mlx"]


def _norm(p):
    return os.path.normcase(os.path.abspath(p)) if p else ""


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out = []
    for state in ("base", "head"):
        o = obs.get(state) or {}
        for mode, r in (o.get("runs") or {}).items():
            out.append((f"{state} {mode}: run finished", r.get("rc") == 0 and r.get("done") and len(r.get("steps") or []) > 0,
                        f"rc={r.get('rc')} steps={len(r.get('steps') or [])} {r.get('error') or ''} {str(r.get('stderr_tail'))[-200:]}"))
        f, co = o.get("unsloth_zoo_file") or "", o.get("checkout") or ""
        out.append((f"{state}: unsloth_zoo imported from its own checkout", bool(f) and _norm(f).startswith(_norm(co)), f"{f}"))
    return out


def _bad(o: dict) -> list[str]:
    bad = []
    for mode, r in (o.get("runs") or {}).items():
        for s in r.get("steps") or []:
            if s.get("error"):
                bad.append(f"{mode} {s.get('arm')} T={s.get('T')}: {str(s['error'])[:160]}")
            elif s.get("arm") == "flex" and s.get("peak_mib") is not None and s["T"] >= 2048:
                dense = 32 * s["T"] * s["T"] * 2 / 2**20
                if s["peak_mib"] > 0.5 * dense:
                    bad.append(f"{mode} flex T={s['T']} peak {s['peak_mib']} MiB (dense one layer {dense:.0f})")
    return bad


def base_shows_defect(base: dict):
    bad = _bad(base)
    return (bool(bad), "; ".join(bad) or "base flat")


def head_is_fixed(head: dict):
    bad = _bad(head)
    return (not bad, "; ".join(bad) or "head flat")


def table(obs: dict) -> str:
    rows = ["| state | run | arm | T | peak MiB | unique graphs |", "|---|---|---|---|---|---|"]
    for state in ("base", "head"):
        for mode, r in ((obs.get(state) or {}).get("runs") or {}).items():
            for s in r.get("steps") or []:
                rows.append(f"| {state} | {mode} | {s.get('arm')} | {s.get('T')} | {s.get('peak_mib')} | {s.get('unique_graphs')} |")
    return "\n".join(rows)
