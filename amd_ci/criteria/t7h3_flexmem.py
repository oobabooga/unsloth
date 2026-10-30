#!/usr/bin/env python3
"""Criteria (T7h.3 follow-up): GPT-OSS flex prefill peak memory stays flat across varied prompt lengths in one process
(zoo head, gfx1151). Pairs with probes/t7h3_flexmem_probe.py (head only).

Gates: each of the three runs (default / static / alternate) finished at the head from its own zoo checkout.
REGRESSION (head-only): in any run, a flex step's peak exceeds 4x the smallest flex peak at the same length in that run,
or a flex step at length T >= 2048 peaks above 50% of the dense bf16 scores of one layer (32 heads x T x T x 2 bytes): the flex
route is then materialising attention scores.
"""

from __future__ import annotations

import os

TITLE = "T7h.3: GPT-OSS flex prefill memory across varied lengths (default / static shapes / eager-flex alternation)"
MODE = "regression"
NEEDS = ["rocm", "gpu", "discrete_gpu", "nvidia", "multi_gpu", "mig", "xpu", "mlx"]


def _norm(p):
    return os.path.normcase(os.path.abspath(p)) if p else ""


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    head = obs.get("head") or {}
    out = []
    for mode, r in (head.get("runs") or {}).items():
        out.append((f"head {mode}: run finished", r.get("rc") == 0 and r.get("done") and len(r.get("steps") or []) > 0,
                    f"rc={r.get('rc')} steps={len(r.get('steps') or [])} {r.get('error') or ''} {str(r.get('stderr_tail'))[-200:]}"))
    f, co = head.get("unsloth_zoo_file") or "", head.get("checkout") or ""
    out.append(("head: unsloth_zoo imported from its own checkout", bool(f) and _norm(f).startswith(_norm(co)), f"{f}"))
    return out


def head_is_worse(base: dict, head: dict) -> tuple[bool, str]:
    bad, notes = [], []
    for mode, r in (head.get("runs") or {}).items():
        steps = [s for s in (r.get("steps") or []) if s.get("arm") == "flex" and s.get("peak_mib") is not None]
        notes.append(f"{mode}: " + ", ".join(f"{s['T']}:{s['peak_mib']}" for s in steps))
        best = {}
        for s in steps:
            best[s["T"]] = min(best.get(s["T"], s["peak_mib"]), s["peak_mib"])
        for s in steps:
            dense = 32 * s["T"] * s["T"] * 2 / 2**20
            if s["peak_mib"] > 4 * best[s["T"]] or (s["T"] >= 2048 and s["peak_mib"] > 0.5 * dense):
                bad.append(f"{mode} flex T={s['T']} peak {s['peak_mib']} MiB (best {best[s['T']]}, dense one layer {dense:.0f})")
        for s in r.get("steps") or []:
            if s.get("error"):
                bad.append(f"{mode} {s['arm']} T={s['T']}: {s['error'][:200]}")
    if bad:
        return True, "; ".join(bad) + " | " + " || ".join(notes)
    return False, "flat: " + " || ".join(notes)


def table(obs: dict) -> str:
    head = obs.get("head") or {}
    rows = ["| run | arm | T | peak MiB | unique graphs | new log lines |", "|---|---|---|---|---|---|"]
    for mode, r in (head.get("runs") or {}).items():
        for s in r.get("steps") or []:
            log = " / ".join(x.replace("|", "/").replace("\n", " ")[:160] for x in (s.get("new_log") or [])[:2])
            rows.append(f"| {mode} | {s.get('arm')} | {s.get('T')} | {s.get('peak_mib')} | {s.get('unique_graphs')} | {log} |")
    return "\n".join(rows)
