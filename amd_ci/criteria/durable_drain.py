#!/usr/bin/env python3
"""Criteria: does the durable chat producer fall behind the model stream on this host's real commits?

The defect (unslothai/unsloth#14135): the producer awaited each SQLite commit before reading on, and once a
commit took longer than one display frame every chunk got its own transaction, so a 125 tok/s stream drained
at 1 / commit latency and the GGUF tool loop stalled behind it. Judged on the cells with NO injected delay,
since the open question is whether real Windows commits trigger it. The injected cell is reported, not judged.

Pairs with probes/durable_drain_probe.py.
"""

from __future__ import annotations

import statistics

TITLE = "Durable chat run drain on the host's real SQLite commits, base versus head"
MODE = "differential"
NEEDS = ["windows", "nvidia"]

# Draining slower than the stream by this factor is falling behind, not noise.
_BEHIND = 1.25


def _ok_runs(state: dict, cell: str) -> list[dict]:
    return [r for r in (state.get("cells") or {}).get(cell, []) if "error" not in r]


def _median(rows: list[dict], key: str) -> float | None:
    vals = [r[key] for r in rows if r.get(key) is not None]
    return statistics.median(vals) if vals else None


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out = []
    for name in ("base", "head"):
        o = obs.get(name) or {}
        cells = o.get("cells") or {}
        errors = [f"{c}: {r.get('error')}" for c, rs in cells.items() for r in rs if "error" in r]
        out.append((f"{name}: every cell ran", bool(cells) and not errors and not o.get("error"),
                    "; ".join(errors)[:300] or o.get("error", "ok")))
        bad = [c for c, rs in cells.items() for r in rs
               if "error" not in r and not (r.get("status") == "completed" and r.get("persisted_in_order"))]
        out.append((f"{name}: every chunk persisted once, in order, run completed", not bad, ", ".join(bad) or "ok"))
        n = sum(r.get("append_calls", 0) for r in _ok_runs(o, "real_paced"))
        out.append((f"{name}: real commits were timed", n > 0, f"{n} commits timed"))
    return out


def _row(state: str, cell: str, rows: list[dict]) -> str:
    if not rows:
        return f"| {state} | {cell} | no data | | | | | |"
    ideal = rows[0].get("ideal_upstream_s") or 0.0
    return (f"| {state} | {cell} | {_median(rows, 'upstream_drained_s'):.2f} s | {ideal:.2f} s | "
            f"{_median(rows, 'append_calls'):.0f} | {_median(rows, 'events_per_call_median'):g} | "
            f"{_median(rows, 'commit_ms_p50'):.2f} / {_median(rows, 'commit_ms_p95'):.2f} / "
            f"{_median(rows, 'commit_ms_max'):.1f} ms | {_median(rows, 'commits_over_one_frame'):.0f} |")


def table(obs: dict) -> str:
    rows = ["| state | cell | stream drained | stream length | commits | events per commit | commit p50 / p95 / max | commits over one frame |",
            "|---|---|---|---|---|---|---|---|"]
    for name in ("base", "head", "merge"):
        o = obs.get(name)
        if not o:
            continue
        for cell in ("real_paced", "real_burst", "real_paced_history", "injected30_paced"):
            rows.append(_row(name, cell, _ok_runs(o, cell)))
    rows.append("")
    rows.append("`real_*` cells use this host's own commits (median of their repeats); `injected30` adds 30 ms "
                "before each real commit and is reported, not judged. One frame = 16.7 ms.")
    return "\n".join(rows)


def _behind(state: dict) -> bool:
    for cell in ("real_paced", "real_paced_history"):
        rows = _ok_runs(state, cell)
        drained, ideal = _median(rows, "upstream_drained_s"), (rows[0].get("ideal_upstream_s") if rows else None)
        if drained is not None and ideal and drained > _BEHIND * ideal:
            return True
    return False


def base_shows_defect(base: dict) -> bool:
    return _behind(base)


def head_is_fixed(head: dict) -> bool:
    rows = _ok_runs(head, "injected30_paced")
    injected_ok = bool(rows) and all(
        r["upstream_drained_s"] <= _BEHIND * r["ideal_upstream_s"] for r in rows
    )
    return not _behind(head) and injected_ok
