#!/usr/bin/env python3
"""Criteria: does Chat Stop end the pre-inference archive embedding, and is the first wait bounded (#14519)?

Base defect: after Stop the embedding keeps the CPU busy until every pending chunk is embedded.
Head fixed: Stop ends it within one batch, the first request embeds at most the cap, and the same
chunks are archived overall with recall still finding the early fact.
"""

from __future__ import annotations

TITLE = "Conversation-archive embedding: wait before inference and work after Stop"
MODE = "differential"
NEEDS = ["windows"]


def _r(state: dict) -> dict:
    return (state or {}).get("repro") or {}


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out = []
    for name in ("base", "head"):
        r = _r(obs.get(name))
        ok = bool(r.get("s1_requests")) and bool(r.get("s2"))
        out.append((f"{name} repro ran", ok,
                    "" if ok else str((obs.get(name) or {}).get("stderr_tail", ""))[-300:]))
    b, h = _r(obs.get("base")), _r(obs.get("head"))
    same = b.get("s1_total_chunks") and b.get("s1_total_chunks") == h.get("s1_total_chunks")
    out.append(("both archived the same chunks", bool(same),
                f"base {b.get('s1_total_chunks')} head {h.get('s1_total_chunks')}"))
    out.append(("Stop pressed mid-pass on both", bool(
        (b.get("s2") or {}).get("stop_pressed_after_s") and (h.get("s2") or {}).get("stop_pressed_after_s")),
        ""))
    return out


def table(obs: dict) -> str:
    rows = ["| state | cpus | first request s | chunks in first | requests to converge | busy after Stop s "
            "| recall finds fact |", "|---|---|---|---|---|---|---|"]
    for name in ("base", "head"):
        r = _r(obs.get(name))
        if not r:
            rows.append(f"| {name} | no result | | | | | |")
            continue
        s2 = r.get("s2") or {}
        rows.append(f"| {name} | {r.get('cpu_count')} | {r.get('s1_first_request_s')} | "
                    f"{(r.get('s1_requests') or [{}])[0].get('chunks_embedded')} | "
                    f"{r.get('s1_requests_to_converge')} | {s2.get('busy_after_stop_s')} | "
                    f"{r.get('s1_recall_finds_fact')} |")
    return "\n".join(rows)


def base_shows_defect(base: dict) -> bool:
    r = _r(base)
    return (r.get("s2") or {}).get("busy_after_stop_s", 0) > 10 and r.get("s1_requests_to_converge") == 1


def head_is_fixed(head: dict) -> bool:
    r = _r(head)
    s2 = r.get("s2") or {}
    first = (r.get("s1_requests") or [{}])[0].get("chunks_embedded") or 0
    return (s2.get("busy_after_stop_s", 1e9) < 10 and 0 < first <= 256
            and r.get("s1_recall_finds_fact") is True)
