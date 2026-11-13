#!/usr/bin/env python3
"""Criteria for probes/npu_compaction_probe.py (unslothai/unsloth#17616): with auto compact on, a chat larger
than the loaded NPU window must be refused at base and answered at head after a context_truncated fit, while
auto compact off still refuses on both. Gated on a short control chat answering on both states, or an NPU
that never served anything would read as a reproduced defect."""

from __future__ import annotations

TITLE = "NPU chat over the loaded window, auto compact on / off, base versus head"
MODE = "differential"
NEEDS = ["windows", "linux", "multi_gpu"]
ROWS = ("control_short", "over_window_auto_compact", "over_window_auto_compact_stream",
        "over_window_auto_compact_off")


def _states(obs: dict) -> dict:
    return {k: v for k, v in obs.items() if not k.startswith("_")}


def _req(state: dict, name: str) -> dict:
    return (state.get("requests") or {}).get(name) or {}


def _answered(r: dict) -> bool:
    return r.get("status") == 200 and not r.get("error") and bool((r.get("content") or "").strip())


def _compacted(r: dict) -> bool:
    return any(t.get("fits") and int(t.get("dropped_messages") or 0) > 0 for t in r.get("truncations") or [])


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out = []
    for name, o in _states(obs).items():
        load = o.get("load") or {}
        out.append((f"{name}: Studio started", bool(o.get("studio_up")), str(o.get("login"))))
        out.append((f"{name}: NPU model loaded", load.get("status") == 200 and bool(load.get("is_npu")),
                    str(load)[:200]))
        out.append((f"{name}: loaded window is small", 0 < int(load.get("context_length") or 0) <= 4096,
                    str(load.get("context_length"))))
        ctl = _req(o, "control_short")
        out.append((f"{name}: short control chat answered", _answered(ctl), str(ctl)[:200]))
    return out


def table(obs: dict) -> str:
    states = _states(obs)
    rows = ["| request | " + " | ".join(states) + " |", "|---|" + "---|" * len(states)]
    for row in ROWS:
        cells = []
        for o in states.values():
            r = _req(o, row)
            cell = f"HTTP {r.get('status')}"
            if _answered(r):
                cell += f", reply {str(r.get('content'))[:30]!r}"
            if r.get("error"):
                cell += f", error {str(r.get('error'))[:80]!r}"
            for t in r.get("truncations") or []:
                cell += f", truncated {t.get('dropped_messages')} msgs ({t.get('prompt_tokens_before')}->{t.get('prompt_tokens_after')} tok)"
            cells.append(cell.replace("|", "/"))
        rows.append(f"| {row} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def base_shows_defect(base: dict) -> bool:
    return not _answered(_req(base, "over_window_auto_compact")) and not _compacted(
        _req(base, "over_window_auto_compact_stream"))


def head_is_fixed(head: dict) -> bool:
    stream = _req(head, "over_window_auto_compact_stream")
    return (_answered(_req(head, "over_window_auto_compact")) and _answered(stream) and _compacted(stream)
            and not _answered(_req(head, "over_window_auto_compact_off")))
