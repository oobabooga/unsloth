#!/usr/bin/env python3
"""Criteria for npu_tools_probe.py (PR 17530). Defect on base: a fresh NPU chat lights Search, and an API
chat with tool_choice none decodes far slower than the same chat with no tools. Head: Search off and the
tool_choice none chat at no-tools speed. Gated on a real NPU, the model loaded on it, and telemetry rows."""

from __future__ import annotations

import statistics

TITLE = "Real AMD NPU (Lemonade + FastFlowLM): Search default and tool_choice none decode, base versus head"
MODE = "differential"
NEEDS = ["windows"]
SLOW = 0.5  # none / no-tools tok/s below this = speculation off
FAST = 0.8


def _states(obs: dict) -> dict:
    return {k: v for k, v in obs.items() if not k.startswith("_")}


def _tps(o: dict, variant: str) -> float | None:
    vals = [r["tele"]["tps"] for r in (o.get("api") or {}).get(variant, []) if r.get("tele")]
    return statistics.median(vals) if vals else None


def _ttft(o: dict, variant: str) -> float | None:
    vals = [r["tele"]["ttft"] for r in (o.get("api") or {}).get(variant, []) if r.get("tele")]
    return statistics.median(vals) if vals else None


def _in(o: dict, variant: str) -> int | None:
    vals = [r["tele"]["in"] for r in (o.get("api") or {}).get(variant, []) if r.get("tele")]
    return vals[0] if vals else None


def _ratio(o: dict) -> float | None:
    a, b = _tps(o, "tools + tool_choice none"), _tps(o, "no tools")
    return a / b if a and b else None


def _ui_on(o: dict) -> bool | None:
    label = (o.get("ui") or {}).get("search_label")
    return None if label is None else label == "Disable web search"


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out = []
    for name, o in _states(obs).items():
        out.append((f"{name}: probe error-free", not o.get("error"), str(o.get("error"))[:200]))
        hw = (o.get("npu_status") or {}).get("hardware") or {}
        out.append((f"{name}: NPU hardware detected", bool(hw), str(hw)[:200]))
        load = o.get("load") or {}
        out.append((f"{name}: model loaded on the NPU", bool(load.get("is_npu")), str(load)[:200]))
        for v in ("no tools", "tools + tool_choice none"):
            n = len([r for r in (o.get("api") or {}).get(v, []) if r.get("tele")])
            out.append((f"{name}: Lemonade telemetry for '{v}'", n >= 2, f"{n} rows"))
        want = name != "base"
        out.append((f"{name}: tree carries the backend fix == {want}", o.get("tree_has_backend_fix") is want,
                    str(o.get("tree_has_backend_fix"))))
    return out


def table(obs: dict) -> str:
    rows = ["| state | no tools tok/s | tools tok/s | tools + none tok/s | none / no-tools | no tools in-tok | "
            "tools in-tok | none in-tok | ttft no tools / tools / none s | fresh-profile Search | composer sends |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, o in _states(obs).items():
        r = _ratio(o)
        ui = o.get("ui") or {}
        reqs = ui.get("requests") or []
        sent = ", ".join(f"enable_tools={q.get('enable_tools')} tools={q.get('enabled_tools')}" for q in reqs[:1])
        tt = " / ".join(f"{_ttft(o, v):.2f}" if _ttft(o, v) is not None else "-"
                        for v in ("no tools", "tools", "tools + tool_choice none"))
        rows.append(
            f"| {name} | {_tps(o, 'no tools')} | {_tps(o, 'tools')} | {_tps(o, 'tools + tool_choice none')} | "
            f"{f'{r:.2f}' if r else '-'} | {_in(o, 'no tools')} | {_in(o, 'tools')} | "
            f"{_in(o, 'tools + tool_choice none')} | {tt} | {ui.get('search_label') or ui.get('error', '-')[:80]} | "
            f"{sent or '-'} |")
    return "\n".join(rows)


def base_shows_defect(base: dict) -> tuple[bool, str]:
    r = _ratio(base)
    ui = _ui_on(base)
    api_bad = r is not None and r < SLOW
    why = f"tool_choice none at {r:.2f}x no-tools speed" if r else "no API ratio"
    why += f"; fresh-profile Search {'on' if ui else 'off' if ui is False else 'not read'}"
    return api_bad and ui is not False, why


def head_is_fixed(head: dict) -> tuple[bool, str]:
    r = _ratio(head)
    ui = _ui_on(head)
    api_ok = r is not None and r >= FAST
    why = f"tool_choice none at {r:.2f}x no-tools speed" if r else "no API ratio"
    why += f"; fresh-profile Search {'on' if ui else 'off' if ui is False else 'not read'}"
    return api_ok and ui is not True, why
