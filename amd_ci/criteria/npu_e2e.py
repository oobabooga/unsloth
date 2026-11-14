#!/usr/bin/env python3
"""Criteria for probes/npu_e2e_probe.py: on the real NPU, does a step that worked at the base fail at the head?

Gates make the run non-vacuous: the NPU was detected, the runtime validated and a model loaded and answered
in both states. PR-specific values (resident context, flm_args, error text) are in the table for the reader.
"""

from __future__ import annotations

TITLE = "Real XDNA2 NPU, base versus head"
MODE = "regression"
NEEDS: list[str] = []

_BASELINE = ("detect", "enable", "catalog", "download", "load_auto", "chat_auto")


def _step(o: dict, name: str) -> dict:
    return (o.get("steps") or {}).get(name) or {}


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out = []
    for name in ("base", "head"):
        o = obs.get(name) or {}
        present = (_step(o, "detect").get("value") or {}).get("present") is True
        out.append((f"{name}: NPU detected", present, str(_step(o, "detect"))[:200] or o.get("error", "")))
        for s in _BASELINE[1:]:
            st = _step(o, s)
            out.append((f"{name}: {s}", st.get("ok") is True, (st.get("error") or str(st.get("value")))[:300]))
    # NPU features stay off unless the NPU is present and FastFlowLM is installed.
    head = obs.get("head") or {}
    for s, supported, allowed in (("gating_real", True, True), ("gating_npu_no_runtime", True, False),
                                  ("gating_no_npu", False, False)):
        st = _step(head, s)
        if not st:
            continue
        v = st.get("value") or {}
        q4nx = v.get("q4nx_export_allowed")
        ok = (st.get("ok") is True and bool(v.get("supported")) is supported
              and (bool(v.get("ready")) is supported if s == "gating_real" else not v.get("ready"))
              and q4nx in (allowed, "absent"))
        out.append((f"head: {s}", ok, str(v or st.get("error"))[:300]))
    return out


def table(obs: dict) -> str:
    rows = ["| state | step | ok | value / error |", "|---|---|---|---|"]
    for name in ("base", "head", "merge"):
        o = obs.get(name)
        if not o:
            continue
        for step, st in (o.get("steps") or {}).items():
            detail = st.get("error") if st.get("ok") is not True else st.get("value")
            text = str(detail).replace("|", "/").replace("\n", " ")[:400]
            rows.append(f"| {name} | {step} | {st.get('ok')} | {text} |")
        rows.append(f"| {name} | lemonade_logs | - | {o.get('lemonade_logs')} |")
    return "\n".join(rows)


def head_is_worse(base: dict, head: dict) -> tuple[bool, str]:
    broke = [s for s, st in (base.get("steps") or {}).items()
             if st.get("ok") is True and _step(head, s).get("ok") is False]
    if broke:
        return True, "worked at the base, fails at the head: " + ", ".join(broke)
    return False, "every step that worked at the base works at the head"
