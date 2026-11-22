#!/usr/bin/env python3
"""Criteria for probes/npu_list_scenarios_probe.py (unsloth#18763): wherever the base lists, the head
lists the same models; the table shows each scenario's text around the JSON and base/head results."""

from __future__ import annotations

TITLE = "unsloth#18763: FastFlowLM 1.0.13 model list on a real NPU, reporter-like scenarios"
MODE = "regression"
NEEDS = ["windows", "exclusive_gpu"]


def _states(obs: dict) -> dict:
    return {k: v for k, v in obs.items() if not k.startswith("_")}


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out = []
    for name, o in _states(obs).items():
        sc = o.get("scenarios") or {}
        out.append((f"{name}: probe ran", "fatal" not in o and bool(sc), str(o.get("fatal") or list(sc))[:300]))
        out.append((f"{name}: real NPU present", bool((o.get("host") or {}).get("dev_accel")), str(o.get("host"))))
        pulled = (sc.get("pulled_by_1013") or {}).get("list", {}).get("installed") or 0
        out.append((f"{name}: models really installed", pulled >= 1, f"{pulled} installed per flm list"))
    return out


def table(obs: dict) -> str:
    base, head = obs.get("base") or {}, obs.get("head") or {}
    rows = ["| scenario | list rc | installed | before JSON | after JSON | stderr | base catalog | head catalog |",
            "|---|---|---|---|---|---|---|---|"]
    for name, s in (base.get("scenarios") or {}).items():
        lst = s.get("list") or {}
        h = ((head.get("scenarios") or {}).get(name) or {}).get("catalog")
        rows.append(f"| {name} | {lst.get('rc')} | {lst.get('installed')} | `{(lst.get('before') or '').strip()[:120]!r}` "
                    f"| `{(lst.get('after') or '').strip()[:200]!r}` | `{(lst.get('stderr') or '').strip()[:300]!r}` "
                    f"| {s.get('catalog')} | {h} |")
    rows += ["", f"pulls: `{base.get('pulls')}`"]
    busy = (base.get("scenarios") or {}).get("npu_busy") or {}
    rows += ["", f"serve alive during busy list: {busy.get('serve_alive')}; serve log tail: `{(busy.get('serve_log') or '')[-300:]!r}`"]
    return "\n".join(rows)


def head_is_worse(base: dict, head: dict) -> tuple[bool, str]:
    bad, notes = [], []
    for name, b in (base.get("scenarios") or {}).items():
        bc = b.get("catalog") or {}
        hc = ((head.get("scenarios") or {}).get(name) or {}).get("catalog") or {}
        if bc.get("ok") and not hc.get("ok"):
            bad.append(f"{name}: head fails ({hc.get('error')})")
        elif bc.get("ok") and hc.get("n") != bc.get("n"):
            bad.append(f"{name}: head lists {hc.get('n')}, base {bc.get('n')}")
        elif not bc.get("ok"):
            notes.append(f"{name}: base fails ({str(bc.get('error'))[:120]}), head {'lists ' + str(hc.get('n')) if hc.get('ok') else 'fails'}")
    if bad:
        return True, "; ".join(bad)
    return False, "head lists wherever base lists" + ("; " + "; ".join(notes) if notes else "")
