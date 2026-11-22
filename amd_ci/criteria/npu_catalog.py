#!/usr/bin/env python3
"""Criteria for probes/issues_npu_apu_probe.py --parts npu (unsloth#18763): the real FastFlowLM's
`flm list --json` must still list through Flm.catalog() at head wherever it did at base, and
the table shows whether this NPU host printed anything around the JSON (the reported defect)."""

from __future__ import annotations

TITLE = "unsloth#18763: real FastFlowLM model list through Flm.catalog(), base versus head"
MODE = "regression"
NEEDS = ["windows", "exclusive_gpu"]


def _states(obs: dict) -> dict:
    return {k: v for k, v in obs.items() if not k.startswith("_")}


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out = []
    for name, o in _states(obs).items():
        npu = o.get("npu") or {}
        inst = npu.get("install") or {}
        out.append((f"{name}: pinned FastFlowLM installed", bool(inst.get("binary")),
                    str(inst.get("error") or inst.get("found_at") or inst.get("binary"))[:200]))
        lst = (npu.get("raw") or {}).get("list") or {}
        out.append((f"{name}: real `flm list --json` printed a model list",
                    lst.get("rc") == 0 and bool(lst.get("json_ok")) and (lst.get("n_models") or 0) > 0,
                    f"rc={lst.get('rc')} models={lst.get('n_models')}"))
        out.append((f"{name}: catalog() ran", "catalog" in npu, str(npu.get("fatal") or "")[:200]))
    return out


def table(obs: dict) -> str:
    rows = []
    base = (obs.get("base") or {}).get("npu") or {}
    rows.append(f"NPU host: `{base.get('host')}`")
    rows.append("")
    rows += ["| command | rc | s | models | text after the JSON on stdout | stderr |", "|---|---|---|---|---|---|"]
    for cmd, r in (base.get("raw") or {}).items():
        rows.append(f"| {cmd} | {r.get('rc')} | {r.get('s')} | {r.get('n_models')} "
                    f"| `{(r.get('after') or '').strip()[:200]!r}` | `{(r.get('stderr') or '').strip()[:300]!r}` |")
    rows += ["", "| state | catalog() |", "|---|---|"]
    for name, o in _states(obs).items():
        c = (o.get("npu") or {}).get("catalog") or {}
        rows.append(f"| {name} | {'ok, ' + str(c.get('n')) + ' models' if c.get('ok') else c.get('error')} |")
    return "\n".join(rows)


def head_is_worse(base: dict, head: dict) -> tuple[bool, str]:
    b = (base.get("npu") or {}).get("catalog") or {}
    h = (head.get("npu") or {}).get("catalog") or {}
    want = (((head.get("npu") or {}).get("raw") or {}).get("list") or {}).get("n_models")
    if b.get("ok") and not h.get("ok"):
        return True, f"head catalog fails where base listed: {h.get('error')}"
    if h.get("ok") and h.get("n") != want:
        return True, f"head listed {h.get('n')} models, flm printed {want}"
    detail = f"head lists {h.get('n')} of {want} models"
    if not b.get("ok"):
        detail += f"; base failed: {b.get('error')}"
    return False, detail
