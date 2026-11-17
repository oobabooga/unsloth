#!/usr/bin/env python3
"""Criteria: on a real XDNA2 NPU, does the head lose any NPU step the base completed, and do the
head's new NPU surfaces (display-name resolution) answer correctly? Pairs with probes/npu_gate_probe.py.
"""

from __future__ import annotations

TITLE = "AMD NPU end to end, base versus head"
MODE = "regression"
NEEDS: list[str] = []

_STEPS = ("detect", "status_before", "enable", "catalog", "download", "load", "chat", "unload")


def _steps(o: dict) -> dict:
    return ((o or {}).get("enabled") or {}).get("steps") or {}


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out = []
    for name in ("base", "head"):
        o = obs.get(name) or {}
        hw = (_steps(o).get("detect") or {}).get("value") or {}
        out.append((f"{name}: XDNA2 NPU detected and supported", bool(hw.get("supported")),
                    f"present={hw.get('present')} family={hw.get('family')} driver={hw.get('driver')} "
                    f"fw={hw.get('firmware_version')}"))
        off = (((o.get("disabled") or {}).get("steps") or {}).get("detect") or {}).get("value") or {}
        out.append((f"{name}: UNSLOTH_STUDIO_ENABLE_NPU=0 hides it", off.get("supported") is False
                    and bool(off.get("present")), f"present={off.get('present')} supported={off.get('supported')}"))
        chat = _steps(o).get("chat") or {}
        out.append((f"{name}: the NPU answered a chat", bool(chat.get("ok") and chat.get("value")),
                    str(chat.get("value") or chat.get("error") or o.get("enabled", {}).get("error"))[:200]))
    return out


def table(obs: dict) -> str:
    rows = ["| step | base | head |", "|---|---|---|"]
    b, h = _steps(obs.get("base") or {}), _steps(obs.get("head") or {})
    for name in (*_STEPS, "resolve_display_name", "shutdown"):
        cells = []
        for s in (b, h):
            st = s.get(name)
            if st is None:
                cells.append("-")
            elif st.get("ok"):
                v = st.get("value")
                cells.append(f"ok {st.get('s', '')}s" + (f": `{str(v)[:80]}`" if name in ("chat", "resolve_display_name", "load") else ""))
            else:
                cells.append(f"FAIL: {str(st.get('error'))[:100]}")
        rows.append(f"| {name} | {cells[0]} | {cells[1]} |")
    return "\n".join(rows)


def head_is_worse(base: dict, head: dict) -> tuple[bool, str]:
    b, h = _steps(base), _steps(head)
    lost = [s for s in _STEPS if (b.get(s) or {}).get("ok") and not (h.get(s) or {}).get("ok")]
    if lost:
        return True, "steps the base completed and the head did not: " + ", ".join(lost)
    res = h.get("resolve_display_name")
    if res is not None:
        v = res.get("value") or []
        good = res.get("ok") and len(v) == 4 and v[0] and v[0] == v[1] and v[0].startswith("lemonade:") \
            and v[2] is None and v[3] is None
        if not good:
            return True, f"display-name resolution wrong at the head: {res.get('value') or res.get('error')}"
    cat = h.get("catalog") or {}
    if any((m or {}).get("local") for m in cat.get("value") or []):
        return True, "a fresh Studio home lists converted (local) NPU rows"
    return False, "the head completed every NPU step the base did" + (
        "; display names resolve (exact, upper case) and unknown / GGUF names do not" if res else "")
