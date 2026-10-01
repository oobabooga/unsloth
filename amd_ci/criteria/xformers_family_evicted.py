#!/usr/bin/env python3
"""Criteria: after the Linux torch repair, is a CUDA-built xFormers left beside a ROCm torch?

Base shows the defect when the extension really fails to load here AND step 13 keeps it.
Head is fixed when step 13 asks to uninstall it. Pairs with probes/xformers_family_probe.py.
"""

from __future__ import annotations

TITLE = "CUDA xFormers beside ROCm torch after the Linux repair"
MODE = "differential"
NEEDS = ["gpu"]


def _states(obs: dict):
    return {n: v for n, v in obs.items() if not n.startswith("_")}


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out = []
    for name, v in _states(obs).items():
        out.append((f"{name}: probe ran", not v.get("error") and v.get("child_rc") == 0,
                    v.get("error") or f"rc={v.get('child_rc')}"))
        out.append((f"{name}: torch is a ROCm build", bool(v.get("hip")),
                    f"{v.get('torch')} hip={v.get('hip')!r} gpu={v.get('gpu')!r}"))
        out.append((f"{name}: the probed xFormers is the CUDA build",
                    "+cu" in str(v.get("xformers_built_for")), str(v.get("xformers_built_for"))))
        out.append((f"{name}: its extension really fails to load here",
                    bool(v.get("cpp_load_error")), str(v.get("cpp_load_error"))[:160]))
        out.append((f"{name}: step 13 has xFormers checks", bool(v.get("step13_calls")),
                    "; ".join(v.get("step13_calls") or []) or "none"))
    return out


def base_shows_defect(state: dict) -> bool:
    return "xformers" not in (state.get("uninstall_requested") or [])


def head_is_fixed(state: dict) -> bool:
    return "xformers" in (state.get("uninstall_requested") or [])


def table(obs: dict) -> str:
    rows = ["| state | torch | xformers built for | extension loads | step-13 calls | uninstall requested |",
            "|---|---|---|---|---|---|"]
    for name, v in _states(obs).items():
        rows.append(f"| {name} | {v.get('torch')} | {v.get('xformers_built_for')} | "
                    f"{'no' if v.get('cpp_load_error') else 'yes'} | "
                    f"{'<br>'.join(v.get('step13_calls') or [])} | {v.get('uninstall_requested')} |")
    return "\n".join(rows)
