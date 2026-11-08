#!/usr/bin/env python3
"""Criteria: does a MOSS-Image-SFT Q8_0 GGUF render through Studio on this Windows host?

Base shows the defect when its render fails; head counts as fixed when its render succeeds. Both states must
have installed Studio and loaded the model on the native engine, or the run says nothing about rendering.
"""

from __future__ import annotations

TITLE = "MOSS-Image-SFT Q8_0 GGUF render through Studio on Windows"
MODE = "differential"
NEEDS = ["gpu", "windows", "nvidia"]


def _states(obs):
    return {n: v for n, v in obs.items() if not n.startswith("_")}


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    st = _states(obs)
    out = [("every state installed Studio", all(v.get("install_rc") == 0 for v in st.values()),
            ", ".join(f"{n}={v.get('install_rc')}" for n, v in st.items()))]
    out.append(("every state loaded the model", all((v.get("status") or {}).get("loaded") for v in st.values()),
                ", ".join(f"{n}={(v.get('status') or {}).get('engine')}" for n, v in st.items())))
    out.append(("every state attempted a render", all(v.get("gen_http") is not None for v in st.values()),
                ", ".join(f"{n}={v.get('gen_http')}" for n, v in st.items())))
    return out


def base_shows_defect(state: dict) -> bool:
    return state.get("gen_http") != 200


def head_is_fixed(state: dict) -> bool:
    return state.get("gen_http") == 200 and bool(state.get("n_images"))


def table(obs: dict) -> str:
    rows = ["| state | rev | engine / device | load s | render | secs | error |", "|---|---|---|---|---|---|---|"]
    for n, v in _states(obs).items():
        s = v.get("status") or {}
        err = (v.get("gen_error") or v.get("load_error") or v.get("probe_error") or "")[:300].replace("|", "/")
        rows.append(f"| {n} | {v.get('rev', '')[:60]} | {s.get('engine')} / {s.get('device')} | {v.get('load_secs')} "
                    f"| {v.get('gen_http')} | {v.get('gen_secs')} | {err} |")
    return "\n".join(rows)
