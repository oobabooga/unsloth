#!/usr/bin/env python3
"""Criteria: a cmd-style `> nul` in the Git Bash Terminal leaves no file named nul (#16991).

Base should leave the file (the defect), head should not, and both must have run
the command through a real Git Bash with its output intact.
"""

from __future__ import annotations

TITLE = "Git Bash Terminal `echo hidden > nul` leaves a nul file"
MODE = "differential"
NEEDS = ["windows"]


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out = []
    for name in ("base", "head"):
        o = obs.get(name) or {}
        out.append((f"{name} imported its own tools.py", bool(o.get("tools_file")),
                    o.get("tools_file") or o.get("error", "no reading")[:400]))
        out.append((f"{name} ran a trusted Git Bash", bool(o.get("bash")), str(o.get("bash"))))
        out.append((f"{name} command ran", "shown" in str(o.get("output", "")),
                    repr(str(o.get("output", ""))[:300])))
    return out


def table(obs: dict) -> str:
    rows = ["| state | nul file left | output | workdir entries |", "|---|---|---|---|"]
    for name in ("base", "head", "merge"):
        o = obs.get(name)
        if o:
            rows.append(f"| {name} | {o.get('nul_present')} | {str(o.get('output', ''))[:80]!r} "
                        f"| {o.get('entries')} |")
    return "\n".join(rows)


def base_shows_defect(state: dict) -> tuple[bool, str]:
    return bool(state.get("nul_present")), f"nul_present={state.get('nul_present')}"


def head_is_fixed(state: dict) -> tuple[bool, str]:
    ok = state.get("nul_present") is False and "hidden" not in str(state.get("output", ""))
    return ok, f"nul_present={state.get('nul_present')}, output={str(state.get('output', ''))[:120]!r}"
