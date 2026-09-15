#!/usr/bin/env python3
"""Criteria: after `import unsloth` and after importing Studio's `main`, does the
AOTriton gate read "1" from an unset start, does an explicit "0" survive, and do
spawned children inherit whatever the parent has?

Defect shape (base): both entry points leave the variable unset.
Fixed shape (head): both read "1" from unset; "0" stays "0"; every spawn shape
(mp spawn, subprocess, subprocess with a copied env) reads the parent's value;
the scrubbed control reads unset at every state, so the reader is not inventing
a value.

No torch is needed, so this runs on the Windows boxes, where spawn (not fork)
is the only process model and inheritance is the question that matters.
Pairs with probes/env_spawn_probe.py.
"""

from __future__ import annotations

TITLE = "AOTriton gate: import-time value and spawn inheritance"
MODE = "differential"
NEEDS = ["windows", "rocm", "gpu", "nvidia", "xpu", "mlx"]

ENTRIES = ("unsloth", "studio_main")
SHAPES = ("mp_spawn", "subprocess", "subprocess_copy")


def _states(obs: dict):
    return [(n, v) for n, v in obs.items() if not n.startswith("_")]


def _cell(state: dict, entry: str, start: str) -> dict:
    return (state.get("cells") or {}).get(f"{entry}/{start}") or {}


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out = []
    for n, v in _states(obs):
        cells = v.get("cells") or {}
        ran = all(f"{e}/{s}" in cells and "after" in cells[f"{e}/{s}"]
                  for e in ENTRIES for s in ("unset", "0", "1"))
        out.append((f"{n}: every import cell reported a value", ran,
                    ", ".join(f"{k}: {c.get('error') or c.get('import_error', 'ok')[:60]}"
                              for k, c in cells.items())))
        sp = v.get("spawn") or {}
        scrub_ok = all((sp.get(s) or {}).get("scrubbed") is None for s in ("unset", "0", "1"))
        out.append((f"{n}: scrubbed control child read unset", scrub_ok,
                    str({s: (sp.get(s) or {}).get("scrubbed") for s in ("unset", "0", "1")})))
        # Reader non-vacuity: a parent that STARTED with "1" must hand "1" to every
        # shape regardless of the state, or the reader cannot see values at all.
        one = sp.get("1") or {}
        reader_ok = all(one.get(sh) == "1" for sh in SHAPES)
        out.append((f"{n}: children can read a value (parent started at \"1\")", reader_ok,
                    str({sh: one.get(sh) for sh in SHAPES})))
    return out


def table(obs: dict) -> str:
    names = [n for n, _ in _states(obs)]
    rows = ["| start | entry | " + " | ".join(names) + " |", "|---|---|" + "---|" * len(names)]
    for start in ("unset", "0", "1"):
        for e in ENTRIES:
            vals = []
            for _, v in _states(obs):
                c = _cell(v, e, start)
                a = c.get("after")
                vals.append("unset" if a is None else f"`{a}`")
            rows.append(f"| {start} | {e} | " + " | ".join(vals) + " |")
    rows.append("")
    rows.append("| parent start | shape | " + " | ".join(names) + " |")
    rows.append("|---|---|" + "---|" * len(names))
    for start in ("unset", "0", "1"):
        for sh in SHAPES + ("scrubbed",):
            vals = []
            for _, v in _states(obs):
                a = ((v.get("spawn") or {}).get(start) or {}).get(sh)
                vals.append("unset" if a is None else f"`{a}`")
            rows.append(f"| {start} | {sh} | " + " | ".join(vals) + " |")
    notes = []
    for n, v in _states(obs):
        for k, c in (v.get("cells") or {}).items():
            if c.get("import_error"):
                notes.append(f"{n} {k}: import stopped at `{c['import_error'][:100]}` (value read in finally)")
                break
    return "\n".join(rows) + ("\n\n" + "\n".join(f"- {x}" for x in notes) if notes else "")


def base_shows_defect(base: dict) -> bool:
    return all(_cell(base, e, "unset").get("after") is None for e in ENTRIES)


def head_is_fixed(head: dict) -> bool:
    for e in ENTRIES:
        if _cell(head, e, "unset").get("after") != "1":
            return False
        if _cell(head, e, "0").get("after") != "0":
            return False
        if _cell(head, e, "1").get("after") != "1":
            return False
    sp = head.get("spawn") or {}
    for sh in SHAPES:
        if (sp.get("unset") or {}).get(sh) != "1":
            return False
        if (sp.get("0") or {}).get(sh) != "0":
            return False
    return True
