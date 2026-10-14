#!/usr/bin/env python3
"""Criteria: does the head fix a diffusion defect the base demonstrably has, on gfx1151?

MODE differential. differential.py applies the rule this toolkit exists for: if the base does NOT show the
defect, the verdict is VOID, never a pass. Nothing here can soften that.

The defect is named by one spec string, from $DBENCH_AMD_DEFECT or the probe's --defect (recorded in every
state's observation; the two must agree, and every state must carry the same one):

  broken:TAG               base: TAG is not ok or has a sanity flag;   head: TAG ok, no flag
  slow:TAG:REF:RATIO       base: s/image(TAG) / s/image(REF) > RATIO (or TAG broken);
                           head: ratio <= RATIO, TAG ok, LPIPS(TAG vs its ref) <= DBENCH_AMD_LPIPS_MAX (0.10)
                           e.g. slow:heavy_q21_fbcache:heavy_q21_bf16:0.9 for "fbcache must engage and pay"
  quality:TAG:MAX_LPIPS    base: LPIPS(TAG vs its ref) > MAX or flagged; head: <= MAX and unflagged
  memory:TAG:GIB           base: torch peak alloc of TAG > GIB;         head: <= GIB and TAG ok

A state skipped with --skip-states (merge, by default) is reported as not measured and does not block CONFIRMED;
the table says so.

Pairs with amd/diffusion_probe.py.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import criteria_common as CC  # noqa: E402

TITLE = "diffusion_bench on gfx1151: does the head fix the defect the base shows?"
MODE = "differential"
NEEDS = CC.NEEDS
LPIPS_MAX = float(os.environ.get("DBENCH_AMD_LPIPS_MAX", "0.10"))
KINDS = {"broken": 1, "slow": 3, "quality": 2, "memory": 2}

_CTX: dict = {"defect": None, "error": None}


def parse(spec: str):
    kind, *rest = (spec or "").split(":")
    if kind not in KINDS or len(rest) != KINDS[kind]:
        raise ValueError(f"defect spec {spec!r}: expected one of broken:TAG, slow:TAG:REF:RATIO, "
                         f"quality:TAG:MAX_LPIPS, memory:TAG:GIB")
    if kind == "slow":
        return kind, rest[0], rest[1], float(rest[2])
    if kind in ("quality", "memory"):
        return kind, rest[0], None, float(rest[1])
    return kind, rest[0], None, None


def gates(obs: dict) -> list:
    out = CC.host_gates(obs)
    recorded = {n: o.get("defect") for n, o in CC.states(obs).items() if CC.measured(o)}
    spec = os.environ.get("DBENCH_AMD_DEFECT") or next((v for v in recorded.values() if v), None)
    consistent = len({v for v in recorded.values()}) <= 1 and (not os.environ.get("DBENCH_AMD_DEFECT")
                                                              or all(v in (None, spec) for v in recorded.values()))
    try:
        _CTX["defect"] = parse(spec) if spec else None
        _CTX["error"] = None if spec else "no defect spec (set --defect on the probe or DBENCH_AMD_DEFECT)"
    except ValueError as exc:
        _CTX["defect"], _CTX["error"] = None, str(exc)
    out.append(("defect spec present and parseable", _CTX["defect"] is not None, _CTX["error"] or spec))
    out.append(("every state probed for the same defect", consistent, str(recorded)))
    if _CTX["defect"]:
        kind, tag, ref, _ = _CTX["defect"]
        for name in ("base", "head"):
            sel = ((obs.get(name) or {}).get("spec") or {}).get("selected") or []
            need = [t for t in (tag, ref) if t]
            out.append((f"{name} selected {', '.join(need)}", all(t in sel for t in need), f"selected: {sel}"))
        if kind == "slow":
            rb = CC.cells(obs.get("base") or {}).get(ref)
            out.append(("reference cell ok at base", CC.ok(rb),
                        f"{ref}: {(rb or {}).get('verdict')}; a ratio against a broken reference means nothing"))
    return out


def _ratio(o: dict, tag: str, ref: str):
    c, r = CC.cells(o).get(tag) or {}, CC.cells(o).get(ref) or {}
    if not (c.get("new_s") and r.get("new_s")):
        return None
    return c["new_s"] / r["new_s"]


def base_shows_defect(base: dict):
    kind, tag, ref, x = _CTX["defect"]
    c = CC.cells(base).get(tag)
    if kind == "broken":
        return (not CC.ok(c)) or bool(CC.bad_flags(c)), f"{tag}: {(c or {}).get('verdict')}"
    if kind == "slow":
        r = _ratio(base, tag, ref)
        return (not CC.ok(c)) or (r is not None and r > x), f"ratio {r}"
    if kind == "quality":
        lp = (c or {}).get("lpips")
        return CC.ok(c) and (bool(CC.bad_flags(c)) or (lp is not None and lp > x)), f"LPIPS {lp}"
    if kind == "memory":
        pk = (c or {}).get("peak_alloc_gib")
        return CC.ok(c) and pk is not None and pk > x, f"peak {pk} GiB"
    return False, "unknown kind"


def head_is_fixed(head: dict):
    if head.get("skipped_state"):
        return True, "not measured (skipped state)"
    kind, tag, ref, x = _CTX["defect"]
    c = CC.cells(head).get(tag)
    if not CC.ok(c) or CC.bad_flags(c):
        return False, f"{tag}: {(c or {}).get('verdict')} {sorted(CC.bad_flags(c))}"
    if kind == "broken":
        lp = c.get("lpips")
        return lp is None or lp <= LPIPS_MAX, f"LPIPS {lp}"
    if kind == "slow":
        r = _ratio(head, tag, ref)
        lp = c.get("lpips")
        return r is not None and r <= x and (lp is None or lp <= LPIPS_MAX), f"ratio {r}, LPIPS {lp}"
    if kind == "quality":
        lp = c.get("lpips")
        return lp is not None and lp <= x, f"LPIPS {lp}"
    if kind == "memory":
        pk = c.get("peak_alloc_gib")
        return pk is not None and pk <= x, f"peak {pk} GiB"
    return False, "unknown kind"


def table(obs: dict) -> str:
    lines = [CC.side_by_side(obs), ""]
    if _CTX["defect"]:
        kind, tag, ref, x = _CTX["defect"]
        for name, o in CC.states(obs).items():
            if not CC.measured(o):
                continue
            shown = base_shows_defect(o)
            fixed = head_is_fixed(o)
            lines.append(f"- defect `{kind}:{tag}{':' + ref if ref else ''}{':' + str(x) if x is not None else ''}` "
                         f"at {name}: shows defect = {bool(shown[0])} ({shown[1]}), fixed = {bool(fixed[0])} "
                         f"({fixed[1]})")
    return "\n".join(lines)
