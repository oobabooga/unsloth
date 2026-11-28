#!/usr/bin/env python3
"""Criteria (unsloth #17492): block compile's strict Dynamo region on a torch without `suppress_errors`, gfx1151 ROCm.

The defect: block compile's first call wrapped torch.compile in torch._dynamo.config.patch(suppress_errors=False);
on a torch build with no such flag (AMD's ROCm torch 2.11) the patch raised, block compile took that as a compile
failure and demoted every decoder layer to the stock forward.

Differential mode.
  base_shows_defect: the base cell finished on a torch without the flag, and printed 'block compile failed' or left
    some decoder layer in a mode other than the compiled one.
  head_is_fixed: the head cell finished, printed no 'block compile failed' line naming suppress_errors, and every
    decoder layer kept a compiled mode; the head's own tests (test_strict_dynamo_flag.py, test_block_compile.py)
    have no failures.
  head_is_worse: head losses differ from base by more than 5e-2 relative (compiled vs stock forward in bf16 may move
    the loss slightly; a large move is a regression).
Gates: each cell imported its state's unsloth; the head defines _strict_dynamo, the base does not.
"""
from __future__ import annotations

TITLE = "unsloth #17492: strict Dynamo regions without suppress_errors, gfx1151 ROCm, base vs head"
MODE = "differential"
NEEDS = ["rocm", "gpu", "nvidia", "windows"]


def _c(o):
    return (o or {}).get("cell") or {}


def _done(c):
    return c.get("stage") == "done" and not c.get("error")


def _demoted(c):
    modes = c.get("layer_modes") or []
    # _block_compile modes: 0 = stock forward (demoted), 1 = full block, 2 = flex outside; None = never ran.
    return not modes or any(m in ("0", "None") for m in modes)


def gates(obs):
    out = []
    for name in ("base", "head"):
        o = obs.get(name) or {}
        c = _c(o)
        f = str(c.get("unsloth_file") or "")
        out.append((f"{name} cell imported the state's unsloth", bool(f) and f.startswith(str(o.get("checkout"))),
                    f or f"stage={c.get('stage')} {c.get('error') or ''}"[:300]))
    out.append(("head defines _strict_dynamo", _c(obs.get("head")).get("has_strict_helper") is True,
                str(_c(obs.get("head")).get("has_strict_helper"))))
    out.append(("base has no _strict_dynamo", _c(obs.get("base")).get("has_strict_helper") is False,
                str(_c(obs.get("base")).get("has_strict_helper"))))
    out.append(("base cell finished", _done(_c(obs.get("base"))), str(_c(obs.get("base")).get("error"))[:300]))
    return out


def base_shows_defect(base):
    c = _c(base)
    return (_done(c) and c.get("has_suppress_errors") is False
            and (bool(c.get("failed_lines")) or _demoted(c)))


def head_is_fixed(head):
    c = _c(head)
    p = (head or {}).get("pytest") or {}
    bad = [l for l in c.get("failed_lines") or [] if "suppress_errors" in l]
    return (_done(c) and not bad and not _demoted(c)
            and p.get("rc") in (0, 5) and p.get("n_failed", 0) == 0 and p.get("n_errors", 0) == 0)


def head_is_worse(base, head):
    lb, lh = _c(base).get("losses") or [], _c(head).get("losses") or []
    if not lh:
        return True, "head produced no losses"
    if len(lb) != len(lh):
        return False, "base has no losses to compare"
    rel = max(abs(x - y) / max(abs(x), 1e-6) for x, y in zip(lb, lh))
    return rel > 5e-2, f"max rel loss diff {rel:.3e}"


def table(obs):
    rows = ["| state | torch | has suppress_errors | layer modes | 'block compile failed' lines | losses | pytest |",
            "|---|---|---|---|---|---|---|"]
    for name in ("base", "head", "merge"):
        o = obs.get(name)
        if not o:
            continue
        c, p = _c(o), o.get("pytest") or {}
        modes = c.get("layer_modes")
        fl = "; ".join(c.get("failed_lines") or [])[:300] or "none"
        losses = ", ".join(f"{x:.4f}" for x in c.get("losses") or []) or f"{c.get('stage')} {c.get('error') or ''}"[:200]
        pt = f"{p.get('n_passed', 0)} passed, {p.get('n_failed', 0)} failed, {p.get('n_skipped', 0)} skipped" if p else "n/a"
        rows.append(f"| {name} | {c.get('torch')} hip {c.get('hip')} | {c.get('has_suppress_errors')} | {modes} | {fl} "
                    f"| {losses} | {pt} |")
    return "\n".join(rows)
