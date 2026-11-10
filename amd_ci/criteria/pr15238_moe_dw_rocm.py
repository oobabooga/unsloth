#!/usr/bin/env python3
"""Criteria (unsloth PR 15238): per-expert MoE dW tile overflowing gfx1151 LDS.

  base_shows_defect  at least one heuristic-tile case raises OutOfResources at the base
  head_is_fixed      every case PASSes at the head (equal to the legacy kernel bit for bit), and no
                     test of the fix's own file (test_moe_dw_rocm_lds_fallback.py) fails at the head;
                     ids failing in the other dW files are listed in the table (base vs head)

Gates (non-vacuity): both states really ran on a ROCm GPU and imported their own checkout's grouped
GEMM interface; the legacy kernel computed a reference for every case at both states; pytest ran and
collected at both states.
"""

from __future__ import annotations

TITLE = "unsloth PR 15238: per-expert MoE dW LDS overflow falls back to the legacy kernel on gfx1151"
MODE = "differential"
# nvidia: the NVIDIA shrink path is unchanged and is not exercised here; windows: Linux ROCm only.
NEEDS = ["rocm", "gpu", "nvidia", "windows"]


def _run(o):
    return (o or {}).get("run") or {}


def gates(obs):
    out = []
    for name in ("base", "head"):
        o = obs.get(name) or {}
        r = _run(o)
        header = str(r.get("header") or "")
        ck = str(o.get("checkout") or "<none>")
        out.append((f"{name} ran on ROCm", "hip None" not in header and " hip " in f" {header} ",
                    header[:300] or f"rc={r.get('rc')} {str(r.get('stderr_tail'))[-300:]}"))
        out.append((f"{name} imported its own grouped GEMM interface", f"interface {ck}" in header, header[:300]))
        res = r.get("results") or []
        out.append((f"{name} ran both cases", len(res) == 2, str(res)[:300]))
        out.append((f"{name} legacy reference computed", not r.get("legacy_errors"),
                    "; ".join(r.get("legacy_errors") or [])[:300]))
        p = o.get("pytest") or {}
        out.append((f"{name} pytest ran", p.get("rc") in (0, 1),
                    f"rc={p.get('rc')} {str(p.get('stderr_tail') or '')[-200:]}"))
        out.append((f"{name} pytest collected",
                    (p.get("n_passed", 0) + p.get("n_failed", 0) + p.get("n_skipped", 0)) > 0,
                    f"passed {p.get('n_passed', 0)}, failed {p.get('n_failed', 0)}, skipped {p.get('n_skipped', 0)}"))
    return out


def table(obs):
    rows = ["| state | case | status | error |", "|---|---|---|---|"]
    for name in ("base", "head", "merge"):
        for r in _run(obs.get(name)).get("results") or []:
            rows.append(f"| {name} | permute_x={r.get('permute_x')} | {r.get('status')} | {(r.get('error') or '')[:160]} |")
    rows += ["", "| state | pytest passed | failed | skipped | failing tests |", "|---|---|---|---|---|"]
    for name in ("base", "head", "merge"):
        p = (obs.get(name) or {}).get("pytest")
        if not p:
            continue
        bad = ", ".join(f"`{t}`" for t in p.get("failed", []) + p.get("errors", [])) or "none"
        absent = ", ".join(p.get("absent_at_this_state") or [])
        rows.append(f"| {name}{f' (absent: {absent})' if absent else ''} | {p.get('n_passed', 0)} "
                    f"| {p.get('n_failed', 0)} | {p.get('n_skipped', 0)} | {bad} |")
    h = _run(obs.get("head"))
    if h.get("header"):
        rows += ["", "Head: " + h["header"]]
    return "\n".join(rows)


def base_shows_defect(base):
    return any(r.get("status") == "OutOfResources" for r in _run(base).get("results") or [])


def head_is_fixed(head):
    res = _run(head).get("results") or []
    p = head.get("pytest") or {}
    own = [t for t in p.get("failed", []) + p.get("errors", []) if "test_moe_dw_rocm_lds_fallback" in t]
    ran_own = "tests/test_moe_dw_rocm_lds_fallback.py" in (p.get("selected") or [])
    return (len(res) == 2 and all(r.get("status") == "PASS" for r in res)
            and ran_own and not own and p.get("n_passed", 0) > 0)
