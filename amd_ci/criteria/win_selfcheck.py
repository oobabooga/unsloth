#!/usr/bin/env python3
"""Criteria (unslothai/unsloth#18313) on Windows ROCm: base never routes packed rows to varlen; head's
self-check finishes without a crash and turns varlen on exactly when the kernel bench matches the reference."""

from __future__ import annotations

TITLE = "Windows gfx1151: ROCm varlen self-check, base versus head"
MODE = "differential"
NEEDS = ["rocm", "gpu", "windows"]
TOL = 0.02


def _states(obs):
    return {k: v for k, v in obs.items() if not k.startswith("_")}


def _bench_ok(o, windowed):
    rows = []
    for case in (o.get("bench") or {}).get("cases", []):
        if (case.get("window") is not None) != windowed:
            continue
        for r in case["results"]:
            if r["cand"] in ("varlen_attn_repeat", "aten_flash_varlen_gqa"):
                rows.append(r)
    return bool(rows) and all(not r.get("error") and max(r.get(k, 1) for k in ("err_out", "err_dq", "err_dk", "err_dv")) < TOL
                              for r in rows)


def gates(obs):
    out = []
    for name, o in _states(obs).items():
        c = o.get("selfcheck") or {}
        out.append((f"{name}: ROCm torch on a GPU", bool(c.get("hip")) and bool(c.get("gpu")), f"{c.get('hip')} {c.get('gpu')}"))
        out.append((f"{name}: self-check process exited cleanly", o.get("selfcheck_rc") == 0 and not c.get("error"),
                    f"rc={o.get('selfcheck_rc')} {c.get('error', '')}"))
    head = _states(obs).get("head") or {}
    out.append(("head: kernel bench ran", bool((head.get("bench") or {}).get("cases")), str(head.get("bench_rc"))))
    return out


def table(obs):
    rows = ["| state | torch_varlen | window_ok | check s | window check s |", "|---|---|---|---|---|"]
    for name, o in _states(obs).items():
        c = o.get("selfcheck") or {}
        rows.append(f"| {name} | {c.get('torch_varlen')} | {c.get('window_ok', '-')} | {c.get('check_s', '-')} | "
                    f"{c.get('window_check_s', '-')} |")
    head = _states(obs).get("head") or {}
    for case in (head.get("bench") or {}).get("cases", []):
        rows += ["", f"bench {case['label']} docs={case['n_docs']}", "| candidate | ms | err out | err dq | err dk | err dv | error |",
                 "|---|---|---|---|---|---|---|"]
        for r in case["results"]:
            f = lambda k: f"{r[k]:.3g}" if isinstance(r.get(k), float) else "-"
            rows.append(f"| {r['cand']} | {f('ms_median')} | {f('err_out')} | {f('err_dq')} | {f('err_dk')} | {f('err_dv')} | "
                        f"{r.get('error', '')[:100]} |")
    return "\n".join(rows)


def base_shows_defect(base):
    return (base.get("selfcheck") or {}).get("torch_varlen") is False


def head_is_fixed(head):
    # Fixed = the self-check agrees with the independent bench (on when the kernel is right, off when not).
    c = head.get("selfcheck") or {}
    return (head.get("selfcheck_rc") == 0 and c.get("torch_varlen") is _bench_ok(head, False)
            and c.get("window_ok") is _bench_ok(head, True))
