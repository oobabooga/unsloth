#!/usr/bin/env python3
"""Criteria: does the Windows GPU monitor read as a standard user with no UAC prompt? (#18005)

Pairs with probes/win_uac_probe.py. Regression mode: head is worse when, as a standard user
(Administrators disabled, Medium integrity), its readings fail where base's succeed, or an
elevation prompt (consent.exe) or a manifest-elevating helper (diskpart.exe) appears while they
run, or they start any process other than PowerShell / its console host. The amd-smi positive
control is reported beside it: if it raises no prompt on this runner either, the watcher is
unproven here and the verdict rests on the spawned-process list alone.
"""

from __future__ import annotations

TITLE = "Windows GPU monitor as a standard user: readings and no UAC prompt"
MODE = "regression"
NEEDS = ["windows", "rocm", "gpu", "integrated_gpu", "discrete_gpu", "multi_gpu", "linux"]
ALLOWED = {"powershell.exe", "conhost.exe"}


def _states(obs):
    return {n: v for n, v in obs.items() if not n.startswith("_")}


def _child(v, mode="readings"):
    return ((v.get(mode) or {}).get("child")) or {}


def gates(obs):
    st = _states(obs)
    out = [("probe ran without error", all(not v.get("error") for v in st.values()),
            "; ".join(f"{n}: {(v.get('error') or 'ok').splitlines()[-1]}" for n, v in st.items()))]
    out.append(("restricted child finished and reported", all(
        _child(v) and not (v.get("readings") or {}).get("timed_out") and not _child(v).get("error")
        for v in st.values()),
        "; ".join(f"{n}: spawn={(v.get('readings') or {}).get('spawn_error')} "
                  f"err={(_child(v).get('error') or '').splitlines()[-1:]}" for n, v in st.items())))
    out.append(("child really was a standard user (not admin, not elevated, Medium IL 0x2000)", all(
        _child(v).get("is_admin") is False and _child(v).get("token_elevated") is False
        and _child(v).get("integrity_rid") == 0x2000 for v in st.values()),
        "; ".join(f"{n}: admin={_child(v).get('is_admin')} elevated={_child(v).get('token_elevated')} "
                  f"il={_child(v).get('integrity_rid')}" for n, v in st.items())))
    return out


def table(obs):
    rows = ["| state | runner admin | child util | child temp | payload util / temp | spawned during readings | prompt processes | amd-smi control (rc, prompt seen) |",
            "|---|---|---|---|---|---|---|---|"]
    for n, v in _states(obs).items():
        c, r, a = _child(v), v.get("readings") or {}, v.get("amd_smi") or {}
        p = c.get("payload") or {}
        rows.append(f"| {n} | {(v.get('runner') or {}).get('is_admin')} | {c.get('util')} | {c.get('temp')} | "
                    f"{p.get('gpu_utilization_pct')} / {p.get('temperature_c')} | {sorted(set(r.get('spawned') or []))} | "
                    f"{r.get('prompt_processes_anywhere')} | {(_child(v, 'amd_smi')).get('amd_smi_rc')}, "
                    f"{a.get('prompt_processes_anywhere')} spawned={sorted(set(a.get('spawned') or []))} |")
    return "\n".join(rows)


def _ok(v):
    c = _child(v)
    return any(x is not None for x in (c.get("util") or [])) and not (v.get("readings") or {}).get("prompt_processes_anywhere")


def head_is_worse(base, head):
    r, c = head.get("readings") or {}, _child(head)
    extra = sorted(set(r.get("spawned") or []) - ALLOWED)
    if r.get("prompt_processes_anywhere"):
        return True, f"elevation prompt process during head readings: {r['prompt_processes_anywhere']}"
    if extra:
        return True, f"head readings started unexpected processes: {extra}"
    if c.get("temp_fn_present") and not any(t is not None for t in (c.get("temp") or [])):
        return True, "head temperature unreadable as a standard user"
    if _ok(base) and not _ok(head):
        return True, "head utilization unreadable as a standard user where base read it"
    return False, (f"standard user: util {c.get('util')}, temp {c.get('temp')}, spawned "
                   f"{sorted(set(r.get('spawned') or []))}, no prompt process")
