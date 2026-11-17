#!/usr/bin/env python3
"""Criteria: the Windows GPU monitor as a real standard local user, under a HIP load: readings, and no elevation. (#18005)

Pairs with probes/win_uac_user_probe.py. Regression mode. Head is worse when, as that user, an
elevation prompt (consent.exe) or diskpart.exe appears while its readings run, any subprocess it
starts fails for elevation (WinError 740) or is not PowerShell / its console host, its temperature
is unreadable, or its utilization misses a load the independent counter saw (< 50 % while the
counter reads >= 50 %). The diskpart control shows what an elevation attempt looks like on this
runner; it is reported, and a run where it shows nothing is called out, not passed silently.
"""

from __future__ import annotations

TITLE = "Windows GPU monitor as a standard local user under a HIP load: readings and no elevation"
MODE = "regression"
NEEDS = ["windows", "rocm", "gpu", "integrated_gpu", "discrete_gpu", "multi_gpu", "linux"]
ALLOWED = {"powershell.exe", "conhost.exe"}


def _states(obs):
    return {n: v for n, v in obs.items() if not n.startswith("_")}


def _c(v, k = "readings"):
    return ((v.get(k) or {}).get("child")) or {}


def _busiest(raw):
    return max(((raw or {}).get("types") or {"": 0}).values())


def gates(obs):
    st = _states(obs)
    out = [("probe ran without error", all(not v.get("error") for v in st.values()),
            "; ".join(f"{n}: {(v.get('error') or 'ok').splitlines()[-1]}" for n, v in st.items()))]
    out.append(("throwaway user created and deleted (account and profile)", all(
        v.get("user") and (v.get("cleanup") or {}).get("user_deleted") and not (v.get("cleanup") or {}).get("still_exists")
        for v in st.values()), "; ".join(f"{n}: {v.get('user')} {v.get('cleanup')}" for n, v in st.items())))
    out.append(("readings child ran as that user to the end", all(
        _c(v).get("user") == v.get("user") and not (v.get("readings") or {}).get("timed_out") and not _c(v).get("error")
        for v in st.values()), "; ".join(f"{n}: user={_c(v).get('user')} exit={(v.get('readings') or {}).get('exit_code')} "
                                         f"err={(_c(v).get('error') or '').splitlines()[-1:]}" for n, v in st.items())))
    out.append(("standard user token: not admin, not elevated, Medium IL", all(
        _c(v).get("is_admin") is False and _c(v).get("token_elevated") is False and _c(v).get("integrity_rid") == 0x2000
        for v in st.values()), "; ".join(f"{n}: admin={_c(v).get('is_admin')} elevated={_c(v).get('token_elevated')} "
                                         f"type={_c(v).get('elevation_type')} il={_c(v).get('integrity_rid')}" for n, v in st.items())))
    out.append(("HIP load seen by the independent counter before and after the readings", all(
        v.get("load_ready") and v.get("load_alive_after") and _busiest(v.get("raw_before")) >= 50 and _busiest(v.get("raw_after")) >= 50
        for v in st.values()), "; ".join(f"{n}: before={_busiest(v.get('raw_before'))} after={_busiest(v.get('raw_after'))} "
                                         f"alive={v.get('load_alive_after')}" for n, v in st.items())))
    return out


def table(obs):
    rows = ["| state | user | util (fn) | temp (fn) | monitor payload util / temp | started | elevation errors | prompt procs | diskpart control: CreateProcess / ShellExecute / prompt procs |",
            "|---|---|---|---|---|---|---|---|---|"]
    for n, v in _states(obs).items():
        c, r, ctl = _c(v), v.get("readings") or {}, v.get("control") or {}
        cc = _c(v, "control")
        pay = "; ".join(f"{p.get('gpu_utilization_pct')} / {p.get('temperature_c')}" for p in c.get("payload") or [])
        rows.append(f"| {n} | {v.get('user')} | {c.get('util_parent_luid')} | {c.get('temp_parent_luid')} | {pay} (child LUID {c.get('luid')}) | {sorted(set(r.get('spawned') or []))} | "
                    f"{c.get('spawn_errors')} | {r.get('prompt_processes_anywhere')} | {cc.get('createprocess')} / "
                    f"rc={cc.get('shellexecute_rc')} err={cc.get('shellexecute_lasterror')} / {ctl.get('prompt_processes_anywhere')} |")
    for n, v in _states(obs).items():
        c = _c(v)
        rows.append(f"\n- {n}: raw counters as the user: adapter memory {c.get('raw_adapter_memory')}, engines {c.get('raw_engine_all')}; "
                    f"HIP adapter ids {c.get('hip_adapter_ids')}; powershell starts {c.get('powershell_start_s')} s; "
                    f"narrowed engine call {c.get('narrowed_call')}")
    return "\n".join(rows)


def head_is_worse(base, head):
    c, r = _c(head), head.get("readings") or {}
    if r.get("prompt_processes_anywhere"):
        return True, f"elevation prompt process during head readings: {r['prompt_processes_anywhere']}"
    if any(e.get("winerror") == 740 for e in c.get("spawn_errors") or []):
        return True, f"head started a process needing elevation: {c['spawn_errors']}"
    extra = sorted(set(r.get("spawned") or []) - ALLOWED - {"python.exe"})
    if extra:
        return True, f"head readings started unexpected processes: {extra}"
    # The PR's readings, on the LUID the runner resolved: the per-device LUID join is pre-existing code.
    if c.get("temp_fn_present") and not any(t is not None for t in c.get("temp_parent_luid") or []):
        return True, "head temperature unreadable as a standard user"
    utils = [u for u in c.get("util_parent_luid") or [] if u is not None]
    if not utils or max(utils) < 50:
        return True, f"head utilization {c.get('util_parent_luid')} misses the load the counter saw"
    return False, (f"standard user under load: util {c.get('util_parent_luid')}, temp {c.get('temp_parent_luid')}, "
                   f"started {sorted(set(r.get('spawned') or []))}, no prompt, no elevation error")
