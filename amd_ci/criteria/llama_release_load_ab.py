#!/usr/bin/env python3
"""Criteria for probes/llama_release_load_ab_probe.py (unslothai/unsloth#18563).

base = b12347-mix-241a0f3 bundle, head = b12540-mix-c4017f9 bundle (gfx1151, one GPU).
Regression = the head bundle AS RELEASED fails to become healthy / answer, crashes with a ROCm
error, or loads (ready_s median) more than 2x slower / decodes (tg tok/s median) below 0.75x
of base in any config base completed. The rtswap arm (runtime DLLs from the other release) is
reported in the table to separate code from runtime; it does not drive the verdict.
"""

from __future__ import annotations

import statistics

TITLE = "llama.cpp ROCm bundle b12347 vs b12540: load + serve (one gfx1151)"
MODE = "regression"
NEEDS = ["rocm", "gpu", "windows", "discrete_gpu", "multi_gpu", "exclusive_gpu"]


def _states(obs):
    return {n: v for n, v in obs.items() if not n.startswith("_")}


def _ok(r):
    return r.get("ready") and r.get("status") == 200 and r.get("alive_after") and not r.get("rocm_error")


def _med(rows, key):
    xs = [r[key] for r in rows if _ok(r) and isinstance(r.get(key), (int, float))]
    return statistics.median(xs) if xs else None


def _spread(rows, key):
    xs = [r[key] for r in rows if _ok(r) and isinstance(r.get(key), (int, float))]
    return (min(xs), max(xs)) if xs else None


def gates(obs):
    st = _states(obs)
    out = [("every state probed both arms", all(v.get("_probe_rc") == 0 and set((v.get("arms") or {})) == {"asis", "rtswap"}
                                                 for v in st.values()),
            ", ".join(f"{n}: rc={v.get('_probe_rc')} arms={list((v.get('arms') or {}))}" for n, v in st.items()))]
    dev = []
    for n, v in st.items():
        r = ((v.get("arms") or {}).get("asis") or {}).get("runs", {}).get("gpu_none") or [{}]
        dev.append((n, any("ROCm" in b[0] for b in r[0].get("buffers", []))))
    out.append(("asis gpu_none put weights in a ROCm buffer (or crashed trying)",
                all(ok or not _ok(((st[n]["arms"]["asis"]["runs"]["gpu_none"]) or [{}])[0]) for n, ok in dev),
                ", ".join(f"{n}={ok}" for n, ok in dev)))
    return out


def head_is_worse(base, head):
    b = (base.get("arms") or {}).get("asis", {}).get("runs", {})
    h = (head.get("arms") or {}).get("asis", {}).get("runs", {})
    bad = []
    for cfg, brows in b.items():
        hrows = h.get(cfg) or []
        if not any(_ok(r) for r in brows):
            continue
        if not hrows or not all(_ok(r) for r in hrows):
            bad.append(f"{cfg}: head failed ({[(r.get('ready'), r.get('status'), r.get('exit_code'), r.get('rocm_error')) for r in hrows]})")
            continue
        br, hr = _med(brows, "ready_s"), _med(hrows, "ready_s")
        if br and hr and hr > 2 * br:
            bad.append(f"{cfg}: ready {hr}s vs {br}s")
        bt, ht = _med(brows, "tg_tps"), _med(hrows, "tg_tps")
        if bt and ht and ht < 0.75 * bt:
            bad.append(f"{cfg}: tg {ht:.1f} vs {bt:.1f} tok/s")
    return (bool(bad), "; ".join(bad) or "head as released matched base in every config base completed")


def table(obs):
    rows = ["| state (code) | arm | config | ok/n | ready s med [min-max] | load_tensors s | pp tok/s | tg tok/s | buffers | failure |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for n, v in _states(obs).items():
        for arm, a in (v.get("arms") or {}).items():
            for cfg, rs in (a.get("runs") or {}).items():
                ok = sum(1 for r in rs if _ok(r))
                sp = _spread(rs, "ready_s")
                fail = next((f"ready={r.get('ready')} st={r.get('status')} exit={r.get('exit_code')} {r.get('rocm_error') or ''}"
                             for r in rs if not _ok(r)), "")
                bufs = ",".join(sorted({b[0] for r in rs for b in r.get("buffers", [])}))
                lt = _med(rs, "load_tensors_s")
                pp, tg = _med(rs, "pp_tps"), _med(rs, "tg_tps")
                rows.append(f"| {n} ({v.get('code_release')}) | {arm} | {cfg} | {ok}/{len(rs)} | {_med(rs, 'ready_s')} "
                            f"[{sp[0] if sp else '-'}-{sp[1] if sp else '-'}] | {lt} | "
                            f"{'%.1f' % pp if pp else '-'} | {'%.1f' % tg if tg else '-'} | {bufs} | {fail[:160]} |")
    return "\n".join(rows)
