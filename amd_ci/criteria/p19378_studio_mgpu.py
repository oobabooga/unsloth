#!/usr/bin/env python3
"""Criteria: PR 19378 (Studio multi-GPU training backend) on gfx1151, base vs head (Linux ROCm or Windows 11).

Regression mode. Head is WORSE than base when any of:
  * Studio's worker (single mode, no parallel field) completed with finite losses at the base and not at the head;
  * both completed and a per-step loss differs beyond max(RTOL, 3 x the base A/A spread), or the step sets differ;
  * a test of the PR's new files fails / errors / crashes at the head (absent at the base by design);
  * the head's live-host parallel plan contradicts this platform:
      Windows: capabilities()["multi_process"] must be False with a reason naming Windows / NCCL, plan(ddp) must raise
               ParallelPlanError code "parallel_platform", plan(single) must plan, and layer_split must not be refused
               for the platform (on one GPU it is refused for the GPU count, "parallel_gpu_ids");
      Linux:   capabilities() device "rocm", distributed_backend "rccl", multi_process True, the ddp mode row
               unavailable with "Needs at least 2 GPUs", plan(single) plans.
Gates (non-vacuity): each state's environment built on a gfx1151 HIP device; `unsloth` resolved from that state's
checkout; the base worker run completed with finite losses; the head's live probe ran parallel_plan; the head
collected tests.
"""
from __future__ import annotations

import math
import os

TITLE = "PR 19378 Studio multi-GPU backend: single-GPU Studio worker no-regression + parallel_plan on this host"
MODE = "regression"
NEEDS = ["rocm", "gpu", "nvidia", "multi_gpu", "windows"]
RTOL = 5e-3


def _norm(p):
    return os.path.normcase(os.path.normpath(str(p or "")))


def _losses(s):
    raw = (s or {}).get("losses") or {}
    try:
        return {int(k): float(v) for k, v in raw.items()}
    except Exception:  # noqa: BLE001
        return {}


def _trained(s):
    ls = _losses(s)
    return bool(s) and s.get("status") == "complete" and bool(ls) and all(math.isfinite(v) for v in ls.values())


def _arch(o):
    return str((((o or {}).get("live") or {}).get("torch") or {}).get("arch") or "")


def gates(obs):
    out = []
    for name in ("base", "head"):
        o = obs.get(name) or {}
        env = o.get("env") or {}
        out.append((f"{name} environment built", not env.get("error") and bool((env.get("env") or {}).get("hip")),
                    str(env.get("error") or env.get("env"))[:300]))
        out.append((f"{name} ran on gfx1151 (HIP)", "gfx1151" in _arch(o), _arch(o) or "no device recorded"))
        origin = (o.get("live") or {}).get("unsloth_origin")
        ck = o.get("checkout")
        out.append((f"{name} `unsloth` resolves from this state's checkout",
                    bool(origin) and bool(ck) and _norm(origin).startswith(_norm(ck)), f"{origin} (checkout {ck})"))
    b = (obs.get("base") or {}).get("smoke")
    out.append(("base Studio worker completed with finite losses", _trained(b),
                f"status={(b or {}).get('status')} losses={(b or {}).get('losses')}"[:300]))
    live = (obs.get("head") or {}).get("live") or {}
    out.append(("head live probe ran parallel_plan", bool(live.get("capabilities")) and not live.get("error"),
                str(live.get("error") or ("has_parallel_plan=%s" % live.get("has_parallel_plan")))[:300]))
    p = (obs.get("head") or {}).get("pytest") or {}
    n = (p.get("n_passed") or 0) + (p.get("n_failed") or 0) + (p.get("n_skipped") or 0)
    out.append(("head collected the new tests", n > 0,
                f"selected={p.get('selected')} passed={p.get('n_passed')} failed={p.get('n_failed')} "
                f"skipped={p.get('n_skipped')} crashed={p.get('crashed_files')}"[:300]))
    return out


def _rel(a, b):
    la, lb = _losses(a), _losses(b)
    if not la or set(la) != set(lb):
        return None
    return max(abs(la[k] - lb[k]) / max(1.0, abs(la[k])) for k in la)


def _platform_findings(head):
    live = (head or {}).get("live") or {}
    cap = live.get("capabilities") or {}
    plans = live.get("plans") or {}
    bad = []
    single = plans.get("single") or {}
    if not single.get("ok"):
        bad.append(f"plan(single) refused: {single.get('payload') or single.get('error')}")
    ddp = plans.get("ddp") or {}
    if (head or {}).get("platform") == "win32":
        reason = str(cap.get("multi_process_reason") or "")
        if cap.get("multi_process") is not False or not ("Windows" in reason or "NCCL" in reason):
            bad.append(f"Windows capabilities multi_process={cap.get('multi_process')} reason={reason!r}")
        if ddp.get("ok") or (ddp.get("payload") or {}).get("code") != "parallel_platform":
            bad.append(f"Windows plan(ddp) not refused with parallel_platform: {ddp}")
        ls = plans.get("layer_split") or {}
        if (ls.get("payload") or {}).get("code") == "parallel_platform" or ls.get("error"):
            bad.append(f"Windows layer_split refused for the platform: {ls}")
    else:
        rows = {r.get("mode"): r for r in cap.get("modes") or []}
        ddp_row = rows.get("ddp") or {}
        if cap.get("device") != "rocm" or cap.get("distributed_backend") != "rccl" or cap.get("multi_process") is not True:
            bad.append(f"Linux ROCm capabilities device={cap.get('device')} backend={cap.get('distributed_backend')} "
                       f"multi_process={cap.get('multi_process')} reason={cap.get('multi_process_reason')}")
        if ddp_row.get("available") is not False or "Needs at least 2 GPUs" not in str(ddp_row.get("reason")):
            bad.append(f"Linux ddp mode row {ddp_row}")
        if ddp.get("ok"):
            bad.append(f"Linux plan(ddp) planned on a one-GPU host: {ddp}")
    return bad


def head_is_worse(base, head):
    worse = []
    b, h = (base or {}).get("smoke"), (head or {}).get("smoke")
    aa = _rel(b, (base or {}).get("smoke_aa"))
    tol = max([RTOL] + ([3 * aa] if aa is not None else []))
    if _trained(b) and not _trained(h):
        worse.append(f"Studio worker trained at base, not at head (status {(h or {}).get('status')}, "
                     f"events {(h or {}).get('events')})"[:600])
    elif _trained(b) and _trained(h):
        d = _rel(b, h)
        if d is None:
            worse.append(f"loss steps differ: base {sorted(_losses(b))} head {sorted(_losses(h))}")
        elif d > tol:
            worse.append(f"per-step loss differs by {d:.3g} > tol {tol:.3g}")
    p = (head or {}).get("pytest") or {}
    fails = sorted(set(p.get("failed") or []) | set(p.get("errors") or []))
    if fails:
        worse.append("new tests failing at head: " + ", ".join(f"`{t}`" for t in fails[:12]))
    if p.get("crashed_files"):
        worse.append("test files crashing at head: " + ", ".join(p["crashed_files"]))
    worse += _platform_findings(head)
    detail = (f"base vs head max rel per-step loss diff {_rel(b, h)}, base A/A {aa}, tol {tol:.3g}; head tests "
              f"passed {p.get('n_passed')} failed {p.get('n_failed')} skipped {p.get('n_skipped')}")
    if worse:
        return True, "; ".join(worse) + ". " + detail
    return False, "Studio worker single-GPU losses match and the head's parallel plan fits this platform; " + detail


def table(obs):
    b, h = obs.get("base") or {}, obs.get("head") or {}
    lb, la, lh = _losses(b.get("smoke")), _losses(b.get("smoke_aa")), _losses(h.get("smoke"))
    rows = [f"Platform `{h.get('platform') or b.get('platform')}`; versions (head): {(h.get('env') or {}).get('env')}", "",
            "| step | base | base A/A | head |", "|---|---|---|---|"]
    for k in sorted(set(lb) | set(lh) | set(la)):
        rows.append(f"| {k} | {lb.get(k)} | {la.get(k)} | {lh.get(k)} |")
    rows.append("")
    rows.append(f"worker status: base={(b.get('smoke') or {}).get('status')} base_aa={(b.get('smoke_aa') or {}).get('status')} "
                f"head={(h.get('smoke') or {}).get('status')}")
    live = h.get("live") or {}
    cap = dict(live.get("capabilities") or {})
    modes = cap.pop("modes", None)
    rows += ["", f"head capabilities(): {cap}", "", f"head mode rows: {modes}", "",
             f"head plans: {live.get('plans')}"]
    for name, o in (("base", b), ("head", h)):
        p = o.get("pytest")
        if p is not None:
            rows.append(f"pytest {name}: passed={p.get('n_passed')} failed={p.get('n_failed')} "
                        f"skipped={p.get('n_skipped')} absent={p.get('absent_at_this_state')} crashed={p.get('crashed_files')}")
    return "\n".join(rows)
