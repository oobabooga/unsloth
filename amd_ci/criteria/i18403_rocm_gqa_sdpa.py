#!/usr/bin/env python3
"""Criteria (unsloth #18453 / issue #18403), REGRESSION mode: gfx1151 does not show the gfx1200 fault, so this
asks whether the head changes anything where the kernels work, and whether the forced-refusal fallback works.
head_is_worse: any head problem below (route, parity, forced refusal, training) OR head default route backends differ
from base OR head serial / plain training losses differ from base (rel > 1e-3) OR a PR test failed / none ran OR the
self-check child did not pass both backends.
Original differential notes follow.
: ROCm GQA SDPA routing, gfx1151 stand-in for gfx1200 (not in the pool).

Differential mode. "Failed backend" = a FLASH / EFFICIENT kernel-matrix cell (fp16 or bf16, any Unsloth layout,
serial or async) that crashed, raised, went non-finite, or missed max|got - ref| <= 0.02 x max|ref| + 1e-3 against
an fp32 MATH reference. no_kernel cells are not failures.
  base_shows_defect: a base training cell crashed / timed out / logged a non-finite loss, OR the base route cell
    crashed, OR a base route call landed on a failed backend.
  head_is_fixed: every head default training cell (serial, plain) finished rc 0 with finite losses; the head route
    cell finished; no head route call landed on a failed backend; default-route logits vs MATH-only within
    max|diff| <= 1e-2 x max|ref| (finite), loss within 1e-2 relative; the head exposes rocm_sdpa_verdict.
  Reported, not judged: head check_off cell (UNSLOTH_ROCM_SDPA_CHECK=0, expected to track the base).
Gates: both states ran on a gfx1151 HIP device, imported their own checkout, the MATH matrix cells all pass
(harness control), head has the verdict API and base does not.
"""
from __future__ import annotations

import math

TITLE = "unsloth #18453 (issue #18403): ROCm AOTriton SDPA self-check, no regression on gfx1151, base vs head"
MODE = "regression"
# discrete_gpu: gfx1200 (RX 9060 XT) is a discrete RDNA4 card; this pool is one gfx1151 APU.
NEEDS = ["rocm", "gpu", "discrete_gpu", "windows", "nvidia"]
_CTX: dict = {}
FUSED = ("FLASH_ATTENTION", "EFFICIENT_ATTENTION")
_CELL_TO_BACKEND = {"FLASH": "FLASH_ATTENTION", "EFFICIENT": "EFFICIENT_ATTENTION", "MATH": "MATH"}


def _matrix(obs_state):
    return ((obs_state or {}).get("matrix") or {}).get("cells") or {}


def failed_backends(cells):
    """{(backend, dtype): [cells]}. A crash / raise / non-finite result condemns the backend for every dtype;
    an accuracy miss only for its own dtype (bf16 rounding must not condemn an fp16 route)."""
    bad = {}
    for name, c in cells.items():
        parts = name.split("|")
        b, dt = _CELL_TO_BACKEND.get(parts[0]), parts[2] if len(parts) > 2 else "?"
        st = (c or {}).get("status")
        if st in ("crash", "error", "nonfinite"):
            for d in ("float16", "bfloat16"):
                bad.setdefault((b, d), []).append(f"{name}={st}")
        elif st == "inaccurate":
            bad.setdefault((b, dt), []).append(f"{name}={st}")
    return bad


def _route_backends(o, cell = "route"):
    """{(backend, dtype): calls} over the route cell's logged SDPA calls."""
    out = {}
    for key, n in (((o or {}).get(cell) or {}).get("route_calls") or {}).items():
        parts = key.split("|")
        k = (parts[0], parts[-1].replace("torch.", ""))
        out[k] = out.get(k, 0) + int(n)
    return out


def _train_ok(c):
    if not c or c.get("stage") != "done" or c.get("rc") != 0:
        return False
    ls = c.get("losses") or {}
    return bool(ls) and all(math.isfinite(float(v)) for v in ls.values())


def _own(o, f):
    f = str(f or "").replace("\\", "/")
    ck = str((o or {}).get("checkout") or "").replace("\\", "/")
    return bool(f) and bool(ck) and f.startswith(ck)


def gates(obs):
    b, h = obs.get("base") or {}, obs.get("head") or {}
    cells = _matrix(h) or _matrix(b)
    _CTX["failed"] = failed_backends(cells)
    out = []
    for name, o in (("base", b), ("head", h)):
        r = o.get("route") or {}
        v = r.get("versions") or {}
        out.append((f"{name}: route cell ran on a gfx1151 HIP device", bool(v.get("hip")) and "gfx1151" in str(v.get("arch")),
                    f"stage={r.get('stage')} hip={v.get('hip')} arch={v.get('arch')} torch={v.get('torch')} "
                    f"err={str(r.get('error') or '')[:200]}"))
        out.append((f"{name}: attention_dispatch imported from its own checkout", _own(o, r.get("ad_file")),
                    str(r.get("ad_file"))))
    maths = {n: c.get("status") for n, c in cells.items() if n.startswith("MATH|")}
    out.append(("kernel matrix measured; every MATH control cell ok", bool(maths) and set(maths.values()) == {"ok"},
                f"{maths}"))
    out.append(("head has rocm_sdpa_verdict, base does not",
                (h.get("route") or {}).get("has_verdict_api") is True and (b.get("route") or {}).get("has_verdict_api") is False,
                f"head={(h.get('route') or {}).get('has_verdict_api')} base={(b.get('route') or {}).get('has_verdict_api')}"))
    return out


def base_shows_defect(base):
    why = []
    for n, c in (base.get("train") or {}).items():
        if not _train_ok(c):
            why.append(f"train {n}: stage={c.get('stage')} rc={c.get('rc')} losses={c.get('losses')} "
                       f"err={str(c.get('error') or '')[:160]}")
    r = base.get("route") or {}
    if r.get("stage") != "done":
        why.append(f"route cell: stage={r.get('stage')} rc={r.get('rc')} err={str(r.get('error') or '')[:160]}")
    failed = _CTX.get("failed")
    if failed is None:
        failed = failed_backends(_matrix(base))
    for b, n in _route_backends(base).items():
        if b in failed:
            why.append(f"route sent {n} calls to {b}, which fails the kernel matrix ({failed[b][:3]})")
    return (bool(why), "; ".join(why) or "base trained, routed only to backends that pass the matrix")


def _problems(head):
    bad = []
    for n in ("serial", "plain"):
        c = (head.get("train") or {}).get(n)
        if not _train_ok(c):
            c = c or {}
            bad.append(f"train {n}: stage={c.get('stage')} rc={c.get('rc')} losses={c.get('losses')} "
                       f"err={str(c.get('error') or '')[:200]}")
    r = head.get("route") or {}
    if r.get("stage") != "done":
        bad.append(f"route cell: stage={r.get('stage')} err={str(r.get('error') or '')[:200]}")
    failed = _CTX.get("failed")
    if failed is None:
        failed = failed_backends(_matrix(head))
    rb = _route_backends(head)
    if not rb:
        bad.append("route logged no SDPA call (path not engaged)")
    for b, n in rb.items():
        if b in failed:
            bad.append(f"route sent {n} calls to {b}, which fails the kernel matrix")
    p = r.get("parity") or {}
    if "rel" in p:
        if not p.get("finite") or p["rel"] > 1e-2:
            bad.append(f"logit parity vs MATH: {p}")
    else:
        bad.append(f"no logit parity: {p}")
    ld, lm = p.get("loss_default"), p.get("loss_math")
    if ld is None or lm is None or not math.isfinite(ld) or abs(ld - lm) > 1e-2 * max(abs(lm), 1e-6):
        bad.append(f"loss parity vs MATH: default={ld} math={lm}")
    if r.get("has_verdict_api") is not True:
        bad.append("head lacks rocm_sdpa_verdict")
    # Forced refusal (gfx1200 stand-in): every call leaves the fused kernels, numerics and training hold.
    rr = head.get("route_refused") or {}
    v = rr.get("verdict") or {}
    if rr.get("stage") != "done" or not rr.get("forced_refusal"):
        bad.append(f"refused route cell: stage={rr.get('stage')} forced={rr.get('forced_refusal')}")
    elif v.get("flash") is not False or v.get("efficient") is not False:
        bad.append(f"refused route: verdict did not refuse: {v}")
    rrb = _route_backends(head, "route_refused")
    if not rrb or any(b != "MATH" for b, _ in rrb):
        bad.append(f"refused route backends {rrb}: expected MATH only")
    p2 = rr.get("parity") or {}
    if not p2.get("finite") or p2.get("rel", 1.0) > 1e-2:
        bad.append(f"refused route logit parity vs MATH: {p2}")
    if not _train_ok((head.get("train") or {}).get("refused")):
        bad.append(f"train refused: {(head.get('train') or {}).get('refused', {}).get('stage')}")
    return bad


def head_is_fixed(head):
    probs = _problems(head)
    return (not probs, "; ".join(probs))


def table(obs):
    b, h = obs.get("base") or {}, obs.get("head") or {}
    cells = _matrix(h) or _matrix(b)
    L = ["| kernel cell (backend, layout, dtype, mode) | status | max rel err | note |", "|---|---|---|---|"]
    for n, c in cells.items():
        mr = c.get("max_rel")
        note = str(c.get("error") or "")[:120] or ("IMA" if c.get("illegal_memory_access") else "")
        if c.get("status") == "crash":
            note = f"rc={c.get('rc')} invalid_value={c.get('hip_invalid_value')} IMA={c.get('illegal_memory_access')}"
        L.append(f"| {n.replace('|', ' / ')} | {c.get('status')} | {'-' if mr is None else f'{mr:.2e}'} | {note} |")
    L += ["", f"failed backends: {_CTX.get('failed') or failed_backends(cells)}", "",
          "| state | route backends | verdict | parity rel | loss default / math |", "|---|---|---|---|---|"]
    for name, o in (("base", b), ("head", h)):
        r = o.get("route") or {}
        p = r.get("parity") or {}
        L.append(f"| {name} | {_route_backends(o)} | {r.get('verdict', '-')} | {p.get('rel', '-')} | "
                 f"{p.get('loss_default')} / {p.get('loss_math')} |")
    rr = h.get("route_refused") or {}
    p = rr.get("parity") or {}
    L.append(f"| head forced refusal | {_route_backends(h, 'route_refused')} | {rr.get('verdict', '-')} | "
             f"{p.get('rel', '-')} | {p.get('loss_default')} / {p.get('loss_math')} |")
    L.append(f"\nself-check seconds: head route {(h.get('route') or {}).get('verdict_seconds')}, "
             f"forced {rr.get('verdict_seconds')}")
    L += ["", "| state | cell | stage | rc | losses | invalid value / IMA | peak GB |", "|---|---|---|---|---|---|---|"]
    for name, o in (("base", b), ("head", h)):
        for n, c in (o.get("train") or {}).items():
            ls = c.get("losses") or {}
            L.append(f"| {name} | {n} | {c.get('stage')} {str(c.get('error') or '')[:100]} | {c.get('rc')} | "
                     f"{', '.join(f'{float(v):.4f}' for _, v in sorted(ls.items(), key = lambda kv: int(kv[0])))} | "
                     f"{c.get('hip_invalid_value')} / {c.get('illegal_memory_access')} | "
                     f"{(c.get('max_memory_allocated') or 0) / 1e9:.2f} |")
    L.append(f"\nbnb 4-bit: {(h.get('bnb') or b.get('bnb') or {}).get('ok')}; pip: {h.get('pip')}")
    L.append("Not reachable here: gfx1200 / gfx1201 (RDNA4 discrete). gfx1151 only stands in for the routing logic.")
    probs = _problems(h)
    L.append("head problems: " + ("none" if not probs else "; ".join(probs)))
    return "\n".join(L)


def _losses(o, cell):
    c = ((o or {}).get("train") or {}).get(cell) or {}
    return {int(k): float(v) for k, v in (c.get("losses") or {}).items()}


def head_is_worse(base, head):
    bad = list(_problems(head))
    rb, rh = _route_backends(base), _route_backends(head)
    if rb != rh:
        bad.append(f"default route backends differ: base {rb} head {rh}")
    for cell in ("serial", "plain"):
        lb, lh = _losses(base, cell), _losses(head, cell)
        if not lb or lb.keys() != lh.keys():
            bad.append(f"train {cell}: steps differ base {sorted(lb)} head {sorted(lh)}")
            continue
        worst = max(abs(lh[k] - lb[k]) / max(abs(lb[k]), 1e-6) for k in lb)
        if worst > 1e-3:
            bad.append(f"train {cell}: losses differ from base (max rel {worst:.2e})")
    t = head.get("pr_tests") or {}
    cases = t.get("cases") or {}
    failed = [k for k, v in cases.items() if v == "failed"]
    if not cases or t.get("rc") not in (0,) or failed:
        bad.append(f"PR tests: rc={t.get('rc')} n={len(cases)} failed={failed[:6]} {t.get('junit_error', '')}")
    cc = head.get("check_child") or {}
    for b in ("flash", "efficient"):
        rep = (cc.get(b) or {}).get("report") or {}
        if not rep.get("ok") or not rep.get("ran"):
            bad.append(f"self-check child {b}: {cc.get(b)}")
    return (bool(bad), "; ".join(bad) or "head routes, trains and computes like base; self-check passes; forced "
                                      "refusal routes every call to MATH with matching logits; PR tests pass")


def _extra_table(obs):
    h, b = obs.get("head") or {}, obs.get("base") or {}
    L = ["", "| self-check child (alone, serialized) | rc | cases ran | ok | wall s | peak alloc MB | peak reserved MB |",
         "|---|---|---|---|---|---|---|"]
    for k in ("flash", "efficient"):
        c = (h.get("check_child") or {}).get(k) or {}
        rep, pk = c.get("report") or {}, c.get("peak") or {}
        L.append(f"| {k} | {c.get('rc')} | {rep.get('ran')} | {rep.get('ok')} | {c.get('seconds')} | "
                 f"{(pk.get('alloc') or 0) / 2**20:.1f} | {(pk.get('reserved') or 0) / 2**20:.1f} |")
    t = h.get("pr_tests") or {}
    cases = t.get("cases") or {}
    n = {k: sum(1 for v in cases.values() if v == k) for k in ("passed", "failed", "skipped")}
    L.append(f"\nPR tests at head (tests/utils/test_rocm_sdpa_check.py): rc={t.get('rc')} {n}; "
             f"failed {[k for k, v in cases.items() if v == 'failed']}; "
             f"skipped {[k.split('::')[-1] for k, v in cases.items() if v == 'skipped']}")
    L.append(f"base has rocm_sdpa_check.py: {(b.get('check_child') or {}).get('present')}; head commit {h.get('commit')} "
             f"tree {h.get('tree')}; base commit {b.get('commit')}; platform {h.get('platform')}")
    if h.get("stack_install"):
        L.append(f"Windows stack install rc: {[s.get('rc') for s in h['stack_install']]}")
    return "\n".join(L)


_table_v1 = table


def table(obs):
    return _table_v1(obs) + "\n" + _extra_table(obs)
