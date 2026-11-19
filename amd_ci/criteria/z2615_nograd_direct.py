#!/usr/bin/env python3
"""Criteria (unsloth-zoo #2615): checkpointed blocks run directly under grad off. REGRESSION mode, base vs head.
head_is_worse: the PR's test file fails / does not run at head; the head's no-grad forward did not take the direct path
(NO_GRAD_CHECKPOINT_STATS["calls"] did not grow); no-grad logits differ from base (rel > 1e-5 on sum / abs sum / first
values) or are non-finite; the 2 SFT step losses / grad norms differ from base (rel > 1e-4); head cell failed.
Gates: both states imported unsloth_zoo.gradient_checkpointing from their own checkout on a gfx1151 HIP device, head
has NO_GRAD_CHECKPOINT_STATS and base does not, base cell finished.
"""
from __future__ import annotations

import math

TITLE = "unsloth-zoo #2615: grad-off checkpoint calls run directly, base vs head on gfx1151"
MODE = "regression"
NEEDS = ["rocm", "gpu", "nvidia"]
NEW = "tests/test_checkpoint_no_grad_direct.py"


def _own(o):
    f = str((((o or {}).get("cell") or {}).get("files") or {}).get("gc") or "").replace("\\", "/")
    ck = str((o or {}).get("checkout") or "").replace("\\", "/")
    return bool(f) and bool(ck) and f.startswith(ck)


def gates(obs):
    b, h = obs.get("base") or {}, obs.get("head") or {}
    out = []
    for name, o in (("base", b), ("head", h)):
        v = (o.get("cell") or {}).get("versions") or {}
        out.append((f"{name}: cell on a gfx1151 HIP device", bool(v.get("hip")) and "gfx1151" in str(v.get("arch")),
                    f"{v} err={str((o.get('cell') or {}).get('error') or '')[:200]}"))
        out.append((f"{name}: gradient_checkpointing from its own checkout", _own(o),
                    str(((o.get("cell") or {}).get("files") or {}).get("gc"))))
    out.append(("head has NO_GRAD_CHECKPOINT_STATS, base does not",
                (h.get("cell") or {}).get("has_stats") is True and (b.get("cell") or {}).get("has_stats") is False,
                f"head={(h.get('cell') or {}).get('has_stats')} base={(b.get('cell') or {}).get('has_stats')}"))
    bc = b.get("cell") or {}
    out.append(("base cell finished (no-grad forward + 2 SFT steps)", bc.get("stage") == "done",
                f"stage={bc.get('stage')} err={str(bc.get('error') or '')[:300]}"))
    fb = sum(((bc.get("nograd") or {}).get("checkpoint_function_forwards") or {}).values())
    out.append(("base no-grad forward went through a checkpoint autograd Function (path engaged)", fb > 0,
                f"{(bc.get('nograd') or {}).get('checkpoint_function_forwards')}"))
    return out


def _rel(x, y):
    return abs(float(x) - float(y)) / max(abs(float(x)), 1e-6)


def head_is_worse(base, head):
    bad = []
    t = head.get("pytest") or {}
    if not t.get("n_passed") or t.get("n_failed"):
        bad.append(f"{NEW} at head: rc={t.get('rc')} passed={t.get('n_passed')} failed={t.get('failed')} {t.get('errors')}")
    bc, hc = base.get("cell") or {}, head.get("cell") or {}
    if hc.get("stage") != "done":
        bad.append(f"head cell: stage={hc.get('stage')} err={str(hc.get('error') or '')[:300]}")
        return True, "; ".join(bad)
    nb, nh = bc.get("nograd") or {}, hc.get("nograd") or {}
    calls = int((nh.get("stats_after") or {}).get("calls", 0)) - int((nh.get("stats_before") or {}).get("calls", 0))
    if calls <= 0:
        bad.append(f"head no-grad forward did not run checkpointed blocks directly (calls delta {calls})")
    fh = sum((nh.get("checkpoint_function_forwards") or {}).values())
    if fh:
        bad.append(f"head no-grad forward still ran checkpoint Functions: {nh.get('checkpoint_function_forwards')}")
    if not nh.get("finite"):
        bad.append("head no-grad logits not finite")
    for k in ("sum", "abs_sum"):
        if k in nb and _rel(nb[k], nh.get(k, math.inf)) > 1e-5:
            bad.append(f"no-grad logits {k}: base {nb[k]} head {nh.get(k)}")
    for i, (x, y) in enumerate(zip(nb.get("first") or [], nh.get("first") or [])):
        if abs(x - y) > 1e-5 * max(abs(x), 1.0):
            bad.append(f"no-grad logits first[{i}]: base {x} head {y}")
    for key in ("losses", "grad_norms"):
        lb, lh = bc.get(key) or {}, hc.get(key) or {}
        if not lb or set(lb) != set(lh):
            bad.append(f"{key}: base {lb} head {lh}")
            continue
        for s in lb:
            if _rel(lb[s], lh[s]) > 1e-4:
                bad.append(f"{key} step {s}: base {lb[s]} head {lh[s]}")
    return (bool(bad), "; ".join(bad) or f"PR tests pass; head no-grad forward ran {calls} checkpointed blocks "
                                      "directly with logits equal to base; SFT losses and grad norms equal to base")


def table(obs):
    b, h = obs.get("base") or {}, obs.get("head") or {}
    L = ["| state | PR tests (passed / failed / skipped) | no-grad calls delta | checkpoint Function forwards (no-grad) | no-grad logits sum / abs sum | "
         "SFT losses | grad norms | peak GB |", "|---|---|---|---|---|---|---|---|"]
    for name, o in (("base", b), ("head", h)):
        t, c = o.get("pytest") or {}, o.get("cell") or {}
        n = c.get("nograd") or {}
        d = int((n.get("stats_after") or {}).get("calls", 0)) - int((n.get("stats_before") or {}).get("calls", 0)) \
            if c.get("has_stats") else "n/a"
        tests = "absent" if t.get("note") else f"{t.get('n_passed')} / {t.get('n_failed')} / {t.get('n_skipped')}"
        L.append(f"| {name} | {tests} | {d} | {n.get('checkpoint_function_forwards')} | {n.get('sum')} / {n.get('abs_sum')} | {c.get('losses')} | "
                 f"{c.get('grad_norms')} | {c.get('peak_gb')} |")
    for name, o in (("base", b), ("head", h)):
        c = o.get("cell") or {}
        if c.get("stage") != "done":
            L.append(f"\n{name} cell stage={c.get('stage')} err={c.get('error')} tb={str(c.get('traceback'))[-800:]} "
                     f"log={str(c.get('log_tail'))[-800:]}")
    L.append(f"\nversions head {(h.get('cell') or {}).get('versions')}; files {(h.get('cell') or {}).get('files')}; "
             f"commits base {b.get('commit')} head {h.get('commit')}")
    L.append(f"venv: {str((h.get('venv') or {}).get('pip_list') or (h.get('venv') or {}).get('windows_stack'))[:600]}")
    return "\n".join(L)
