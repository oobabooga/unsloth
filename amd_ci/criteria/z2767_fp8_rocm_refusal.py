#!/usr/bin/env python3
"""Criteria (unsloth-zoo #2767): the W8A16 grouped GEMM for block-FP8 experts must never engage on ROCm, so on gfx1151
the dequant path runs unchanged. REGRESSION mode, base vs head.
head_is_worse: tests/test_moe_fp8_w8a16_grouped.py has any failed / errored case at head (or collected nothing it
passed); tests/test_fp8_bank_release_checkpointing.py fails a case at head that passed at base; the head cell did not
finish; any head run (auto, auto_rep, UNSLOTH_MOE_FP8_FUSED=0) has a non-finite out / dX, an out or dX sha256 that
differs from base auto (when base auto and auto_rep agree, i.e. the base is deterministic; otherwise a 1e-3 relative
tolerance on sum / abs sum), FP8_FUSED_STATS that moved, fp8_experts_as_packed not None, or a backend message that
differs from base.
Gates: each state imported unsloth_zoo from its own checkout, the cell ran on a gfx1151 HIP device, the backend was
grouped_mm (so the head's step 1b dispatch was reached), the base cell finished with the dequant-plus-grouped_mm
path engaged, head carries moe_fp8_grouped and base does not, head collected the PR's tests, unsloth pinned alike.
"""
from __future__ import annotations

import math

TITLE = "unsloth-zoo #2767: block-FP8 experts keep the dequant path on gfx1151 (fused W8A16 refused), base vs head"
MODE = "regression"
NEEDS = ["rocm", "gpu", "nvidia"]
NEW = "tests/test_moe_fp8_w8a16_grouped.py"
BANK = "tests/test_fp8_bank_release_checkpointing.py"
UNSLOTH_SHA = "3dd8b14c58965de4a122f7ed5bfcea7a67649797"
BASE_SHA = "ebf37d36b9402b44c55f6d1108464380d3867d69"
HEAD_SHA = "c17fd5df00f4b1c3642f7685136a61cf40dadd80"
RUNS = ("auto", "auto_rep", "fused0")
DEQUANT_MSG = "Unsloth: MoE FP8 is using dequantize-plus-grouped_mm."


def _norm(p):
    return str(p or "").replace("\\", "/")


def _own(o):
    f = _norm(((o.get("cell") or {}).get("files") or {}).get("unsloth_zoo"))
    ck = _norm(o.get("checkout")).rstrip("/")
    return bool(f) and bool(ck) and f.startswith(ck + "/")


def gates(obs):
    b, h = obs.get("base") or {}, obs.get("head") or {}
    out = [("states are the commits this run was built for (base merge base, head PR head)",
            b.get("commit") == BASE_SHA and h.get("commit") == HEAD_SHA, f"base {b.get('commit')} head {h.get('commit')}")]
    for name, o in (("base", b), ("head", h)):
        c = o.get("cell") or {}
        v = c.get("versions") or {}
        out.append((f"{name}: cell on a gfx1151 HIP device", bool(v.get("hip")) and "gfx1151" in str(v.get("arch")),
                    f"{v} err={str(c.get('error') or '')[:200]}"))
        out.append((f"{name}: unsloth_zoo from its own checkout", _own(o),
                    f"{((c.get('files') or {}).get('unsloth_zoo'))} vs {o.get('checkout')}"))
        out.append((f"{name}: MoE backend grouped_mm (step 1b reachable)", c.get("backend") == "grouped_mm",
                    str(c.get("backend"))))
    bc, hc = b.get("cell") or {}, h.get("cell") or {}
    hf = _norm((hc.get("files") or {}).get("moe_fp8_grouped"))
    out.append(("head carries moe_fp8_grouped (from its checkout), base does not",
                hc.get("has_fused_module") is True and hf.startswith(_norm(h.get("checkout")).rstrip("/") + "/")
                and bc.get("has_fused_module") is False,
                f"head={hc.get('has_fused_module')} {hf} base={bc.get('has_fused_module')}"))
    ba = (bc.get("runs") or {}).get("auto") or {}
    out.append(("base cell finished (all runs)", bc.get("stage") == "done",
                f"stage={bc.get('stage')} err={str(bc.get('error') or ba.get('error') or '')[:400]}"))
    out.append(("base auto run took the dequant-plus-grouped_mm path (path engaged)",
                DEQUANT_MSG in (ba.get("backend_messages") or []), str(ba.get("backend_messages"))))
    t = h.get("pytest_new") or {}
    n = int(t.get("n_passed") or 0) + int(t.get("n_failed") or 0) + int(t.get("n_skipped") or 0)
    out.append((f"head collected {NEW} and passed at least one case", n > 0 and int(t.get("n_passed") or 0) > 0,
                f"passed={t.get('n_passed')} failed={t.get('n_failed')} skipped={t.get('n_skipped')} rc={t.get('rc')} "
                f"note={t.get('note')} {str(t.get('stderr_tail') or '')[-200:]}"))
    out.append((f"{NEW} absent at base (by construction)", bool((b.get("pytest_new") or {}).get("note")),
                str((b.get("pytest_new") or {}).get("note"))))
    tb = b.get("pytest_bank") or {}
    out.append((f"base collected {BANK}", int(tb.get("n_passed") or 0) + int(tb.get("n_failed") or 0) > 0,
                f"passed={tb.get('n_passed')} failed={tb.get('n_failed')} rc={tb.get('rc')} note={tb.get('note')}"))
    ub, uh = (b.get("unsloth_main") or {}), (h.get("unsloth_main") or {})
    out.append(("both arms run unsloth main at the same pinned commit",
                ub.get("rc") == 0 and UNSLOTH_SHA in str(ub.get("direct_url")) and UNSLOTH_SHA in str(uh.get("direct_url")),
                f"base {ub} head {uh}"[:400]))
    return out


def _rel(x, y):
    return abs(float(x) - float(y)) / max(abs(float(x)), 1e-6)


def _deterministic(bc):
    r = bc.get("runs") or {}
    a, b = r.get("auto") or {}, r.get("auto_rep") or {}
    return all((a.get(k) or {}).get("sha256") and (a.get(k) or {}).get("sha256") == (b.get(k) or {}).get("sha256")
               for k in ("out", "dx"))


def head_is_worse(base, head):
    bad = []
    t = head.get("pytest_new") or {}
    if t.get("note") or not t.get("n_passed") or t.get("n_failed"):
        bad.append(f"{NEW} at head: rc={t.get('rc')} passed={t.get('n_passed')} failed={t.get('failed')} "
                   f"errors={t.get('errors')} note={t.get('note')}")
    tb, th = base.get("pytest_bank") or {}, head.get("pytest_bank") or {}
    hb_bad = set(th.get("failed") or []) | set(th.get("errors") or [])
    bb_bad = set(tb.get("failed") or []) | set(tb.get("errors") or [])
    new_bad = sorted(hb_bad - bb_bad)
    if new_bad or (th.get("note") and not tb.get("note")):
        bad.append(f"{BANK} at head: new failures {new_bad} note={th.get('note')}")
    if not th.get("n_passed") and tb.get("n_passed"):
        bad.append(f"{BANK} at head passed nothing (base passed {tb.get('n_passed')})")
    bc, hc = base.get("cell") or {}, head.get("cell") or {}
    if hc.get("stage") != "done":
        bad.append(f"head cell: stage={hc.get('stage')} err={str(hc.get('error') or '')[:300]} "
                   f"runs={ {k: (v.get('stage'), str(v.get('error'))[:200]) for k, v in (hc.get('runs') or {}).items()} }")
        return True, "; ".join(bad)
    ref = (bc.get("runs") or {}).get("auto") or {}
    det = _deterministic(bc)
    for name in RUNS:
        r = (hc.get("runs") or {}).get(name) or {}
        if r.get("stage") != "done":
            bad.append(f"head {name}: stage={r.get('stage')} err={r.get('error')}")
            continue
        for k in ("out", "dx"):
            hv, bv = r.get(k) or {}, ref.get(k) or {}
            if not hv.get("finite"):
                bad.append(f"head {name} {k} not finite")
            if det:
                if hv.get("sha256") != bv.get("sha256"):
                    bad.append(f"head {name} {k} sha256 {str(hv.get('sha256'))[:16]} != base {str(bv.get('sha256'))[:16]}")
            else:
                for q in ("sum", "abs_sum"):
                    if _rel(bv.get(q, math.inf), hv.get(q, math.inf)) > 1e-3:
                        bad.append(f"head {name} {k} {q}: base {bv.get(q)} head {hv.get(q)} (base not deterministic)")
        moved = {k: v for k, v in (r.get("fused_stats_delta") or {}).items() if v}
        if r.get("fused_stats_delta") is None or moved:
            bad.append(f"head {name} FP8_FUSED_STATS moved or missing: {r.get('fused_stats_delta')}")
        if r.get("as_packed_is_none") is not True:
            bad.append(f"head {name} fp8_experts_as_packed returned {r.get('as_packed_type')}, not None")
        if r.get("fused_enabled") is not False:
            bad.append(f"head {name} fp8_fused_enabled = {r.get('fused_enabled')}")
        if (r.get("backend_messages") or []) != (ref.get("backend_messages") or []):
            bad.append(f"head {name} backend messages {r.get('backend_messages')} != base {ref.get('backend_messages')}")
    det_note = "exact sha256" if det else "1e-3 relative on sums (base A/A sha differed)"
    return (bool(bad), "; ".join(bad) or f"head refused the fused W8A16 GEMM on gfx1151 (FP8_FUSED_STATS 0, "
                                         f"fp8_experts_as_packed None) and out / dX equal base ({det_note}) in auto, "
                                         "auto_rep and UNSLOTH_MOE_FP8_FUSED=0; PR tests pass")


def table(obs):
    b, h = obs.get("base") or {}, obs.get("head") or {}
    L = [f"| state | {BANK} (p / f / s) | {NEW} (p / f / s) | run | backend message | dequant (triton / vectorized) | "
         "out sha / finite | dX sha / finite | FP8_FUSED_STATS delta | as_packed None |",
         "|---|---|---|---|---|---|---|---|---|---|"]

    def pt(t):
        return "absent" if t.get("note") else f"{t.get('n_passed')} / {t.get('n_failed')} / {t.get('n_skipped')}"
    for name, o in (("base", b), ("head", h)):
        c = o.get("cell") or {}
        for run in RUNS:
            r = (c.get("runs") or {}).get(run) or {}
            dq = r.get("dequant_calls") or {}
            L.append(f"| {name} | {pt(o.get('pytest_bank') or {})} | {pt(o.get('pytest_new') or {})} | {run} | "
                     f"{r.get('backend_messages')} | {dq.get('unsloth_triton')} / {dq.get('vectorized')} | "
                     f"{str((r.get('out') or {}).get('sha256'))[:16]} / {(r.get('out') or {}).get('finite')} | "
                     f"{str((r.get('dx') or {}).get('sha256'))[:16]} / {(r.get('dx') or {}).get('finite')} | "
                     f"{r.get('fused_stats_delta')} | {r.get('as_packed_is_none')} |")
    for name, o in (("base", b), ("head", h)):
        c = o.get("cell") or {}
        if c.get("stage") != "done":
            errs = {k: (v.get("error"), str(v.get("traceback"))[-600:]) for k, v in (c.get("runs") or {}).items()
                    if v.get("stage") != "done"}
            L.append(f"\n{name} cell stage={c.get('stage')} err={c.get('error')} runs={errs} "
                     f"tb={str(c.get('traceback'))[-600:]} log={str(c.get('log_tail'))[-800:]}")
        for key in ("pytest_bank", "pytest_new"):
            t = o.get(key) or {}
            if t.get("failed") or t.get("errors"):
                L.append(f"\n{name} {key} failed={t.get('failed')} errors={t.get('errors')} "
                         f"msgs={str(t.get('fail_msgs'))[:800]}")
    hs = (h.get("pytest_new") or {}).get("skipped") or {}
    if hs:
        L.append(f"\nhead {NEW} skipped: {hs}")
    L.append(f"\nversions head {(h.get('cell') or {}).get('versions')}; files {(h.get('cell') or {}).get('files')}; "
             f"commits base {b.get('commit')} head {h.get('commit')}; base A/A deterministic: "
             f"{_deterministic(b.get('cell') or {})}")
    L.append(f"venv: {str((h.get('venv') or {}).get('pip_list'))[:600]}")
    return "\n".join(L)
