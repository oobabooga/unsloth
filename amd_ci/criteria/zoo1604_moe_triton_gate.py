#!/usr/bin/env python3
"""Criteria (unsloth-zoo PR 1604): Triton grouped GEMM for every MoE behind a static arch / shape gate. On ROCm /
HIP the gate must be OFF, so everything must be unchanged vs base. gfx1151 ROCm, base vs head.

Regression mode. Head is WORSE than base when any of:
  * the run is not on HIP (versions.hip is None in any head cell)
  * a head training cell crashes or has a non-finite loss (compiled: only if a base compiled cell finished)
  * head moe_utils._triton_grouped_mm_max_rows(i, k) != 0 for any i in (0, None), k in (lora, many, few, dw), under
    the cell's env or with UNSLOTH_MOE_GROUPED_TRITON forced to "1" / "auto"
  * head moe_grouped_fp16.GENERIC_CALLS is not all-zero after training in any cell (eager, forced, compiled)
  * a head cell's per-step loss reprs, per-step LoRA-grad digests or final LoRA digest differ from the same base
    cell (eager and forced: the base A/A pair is bit-stable, else beyond its spread); a compiled head cell that
    is bit-identical to no base compiled run (r1..r3) and differs beyond the base compiled A/A loss spread
  * eager head calls torch._grouped_mm a different number of times than eager base (the path changed)
  * a test that passed at the base does not pass at the head, or a new test file fails (not skips) at the head

Gates: each state imported its own checkout; pytest ran and collected; the new test files ran in both states
(copied from head into base); base eager cells finished; only the head has _triton_grouped_mm_max_rows /
GENERIC_CALLS.
"""

from __future__ import annotations

TITLE = "unsloth-zoo PR 1604: Triton grouped GEMM gate for every MoE on gfx1151 ROCm (must be off), base vs head"
MODE = "regression"
NEEDS = ["rocm", "gpu", "nvidia"]

_RAN = (0, 1)
NEW = ("test_moe_grouped_triton_generic", "test_moe_grouped_generic_config")
KINDS = ("lora", "many", "few", "dw")


def _cells(o):
    return (o or {}).get("train") or {}


def _done(c):
    return c.get("stage") == "done" and c.get("ok") is True and not c.get("error")


def _sig(c):
    return (c.get("loss_reprs"), c.get("grad_digests"), c.get("lora_digest"))


def gates(obs):
    out = []
    for name in ("base", "head"):
        o = obs.get(name) or {}
        ck = str(o.get("checkout") or "<none>")
        cells = _cells(o)
        files = {str(c.get("unsloth_zoo_file") or "") for c in cells.values()}
        out.append((f"{name} train cells imported the state's unsloth_zoo",
                    bool(cells) and all(f and f.startswith(ck) for f in files), ", ".join(sorted(files)) or "no cells"))
        p = o.get("pytest") or {}
        out.append((f"{name} pytest ran", p.get("rc") in _RAN,
                    f"rc={p.get('rc')} {str(p.get('stderr_tail') or '')[-200:] if p.get('rc') not in _RAN else ''}"))
        n = p.get("n_passed", 0) + p.get("n_failed", 0) + p.get("n_skipped", 0)
        out.append((f"{name} pytest collected", n > 0,
                    f"passed {p.get('n_passed', 0)}, failed {p.get('n_failed', 0)}, skipped {p.get('n_skipped', 0)}"))
        ids = list(p.get("passed") or []) + list(p.get("failed") or []) + list(p.get("errors") or []) + list((p.get("skipped") or {}).keys())
        out.append((f"{name} ran the new test files", all(any(t in i for i in ids) for t in NEW),
                    f"borrowed {o.get('borrowed_tests_from_head')}"))
    b, h = _cells(obs.get("base")), _cells(obs.get("head"))
    be = [k for k in b if k.startswith("eager")]
    bad = [k for k in be if not _done(b[k])]
    out.append(("base eager cells finished, finite", bool(be) and not bad,
                "; ".join(f"{k}: {b[k].get('error') or b[k].get('stage')} rc={b[k].get('rc')}" for k in bad) or f"{len(be)} cells"))
    bs = {(c.get("has_max_rows"), c.get("has_generic_calls")) for c in b.values() if c.get("stage") != "import"}
    hs = {(c.get("has_max_rows"), c.get("has_generic_calls")) for c in h.values() if c.get("stage") != "import"}
    out.append(("only the head has _triton_grouped_mm_max_rows and GENERIC_CALLS", bs == {(False, False)} and hs == {(True, True)},
                f"base {bs}, head {hs}"))
    hip = {str((c.get("versions") or {}).get("hip")) for c in list(b.values()) + list(h.values())}
    out.append(("every cell ran on HIP", bool(hip) and "None" not in hip, f"hip {hip}"))
    return out


def head_is_worse(base, head):
    problems, notes = [], []
    bc, hc = _cells(base), _cells(head)
    # gate off on HIP
    for k, c in hc.items():
        if (c.get("versions") or {}).get("hip") is None:
            problems.append(f"head {k}: not HIP ({c.get('versions')})")
        rows = c.get("max_rows_env") or {}
        forced = c.get("max_rows_forced_modes_dev0") or {}
        nz = {f"env idx={i} {kk}": v for i, d in rows.items() for kk, v in d.items() if v != 0}
        nz.update({f"mode={m} {kk}": v for m, d in forced.items() for kk, v in d.items() if v != 0})
        if c.get("stage") not in ("import",) and (not rows or not forced or set(next(iter(rows.values()), {})) != set(KINDS)):
            problems.append(f"head {k}: max_rows not recorded for every kind ({rows}, {forced})")
        if nz:
            problems.append(f"head {k}: Triton grouped gate ON on HIP: {nz}")
        gcd = c.get("generic_calls_final")
        if c.get("stage") == "done" and (not isinstance(gcd, dict) or any(v != 0 for v in gcd.values())):
            problems.append(f"head {k}: GENERIC_CALLS not all-zero: final {gcd}, delta {c.get('generic_calls_delta')}")
    # cells
    b1, b2 = bc.get("eager/r1") or {}, bc.get("eager/r2") or {}
    aa = _done(b1) and _done(b2) and _sig(b1) == _sig(b2)
    notes.append(f"base eager A/A bit-identical={aa}")
    # compiled: Unsloth defaults (dynamo + inductor). Judged against the base compiled A/A set (r1..r3).
    bcomp = [c for k, c in bc.items() if k.startswith("compiled") and _done(c)]
    bsigs = [_sig(c) for c in bcomp]
    caa = len(set(map(repr, bsigs))) <= 1
    cspread = 0.0
    for i in range(len(bcomp)):
        for j in range(i + 1, len(bcomp)):
            cspread = max([cspread] + [abs(float(x) - float(y)) for x, y in zip(bcomp[i].get("loss_reprs") or [], bcomp[j].get("loss_reprs") or [])])
    notes.append(f"base compiled A/A: {len(bcomp)} finished, bit-identical={caa}, loss spread {cspread:.3g}")
    for k, h in hc.items():
        b = bc.get(k) or {}
        if k.startswith("compiled"):
            if not _done(h):
                (problems if bcomp else notes).append(f"head {k} failed: {str(h.get('error'))[:200]} (base compiled finished: {len(bcomp)})")
                continue
            if not bcomp:
                continue
            if _sig(h) in bsigs:
                notes.append(f"{k}: head bit-identical to a base compiled run; GENERIC_CALLS {h.get('generic_calls_final')}")
                continue
            d = min(max((abs(float(x) - float(y)) for x, y in zip(h.get("loss_reprs") or [], c.get("loss_reprs") or [])), default = float("inf"))
                    for c in bcomp)
            if caa:
                problems.append(f"{k}: head differs from base though base compiled A/A is bit-stable (loss diff {d:.3g})")
            else:
                (problems if d > cspread + 1e-7 else notes).append(
                    f"{k}: head not bit-identical to any base compiled run; nearest loss diff {d:.3g} vs base compiled A/A spread {cspread:.3g}")
            continue
        if not _done(h):
            problems.append(f"head {k} did not finish: {h.get('error') or ('rc=' + str(h.get('rc')) + ' stage=' + str(h.get('stage')))}")
            continue
        if not _done(b):
            continue
        if _sig(h) == _sig(b):
            notes.append(f"{k}: head == base bit-identical (losses {h.get('loss_reprs')}, lora {h.get('lora_digest')}, "
                         f"grads {h.get('grad_digests')}); GENERIC_CALLS {h.get('generic_calls_final')}; "
                         f"torch._grouped_mm calls base {b.get('torch_grouped_mm_calls')} head {h.get('torch_grouped_mm_calls')}")
        elif aa or k.startswith("eager-forced"):
            problems.append(f"{k}: head differs from base (base A/A bit-stable={aa}): base {_sig(b)}, head {_sig(h)}")
        else:
            lb, lh = [float(x) for x in b.get("loss_reprs") or []], [float(x) for x in h.get("loss_reprs") or []]
            spread = max((abs(x - y) for x, y in zip([float(x) for x in b1.get("loss_reprs") or []],
                                                       [float(x) for x in b2.get("loss_reprs") or []])), default = 0.0)
            d = max((abs(x - y) for x, y in zip(lb, lh)), default = float("inf"))
            (problems if d > spread + 1e-7 else notes).append(f"{k}: base A/A not bit-stable (spread {spread:.3g}); head-vs-base {d:.3g}")
        if h.get("instrumented") and b.get("instrumented") and h.get("torch_grouped_mm_calls") != b.get("torch_grouped_mm_calls"):
            problems.append(f"{k}: torch._grouped_mm calls changed base {b.get('torch_grouped_mm_calls')} -> head {h.get('torch_grouped_mm_calls')}")
    # pytest
    bp, hp = (base or {}).get("pytest") or {}, (head or {}).get("pytest") or {}
    bpass = set(bp.get("passed") or [])
    hbad = set(hp.get("failed") or []) | set(hp.get("errors") or [])
    bbad = set(bp.get("failed") or []) | set(bp.get("errors") or [])
    lost = sorted(t for t in bpass if t not in set(hp.get("passed") or []))
    if lost:
        problems.append(f"passed at base, not at head: {lost}")
    new_head_fail = sorted(t for t in hbad - bbad)
    if new_head_fail:
        problems.append(f"failing at head, not at base: {new_head_fail}")
    pre = sorted(hbad & bbad)
    if pre:
        notes.append(f"failing at both (pre-existing): {pre}")
    fixed = sorted(t for t in bbad if t in set(hp.get("passed") or []))
    if fixed:
        notes.append(f"failing at base, passing at head: {fixed}")
    if problems:
        return True, "; ".join(problems)
    return False, "; ".join(notes)


def table(obs):
    rows = ["| state | cell | stage | losses | grad digests | lora digest | GENERIC_CALLS final | max_rows (env) | "
            "max_rows forced | torch._grouped_mm calls | moe backend / experts | grouped_mm supported |",
            "|" + "---|" * 12]
    for name in ("base", "head", "merge"):
        for k, c in _cells(obs.get(name)).items():
            rows.append(f"| {name} | {k} | {c.get('stage')}{' ERR ' + str(c.get('error'))[:200] if c.get('error') else ''} | "
                        f"{', '.join(c.get('loss_reprs') or [])} | {', '.join(c.get('grad_digests') or [])} | {c.get('lora_digest')} | "
                        f"{c.get('generic_calls_final')} | {c.get('max_rows_env')} | {c.get('max_rows_forced_modes_dev0')} | "
                        f"{c.get('torch_grouped_mm_calls')} | {c.get('experts_class')} | {c.get('grouped_mm_supported')} |")
    rows += ["", "| state | pytest passed | failed | skipped | failing tests |", "|---|---|---|---|---|"]
    for name in ("base", "head"):
        p = (obs.get(name) or {}).get("pytest") or {}
        if not p:
            continue
        bad = sorted(set(p.get("failed") or []) | set(p.get("errors") or []))
        rows.append(f"| {name} | {p.get('n_passed')} | {p.get('n_failed')} | {p.get('n_skipped')} | {', '.join(bad) or 'none'} |")
    rows.append("")
    for name in ("base", "head"):
        p = (obs.get(name) or {}).get("pytest") or {}
        newt = {t: ("FAIL" if t in (p.get("failed") or []) + (p.get("errors") or []) else
                    "skip: " + (p.get("skipped") or {})[t] if t in (p.get("skipped") or {}) else "pass")
                for t in (p.get("passed") or []) + (p.get("failed") or []) + (p.get("errors") or []) + list((p.get("skipped") or {}).keys())
                if any(n in t for n in NEW)}
        rows.append(f"{name} new tests: " + "; ".join(f"`{t}` {v}" for t, v in sorted(newt.items())))
        rows.append("")
    v = next((c.get("versions") for o in obs.values() if isinstance(o, dict) for c in _cells(o).values() if c.get("versions")), None)
    rows.append(f"versions: {v}")
    return "\n".join(rows)
