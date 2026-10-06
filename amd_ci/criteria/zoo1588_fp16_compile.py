#!/usr/bin/env python3
"""Criteria (unsloth-zoo PR 1588): float16 grouped GEMM under torch.compile via an opaque custom op, plus a compiler
fallback importing a torch module whose standalone copy failed. gfx1151 ROCm, base vs head.

Regression mode. Head is WORSE than base when any of:
  * a head training cell crashes or has a non-finite loss
  * head bfloat16 training is not identical to base bfloat16 (loss reprs + LoRA digest), or, when the base A/A
    pair itself is not bit-stable, differs from base by more than the base A/A spread
  * head float16 has more dynamo failure lines / failure records than base float16
  * head float16 loss is further from head bfloat16 than max(2x the base fp16-vs-bf16 gap, 5% of bf16 loss)
  * a test that passed at the base does not pass at the head, or a test in the two new files fails at the head

Gates: each state imported its own checkout; pytest ran and collected; the two new test files ran in both
states (copied from head into base); base bfloat16 cells finished; only the head has _GROUPED_MM_FP16_OP.
"""

from __future__ import annotations

TITLE = "unsloth-zoo PR 1588: compiled float16 grouped GEMM + compiler import fallback on gfx1151 ROCm, base vs head"
MODE = "regression"
NEEDS = ["rocm", "gpu", "nvidia"]

_RAN = (0, 1)
NEW = ("test_moe_grouped_mm_fp16_compile", "test_compiler_failed_standalone_import")


def _cells(o):
    return (o or {}).get("train") or {}


def _done(c):
    return c.get("stage") == "done" and c.get("ok") is True and not c.get("error")


def _fails(c):
    return int(c.get("n_log_failure_lines") or 0) + int(c.get("n_dynamo_failures") or 0)


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
        out.append((f"{name} pytest ran", p.get("rc") in _RAN, f"rc={p.get('rc')} {str(p.get('stderr_tail') or '')[-200:] if p.get('rc') not in _RAN else ''}"))
        n = p.get("n_passed", 0) + p.get("n_failed", 0) + p.get("n_skipped", 0)
        out.append((f"{name} pytest collected", n > 0,
                    f"passed {p.get('n_passed', 0)}, failed {p.get('n_failed', 0)}, skipped {p.get('n_skipped', 0)}"))
        ids = list(p.get("passed") or []) + list(p.get("failed") or []) + list(p.get("errors") or []) + list((p.get("skipped") or {}).keys())
        out.append((f"{name} ran both new test files", all(any(t in i for i in ids) for t in NEW),
                    f"borrowed {o.get('borrowed_tests_from_head')}"))
    b, h = _cells(obs.get("base")), _cells(obs.get("head"))
    bb = [k for k in b if k.startswith("bfloat16/")]
    bad = [k for k in bb if not _done(b[k])]
    out.append(("base bfloat16 cells finished, finite", bool(bb) and not bad,
                "; ".join(f"{k}: {b[k].get('error') or b[k].get('stage')} rc={b[k].get('rc')}" for k in bad) or f"{len(bb)} cells"))
    battr = {c.get("has_fp16_op_attr") for c in b.values() if c.get("stage") != "import"}
    hreg = {c.get("fp16_op_registered") for c in h.values() if c.get("stage") != "import"}
    out.append(("only the head has _GROUPED_MM_FP16_OP (registered)", battr == {False} and hreg == {True},
                f"base has_attr {battr}, head registered {hreg}"))
    return out


def _l(c):
    return c.get("losses") or []


def head_is_worse(base, head):
    problems, notes = [], []
    bc, hc = _cells(base), _cells(head)
    for k, c in hc.items():
        if not _done(c):
            problems.append(f"head {k} did not finish: {c.get('error') or ('rc=' + str(c.get('rc')) + ' stage=' + str(c.get('stage')))}")
    # bf16 identity
    b1, b2, h1, h2 = (bc.get("bfloat16/r1") or {}), (bc.get("bfloat16/r2") or {}), (hc.get("bfloat16/r1") or {}), (hc.get("bfloat16/r2") or {})
    if all(_done(x) for x in (b1, b2, h1, h2)):
        aa_exact = b1.get("loss_reprs") == b2.get("loss_reprs") and b1.get("lora_digest") == b2.get("lora_digest")
        hb_exact = all(x.get("loss_reprs") == b1.get("loss_reprs") and x.get("lora_digest") == b1.get("lora_digest") for x in (h1, h2))
        if hb_exact:
            notes.append(f"bf16 head == base bit-identical (losses {b1.get('loss_reprs')}, digest {b1.get('lora_digest')}); base A/A exact={aa_exact}")
        elif aa_exact:
            problems.append(f"bf16 head differs from base though base A/A is bit-stable: base {b1.get('loss_reprs')} / {b1.get('lora_digest')}, "
                            f"head {h1.get('loss_reprs')} / {h1.get('lora_digest')}, {h2.get('loss_reprs')} / {h2.get('lora_digest')}")
        else:
            spread = max(abs(x - y) for x, y in zip(_l(b1), _l(b2)))
            d = max(abs(x - y) for hx in (h1, h2) for x, y in zip(_l(hx), _l(b1)))
            (problems if d > spread + 1e-6 else notes).append(f"bf16 base A/A not bit-stable (spread {spread:.3g}); head-vs-base max {d:.3g}")
    # fp16
    for r in ("r1", "r2"):
        bf, hf = bc.get(f"float16/{r}") or {}, hc.get(f"float16/{r}") or {}
        if _fails(hf) > _fails(bf):
            problems.append(f"float16/{r}: head dynamo failures {_fails(hf)} > base {_fails(bf)}: {(hf.get('log_failure_lines') or [])[:3]} {hf.get('dynamo_failures', [])[:3]}")
        notes.append(f"float16/{r}: dynamo failures base {_fails(bf)} -> head {_fails(hf)}; frames ok/total base "
                     f"{bf.get('frames_ok')}/{bf.get('frames_total')} -> head {hf.get('frames_ok')}/{hf.get('frames_total')}; "
                     f"grouped_mm_fp16 op calls base {bf.get('fp16_op_calls')} -> head {hf.get('fp16_op_calls')}; "
                     f"aten::_grouped_mm base {bf.get('aten_grouped_mm_calls')} -> head {hf.get('aten_grouped_mm_calls')}")
        hb = hc.get(f"bfloat16/{r}") or {}
        bbf = bc.get(f"bfloat16/{r}") or {}
        if _done(hf) and _done(hb) and _l(hf) and _l(hb):
            hgap = max(abs(x - y) for x, y in zip(_l(hf), _l(hb)))
            bgap = max(abs(x - y) for x, y in zip(_l(bf), _l(bbf))) if _done(bf) and _done(bbf) and _l(bf) else None
            lim = max(2 * (bgap or 0.0), 0.05 * max(abs(v) for v in _l(hb)))
            (problems if hgap > lim else notes).append(
                f"float16/{r} vs bfloat16 loss max gap: head {hgap:.4g}, base {bgap if bgap is None else f'{bgap:.4g}'} (limit {lim:.4g})")
    # pytest
    bp, hp = (base or {}).get("pytest") or {}, (head or {}).get("pytest") or {}
    bpass = set(bp.get("passed") or [])
    hbad = set(hp.get("failed") or []) | set(hp.get("errors") or [])
    bbad = set(bp.get("failed") or []) | set(bp.get("errors") or [])
    lost = sorted(t for t in bpass if t not in set(hp.get("passed") or []))
    if lost:
        problems.append(f"passed at base, not at head: {lost}")
    new_fail = sorted(t for t in hbad if any(n in t for n in NEW))
    if new_fail:
        problems.append(f"new-test failures at head: {new_fail}")
    pre = sorted(hbad & bbad)
    if pre:
        notes.append(f"failing at both (pre-existing): {pre}")
    fixed = sorted(t for t in bbad if t in set(hp.get("passed") or []))
    if fixed:
        notes.append(f"failing at base, passing at head: {fixed}")
    if problems:
        return True, "; ".join(problems)
    return False, "; ".join(notes)


def _f(v):
    return f"{v:.6g}" if isinstance(v, float) else str(v)


def table(obs):
    rows = ["| state | cell | stage | losses | lora digest | dynamo failures (log / records) | frames ok/total | "
            "grouped_mm_fp16 op calls | aten::_grouped_mm calls | grouped_mm supported | expert LoRA params | expert grad max |",
            "|" + "---|" * 12]
    for name in ("base", "head", "merge"):
        for k, c in _cells(obs.get(name)).items():
            rows.append(f"| {name} | {k} | {c.get('stage')}{' ERR ' + str(c.get('error'))[:200] if c.get('error') else ''} | "
                        f"{', '.join(_f(x) for x in _l(c))} | {c.get('lora_digest')} | {c.get('n_log_failure_lines')} / {c.get('n_dynamo_failures')} | "
                        f"{c.get('frames_ok')}/{c.get('frames_total')} | {c.get('fp16_op_calls')} | {c.get('aten_grouped_mm_calls')} | "
                        f"{c.get('grouped_mm_supported')} | {c.get('n_expert_lora_params')} | {c.get('expert_grad_max')} |")
    rows += ["", "| state | pytest passed | failed | skipped | failing tests |", "|---|---|---|---|---|"]
    for name in ("base", "head", "merge"):
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
        rows.append(f"{name} new tests: " + "; ".join(f"`{t.split('::')[-1]}` {v}" for t, v in sorted(newt.items())))
        rows.append("")
    v = next((c.get("versions") for o in obs.values() if isinstance(o, dict) for c in _cells(o).values() if c.get("versions")), None)
    rows.append(f"versions: {v}")
    return "\n".join(rows)
