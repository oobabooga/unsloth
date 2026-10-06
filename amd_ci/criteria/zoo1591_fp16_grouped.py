#!/usr/bin/env python3
"""Criteria (unsloth-zoo PR 1591): gpt-oss grouped QLoRA on float16 (Triton / cuBLAS grouped GEMMs with an fp32
down output). gfx1151 ROCm, base vs head.

Regression mode. Head is WORSE than base when any of:
  * a head training cell crashes or has a non-finite loss
  * head bfloat16 is not identical to base bfloat16 (loss reprs + LoRA digest) while the base A/A pair is
    bit-stable (else: differs by more than the base A/A spread), or the fp16 grouped path ran in a bf16 cell
  * head float16 runs a MIXED path (some layers grouped, some loop) in one cell
  * head float16 keeps the per-expert loop WITHOUT a logged reason (LAST_DECLINE / readiness log), or its
    losses then differ from base float16 (same loop) beyond the base A/A spread
  * head float16 engages the grouped path and its losses differ from base float16 by more than
    max(4x the base fp16 A/A spread, 1e-4 relative)
  * a test that passed at the base does not pass at the head, or a test in tests/test_moe_grouped_fp16.py
    fails (not skips) at the head

Gates: each state imported its own checkout; pytest ran and collected; the new test file ran in both states
(copied from head into base); every cell loaded NF4 (Params4bit) experts into GptOssExpertsBnb4bit;
base bfloat16 cells finished; only the head has LAST_DECLINE.
"""

from __future__ import annotations

TITLE = "unsloth-zoo PR 1591: gpt-oss grouped QLoRA on float16 on gfx1151 ROCm, base vs head"
MODE = "regression"
NEEDS = ["rocm", "gpu", "nvidia"]

_RAN = (0, 1)
NEW = ("test_moe_grouped_fp16",)


def _cells(o):
    return (o or {}).get("train") or {}


def _done(c):
    return c.get("stage") == "done" and c.get("ok") is True and not c.get("error")


def _l(c):
    return c.get("losses") or []


def _nf4(c):
    d = c.get("expert0_dtypes") or {}
    return bool(c.get("is_bnb4bit_experts")) and all((d.get(k) or [None] * 4)[3] == "Params4bit" for k in ("gate_up", "down"))


def gates(obs):
    out = []
    for name in ("base", "head"):
        o = obs.get(name) or {}
        ck = str(o.get("checkout") or "<none>")
        cells = _cells(o)
        files = {str(c.get("unsloth_zoo_file") or "") for c in cells.values()}
        out.append((f"{name} train cells imported the state's unsloth_zoo",
                    bool(cells) and all(f and f.startswith(ck) for f in files), ", ".join(sorted(files)) or "no cells"))
        bad = [k for k, c in cells.items() if not _nf4(c)]
        out.append((f"{name} every cell loaded NF4 Params4bit experts (GptOssExpertsBnb4bit)", bool(cells) and not bad,
                    "; ".join(f"{k}: {cells[k].get('expert_module_class')} {cells[k].get('expert0_dtypes')} "
                              f"{cells[k].get('error') or ''}" for k in bad) or f"{len(cells)} cells"))
        p = o.get("pytest") or {}
        out.append((f"{name} pytest ran", p.get("rc") in _RAN,
                    f"rc={p.get('rc')} {str(p.get('stderr_tail') or '')[-200:] if p.get('rc') not in _RAN else ''}"))
        n = p.get("n_passed", 0) + p.get("n_failed", 0) + p.get("n_skipped", 0)
        out.append((f"{name} pytest collected", n > 0,
                    f"passed {p.get('n_passed', 0)}, failed {p.get('n_failed', 0)}, skipped {p.get('n_skipped', 0)}"))
        ids = list(p.get("passed") or []) + list(p.get("failed") or []) + list(p.get("errors") or []) + list((p.get("skipped") or {}).keys())
        out.append((f"{name} ran the new test file", all(any(t in i for i in ids) for t in NEW),
                    f"borrowed {o.get('borrowed_tests_from_head')}"))
    b, h = _cells(obs.get("base")), _cells(obs.get("head"))
    bb = [k for k in b if k.startswith("bfloat16/")]
    bad = [k for k in bb if not _done(b[k])]
    out.append(("base bfloat16 cells finished, finite", bool(bb) and not bad,
                "; ".join(f"{k}: {b[k].get('error') or b[k].get('stage')} rc={b[k].get('rc')}" for k in bad) or f"{len(bb)} cells"))
    battr = {c.get("has_last_decline") for c in b.values() if c.get("stage") != "import"}
    hattr = {c.get("has_last_decline") for c in h.values() if c.get("stage") != "import"}
    out.append(("only the head has gpt_oss_grouped_qlora.LAST_DECLINE", battr == {False} and hattr == {True},
                f"base {battr}, head {hattr}"))
    return out


def _maxdiff(a, b):
    return max(abs(x - y) for x, y in zip(_l(a), _l(b))) if _l(a) and _l(b) else None


def _same(a, b):
    return a.get("loss_reprs") == b.get("loss_reprs") and a.get("lora_digest") == b.get("lora_digest")


def head_is_worse(base, head):
    problems, notes = [], []
    bc, hc = _cells(base), _cells(head)
    for k, c in hc.items():
        if not _done(c):
            problems.append(f"head {k} did not finish: {c.get('error') or ('rc=' + str(c.get('rc')) + ' stage=' + str(c.get('stage')))}")
    # bf16: identical to base
    b1, b2 = bc.get("bfloat16/r1") or {}, bc.get("bfloat16/r2") or {}
    hs = [hc.get(f"bfloat16/{r}") or {} for r in ("r1", "r2")]
    for k in ("bfloat16/r1", "bfloat16/r2"):
        fp = int(((hc.get(k) or {}).get("calls_delta") or {}).get("forward_fp16", 0))
        if fp:
            problems.append(f"head {k}: fp16 grouped path ran {fp} times in a bf16 cell")
    if all(_done(x) for x in (b1, b2, *hs)):
        aa = _same(b1, b2)
        if all(_same(x, b1) for x in hs):
            notes.append(f"bf16 head == base bit-identical (losses {b1.get('loss_reprs')}, digest {b1.get('lora_digest')}); "
                         f"base A/A exact={aa}; path base {b1.get('engaged')} {b1.get('calls_delta')}, head {hs[0].get('engaged')} {hs[0].get('calls_delta')}")
        elif aa:
            problems.append(f"bf16 head differs from base though base A/A is bit-stable: base {b1.get('loss_reprs')} / {b1.get('lora_digest')}, "
                            f"head {[(x.get('loss_reprs'), x.get('lora_digest')) for x in hs]}")
        else:
            spread = _maxdiff(b1, b2) or 0.0
            d = max(_maxdiff(x, b1) or 0.0 for x in hs)
            (problems if d > spread + 1e-6 else notes).append(f"bf16 base A/A not bit-stable (spread {spread:.3g}); head-vs-base max {d:.3g}")
    # fp16
    f1, f2 = bc.get("float16/r1") or {}, bc.get("float16/r2") or {}
    f_aa = _maxdiff(f1, f2) if _done(f1) and _done(f2) else None
    notes.append(f"fp16 base A/A: bit-identical={_same(f1, f2)}, loss spread {f_aa}")
    for r in ("r1", "r2"):
        bf, hf = bc.get(f"float16/{r}") or {}, hc.get(f"float16/{r}") or {}
        if not (_done(hf) and _done(bf)):
            continue
        eng = hf.get("engaged")
        cd = hf.get("calls_delta") or {}
        d = _maxdiff(hf, bf)
        desc = (f"float16/{r}: head path {eng} (calls {cd}, LAST_DECLINE {hf.get('last_decline')}, readiness "
                f"{hf.get('grouped_ready')}, log {hf.get('grouped_log')}); base path {bf.get('engaged')} "
                f"(log {bf.get('grouped_log')}); head-vs-base loss max diff {d}")
        if eng == "mixed" or eng == "none":
            problems.append(desc + " -> mixed / no expert path")
            continue
        if eng == "loop":
            reason = ((hf.get("last_decline") or {}).get("reason") if isinstance(hf.get("last_decline"), dict) else None)
            logged = reason or any("disabled" in x or "per-expert loop" in x for x in (hf.get("grouped_log") or []))
            if not logged:
                problems.append(desc + " -> kept the loop with no logged reason")
                continue
            if _same(hf, bf):
                notes.append(desc + " -> declined cleanly, bit-identical to base (same loop)")
            elif f_aa is not None and d is not None and d <= f_aa + 1e-6:
                notes.append(desc + f" -> declined cleanly, within base A/A spread {f_aa}")
            else:
                problems.append(desc + f" -> declined but losses differ from base beyond A/A spread {f_aa}")
            continue
        lim = max(4 * (f_aa or 0.0), 1e-4 * max(abs(v) for v in _l(bf)))
        (problems if d is None or d > lim else notes).append(desc + f" -> grouped engaged (limit {lim:.3g})")
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
    return f"{v:.9g}" if isinstance(v, float) else str(v)


def table(obs):
    rows = ["| state | cell | stage | losses | lora digest | engaged | calls delta | LAST_DECLINE | readiness | grouped log | "
            "grouped_mm supported | fp16 grouped available | expert dtypes (gate_up / down) | dynamo reset retries |",
            "|" + "---|" * 14]
    for name in ("base", "head", "merge"):
        for k, c in _cells(obs.get(name)).items():
            d = c.get("expert0_dtypes") or {}
            rows.append(f"| {name} | {k} | {c.get('stage')}{' ERR ' + str(c.get('error'))[:200] if c.get('error') else ''} | "
                        f"{', '.join(_f(x) for x in _l(c))} | {c.get('lora_digest')} | {c.get('engaged')} | {c.get('calls_delta')} | "
                        f"{c.get('last_decline')} | {c.get('grouped_ready')} | {'; '.join(c.get('grouped_log') or [])[:300]} | "
                        f"{c.get('grouped_mm_supported')} | {c.get('fp16_grouped_available')} {c.get('fp16_unavailable_reason') or ''} | "
                        f"{d.get('gate_up')} / {d.get('down')} | {[r.get('step') for r in (c.get('dynamo_reset_retries') or [])]} |")
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
                if any(n in t for n in NEW) or "fp16" in t or "float16" in t}
        rows.append(f"{name} float16 / new tests: " + "; ".join(f"`{t}` {v}" for t, v in sorted(newt.items())))
        rows.append("")
    v = next((c.get("versions") for o in obs.values() if isinstance(o, dict) for c in _cells(o).values() if c.get("versions")), None)
    rows.append(f"versions: {v}")
    return "\n".join(rows)
