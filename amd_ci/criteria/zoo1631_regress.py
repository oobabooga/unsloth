#!/usr/bin/env python3
"""Criteria (unsloth-zoo PR 1631, reuse flex attention block masks), gfx1151 ROCm, base vs head.
Leg from the observations ("t4" = transformers 4.57.6, "t5" = installed transformers 5.x).

Regression mode. Head is WORSE than base when any of:
  * a test that passed at the base does not pass at the head, or fails at head and not at base
  * (train) a head cell crashes or has a non-finite loss / grad
  * (train) head reuse/r1 is not bit-identical (per-step loss, logits digest, LoRA grad digest, post-step param digest)
    to head off/r1 (UNSLOTH_FLEX_MASK_REUSE=0) although the head A/A is; or head vs base differs although the base A/A
    is; or any of those pairs differs in loss by more than DIVERGE
Gates: each state imported its own checkout; pytest ran and collected; the new test file ran in both states;
(train) every cell on HIP with the expected transformers major, base cells finished, the patched gpt-oss attention
called flex_attention_with_sink in every cell (else flex attention was not exercised), head reuse/r1 built FEWER
block masks than head off/r1 (reuse engaged), head off/r1 built as many as base reuse/r1 (kill switch restores base).
"""

from __future__ import annotations

TITLE = "unsloth-zoo PR 1631 on gfx1151 ROCm, base vs head (pytest + tiny gpt-oss flex-attention LoRA cells)"
MODE = "regression"
NEEDS = ["rocm", "gpu", "nvidia"]

_RAN = (0, 1)
DIVERGE = 0.05


def _cells(o):
    return (o or {}).get("train") or {}


def _done(c):
    return c.get("stage") == "done" and c.get("ok") is True and not c.get("error")


def _sig(c):
    return [(s.get("loss"), s.get("logits_digest"), s.get("grad_digest"), s.get("param_digest")) for s in c.get("steps") or []]


def _meta(obs, k):
    return next((o.get(k) for o in obs.values() if isinstance(o, dict) and o.get(k) is not None), None)


def _dloss(a, b):
    la = [float(s["loss"]) for s in a.get("steps") or []]
    lb = [float(s["loss"]) for s in b.get("steps") or []]
    if not la or len(la) != len(lb):
        return float("inf")
    return max(abs(x - y) for x, y in zip(la, lb))


def _dlogit(a, b):
    out = 0.0
    for x, y in zip(a.get("steps") or [], b.get("steps") or []):
        for u, v in zip(x.get("logits_slice") or [], y.get("logits_slice") or []):
            out = max(out, abs(float(u) - float(v)))
    return out


def gates(obs):
    out = []
    leg = _meta(obs, "leg")
    train = bool(_meta(obs, "train_enabled"))
    new = [t.rsplit("/", 1)[-1].removesuffix(".py") for t in (_meta(obs, "new_tests") or [])]
    out.append(("leg recorded", leg in ("t4", "t5"), f"leg {leg}, pr {_meta(obs, 'pr')}, new tests {new}"))
    out.append(("new test files named", bool(new), str(new)))
    allc = []
    for name in ("base", "head"):
        o = obs.get(name) or {}
        ck = str(o.get("checkout") or "<none>")
        p = o.get("pytest") or {}
        out.append((f"{name} pytest ran", p.get("rc") in _RAN,
                    f"rc={p.get('rc')} {str(p.get('stderr_tail') or '')[-200:] if p.get('rc') not in _RAN else ''}"))
        n = p.get("n_passed", 0) + p.get("n_failed", 0) + p.get("n_skipped", 0)
        out.append((f"{name} pytest collected", n > 0,
                    f"passed {p.get('n_passed', 0)}, failed {p.get('n_failed', 0)}, skipped {p.get('n_skipped', 0)}"))
        ids = list(p.get("passed") or []) + list(p.get("failed") or []) + list(p.get("errors") or []) + list((p.get("skipped") or {}).keys())
        out.append((f"{name} ran the new test files", bool(new) and all(any(t in i for i in ids) for t in new),
                    f"borrowed {o.get('borrowed_tests_from_head')}"))
        if train:
            cells = _cells(o)
            files = {str(c.get("unsloth_zoo_file") or "") for c in cells.values()}
            out.append((f"{name} train cells imported the state's unsloth_zoo",
                        bool(cells) and all(f and f.startswith(ck) for f in files), ", ".join(sorted(files)) or "no cells"))
            allc += [c for c in cells.values() if c.get("versions")]
    if train:
        b, h = _cells(obs.get("base")), _cells(obs.get("head"))
        bad = [k for k in b if not _done(b[k])]
        out.append(("base train cells finished, finite", bool(b) and not bad,
                    "; ".join(f"{k}: {str(b[k].get('error') or b[k].get('stage'))[:300]} rc={b[k].get('rc')}" for k in bad) or f"{len(b)} cells"))
        hip = {str(c["versions"].get("hip")) for c in allc}
        out.append(("every cell ran on HIP", bool(hip) and "None" not in hip, f"hip {hip}"))
        tv = {str(c["versions"].get("transformers")) for c in allc}
        out.append((f"{leg} leg ran the expected transformers major",
                    bool(tv) and all(v.startswith("4." if leg == "t4" else "5.") for v in tv), f"transformers {tv}"))
        sink = {f"{s}:{k}": c.get("sink_calls_total") for s, cc in (("base", b), ("head", h)) for k, c in cc.items()}
        out.append(("flex_attention_with_sink ran in every cell (gpt-oss flex path exercised)",
                    bool(sink) and all((v or 0) > 0 for v in sink.values()),
                    f"sink calls {sink}; hooks {[c.get('sink_counter_hooks') for c in list(b.values()) + list(h.values())]}; "
                    f"attention forward module {sorted({str(c.get('attention_forward_module')) for c in list(b.values()) + list(h.values())})}"))
        hr, ho, br = h.get("reuse/r1") or {}, h.get("off/r1") or {}, b.get("reuse/r1") or {}
        out.append(("head reuse/r1 built fewer block masks than head off/r1 (reuse engaged)",
                    _done(hr) and _done(ho) and (hr.get("mask_builds_total") or 0) > 0
                    and hr.get("mask_builds_total") < (ho.get("mask_builds_total") or 0),
                    f"builds reuse {hr.get('mask_builds_total')} off {ho.get('mask_builds_total')}; per step reuse "
                    f"{[s.get('mask_builds') for s in hr.get('steps') or []]} off {[s.get('mask_builds') for s in ho.get('steps') or []]}"))
        out.append(("head off/r1 (UNSLOTH_FLEX_MASK_REUSE=0) builds as many masks as base",
                    _done(ho) and _done(br) and ho.get("mask_builds_total") == br.get("mask_builds_total"),
                    f"head off {ho.get('mask_builds_total')} base reuse/r1 {br.get('mask_builds_total')}"))
    return out


def head_is_worse(base, head):
    problems, notes = [], []
    leg = (head or {}).get("leg") or (base or {}).get("leg")
    bc, hc = _cells(base), _cells(head)
    if hc:
        b1, b2 = bc.get("reuse/r1") or {}, bc.get("reuse/r2") or {}
        h1, h2, ho = hc.get("reuse/r1") or {}, hc.get("reuse/r2") or {}, hc.get("off/r1") or {}
        baa = _done(b1) and _done(b2) and _sig(b1) == _sig(b2)
        haa = _done(h1) and _done(h2) and _sig(h1) == _sig(h2)
        notes.append(f"[{leg}] base A/A bit-identical={baa}; head A/A bit-identical={haa}")
        for k, h in hc.items():
            if not _done(h):
                problems.append(f"head {k} did not finish: {str(h.get('error') or ('rc=' + str(h.get('rc')) + ' stage=' + str(h.get('stage'))))[:300]}")
        if _done(h1) and _done(ho):
            same = _sig(h1) == _sig(ho)
            line = (f"[{leg}] head reuse vs head UNSLOTH_FLEX_MASK_REUSE=0 bit-identical (loss, logits, grads, params)={same}, "
                    f"max |dloss| {_dloss(h1, ho):.3g}, max |dlogit| {_dlogit(h1, ho):.3g}; mask builds reuse "
                    f"{h1.get('mask_builds_total')} off {ho.get('mask_builds_total')}; losses {[s.get('loss') for s in h1.get('steps') or []]}")
            (problems.append("REUSE != OFF " + line) if (haa and not same) or _dloss(h1, ho) > DIVERGE else notes.append(line))
        for k in ("reuse/r1", "off/r1"):
            h, ref = hc.get(k) or {}, bc.get(k) or {}
            if _done(h) and _done(ref):
                same = _sig(h) == _sig(ref)
                line = (f"[{leg}] {k} head vs base bit-identical={same}, max |dloss| {_dloss(h, ref):.3g}, "
                        f"max |dlogit| {_dlogit(h, ref):.3g}; mask builds head {h.get('mask_builds_total')} base {ref.get('mask_builds_total')}")
                (problems.append("HEAD != BASE " + line) if (baa and not same) or _dloss(h, ref) > DIVERGE else notes.append(line))
    bp, hp = (base or {}).get("pytest") or {}, (head or {}).get("pytest") or {}
    bpass = set(bp.get("passed") or [])
    hbad = set(hp.get("failed") or []) | set(hp.get("errors") or [])
    bbad = set(bp.get("failed") or []) | set(bp.get("errors") or [])
    lost = sorted(t for t in bpass if t not in set(hp.get("passed") or []))
    if lost:
        problems.append(f"[{leg}] passed at base, not at head: {lost}")
    new_head_fail = sorted(hbad - bbad)
    if new_head_fail:
        problems.append(f"[{leg}] failing at head, not at base: {new_head_fail}")
    pre = sorted(hbad & bbad)
    if pre:
        notes.append(f"[{leg}] failing at both: {pre}")
    fixed = sorted(t for t in bbad if t in set(hp.get("passed") or []))
    if fixed:
        notes.append(f"[{leg}] failing at base, passing at head: {len(fixed)} tests, e.g. {fixed[:8]}")
    notes.append(f"[{leg}] pytest base {bp.get('n_passed')}/{bp.get('n_failed')}/{bp.get('n_skipped')} head "
                 f"{hp.get('n_passed')}/{hp.get('n_failed')}/{hp.get('n_skipped')} (passed/failed/skipped)")
    if problems:
        return True, "; ".join(problems)
    return False, "; ".join(notes)


def table(obs):
    rows = ["| state | cell | stage | sink calls | mask builds (per step) | losses | logits digests | grad digests | final digest | transformers |",
            "|" + "---|" * 10]
    for name in ("base", "head"):
        for k, c in _cells(obs.get(name)).items():
            st = c.get("steps") or []
            rows.append(f"| {name} | {k} | {c.get('stage')}{' ERR ' + str(c.get('error'))[:200] if c.get('error') else ''} | "
                        f"{c.get('sink_calls_total')} | {c.get('mask_builds_total')} ({[s.get('mask_builds') for s in st]}) | "
                        f"{', '.join(str(s.get('loss')) for s in st)} | {', '.join(str(s.get('logits_digest')) for s in st)} | "
                        f"{', '.join(str(s.get('grad_digest')) for s in st)} | {c.get('final_digest')} | "
                        f"{(c.get('versions') or {}).get('transformers')} |")
    rows += ["", "| state | pytest passed | failed | skipped | failing tests |", "|---|---|---|---|---|"]
    for name in ("base", "head"):
        p = (obs.get(name) or {}).get("pytest") or {}
        if not p:
            continue
        bad = sorted(set(p.get("failed") or []) | set(p.get("errors") or []))
        rows.append(f"| {name} | {p.get('n_passed')} | {p.get('n_failed')} | {p.get('n_skipped')} | {', '.join(bad) or 'none'} |")
    rows.append("")
    new = [t.rsplit("/", 1)[-1].removesuffix(".py") for t in (_meta(obs, "new_tests") or [])]
    for name in ("base", "head"):
        p = (obs.get(name) or {}).get("pytest") or {}
        ids = (p.get("passed") or []) + (p.get("failed") or []) + (p.get("errors") or []) + list((p.get("skipped") or {}).keys())
        newt = {t: ("FAIL" if t in (p.get("failed") or []) + (p.get("errors") or []) else
                    "skip: " + (p.get("skipped") or {})[t] if t in (p.get("skipped") or {}) else "pass")
                for t in ids if any(n in t for n in new)}
        rows.append(f"{name} new tests: " + "; ".join(f"`{t}` {v}" for t, v in sorted(newt.items())))
        rows.append("")
        for t, m in sorted((p.get("fail_msgs") or {}).items()):
            rows.append(f"{name} FAIL `{t}`: {m[:600]!r}")
        rows.append("")
    return "\n".join(rows)
