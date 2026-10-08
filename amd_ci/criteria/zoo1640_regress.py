#!/usr/bin/env python3
"""Criteria (unsloth-zoo PR 1640, moe_ready_epoch readiness / NF4 table reuse), gfx1151 ROCm, base vs head.
Leg from the observations ("t4" = transformers 4.57.6, "t5" = installed transformers 5.x).

Observations: per state, processes bf16/p1, bf16/p2, nf4/p1, nf4/p2; each process trains fast/r1, off
(UNSLOTH_MOE_FAST_READY=0), fast/r2 on a fresh tiny Qwen3-MoE + expert LoRA, 6 AdamW steps.

Regression mode. Head is WORSE than base when any of:
  * a test that passed at the base does not pass at the head, or fails at head and not at base
  * a head run crashes or has a non-finite loss
  * in-process: head fast/r1 differs from head off (losses, grad norms, per-step digests, final digest) although
    head fast/r1 == head fast/r2, or |dloss| > DIVERGE
  * cross-process: head fast/r1 differs from base fast/r1 although both states' p1 == p2, or |dloss| > DIVERGE
  * where the experts are a ModuleList (the path this PR changes): a head fast run does a full check after
    optimizer step 1, or has no cheap / lean hits in steps 2..N; a head off run records any cheap / lean hit
Gates: each state imported its own checkout; pytest ran and collected; new test files ran in both states; every
run on HIP with the leg's transformers major; base runs finished; moe_ready_epoch absent at base, present at head;
where experts are a ModuleList, the head fast runs recorded full checks (the path engaged).
"""

from __future__ import annotations

TITLE = "unsloth-zoo PR 1640 on gfx1151 ROCm, base vs head (pytest + Qwen3-MoE expert LoRA readiness cells)"
MODE = "regression"
NEEDS = ["rocm", "gpu", "nvidia"]

_RAN = (0, 1)
DIVERGE = 0.05
PROCS = ("bf16/p1", "bf16/p2", "nf4/p1", "nf4/p2")


def _procs(o):
    return (o or {}).get("train") or {}


def _runs(p):
    return (p or {}).get("runs") or {}


def _done(r):
    return (r or {}).get("stage") == "done" and r.get("ok") is True and not r.get("error")


def _gn(r):
    return [x.get("grad_norm") for x in (r or {}).get("logs") or [] if "grad_norm" in x]


def _sig(r):
    r = r or {}
    return (r.get("loss_reprs"), _gn(r), r.get("step_digests"), r.get("final_digest"))


def _meta(obs, k):
    return next((o.get(k) for o in obs.values() if isinstance(o, dict) and o.get(k) is not None), None)


def _dloss(a, b):
    la, lb = [float(x) for x in a.get("loss_reprs") or []], [float(x) for x in b.get("loss_reprs") or []]
    if not la or len(la) != len(lb):
        return float("inf")
    return max(abs(x - y) for x, y in zip(la, lb))


def _modulelist(r):
    return "ModuleList" in ((r or {}).get("experts_class") or [])


def _all_runs(o):
    for pk, p in _procs(o).items():
        for rk, r in _runs(p).items():
            yield pk, rk, p, r


def gates(obs):
    out = []
    leg = _meta(obs, "leg")
    train = bool(_meta(obs, "train_enabled"))
    new = [t.rsplit("/", 1)[-1].removesuffix(".py") for t in (_meta(obs, "new_tests") or [])]
    out.append(("leg recorded", leg in ("t4", "t5"), f"leg {leg}, pr {_meta(obs, 'pr')}, new tests {new}"))
    out.append(("new test files named", bool(new), str(new)))
    allp = []
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
            procs = _procs(o)
            files = {str(c.get("unsloth_zoo_file") or "") for c in procs.values()}
            out.append((f"{name} train processes imported the state's unsloth_zoo",
                        len(procs) == len(PROCS) and all(f and f.startswith(ck) for f in files),
                        ", ".join(sorted(files)) or "no processes"))
            pres = {c.get("moe_ready_epoch_present") for c in procs.values()}
            want = name == "head"
            out.append((f"{name} moe_ready_epoch present = {want}", pres == {want}, f"{pres}"))
            allp += [c for c in procs.values() if c.get("versions")]
    if train:
        b = obs.get("base") or {}
        bad = [f"{pk}:{rk}: {str(r.get('error') or r.get('stage'))[:200]}" for pk, rk, p, r in _all_runs(b) if not _done(r)]
        nb = sum(1 for _ in _all_runs(b))
        out.append(("base train runs finished, finite", nb == 3 * len(PROCS) and not bad,
                    "; ".join(bad) or f"{nb} runs"))
        hip = {str(c["versions"].get("hip")) for c in allp}
        out.append(("every process ran on HIP", bool(hip) and "None" not in hip, f"hip {hip}"))
        tv = {str(c["versions"].get("transformers")) for c in allp}
        out.append((f"{leg} leg ran the expected transformers major",
                    bool(tv) and all(v.startswith("4." if leg == "t4" else "5.") for v in tv), f"transformers {tv}"))
        h = obs.get("head") or {}
        ml = [(pk, rk, r) for pk, rk, p, r in _all_runs(h) if rk.startswith("fast") and _modulelist(r)]
        if ml:
            eng = [(pk, rk, (r.get("counts_train_total") or {}).get("full", 0)) for pk, rk, r in ml]
            out.append(("head (ModuleList experts): readiness checks engaged in every fast run (full > 0)",
                        all(f > 0 for _, _, f in eng), "; ".join(f"{pk}:{rk} full={f}" for pk, rk, f in eng)))
        else:
            ec = sorted({str(r.get("experts_class")) for pk, rk, p, r in _all_runs(h)})
            out.append(("head experts layout recorded (ModuleList path not applicable on this leg)", bool(ec),
                        f"experts {ec}"))
    return out


def _cmp(a, b):
    return _sig(a) == _sig(b), _dloss(a, b)


def head_is_worse(base, head):
    problems, notes = [], []
    leg = (head or {}).get("leg") or (base or {}).get("leg")
    bp_, hp_ = _procs(base), _procs(head)
    if hp_:
        for pk, rk, p, r in _all_runs(head):
            if not _done(r):
                problems.append(f"[{leg}] head {pk}:{rk} did not finish: {str(r.get('error') or ('stage=' + str(r.get('stage'))))[:300]}")
        # in-process: fast/r1 vs off vs fast/r2
        for state, procs in (("base", bp_), ("head", hp_)):
            for pk in PROCS:
                rs = _runs(procs.get(pk))
                r1, r2, off = rs.get("fast/r1"), rs.get("fast/r2"), rs.get("off")
                if not (_done(r1) and _done(r2) and _done(off)):
                    continue
                aa, daa = _cmp(r1, r2)
                same, d = _cmp(r1, off)
                line = (f"[{leg}] {state} {pk} in-process: fast/r1==fast/r2 {aa}; fast/r1==off(FAST_READY=0) {same} "
                        f"max|dloss| {d:.3g}")
                if state == "head" and ((aa and not same) or d > DIVERGE):
                    problems.append("FAST != OFF " + line)
                else:
                    notes.append(line)
        # cross-process: base vs head, gated by each state's p1 == p2
        for dt in ("bf16", "nf4"):
            b1, b2 = _runs(bp_.get(f"{dt}/p1")).get("fast/r1"), _runs(bp_.get(f"{dt}/p2")).get("fast/r1")
            h1, h2 = _runs(hp_.get(f"{dt}/p1")).get("fast/r1"), _runs(hp_.get(f"{dt}/p2")).get("fast/r1")
            if not all(_done(x) for x in (b1, b2, h1, h2)):
                continue
            baa, haa = _sig(b1) == _sig(b2), _sig(h1) == _sig(h2)
            same, d = _cmp(h1, b1)
            line = (f"[{leg}] {dt} head vs base fast/r1 bit-identical={same} (cross-process A/A base {baa}, head {haa}), "
                    f"max|dloss| {d:.3g}; losses head {h1.get('loss_reprs')[:4]} base {b1.get('loss_reprs')[:4]}; "
                    f"grad_norm head {_gn(h1)} base {_gn(b1)}; final digest head {h1.get('final_digest')} base {b1.get('final_digest')}")
            if (baa and haa and not same) or d > DIVERGE:
                problems.append("HEAD != BASE " + line)
            else:
                notes.append(line)
        # engagement per step (ModuleList experts only)
        for pk, rk, p, r in _all_runs(head):
            if not _done(r):
                continue
            steps = r.get("counts_per_step") or []
            if not _modulelist(r):
                notes.append(f"[{leg}] head {pk}:{rk} experts {r.get('experts_class')}: ModuleList path n/a; "
                             f"COUNTS train total {r.get('counts_train_total')}")
                continue
            compact = [f"{s.get('full')}/{s.get('lean')}/{s.get('cheap')}/{s.get('table_full')}/{s.get('table_cheap')}"
                       for s in steps if s]
            line = f"[{leg}] head {pk}:{rk} COUNTS per step full/lean/cheap/table_full/table_cheap {compact}"
            if rk.startswith("fast"):
                late_full = [i + 1 for i, s in enumerate(steps) if i > 0 and (s or {}).get("full", 0) > 0]
                no_hit = [i + 1 for i, s in enumerate(steps) if i > 0 and ((s or {}).get("cheap", 0) + (s or {}).get("lean", 0)) == 0]
                first_full = (steps[0] or {}).get("full", 0) if steps else 0
                if late_full or no_hit or first_full <= 0:
                    problems.append(f"ENGAGEMENT {line}: first-step full {first_full}, full in later steps {late_full}, "
                                    f"steps without cheap/lean hits {no_hit}")
                else:
                    notes.append(line)
            else:
                hits = sum((s or {}).get("cheap", 0) + (s or {}).get("lean", 0) for s in steps)
                (problems.append(f"KILL SWITCH {line}: {hits} cheap/lean hits with UNSLOTH_MOE_FAST_READY=0")
                 if hits else notes.append(line))
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
    rows = ["| state | process | run | stage | experts | COUNTS/step full/lean/cheap/tf/tc | losses (first 3) | grad_norm | final digest | transformers |",
            "|" + "---|" * 10]
    for name in ("base", "head"):
        for pk, rk, p, r in _all_runs(obs.get(name)):
            cs = " ".join(f"{s.get('full')}/{s.get('lean')}/{s.get('cheap')}/{s.get('table_full')}/{s.get('table_cheap')}"
                          for s in r.get("counts_per_step") or [] if s) or "-"
            rows.append(f"| {name} | {pk} | {rk} | {r.get('stage')}{' ERR ' + str(r.get('error'))[:200] if r.get('error') else ''} | "
                        f"{r.get('experts_class')} | {cs} | {', '.join((r.get('loss_reprs') or [])[:3])} | "
                        f"{', '.join(str(x) for x in _gn(r))} | {r.get('final_digest')} | "
                        f"{(p.get('versions') or {}).get('transformers')} |")
        for pk, p in _procs(obs.get(name)).items():
            if p.get("error"):
                rows.append(f"| {name} | {pk} | - | process ERR {str(p.get('error'))[:300]} | | | | | | |")
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
        rows.append(f"{name} skipped: " + "; ".join(f"`{t}` {m[:120]}" for t, m in sorted((p.get("skipped") or {}).items())))
        rows.append("")
        for t, m in sorted((p.get("fail_msgs") or {}).items()):
            rows.append(f"{name} FAIL `{t}`: {m[:600]!r}")
        rows.append("")
    return "\n".join(rows)
