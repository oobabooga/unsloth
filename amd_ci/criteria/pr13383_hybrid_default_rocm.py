#!/usr/bin/env python3
"""Criteria (unsloth PR 13383): gated-delta hybrid packing default on gfx1151 ROCm, base vs head.

PR 13383 turns Qwen3.5 / Qwen3-Next gated-delta hybrid packing ON by default, on CUDA only. On
ROCm the auto mode must stay OFF: the default run has to be the padded run the base trains.

Regression mode. Head is WORSE than base when any of:
  * head/default or head/off did not finish
  * head/default PACKED on this ROCm host: trainer args.packing / args.padding_free, the collator's
    padding_free, the varlen forward wrapper, a packed batch reaching the loss, or
    HYBRID_PACKING_STATS["patched"] > 0
  * head/off packed (=0 must always disable)
  * head/default or head/off per-step losses differ from the base's same arm beyond the A/A noise
    (bit-identical expected: both states train the padded path). The A/A is free: at the base all
    three arms run the same padded path, so their pairwise spread is the run-to-run noise. Zero
    spread means zero tolerance; otherwise 4x the spread, named in the verdict (on CUDA the fla
    Triton backward is not bit-deterministic run to run: 2.4e-4 abs between two padded base arms)
  * a test id fails / errors at the head that did not at the base (by id, not count)

head/on (=1, explicit opt-in) is recorded, never judged: it may pack or fail closed.

Gates (non-vacuity): every arm imported its own checkout's unsloth (a missing Qwen3.5 in the
installed transformers fails this gate with the reason, so the verdict is INCONCLUSIVE); base
default / off finished and trained 8 logged steps; the trainer was really asked to pack (the
probe saw the trainer's resolved flags and its batches); pytest ran and collected at both states.
"""

from __future__ import annotations

TITLE = "unsloth PR 13383: gated-delta hybrid packing stays off by default on gfx1151 ROCm, base vs head"
MODE = "regression"
# nvidia: the default-on path the PR ships lives there; windows: no ROCm torch run here.
NEEDS = ["rocm", "gpu", "nvidia", "windows"]

ARMS = ("default", "off", "on")
JUDGED = ("default", "off")
STEPS = 8


def _arm(o, arm):
    return ((o or {}).get("train") or {}).get(arm) or {}


def _done(a):
    return a.get("stage") == "done" and a.get("ok") is True and not a.get("error")


def packed_evidence(a) -> list[str]:
    """Every observed sign that this arm trained packed rows."""
    ev = []
    if a.get("args_packing"):
        ev.append("args.packing=True")
    if a.get("args_padding_free"):
        ev.append("args.padding_free=True")
    if a.get("collator_padding_free"):
        ev.append("collator.padding_free=True")
    vw = a.get("varlen_wrapped_after_init") or {}
    if (vw.get("forward_wrapped_modules") or 0) > 0 or (vw.get("mixers_wrapped") or 0) > 0:
        ev.append(f"varlen wrapper {vw}")
    b = a.get("batches") or {}
    if (b.get("packed") or 0) > 0:
        ev.append(f"{b.get('packed')}/{b.get('n')} packed batches")
    st = a.get("stats") or {}
    if (st.get("patched") or 0) > 0:
        ev.append(f"HYBRID_PACKING_STATS.patched={st.get('patched')}")
    return ev


def _losses_same(x, y):
    lx, ly = x.get("losses") or [], y.get("losses") or []
    if not lx or len(lx) != len(ly):
        return None, False
    return max(abs(a - b) for a, b in zip(lx, ly)), lx == ly


def aa_noise(base):
    """Max abs per-step loss spread between base arms that trained the padded path."""
    arms = [_arm(base, a) for a in ARMS]
    arms = [a for a in arms if _done(a) and not packed_evidence(a) and len(a.get("losses") or []) == STEPS]
    if len(arms) < 2:
        return None
    return max(max(abs(x - y) for x, y in zip(a["losses"], b["losses"]))
               for i, a in enumerate(arms) for b in arms[i + 1:])


def gates(obs):
    out = []
    for name in ("base", "head"):
        o = obs.get(name) or {}
        ck = str(o.get("checkout") or "<none>")
        for arm in ARMS:
            a = _arm(o, arm)
            f = str(a.get("unsloth_file") or "")
            out.append((f"{name}/{arm} imported the state's unsloth", bool(f) and f.startswith(ck),
                        f or (a.get("skip") or f"stage={a.get('stage')} rc={a.get('rc')} "
                              f"{a.get('error') or ''}")[:300]))
        for arm in JUDGED:
            a = _arm(o, arm)
            if name == "base":
                out.append((f"base/{arm} finished {STEPS} steps",
                            _done(a) and len(a.get("losses") or []) == STEPS,
                            (a.get("error") or f"stage={a.get('stage')} rc={a.get('rc')} "
                             f"losses={len(a.get('losses') or [])}")[:300]))
            if name == "head" and not _done(a):
                continue  # a head crash is a finding, judged in head_is_worse
            b = a.get("batches") or {}
            out.append((f"{name}/{arm} trainer observed (packing=True requested, batches seen)",
                        a.get("args_packing") is not None and (b.get("n") or 0) > 0,
                        f"args_packing={a.get('args_packing')} batches={b} collator={a.get('collator')}"))
        if name == "base":
            aa = aa_noise(o)
            out.append(("base A/A: >= 2 padded base arms to measure loss noise", aa is not None,
                        f"max abs spread {aa}"))
        p = o.get("pytest") or {}
        out.append((f"{name} pytest ran", p.get("rc") in (0, 1),
                    f"rc={p.get('rc')} {str(p.get('stderr_tail') or '')[-200:]}"))
        out.append((f"{name} pytest collected",
                    (p.get("n_passed", 0) + p.get("n_failed", 0) + p.get("n_skipped", 0)) > 0,
                    f"passed {p.get('n_passed', 0)}, failed {p.get('n_failed', 0)}, skipped {p.get('n_skipped', 0)}"))
    return out


def table(obs):
    rows = ["| state | arm | env | packed? | HYBRID_PACKING_STATS | rejection / ignored | losses | vs base | rc |",
            "|---|---|---|---|---|---|---|---|---|"]
    vers = None
    for name in ("base", "head", "merge"):
        o = obs.get(name)
        if not o:
            continue
        for arm in ARMS:
            a = _arm(o, arm)
            vers = vers or a.get("versions")
            status = "" if _done(a) else f" CRASH {a.get('skip') or a.get('error') or a.get('rc')}"[:160]
            ev = packed_evidence(a)
            st = a.get("stats")
            st_s = "absent" if st is None else ", ".join(f"{k}={v}" for k, v in st.items())
            why = "; ".join(dict.fromkeys((a.get("hybrid_rejections") or []) + (a.get("packing_ignored") or [])))[:300] or "-"
            why = why.replace("|", "/")
            diff, same = _losses_same(a, _arm(obs.get("base"), arm))
            losses = ", ".join(f"{x:.4f}" for x in (a.get("losses") or []))
            vs = "n/a" if diff is None else ("bit-identical" if same else f"max abs {diff:.2e}")
            rows.append(f"| {name} | {arm}{status} | {a.get('env_hybrid_packing')} | "
                        f"{'YES: ' + ', '.join(ev) if ev else 'no'} | {st_s} | {why} | {losses} | {vs} | {a.get('rc')} |")
    rows += ["", "| state | pytest passed | failed | skipped | failing tests |", "|---|---|---|---|---|"]
    for name in ("base", "head", "merge"):
        p = (obs.get(name) or {}).get("pytest")
        if not p:
            continue
        bad = ", ".join(f"`{t}`" for t in p.get("failed", []) + p.get("errors", [])) or "none"
        absent = ", ".join(p.get("absent_at_this_state") or [])
        rows.append(f"| {name}{f' (absent: {absent})' if absent else ''} | {p.get('n_passed', 0)} "
                    f"| {p.get('n_failed', 0)} | {p.get('n_skipped', 0)} | {bad} |")
    hp = (obs.get("head") or {}).get("pytest") or {}
    if hp.get("skipped"):
        reasons = {}
        for t, m in hp["skipped"].items():
            reasons.setdefault(m, []).append(t.split("::")[-1])
        rows += ["", "Head skips: " + "; ".join(f"{len(v)} x {k!r}" for k, v in reasons.items())]
    if vers:
        rows += ["", "Versions: " + ", ".join(f"{k}={v}" for k, v in vers.items())]
    return "\n".join(rows)


def head_is_worse(base, head):
    problems, notes = [], []
    aa = aa_noise(base) or 0.0
    tol = 0.0 if aa == 0 else 4 * aa
    notes.append("base A/A loss spread 0: tolerance 0 (bit-identical required)" if aa == 0 else
                 f"base A/A loss spread {aa:.2e} (padded arms not bit-reproducible on this build): tolerance {tol:.2e}")
    for arm in JUDGED:
        a, b = _arm(head, arm), _arm(base, arm)
        if not _done(a):
            problems.append(f"head/{arm} did not finish: {(a.get('error') or str(a.get('rc')))[:300]}")
            continue
        ev = packed_evidence(a)
        if ev:
            problems.append(f"head/{arm} PACKED the hybrid on this host ({', '.join(ev)})")
        else:
            notes.append(f"head/{arm} stayed on the padded path")
        diff, same = _losses_same(a, b)
        if diff is None or diff > tol:
            problems.append(f"head/{arm} losses differ from base beyond {tol:.2e}: max abs {diff}")
        else:
            notes.append(f"head/{arm} losses {'bit-identical to' if same else f'within A/A noise ({diff:.2e}) of'} base")
    d = _arm(head, "default")
    rej = list(dict.fromkeys(d.get("hybrid_rejections") or []))
    if rej:
        notes.append("head/default rejection: " + "; ".join(rej)[:300])
    if _done(d) and not packed_evidence(d):
        cuda_gate = any("default-on only on CUDA" in r for r in rej)
        notes.append("the ROCm refusal (`default-on only on CUDA`) fired" if cuda_gate else
                     "head/default stayed padded WITHOUT the `default-on only on CUDA` refusal firing "
                     "(auto mode declined earlier, e.g. mixers not auto-eligible)")
    on = _arm(head, "on")
    if _done(on):
        ev = packed_evidence(on)
        notes.append(f"head/on (=1, recorded only): {'packed (' + ', '.join(ev) + ')' if ev else 'fail-closed, padded'}"
                     + (f"; {'; '.join(dict.fromkeys(on.get('hybrid_rejections') or []))[:200]}" if on.get("hybrid_rejections") else ""))
    else:
        notes.append(f"head/on (=1, recorded only) did not finish: {(on.get('error') or str(on.get('rc')))[:200]}")
    bp, hp = base.get("pytest") or {}, head.get("pytest") or {}
    b = set(bp.get("failed", [])) | set(bp.get("errors", []))
    h = set(hp.get("failed", [])) | set(hp.get("errors", []))
    new = sorted(h - b)
    if new:
        problems.append("newly failing at the head: " + ", ".join(f"`{t}`" for t in new))
    else:
        notes.append(f"head tests: {hp.get('n_passed', 0)} passed, {hp.get('n_skipped', 0)} skipped, "
                     f"{len(h)} failed; no new failing ids")
    if h & b:
        notes.append("failing at BOTH states (pre-existing): " + ", ".join(f"`{t}`" for t in sorted(h & b)))
    if problems:
        return True, "; ".join(problems) + (". Also: " + "; ".join(notes) if notes else "")
    return False, "; ".join(notes)
