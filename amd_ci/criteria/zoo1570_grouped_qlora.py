#!/usr/bin/env python3
"""Criteria (unsloth-zoo PR 1570): grouped QLoRA for gpt-oss NF4 experts, on ROCm.

Regression mode. On HIP the PR's stacked Triton dequant is disabled, so the head
either takes bnb concat dequant + torch._grouped_mm LoRA, or declines to the
per-expert loop. Head is WORSE than base when any of:
  * a training arm crashes / produces non-finite loss or grads at the head
  * the head's forced loop arm (UNSLOTH_GPTOSS_GROUPED=0) does not reproduce the
    base's loop arm losses (rel 1e-4)
  * the head's default arm took the grouped path and its losses differ from the
    head loop arm by > 2e-2 rel, or its step-0 LoRA grads by > 0.05 rel (the
    PR's own test tolerance), or it declined and is not identical to the loop
  * a test that ran at the base fails at the head, or a test the PR adds fails
    at the head (added tests are part of the change, so their failure counts)

Gates (non-vacuity): each state imported unsloth_zoo from its own checkout; the
base's training arms both finished; the base actually ran the per-expert loop;
pytest ran (exit 0 / 1) and collected tests at both states.
"""

from __future__ import annotations

TITLE = "unsloth-zoo PR 1570: grouped QLoRA (gpt-oss NF4 experts) on gfx1151 ROCm, base vs head"
MODE = "regression"
# nvidia: the stacked Triton NF4 dequant the PR adds runs only on CUDA; unreachable here.
NEEDS = ["rocm", "gpu", "nvidia"]

_RAN = (0, 1)


def _arm(o: dict, arm: str) -> dict:
    return ((o or {}).get("train") or {}).get(arm) or {}


def _finished(a: dict) -> bool:
    return a.get("stage") == "done" and a.get("ok") is True and not a.get("error")


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out = []
    for name in ("base", "head"):
        o = obs.get(name) or {}
        for arm in ("default", "loop"):
            a = _arm(o, arm)
            f = str(a.get("unsloth_zoo_file") or "")
            ck = str(o.get("checkout") or "<none>")
            out.append((f"{name}/{arm} imported the state's unsloth_zoo", bool(f) and f.startswith(ck),
                        f or f"not imported (stage={a.get('stage')}, rc={a.get('rc')})"))
        p = o.get("pytest") or {}
        out.append((f"{name} pytest ran", p.get("rc") in _RAN,
                    f"rc={p.get('rc')} " + (str(p.get("stderr_tail") or "")[-200:] if p.get("rc") not in _RAN else "")))
        out.append((f"{name} pytest collected", (p.get("n_passed", 0) + p.get("n_failed", 0) + p.get("n_skipped", 0)) > 0,
                    f"passed {p.get('n_passed', 0)}, failed {p.get('n_failed', 0)}, skipped {p.get('n_skipped', 0)}"))
    b = obs.get("base") or {}
    for arm in ("default", "loop"):
        a = _arm(b, arm)
        out.append((f"base/{arm} training finished", _finished(a),
                    a.get("error") or f"stage={a.get('stage')} rc={a.get('rc')} losses={a.get('losses')}"))
        out.append((f"base/{arm} ran the per-expert loop", a.get("engaged") == "loop",
                    f"engaged={a.get('engaged')} counters={a.get('counters')}"))
    return out


def _fmt(v):
    if isinstance(v, list):
        return "[" + ", ".join(f"{x:.6g}" if isinstance(x, float) else str(x) for x in v) + "]"
    return str(v)


def table(obs: dict) -> str:
    rows = ["| state | arm | engaged | grouped calls (returned/attempted) | loop expert calls | gq.CALLS | losses | LoRA grad norms |",
            "|---|---|---|---|---|---|---|---|"]
    vers = None
    for name in ("base", "head", "merge"):
        o = obs.get(name)
        if not o:
            continue
        for arm in ("default", "loop"):
            a = _arm(o, arm)
            vers = vers or a.get("versions")
            c = a.get("counters") or {}
            rows.append(f"| {name} | {arm} | {a.get('engaged') or ('CRASH: ' + str(a.get('error') or a.get('rc')))} "
                        f"| {c.get('grouped_returned')}/{c.get('grouped_attempt')} | {c.get('loop_gate_up_calls')} "
                        f"| {a.get('gq_calls_delta', 'n/a (module absent)')} | {_fmt(a.get('losses'))} | {_fmt(a.get('grad_norms'))} |")
    rows += ["", "| state | default-vs-loop step-0 LoRA grads | grouped_mm supported | stacked dequant available |",
             "|---|---|---|---|"]
    for name in ("base", "head", "merge"):
        o = obs.get(name)
        if not o:
            continue
        d = o.get("grad_diff_default_vs_loop") or {}
        a = _arm(o, "default")
        rows.append(f"| {name} | max_rel={d.get('max_rel')} bit_identical={d.get('bit_identical')} n={d.get('n')} "
                    f"| {a.get('grouped_mm_supported')} | {a.get('stacked_dequant_available', 'n/a')} |")
    rows += ["", "| state | pytest passed | failed | skipped | failing tests |", "|---|---|---|---|---|"]
    for name in ("base", "head", "merge"):
        p = (obs.get(name) or {}).get("pytest")
        if not p:
            continue
        bad = ", ".join(f"`{t.split('::')[-1]}`" for t in p.get("failed", []) + p.get("errors", [])) or "none"
        rows.append(f"| {name} | {p.get('n_passed')} | {p.get('n_failed')} | {p.get('n_skipped')} | {bad} |")
    if vers:
        rows += ["", "Versions: " + ", ".join(f"{k}={v}" for k, v in vers.items())]
    return "\n".join(rows)


def _rel(a: float, b: float) -> float:
    return abs(a - b) / (abs(b) + 1e-12)


def head_is_worse(base: dict, head: dict) -> tuple[bool, str]:
    problems, notes = [], []
    hd, hl, bl = _arm(head, "default"), _arm(head, "loop"), _arm(base, "loop")
    for arm, a in (("default", hd), ("loop", hl)):
        if not _finished(a):
            problems.append(f"head/{arm} did not finish: {a.get('error') or ('rc=' + str(a.get('rc')) + ' stage=' + str(a.get('stage')))}")
    if _finished(hl) and _finished(bl):
        worst = max(_rel(x, y) for x, y in zip(hl["losses"], bl["losses"]))
        if worst > 1e-4:
            problems.append(f"head loop arm losses {hl['losses']} differ from base loop {bl['losses']} (rel {worst:.2e})")
        else:
            notes.append(f"head loop arm reproduces base loop losses (max rel {worst:.2e})")
    if _finished(hd) and _finished(hl):
        worst = max(_rel(x, y) for x, y in zip(hd["losses"], hl["losses"]))
        gd = head.get("grad_diff_default_vs_loop") or {}
        if hd.get("engaged") == "grouped":
            notes.append(f"head default arm ENGAGED the grouped path (gq.CALLS {hd.get('gq_calls_delta')}); "
                         f"loss max rel vs loop {worst:.2e}, step-0 LoRA grad max rel {gd.get('max_rel')}")
            if worst > 2e-2:
                problems.append(f"grouped losses differ from loop by rel {worst:.2e} > 2e-2")
            if gd.get("max_rel") is None or gd["max_rel"] > 0.05 + 1e-6:
                problems.append(f"grouped step-0 LoRA grads differ from loop: {gd}")
        elif hd.get("engaged") == "loop":
            notes.append("head default arm DECLINED the grouped path and ran the per-expert loop")
            if worst > 0 or not gd.get("bit_identical"):
                problems.append(f"declined but not identical to the loop arm (loss rel {worst:.2e}, grads {gd})")
        else:
            problems.append(f"head default arm ran a mixed path: {hd.get('counters')}")
    bp, hp = base.get("pytest") or {}, head.get("pytest") or {}
    b = set(bp.get("failed", [])) | set(bp.get("errors", []))
    h = set(hp.get("failed", [])) | set(hp.get("errors", []))
    new = sorted(h - b)
    if new:
        problems.append("failing at the head (not failing at the base): " + ", ".join(f"`{n.split('::')[-1]}`" for n in new))
    if b & h:
        notes.append("failing at BOTH states (pre-existing): " + ", ".join(f"`{n.split('::')[-1]}`" for n in sorted(b & h)))
    if problems:
        return True, "; ".join(problems) + (". Also: " + "; ".join(notes) if notes else "")
    return False, "; ".join(notes)
