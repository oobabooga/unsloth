#!/usr/bin/env python3
"""Criteria (unsloth PR 12739): the bitsandbytes NF4 Linear4bit.forward override is inert on ROCm.

Regression mode. Head is WORSE than base when any of:
  * a head arm crashes
  * head, default arm: the override installed AND engaged (any call into its Triton _linear)
  * head, forced arm (UNSLOTH_BNB_NF4_LINEAR=1): any engagement (the HIP guard in _eligible must
    hand every call back to bitsandbytes)
  * any arm's model loss / logits / LoRA grads or standalone-layer output / dX hashes differ
    from the base's same arm (bit identity)
  * a test fails at the head (the PR's own test file is absent at the base, so every head
    failure counts)

Gates (non-vacuity): every arm imported its own checkout's unsloth; base arms finished;
Linear4bit.forward really ran in the probes (bnb_forward_total > 0 at every state, so a 0
engagement count is not a probe that never reached the layer); the head forced arm really
routed through the installed override (override_forward > 0), so the HIP guard was exercised;
pytest ran and collected tests at the head.
"""

from __future__ import annotations

TITLE = "unsloth PR 12739: bitsandbytes NF4 override stays off on gfx1151 ROCm, base vs head"
MODE = "regression"
# nvidia: where the override is meant to engage (sm100/sm120 or bnb < 0.50); unreachable here.
NEEDS = ["rocm", "gpu", "nvidia"]

ARMS = ("default", "forced")


def _arm(o, arm):
    return ((o or {}).get("train") or {}).get(arm) or {}


def _done(a):
    return a.get("stage") == "done" and a.get("ok") is True and not a.get("error")


def _total(a, key):
    return ((a.get("counts") or {}).get(key)) or 0


def gates(obs):
    out = []
    for name in ("base", "head"):
        o = obs.get(name) or {}
        ck = str(o.get("checkout") or "<none>")
        for arm in ARMS:
            a = _arm(o, arm)
            f = str(a.get("unsloth_file") or "")
            out.append((f"{name}/{arm} imported the state's unsloth", bool(f) and f.startswith(ck),
                        f or f"stage={a.get('stage')} rc={a.get('rc')} {a.get('error') or ''}"[:300]))
            if name == "base":
                out.append((f"base/{arm} finished", _done(a), a.get("error") or f"stage={a.get('stage')} rc={a.get('rc')}"))
            out.append((f"{name}/{arm} Linear4bit.forward ran", _total(a, "bnb_forward_total") > 0,
                        f"counts={a.get('counts')} model={a.get('model')} n_linear4bit={a.get('n_linear4bit')}"))
    hf = _arm(obs.get("head"), "forced")
    out.append(("head/forced routed through the installed override", _total(hf, "override_forward") > 0
                and bool(hf.get("installed_mark_after_load")),
                f"installed_mark_after_load={hf.get('installed_mark_after_load')} counts={hf.get('counts')}"))
    p = (obs.get("head") or {}).get("pytest") or {}
    out.append(("head pytest ran", p.get("rc") in (0, 1), f"rc={p.get('rc')} {str(p.get('stderr_tail') or '')[-200:]}"))
    out.append(("head pytest collected", (p.get("n_passed", 0) + p.get("n_failed", 0) + p.get("n_skipped", 0)) > 0,
                f"passed {p.get('n_passed', 0)}, failed {p.get('n_failed', 0)}, skipped {p.get('n_skipped', 0)}"))
    return out


_HASHES = (("model_probe", "loss"), ("model_probe", "logits_sha"), ("model_probe", "grads_sha"),
           ("layer_probe", "y_sha"), ("layer_probe", "dx_sha"))


def table(obs):
    rows = ["| state | arm | module | wanted | installed (mark after load) | override _forward calls | Triton _linear engagements | Linear4bit.forward calls | loss | logits | LoRA grads | layer y | layer dX |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    vers = None
    for name in ("base", "head", "merge"):
        o = obs.get(name)
        if not o:
            continue
        for arm in ARMS:
            a = _arm(o, arm)
            vers = vers or a.get("versions")
            c = a.get("counts") or {}
            h = [str((a.get(s) or {}).get(k)) for s, k in _HASHES]
            status = "" if _done(a) else f" CRASH {a.get('error') or a.get('rc')}"[:120]
            rows.append(f"| {name} | {arm}{status} | {a.get('has_override_module')} | {a.get('wanted', 'n/a')} "
                        f"| {a.get('installed_mark_after_load')} | {c.get('override_forward')} | {c.get('engaged_linear')} "
                        f"| {c.get('bnb_forward_total')} | " + " | ".join(h) + " |")
    rows += ["", "| state | pytest passed | failed | skipped | failing tests |", "|---|---|---|---|---|"]
    for name in ("base", "head", "merge"):
        p = (obs.get(name) or {}).get("pytest")
        if not p:
            continue
        bad = ", ".join(f"`{t.split('::')[-1]}`" for t in p.get("failed", []) + p.get("errors", [])) or "none"
        note = " (file absent at this state)" if p.get("absent_at_this_state") and not p.get("selected") else ""
        rows.append(f"| {name}{note} | {p.get('n_passed', 0)} | {p.get('n_failed', 0)} | {p.get('n_skipped', 0)} | {bad} |")
    hp = (obs.get("head") or {}).get("pytest") or {}
    if hp.get("skipped"):
        reasons = {}
        for t, m in hp["skipped"].items():
            reasons.setdefault(m, []).append(t.split("::")[-1])
        rows += ["", "Head skips: " + "; ".join(f"{len(v)} x {k!r}" for k, v in reasons.items())]
    m = (_arm(obs.get("base"), "default")).get("model")
    if vers:
        rows += ["", f"Model: {m}. Versions: " + ", ".join(f"{k}={v}" for k, v in vers.items())]
    return "\n".join(rows)


def head_is_worse(base, head):
    problems, notes = [], []
    for arm in ARMS:
        a, b = _arm(head, arm), _arm(base, arm)
        if not _done(a):
            problems.append(f"head/{arm} did not finish: {a.get('error') or a.get('rc')}")
            continue
        eng = _total(a, "engaged_linear")
        if eng:
            problems.append(f"head/{arm}: override ENGAGED {eng} times on ROCm")
        else:
            notes.append(f"head/{arm}: 0 engagements (installed={a.get('installed_mark_after_load')}, "
                         f"override _forward calls={_total(a, 'override_forward')}, Linear4bit.forward calls={_total(a, 'bnb_forward_total')})")
        diff = [f"{s}.{k}" for s, k in _HASHES if (a.get(s) or {}).get(k) != (b.get(s) or {}).get(k)]
        if diff:
            problems.append(f"head/{arm} not bit-identical to base/{arm}: {diff}")
        else:
            notes.append(f"head/{arm} bit-identical to base (loss {(a.get('model_probe') or {}).get('loss')})")
    hp = head.get("pytest") or {}
    bad = sorted(set(hp.get("failed", [])) | set(hp.get("errors", [])))
    if bad:
        problems.append("failing at the head: " + ", ".join(f"`{t.split('::')[-1]}`" for t in bad))
    else:
        notes.append(f"head tests: {hp.get('n_passed', 0)} passed, {hp.get('n_skipped', 0)} skipped, 0 failed")
    if problems:
        return True, "; ".join(problems) + (". Also: " + "; ".join(notes) if notes else "")
    return False, "; ".join(notes)
