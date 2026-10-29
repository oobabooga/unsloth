#!/usr/bin/env python3
"""Criteria (unsloth PR 13326): packed rows on sliding-window layers on gfx1151 ROCm, base vs head.

Regression mode. Head is WORSE than base when any of:
  * a head arm crashes
  * any head arm's per-step losses differ from the base's same arm beyond tolerance
  * head/sliding took the fast (varlen) route and its losses differ from head/sliding_ref
    (UNSLOTH_HF_PACKED_VARLEN=0, the masked sdpa path) beyond tolerance: a wrong varlen route
  * an arm that took the fast route at the base falls back at the head (lost route)
  * the opt-in flash impl is active at the head with UNSLOTH_FLASH_ATTENTION_IMPL unset
  * a test id fails / errors at the head that did not at the base (by id, not count)

Gates (non-vacuity): every arm imported its own checkout's unsloth; base arms finished; the
wrapper was installed and packed rows reached it (fast + fallback > 0) in the wrapper-on arms at
both states; pytest ran and collected tests at both states.
"""

from __future__ import annotations

TITLE = "unsloth PR 13326: packed sliding-window attention route on gfx1151 ROCm, base vs head"
MODE = "regression"
# nvidia: torch varlen / FA2 / FA4 / FA3 routes the PR enables live there; windows: no ROCm torch run here.
NEEDS = ["rocm", "gpu", "nvidia", "windows"]

ARMS = ("sliding", "sliding_ref", "full")
WRAPPED = ("sliding", "full")
RTOL = 2e-2


def _arm(o, arm):
    return ((o or {}).get("train") or {}).get(arm) or {}


def _done(a):
    return a.get("stage") == "done" and a.get("ok") is True and not a.get("error")


def _stats(a):
    return a.get("stats") or {}


def _seen(a):
    s = _stats(a)
    return (s.get("fast") or 0) + (s.get("fallback") or 0)


def _loss_diff(x, y):
    lx, ly = x.get("losses") or [], y.get("losses") or []
    if not lx or len(lx) != len(ly):
        return None, False
    rel = max(abs(a - b) / max(abs(b), 1e-6) for a, b in zip(lx, ly))
    return rel, lx == ly


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
                out.append((f"base/{arm} finished", _done(a),
                            (a.get("error") or f"stage={a.get('stage')} rc={a.get('rc')}")[:300]))
        for arm in WRAPPED:
            a = _arm(o, arm)
            if name == "head" and not _done(a):
                continue  # a head crash is a finding, judged in head_is_worse
            out.append((f"{name}/{arm} packed rows reached the installed wrapper",
                        bool(a.get("sdpa_wrapped")) and _seen(a) > 0,
                        f"sdpa_entry={a.get('sdpa_entry')} stats={a.get('stats')} collator={a.get('collator')}"))
        p = o.get("pytest") or {}
        out.append((f"{name} pytest ran", p.get("rc") in (0, 1),
                    f"rc={p.get('rc')} {str(p.get('stderr_tail') or '')[-200:]}"))
        out.append((f"{name} pytest collected",
                    (p.get("n_passed", 0) + p.get("n_failed", 0) + p.get("n_skipped", 0)) > 0,
                    f"passed {p.get('n_passed', 0)}, failed {p.get('n_failed', 0)}, skipped {p.get('n_skipped', 0)}"))
    return out


def table(obs):
    rows = ["| state | arm | stats (fast / fallback / mask_checks) | flash impl active | losses | max rel vs base | |",
            "|---|---|---|---|---|---|---|"]
    vers = kern = None
    for name in ("base", "head", "merge"):
        o = obs.get(name)
        if not o:
            continue
        for arm in ARMS:
            a = _arm(o, arm)
            vers = vers or a.get("versions")
            kern = kern or (a.get("kernels") if name == "head" else None)
            s = _stats(a)
            status = "" if _done(a) else f" CRASH {a.get('error') or a.get('rc')}"[:160]
            rel, same = _loss_diff(a, _arm(obs.get("base"), arm))
            fi = (a.get("flash_impl_state") or {}).get("active") if a.get("flash_impl_module") else "n/a"
            losses = ", ".join(f"{x:.4f}" for x in (a.get("losses") or []))
            rows.append(f"| {name} | {arm}{status} | {s.get('fast')} / {s.get('fallback')} / {s.get('mask_checks')} "
                        f"| {fi} | {losses} | {'n/a' if rel is None else f'{rel:.2e}'} | {'bit-identical' if same else ''} |")
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
    if kern:
        rows += ["", "Head varlen kernels on this build: " + ", ".join(f"{k}={v}" for k, v in kern.items())]
    if vers:
        rows += ["", "Versions: " + ", ".join(f"{k}={v}" for k, v in vers.items())]
    return "\n".join(rows)


def head_is_worse(base, head):
    problems, notes = [], []
    for arm in ARMS:
        a, b = _arm(head, arm), _arm(base, arm)
        if not _done(a):
            problems.append(f"head/{arm} did not finish: {(a.get('error') or str(a.get('rc')))[:300]}")
            continue
        rel, same = _loss_diff(a, b)
        if rel is None or rel > RTOL:
            problems.append(f"head/{arm} losses differ from base beyond {RTOL}: max rel {rel}")
        else:
            notes.append(f"head/{arm} losses {'bit-identical to' if same else f'within {rel:.2e} of'} base")
        hs, bs = _stats(a), _stats(b)
        if arm in WRAPPED:
            notes.append(f"{arm}: base fast/fallback {bs.get('fast')}/{bs.get('fallback')}, "
                         f"head {hs.get('fast')}/{hs.get('fallback')}")
            if (bs.get("fast") or 0) > 0 and not (hs.get("fast") or 0):
                problems.append(f"head/{arm} lost the fast route the base took")
        st = a.get("flash_impl_state") or {}
        if st.get("active"):
            problems.append(f"head/{arm}: flash impl ACTIVE ({st.get('active')}) with the env unset")
    hs, ref = _arm(head, "sliding"), _arm(head, "sliding_ref")
    if _done(hs) and _done(ref) and (_stats(hs).get("fast") or 0) > 0:
        rel, same = _loss_diff(hs, ref)
        if rel is None or rel > RTOL:
            problems.append(f"head/sliding varlen route disagrees with the masked sdpa reference: max rel {rel}")
        else:
            notes.append(f"head/sliding varlen route matches the masked reference (max rel {rel:.2e})")
    elif _done(hs):
        notes.append("head/sliding took no fast route on this build: falls back to the wrapped sdpa as at the base")
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
