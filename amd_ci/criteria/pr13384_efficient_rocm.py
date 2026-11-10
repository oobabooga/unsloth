#!/usr/bin/env python3
"""Criteria (unsloth PR 13384): packed Gemma-4 global layers (head_dim 512) and the
aten._efficient_attention varlen route, base vs head.

The route is gated off on ROCm (`torch.version.hip`), so on gfx1151 the head must behave exactly
as the base. Regression mode. Head is WORSE than base when any of:
  * a head arm crashes (a finding, not a gate: a gate failure would hide it as INCONCLUSIVE)
  * on ROCm, any head arm counts efficient > 0 (the ROCm gate leaked)
  * on ROCm, head/default per-step losses are not bit-identical to base/default
    (off ROCm the route is expected to move losses: head/default must instead agree with
    head/ref, the masked SDPA path, within RTOL)
  * a test id fails / errors at the head that did not at the base (by id, not count)

Gates (non-vacuity): every arm at both states imported its own checkout's unsloth; base arms
finished; Gemma 4 is available in the installed transformers (else INCONCLUSIVE with the reason);
the tiny model really has a 512 global layer; packed rows reached the wrapper (fast + fallback
> 0) in base/default and head/default; pytest ran and collected at both states.
"""

from __future__ import annotations

TITLE = "unsloth PR 13384: packed Gemma-4 wide-head efficient varlen route, gated off on gfx1151 ROCm, base vs head"
MODE = "regression"
# nvidia: the efficient route itself only runs there; windows: no ROCm torch measured here.
NEEDS = ["rocm", "gpu", "nvidia", "windows"]

ARMS = ("default", "efficient_off", "ref")
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


def _is_rocm(o):
    for arm in ARMS:
        v = _arm(o, arm).get("versions") or {}
        if v:
            return bool(v.get("hip"))
    return None


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
            if a.get("stage") == "unsupported":
                out.append((f"{name}/{arm} transformers has Gemma 4", False, str(a.get("unsupported"))[:300]))
                continue
            f = str(a.get("unsloth_file") or "")
            out.append((f"{name}/{arm} imported the state's unsloth", bool(f) and f.startswith(ck),
                        f or f"stage={a.get('stage')} rc={a.get('rc')} {a.get('error') or ''}"[:300]))
            if name == "base":
                out.append((f"base/{arm} finished", _done(a),
                            (a.get("error") or f"stage={a.get('stage')} rc={a.get('rc')}")[:300]))
        a = _arm(o, "default")
        if name == "head" and not _done(a):
            continue  # a head crash is a finding, judged in head_is_worse
        if _done(a):
            dims = a.get("attn_head_dims") or []
            out.append((f"{name}/default model has a head_dim 512 global layer", 512 in dims,
                        f"attn_head_dims={dims} config={a.get('config')}"))
            out.append((f"{name}/default packed rows reached the installed wrapper",
                        bool(a.get("sdpa_wrapped")) and _seen(a) > 0,
                        f"sdpa_entry={a.get('sdpa_entry')} stats={a.get('stats')} collator={a.get('collator')}"))
    for name in ("base", "head"):
        p = (obs.get(name) or {}).get("pytest") or {}
        out.append((f"{name} pytest ran", p.get("rc") in (0, 1),
                    f"rc={p.get('rc')} {str(p.get('stderr_tail') or '')[-200:]}"))
        out.append((f"{name} pytest collected",
                    (p.get("n_passed", 0) + p.get("n_failed", 0) + p.get("n_skipped", 0)) > 0,
                    f"passed {p.get('n_passed', 0)}, failed {p.get('n_failed', 0)}, skipped {p.get('n_skipped', 0)}"))
    return out


def table(obs):
    rows = ["| state | arm | stats (fast / fallback / efficient / mask_checks) | losses | max rel vs base | |",
            "|---|---|---|---|---|---|"]
    vers = None
    for name in ("base", "head", "merge"):
        o = obs.get(name)
        if not o:
            continue
        for arm in ARMS:
            a = _arm(o, arm)
            vers = vers or a.get("versions")
            s = _stats(a)
            status = "" if _done(a) else f" CRASH {a.get('unsupported') or a.get('error') or a.get('rc')}"[:160]
            rel, same = _loss_diff(a, _arm(obs.get("base"), arm))
            losses = ", ".join(f"{x:.4f}" for x in (a.get("losses") or []))
            rows.append(f"| {name} | {arm}{status} | {s.get('fast')} / {s.get('fallback')} / {s.get('efficient', '-')} "
                        f"/ {s.get('mask_checks')} | {losses} | {'n/a' if rel is None else f'{rel:.2e}'} "
                        f"| {'bit-identical' if same else ''} |")
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
    h = obs.get("head") or {}
    rows += ["", f"UNSLOTH_COMPILE_DISABLE=1 for every arm: {h.get('compile_disable')}; "
                 f"unsloth_zoo override: {h.get('zoo_override') or 'none (installed)'}"]
    if vers:
        rows += ["", "Versions: " + ", ".join(f"{k}={v}" for k, v in vers.items())]
    return "\n".join(rows)


def head_is_worse(base, head):
    problems, notes = [], []
    rocm = _is_rocm(head)
    for arm in ARMS:
        a, b = _arm(head, arm), _arm(base, arm)
        if not _done(a):
            problems.append(f"head/{arm} did not finish: {(a.get('error') or str(a.get('rc')))[:300]}")
            continue
        hs, bs = _stats(a), _stats(b)
        eff = hs.get("efficient") or 0
        notes.append(f"{arm}: base fast/fallback {bs.get('fast')}/{bs.get('fallback')}, "
                     f"head fast/fallback/efficient {hs.get('fast')}/{hs.get('fallback')}/{eff}")
        if rocm and eff > 0:
            problems.append(f"head/{arm}: efficient={eff} on ROCm, the torch.version.hip gate leaked")
        if rocm and (bs.get("fast") or 0) > 0 and not (hs.get("fast") or 0):
            problems.append(f"head/{arm} lost the fast route the base took")
        rel, same = _loss_diff(a, b)
        if rocm and arm == "default":
            if not same:
                problems.append(f"head/default losses not bit-identical to base on ROCm (max rel {rel})")
            else:
                notes.append("head/default losses bit-identical to base")
        elif rel is None:
            problems.append(f"head/{arm} losses not comparable with base")
        else:
            notes.append(f"head/{arm} losses {'bit-identical to' if same else f'within {rel:.2e} of'} base")
            if rel > RTOL and (rocm or arm != "default"):
                problems.append(f"head/{arm} losses differ from base beyond {RTOL}: max rel {rel}")
    if not rocm:
        d, ref = _arm(head, "default"), _arm(head, "ref")
        if _done(d) and _done(ref):
            rel, _ = _loss_diff(d, ref)
            if rel is None or rel > RTOL:
                problems.append(f"head/default (efficient {_stats(d).get('efficient')}) disagrees with the masked "
                                f"sdpa reference: max rel {rel}")
            else:
                notes.append(f"off ROCm: head/default (efficient {_stats(d).get('efficient')}) matches head/ref "
                             f"within {rel:.2e}")
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
