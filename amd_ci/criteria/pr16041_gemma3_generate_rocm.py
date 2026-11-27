#!/usr/bin/env python3
"""Criteria (unsloth PR 16041): Gemma 3 generate() off flex_attention, gfx1151 ROCm, base vs head.

PR 16041 wraps unsloth_base_fast_generate in _NonFlexGenerate: when Unsloth put the Gemma 3 decoder on
flex_attention, the decoder attention configs read "eager" for that one generate() and are restored on exit.

First question: does the base pick flex_attention for gemma-3-270m-it on this host? Recorded from
configs_after_load. If it does not, the PR must be a no-op here: _NonFlexGenerate never active, attn impl
during generate unchanged, tokens identical, timing within noise.

Regression mode. Head is WORSE than base when any of:
  * a head rep did not finish (load, generate, LoRA SFT)
  * a head config does not read its after-load value after generate or after training
  * the attn impl read by the attention modules during TRAINING differs from the base's
  * head ms/token median exceeds the base's by more than max(5%, 3x the base A/A spread)
  * head losses differ from base beyond max(1e-3 relative, 4x the larger A/A loss spread)
  * base did not pick flex, yet the head changed anything (engaged, attn impl during generate, tokens)
  * a test id fails at the head (the base copy of the PR's test is expected to fail collection)

Gates: each rep imported its own checkout's unsloth; base reps finished; 128 tokens generated with 3
timed repeats; the attention pre-hook saw calls during generate and training (the probe observed the path);
when base picked flex, the head's _NonFlexGenerate engaged (else the new path never ran: VOID-like).
"""

from __future__ import annotations

import statistics

TITLE = "unsloth PR 16041: Gemma 3 generate off flex_attention on gfx1151 ROCm, base vs head"
MODE = "regression"
# nvidia: the sm80+ flex decode slowdown the PR targets lives there; windows: no ROCm torch run here.
NEEDS = ["rocm", "gpu", "exclusive_gpu", "nvidia", "windows"]
REPS = ("a", "b")
NEW_TOKENS = 128


def _runs(o):
    return [((o or {}).get("runs") or {}).get(r) or {} for r in REPS]


def _done(r):
    return r.get("stage") == "done" and not r.get("error")


def _picks_flex(r):
    return any(v.get("_attn_implementation") == "flex_attention" for v in (r.get("configs_after_load") or {}).values())


def _impls(snapshot):
    return {k: v.get("_attn_implementation") for k, v in (snapshot or {}).items()}


def _med(o):
    xs = [r.get("ms_per_token_median") for r in _runs(o) if r.get("ms_per_token_median") is not None]
    return statistics.median(xs) if xs else None


def _aa_timing(o):
    xs = [r.get("ms_per_token_median") for r in _runs(o) if r.get("ms_per_token_median") is not None]
    within = [max(r["ms_per_token"]) - min(r["ms_per_token"]) for r in _runs(o) if r.get("ms_per_token")]
    across = (max(xs) - min(xs)) if len(xs) >= 2 else None
    return across, (max(within) if within else None)


def _aa_loss(o):
    ls = [r.get("losses") or [] for r in _runs(o)]
    if len(ls) < 2 or not ls[0] or len(ls[0]) != len(ls[1]):
        return None
    return max(abs(x - y) for x, y in zip(ls[0], ls[1]))


def _first_div(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def gates(obs):
    out = []
    base_flex = any(_picks_flex(r) for r in _runs(obs.get("base")))
    for name in ("base", "head"):
        o = obs.get(name) or {}
        ck = str(o.get("checkout") or "<none>")
        for rep, r in zip(REPS, _runs(o)):
            f = str(r.get("unsloth_file") or "")
            out.append((f"{name}/{rep} imported the state's unsloth", bool(f) and f.startswith(ck),
                        f or f"stage={r.get('stage')} rc={r.get('rc')} {str(r.get('error') or '')[:200]}"))
            if name == "base":
                out.append((f"base/{rep} finished (load, generate, SFT)", _done(r),
                            str(r.get("error") or f"stage={r.get('stage')} rc={r.get('rc')}")[:300]))
            if not _done(r) and name == "head":
                continue  # a head crash is a finding, judged in head_is_worse
            out.append((f"{name}/{rep} generated {NEW_TOKENS} tokens x 3 timed repeats",
                        r.get("new_tokens") == NEW_TOKENS and len(r.get("ms_per_token") or []) == 3,
                        f"new_tokens={r.get('new_tokens')} ms/token={r.get('ms_per_token')}"))
            out.append((f"{name}/{rep} attention pre-hook observed generate and training",
                        sum((r.get("attn_impl_seen_during_generate") or {}).values()) > 0
                        and sum((r.get("attn_impl_seen_during_train") or {}).values()) > 0,
                        f"generate={r.get('attn_impl_seen_during_generate')} train={r.get('attn_impl_seen_during_train')}"))
        if name == "head" and base_flex:
            eng = [r.get("nonflex_generate_engaged") for r in _runs(o)]
            out.append(("head: base picked flex, so _NonFlexGenerate must have engaged",
                        all(isinstance(e, dict) and e.get("active", 0) > 0 for e in eng), str(eng)))
        if name == "head":
            p = o.get("pytest") or {}
            out.append(("head pytest collected the PR's test",
                        (p.get("n_passed", 0) + p.get("n_failed", 0) + p.get("n_skipped", 0)) > 0,
                        f"rc={p.get('rc')} passed {p.get('n_passed', 0)} failed {p.get('n_failed', 0)}"))
    return out


def table(obs):
    rows = ["| state | rep | attn after load | engaged | attn seen in generate | dispatch (8-token gen) | "
            "attn after generate | warm-up ms/tok | ms/tok (3) | median | tokens vs base/a | attn seen in train | "
            "losses | attn after train |",
            "|" + "---|" * 14]
    base_ids = (_runs(obs.get("base"))[0] or {}).get("token_ids") or []
    vers = None
    for name in ("base", "head", "merge"):
        o = obs.get(name)
        if not o:
            continue
        for rep, r in zip(REPS, _runs(o)):
            vers = vers or r.get("versions")
            crash = "" if _done(r) else f" CRASH {str(r.get('error') or r.get('rc'))[:150]}"
            ids = r.get("token_ids") or []
            d = _first_div(ids, base_ids) if ids and base_ids else "n/a"
            tok = "equal" if d is None else f"diverge at {d}"
            losses = ", ".join(f"{x:.4f}" for x in (r.get("losses") or []))
            rows.append(
                f"| {name} | {rep}{crash} | {_impls(r.get('configs_after_load'))} | {r.get('nonflex_generate_engaged')} "
                f"| {r.get('attn_impl_seen_during_generate')} | {r.get('dispatch_during_generate') or r.get('dispatch_error')} "
                f"| {_impls(r.get('configs_after_generate'))} | {r.get('warmup_ms_per_token')} | {r.get('ms_per_token')} "
                f"| {r.get('ms_per_token_median')} | {tok} | {r.get('attn_impl_seen_during_train')} | {losses} "
                f"| {_impls(r.get('configs_after_train'))} |".replace("\n", " "))
    rows += ["", "| state | timing median of rep medians (ms/token) | A/A across reps | max within-rep spread | "
             "loss A/A |", "|---|---|---|---|---|"]
    for name in ("base", "head"):
        o = obs.get(name)
        if o:
            ac, wi = _aa_timing(o)
            rows.append(f"| {name} | {_med(o)} | {ac} | {wi} | {_aa_loss(o)} |")
    rows += ["", "| state | pytest passed | failed | skipped | failing tests | note |", "|---|---|---|---|---|---|"]
    for name in ("base", "head", "merge"):
        p = (obs.get(name) or {}).get("pytest")
        if not p:
            continue
        bad = ", ".join(f"`{t}`" for t in p.get("failed", []) + p.get("errors", [])) or "none"
        note = "head's test copied in" if p.get("copied_from_head") else (p.get("note") or "")
        rows.append(f"| {name} | {p.get('n_passed', 0)} | {p.get('n_failed', 0)} | {p.get('n_skipped', 0)} | {bad} | {note} |")
    if vers:
        rows += ["", "Versions: " + ", ".join(f"{k}={v}" for k, v in vers.items())]
    return "\n".join(rows)


def head_is_worse(base, head):
    problems, notes = [], []
    br, hr = _runs(base), _runs(head)
    base_flex = any(_picks_flex(r) for r in br)
    notes.append(f"base picks flex_attention after from_pretrained on this host: {base_flex} "
                 f"({_impls(br[0].get('configs_after_load'))})")
    for rep, r in zip(REPS, hr):
        if not _done(r):
            problems.append(f"head/{rep} did not finish: {str(r.get('error') or r.get('rc'))[:300]}")
    if problems:
        return True, "; ".join(problems)
    for rep, r in zip(REPS, hr):
        after_load = _impls(r.get("configs_after_load"))
        for k in ("configs_after_generate", "configs_after_dispatch_generate", "configs_after_train"):
            if _impls(r.get(k)) != after_load:
                problems.append(f"head/{rep} {k} {_impls(r.get(k))} != after load {after_load}")
    bt = set(br[0].get("attn_impl_seen_during_train") or {})
    for rep, r in zip(REPS, hr):
        ht = set(r.get("attn_impl_seen_during_train") or {})
        if ht != bt:
            problems.append(f"head/{rep} attn impl during training {sorted(ht)} != base {sorted(bt)}")
    # Timing.
    bm, hm = _med(base), _med(head)
    b_ac, b_wi = _aa_timing(base)
    noise = max(x for x in (b_ac, b_wi, 0.0) if x is not None)
    thr = max(0.05 * bm, 3 * noise)
    delta = hm - bm
    notes.append(f"ms/token base {bm:.3f} head {hm:.3f} (delta {delta:+.3f}, {100 * delta / bm:+.1f}%); "
                 f"base A/A across reps {b_ac}, within rep {b_wi}; threshold +-{thr:.3f}")
    if delta > thr:
        problems.append(f"head decode slower than base beyond noise: {bm:.3f} -> {hm:.3f} ms/token")
    elif delta < -thr:
        notes.append("head decode FASTER than base beyond noise")
    else:
        notes.append("decode timing within noise")
    # Tokens.
    b_ids = br[0].get("token_ids") or []
    divs = [_first_div(r.get("token_ids") or [], b_ids) for r in hr]
    aa_div = _first_div(br[1].get("token_ids") or [], b_ids)
    notes.append(f"tokens: base A/A {'equal' if aa_div is None else f'diverge at {aa_div}'}; head vs base "
                 + ", ".join("equal" if d is None else f"first divergence at token {d}" for d in divs))
    # Losses.
    aa = max(x for x in (_aa_loss(base), _aa_loss(head), 0.0) if x is not None)
    bl = br[0].get("losses") or []
    for rep, r in zip(REPS, hr):
        hl = r.get("losses") or []
        if len(hl) != len(bl) or not bl:
            problems.append(f"head/{rep} losses {hl} vs base {bl}: length mismatch")
            continue
        diff = max(abs(x - y) for x, y in zip(hl, bl))
        tol = max(1e-3 * max(abs(x) for x in bl), 4 * aa)
        if diff > tol:
            problems.append(f"head/{rep} losses differ from base by {diff:.3e} > {tol:.3e}")
        else:
            notes.append(f"head/{rep} losses {'bit-identical to' if diff == 0 else f'within {tol:.1e} ({diff:.1e}) of'} base")
    # No-op requirement when base never picked flex.
    if not base_flex:
        for rep, r in zip(REPS, hr):
            eng = r.get("nonflex_generate_engaged")
            if isinstance(eng, dict) and eng.get("active", 0) > 0:
                problems.append(f"head/{rep}: base not on flex, yet _NonFlexGenerate engaged {eng}")
            if set(r.get("attn_impl_seen_during_generate") or {}) != set(br[0].get("attn_impl_seen_during_generate") or {}):
                problems.append(f"head/{rep}: attn impl during generate changed although base not on flex")
        if any(d is not None for d in divs) and aa_div is None:
            problems.append("base not on flex, yet head tokens differ from base")
        notes.append("PR is expected to be a no-op on this host (base not on flex)")
    else:
        notes.append("head attn impl during generate: "
                     + "; ".join(str(r.get("attn_impl_seen_during_generate")) for r in hr)
                     + " vs base " + str(br[0].get("attn_impl_seen_during_generate")))
    # pytest.
    hp, bp = head.get("pytest") or {}, base.get("pytest") or {}
    hbad = set(hp.get("failed", [])) | set(hp.get("errors", []))
    if hbad:
        problems.append("PR test fails at head: " + ", ".join(sorted(hbad)))
    else:
        notes.append(f"PR test at head: {hp.get('n_passed', 0)} passed, {hp.get('n_skipped', 0)} skipped")
    notes.append(f"PR test at base (head copy): rc={bp.get('rc')}, {bp.get('n_passed', 0)} passed, "
                 f"{bp.get('n_failed', 0)} failed/errors (collection failure expected)")
    if problems:
        return True, "; ".join(problems) + ". Also: " + "; ".join(notes)
    return False, "; ".join(notes)
