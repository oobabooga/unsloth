#!/usr/bin/env python3
"""Criteria (unsloth #16197 + unsloth-zoo #2296): reset the inference stance counter at every generate, gfx1151 ROCm.

The defect: unsloth_zoo's generated forwards count inference forwards and switch Dynamo to eager_on_recompile once
one has run; nothing reset the count between generate() calls, so after a max_new_tokens=1 generate the next
generate switched right after its prefill and decode step 1 ran under eager_on_recompile (its decode graph never
compiled; flex_attention then falls back to math_attention eagerly).

Differential mode.
  base_shows_defect: in the base 270m sequence (2048x1, 2048x32, 2048x1, 2048x1, 2048x32), some multi-token
    generate that follows a 1-token generate entered decode step 1 under eager_on_recompile.
  head_is_fixed: every head / ufix cell finished; every multi-token generate entered decode step 1 under
    "default"; and no regression against the base:
      * decode step 2 onwards still eager_on_recompile in every multi-token generate (the guard skip is kept)
      * same-shape generates 2..N compile no new graph (graphs compiled once, not per generate)
      * same-shape ms/token median not slower than the base's by more than max(5%, 3x the base's spread over
        those generates)
      * SFT losses after generate within max(2e-3 relative, 1e-4) of the base's
  Greedy ids head vs base are reported per generate (eager vs compiled decode may legitimately differ in bf16).

Gates: each cell imported its arm's unsloth and unsloth_zoo; base zoo has no reset helper and base for_inference
does not call it; head zoo has it and head for_inference calls it (ufix: helper absent, for_inference calls the
fallback); base cells finished; the stance hook saw one top-level forward per generated token.
"""

from __future__ import annotations

import statistics

TITLE = "unsloth #16197 + unsloth-zoo #2296: inference stance counter reset per generate, gfx1151 ROCm"
MODE = "differential"
# nvidia: the L4 slowdown that motivated it; windows: no ROCm torch run here; exclusive_gpu: timing is reported.
NEEDS = ["rocm", "gpu", "exclusive_gpu", "nvidia", "windows"]
EAGER = "eager_on_recompile"
_CTX: dict = {}


def _arms(o):
    return (o or {}).get("arms") or {}


def _cell(o, arm, model):
    return ((_arms(o).get(arm) or {}).get("cells") or {}).get(model) or {}


def _gens(c, prefix = None):
    return [g for g in (c.get("generates") or []) if prefix is None or g.get("tag", "").startswith(prefix)]


def _multi(c):
    return [g for g in _gens(c) if (g.get("new") or 0) > 1]


def _after_one(c):
    """Multi-token sequence generates whose predecessor was a 1-token generate."""
    gs = _gens(c, "seq")
    return [g for prev, g in zip(gs, gs[1:]) if prev.get("new") == 1 and (g.get("new") or 0) > 1]


def _done(c):
    return c.get("stage") == "done" and not c.get("error") and not any(g.get("error") for g in _gens(c))


def _expected_cells(state):
    return [("base", "270m"), ("base", "1b")] if state == "base" else [("head", "270m"), ("head", "1b"), ("ufix", "270m")]


def gates(obs):
    _CTX["base"] = obs.get("base") or {}
    out = []
    for st in ("base", "head"):
        o = obs.get(st) or {}
        ck = str(o.get("checkout") or "<none>")
        if o.get("zoo_error"):
            out.append((f"{st}: zoo worktrees", False, o["zoo_error"][:300]))
            continue
        for arm, model in _expected_cells(st):
            c = _cell(o, arm, model)
            zp = str((_arms(o).get(arm) or {}).get("zoo_path") or "<none>")
            uf, zf = str(c.get("unsloth_file") or ""), str(c.get("zoo_file") or "")
            out.append((f"{arm}/{model} imported its arm's unsloth + unsloth_zoo",
                        uf.startswith(ck) and zf.startswith(zp),
                        f"{uf or 'none'} | {zf or 'none'} (stage={c.get('stage')} rc={c.get('rc')} "
                        f"{str(c.get('error') or c.get('json_err') or '')[:200]})"))
            want = {"base": (False, False), "head": (True, True), "ufix": (False, True)}[arm]
            got = (c.get("zoo_has_reset"), c.get("for_inference_calls_reset"))
            out.append((f"{arm}/{model} tree identity (zoo helper, for_inference reset) == {want}", got == want, str(got)))
            if arm == "base":
                out.append((f"base/{model} finished", _done(c),
                            str(c.get("error") or [g.get("error", "")[-200:] for g in _gens(c) if g.get("error")]
                                or f"stage={c.get('stage')} rc={c.get('rc')}")[:400]))
            if _done(c) or arm == "base":
                gs = _gens(c)
                bad = [(g.get("tag"), g.get("n_forwards"), g.get("new")) for g in gs if g.get("n_forwards") != g.get("new")]
                out.append((f"{arm}/{model} stance hook saw one forward per token in every generate",
                            bool(gs) and not bad, f"{len(gs)} generates; mismatches {bad[:5]}"))
    return out


def _fmt_ids(g, bg):
    a, b = g.get("ids") or [], (bg or {}).get("ids") or []
    if not a or not b:
        return "n/a"
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return f"diverge@{i}"
    return "equal" if len(a) == len(b) else "len differs"


def table(obs):
    base = obs.get("base") or {}
    rows = []
    vers = None
    for st in ("base", "head"):
        o = obs.get(st) or {}
        for arm, model in _expected_cells(st):
            c = _cell(o, arm, model)
            if not c:
                continue
            vers = vers or c.get("versions")
            bc = _cell(base, "base", model)
            bmap = {g.get("tag"): g for g in _gens(bc)}
            rows += ["", f"**{arm} / gemma-3-{model}-it** (unsloth {str((_arms(o).get(arm) or {}).get('unsloth_sha'))[:9]}, "
                     f"zoo {str((_arms(o).get(arm) or {}).get('zoo_sha'))[:9]}; attn `{c.get('attn_impl')}`, "
                     f"dtype {c.get('dtype')}, stage {c.get('stage')}, {c.get('seconds')} s"
                     + (f"; ERROR {str(c.get('error'))[:200]}" if c.get("error") else "") + ")", "",
                     "| generate | prompt x new | stance prefill / step1 / step2 | eager fwds | math_attention calls "
                     "| new graphs | runs before -> after | ms/token | ids vs base |",
                     "|---|---|---|---|---|---|---|---|---|"]
            for g in _gens(c):
                s = g.get("stances") or []
                rows.append(f"| {g.get('tag')} | {g.get('prompt')}x{g.get('new')} | {' / '.join(map(str, s[:3]))} "
                            f"| {(g.get('stance_counts') or {}).get(EAGER, 0)}/{g.get('n_forwards')} | {g.get('math')} "
                            f"| {g.get('new_graphs')} | {(g.get('runs_before') or {}).get('shared')}"
                            f"{(g.get('runs_before') or {}).get('modules')} -> {(g.get('runs_after') or {}).get('shared')}"
                            f"{(g.get('runs_after') or {}).get('modules')} | {g.get('ms_per_token')} "
                            f"| {_fmt_ids(g, bmap.get(g.get('tag'))) if arm != 'base' else '-'} |"
                            + (f" ERROR {g['error'][-150:]}" if g.get("error") else ""))
            if c.get("losses") is not None:
                rows.append(f"\nSFT after generate: losses {c.get('losses')}, grad norms {c.get('grad_norms')}, stance "
                            f"before/after train {c.get('stance_before_train')}/{c.get('stance_after_train')}")
    if obs.get("head"):
        head_is_fixed(obs["head"])
    if obs.get("base"):
        rows.append("\n- defect check: " + base_shows_defect(obs["base"])[1])
    rows.append("- merge state: not probed (base and head carry the evidence); its head_is_fixed is declared True.")
    if "head_parts" in _CTX:
        rows.append("- head checks: " + "; ".join(f"{k}: {v}" for k, v in _CTX["head_parts"].items()))
    if vers:
        rows.append("- Versions: " + ", ".join(f"{k}={v}" for k, v in vers.items()))
    return "\n".join(rows)


def base_shows_defect(base):
    c = _cell(base, "base", "270m")
    hits = [g.get("tag") for g in _after_one(c) if g.get("decode_step1_stance") == EAGER]
    return bool(hits), f"base 270m generates after a 1-token generate entering decode step 1 eager: {hits}"


def _spread(xs):
    return (max(xs) - min(xs)) if len(xs) >= 2 else 0.0


def head_is_fixed(head):
    if head.get("state") == "merge":
        return True
    base = _CTX.get("base") or {}
    parts, ok = {}, True
    for arm, model in _expected_cells("head"):
        c = _cell(head, arm, model)
        key = f"{arm}/{model}"
        if not _done(c):
            parts[key] = f"did not finish: {str(c.get('error') or c.get('rc'))[:200]}"
            ok = False
            continue
        bad1 = [g.get("tag") for g in _multi(c) if g.get("decode_step1_stance") != "default"]
        bad2 = [g.get("tag") for g in _multi(c) if g.get("new", 0) > 2 and (g.get("stances") or [None] * 3)[2] != EAGER]
        parts[f"{key} decode step 1 default"] = "yes" if not bad1 else f"NO {bad1}"
        parts[f"{key} decode step 2 still eager"] = "yes" if not bad2 else f"NO {bad2}"
        ok &= not bad1 and not bad2
        same = _gens(c, "same")
        if same:
            regraph = [(g.get("tag"), g.get("new_graphs")) for g in same[1:] if g.get("new_graphs")]
            parts[f"{key} same-shape generates 2..N new graphs"] = "0" if not regraph else f"NO {regraph}"
            ok &= not regraph
            bsame = _gens(_cell(base, "base", model), "same")
            hm = statistics.median(g["ms_per_token"] for g in same[1:])
            bt = [g["ms_per_token"] for g in bsame[1:]]
            if bt:
                bm = statistics.median(bt)
                thr = max(0.05 * bm, 3 * _spread(bt))
                parts[f"{key} same-shape ms/token median (gens 2..N)"] = (
                    f"head {hm:.2f} vs base {bm:.2f} (base spread {_spread(bt):.2f}, threshold +{thr:.2f})")
                if hm - bm > thr:
                    ok = False
                    parts[f"{key} same-shape timing"] = "head SLOWER beyond noise"
        if c.get("losses") is not None or model == "270m" and arm == "head":
            bl, hl = _cell(base, "base", model).get("losses") or [], c.get("losses") or []
            if not bl or len(bl) != len(hl):
                parts[f"{key} losses"] = f"NO comparable losses: head {hl} base {bl}"
                ok = False
            else:
                d = max(abs(x - y) for x, y in zip(hl, bl))
                tol = max(2e-3 * max(abs(x) for x in bl), 1e-4)
                parts[f"{key} losses vs base"] = f"max diff {d:.2e} (tol {tol:.1e}) {'ok' if d <= tol else 'NO'}"
                ok &= d <= tol
    _CTX["head_parts"] = parts
    return ok, "; ".join(f"{k}: {v}" for k, v in parts.items())
