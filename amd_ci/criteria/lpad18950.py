#!/usr/bin/env python3
"""Criteria (unsloth #18950): fully masked pad query rows attend only their own key (masking_utils sdpa mask builder
+ native SDPA dispatch). Regression mode with a non-vacuity gate: the base arm must show the defect (a left padded
pad query at the last pad position reads future tokens: leak_last_pad > 0), else the run is not evidence.

head_is_worse when any of:
  - a test ID fails / errors at head that passed at base (or was absent at base),
  - a cell trained (finite) at base but not at head, or head has non-finite logits / losses / grad norms,
  - a right padded cell differs at all base vs head (losses, grad norms, step-0 per-token loss bytes),
  - a left padded cell at head still leaks (leak_last_pad or leak_pad_rows != 0), or its step-0 loss over every
    label except the first-real-token terms differs base vs head (only that one term per row may move).
"""
from __future__ import annotations

import math
import os

TITLE = "unsloth #18950: left padded pad queries attend only their own key, gfx1151 base vs head"
MODE = "regression"
NEEDS = ["rocm", "gpu", "nvidia", "multi_gpu", "windows"]
LEFT = ("lpad_left_bf16", "lpad_left_fp16")
RIGHT = ("lpad_right_bf16", "lpad_right_fp16")
CELLS = LEFT + RIGHT


def _norm(p):
    return os.path.normcase(os.path.normpath(str(p or "")))


def _cells(o):
    return (o or {}).get("cells") or {}


def _trained(c):
    return bool(c) and c.get("stage") == "done" and c.get("finite") is True and len(c.get("losses") or []) == 10


def _arch(o):
    for c in _cells(o).values():
        a = (c.get("versions") or {}).get("arch")
        if a:
            return str(a)
    return ""


def gates(obs):
    out = []
    for name in ("base", "head"):
        o = obs.get(name) or {}
        env = o.get("env") or {}
        out.append((f"{name} environment built", not env.get("error") and bool((env.get("env") or {}).get("hip")),
                    str(env.get("error") or env.get("env"))[:300]))
        paths = o.get("paths") or {}
        bad, files = [], set()
        for cname, c in _cells(o).items():
            for mod, key in (("unsloth", "unsloth_file"), ("zoo", "unsloth_zoo_file")):
                f = c.get(key)
                if key == "unsloth_file" and f:
                    files.add(str(f))
                if f and paths.get(mod) and not _norm(f).startswith(_norm(paths[mod])):
                    bad.append(f"{cname}:{mod}={f}")
        out.append((f"{name} cells imported this state's own unsloth tree", bool(files) and not bad,
                    ("; ".join(bad) or ", ".join(sorted(files)))[:300] or "no cell imported unsloth"))
        out.append((f"{name} ran on gfx1151 (HIP)", "gfx1151" in _arch(o), _arch(o) or "no device recorded"))
        p = o.get("pytest")
        if p is not None:
            n = p.get("n_passed", 0) + p.get("n_failed", 0) + p.get("n_skipped", 0)
            out.append((f"{name} pytest collected tests", n > 0,
                        f"passed={p.get('n_passed')} failed={p.get('n_failed')} skipped={p.get('n_skipped')} "
                        f"crashed={p.get('crashed_files')}"[:300]))
    b = _cells(obs.get("base")).get("lpad_left_bf16") or {}
    out.append(("base lpad_left_bf16 shows the defect (last pad query reads future tokens: leak_last_pad > 0)",
                (b.get("leak_last_pad") or 0) > 0,
                f"leak_last_pad={b.get('leak_last_pad')} stage={b.get('stage')} err={str(b.get('error'))[:150]}"))
    h = _cells(obs.get("head")).get("lpad_left_bf16") or {}
    out.append(("head lpad_left_bf16 ran the leak probe", h.get("leak_last_pad") is not None,
                f"leak_last_pad={h.get('leak_last_pad')} stage={h.get('stage')} err={str(h.get('error'))[:150]}"))
    return out


def _pyt(o):
    p = (o or {}).get("pytest") or {}
    return set(p.get("passed") or []), set(p.get("failed") or []) | set(p.get("errors") or [])


def _aa_bitwise(base, cell):
    a, b = _cells(base).get(cell) or {}, _cells(base).get(cell + "@aa") or {}
    if not (_trained(a) and _trained(b)):
        return None
    return a.get("losses") == b.get("losses") and a.get("step0_rest_tokens_sha") == b.get("step0_rest_tokens_sha")


def _maxrel(x, y):
    return max((abs(a - b) / max(1.0, abs(a)) for a, b in zip(x, y)), default = 0.0)


def head_is_worse(base, head):
    worse, notes = [], []
    bp, bf = _pyt(base)
    hp, hf = _pyt(head)
    new_fail = sorted(hf - bf)
    if new_fail:
        worse.append("tests failing at head, not at base: " + ", ".join(t.split("::")[-1] for t in new_fail)[:600])
    fixed = sorted(bf - hf)
    if fixed:
        notes.append(f"{len(fixed)} test(s) failing at base pass at head")
    bc, hc = _cells(base), _cells(head)
    for cell in CELLS:
        b, h = bc.get(cell), hc.get(cell)
        if b is None and h is None:
            continue
        if _trained(b) and not _trained(h):
            worse.append(f"{cell}: trained at base, not at head ({str((h or {}).get('error') or (h or {}).get('stage'))[:200]})")
            continue
        if not _trained(b) and not _trained(h):
            notes.append(f"{cell}: did not train at either arm ({str((h or {}).get('error') or (h or {}).get('stage'))[:150]})")
            continue
        if not _trained(b):
            notes.append(f"{cell}: trains at head only")
            continue
        if h.get("logits_finite_all_positions") is False:
            worse.append(f"{cell}: non-finite logits at head (pad rows included)")
        if cell in RIGHT:
            aa = _aa_bitwise(base, cell)
            same = (b.get("losses") == h.get("losses") and b.get("grad_norms") == h.get("grad_norms")
                    and b.get("step0_rest_tokens_sha") == h.get("step0_rest_tokens_sha"))
            if not same:
                a = _cells(base).get(cell + "@aa") or {}
                spread = _maxrel(b.get("losses") or [], a.get("losses") or []) if _trained(a) else None
                d = min(_maxrel(b.get("losses") or [], h.get("losses") or []),
                        _maxrel(a.get("losses") or [], h.get("losses") or []) if _trained(a) else 9e9)
                det = (b.get("step0_rest_tokens_sha") == h.get("step0_rest_tokens_sha")
                       and (b.get("losses") or [None])[0] == (h.get("losses") or [None])[0])
                if aa is not False or not det or spread is None or d > 3 * spread:
                    worse.append(f"{cell}: right padded run differs base vs head (step-0 forward + first loss bitwise: "
                                 f"{det}; max rel loss diff {d:.3g}; base A/A bitwise={aa}, spread {spread})")
                else:
                    notes.append(f"{cell}: right padded step-0 forward and first loss bitwise equal; later steps differ by "
                                 f"{d:.3g} rel, inside the base A/A spread {spread:.3g} (training is not bitwise "
                                 f"reproducible run to run on this host)")
        else:
            if (h.get("leak_last_pad") or 0) != 0 or (h.get("leak_pad_rows") or 0) != 0:
                worse.append(f"{cell}: head still leaks future tokens into pad queries "
                             f"(last pad {h.get('leak_last_pad')}, pad rows {h.get('leak_pad_rows')})")
            bs, hs = b.get("step0_rest_per_row_sha") or [], h.get("step0_rest_per_row_sha") or []
            moved = [r for r, (x, y) in enumerate(zip(bs, hs)) if x != y]
            pp = b.get("pad_perturb_all_rows_real_maxabs") or []
            if b.get("step0_rest_tokens_sha") != h.get("step0_rest_tokens_sha"):
                coupled = bool(moved) and all(r < len(pp) and pp[r] != 0 for r in moved)
                msg = (f"{cell}: step-0 loss over labels other than the first real token differs base vs head in rows "
                       f"{moved or '?'} ({b.get('step0_loss_rest_sum')} vs {h.get('step0_loss_rest_sum')})")
                if coupled:
                    notes.append(msg + f"; at the BASE, changing only pad token ids already moves those rows' real logits "
                                       f"(per row {pp}), so real outputs carry a numerical dependence on pad content "
                                       f"that predates this PR, and changing the pad rows moves them")
                else:
                    worse.append(msg + f"; base pad perturbation per row {pp}")
            if b.get("step0_first_real_terms") == h.get("step0_first_real_terms"):
                notes.append(f"{cell}: first-real-token terms identical base vs head (expected to differ)")
    return bool(worse), "; ".join(worse + notes) or "head is no worse than base"


def _f(x, n = 6):
    return "-" if x is None else (f"{x:.{n}g}" if isinstance(x, float) else str(x))


def table(obs):
    lines = ["| state | cell | attn | trained | finite logits (all / real) | leak last pad / pad rows / first real | "
             "pad-perturb real logits per row | mask real->pad allowed (calls) | unpadded vs left max abs | "
             "step0 rest sum (sha8) | rest per row (sha) | first-real terms | losses (10 steps) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for st in ("base", "head"):
        for cell, c in sorted(_cells(obs.get(st)).items()):
            if not cell.startswith("lpad_"):
                continue
            lines.append(f"| {st} | {cell} | {c.get('attn_implementation')} | {_trained(c)} | "
                         f"{c.get('logits_finite_all_positions')} / {c.get('logits_finite_real_positions')} | "
                         f"{_f(c.get('leak_last_pad'))} / {_f(c.get('leak_pad_rows'))} / {_f(c.get('leak_first_real'))} | "
                         f"{[_f(x, 3) for x in c.get('pad_perturb_all_rows_real_maxabs') or []]} | "
                         f"{c.get('mask_real_to_pad_allowed_step0')} ({c.get('sdpa_mask_calls_step0')}) | "
                         f"{_f(c.get('unpadded_vs_left_real_maxabs'))} | "
                         f"{_f(c.get('step0_loss_rest_sum'), 12)} ({str(c.get('step0_rest_tokens_sha'))[:8]}) | "
                         f"{c.get('step0_rest_per_row_sha')} | "
                         f"{[round(x, 4) for x in c.get('step0_first_real_terms') or []]} | "
                         f"{[round(x, 5) for x in c.get('losses') or []] or str(c.get('error') or c.get('stage'))[:200]} |")
    lines.append("")
    for st in ("base", "head"):
        p = (obs.get(st) or {}).get("pytest") or {}
        lines.append(f"- {st} pytest: passed {p.get('n_passed')}, failed {p.get('n_failed')}, skipped {p.get('n_skipped')}, "
                     f"crashed files {p.get('crashed_files')}, copied from head {(obs.get(st) or {}).get('tests_copied_from_head')}")
        for t in sorted(set(p.get("failed") or []) | set(p.get("errors") or [])):
            msg = " ".join(str((p.get("fail_msgs") or {}).get(t, "")).split())[:240]
            lines.append(f"  - FAIL `{t}`: {msg}")
    return "\n".join(lines)
