#!/usr/bin/env python3
"""Criteria (unsloth-zoo PR 2240, GLM-5.3-Flash / glm5_next sparse flex), gfx1151 ROCm, base (zoo main) vs head.
Pairs with probes/zoo2240_probe.py.

Regression mode. Head is WORSE than base when any of:
  * a test that passed at the base does not pass at the head, or a test fails at the head and not at the base
    (the new tests/test_glm5_next_sparse.py is head-only, so any failure in it counts against the head)
  * a head GPU cell (tiny glm5_next model above the index budget, forward + backward) crashes or is non-finite
  * the flex fallback latch (sparse_blocks._FAILED) fired in the flex-on cell: flex was tried and failed on ROCm
  * flex on is further from the fp32 upstream reference than transformers' own bf16 path by more than the
    tolerance used for the other DSA families (test_tiny_model_flex_on_vs_off):
      |loss_on - loss_fp32| <= |loss_up - loss_fp32| + 5e-3   and   grad_rel_on <= 1.5 * grad_rel_up + 1e-2
Gates (non-vacuity): venv on HIP gfx1151 with transformers >= 5.21; each state imported its own unsloth_zoo;
pytest ran and collected in both states; the head's glm5_next test file EXECUTED tests (not all skipped); the head
cells ran in all four modes on the gfx1151 GPU with the patch installed; and the flex path ENGAGED in the
flex-on cell (counter > 0) or latched a failure. Zero engagement with no latch means the GPU path under test never
ran, which is no result.
"""

from __future__ import annotations

TITLE = "unsloth-zoo PR 2240 (GLM-5.3-Flash sparse flex) on gfx1151 ROCm, base vs head"
MODE = "regression"
NEEDS = ["rocm", "gpu", "nvidia", "exclusive_gpu"]

_RAN = (0, 1)
MODES = ("on", "off", "upstream", "fp32")
NEW_TEST = "test_glm5_next_sparse"


def _vt(s):
    try:
        return tuple(int(x) for x in str(s).split("+")[0].split(".")[:2])
    except ValueError:
        return (0, 0)


def _cells(o):
    return (o or {}).get("cells") or {}


def _flex_calls(cell):
    return sum(int((c.get("stats") or {}).get("flex", 0)) for c in (cell.get("cases") or {}).values())


def gates(obs):
    out = []
    for name in ("base", "head"):
        o = obs.get(name) or {}
        v = o.get("venv") or {}
        env = v.get("env") or {}
        out.append((f"{name} venv on HIP gfx1151, transformers >= 5.21",
                    not v.get("error") and bool(env.get("hip")) and str(env.get("arch") or "").startswith("gfx1151")
                    and env.get("cuda_available") is True and _vt(env.get("transformers")) >= (5, 21),
                    f"{v.get('error') or ''} {env} steps {v.get('steps')} {str(v.get('log_tail') or '')[-400:]}"))
        ck = str(o.get("checkout") or "<none>")
        f = str(o.get("unsloth_zoo_file") or "")
        out.append((f"{name} imported its own unsloth_zoo", bool(f) and f.startswith(ck),
                    f"{f or o.get('unsloth_zoo_import_error')}"))
        p = o.get("pytest") or {}
        n = (p.get("n_passed") or 0) + (p.get("n_failed") or 0) + (p.get("n_skipped") or 0)
        out.append((f"{name} pytest ran and collected", p.get("rc") in _RAN and n > 0,
                    f"rc={p.get('rc')} passed {p.get('n_passed')} failed {p.get('n_failed')} skipped {p.get('n_skipped')} "
                    f"per file {p.get('per_file')} {str(p.get('stderr_tail') or '')[-300:]}"))
    h = obs.get("head") or {}
    hp = h.get("pytest") or {}
    ran = [t for t in (hp.get("passed") or []) + (hp.get("failed") or []) + (hp.get("errors") or []) if NEW_TEST in t]
    skipped = [t for t in (hp.get("skipped") or {}) if NEW_TEST in t]
    out.append(("head executed the glm5_next tests (not skipped)", len(ran) > 0,
                f"{len(ran)} executed, {len(skipped)} skipped; reasons {sorted(set((hp.get('skipped') or {}).get(t) for t in skipped))[:3]}"))
    out.append(("head has the glm5_next patch, base does not",
                h.get("has_glm5_next_patch") is True and (obs.get("base") or {}).get("has_glm5_next_patch") is False,
                f"head {h.get('has_glm5_next_patch')} base {(obs.get('base') or {}).get('has_glm5_next_patch')}"))
    hc = _cells(h)
    stage = {m: (hc.get(m) or {}).get("stage") for m in MODES}
    out.append(("head cells ran in all four modes (harness reached the model)", all(s == "done" for s in stage.values()),
                f"{stage}; fatal {[(m, (hc.get(m) or {}).get('fatal'), str((hc.get(m) or {}).get('log_tail'))[-300:]) for m, s in stage.items() if s != 'done']}"))
    arch = {m: ((hc.get(m) or {}).get("device") or {}).get("arch") for m in MODES}
    out.append(("every head cell ran on the gfx1151 GPU with the patch installed",
                all(str(a or "").startswith("gfx1151") for a in arch.values())
                and all((hc.get(m) or {}).get("patch_installed") is True for m in MODES),
                f"arch {arch} patch_installed {[(m, (hc.get(m) or {}).get('patch_installed')) for m in MODES]}"))
    on = hc.get("on") or {}
    out.append(("flex-on cell engaged flex (or latched a failure)",
                _flex_calls(on) > 0 or bool(on.get("failed_latch")),
                f"flex calls {_flex_calls(on)}; gates {on.get('gates')}; per case {[(k, c.get('stats')) for k, c in (on.get('cases') or {}).items()]}; "
                f"latch {on.get('failed_latch')}"))
    return out


def head_is_worse(base, head):
    problems, notes = [], []
    hc = _cells(head)
    on = hc.get("on") or {}
    notes.append(f"flex on: gates {on.get('gates')}; flex calls {_flex_calls(on)}; per case "
                 f"{[(k, c.get('stats')) for k, c in (on.get('cases') or {}).items()]}")
    if on.get("failed_latch"):
        problems.append(f"FLEX FALLBACK LATCHED on ROCm: {on.get('failed_latch')}")
    for mode in MODES:
        for case, c in ((hc.get(mode) or {}).get("cases") or {}).items():
            if c.get("stage") != "done" or not c.get("finite"):
                problems.append(f"{mode}/{case} {'crashed at ' + str(c.get('stage')) if c.get('stage') != 'done' else 'non-finite'}: "
                                f"{str(c.get('error'))[:400]}")
    cmp_ = (hc.get("compare") or {}).get("cases") or {}
    if not cmp_:
        problems.append(f"no compare output: {str((hc.get('compare') or {}).get('log_tail'))[-400:]}")
    for case, m in cmp_.items():
        if m.get("missing"):
            problems.append(f"{case}: no comparison, missing {m['missing']}")
            continue
        a, u, o = m["on_vs_fp32"], m["upstream_vs_fp32"], m["off_vs_fp32"]
        loss_ok = a["loss_abs"] <= u["loss_abs"] + 5e-3
        grad_ok = (a["grad_rel_max"] or 0) <= 1.5 * (u["grad_rel_max"] or 0) + 1e-2
        out_ok = a["out_rel"] <= 1.5 * u["out_rel"] + 1e-2
        line = (f"{case}: vs fp32 upstream: loss abs on {a['loss_abs']:.2e} off {o['loss_abs']:.2e} upstream-bf16 {u['loss_abs']:.2e}; "
                f"out rel on {a['out_rel']:.4f} off {o['out_rel']:.4f} up {u['out_rel']:.4f}; grad rel on {a['grad_rel_max']:.4f} "
                f"(worst {a['grad_rel_worst_param']}) off {o['grad_rel_max']:.4f} up {u['grad_rel_max']:.4f}")
        ok = loss_ok and grad_ok and out_ok
        (notes if ok else problems).append(("" if ok else "FLEX OVER TOLERANCE ") + line)
    bp, hp = (base or {}).get("pytest") or {}, (head or {}).get("pytest") or {}
    bpass, hpass = set(bp.get("passed") or []), set(hp.get("passed") or [])
    hbad = set(hp.get("failed") or []) | set(hp.get("errors") or [])
    bbad = set(bp.get("failed") or []) | set(bp.get("errors") or [])
    lost = sorted(bpass - hpass)
    if lost:
        problems.append(f"passed at base, not at head: {lost}")
    if hbad - bbad:
        problems.append(f"failing at head, not at base: {sorted(hbad - bbad)}")
    if hbad & bbad:
        notes.append(f"failing at both (pre-existing): {sorted(hbad & bbad)}")
    for name, p in (("base", bp), ("head", hp)):
        notes.append(f"{name} pytest {p.get('n_passed')}/{p.get('n_failed')}/{p.get('n_skipped')} (passed/failed/skipped)")
    if problems:
        return True, "; ".join(problems)
    return False, "; ".join(notes)


def table(obs):
    hc = _cells(obs.get("head"))
    rows = ["| mode | case | stage | loss | glm5_next counters | new _FAILED | s |", "|" + "---|" * 7]
    for mode in MODES:
        for case, c in ((hc.get(mode) or {}).get("cases") or {}).items():
            rows.append(f"| {mode} | {case} | {c.get('stage')}{' ERR ' + str(c.get('error'))[:200] if c.get('error') else ''} | "
                        f"{c.get('loss')} | {c.get('stats')} | {c.get('failed_new') or ''} | {c.get('seconds')} |")
    rows += ["", "| case | out rel vs fp32 on / off / upstream-bf16 | grad rel vs fp32 on / off / upstream-bf16 | on vs upstream out / grad |",
             "|---|---|---|---|"]
    for case, m in ((hc.get("compare") or {}).get("cases") or {}).items():
        if m.get("missing"):
            rows.append(f"| {case} | missing {m['missing']} | | |")
            continue
        a, o, u, d = m["on_vs_fp32"], m["off_vs_fp32"], m["upstream_vs_fp32"], m["on_vs_upstream"]
        rows.append(f"| {case} | {a['out_rel']:.4f} / {o['out_rel']:.4f} / {u['out_rel']:.4f} | "
                    f"{a['grad_rel_max']:.4f} / {o['grad_rel_max']:.4f} / {u['grad_rel_max']:.4f} | {d['out_rel']:.4f} / {d['grad_rel_max']:.4f} |")
    rows += ["", "| state | pytest passed | failed | skipped | failing tests |", "|---|---|---|---|---|"]
    for name in ("base", "head"):
        p = (obs.get(name) or {}).get("pytest") or {}
        bad = sorted(set(p.get("failed") or []) | set(p.get("errors") or []))
        rows.append(f"| {name} | {p.get('n_passed')} | {p.get('n_failed')} | {p.get('n_skipped')} | {', '.join(bad) or 'none'} |")
    absent = (obs.get("base") or {}).get("pytest", {}).get("absent_at_this_state") or []
    if absent:
        rows.append(f"\nAbsent at the base (added by the PR): {absent}")
    rows.append("")
    for name in ("base", "head"):
        p = (obs.get(name) or {}).get("pytest") or {}
        rows.append(f"{name} skip reasons: {sorted({v for v in (p.get('skipped') or {}).values()})[:8]}")
        for t, m in sorted((p.get("fail_msgs") or {}).items()):
            rows.append(f"{name} FAIL `{t}`: {m[:500]!r}")
    v = (obs.get("head") or {}).get("venv") or {}
    rows += ["", f"env: {v.get('env')}; {v.get('pip_list')}; triton fixup {v.get('triton_fixup')}"]
    return "\n".join(rows)
