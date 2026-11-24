#!/usr/bin/env python3
"""Criteria (unsloth-zoo PR 2106 sparse flex attention, with #2101's exact skip), gfx1151 ROCm, base vs head.
Pairs with probes/zoo2106_probe.py. Base = the commit before #2106 (#2101 and #2104 already in), head = #2106.

Regression mode. Head is WORSE than base when any of:
  * a test that passed at the base does not pass at the head, or a test fails at the head and not at the base
  * a head cell (MiniMax-M3 / GLM-5 tiny model above the budget, forward + backward) crashes or is non-finite with
    flex on (the default), i.e. the fallback is not clean
  * flex on is further from the fp32 dense reference than the model's own bf16 path by more than the tolerance of
    tests/test_sparse_attention_flex.py::test_tiny_model_flex_on_vs_off:
      |loss_on - loss_fp32| <= |loss_off - loss_fp32| + 5e-3   and   grad_rel_on <= 1.5 * grad_rel_off + 1e-2
  * a ROCm-unskipped flex parity test (the head's own test file, ROCm marker removed) fails on an assertion about
    numbers (output / gradient tolerance), not about engagement
Gates: venv built on HIP + gfx1151 + transformers >= 5.17; each state imported its own checkout; pytest ran and
collected; the #2101 skip test file ran in both states; head cells ran for both families in all three modes.
Engagement (flex counters, the _FAILED latch) is REPORTED, not gated: either answer is a finding.
"""

from __future__ import annotations

TITLE = "unsloth-zoo PR 2106 (sparse flex) + #2101 skip on gfx1151 ROCm, base vs head"
MODE = "regression"
NEEDS = ["rocm", "gpu", "nvidia", "exclusive_gpu"]

_RAN = (0, 1)
FAMILIES = ("minimax_m3_vl", "glm_moe_dsa")
MODES = ("on", "off", "fp32")
SKIP_TEST = "test_sparse_attention_skip"
NUMERIC_MARKERS = ("err", "rel", "allclose", "flex_err", "Mismatch", "not close")


def _cells(o):
    return (o or {}).get("cells") or {}


def _vt(s):
    try:
        return tuple(int(x) for x in str(s).split("+")[0].split(".")[:2])
    except ValueError:
        return (0, 0)


def gates(obs):
    out = []
    for name in ("base", "head"):
        o = obs.get(name) or {}
        v = o.get("venv") or {}
        env = v.get("env") or {}
        out.append((f"{name} venv on HIP gfx1151, transformers >= 5.17",
                    not v.get("error") and bool(env.get("hip")) and str(env.get("arch") or "").startswith("gfx1151")
                    and env.get("cuda_available") is True and _vt(env.get("transformers")) >= (5, 17),
                    f"{v.get('error') or ''} {env} steps {v.get('steps')} {str(v.get('log_tail') or '')[-400:]}"))
        ck = str(o.get("checkout") or "<none>")
        f = str(o.get("unsloth_zoo_file") or "")
        out.append((f"{name} imported its own unsloth_zoo", bool(f) and f.startswith(ck),
                    f"{f or o.get('unsloth_zoo_import_error')}"))
        p = o.get("pytest") or {}
        n = p.get("n_passed", 0) + p.get("n_failed", 0) + p.get("n_skipped", 0)
        out.append((f"{name} pytest ran and collected", p.get("rc") in _RAN and n > 0,
                    f"rc={p.get('rc')} passed {p.get('n_passed')} failed {p.get('n_failed')} skipped {p.get('n_skipped')} "
                    f"absent {p.get('absent_at_this_state')} {str(p.get('stderr_tail') or '')[-200:] if p.get('rc') not in _RAN else ''}"))
        ids = list(p.get("passed") or []) + list(p.get("failed") or []) + list(p.get("errors") or [])
        out.append((f"{name} ran the #2101 skip tests (not all skipped)", any(SKIP_TEST in i for i in ids),
                    f"{sum(SKIP_TEST in i for i in ids)} executed"))
    h = obs.get("head") or {}
    out.append(("head has sparse_blocks.py, base does not",
                h.get("has_sparse_blocks") is True and (obs.get("base") or {}).get("has_sparse_blocks") is False,
                f"head {h.get('has_sparse_blocks')} base {(obs.get('base') or {}).get('has_sparse_blocks')}"))
    hc = _cells(h)
    ran = {f"{f}/{m}": (hc.get(f) or {}).get(m, {}).get("stage") for f in FAMILIES for m in MODES}
    out.append(("head cells ran for both families, all modes (harness reached the model)",
                all(v == "done" for v in ran.values()),
                f"{ran}; fatal {[(k, (hc.get(k.split('/')[0]) or {}).get(k.split('/')[1], {}).get('fatal')) for k, v in ran.items() if v != 'done']}"))
    gpu_seen = {f"{f}/{m}": ((hc.get(f) or {}).get(m, {}).get("device") or {}).get("arch") for f in FAMILIES for m in MODES}
    out.append(("every head cell ran on the gfx1151 GPU",
                all(str(a or "").startswith("gfx1151") for a in gpu_seen.values()), str(gpu_seen)))
    u = h.get("pytest_rocm_unskipped") or {}
    out.append(("ROCm-unskipped flex test copy ran",
                h.get("unskip_needle_count", 0) >= 1 and u.get("rc") in _RAN and (u.get("n_passed", 0) + u.get("n_failed", 0)) > 0,
                f"needles {h.get('unskip_needle_count')} rc {u.get('rc')} passed {u.get('n_passed')} failed {u.get('n_failed')} skipped {u.get('n_skipped')}"))
    tr = (h.get("tiny_repeats") or {}).get("runs") or []
    fams = {row["family"] for r in tr for row in r.get("rows") or []}
    out.append(("tiny-model flex on/off test repeated 5x, numbers parsed for all 3 families",
                len(tr) == 5 and fams == {"minimax_m3_vl", "deepseek_v32", "glm_moe_dsa"}
                and all(len(r.get("rows") or []) == 3 for r in tr),
                f"{[(r.get('rc'), len(r.get('rows') or []), r.get('summary')) for r in tr]}"))
    aa = h.get("compile_aa") or {}
    reps = aa.get("repeats") or []
    out.append(("compile A/A ran (3 repeats, both tests executed)",
                aa.get("found") is True and len(reps) == 3 and all((r.get("n_passed") or 0) + (r.get("n_failed") or 0) >= 4 for r in reps),
                f"found {aa.get('found')} repeats {[(r.get('rc'), r.get('n_passed'), r.get('n_failed')) for r in reps]}"))
    return out


def _engaged(cell):
    flex = 0
    for c in (cell.get("cases") or {}).values():
        flex += sum(((c.get("stats_delta") or {}).get("flex") or {}).values())
    return flex


def head_is_worse(base, head):
    problems, notes = [], []
    hc = _cells(head)
    for fam in FAMILIES:
        cells = hc.get(fam) or {}
        on = cells.get("on") or {}
        cmp_ = (cells.get("compare") or {}).get("cases") or {}
        notes.append(f"[{fam}] flex on: gates {on.get('gates')}; flex calls {_engaged(on)}; "
                     f"stats {on.get('stats_total')}; _FAILED latch {on.get('failed_latch') or '{}'}")
        for mode in MODES:
            for case, c in ((cells.get(mode) or {}).get("cases") or {}).items():
                if c.get("stage") != "done" or not c.get("finite"):
                    problems.append(f"[{fam}] {mode}/{case} {'crashed at ' + str(c.get('stage')) if c.get('stage') != 'done' else 'non-finite'}: "
                                    f"{str(c.get('error'))[:400]}")
        for case, m in cmp_.items():
            if m.get("missing"):
                problems.append(f"[{fam}] {case}: no comparison, missing {m['missing']}")
                continue
            a, b, d = m["on_vs_fp32"], m["off_vs_fp32"], m["on_vs_off"]
            loss_ok = a["loss_abs"] <= b["loss_abs"] + 5e-3
            grad_ok = (a["grad_rel_max"] or 0) <= 1.5 * (b["grad_rel_max"] or 0) + 1e-2
            line = (f"[{fam}] {case}: loss fp32 {m['loss']['fp32']:.5f} off {m['loss']['off']:.5f} on {m['loss']['on']:.5f}; "
                    f"vs fp32 grad rel off {b['grad_rel_max']:.4f} on {a['grad_rel_max']:.4f} (worst {a['grad_rel_worst_param']}); "
                    f"logits rel off {b['logits_rel']:.4f} on {a['logits_rel']:.4f}; on vs off logits rel {d['logits_rel']:.4f} "
                    f"grad rel {d['grad_rel_max']:.4f} bit-identical {d['bit_identical_logits']}")
            (notes if loss_ok and grad_ok else problems).append(("" if loss_ok and grad_ok else "FLEX OFF TOLERANCE ") + line)
    u = (head or {}).get("pytest_rocm_unskipped") or {}
    aa = (head or {}).get("compile_aa") or {}
    reps = aa.get("repeats") or []

    tiny_runs = ((head or {}).get("tiny_repeats") or {}).get("runs") or []
    tiny: dict = {}
    for r in tiny_runs:
        for row in r.get("rows") or []:
            tiny.setdefault(row["family"], []).append(row)

    def _aa_fails(fam, name):
        return sum(any(f"{name}[{fam}]" in t for t in (r.get("failed") or []) + (r.get("errors") or [])) for r in reps)

    for t in sorted(set(u.get("failed") or []) | set(u.get("errors") or [])):
        msg = (u.get("fail_msgs") or {}).get(t, "")
        numeric = "AssertionError" in msg and any(k in msg for k in NUMERIC_MARKERS) and "_FAILED" not in msg.split("\n")[0]
        fam = t.rsplit("[", 1)[-1].rstrip("]") if "[" in t else ""
        if numeric and "inside_torch_compile_matches_eager" in t and reps:
            orig, twin = _aa_fails(fam, "test_attention_inside_torch_compile_matches_eager"), _aa_fails(fam, "test_compile_aa_flex_off_twice")
            line = (f"ROCm-unskipped failure {t}: {msg[:300]!r}; repeats: flex-on vs flex-off compiled failed {orig}/{len(reps)}, "
                    f"A/A (flex-off compiled twice, same assertion) failed {twin}/{len(reps)}")
            if twin:
                notes.append("COMPILE-TO-COMPILE NOISE (A/A fails alike, flex is gated off inside compile) " + line)
            else:
                problems.append("NUMERIC (A/A clean) " + line)
            continue
        if numeric and "test_tiny_model_flex_on_vs_off" in t and tiny:
            notes.append(f"ROCm-unskipped single draw failed {t}: {msg[:200]!r}; judged on the {len(tiny_runs)}-repeat median below")
            continue
        (problems if numeric else notes).append(f"ROCm-unskipped {'NUMERIC ' if numeric else ''}failure {t}: {msg[:500]!r}")
    for fam, rs in sorted(tiny.items()):
        import statistics
        fl, de = statistics.median(r["grad_flex"] for r in rs), statistics.median(r["grad_dense"] for r in rs)
        lf = statistics.median(abs(r["loss_flex"] - r["loss_fp32"]) for r in rs)
        ld = statistics.median(abs(r["loss_dense"] - r["loss_fp32"]) for r in rs)
        over = sum(r["grad_flex"] > 1.5 * r["grad_dense"] + 1e-2 or abs(r["loss_flex"] - r["loss_fp32"]) > abs(r["loss_dense"] - r["loss_fp32"]) + 5e-3 for r in rs)
        line = (f"[tiny {fam}] {len(rs)} repeats: median grad rel vs fp32 flex {fl:.4f} dense {de:.4f} "
                f"(flex {[round(r['grad_flex'], 4) for r in rs]} dense {[round(r['grad_dense'], 4) for r in rs]}); "
                f"median |dloss| flex {lf:.2e} dense {ld:.2e}; draws over the test bound {over}/{len(rs)}")
        ok = fl <= 1.5 * de + 1e-2 and lf <= ld + 5e-3
        (notes if ok else problems).append(("" if ok else "TINY MEDIAN OVER TOLERANCE ") + line)
    if u:
        notes.append(f"ROCm-unskipped flex tests: passed {u.get('n_passed')} failed {u.get('n_failed')} skipped {u.get('n_skipped')}")
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
        notes.append(f"failing at both: {sorted(hbad & bbad)}")
    for name, p in (("base", bp), ("head", hp)):
        sk = [t for t in p.get("passed") or [] if SKIP_TEST in t]
        notes.append(f"{name} pytest {p.get('n_passed')}/{p.get('n_failed')}/{p.get('n_skipped')} (passed/failed/skipped); "
                     f"#2101 skip tests passed {len(sk)} (cuda params {sum('cuda' in t for t in sk)})")
    if problems:
        return True, "; ".join(problems)
    return False, "; ".join(notes)


def table(obs):
    rows = ["| family | mode | case | stage | loss | flex calls | other counters | new _FAILED | s |", "|" + "---|" * 9]
    for fam, cells in _cells(obs.get("head")).items():
        for mode in MODES:
            c0 = cells.get(mode) or {}
            for case, c in (c0.get("cases") or {}).items():
                sd = c.get("stats_delta") or {}
                rows.append(f"| {fam} | {mode} | {case} | {c.get('stage')}{' ERR ' + str(c.get('error'))[:200] if c.get('error') else ''} | "
                            f"{c.get('loss')} | {sum((sd.get('flex') or {}).values())} | "
                            f"{ {k: v for k, v in sd.items() if k != 'flex'} } | {c.get('failed_new') or ''} | {c.get('seconds')} |")
    rows += ["", "| family | case | loss on/off/fp32 | logits rel vs fp32 on / off | grad rel vs fp32 on / off | on vs off logits / grad |",
             "|---|---|---|---|---|---|"]
    for fam, cells in _cells(obs.get("head")).items():
        for case, m in ((cells.get("compare") or {}).get("cases") or {}).items():
            if m.get("missing"):
                rows.append(f"| {fam} | {case} | missing {m['missing']} | | | |")
                continue
            a, b, d, lo = m["on_vs_fp32"], m["off_vs_fp32"], m["on_vs_off"], m["loss"]
            rows.append(f"| {fam} | {case} | {lo['on']:.5f} / {lo['off']:.5f} / {lo['fp32']:.5f} | {a['logits_rel']:.4f} / {b['logits_rel']:.4f} | "
                        f"{a['grad_rel_max']:.4f} / {b['grad_rel_max']:.4f} | {d['logits_rel']:.4f} / {d['grad_rel_max']:.4f} |")
    rows += ["", "| state | pytest passed | failed | skipped | failing tests |", "|---|---|---|---|---|"]
    for name in ("base", "head"):
        p = (obs.get(name) or {}).get("pytest") or {}
        bad = sorted(set(p.get("failed") or []) | set(p.get("errors") or []))
        rows.append(f"| {name} | {p.get('n_passed')} | {p.get('n_failed')} | {p.get('n_skipped')} | {', '.join(bad) or 'none'} |")
    u = (obs.get("head") or {}).get("pytest_rocm_unskipped") or {}
    if u:
        bad = sorted(set(u.get("failed") or []) | set(u.get("errors") or []))
        rows.append(f"| head, flex tests ROCm-unskipped | {u.get('n_passed')} | {u.get('n_failed')} | {u.get('n_skipped')} | {', '.join(bad) or 'none'} |")
    rows.append("")
    for name in ("base", "head"):
        p = (obs.get(name) or {}).get("pytest") or {}
        sk = sorted({v for v in (p.get("skipped") or {}).values()})
        rows.append(f"{name} skip reasons: {sk[:8]}")
        for t, m in sorted((p.get("fail_msgs") or {}).items()):
            rows.append(f"{name} FAIL `{t}`: {m[:500]!r}")
        rows.append("")
    for t, m in sorted((u.get("fail_msgs") or {}).items()):
        rows.append(f"head unskipped FAIL `{t}`: {m[:600]!r}")
    for i, r in enumerate(((obs.get("head") or {}).get("compile_aa") or {}).get("repeats") or []):
        rows.append(f"compile A/A repeat {i}: passed {r.get('passed')} failed {r.get('failed')} {list((r.get('fail_msgs') or {}).values())[:2]}")
    v = ((obs.get("head") or {}).get("venv") or {})
    rows += ["", f"env: {v.get('env')}; {v.get('pip_list')}"]
    return "\n".join(rows)
