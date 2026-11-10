#!/usr/bin/env python3
"""Criteria (unsloth-zoo PR 2241, Inductor rand4x off on torch 2.13 / 2.14), lane `gate`: stock gfx1151 torch 2.11,
base (zoo main) vs head. Pairs with probes/zoo2241_probe.py --lane gate.

torch 2.11 has no `_can_use_4x_random`, so the patch must NOT install: "rest unchanged".
Regression mode. Head is WORSE than base when any of:
  * a test that passed at the base does not pass at the head, or a test fails at the head and not at the base
    (tests/test_inductor_rand4x.py is head-only, so any failure in it counts against the head)
  * importing the head's unsloth_zoo replaced TritonOverrides._can_use_4x_random or added the cache key tag
Gates (non-vacuity): venv on HIP gfx1151 with torch 2.11; the 4x path is absent on this torch (else this is not
the gate lane); each state imported its own unsloth_zoo, in the probe and in the import-state cell; pytest ran
and collected in both; the head ran tests/test_inductor_rand4x.py tests (not all skipped); head has the patch
module and base does not.
"""

from __future__ import annotations

TITLE = "unsloth-zoo PR 2241 (rand4x off) on gfx1151 torch 2.11: patch stays out, base vs head"
MODE = "regression"
NEEDS = ["rocm", "gpu", "nvidia", "exclusive_gpu"]

_RAN = (0, 1)
NEW_TEST = "test_inductor_rand4x"


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
        out.append((f"{name} venv on HIP gfx1151, torch 2.11",
                    not v.get("error") and bool(env.get("hip")) and str(env.get("arch") or "").startswith("gfx1151")
                    and env.get("cuda_available") is True and _vt(env.get("torch")) == (2, 11),
                    f"{v.get('error') or ''} {env} steps {v.get('steps')} {str(v.get('log_tail') or '')[-400:]}"))
        ck = str(o.get("checkout") or "<none>")
        st = (o.get("import_state") or {}).get("state") or {}
        f, f2 = str(o.get("unsloth_zoo_file") or ""), str(st.get("unsloth_zoo_file") or "")
        out.append((f"{name} imported its own unsloth_zoo (probe and cell)", f.startswith(ck) and f2.startswith(ck) and bool(f),
                    f"{f or o.get('unsloth_zoo_import_error')} | cell {f2 or (o.get('import_state') or {}).get('fatal')} "
                    f"{str((o.get('import_state') or {}).get('log_tail'))[-300:] if not f2 else ''}"))
        out.append((f"{name}: torch has no 4x path (gate lane)", st.get("has_4x_path_before_import") is False,
                    f"has_4x_path {st.get('has_4x_path_before_import')} torch {st.get('torch')}"))
        p = o.get("pytest") or {}
        n = (p.get("n_passed") or 0) + (p.get("n_failed") or 0) + (p.get("n_skipped") or 0)
        out.append((f"{name} pytest ran and collected", p.get("rc") in _RAN and n > 0,
                    f"rc={p.get('rc')} passed {p.get('n_passed')} failed {p.get('n_failed')} skipped {p.get('n_skipped')} "
                    f"per file {p.get('per_file')} {str(p.get('stderr_tail') or '')[-300:]}"))
    h, b = obs.get("head") or {}, obs.get("base") or {}
    hp = h.get("pytest") or {}
    ran = [t for t in (hp.get("passed") or []) + (hp.get("failed") or []) + (hp.get("errors") or []) if NEW_TEST in t]
    out.append(("head executed test_inductor_rand4x tests", len(ran) > 0,
                f"{len(ran)} executed: {[t.split('::')[-1] for t in ran]}"))
    out.append(("head has the patch module, base does not",
                h.get("has_patch_module") is True and b.get("has_patch_module") is False,
                f"head {h.get('has_patch_module')} base {b.get('has_patch_module')}"))
    return out


def head_is_worse(base, head):
    problems, notes = [], []
    for name, o in (("base", base), ("head", head)):
        st = ((o or {}).get("import_state") or {}).get("state") or {}
        line = f"{name}: patched {st.get('patched')} cache_key_tag {st.get('cache_key_tag')!r}"
        if name == "head" and (st.get("patched") or st.get("tag_present")):
            problems.append("PATCH INSTALLED ON TORCH 2.11 " + line)
        else:
            notes.append(line)
    bp, hp = (base or {}).get("pytest") or {}, (head or {}).get("pytest") or {}
    hbad = set(hp.get("failed") or []) | set(hp.get("errors") or [])
    bbad = set(bp.get("failed") or []) | set(bp.get("errors") or [])
    lost = sorted(set(bp.get("passed") or []) - set(hp.get("passed") or []))
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
    rows = ["| state | torch | 4x path | patched | cache key tag | pytest passed | failed | skipped | failing tests |",
            "|" + "---|" * 9]
    for name in ("base", "head"):
        o = obs.get(name) or {}
        st = (o.get("import_state") or {}).get("state") or {}
        p = o.get("pytest") or {}
        bad = sorted(set(p.get("failed") or []) | set(p.get("errors") or []))
        rows.append(f"| {name} | {st.get('torch')} | {st.get('has_4x_path_before_import')} | {st.get('patched')} | "
                    f"{st.get('cache_key_tag')!r} | {p.get('n_passed')} | {p.get('n_failed')} | {p.get('n_skipped')} | {', '.join(bad) or 'none'} |")
    absent = ((obs.get("base") or {}).get("pytest") or {}).get("absent_at_this_state") or []
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
