#!/usr/bin/env python3
"""Criteria (unsloth #18285): NF4 dequant cache for QLoRA, gfx1151, base vs head.

Regression mode (training-path rule: head must be no worse than the merge base on this platform). Head is WORSE when:
  * tests/test_nf4_dequant_cache.py (head-only, absent at the base by construction: that is not a regression and
    not VOID) has any failed / errored test at the head
  * a head SFT cell (auto or UNSLOTH_NF4_DEQUANT_CACHE=0) did not finish while the base finished, or has
    non-finite LoRA weights
  * a head cell's per-step loss differs from base run1 by more than tol, tol = max(3 x the largest per-step
    |run1 - run2| at the base, 1e-6 x |loss|) (step 1 included: the cache admits nothing before the first optimizer
    step, so step 1 runs the same code as the base)
  * head auto: the cache neither engaged (admits > 0 and hits > 0) nor stayed cleanly off (admits 0, hits 0, not
    disabled by an OOM); a cache_mode() of None under the default env is also reported
  * head off (=0): anything was admitted / hit (kill switch ignored)
Gates (non-vacuity): both states ran their own checkout's unsloth; head carries the dequant_cache module and the
base does not; head pytest collected tests; base run1 finished on the gfx1151 GPU with Linear4bit layers. If
bitsandbytes NF4 cannot run at the base the gate fails (INCONCLUSIVE): a host gap, not a PR result.
"""
from __future__ import annotations

TITLE = "unsloth #18285: NF4 dequant cache (QLoRA) on gfx1151, base vs head (unit tests + tiny 4-bit LoRA SFT)"
MODE = "regression"
NEEDS = ["rocm", "gpu", "nvidia", "multi_gpu", "windows"]

BASE_CELLS = ("run1", "run2")
EXPECT_BASE = "40cbeb7c5f03b559e564b2d41f514e67388a8dc9"
EXPECT_HEAD_TREE = "55809c154f9d9731e43e2ebe9a2193c81e955e5d"  # tree of PR head 35ceab15c43c2b6236db72901ea449b5daaca467
EXPECT_TESTS = 22
HEAD_CELLS = ("auto", "off")


def _sft(o, k):
    return (((o or {}).get("sft") or {}).get(k)) or {}


def _cases(o):
    return ((o or {}).get("test") or {}).get("cases") or {}


def _ok(c):
    return c.get("stage") == "done" and c.get("finite_lora") is True and bool(c.get("losses"))


def _ls(c):
    return {int(k): float(v) for k, v in (c.get("losses") or {}).items()}


def _spread(base):
    a, b = _ls(_sft(base, "run1")), _ls(_sft(base, "run2"))
    if not a or not b:
        return None
    return max(abs(a[s] - b[s]) for s in a if s in b) if set(a) & set(b) else None


def _tol(base, loss):
    sp = _spread(base)
    return max(3 * (sp or 0.0), 1e-6 * abs(loss))


def _own(o, c):
    f = str(c.get("unsloth_file") or "").replace("\\", "/")
    ck = str((o or {}).get("checkout") or "").replace("\\", "/")
    return bool(f) and bool(ck) and f.startswith(ck)


def gates(obs):
    out = []
    b, h = obs.get("base") or {}, obs.get("head") or {}
    out.append(("base is the merge base 40cbeb7c5f", str(b.get("commit")) == EXPECT_BASE, f"base={b.get('commit')}"))
    out.append(("head tree is PR head 35ceab15c4's tree", str(h.get("tree")) == EXPECT_HEAD_TREE,
                f"head commit={h.get('commit')} tree={h.get('tree')}"))
    out.append((f"head pytest collected all {EXPECT_TESTS} tests", len(_cases(h)) == EXPECT_TESTS, f"cases={len(_cases(h))}"))
    hf = b.get("test_headfile") or {}
    out.append(("head's test file ran against the base checkout", hf.get("rc") not in (None, "timeout") and len(hf.get("cases") or {}) > 0,
                f"rc={hf.get('rc')} cases={len(hf.get('cases') or {})} {str(hf.get('stderr_tail', ''))[-200:]}"))
    for name, o, k in (("base", b, "run1"), ("head", h, "auto")):
        c = _sft(o, k)
        out.append((f"{name} {k} imported its own checkout's unsloth", _own(o, c),
                    f"{c.get('unsloth_file')} stage={c.get('stage')} {str(c.get('error') or '')[:200]}"))
    out.append(("head carries unsloth/kernels/dequant_cache.py, base does not",
                h.get("dc_file_present") is True and b.get("dc_file_present") is False,
                f"head={h.get('dc_file_present')} base={b.get('dc_file_present')}"))
    t = h.get("test") or {}
    out.append(("head pytest ran and collected", t.get("present") is True and t.get("rc") in (0, 1) and len(_cases(h)) > 0,
                f"present={t.get('present')} rc={t.get('rc')} cases={len(_cases(h))} {str(t.get('stderr_tail', ''))[-200:]}"))
    c = _sft(b, "run1")
    v = c.get("versions") or {}
    out.append(("base run1 4-bit LoRA SFT finished on the gfx1151 GPU (bitsandbytes NF4 works on this host)",
                _ok(c) and str(c.get("model_device", "")).startswith("cuda") and "gfx1151" in str(v.get("arch"))
                and (c.get("n_linear4bit") or 0) > 0,
                f"stage={c.get('stage')} err={str(c.get('error') or '')[:300]} device={c.get('model_device')} arch={v.get('arch')} "
                f"linear4bit={c.get('n_linear4bit')} bnb={v.get('bitsandbytes')} nf4={c.get('bnb_nf4_roundtrip')}"[:700]))
    return out


def head_is_worse(base, head):
    bad, notes = [], []
    for tid, c in _cases(head).items():
        if c.get("outcome") in ("failed", "error"):
            bad.append(f"head test `{tid.split('::')[-1]}` {c.get('outcome')}: {' '.join(c.get('message', '').split())[:200]}")
    ref = _sft(base, "run1")
    for k in HEAD_CELLS:
        hc = _sft(head, k)
        if not _ok(hc):
            if _ok(ref):
                bad.append(f"head {k}: base finished, head {hc.get('stage')} {str(hc.get('error') or '')[:300]}")
            continue
        if _ok(ref):
            rl, hl = _ls(ref), _ls(hc)
            if set(rl) != set(hl):
                bad.append(f"head {k}: logged steps {sorted(hl)} != base {sorted(rl)}")
            for s in sorted(set(rl) & set(hl)):
                d = abs(hl[s] - rl[s])
                if d > _tol(base, rl[s]):
                    bad.append(f"head {k} step {s}: loss {hl[s]!r} vs base {rl[s]!r} (|d| {d:.3g} > tol {_tol(base, rl[s]):.3g})")
        st = hc.get("dc_stats") or {}
        adm, hits = st.get("admits", 0) or 0, st.get("hits", 0) or 0
        if k == "auto":
            if hc.get("dc_mode") is None:
                notes.append("head auto: cache_mode() is None under the default env")
            engaged = adm > 0 and hits > 0
            clean_off = adm == 0 and hits == 0 and not hc.get("dc_disabled") and not st.get("oom_clears")
            if not (engaged or clean_off):
                bad.append(f"head auto: cache neither engaged nor cleanly off (stats {st}, disabled {hc.get('dc_disabled')})")
            notes.append("head auto: cache ENGAGED" if engaged else "head auto: cache stayed OFF (no admits)")
        else:
            if hc.get("dc_mode") is not None or adm or hits:
                bad.append(f"head off (=0): mode {hc.get('dc_mode')!r}, admits {adm}, hits {hits}: kill switch not honoured")
    hf = ((base or {}).get("test_headfile") or {}).get("cases") or {}
    passed_at_base = sorted(t.split("::")[-1] for t, c in hf.items() if c.get("outcome") == "passed")
    notes.append(f"head test file at base: {len(hf)} junit cases, passed {len(passed_at_base)} {passed_at_base}"
                 f" (new tests must fail on base)")
    if bad:
        return True, "; ".join(bad + notes)
    return False, ("head tests pass, both head cells finished with losses within the base run-to-run tolerance; "
                   + "; ".join(notes))


def _fmt(c):
    if not _ok(c):
        return f"{c.get('stage')}: {' '.join(str(c.get('error') or '').split())[:200]}"
    ls = _ls(c)
    gn = {int(k): v for k, v in (c.get("grad_norms") or {}).items()}
    return (" ".join(f"{s}:{ls[s]!r}" for s in sorted(ls)) + " <br>gn " + " ".join(f"{gn[s]:.5g}" for s in sorted(gn))
            + f" <br>maxalloc {c.get('max_memory_allocated')} bf16={c.get('bf16')} mode={c.get('dc_mode')!r} "
            + f"stats={c.get('dc_stats')} cached={c.get('dc_cached_count')} budget={c.get('dc_budget')} "
            + f"disabled={c.get('dc_disabled')} foreign={c.get('dc_foreign')} meminfo={c.get('mem_get_info_end')}")


def table(obs):
    rows = ["| state | commit | versions | cell | result |", "|---|---|---|---|---|"]
    for name in ("base", "head", "merge"):
        o = obs.get(name)
        if not o:
            continue
        for k, c in (o.get("sft") or {}).items():
            v = c.get("versions") or {}
            ver = f"torch {v.get('torch')} hip {v.get('hip')} {v.get('arch')} bnb {v.get('bitsandbytes')} nf4 {c.get('bnb_nf4_roundtrip')}"
            rows.append(f"| {name} | {str(o.get('commit'))[:10]} | {ver} | {k} | {_fmt(c).replace('|', '/')} |")
    lines = ["\n".join(rows), "", f"Base run-to-run max per-step loss spread: {_spread(obs.get('base') or {})}", "",
             "Per-test ids:"]
    for name in ("base", "head", "merge"):
        o = obs.get(name)
        if not o:
            continue
        t = o.get("test") or {}
        if not t.get("present"):
            lines.append(f"- {name}: {t.get('file')} absent at this state (not run)")
            continue
        lines.append(f"- {name} (rc {t.get('rc')}): " + ", ".join(f"`{tid.split('::')[-1]}` {c.get('outcome')}"
                                                              for tid, c in sorted(_cases(o).items())))
        for tid, c in sorted(_cases(o).items()):
            if c.get("outcome") in ("failed", "error", "skipped"):
                lines.append(f"  - {name} {c.get('outcome')} `{tid}`: {' '.join(c.get('message', '').split())[:300]}")
    hf = (obs.get("base") or {}).get("test_headfile") or {}
    if hf:
        lines.append(f"- base running HEAD's test file (rc {hf.get('rc')}): " + ", ".join(
            f"`{tid}` {c.get('outcome')}" for tid, c in sorted((hf.get("cases") or {}).items())))
        for tid, c in sorted((hf.get("cases") or {}).items()):
            if c.get("outcome") != "passed":
                lines.append(f"  - base(headfile) {c.get('outcome')} `{tid}`: {' '.join(c.get('message', '').split())[:300]}")
    for name in ("base", "head", "merge"):
        o = obs.get(name) or {}
        lines.append(f"- {name} commit {o.get('commit')} tree {o.get('tree')}")
    for name in ("base", "head"):
        o = obs.get(name) or {}
        lines.append(f"- {name} pip: {o.get('pip')} preinstalled={o.get('stack_preinstalled')} "
                     f"install_rcs={[s.get('rc') for s in (o.get('stack_install') or [])]}")
    return "\n".join(lines)
