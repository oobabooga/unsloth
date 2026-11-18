#!/usr/bin/env python3
"""Criteria (unsloth #18287 + follow-up fix): FSDP2 replicated-table gather moves the chunk off the GPU; torchrun
ranks > 0 re-apply unsloth_zoo's allocator settings after CUDA/HIP started; follow-up reads the variables in torch's
order and keeps torch 2.10+'s FutureWarning out.

Differential. Base shows the defect when any flip test fails at the base (the GPU peak test
test_chunk_leaves_the_gpu_before_the_full_table where a GPU ran it; the CPU allocator tests, absent API at the base).
Head is fixed when no selected test fails or errors at the head, every `import unsloth` cell that imported at the base
imports at the head, and no head cell raised or emitted an allocator FutureWarning.
Gates: each state ran its own unsloth/parallel_env.py; the head carries the follow-up fix; pytest ran and collected.
"""
from __future__ import annotations

TITLE = "unsloth #18287: late allocator settings + FSDP2 gather off-GPU, base vs head (+ follow-up fix)"
MODE = "differential"
NEEDS = ["rocm", "gpu", "nvidia", "multi_gpu", "windows"]
FLIP = ("test_single_process_import_leaves_the_allocator_alone", "test_chunk_leaves_the_gpu_before_the_full_table", "test_late_allocator_env_is_applied_once_cuda_started",
        "test_late_allocator_env_reads_variables_in_torch_order", "test_deprecated_setter_neither_warns_nor_is_skipped")
CELLS = ("rank_gt0", "single", "rank_gt0_user_hip_expandable")


def _cases(o, key = "test"):
    return ((o or {}).get(key) or {}).get("cases") or {}


def _short(t):
    return t.split("::")[-1]


def gates(obs):
    out = []
    for name in ("base", "head"):
        o = obs.get(name) or {}
        m = o.get("module") or {}
        f, ck = str(m.get("file") or "").replace("\\", "/"), str(o.get("checkout") or "").replace("\\", "/")
        out.append((f"{name} ran its own parallel_env.py", bool(f) and bool(ck) and f.startswith(ck), f or str(m)[:300]))
        t = o.get("test") or {}
        out.append((f"{name} pytest ran and collected", t.get("rc") in (0, 1) and len(_cases(o)) > 0,
                    f"rc={t.get('rc')} cases={len(_cases(o))} {'' if t.get('rc') in (0, 1) else str(t.get('tail'))[-300:]}"))
    h = obs.get("head") or {}
    out.append(("head carries the follow-up fix", (h.get("module") or {}).get("has_fix") is True,
                str(h.get("fix_apply") or "already in the head")[:300]))
    return out


def base_shows_defect(base):
    return any(c.get("outcome") in ("failed", "error") for t, c in _cases(base).items() if _short(t).split("[")[0] in FLIP)


def _cell_problems(o, key = "cells"):
    bad = []
    for n in CELLS:
        c = ((o or {}).get(key) or {}).get(n)
        if not c:
            continue
        if c.get("error") or c.get("import_error") or c.get("rc") not in (0, None):
            bad.append(f"{n}: {(c.get('error') or c.get('import_error') or c.get('rc'))!s:.300}")
        if any("deprecated" in w for w in c.get("allocator_warnings") or []):
            bad.append(f"{n}: allocator FutureWarning {c['allocator_warnings'][:1]}")
    return bad


def head_is_fixed(head):
    bad = [t for t, c in _cases(head).items() if c.get("outcome") in ("failed", "error")]
    return not bad and not _cell_problems(head)


def table(obs):
    lines = []
    for name, key in (("base", "test"), ("head (PR as pushed)", "pr_test"), ("head (PR + follow-up)", "test")):
        o = obs.get("head" if name.startswith("head") else "base") or {}
        cs = _cases(o, key)
        if not cs:
            continue
        lines.append(f"- {name} `{str(o.get('commit'))[:10]}`: " + ", ".join(f"`{_short(t)}` {c.get('outcome')}" for t, c in sorted(cs.items())))
        for t, c in sorted(cs.items()):
            if c.get("outcome") in ("failed", "error"):
                lines.append(f"  - {c.get('outcome')} `{_short(t)}`: {' '.join(c.get('message', '').split())[:300]}")
    lines.append("")
    lines.append("| state | cell | torch | init before import | setter calls | allocator warnings | expandable | segs | import error |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for name, st, key in (("base", "base", "cells"), ("PR", "head", "pr_cells"), ("PR+fix", "head", "cells")):
        cc = (obs.get(st) or {}).get(key) or {}
        if not cc.get("zoo_importable"):
            lines.append(f"| {name} | (unsloth_zoo not importable: cells skipped) | | | | | | | |")
            continue
        for n in CELLS:
            c = cc.get(n) or {}
            lines.append(f"| {name} | {n} | {c.get('torch')} hip {c.get('hip')} | {c.get('initialized_before_import')} | "
                         f"{c.get('setter_calls')} | {str(c.get('allocator_warnings'))[:160]} | {c.get('expandable')} | "
                         f"{c.get('n_segments')} | {' '.join(str(c.get('import_error') or c.get('error') or '').split())[:200]} |")
    return "\n".join(lines)
