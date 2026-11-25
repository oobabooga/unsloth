#!/usr/bin/env python3
"""Criteria (unsloth-zoo PR 2765, block swap disk tier) on gfx1151, base (zoo main) vs head. Pairs with
probes/zoo2765_probe.py.

Regression mode. Head is WORSE than base when a test that passed at the base does not pass at the head, or any
test fails at the head and not at the base (tests/test_block_swap_disk.py is head-only: any failure counts).
Gates: venv on HIP gfx1151 with a working GPU; each state imported its own unsloth_zoo; pytest ran and collected
in both; the head EXECUTED the disk tier GPU tests (bitwise RAM vs disk training), not skipped them.
"""

from __future__ import annotations

TITLE = "unsloth-zoo PR 2765 (block swap disk tier) on gfx1151: base vs head"
MODE = "regression"
NEEDS = ["rocm", "gpu", "nvidia", "exclusive_gpu"]

_RAN = (0, 1)
GPU_TEST = "test_disk_tier_trains_bitwise_like_host_ram"


def gates(obs):
    out = []
    for name in ("base", "head"):
        o = obs.get(name) or {}
        v = o.get("venv") or {}
        env = v.get("env") or {}
        out.append((f"{name} venv on HIP gfx1151 with the GPU",
                    not v.get("error") and bool(env.get("hip")) and str(env.get("arch") or "").startswith("gfx1151")
                    and env.get("cuda_available") is True,
                    f"{v.get('error') or ''} {env} {str(v.get('log_tail') or '')[-300:]}"))
        ck = str(o.get("checkout") or "<none>")
        f = str(o.get("unsloth_zoo_file") or "")
        out.append((f"{name} imported its own unsloth_zoo", bool(f) and f.startswith(ck),
                    f or str(o.get("unsloth_zoo_import_error"))))
        p = o.get("pytest") or {}
        n = (p.get("n_passed") or 0) + (p.get("n_failed") or 0) + (p.get("n_skipped") or 0)
        out.append((f"{name} pytest ran and collected", p.get("rc") in _RAN and n > 0,
                    f"rc={p.get('rc')} passed {p.get('n_passed')} failed {p.get('n_failed')} skipped "
                    f"{p.get('n_skipped')} per file {p.get('per_file')} {str(p.get('stderr_tail') or '')[-300:]}"))
    hp = (obs.get("head") or {}).get("pytest") or {}
    ran = [t for t in (hp.get("passed") or []) + (hp.get("failed") or []) + (hp.get("errors") or []) if GPU_TEST in t]
    out.append(("head executed the disk tier GPU training tests", len(ran) >= 4,
                f"{len(ran)} executed: {[t.split('::')[-1] for t in ran]}"))
    return out


def table(obs):
    rows = ["| state | passed | failed | skipped | failing tests |", "|---|---|---|---|---|"]
    for name in ("base", "head", "merge"):
        p = (obs.get(name) or {}).get("pytest")
        if not p:
            continue
        bad = sorted(set(p.get("failed") or []) | set(p.get("errors") or []))
        rows.append(f"| {name} | {p.get('n_passed')} | {p.get('n_failed')} | {p.get('n_skipped')} | "
                    + (", ".join(f"`{t.split('::')[-1]}`" for t in bad) or "none") + " |")
    return "\n".join(rows)


def head_is_worse(base, head):
    bp, hp = (base or {}).get("pytest") or {}, (head or {}).get("pytest") or {}
    hbad = set(hp.get("failed") or []) | set(hp.get("errors") or [])
    bbad = set(bp.get("failed") or []) | set(bp.get("errors") or [])
    lost = sorted(set(bp.get("passed") or []) - set(hp.get("passed") or []))
    new = sorted(hbad - bbad)
    if lost or new:
        return True, "lost at head: " + ", ".join(t.split("::")[-1] for t in lost) + "; newly failing: " + \
            ", ".join(t.split("::")[-1] for t in new)
    note = "no test that passed at the base fails at the head"
    if hbad:
        note += "; failing at both states (pre-existing): " + ", ".join(t.split("::")[-1] for t in sorted(hbad))
    return False, note
