#!/usr/bin/env python3
"""Criteria for PR 9829: torch installed from an index that has kernels for the GPU.

The defect: a Linux host whose runtime target is absent from the generic
pytorch.org ROCm wheel (gfx1103, gfx1031-gfx1036) gets that wheel anyway. Torch
imports, reports the GPU as available, and faults on the first GPU operation.

Expressed here as: the head's routing tests, run against the base source, fail.
Judges, never observes. Pairs with probes/spec_backport_pytest_probe.py.

`base_shows_defect` deliberately requires failures in the ROUTING module, not
merely somewhere in the selection. Any three-file backport onto older source will
produce some failure; only a failure in the module that encodes the missing-kernel
routing is the defect this PR is about.
"""

from __future__ import annotations

TITLE = "PR 9829: missing-kernel ROCm routing, head's tests against each state"
MODE = "differential"

# Authored, not computed. Every capability the CHANGE touches, so the report
# bounds itself even where this host happens to satisfy the ones it uses.
# The change routes Windows arch indexes, mixed discrete+integrated hosts and
# multi-GPU visible-device selection, none of which this single integrated
# gfx1151 APU on Linux can exercise for real.
NEEDS = [
    "linux", "rocm", "gpu", "integrated_gpu",
    "windows", "multi_gpu", "multi_gpu_amd", "discrete_gpu",
]

# The module carrying the missing-kernel routing specification.
DEFECT_MODULE = "test_missing_kernel_arch_routing_9396.py"

# Failing ids seen at the base, recorded by gates() so head_is_fixed can tell a
# failure this change caused from one that was already there. differential.py
# calls gates() before it decides, and hands only ONE state to head_is_fixed, so
# there is no other way for the head's judgement to see the base. If the hook
# never ran, the fallback below is the strict reading, never the lenient one.
_BASE_FAILURES: "set[str] | None" = None


def _ids(o: dict) -> set[str]:
    return set(o.get("failed", [])) | set(o.get("errors", []))


def _defect_ids(o: dict) -> set[str]:
    return {i for i in _ids(o) if DEFECT_MODULE in i}


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    global _BASE_FAILURES
    _BASE_FAILURES = _ids(obs.get("base") or {})

    out: list[tuple[str, bool, str]] = []
    for name in ("base", "head"):
        o = obs.get(name) or {}
        out.append((f"{name} suite actually ran", o.get("rc") is not None,
                    o.get("error") or o.get("note") or f"rc={o.get('rc')}"))

    base = obs.get("base") or {}
    head = obs.get("head") or {}

    # Non-vacuity: the base must really have been handed the head's tests. If the
    # backport copied nothing, a base failure would be about something else and a
    # base pass would be about the base's own (absent) tests.
    applied = base.get("spec_files") or []
    changed = [f for f in applied if f.get("action") in ("ADDED", "OVERWRITTEN")]
    out.append(("head's tests were backported onto the base",
                bool(changed) and not [f for f in applied if f.get("action") == "MISSING_AT_SPEC_STATE"],
                ", ".join(f"{f['path'].split('/')[-1]}={f['action']}" for f in applied) or "none"))

    # Non-vacuity: the routing module must have been among them, and must have
    # actually collected tests at the base. A base that collected nothing cannot
    # exhibit anything.
    out.append((f"{DEFECT_MODULE} present at the base",
                any(DEFECT_MODULE in (f.get("path") or "") and
                    f.get("action") in ("ADDED", "OVERWRITTEN") for f in applied),
                str([f.get("action") for f in applied
                     if DEFECT_MODULE in (f.get("path") or "")]) or "absent"))

    out.append(("base collected tests", (base.get("n_collected") or 0) > 0,
                f"{base.get('n_collected', 0)} collected, rc={base.get('rc')}"))
    out.append(("head collected tests", (head.get("n_collected") or 0) > 0,
                f"{head.get('n_collected', 0)} collected, rc={head.get('rc')}"))
    return out


def table(obs: dict) -> str:
    rows = ["| state | passed | failed | errors | skipped | failing in the routing module |",
            "|---|---|---|---|---|---|"]
    for name in ("base", "head", "merge"):
        o = obs.get(name)
        if not o:
            continue
        d = sorted(_defect_ids(o))
        shown = ", ".join(f"`{i.split('::')[-1]}`" for i in d[:6])
        if len(d) > 6:
            shown += f", and {len(d) - 6} more"
        rows.append(
            f"| {name} | {o.get('n_passed', 0)} | {o.get('n_failed', 0)} "
            f"| {o.get('n_errors', 0)} | {o.get('n_skipped', 0)} | {shown or 'none'} |")

    note = [""]
    base = obs.get("base") or {}
    other = sorted(_ids(base) - _defect_ids(base))
    if other:
        note.append("Also failing at the base, in the other backported modules: "
                    + ", ".join(f"`{i.split('::')[-1]}`" for i in other[:10])
                    + (f", and {len(other) - 10} more" if len(other) > 10 else ""))
    note.append("")
    base = obs.get("base") or {}
    head = obs.get("head") or {}
    note.append(f"Backported into each state: "
                f"{', '.join(f['path'].split('/')[-1] for f in base.get('spec_files') or []) or 'none'}. "
                f"Run at each state: {', '.join(base.get('selected') or []) or 'none'}. The wider "
                f"selection is deliberate: `head_is_fixed` demands zero failures in the routing "
                f"module and, everywhere else, no failure the base does not also show, so the "
                f"regression reading and the fix reading come from one leg.")
    note.append("")
    note.append("The comparison runs the HEAD's test files at every state, so the base is "
                "measured against the specification this change introduces. Hardware is "
                "mocked inside those tests; what this host contributes is a real ROCm "
                "Python environment on real AMD hardware, not a real gfx1103.")
    if head:
        note.append("")
        note.append(f"Head selection: {head.get('n_passed', 0)} passed, "
                    f"{head.get('n_failed', 0)} failed, {head.get('n_errors', 0)} errors.")
        both = sorted(_ids(head) & _ids(base))
        new = sorted(_ids(head) - _ids(base))
        if both:
            note.append("")
            note.append("Failing at BOTH states, so pre-dating this change: "
                        + ", ".join(f"`{i.split('::')[-1]}`" for i in both[:10])
                        + (f", and {len(both) - 10} more" if len(both) > 10 else ""))
        if new:
            note.append("")
            note.append("**Failing only at the head**: "
                        + ", ".join(f"`{i.split('::')[-1]}`" for i in new))
    return "\n".join(rows + note)


def base_shows_defect(base: dict) -> bool:
    return bool(_defect_ids(base))


def head_is_fixed(head: dict) -> bool:
    if (head.get("n_collected") or 0) <= 0:
        return False
    failures = _ids(head)
    # Nothing in the module the defect lives in, ever: those tests exist at both
    # states here, so a failure among them is always this change's.
    if any(DEFECT_MODULE in i for i in failures):
        return False
    # Elsewhere in the widened selection, only failures the base does not also
    # show. A suite whose environment already fails a test would otherwise make
    # every head look unfixed, which is a reason to narrow the selection rather
    # than a finding about the change.
    if _BASE_FAILURES is None:
        return not failures
    return not (failures - _BASE_FAILURES)
