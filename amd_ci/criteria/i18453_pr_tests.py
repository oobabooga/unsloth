#!/usr/bin/env python3
"""Criteria (unsloth #18453, Windows test fix): only the PR's tests/utils/test_rocm_sdpa_check.py, base vs head.
REGRESSION mode. The file is new in the PR, so base has none; head is worse if pytest did not collect it, any case
failed or errored, or nothing passed. Gates: both states imported unsloth from a ROCm torch on gfx1151 (head's
self-check child ran on the GPU), head has the test file and base does not.
"""
from __future__ import annotations

TITLE = "unsloth #18453: tests/utils/test_rocm_sdpa_check.py on gfx1151 (Windows test fix), base vs head"
MODE = "regression"
NEEDS = ["rocm", "gpu", "nvidia"]


def gates(obs):
    b, h = obs.get("base") or {}, obs.get("head") or {}
    out = []
    for name, o in (("base", b), ("head", h)):
        out.append((f"{name}: unsloth importable from the job venv", o.get("unsloth_importable") is True,
                    str(o.get("unsloth_import_tail"))[-300:]))
    out.append(("head has rocm_sdpa_check.py, base does not",
                (h.get("check_child") or {}).get("present") is True and (b.get("check_child") or {}).get("present") is False,
                f"head={(h.get('check_child') or {}).get('present')} base={(b.get('check_child') or {}).get('present')}"))
    cc = h.get("check_child") or {}
    ok = all(((cc.get(k) or {}).get("report") or {}).get("ran") for k in ("flash", "efficient"))
    out.append(("head: self-check child ran cases on the GPU (ROCm torch present)", ok,
                str({k: (cc.get(k) or {}).get("report") for k in ("flash", "efficient")})[:400]))
    return out


def _cases(o):
    return ((o or {}).get("pr_tests") or {}).get("cases") or {}


def head_is_worse(base, head):
    t = head.get("pr_tests") or {}
    c = _cases(head)
    failed = sorted(k for k, v in c.items() if v == "failed")
    passed = sum(1 for v in c.values() if v == "passed")
    if not c or failed or not passed or t.get("rc") != 0:
        return True, f"head PR tests: rc={t.get('rc')} passed={passed} failed={failed} {t.get('junit_error', '')}"
    return False, f"head PR tests: {passed} passed, 0 failed (base has no such file)"


def table(obs):
    L = ["| state | file present | rc | passed | failed | skipped | failing / skipped ids |", "|---|---|---|---|---|---|---|"]
    for name in ("base", "head"):
        o = obs.get(name) or {}
        t, c = o.get("pr_tests") or {}, _cases(o)
        n = {k: sum(1 for v in c.values() if v == k) for k in ("passed", "failed", "skipped")}
        ids = sorted(k.split("::")[-1] for k, v in c.items() if v != "passed")
        L.append(f"| {name} | {(o.get('check_child') or {}).get('present')} | {t.get('rc')} | {n['passed']} | "
                 f"{n['failed']} | {n['skipped']} | {ids} |")
    h = obs.get("head") or {}
    L.append(f"\nhead commit {h.get('commit')}; base commit {(obs.get('base') or {}).get('commit')}; platform "
             f"{h.get('platform')}; pip {h.get('pip')}")
    L.append(f"\nhead pytest tail:\n```\n{str((h.get('pr_tests') or {}).get('tail'))[-1500:]}\n```")
    return "\n".join(L)
