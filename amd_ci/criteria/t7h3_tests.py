#!/usr/bin/env python3
"""Criteria (T7h.3): zoo #2255 / #2256 on real gfx1151. tests/test_gpt_oss_rope_decode.py and
tests/test_gpt_oss_decode_fused.py fail at the base (zoo 4a74d204, before #2255) because the ROCm build switches the
fused decode module off in the tests' subprocesses, and pass at the head (base + the two PRs' test diffs = b6413947).
Pairs with probes/zoo_pytest_probe.py.

Gates (INCONCLUSIVE when any fails): pytest ran and collected at both states; each state imported unsloth_zoo from its
own checkout; a HIP gfx1151 GPU was visible inside pytest at the head.
Defect (base): at least one of the selected tests fails or errors. Fixed (head): none fail or error and >= 1 passed.
"""

from __future__ import annotations

import os

TITLE = "T7h.3: zoo #2255 / #2256 GPT-OSS decode tests on real gfx1151, base 4a74d204 vs head b6413947"
MODE = "differential"
NEEDS = ["rocm", "gpu", "discrete_gpu", "nvidia", "multi_gpu", "mig", "xpu", "mlx"]
_RAN = (0, 1)


def _norm(p):
    return os.path.normcase(os.path.abspath(p)) if p else ""


def _states(obs):
    return {n: v for n, v in obs.items() if not n.startswith("_") and not (v or {}).get("skipped")}


def _bad(o):
    return sorted(set(o.get("failed") or []) | set(o.get("errors") or []))


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out = []
    for name, o in _states(obs).items():
        rc = o.get("rc")
        n = sum((o.get(k) or 0) for k in ("n_passed", "n_failed", "n_skipped", "n_errors"))
        out.append((f"{name}: pytest ran and collected", rc in _RAN and n > 0,
                    f"rc={rc}, {n} outcomes {str(o.get('stderr_tail'))[-200:]}"))
        f, co = o.get("unsloth_zoo_file") or "", o.get("checkout") or ""
        out.append((f"{name}: unsloth_zoo imported from its own checkout", bool(f) and _norm(f).startswith(_norm(co)),
                    f"{f} (checkout {co}, zoo {str(o.get('zoo_sha'))[:9]})"))
    t = (obs.get("head") or {}).get("torch") or {}
    out.append(("head: HIP gfx1151 GPU visible inside pytest",
                bool(t.get("is_available")) and bool(t.get("hip")) and "gfx1151" in str(t.get("arch")),
                f"is_available={t.get('is_available')} hip={t.get('hip')} arch={t.get('arch')} torch={t.get('version')} "
                f"triton={t.get('triton')}"))
    return out


def base_shows_defect(base: dict):
    b = _bad(base)
    return bool(b), f"{len(b)} failing at the base: " + ", ".join(x.split("::")[-1] for x in b)


def head_is_fixed(head: dict):
    h = _bad(head)
    return (not h and (head.get("n_passed") or 0) > 0), f"head {head.get('n_passed')} passed, failing: {h}"


def table(obs: dict) -> str:
    rows = ["| state | zoo | passed | failed | errors | skipped | failing / erroring IDs |", "|---|---|---|---|---|---|---|"]
    for name, o in _states(obs).items():
        bad = ", ".join(f"`{x.split('::')[-1]}`" for x in _bad(o)) or "none"
        rows.append(f"| {name} | {str(o.get('zoo_sha'))[:9]} | {o.get('n_passed', 0)} | {o.get('n_failed', 0)} | "
                    f"{o.get('n_errors', 0)} | {o.get('n_skipped', 0)} | {bad} |")
    return "\n".join(rows)
