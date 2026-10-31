#!/usr/bin/env python3
"""Criteria (T7h.6q): zoo #2273 on the gfx1151 Windows 11 runner. The GPT-OSS decode tests' CPU subprocess checks
need CPU Inductor, i.e. a C++ compiler (cl on Windows); with none on PATH they fail at the base (zoo main = the PR's
merge base) with "Compiler: cl is not found.", and the head skips them up front via tests/_cpu_compiler_skip.py.
Pairs with probes/zoo_pytest_probe.py.

Gates (INCONCLUSIVE when any fails): pytest ran and collected at both states; each state imported unsloth_zoo from its
own checkout; a HIP gfx1151 GPU was visible inside pytest at the head.
Defect (base): at least one selected test fails or errors.
Fixed (head): none fail or error, >= 1 passed, and >= 1 skipped with the probe's missing-compiler reason.
"""

from __future__ import annotations

import os

TITLE = "T7h.6q: zoo #2273 GPT-OSS decode tests without a C++ compiler on Windows gfx1151, base (main) vs head"
MODE = "differential"
NEEDS = ["rocm", "gpu", "discrete_gpu", "nvidia", "multi_gpu", "mig", "xpu", "mlx"]
_RAN = (0, 1)
SKIP_TEXT = "no C++ compiler"


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
                    f"rc={rc}, {n} outcomes, tests {o.get('tests')} {str(o.get('stderr_tail'))[-200:]}"))
        f, co = o.get("unsloth_zoo_file") or "", o.get("checkout") or ""
        out.append((f"{name}: unsloth_zoo imported from its own checkout", bool(f) and _norm(f).startswith(_norm(co)),
                    f"{f} (checkout {co}, zoo {str(o.get('zoo_sha'))[:9]})"))
    t = (obs.get("head") or {}).get("torch") or {}
    out.append(("head: HIP gfx1151 GPU visible inside pytest",
                bool(t.get("is_available")) and bool(t.get("hip")) and "gfx1151" in str(t.get("arch")),
                f"is_available={t.get('is_available')} hip={t.get('hip')} arch={t.get('arch')} torch={t.get('version')}"))
    return out


def base_shows_defect(base: dict):
    b = _bad(base)
    msgs = base.get("fail_msgs") or {}
    cl = sorted(k.split("::")[-1] for k, m in msgs.items() if "is not found" in m)
    return bool(b), (f"{len(b)} failing at the base: " + ", ".join(x.split("::")[-1] for x in b)
                     + f" | with 'Compiler: ... is not found.' in the failure: {len(cl)} {cl}")


def head_is_fixed(head: dict):
    h = _bad(head)
    sk = {k: v for k, v in (head.get("skip_reasons") or {}).items() if SKIP_TEXT in v}
    ok = not h and (head.get("n_passed") or 0) > 0 and bool(sk)
    return ok, (f"head {head.get('n_passed')} passed, {head.get('n_skipped')} skipped "
                f"({len(sk)} with the missing-compiler reason: {sorted(x.split('::')[-1] for x in sk)[:12]}), failing: {h}")
