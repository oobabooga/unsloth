#!/usr/bin/env python3
"""Criteria: the unsloth-zoo fused-linear GPU unit tests pass on gfx1151 at the head, and the GPU path really ran.

Pairs with probes/zoo_pytest_probe.py. Regression by test ID (a test added by the PR that fails at the head is a
regression: it is absent at the base). Non-vacuity gates, all must hold or the verdict is INCONCLUSIVE:
  - pytest exited 0/1 and collected tests at base and head
  - each state imported unsloth_zoo from its OWN checkout (not the venv's copy)
  - head: torch.cuda.is_available() and torch.version.hip set inside the pytest process, arch gfx1151
  - head: >= 1 cuda-parametrised test passed and >= 1 test launched the fused_linear Triton kernel
  - head: no device left in fused_linear._TRITON_BROKEN (the autouse fixture also fails the test that leaves one)
"""

from __future__ import annotations

TITLE = "unsloth-zoo fused-linear GPU unit tests on gfx1151, zoo base vs head"
MODE = "regression"
NEEDS = ["rocm", "gpu", "discrete_gpu", "windows", "nvidia", "multi_gpu", "mig", "xpu", "mlx"]
_RAN = (0, 1)


def _states(obs):
    # A state the probe was told to skip (merge, for time) is reported, not judged.
    return {n: v for n, v in obs.items() if not n.startswith("_") and not (v or {}).get("skipped")}


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out = []
    for name, o in _states(obs).items():
        rc = o.get("rc")
        n = (o.get("n_passed") or 0) + (o.get("n_failed") or 0) + (o.get("n_skipped") or 0) + (o.get("n_errors") or 0)
        out.append((f"{name}: pytest ran and collected", rc in _RAN and n > 0,
                    f"rc={rc}, {n} outcomes; {o.get('error') or o.get('note') or ''} {str(o.get('stderr_tail'))[-200:]}"))
        f, co = o.get("unsloth_zoo_file") or "", o.get("checkout") or "?"
        out.append((f"{name}: unsloth_zoo imported from its own checkout", bool(f) and f.startswith(co),
                    f"{f} (checkout {co}, zoo {str(o.get('zoo_sha'))[:9]})"))
    head = obs.get("head") or {}
    t = head.get("torch") or {}
    out.append(("head: GPU visible in the pytest process (HIP)",
                bool(t.get("is_available")) and bool(t.get("hip")) and "gfx1151" in str(t.get("arch")),
                f"is_available={t.get('is_available')} hip={t.get('hip')} arch={t.get('arch')} "
                f"name={t.get('device_name')} torch={t.get('version')} triton={t.get('triton')} {t.get('error') or ''}"))
    cp = head.get("cuda_passed") or []
    out.append(("head: cuda-parametrised tests passed", len(cp) > 0,
                f"{len(cp)} passed, {len(head.get('cuda_skipped') or [])} skipped"))
    tt = head.get("triton_tests") or []
    out.append(("head: fused_linear Triton kernel launched", len(tt) > 0,
                f"{len(tt)} tests incremented CALLS['triton']; CALLS at end {head.get('calls')}"))
    tb = head.get("triton_broken")
    out.append(("head: no device left in _TRITON_BROKEN", tb is not None and not tb, f"{tb}"))
    return out


def table(obs: dict) -> str:
    rows = ["| state | zoo | passed | failed | errors | skipped | cuda passed | Triton tests | failing |",
            "|---|---|---|---|---|---|---|---|---|"]
    for name, o in _states(obs).items():
        bad = ", ".join(f"`{x.split('::')[-1]}`" for x in (o.get("failed") or []) + (o.get("errors") or [])) or "none"
        rows.append(f"| {name} | {str(o.get('zoo_sha'))[:9]} | {o.get('n_passed', 0)} | {o.get('n_failed', 0)} | "
                    f"{o.get('n_errors', 0)} | {o.get('n_skipped', 0)} | {len(o.get('cuda_passed') or [])} | "
                    f"{len(o.get('triton_tests') or [])} | {bad} |")
    absent = (obs.get("base") or {}).get("absent_at_this_state") or []
    note = ("\n\nAbsent at the base (added by the PR), so their head result is judged on its own: "
            + ", ".join(f"`{a}`" for a in absent)) if absent else ""
    return "\n".join(rows) + note


def head_is_worse(base: dict, head: dict) -> tuple[bool, str]:
    b = set(base.get("failed") or []) | set(base.get("errors") or [])
    h = set(head.get("failed") or []) | set(head.get("errors") or [])
    new = sorted(h - b)
    if new:
        return True, "failing at the head and not at the base: " + ", ".join(f"`{n}`" for n in new)
    detail = f"head {head.get('n_passed')} passed, 0 new failures"
    if h:
        detail += "; failing at BOTH states (pre-existing): " + ", ".join(f"`{n}`" for n in sorted(h))
    return False, detail
