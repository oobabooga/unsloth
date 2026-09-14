#!/usr/bin/env python3
"""Criteria: does PR #10819 leave a non-Windows host exactly as it found it?

The cap is `accelerate<1.15.0; sys_platform == "win32"`, and the repair step it pairs
with returns early unless `IS_WINDOWS`. Both are therefore supposed to be inert here, and
"inert" is a claim worth measuring on the platform that carries the load rather than
asserting from the marker text.

Regression mode, not differential, and deliberately so. On Linux the base does not exhibit
the defect and never should, so a differential could only ever return VOID; asking "is the
head worse" is the question that actually has an answer off Windows.

Two ways to be worse, and the version one matters more than it looks: a marker that leaked
would silently downgrade every Linux and macOS user to accelerate 1.14.0, which is a real
regression that no error message would ever report.

Pairs with probes/accelerate_dtensor_probe.py.
"""

from __future__ import annotations

TITLE = "accelerate off Windows: the cap must change nothing"
MODE = "regression"
NEEDS = ["gpu", "rocm", "linux", "multi_gpu", "nvidia", "mig", "xpu", "mlx"]


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out: list[tuple[str, bool, str]] = []
    for name, v in obs.items():
        if name.startswith("_"):
            continue
        out.append((f"{name}: torch importable",
                    bool(v.get("torch")) and not v.get("torch_error"),
                    v.get("torch") or v.get("torch_error") or "no torch"))
        out.append((f"{name}: constraints.txt resolved",
                    bool(v.get("selected_accelerate")),
                    v.get("selected_accelerate") or v.get("resolve_error") or "unresolved"))
    # Non-vacuity: off Windows this torch is expected to HAVE c10d. If it did not, the
    # host would be a Windows-ROCm-alike and this criteria would be the wrong question.
    states = [v for k, v in obs.items() if not k.startswith("_")]
    have = [v for v in states if v.get("has_c10d") is True]
    out.append(("this torch ships torch._C._distributed_c10d",
                len(have) == len(states) and bool(states),
                "yes, so the Windows defect cannot apply here"
                if len(have) == len(states) and states
                else "no: this host looks like the Windows ROCm case, so ask that question instead"))
    return out


def table(obs: dict) -> str:
    rows = ["| state | constraint line | selected | torch | has c10d | prepare_model |",
            "|---|---|---|---|---|---|"]
    for name, v in obs.items():
        if name.startswith("_"):
            continue
        verdict = ("started" if v.get("prepare_ok") is True
                   else f"`{str(v.get('prepare_error', 'not reached'))[:80]}`")
        rows.append(
            f"| {name} | `{v.get('constraint_line') or 'none'}` | "
            f"{v.get('selected_accelerate') or '-'} | {v.get('torch') or '-'} | "
            f"{v.get('has_c10d')} | {verdict} |"
        )
    return "\n".join(rows)


def head_is_worse(base: dict, head: dict) -> tuple[bool, str]:
    bv, hv = base.get("selected_accelerate"), head.get("selected_accelerate")
    if bv != hv:
        return True, (f"the cap leaked off Windows: base selects {bv}, head selects {hv}. "
                      "The marker is sys_platform == \"win32\" and this host is not Windows")
    if base.get("prepare_ok") is True and head.get("prepare_ok") is not True:
        return True, (f"prepare_model started at the base and not at the head: "
                      f"{head.get('prepare_error', 'no reason recorded')}")
    return False, f"both states select accelerate {hv} and start a trainer"
