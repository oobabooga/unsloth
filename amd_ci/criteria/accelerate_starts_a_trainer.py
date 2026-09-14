#!/usr/bin/env python3
"""Criteria: does the accelerate this checkout selects survive trainer start?

PR #10819 caps `accelerate<1.15.0` on Windows because 1.15.0's unconditional
`model_has_dtensor` call reaches `torch._C._distributed_c10d`, which AMD's Windows ROCm
wheels do not ship. The base state has no cap and should select 1.15.0 and fail; the head
should select 1.14.0 and start.

The defect is a property of the torch build, not of the OS label. A Windows box carrying a
torch that DOES ship c10d cannot exhibit it, and on that host the honest answer is VOID
rather than a pass. That is why `base_shows_defect` insists on the specific
`_distributed_c10d` failure and not merely on "base was unhappy": a base that fails for
some other reason is a broken harness wearing a confirmation.

The gate on the two states choosing different versions is what stops this reading green on
a host where the diff changed nothing, which is every non-Windows host by design.

Pairs with probes/accelerate_dtensor_probe.py.
"""

from __future__ import annotations

TITLE = "accelerate selection and trainer start, base versus head"
MODE = "differential"
NEEDS = ["gpu", "rocm", "windows", "windows_rocm_wddm",
         "multi_gpu", "nvidia", "mig", "xpu", "mlx"]

_MARKER = "_distributed_c10d"


def _states(obs: dict) -> dict:
    return {k: v for k, v in obs.items() if not k.startswith("_")}


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    states = _states(obs)
    out: list[tuple[str, bool, str]] = []

    for name, v in states.items():
        ok = bool(v.get("torch")) and not v.get("torch_error")
        out.append((f"{name}: torch importable",
                    ok, v.get("torch") or v.get("torch_error") or "no torch"))
        out.append((f"{name}: constraints.txt resolved",
                    bool(v.get("selected_accelerate")),
                    v.get("selected_accelerate") or v.get("resolve_error") or "unresolved"))

    # Without a torch lacking c10d there is no defect on this host, whatever the OS says.
    lacking = [n for n, v in states.items() if v.get("has_c10d") is False]
    out.append(("this torch lacks torch._C._distributed_c10d",
                len(lacking) == len(states) and bool(states),
                "yes, on every state" if len(lacking) == len(states) and states
                else "no: this build ships a distributed backend, so it cannot show the defect"))

    base, head = states.get("base") or {}, states.get("head") or {}
    bv, hv = base.get("selected_accelerate"), head.get("selected_accelerate")
    out.append(("the diff changed the selected version", bool(bv and hv and bv != hv),
                f"base={bv} head={hv}"))
    return out


def table(obs: dict) -> str:
    rows = ["| state | constraint line | selected | torch | has c10d | prepare_model |",
            "|---|---|---|---|---|---|"]
    for name, v in _states(obs).items():
        if v.get("prepare_ok") is True:
            verdict = "started"
        elif v.get("prepare_ok") is False:
            verdict = f"`{str(v.get('prepare_error', 'failed'))[:90]}`"
        else:
            verdict = v.get("install_error") or v.get("resolve_error") or "not reached"
        rows.append(
            f"| {name} | `{v.get('constraint_line') or 'none'}` | "
            f"{v.get('selected_accelerate') or '-'} | {v.get('torch') or '-'} | "
            f"{v.get('has_c10d')} | {verdict} |"
        )
    return "\n".join(rows)


def base_shows_defect(base: dict) -> bool:
    if base.get("prepare_ok") is not False:
        return False
    return _MARKER in str(base.get("prepare_error", ""))


def head_is_fixed(head: dict) -> bool:
    return head.get("prepare_ok") is True
