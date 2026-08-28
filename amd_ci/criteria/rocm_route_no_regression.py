#!/usr/bin/env python3
"""Criteria: does PR 9152 move any host it was not supposed to move?

Deliberately a regression check rather than a differential. The fix in 9152 does NOT
land in `get_torch_index_url` -- that function still returns the CPU index on the
Fedora shape at head, by design, because it DEFERS to a reroute in install.sh's
top-level code. Top-level code is not a function and no harness that sources functions
can reach it, so this run cannot see the fix and must not claim to.

(The end-to-end reroute is settled separately, by
tests/amd_pr_sims/sh/reroute_e2e.sh, which splices and executes that block.)

What this run is for is the other half of the question, and it is the half that
decides whether merging is safe: 9152 adds version SOURCES (rpm asked about
rocm-runtime and rocm-hip, hipconfig read from ROCM_PATH/bin) and memoises detection
to a file. Those are consulted on every Linux host. A regression here is any host
whose resolved index or resolved tag changes -- most importantly this runner's own
real ROCm stack, a working NVIDIA box with stale ROCm packaging, and a CPU-only box
with the same.
"""

from __future__ import annotations

TITLE = "ROCm wheel routing, unchanged for hosts outside the fix (PR 9152)"
MODE = "regression"

NEEDS = ["rocm", "gpu", "linux", "nvidia", "discrete_gpu", "integrated_gpu",
         "multi_gpu_amd", "windows", "xpu", "mlx"]


def _sc(state: dict, key: str) -> dict:
    return (state.get("scenarios") or {}).get(key) or {}


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out: list[tuple[str, bool, str]] = []
    states = {n: v for n, v in obs.items() if not n.startswith("_")}

    bad = {n: v.get("missing_required") for n, v in states.items()
           if v.get("missing_required")}
    out.append(("every required install.sh function was spliced", not bad,
                str(bad) if bad else "no required function missing in any state"))

    real = {n: _sc(v, "bash:real_host").get("index_url") for n, v in states.items()}
    out.append(("the unstubbed host resolved an index", all(bool(v) for v in real.values()),
                ", ".join(f"{k}={v}" for k, v in real.items())))

    b, h = obs.get("base") or {}, obs.get("head") or {}
    shape = (b.get("has_agreed_index_family") is False
             and h.get("has_agreed_index_family") is True)
    out.append(("base predates the arch-agreement helpers and head carries them", shape,
                f"base={b.get('has_agreed_index_family')} "
                f"head={h.get('has_agreed_index_family')}"))
    return out


def table(obs: dict) -> str:
    names = [n for n in obs if not n.startswith("_")]
    keys: list[str] = []
    for n in names:
        for k in (obs[n].get("scenarios") or {}):
            if k not in keys:
                keys.append(k)
    rows = ["| scenario | " + " | ".join(f"{n} tag / index" for n in names) + " |",
            "|---" * (1 + len(names)) + "|"]
    for k in keys:
        cells = []
        for n in names:
            s = _sc(obs[n], k)
            u = str(s.get("index_url") or "-").replace(
                "https://download.pytorch.org/whl", "pt")
            cells.append(f"`{s.get('rocm_tag') or '-'}` / `{u}`")
        rows.append(f"| {k} | " + " | ".join(cells) + " |")
    rows += [
        "",
        "**This run does not, and cannot, show 9152 fixing issue 8731.** The fix is in "
        "install.sh's top-level reroute, which is not a function and is therefore "
        "unreachable from any harness that sources functions -- including this one and "
        "including the PR's own `tests/sh` suite, which asserts the reroute by grepping "
        "install.sh for an identifier rather than by running it. What is scored here is "
        "only whether hosts OUTSIDE the fix kept their previous routing.",
    ]
    return "\n".join(rows)


def head_is_worse(base: dict, head: dict) -> tuple[bool, str]:
    problems: list[str] = []
    keys = set((base.get("scenarios") or {})) | set((head.get("scenarios") or {}))
    for k in sorted(keys):
        b, h = _sc(base, k), _sc(head, k)
        bu, hu = b.get("index_url"), h.get("index_url")
        if bu is None or hu is None:
            continue
        if bu != hu:
            problems.append(f"{k}: index moved from {bu!r} to {hu!r}")
        bt, ht = b.get("rocm_tag"), h.get("rocm_tag")
        if bt != ht:
            # A tag change without an index change is not user-visible, so it is
            # reported rather than scored -- appended to the detail either way.
            problems.append(f"{k}: resolved tag changed {bt!r} -> {ht!r}"
                            if bu != hu else "")
    problems = [p for p in problems if p]
    moved = [p for p in problems if "index moved" in p]
    if moved:
        return True, "; ".join(problems)
    return False, ("every scenario resolves the same index at base and head, "
                   "including the unstubbed ROCm host, a working NVIDIA host with "
                   "stale ROCm packaging, and a CPU-only host with the same")
