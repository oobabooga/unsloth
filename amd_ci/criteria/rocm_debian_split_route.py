#!/usr/bin/env python3
"""Criteria: does PR 8886 read the Debian split stack right, without moving anyone else?

Issue 8402: on Debian 13 / LMDE, `hipconfig` reports 5.7 while the installed HSA
runtime is 6.1 and `rocm-core` is absent entirely. The resolver saw only 5.7, which is
below the ROCm 6.0 wheel floor, so a working gfx1100 got CPU-only PyTorch.

  base_shows_defect -- the Debian layout resolves rocm5.7 and lands on the cpu index.
  head_is_fixed     -- it resolves rocm6.1 and lands on a rocm index, AND every
                       control host still routes exactly where it did.

The controls are not decoration. This PR changes a version SOURCE, and a version
source is consulted on hosts that have nothing to do with Debian: a working NVIDIA box
with stale ROCm packages lying around, a CPU-only box with the same, and an Ubuntu box
where AMD's `rocm-core` sits beside the distro's much older `libhsa-runtime64-1`. Any
of those moving is a worse outcome than the bug being fixed.

WHAT THIS CANNOT SAY. The Debian and Fedora layouts are stubbed: `dpkg-query`, `rpm`,
`hipconfig`, `amd-smi` and `rocminfo` are fake binaries on PATH, and `/opt/rocm`,
`/proc/driver/nvidia` and `/dev/kfd` are redirected. That exercises the resolver, not a
real Debian 13 host, and it says nothing about whether the rocm6.1 wheels then run.
"""

from __future__ import annotations

TITLE = "ROCm version resolution on a split Debian stack (PR 8886)"
MODE = "differential"

# A version source is read on every Linux host, so the shapes this could move are
# declared even where this runner is not one of them.
NEEDS = ["rocm", "gpu", "linux", "nvidia", "discrete_gpu", "integrated_gpu",
         "multi_gpu_amd", "windows", "xpu", "mlx"]

DEFECT = "bash:debian_split_stack"
# (scenario, what it must keep doing). These are the hosts that must not move.
CONTROLS = {
    "bash:nvidia_with_stale_rocm": "cu",
    "bash:cpu_only_with_stale_rocm": "/cpu",
    "bash:ubuntu_rocm_core_and_hsa": "/rocm7.2",
}


def _sc(state: dict, key: str) -> dict:
    return (state.get("scenarios") or {}).get(key) or {}


def _url(state: dict, key: str) -> str:
    return str(_sc(state, key).get("index_url") or "")


def _tag(state: dict, key: str) -> str:
    return str(_sc(state, key).get("rocm_tag") or "")


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out: list[tuple[str, bool, str]] = []
    states = {n: v for n, v in obs.items() if not n.startswith("_")}

    # The splice is the harness's single point of silent failure: a helper install.sh
    # calls but the spliced file never defined makes the ROCm branch die and the whole
    # thing resolve to "cpu", which is indistinguishable from a routing decision.
    bad = {n: v.get("missing_required") for n, v in states.items()
           if v.get("missing_required")}
    out.append(("every required install.sh function was spliced", not bad,
                str(bad) if bad else "no required function missing in any state"))

    # Every scenario must have produced a URL, or an empty string reads as "cpu-ish"
    # and the comparison silently degrades.
    empty = [f"{n}/{k}" for n, v in states.items()
             for k in list(CONTROLS) + [DEFECT] if not _url(v, k)]
    out.append(("every scored scenario produced an index URL", not empty,
                "; ".join(empty) or "all scored scenarios answered"))

    b, h = obs.get("base") or {}, obs.get("head") or {}
    shape = (b.get("has_hsa_runtime_source") is False
             and h.get("has_hsa_runtime_source") is True)
    out.append(("base predates the HSA runtime source and head carries it", shape,
                f"base={b.get('has_hsa_runtime_source')} "
                f"head={h.get('has_hsa_runtime_source')}"))

    # dash as well as bash: install.sh is #!/bin/sh, so a bashism is a real defect.
    for n, v in states.items():
        dash = [k for k in (v.get("scenarios") or {}) if k.startswith("dash:")]
        if not dash:
            continue
        mismatch = [k[5:] for k in dash
                    if _url(v, k) != _url(v, "bash:" + k[5:])]
        out.append((f"{n}: dash and bash agree", not mismatch,
                    ", ".join(mismatch) or "every scenario routes identically "
                    "under dash and bash"))
    return out


def table(obs: dict) -> str:
    keys: list[str] = []
    for v in obs.values():
        if isinstance(v, dict):
            for k in (v.get("scenarios") or {}):
                if k.startswith("bash:") and k not in keys:
                    keys.append(k)
    rows = ["| scenario | " + " | ".join(
        f"{n} tag / index" for n in obs if not n.startswith("_")) + " |",
        "|---" * (1 + len([n for n in obs if not n.startswith("_")])) + "|"]
    for k in keys:
        cells = []
        for n, v in obs.items():
            if n.startswith("_"):
                continue
            u = _url(v, k).replace("https://download.pytorch.org/whl", "pt")
            cells.append(f"`{_tag(v, k) or '-'}` / `{u or '-'}`")
        rows.append(f"| {k[5:]} | " + " | ".join(cells) + " |")
    rows += [
        "",
        "`real_host` is this runner's own ROCm stack, unstubbed: it is the control "
        "saying the change did not move a healthy host. Every other row stubs the "
        "probe binaries and redirects `/opt/rocm`, `/proc/driver/nvidia` and "
        "`/dev/kfd`, so they exercise the resolver rather than real packaging.",
        "",
        "`debian_split_stack` is issue 8402: hipconfig 5.7, libhsa-runtime64-1 6.1, "
        "no rocm-core.",
    ]
    return "\n".join(rows)


def base_shows_defect(base: dict) -> bool:
    # 5.7 is below the ROCm 6.0 wheel floor, so the Debian host takes CPU wheels.
    return _tag(base, DEFECT).startswith("rocm5.") and _url(base, DEFECT).endswith("/cpu")


def head_is_fixed(head: dict) -> bool:
    fixed = _tag(head, DEFECT).startswith("rocm6.") and "/rocm" in _url(head, DEFECT)
    if not fixed:
        return False
    # A fix that moved a control host is not a fix.
    return all(want in _url(head, key) for key, want in CONTROLS.items())
