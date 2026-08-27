#!/usr/bin/env python3
"""Criteria: does the installer resolve this AMD GPU's wheel index? (PR 9829)

**Read the defect shape before the verdict.** This PR is mostly about gfx1103 and
gfx1031-gfx1036, whose kernels the generic pytorch.org ROCm wheel does not carry.
This host is gfx1151, which IS in the generic wheel's arch list and already had a
route of its own, so on the unmodified detection path base and head choose the same
index and the PR's headline repair is not reachable here at all.

What IS reachable is the other half of the change: the resolver now falls back to
KFD topology sysfs. A runtime-only ROCm install ships neither `rocminfo` nor
`amd-smi`, and at the base that host resolves no target and keeps a wheel chosen
without knowing the architecture. So the defect predicate is stated on that host
shape, reproduced by removing the two executables from PATH while the KFD read
underneath remains this machine's real kernel topology.

That mask is a CONTROLLED condition, not a naturally occurring failure of this
runner, and a CONFIRMED here means the KFD fallback works on real gfx1151 hardware.
It does not mean the gfx1103 reroute was tested. The gate below requires base and
head to agree on the unmasked path, so the mask is the only variable.

Pairs with probes/rocm_index_route_probe.py.
"""

from __future__ import annotations

TITLE = "AMD wheel index routing on gfx1151, base versus head"
MODE = "differential"

NEEDS = [
    "linux", "rocm", "gpu", "integrated_gpu", "amd_smi",
    "discrete_gpu", "multi_gpu", "multi_gpu_amd", "windows",
]

_ARCH_LEAF = "gfx1151"


def _s(o: dict, scenario: str) -> dict:
    return o.get(scenario) or {}


def _is_arch_index(url) -> bool:
    return bool(url) and f"/{_ARCH_LEAF}/" in str(url)


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    states = [n for n in obs if not n.startswith("_")]
    out: list[tuple[str, bool, str]] = []

    bad = [f"{n}.{sc}" for n in states
           for sc in ("natural", "runtime_only", "runtime_only_unnamed_product")
           if _s(obs[n], sc).get("worker_error") or _s(obs[n], sc).get("no_output")
           or _s(obs[n], sc).get("install_calls") is None]
    out.append(("every scenario loaded the installer and reached a decision", not bad,
                ", ".join(bad) if bad else ", ".join(
                    f"{n}={_s(obs[n], 'natural').get('chosen_label') or 'no install call'}"
                    for n in states)))

    kfd = {n: (_s(obs[n], "runtime_only").get("kfd_targets") or []) for n in states}
    kfd_ok = all(t == [_ARCH_LEAF] for t in kfd.values())
    out.append((f"the kernel's KFD topology really reports one {_ARCH_LEAF}", kfd_ok,
                ", ".join(f"{n}={t or 'none'}" for n, t in kfd.items())))

    # The toolkit's own masking pitfall: a difference that is really the mask, read as a
    # difference between the states. Unmasked, both states must agree.
    nat = {n: _s(obs[n], "natural").get("chosen_index_url") for n in states}
    agree = len(set(map(str, nat.values()))) == 1 and _is_arch_index(nat[states[0]])
    out.append(("unmasked, every state picks the same gfx1151 index, so the PATH mask is "
                "the only variable", agree,
                ", ".join(f"{n}={v or 'none'}" for n, v in nat.items())))

    masked = all(not _s(obs[n], "runtime_only").get("rocminfo_on_path")
                 and not _s(obs[n], "runtime_only").get("amd_smi_on_path") for n in states)
    out.append(("rocminfo and amd-smi were really absent in the masked scenario", masked,
                ", ".join(f"{n}: rocminfo={_s(obs[n], 'runtime_only').get('rocminfo_on_path')}"
                          for n in states)))

    # Nothing may have been installed for real: every recorded call is a recorder's.
    forced = [f"{n}.{sc}" for n in states for sc in ("natural", "runtime_only",
                                                    "runtime_only_unnamed_product")
              if (_s(obs[n], sc).get("errors") or {}).get("ensure_rocm_torch")]
    out.append(("the installer drive raised nothing", not forced,
                ", ".join(forced) if forced else "none"))
    return out


def table(obs: dict) -> str:
    states = [n for n in obs if not n.startswith("_")]
    rows = ["| scenario | observation | " + " | ".join(states) + " |",
            "|---" * (2 + len(states)) + "|"]

    def row(scenario, label, key, fn = None):
        vals = []
        for n in states:
            v = _s(obs[n], scenario).get(key)
            vals.append(_fmt(fn(v) if fn else v))
        rows.append(f"| {scenario} | {label} | " + " | ".join(vals) + " |")

    for sc in ("natural", "runtime_only", "runtime_only_unnamed_product"):
        row(sc, "userland probe result", "detected_gfx")
        row(sc, "KFD topology", "kfd_targets")
        row(sc, "product-name inference", "inferred_product_gfx")
        row(sc, "`_has_rocm_gpu()`", "has_rocm_gpu")
        row(sc, "resolved target", "runtime_gfx_target")
        row(sc, "index it would install from", "chosen_index_url")
        row(sc, "install label", "chosen_label")
    rows.append("")
    rows.append("`runtime_only` removes rocminfo and amd-smi from PATH to reproduce a "
                "runtime-only ROCm install; the KFD reading underneath is this machine's real "
                "kernel topology. `runtime_only_unnamed_product` additionally suppresses "
                "product-name inference and is supplementary evidence only, outside the "
                "verdict. The gfx1103 / gfx1031-gfx1036 reroute this PR is chiefly about is "
                "not reachable on a gfx1151 host at all.")
    return "\n".join(rows)


def _fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, list):
        return ", ".join(f"`{x}`" for x in v) if v else "(none)"
    return f"`{v}`"


def base_shows_defect(base: dict) -> bool:
    """On a host with no ROCm userland, the base resolves nothing and keeps a wheel
    picked without knowing the architecture, while the kernel plainly reports gfx1151."""
    o = _s(base, "runtime_only")
    return (o.get("kfd_targets") == [_ARCH_LEAF]
            and not _is_arch_index(o.get("chosen_index_url")))


def head_is_fixed(head: dict) -> bool:
    o = _s(head, "runtime_only")
    nat = _s(head, "natural")
    return (_is_arch_index(o.get("chosen_index_url"))
            and o.get("runtime_gfx_target") == _ARCH_LEAF
            # and the fix did not come at the cost of the path that already worked
            and _is_arch_index(nat.get("chosen_index_url")))
