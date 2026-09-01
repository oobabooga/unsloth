#!/usr/bin/env python3
"""Criteria: does PR 9152 reroute the reported Fedora/Bazzite host, and only that host?

Issue 8731: on Fedora/Bazzite none of the five ROCm version sources answers (amd-smi says
N/A, /opt/rocm/.info/version does not exist because no Fedora package owns anything under
/opt/rocm, hipconfig is on PATH but silent, rpm has no rocm-core installed, dpkg-query is
absent), while rocminfo reads the arch perfectly well. A working RX 9070 XT therefore got
CPU-only PyTorch.

  base_shows_defect -- the reported host resolves */cpu at the base.
  head_is_fixed     -- it resolves the per-arch AMD index with the 2.11 constraint and the
                       arch handed to setup.sh, AND every control host still routes exactly
                       where it did.

This is a differential over install.sh's TOP-LEVEL reroute, executed rather than grepped
for. The distinction is not pedantic: the PR's unit suite passes byte-identically when the
reroute is forced to decide "no", because its assertions grep the gate's own variable names,
and a companion probe that only sourced functions concluded it "cannot show 9152 fixing
issue 8731". The block runs here.

The controls are not decoration. This PR adds a decision consulted on every Linux host with
an AMD GPU and no readable ROCm version, so an NVIDIA box with stale ROCm packaging, a
CPU-only box with the same, a Mac, an aarch64 host, a gfx906 card with no per-arch family to
route to, and a Strix iGPU sitting beside a discrete RDNA4 card must all keep their previous
routing. `strix_gfx1151_alone` earns its place specifically: the per-arch mirror's own base
path is repo.amd.com/ROCM/whl/..., which once branded every rerouted host a Radeon-repo
install, so its `radeon` flag is scored.

WHAT THIS CANNOT SAY. Every AMD host shape here is faked -- stub rocminfo / amd-smi /
hipconfig / rpm / dpkg-query / lspci / uname on a hermetic PATH, with /opt/rocm, /dev/kfd,
/sys/class/kfd, /sys/bus/pci/devices and /proc/driver/nvidia redirected. That exercises the
routing decision, not a real Fedora 43 host, and it says nothing about whether the resolved
wheels then import or train. `real_host` is the single unstubbed reading, and it is this
runner's gfx1151 APU, not the reported RX 9070 XT.
"""

from __future__ import annotations

TITLE = "Per-arch ROCm wheel routing when no version is readable (PR 9152, issue 8731)"
MODE = "differential"

# The decision touches every Linux host with an AMD GPU, and its guards are about hosts this
# runner is not. Declared even where the runner cannot be one of them.
NEEDS = ["rocm", "gpu", "linux", "nvidia", "discrete_gpu", "integrated_gpu",
         "multi_gpu_amd", "windows", "xpu", "mlx"]

DEFECT = "bash:fedora_no_version_gfx1201"
AMD = "https://repo.amd.com/rocm/whl"
C211 = "torch>=2.11.0,<2.12.0"

# scenario -> (index substring that must hold at the head, expected gfx export or None)
CONTROLS = {
    "bash:nvidia_with_stale_rocm": ("/cu", ""),
    "bash:cpu_only_with_stale_rocm": ("/cpu", ""),
    "bash:fedora_no_version_on_darwin": ("/cpu", ""),
    "bash:fedora_no_version_aarch64": ("/cpu", ""),
    "bash:gfx906_no_version": ("/cpu", ""),
    "bash:strix_beside_discrete_gfx1201": ("/cpu", ""),
    "bash:strix_gfx1151_alone": (f"{AMD}/gfx1151/", "gfx1151"),
}


def _sc(state: dict, key: str) -> dict:
    return (state.get("scenarios") or {}).get(key) or {}


def _f(state: dict, key: str, field: str) -> str:
    return str(_sc(state, key).get(field) or "")


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out: list[tuple[str, bool, str]] = []
    states = {n: v for n, v in obs.items() if not n.startswith("_")}

    bad = {n: v.get("missing_required") for n, v in states.items()
           if v.get("missing_required")}
    out.append(("every required install.sh function was spliced", not bad,
                str(bad) if bad else "no required function missing in any state"))

    noend = [n for n, v in states.items() if not v.get("block_reached_end_marker")]
    out.append(("the top-level block splice reached its end marker", not noend,
                ", ".join(noend) or "every state ended on 'fi  # _torch_index_pinned guard'"))

    nocall = [n for n, v in states.items() if not v.get("block_has_index_call")]
    out.append(("the block splice kept the get_torch_index_url call", not nocall,
                ", ".join(nocall) or "present in every state"))

    b, h = obs.get("base") or {}, obs.get("head") or {}
    shape = (b.get("has_no_version_reroute") is False
             and h.get("has_no_version_reroute") is True)
    out.append(("base predates the no-version reroute and head carries it", shape,
                f"base={b.get('has_no_version_reroute')} "
                f"head={h.get('has_no_version_reroute')}"))

    real = {n: _f(v, "bash:real_host", "index_url") for n, v in states.items()}
    same = len(set(real.values())) == 1 and all(real.values())
    out.append(("this runner's own unstubbed ROCm stack routes identically in every state",
                same, ", ".join(f"{k}={v or '-'}" for k, v in real.items())))

    scored = [DEFECT, *CONTROLS]
    empty = [f"{n}/{k}" for n, v in states.items() for k in scored
             if not _f(v, k, "index_url")]
    out.append(("every scored scenario produced an index URL", not empty,
                "; ".join(empty) or "all scored scenarios answered"))

    # install.sh is #!/bin/sh with set -e and no set -u, so POSIX parity is a contract.
    for n, v in states.items():
        dash = [k for k in (v.get("scenarios") or {}) if k.startswith("dash:")]
        if not dash:
            continue
        mismatch = [k[5:] for k in dash
                    if _f(v, k, "index_url") != _f(v, "bash:" + k[5:], "index_url")
                    or _f(v, k, "gfx_arch") != _f(v, "bash:" + k[5:], "gfx_arch")]
        out.append((f"{n}: dash and bash agree", not mismatch,
                    ", ".join(mismatch) or "every scenario routes identically under "
                    "dash and bash"))
    return out


def table(obs: dict) -> str:
    names = [n for n in obs if not n.startswith("_")]
    keys: list[str] = []
    for n in names:
        for k in (obs[n].get("scenarios") or {}):
            if k.startswith("bash:") and k not in keys:
                keys.append(k)
    rows = ["| scenario | " + " | ".join(f"{n} index / gfx / constraint" for n in names)
            + " |",
            "|---" * (1 + len(names)) + "|"]
    for k in keys:
        cells = []
        for n in names:
            u = (_f(obs[n], k, "index_url")
                 .replace("https://download.pytorch.org/whl", "pt")
                 .replace("https://repo.amd.com/rocm/whl", "amd"))
            cells.append(f"`{u or '-'}` / `{_f(obs[n], k, 'gfx_arch') or '-'}` / "
                         f"`{_f(obs[n], k, 'torch_constraint') or '-'}`")
        rows.append(f"| {k[5:]} | " + " | ".join(cells) + " |")
    rows += [
        "",
        "`real_host` is this runner's own gfx1151 ROCm stack, unstubbed: the control saying "
        "the change did not move a healthy host. Every other row stubs rocminfo, amd-smi, "
        "hipconfig, rpm, dpkg-query, lspci and uname on a hermetic PATH and redirects "
        "/opt/rocm, /dev/kfd, /sys/class/kfd, /sys/bus/pci/devices and /proc/driver/nvidia, "
        "so they exercise the routing decision rather than real packaging.",
        "",
        "`fedora_no_version_gfx1201` is issue 8731: rocminfo reads gfx1201, and all five "
        "version sources miss.",
        "",
        "Unlike the function-sourcing probe alongside it, this run executes install.sh's "
        "top-level reroute, so it can see the fix rather than only its absence of "
        "collateral damage.",
    ]
    return "\n".join(rows)


def base_shows_defect(base: dict) -> bool:
    # A working RX 9070 XT lands on CPU-only wheels, with no arch handed to setup.sh.
    return (_f(base, DEFECT, "index_url").endswith("/cpu")
            and not _f(base, DEFECT, "gfx_arch"))


def head_is_fixed(head: dict) -> bool:
    if _f(head, DEFECT, "index_url") != f"{AMD}/gfx120X-all/":
        return False
    if _f(head, DEFECT, "gfx_arch") != "gfx1201":
        return False
    if _f(head, DEFECT, "torch_constraint") != C211:
        return False
    if _f(head, DEFECT, "backend") != "rocm":
        return False
    # The per-arch mirror's base path contains "rocm"; branding off the whole URL once made
    # every rerouted host report repo.radeon.com wheels it never fetched.
    if _f(head, DEFECT, "radeon") != "false":
        return False
    for key, (want_url, want_gfx) in CONTROLS.items():
        if want_url not in _f(head, key, "index_url"):
            return False
        if _f(head, key, "gfx_arch") != want_gfx:
            return False
    if _f(head, "bash:strix_gfx1151_alone", "radeon") != "false":
        return False
    return True
