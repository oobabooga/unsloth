#!/usr/bin/env python3
"""Criteria: does PR 9672 move any Windows AMD routing decision it should not?

Pairs with probes/windows_amd_routing_probe.py.

The PR's declared Windows-reachable change is exactly one thing: a per-architecture
generic-wheel floor, adding gfx1102 -> ROCm 6.3 to ``_GENERIC_WHEEL_GFX_MIN_ROCM``
in studio/install_python_stack.py (gfx1200 / gfx1201 -> ROCm 6.4 already exist at
the base). Everything else the probe reads -- install.ps1's adapter-name table,
its gfx -> repo.amd.com family map, setup.ps1's leaf classifiers, and what this
real Strix Halo box's own adapter name resolves to -- is supposed to be untouched.

So this is a REGRESSION question, not a differential one: the base does not
exhibit a defect on THIS host (a Radeon 8060S is gfx1151, which the PR does not
re-floor), and writing it as a differential would produce a VOID. What can be
measured here is whether the head moves a decision on a real Windows machine.

head_is_worse() is therefore: any decision that changed, other than the declared
gfx1102 floor, is a regression. An INTENDED change that fails to appear is
reported too, because a run that shows nothing moved cannot tell "no regression"
from "the probe reached nothing".
"""

from __future__ import annotations

TITLE = "PR 9672 AMD arch detection and ROCm wheel routing, on a real Windows gfx1151 host"
MODE = "regression"

# Authored, not computed. The change rewrites AMD architecture detection and ROCm
# wheel-index routing across Linux install.sh, Windows install.ps1 / setup.ps1 and
# the shared install_python_stack.py, and its new floors are about DISCRETE parts
# (gfx1102 = RX 7600, gfx1200 / gfx1201 = RDNA 4) and about hosts with more than
# one AMD GPU. This host is one integrated Strix Halo iGPU on Windows with no
# ROCm torch, so most of that is declared here rather than quietly left out.
NEEDS = [
    "windows",            # met: this job runs on the Windows half of the pool
    "windows_rocm_wddm",  # whether a ROCm torch even exists on these boxes is unmeasured
    "rocm",
    "gpu",
    "discrete_gpu",       # gfx1102 / gfx1200 / gfx1201 are discrete cards
    "multi_gpu_amd",      # the per-device visible-mask indexing the PR adds
    "linux",              # install.sh, the PR's largest hunk, is not reachable from here
    "amd_smi",
]

# The one decision the PR is allowed to move, stated as a whitelist so anything
# else is caught. gfx1102 gains a (6, 3) floor, so tags older than rocm6.3 must
# start reading "this wheel has no kernels for it".
_INTENDED_FLOOR_ARCH = "gfx1102"
_INTENDED_FLOOR = [6, 3]


def _stack(o: dict) -> dict:
    return (o or {}).get("python_stack") or {}


def _ps(o: dict) -> dict:
    return (o or {}).get("install_ps1") or {}


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out: list[tuple[str, bool, str]] = []
    states = [n for n in ("base", "head", "merge") if n in obs]

    for name in states:
        o = obs.get(name) or {}
        host = o.get("host") or {}
        psres = (host.get("powershell") or {}).get("result") or {}

        # 1. It really ran on Windows. A Windows report produced on Linux is the
        #    failure this whole job exists to avoid.
        on_windows = host.get("sys_platform") == "win32" and host.get("system") == "Windows"
        out.append((f"{name}: ran on Windows", on_windows,
                    f"sys.platform={host.get('sys_platform')!r} "
                    f"platform.system()={host.get('system')!r} "
                    f"node={host.get('node')!r}"))

        # 2. PowerShell answered, and it is the 5.1 Desktop edition these boxes
        #    carry. `pwsh` does not exist here; a run reporting 7.x would mean the
        #    job landed somewhere else entirely.
        ver = str(psres.get("PSVersion") or "")
        edition = str(psres.get("PSEdition") or "")
        out.append((f"{name}: PowerShell answered", bool(ver),
                    f"PSVersion={ver or 'NONE'} PSEdition={edition or 'NONE'} "
                    f"OS={psres.get('OSCaption') or '?'} build={psres.get('OSBuild') or '?'}"))

        # 3. A real adapter name was read. Without one the name -> gfx table was
        #    fed synthetic strings only, and the "real host" claim is empty.
        real = [n for n in (o.get("real_adapter_names") or []) if n.strip()]
        out.append((f"{name}: read a real Win32_VideoController name", bool(real),
                    ", ".join(real) if real else "no adapter name returned"))

        # 4. install.ps1's tables were actually lifted and executed. An extractor
        #    that silently found nothing decides nothing, and comparing two empty
        #    decision sets always shows no regression.
        dec = (_ps(o).get("decisions") or {})
        out.append((f"{name}: install.ps1 name/arch tables executed", len(dec) > 0,
                    f"{len(dec)} adapter name(s) routed; tables_found="
                    f"{_ps(o).get('tables_found')}"))

        # 5. The cross-platform module the PR edits imported, and its floor table
        #    is non-empty. This is the only place a Windows run can see the change.
        st = _stack(o)
        floor = st.get("generic_wheel_floor") or {}
        out.append((f"{name}: install_python_stack imported", bool(st.get("imported")),
                    str(st.get("import_error") or f"floor table has {len(floor)} entries")))
        out.append((f"{name}: floor decisions computed", len(st.get("decisions") or {}) > 0,
                    f"{len(st.get('decisions') or {})} arch(es) evaluated"))

    # 6. Both halves of the comparison exist.
    out.append(("base and head were both probed",
                ("base" in obs) and ("head" in obs),
                "states probed: " + ", ".join(states)))

    # 7. The PR's intended change is visible at the head. If it is not, the probe
    #    did not reach the code under test and "nothing moved" means nothing.
    hb = (_stack(obs.get("base") or {}).get("generic_wheel_floor") or {})
    hh = (_stack(obs.get("head") or {}).get("generic_wheel_floor") or {})
    intended = (_INTENDED_FLOOR_ARCH not in hb) and (hh.get(_INTENDED_FLOOR_ARCH) == _INTENDED_FLOOR)
    out.append(("the change under test is visible between base and head", intended,
                f"base floor keys {sorted(hb)}; head floor keys {sorted(hh)}"))
    return out


def _decision_map(o: dict) -> dict:
    """Every routing decision this probe observed, flattened to comparable keys."""
    flat: dict[str, object] = {}
    for name, d in (_ps(o).get("decisions") or {}).items():
        for field in ("arch", "unsupported", "family", "has_gpu_wheels"):
            flat[f"install.ps1|{name}|{field}"] = d.get(field)
    for leaf, d in (((o or {}).get("setup_ps1") or {}).get("leaf_classification") or {}).items():
        for fn, val in (d or {}).items():
            flat[f"setup.ps1|{leaf}|{fn}"] = val
    st = _stack(o)
    for gfx, d in (st.get("decisions") or {}).items():
        for field in ("has_wheel_route", "route_on_host_single", "route_on_host_mixed",
                      "reroute_no_version", "generic_only_below_floor_no_version"):
            flat[f"python|{gfx}|{field}"] = d.get(field)
        for field in ("tag_lacks_kernels", "reroute_lacks_kernels",
                      "generic_only_below_floor"):
            for ver, val in (d.get(field) or {}).items():
                flat[f"python|{gfx}|{field}|rocm{ver}"] = val
    flat["python|_detect_windows_gfx_arch"] = st.get("detect_windows_gfx_arch")
    flat["python|_detect_amd_gfx_codes"] = str(st.get("detect_amd_gfx_codes"))
    return flat


def _is_intended(key: str) -> bool:
    """Whether a moved decision is the declared gfx1102 floor and nothing else."""
    return key.startswith(f"python|{_INTENDED_FLOOR_ARCH}|")


def _changes(base: dict, head: dict) -> tuple[list, list]:
    b, h = _decision_map(base), _decision_map(head)
    moved = [(k, b.get(k), h.get(k)) for k in sorted(set(b) | set(h))
             if b.get(k) != h.get(k)]
    return ([m for m in moved if _is_intended(m[0])],
            [m for m in moved if not _is_intended(m[0])])


def table(obs: dict) -> str:
    base, head = obs.get("base") or {}, obs.get("head") or {}
    rows: list[str] = []

    host = base.get("host") or {}
    psres = (host.get("powershell") or {}).get("result") or {}
    rows += [
        "**Host as the probe saw it**", "",
        "| field | value |", "|---|---|",
        f"| machine | `{host.get('node') or psres.get('ComputerName') or '?'}` |",
        f"| OS | {psres.get('OSCaption') or '?'} build {psres.get('OSBuild') or '?'} |",
        f"| PowerShell | {psres.get('PSVersion') or '?'} "
        f"{psres.get('PSEdition') or '?'} edition |",
        f"| sys.platform | `{host.get('sys_platform')}` |",
        f"| RUNNER_OS env | `{host.get('runner_os_env')}` (blank is the known W103 trap) |",
        f"| adapter(s) | {', '.join(base.get('real_adapter_names') or []) or 'none'} |",
        f"| amd-smi | `{host.get('amd_smi_path') or 'ABSENT'}` |",
        f"| bash | `{host.get('bash_path') or 'ABSENT'}` |",
        "",
    ]

    rows += ["**This host's own adapter, through each state's routing**", "",
             "| state | install.ps1 arch | index family | python _detect_windows_gfx_arch |",
             "|---|---|---|---|"]
    for name in ("base", "head", "merge"):
        o = obs.get(name)
        if not o:
            continue
        real = (o.get("real_adapter_names") or [None])[0]
        d = (_ps(o).get("decisions") or {}).get(real) or {}
        rows.append(f"| {name} | `{d.get('arch')}` | `{d.get('family')}` | "
                    f"`{_stack(o).get('detect_windows_gfx_arch')}` |")
    rows.append("")

    rows += ["**Generic-wheel floor table (`_GENERIC_WHEEL_GFX_MIN_ROCM`)**", "",
             "| state | entries |", "|---|---|"]
    for name in ("base", "head", "merge"):
        o = obs.get(name)
        if not o:
            continue
        f = _stack(o).get("generic_wheel_floor") or {}
        rows.append(f"| {name} | " + (", ".join(
            f"`{k}`>={v[0]}.{v[1]}" for k, v in sorted(f.items())) or "none") + " |")
    rows.append("")

    intended, unintended = _changes(base, head)
    rows += ["**Decisions that moved between base and head**", "",
             "| decision | base | head | declared? |", "|---|---|---|---|"]
    if not intended and not unintended:
        rows.append("| none | - | - | - |")
    for key, b, h in intended + unintended:
        rows.append(f"| `{key}` | `{b}` | `{h}` | "
                    f"{'yes, the gfx1102 floor' if _is_intended(key) else '**NO**'} |")
    rows.append("")
    rows.append(f"{len(_decision_map(base))} decisions compared per state.")
    return "\n".join(rows)


def head_is_worse(base: dict, head: dict) -> tuple[bool, str]:
    intended, unintended = _changes(base, head)
    if unintended:
        shown = "; ".join(f"`{k}`: {b!r} -> {h!r}" for k, b, h in unintended[:8])
        return True, (f"{len(unintended)} Windows-reachable routing decision(s) moved that "
                      f"this change does not declare: {shown}")
    detail = (f"every one of the {len(_decision_map(base))} Windows-reachable routing "
              f"decisions this probe compared is identical at base and head, except the "
              f"{len(intended)} that are the declared gfx1102 floor")
    if intended:
        detail += ": " + "; ".join(f"`{k.split('|', 2)[-1]}` {b!r} -> {h!r}"
                                   for k, b, h in intended[:6])
        if len(intended) > 6:
            detail += f" (+{len(intended) - 6} more)"
    return False, detail
