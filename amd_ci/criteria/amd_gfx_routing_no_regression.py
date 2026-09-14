"""Judge PR 9672's AMD architecture routing against one real host.

The PR's own claim is about a MIXED multi-adapter host, which this pool cannot
produce: a gfx1151 Strix Halo has one GPU, and `--spoof-devices` only fools HIP,
not rocminfo or amd-smi, so it cannot fabricate a second adapter for a parser
that reads their text. So the question this job can answer is the other one, and
it is the one that decides whether the PR is safe to merge:

    on a real AMD host, does the rewritten routing reach the same decision the
    old one did, from the same real probe output?

Hence regression mode. A change of leaf, URL, runtime arch or the Radeon flag on
this hardware, in any of the leaf and mask combinations the probe drives, is a
regression and is reported with both readings. A parser that silently produces
nothing is also a regression: the routing then falls through and the arch-
specific reroute this host depends on never fires.
"""

MODE = "regression"
TITLE = "PR 9672: does the per-device routing decide the same thing on real hardware?"

# Authored for what the change touches. The mixed-adapter case, the archs the
# floor is about, Windows and the amd-smi HIP_ID map on a box that has one KFD
# node are all out of reach here and must stay declared.
NEEDS = ["rocm", "gpu", "multi_gpu", "multi_gpu_amd", "windows", "nvidia", "xpu", "mlx"]

_FIELDS = ("leaf", "url", "runtime_gfx", "radeon")


def gates(obs):
    out = []
    for name in ("base", "head"):
        state = obs.get(name) or {}
        out.append(
            (
                f"{name}: install.sh was readable",
                "install_sh_error" not in state,
                str(state.get("install_sh_error", "ok")),
            )
        )
        tools = state.get("tools_present") or {}
        out.append(
            (
                f"{name}: a real AMD probe answered",
                bool(tools.get("rocminfo") or tools.get("amd_smi_list")),
                f"rocminfo={tools.get('rocminfo')} amd-smi={tools.get('amd_smi_list')}",
            )
        )
        routing = state.get("routing") or {}
        ran = [k for k, v in routing.items() if v.get("rc") == 0 and "leaf" in v]
        out.append(
            (
                f"{name}: the routing block ran",
                len(ran) == len(routing) and bool(routing),
                f"{len(ran)}/{len(routing)} combinations produced a decision",
            )
        )
        out.append(
            (
                f"{name}: the probe read an architecture",
                bool((routing.get("rocm6.1") or {}).get("runtime_gfx")),
                f"runtime_gfx={(routing.get('rocm6.1') or {}).get('runtime_gfx')!r}",
            )
        )
    return out


def table(obs):
    names = [n for n in obs if not n.startswith("_")]
    combos = sorted(set().union(*[set((obs[n] or {}).get("routing") or {}) for n in names]))
    rows = ["| combination | " + " | ".join(f"{n} leaf / arch / radeon" for n in names) + " |",
            "|---|" + "---|" * len(names)]
    for combo in combos:
        cells = []
        for n in names:
            r = ((obs[n] or {}).get("routing") or {}).get(combo) or {}
            cells.append(
                f"`{r.get('leaf', '-')}` / `{r.get('runtime_gfx', '-') or 'none'}` / "
                f"`{r.get('radeon', '-')}`"
            )
        rows.append(f"| `{combo}` | " + " | ".join(cells) + " |")
    rows.append("")
    for n in names:
        recs = (obs[n] or {}).get("records") or {}
        rows.append(f"**{n} records from this host's real output**")
        for key, value in recs.items():
            got = (value.get("stdout") or "").strip().replace("\n", " / ") or "(empty)"
            rows.append(f"- `{key}`: `{got}`")
    return "\n".join(rows)


def head_is_worse(base, head):
    base_routing = (base or {}).get("routing") or {}
    head_routing = (head or {}).get("routing") or {}

    missing = sorted(set(base_routing) - set(head_routing))
    if missing:
        return True, f"the head produced no decision for {missing}"

    changed = []
    for combo, base_result in sorted(base_routing.items()):
        head_result = head_routing.get(combo, {})
        for field in _FIELDS:
            before, after = base_result.get(field), head_result.get(field)
            if before != after:
                changed.append(f"{combo}.{field}: {before!r} -> {after!r}")
    if changed:
        return True, (
            "the routing decision moved on this host, which is a single-adapter "
            "gfx1151 the rewrite is not supposed to change: " + "; ".join(changed)
        )

    # A parser that answers nothing routes nothing, and `|| true` hides it.
    empty = [
        key
        for key, value in ((head or {}).get("records") or {}).items()
        if key == "rocminfo_gpu_records" and not (value.get("stdout") or "").strip()
    ]
    if empty and ((head or {}).get("tools_present") or {}).get("rocminfo"):
        return True, (
            "rocminfo is present on this host but the head's record parser returned "
            "nothing, so the routing it feeds is silently inert"
        )

    absent = [
        name
        for name, present in ((head or {}).get("helpers_present") or {}).items()
        if not present and name in ("_rocminfo_gpu_records", "_gfx_arch_slots")
    ]
    if absent:
        return True, f"the head calls helpers install.sh does not define: {absent}"

    return False, (
        f"all {len(base_routing)} leaf and mask combinations reached the identical "
        f"decision at base and head on a real "
        f"{(head.get('routing', {}).get('rocm6.1') or {}).get('runtime_gfx', 'AMD')} host"
    )
