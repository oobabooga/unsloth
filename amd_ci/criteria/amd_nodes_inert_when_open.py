#!/usr/bin/env python3
"""Criteria: on a REAL AMD host, does this change alter anything it should not?

Pairs with probes/amd_node_permission_probe.py.

The change makes Studio name an AMD device node this account cannot open instead
of reading it as "no GPU". On the machines that run this, ROCm works and the
runner account can open the nodes, so the defect is not present and cannot be
created without root. That makes this a REGRESSION question, not a differential
one, and deliberately so: a differential here would report VOID, which is correct
but answers nothing.

What real hardware answers that a spoofed host cannot is the other direction.
On the dev box there is no AMD GPU, so ``/sys/class/kfd/kfd/topology/nodes`` is
absent, every topology read collapses to "could not be read", and an entire
family of branches is never entered. Here the topology is real and names an AMD
GPU, the render node vendor really reads 0x1002, and the account really is in
the owning group. So the question this run exists to answer is:

    on a working gfx1151 whose nodes open, is the new code inert?

If it is not -- if it reports a node closed, or produces a hint, or changes what
the derivation says about the real nodes -- then every working AMD user gets a
message about group membership they do not need, which is the mirror image of the
bug being fixed and strictly worse than it.

The gates FAIL rather than skip. A host that turns out not to be AMD, or whose
head checkout does not carry the new symbols, makes the comparison vacuous, and a
vacuous comparison reported as a pass is the failure this harness exists to stop.
"""

from __future__ import annotations

TITLE = "AMD node-permission logic on real gfx1151: inert when the nodes open"
MODE = "regression"

# Authored, not derived from the host: every capability the CHANGE touches. The
# node probe is Linux-only by construction, reads KFD's sysfs topology, reads the
# DRM vendor, and shells out for the installer twin.
NEEDS = ["rocm", "amd_smi", "multi_gpu", "windows", "windows_docker"]


def _module(obs: dict, state: str) -> dict:
    return ((obs.get(state) or {}).get("module") or {})


def _host(obs: dict, state: str) -> dict:
    return ((obs.get(state) or {}).get("host") or {})


def _closed(mod: dict) -> object:
    """The closed list this state reported, or a marker for absent/raised."""
    entry = mod.get("amd_nodes_closed_to_this_user") or {}
    if entry.get("absent"):
        return "<absent>"
    if entry.get("raised"):
        return f"<raised {entry['raised']}>"
    return entry.get("value")


def _hint(mod: dict) -> object:
    entry = mod.get("amd_node_permission_hint") or {}
    if entry.get("absent"):
        return "<absent>"
    if entry.get("raised"):
        return f"<raised {entry['raised']}>"
    return entry.get("value")


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out: list[tuple[str, bool, str]] = []
    head_host = _host(obs, "head")
    head_mod = _module(obs, "head")
    system = head_host.get("platform")

    # 0. The probe has to have produced anything at all, and this gate has to be
    #    FIRST and POSITIVE. Live: the gate below read `.get("import_error") is
    #    None`, which is trivially true for an EMPTY dict, so a probe that never
    #    ran at all reported "imported: yes, from None" and the only thing that
    #    caught it was a different gate noticing the platform was None too. A
    #    gate written as the absence of a failure passes hardest when nothing ran.
    for _state in ("base", "head"):
        _seen = obs.get(_state) or {}
        out.append((
            f"{_state} probe produced observations",
            bool(_seen.get("host")) and _seen.get("host", {}).get("platform") is not None,
            f"keys {sorted(_seen)}; if empty the probe did not run, usually because "
            f"differential.py passes --state/--checkout and the probe wanted something else",
        ))

    # 1. And it has to have imported the module under test.
    out.append((
        "head probe imported utils.hardware.amd",
        bool(head_mod) and head_mod.get("import_error") is None
        and head_mod.get("module_file") is not None,
        head_mod.get("import_error") or f"imported from {head_mod.get('module_file')}",
    ))

    # 2. This must really be AMD hardware, or the run bounds nothing. On Linux
    #    that is KFD's own topology naming vendor 4098. On Windows there is no
    #    KFD, so the render nodes and the topology are both absent by design and
    #    the gate is that the probe confirmed which OS it was on.
    if system == "Linux":
        out.append((
            "this host is really an AMD GPU (KFD topology names vendor 4098)",
            bool(head_host.get("topology_names_an_amd_gpu")),
            f"topology nodes {head_host.get('topology_nodes')}, "
            f"vendor ids {head_host.get('topology_vendor_ids')}",
        ))
        out.append((
            "this host really has a render node",
            len(head_host.get("render_nodes") or []) > 0,
            f"{len(head_host.get('render_nodes') or [])} render node(s): "
            f"{[n.get('path') for n in head_host.get('render_nodes') or []]}",
        ))
        # 3. The head checkout must actually carry the change, or base and head
        #    agree for the boring reason and the run reads as a pass.
        out.append((
            "the head checkout carries the new probe",
            _closed(head_mod) != "<absent>",
            f"amd_nodes_closed_to_this_user -> {_closed(head_mod)!r}",
        ))
        out.append((
            "the head checkout carries the installer twin",
            not ((obs.get("head") or {}).get("installer") or {}).get("absent", True),
            str(((obs.get("head") or {}).get("installer") or {}).get(
                "helpers_lifted", "absent")) + " helpers lifted from install.sh",
        ))
    else:
        out.append((
            "the probe knows which OS it ran on",
            bool(system),
            f"platform.system() -> {system!r}",
        ))
        out.append((
            "the head checkout carries the new probe",
            _closed(head_mod) != "<absent>",
            f"amd_nodes_closed_to_this_user -> {_closed(head_mod)!r}",
        ))
    return out


def table(obs: dict) -> str:
    rows = [
        "| state | closed nodes | hint | blocks the runtime | installer closed set |",
        "|---|---|---|---|---|",
    ]
    for name in ("base", "head", "merge"):
        if not obs.get(name):
            continue
        mod = _module(obs, name)
        inst = (obs.get(name) or {}).get("installer") or {}
        blocks = mod.get("amd_closed_nodes_block_the_runtime") or {}
        blocks_s = "absent" if blocks.get("absent") else repr(blocks.get("value"))
        inst_s = "absent" if inst.get("absent") else repr(inst.get("closed"))
        hint = _hint(mod)
        hint_s = "None" if hint is None else (
            "absent" if hint == "<absent>" else f"`{str(hint)[:90]}`"
        )
        rows.append(f"| {name} | `{_closed(mod)}` | {hint_s} | {blocks_s} | {inst_s} |")

    host = _host(obs, "head")
    facts = [
        "",
        "",
        "Host as the probe read it, with no product code involved:",
        "",
        f"- platform: `{host.get('platform')}`",
        f"- `/dev/kfd` exists: `{host.get('kfd_exists')}`, "
        f"openable by this account: `{host.get('kfd_readable_writable')}`",
        f"- KFD topology vendor ids: `{host.get('topology_vendor_ids')}` "
        f"(4098 is AMD, 4318 NVIDIA, 0 the CPU agent)",
        f"- account uid `{host.get('account_uid')}`, groups `{host.get('account_groups')}`",
    ]
    for node in host.get("render_nodes") or []:
        facts.append(
            f"- `{node.get('path')}` vendor `{node.get('vendor')}` "
            f"openable `{node.get('openable')}`"
        )
    return "\n".join(rows + facts)


def head_is_worse(base: dict, head: dict) -> tuple[bool, str]:
    head_host = head.get("host") or {}
    head_mod = head.get("module") or {}
    base_mod = base.get("module") or {}
    system = head_host.get("platform")

    closed = _closed(head_mod)
    hint = _hint(head_mod)

    # Windows: the probe is Linux-only by construction, so the only correct answer
    # there is "nothing closed, no hint", whatever the box has. A non-empty answer
    # would mean the Linux-only guard stopped holding.
    if system != "Linux":
        if closed not in ([], "<absent>"):
            return True, (f"on {system} the probe must answer [] because it is Linux-only, "
                          f"but it answered {closed!r}")
        if hint not in (None, "<absent>"):
            return True, (f"on {system} the hint must be None because the probe is "
                          f"Linux-only, but it was {str(hint)[:160]!r}")
        return False, (f"on {system} the Linux-only guard holds: closed {closed!r}, "
                       f"hint {hint!r}, and the base said {_closed(base_mod)!r}")

    # Linux, real AMD hardware. Whether the nodes open decides which answer is
    # right, and a node that does not EXIST is not a node that is shut: the probe
    # only reports openability for paths it found, and conflating the two made a
    # host with no /dev/kfd at all read as a reproduction of the defect.
    kfd_shut = bool(head_host.get("kfd_exists")) and not head_host.get("kfd_readable_writable")
    renders = head_host.get("render_nodes") or []
    amd_renders_shut = [
        n for n in renders
        if (n.get("vendor") == "0x1002") and n.get("openable") is False
    ]
    everything_opens = not kfd_shut and not amd_renders_shut

    if everything_opens:
        if closed:
            return True, (
                "every AMD node on this host opens for this account, so the new probe must "
                f"report nothing closed, but it reported {closed!r}. A working AMD host "
                "would be told to join a group it is already effectively in."
            )
        if hint is not None:
            return True, (
                "every AMD node on this host opens, so there is no repair to print, but the "
                f"hint was {str(hint)[:200]!r}"
            )
        inst = (head.get("installer") or {})
        if not inst.get("absent") and inst.get("closed"):
            return True, (
                "the Python half reported nothing closed but the installer twin reported "
                f"{inst.get('closed')!r}; the two halves disagree about the same host"
            )
        return False, (
            "on a real gfx1151 whose nodes all open, the new code is inert: nothing reported "
            f"closed, no hint, and the installer twin agrees (it lifted "
            f"{inst.get('helpers_lifted')} helpers and reported nothing). The base answered "
            f"{_closed(base_mod)!r}."
        )

    # The nodes are genuinely shut on the runner. That is #10466's own shape, and
    # now the head MUST diagnose it while the base must not.
    shut = [n.get("path") for n in amd_renders_shut] + (["/dev/kfd"] if kfd_shut else [])
    if not closed:
        return True, (
            f"this account cannot open {shut}, which is exactly the reported defect, but the "
            "head reported nothing closed"
        )
    if hint is None:
        return True, f"nodes {closed!r} are shut but no repair was printed"
    return False, (
        f"this runner reproduces the defect: {shut} are shut, the head names them "
        f"({closed!r}) and prints a repair, where the base answered {_closed(base_mod)!r}"
    )
