#!/usr/bin/env python3
"""Probe: what the AMD device-node permission logic says about THIS host.

Observes only. It reads the host, then asks this checkout's
``utils.hardware.amd`` what it makes of it, and records both. Every judgement --
whether an answer is right, whether base and head may differ -- belongs to the
criteria module beside it.

The point of running this on real gfx1151 hardware is that the dev box has no AMD
GPU at all, so ``/sys/class/kfd/kfd/topology/nodes`` is absent, every topology
read collapses to "could not be read", and a whole family of branches is never
entered. Here the topology is real, the render nodes are real, their vendor is
really 0x1002, and the account really does or does not sit in the owning group.

Base has an ``amd.py`` too, but without the symbols this change adds. A missing
symbol is recorded as absent rather than raising, because "the base did not have
this function" is an observation the criteria needs, not an error.

JSON to --out, never stdout: an import banner on stdout corrupts it.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

_KFD = "/dev/kfd"
_RENDER_GLOB = "/dev/dri/renderD*"
_TOPOLOGY = "/sys/class/kfd/kfd/topology/nodes"


def _host_facts() -> dict:
    """What the operating system says, with no product code involved."""
    facts: dict = {
        "platform": platform.system(),
        "kfd_exists": False,
        "kfd_readable_writable": False,
        "render_nodes": [],
        "topology_nodes": [],
        "topology_vendor_ids": [],
        "account_uid": None,
        "account_groups": [],
    }
    try:
        facts["account_uid"] = os.getuid()
        facts["account_groups"] = sorted({os.getgid(), *os.getgroups()})
    except AttributeError:
        pass  # Windows

    try:
        facts["kfd_exists"] = os.path.exists(_KFD)
        if facts["kfd_exists"]:
            facts["kfd_readable_writable"] = os.access(_KFD, os.R_OK | os.W_OK)
    except OSError as e:
        facts["kfd_error"] = f"{type(e).__name__}: {e}"

    for path in sorted(glob.glob(_RENDER_GLOB)):
        entry = {"path": path, "openable": None, "vendor": None}
        try:
            entry["openable"] = os.access(path, os.R_OK | os.W_OK)
        except OSError:
            pass
        vendor_file = f"/sys/class/drm/{os.path.basename(path)}/device/vendor"
        try:
            entry["vendor"] = Path(vendor_file).read_text(encoding = "utf-8").strip().lower()
        except (OSError, UnicodeDecodeError):
            pass
        facts["render_nodes"].append(entry)

    try:
        for node in sorted(os.listdir(_TOPOLOGY)):
            facts["topology_nodes"].append(node)
            try:
                props = Path(_TOPOLOGY, node, "properties").read_text(encoding = "utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            for line in props.splitlines():
                if line.startswith("vendor_id "):
                    facts["topology_vendor_ids"].append(line.split()[1])
    except OSError:
        pass
    # 4098 is AMD, 4318 is NVIDIA's open kernel module, 0 is the CPU agent.
    facts["topology_names_an_amd_gpu"] = "4098" in facts["topology_vendor_ids"]
    return facts


def _ask_the_module(repo: Path) -> dict:
    """What THIS checkout's utils.hardware.amd makes of the host above."""
    backend = repo / "studio" / "backend"
    sys.path.insert(0, str(backend))
    answers: dict = {"import_error": None}
    try:
        from utils.hardware import amd  # noqa: PLC0415
    except Exception as e:  # noqa: BLE001 -- an unimportable module is an observation
        answers["import_error"] = f"{type(e).__name__}: {e}"
        return answers

    answers["module_file"] = getattr(amd, "__file__", None)

    def _call(name, *args, **kwargs):
        fn = getattr(amd, name, None)
        if fn is None:
            return {"absent": True}
        try:
            return {"value": fn(*args, **kwargs)}
        except Exception as e:  # noqa: BLE001
            return {"raised": f"{type(e).__name__}: {e}"}

    answers["amd_nodes_closed_to_this_user"] = _call("amd_nodes_closed_to_this_user")
    answers["amd_node_permission_hint"] = _call("amd_node_permission_hint")
    answers["amd_node_permission_hint_vulkan"] = _call(
        "amd_node_permission_hint", needs_kfd = False
    )
    answers["amd_closed_nodes_block_the_runtime"] = _call("amd_closed_nodes_block_the_runtime")
    answers["kfd_topology_amd_state"] = _call("_kfd_topology_amd_state")
    answers["amd_kfd_gpu_node_count"] = _call("amd_kfd_gpu_node_count")
    answers["a_confirmed_amd_render_node_exists"] = _call("_a_confirmed_amd_render_node_exists")
    # The derivation, against the REAL nodes: the dev box can only patch these.
    _present = [p for p in (_KFD, *sorted(glob.glob(_RENDER_GLOB))) if os.path.exists(p)]
    answers["groups_that_own_the_real_nodes"] = _call("_groups_that_own", _present)
    answers["nodes_asked_about"] = _present
    return answers


def _ask_the_installer(repo: Path) -> dict:
    """The shell twin, run from THIS checkout's install.sh against the same host.

    The function is lifted by name rather than the installer being run, because
    running the installer would install something. A checkout without the
    function records it as absent, which is what the base looks like.
    """
    install_sh = repo / "install.sh"
    out: dict = {"absent": True}
    if not install_sh.exists():
        out["error"] = "no install.sh at this state"
        return out
    try:
        lines = install_sh.read_text(encoding = "utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as e:
        out["error"] = f"{type(e).__name__}: {e}"
        return out

    def _fn(name: str):
        start = next((i for i, l in enumerate(lines) if l.startswith(f"{name}() {{")), None)
        if start is None:
            return None
        depth = 0
        for end in range(start, len(lines)):
            depth += lines[end].count("{") - lines[end].count("}")
            if depth == 0:
                return "\n".join(lines[start:end + 1])
        return None

    import re as _re

    wanted = "_amd_nodes_closed_to_this_user"
    if _fn(wanted) is None:
        return out  # absent: this is the base
    lifted, seen, queue = [], set(), [wanted]
    while queue:
        name = queue.pop(0)
        if name in seen:
            continue
        seen.add(name)
        body = _fn(name)
        if body is None:
            continue
        lifted.append(body)
        queue += sorted(set(_re.findall(r"\b(_[a-z0-9_]+)\b", body)) - seen)
    script = "\n".join([*lifted, f"{wanted}"])
    try:
        run = subprocess.run(
            ["bash", "-c", script], capture_output = True, text = True, timeout = 120
        )
    except (OSError, subprocess.SubprocessError) as e:
        return {"absent": False, "error": f"{type(e).__name__}: {e}"}
    return {
        "absent": False,
        "rc": run.returncode,
        "closed": [l for l in run.stdout.splitlines() if l.strip()],
        "stderr_tail": run.stderr.strip()[-400:],
        "helpers_lifted": len(lifted),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    # The names differential.py actually passes. Getting these wrong is not a loud
    # failure: argparse exits 2, the probe writes nothing, every observation comes
    # back empty, and the criteria then reads None for every field. Live: the first
    # version of this probe took --repo, and both the Linux and the Windows leg
    # reported INCONCLUSIVE with "platform.system() -> None".
    ap.add_argument("--state", default = "unknown", help = "base / head / merge")
    ap.add_argument("--checkout", required = True, help = "worktree for the state")
    ap.add_argument("--out", required = True)
    # Accepted and ignored: differential.py forwards its own --python to probes.
    ap.add_argument("--python", default = None)
    args, _unknown = ap.parse_known_args()

    repo = Path(args.checkout).resolve()
    observations = {
        "state": args.state,
        "host": _host_facts(),
        "module": _ask_the_module(repo),
        "installer": _ask_the_installer(repo) if platform.system() == "Linux" else {
            "absent": True, "error": "not Linux, install.sh is not the installer here"
        },
    }
    Path(args.out).write_text(json.dumps(observations, indent = 2, default = str),
                              encoding = "utf-8")


if __name__ == "__main__":
    main()
