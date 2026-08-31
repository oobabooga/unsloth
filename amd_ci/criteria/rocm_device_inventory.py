#!/usr/bin/env python3
"""Criteria: an inventory, not a verdict on a change.

The question is about the HOST, not about a diff: can the AMD CI machine present more
than one GPU to torch? So this is regression mode with nothing to regress -- base and
head are the same machine and are expected to agree. The answer lives in the table.

Scored only on the states disagreeing with each other, which would mean the reading is
unstable and nothing in it can be trusted.
"""

from __future__ import annotations

TITLE = "What GPUs this runner has, and whether it can be made to show more"
MODE = "regression"

NEEDS = ["rocm", "gpu", "integrated_gpu", "discrete_gpu", "multi_gpu", "multi_gpu_amd",
         "gpu_partitions", "nvidia", "mig", "windows", "xpu", "mlx"]


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    states = {n: v for n, v in obs.items() if not n.startswith("_")}
    out: list[tuple[str, bool, str]] = []

    got = {n: v.get("kfd_node_count") for n, v in states.items()}
    out.append(("the KFD topology was readable",
                all(isinstance(v, int) and v > 0 for v in got.values()),
                ", ".join(f"{k}={v}" for k, v in got.items())))

    base = {n: (v.get("torch_counts") or {}).get("baseline", {}).get("count")
            for n, v in states.items()}
    out.append(("torch reported a device count",
                all(isinstance(v, int) for v in base.values()),
                ", ".join(f"{k}={v}" for k, v in base.items())))
    return out


def table(obs: dict) -> str:
    ref = obs.get("head") or obs.get("base") or {}
    rows: list[str] = []

    rows += ["### What the machine has", "",
             "| layer | reading |", "|---|---|"]
    rows.append(f"| KFD topology nodes (kernel) | {ref.get('kfd_node_count')} |")
    rows.append(f"| of those, AMD GPU nodes | {ref.get('kfd_amd_gpu_nodes')} |")
    rows.append(f"| DRM render nodes | {ref.get('render_nodes')} |")
    rows.append(f"| DRM card nodes | {ref.get('card_nodes')} |")
    rows.append(f"| rocminfo GPU agents | "
                f"{(ref.get('rocminfo_gpu_agent_count') or {}).get('out','').strip()} |")

    for label, key in (("KFD node detail", "kfd_nodes"),):
        rows += ["", f"### {label}", "", "```", str(ref.get(key)), "```"]

    for label, key in (
        ("rocminfo agents", "rocminfo_agents"),
        ("amd-smi list", "amd_smi_list"),
        ("amd-smi static --partition", "amd_smi_partition"),
        ("amd-smi set, partition options", "amd_smi_set_help"),
        ("compute-partition sysfs", "sysfs_compute_partition"),
        ("rocm-smi --showcomputepartition", "rocm_smi_partition"),
    ):
        rec = ref.get(key) or {}
        body = (rec.get("out") or rec.get("err") or rec.get("error") or "").strip()
        rows += ["", f"### {label}", "", "```",
                 body[:1500] or "(no output)", "```"]

    rows += ["", "### Can torch be made to see more than one device", "",
             "| mechanism | env | torch device_count |", "|---|---|---|"]
    for label, rec in (ref.get("torch_counts") or {}).items():
        env = ", ".join(f"{k}={v!r}" for k, v in (rec.get("env") or {}).items()) or "-"
        rows.append(f"| {label} | `{env}` | "
                    f"{rec.get('count', rec.get('error', '-'))} |")

    rows += [
        "",
        "Read the partition section first: CPX compute partitioning is the only AMD "
        "mechanism that turns one package into several HIP devices, and it is CDNA2/"
        "CDNA3 only. If this host offers no compute-partition sysfs and amd-smi has no "
        "partition verb, then no amount of environment manipulation will produce a "
        "second torch device here, and the visibility variables can only ever filter "
        "the list shown above.",
        "",
        "`GGML_CUDA_DEVICES` is included because it DOES give llama.cpp real virtual "
        "devices on this machine. It lives in the ggml CUDA-family backend, not in "
        "torch, so it cannot move the count in this table.",
    ]
    return "\n".join(rows)


def head_is_worse(base: dict, head: dict) -> tuple[bool, str]:
    b = (base.get("torch_counts") or {}).get("baseline", {}).get("count")
    h = (head.get("torch_counts") or {}).get("baseline", {}).get("count")
    if b != h:
        return True, f"the two states disagree on the device count ({b} vs {h}), so the reading is unstable"
    return False, (f"both states see {h} torch device(s); this is an inventory of the "
                   f"host rather than a test of the change")
