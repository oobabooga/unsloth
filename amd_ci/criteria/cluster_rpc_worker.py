#!/usr/bin/env python3
"""Criteria: does Studio serve a GGUF on a ROCm ggml-rpc-server worker when a cluster is configured?

Base (before the N-node cluster PR) ignores UNSLOTH_CLUSTER_NODES on a machine that is not a
paired DGX Spark, so the coordinator, whose own GPU is hidden, serves on its CPU: no layer reaches
the worker. Head plans a layer split onto the worker, so every layer's buffer lands on the RPC
device backed by the gfx1151 GPU, and the completion still answers.
"""

from __future__ import annotations

TITLE = "N-node cluster serving onto a ROCm ggml-rpc-server worker"
MODE = "differential"
NEEDS = ["gpu"]


def _states(obs: dict):
    return {n: v for n, v in obs.items() if not n.startswith("_")}


def _rpc_mib(state: dict) -> float:
    return sum(float(mib) for name, mib in (state.get("buffers") or []) if name.startswith("RPC"))


def _rpc_device_mib(state: dict) -> int:
    import re  # noqa: PLC0415

    for line in state.get("coordinator_devices") or []:
        m = re.match(r"RPC\d+:.*\((\d+) MiB", line)
        if m:
            return int(m.group(1))
    return 0


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    states = _states(obs)
    out = []
    out.append((
        "Studio's install ships ggml-rpc-server",
        all(v.get("rpc_server") for v in states.values()),
        ", ".join(f"{n}={v.get('rpc_server')}" for n, v in states.items()),
    ))
    out.append((
        "the worker exposes the GPU over RPC (>= 1 GiB)",
        all(_rpc_device_mib(v) >= 1024 for v in states.values()),
        ", ".join(f"{n}={_rpc_device_mib(v)} MiB" for n, v in states.items()),
    ))
    out.append((
        "the coordinator's own GPU is hidden (no local ROCm device listed)",
        all(not any(l.startswith("ROCm") for l in (v.get("coordinator_devices") or [])) for v in states.values()),
        "; ".join(f"{n}: {v.get('coordinator_devices')}" for n, v in states.items()),
    ))
    out.append((
        "every state served a completion",
        all(v.get("healthy") and v.get("text") for v in states.values()),
        ", ".join(f"{n}={v.get('error') or v.get('healthy')}" for n, v in states.items()),
    ))
    return out


def table(obs: dict) -> str:
    rows = [
        "| state | cluster module | topology | planned args | RPC MiB | offloaded | tok/s | text |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, v in _states(obs).items():
        rows.append(
            f"| {name} | {v.get('has_cluster_module')} | {v.get('topology')} | "
            f"`{' '.join(v.get('planned_args') or []) or '-'}` | {_rpc_mib(v):.1f} | {v.get('offloaded')} | "
            f"{round(v.get('tok_s') or 0, 1)} | {(v.get('text') or '').strip()[:40]!r} |"
        )
    return "\n".join(rows)


def base_shows_defect(base: dict) -> bool:
    return bool(base.get("healthy")) and _rpc_mib(base) == 0


def head_is_fixed(head: dict) -> bool:
    off = head.get("offloaded") or [0, 1]
    return (
        bool(head.get("healthy"))
        and _rpc_mib(head) > 0
        and off[0] == off[1] > 0
        and "paris" in (head.get("text") or "").lower()
    )
