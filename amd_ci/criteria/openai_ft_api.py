#!/usr/bin/env python3
"""Criteria: the OpenAI fine-tuning API trains on gfx1151 (unslothai/unsloth#14496).

Pairs with probes/openai_ft_api_probe.py. Base (the merge commit's parent) has no /v1/files
route, so the upload answers 404. Head trains: the job reaches `succeeded`, logs per-step
metric events with finite losses, and names a fine-tuned model. The gates prove the run used a
ROCm device and that each state was probed.
"""

from __future__ import annotations

TITLE = "OpenAI-compatible fine-tuning API: real training on ROCm"
MODE = "differential"
NEEDS = ["rocm", "gpu", "integrated_gpu", "linux"]


def _states(obs: dict):
    return {n: v for n, v in obs.items() if not n.startswith("_")}


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    states = _states(obs)
    probed = bool(states) and all(v.get("_probe_rc") == 0 and "probe_error" not in v for v in states.values())
    rocm = bool(states) and all(v.get("torch_hip") and v.get("device") for v in states.values())
    return [
        ("every state probed without a probe error", probed,
         ", ".join(f"{n}: rc={v.get('_probe_rc')} err={v.get('probe_error')}" for n, v in states.items())),
        ("each state saw a ROCm torch device", rocm,
         ", ".join(f"{n}: hip={v.get('torch_hip')} device={v.get('device')}" for n, v in states.items())),
    ]


def table(obs: dict) -> str:
    rows = ["| state | POST /v1/files | POST /v1/fine_tuning/jobs | statuses | final | metric steps | first/last loss | checkpoints |",
            "|---|---|---|---|---|---|---|---|"]
    for n, v in _states(obs).items():
        ev = sorted(v.get("metric_events") or [], key = lambda d: d.get("step") or 0)
        losses = [d.get("train_loss") for d in ev if d.get("train_loss") is not None]
        loss = f"{losses[0]:.3f}/{losses[-1]:.3f}" if losses else "-"
        final = (v.get("final_job") or {}).get("status")
        rows.append(f"| {n} | {v.get('files_create_status')} | {v.get('jobs_create_status')} | "
                    f"{v.get('status_sequence')} | {final} | {len(ev)} | {loss} | {len(v.get('checkpoints') or [])} |")
    return "\n".join(rows)


def base_shows_defect(base: dict) -> bool:
    return base.get("files_create_status") == 404


def head_is_fixed(head: dict) -> bool:
    final = head.get("final_job") or {}
    losses = [d.get("train_loss") for d in head.get("metric_events") or [] if d.get("train_loss") is not None]
    return (head.get("files_create_status") == 200 and head.get("jobs_create_status") == 200
            and final.get("status") == "succeeded" and bool(final.get("fine_tuned_model")) and bool(losses))
