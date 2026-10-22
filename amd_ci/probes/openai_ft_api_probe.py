#!/usr/bin/env python3
"""Probe: does this checkout's Studio answer the OpenAI fine-tuning API, and does a job train on this GPU?

Observes only (unslothai/unsloth#14496). Runs the checkout's real FastAPI app in process with a
real API key, uploads a JSONL file through /v1/files, starts /v1/fine_tuning/jobs and polls it to a
terminal state, so the training worker is the checkout's own. Records HTTP statuses, the job's
status sequence, events and the torch device, and leaves the verdict to the criteria module.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path


def run(checkout: Path, model: str, steps: int, budget_s: int) -> dict:
    backend = checkout / "studio" / "backend"
    os.environ["UNSLOTH_STUDIO_HOME"] = tempfile.mkdtemp(prefix = "ft_api_home_")
    sys.path.insert(0, str(backend))
    os.chdir(backend)
    obs: dict = {"backend": str(backend)}
    try:
        import torch

        obs["torch_hip"] = getattr(torch.version, "hip", None)
        obs["device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception as e:  # noqa: BLE001
        obs["torch_error"] = f"{type(e).__name__}: {e}"

    import httpx
    import main as studio
    from auth import storage

    storage.ensure_default_admin()
    key, _ = storage.create_api_key("unsloth", "amd-ci")
    rows = [
        {"messages": [{"role": "user", "content": f"Code for item {i}?"},
                      {"role": "assistant", "content": f"UNSLOTH-{i * 7}"}]}
        for i in range(32)
    ]
    body = "\n".join(json.dumps(r) for r in rows).encode()

    import asyncio

    async def flow() -> None:
        transport = httpx.ASGITransport(app = studio.app, client = ("127.0.0.1", 5555))
        async with httpx.AsyncClient(transport = transport, base_url = "http://127.0.0.1:8888",
                                     headers = {"Authorization": f"Bearer {key}"}, timeout = 600) as c:
            r = await c.post("/v1/files", files = {"file": ("codes.jsonl", body)}, data = {"purpose": "fine-tune"})
            obs["files_create_status"] = r.status_code
            if r.status_code != 200:
                return
            file_id = r.json()["id"]
            r = await c.post("/v1/fine_tuning/jobs", json = {
                "model": model, "training_file": file_id,
                "method": {"type": "supervised", "supervised": {"hyperparameters": {"n_epochs": 1, "batch_size": 2}}},
                "max_steps": steps, "max_seq_length": 256,
            })
            obs["jobs_create_status"] = r.status_code
            obs["jobs_create_body"] = r.json() if r.headers.get("content-type", "").startswith("application/json") else r.text[:500]
            if r.status_code != 200:
                return
            job_id = r.json()["id"]
            seen, t0 = [], time.time()
            job = {}
            while time.time() - t0 < budget_s:
                job = (await c.get(f"/v1/fine_tuning/jobs/{job_id}")).json()
                if not seen or seen[-1] != job.get("status"):
                    seen.append(job.get("status"))
                if job.get("status") in ("succeeded", "failed", "cancelled"):
                    break
                await asyncio.sleep(10)
            obs["status_sequence"] = seen
            obs["final_job"] = job
            obs["wall_s"] = round(time.time() - t0, 1)
            events = (await c.get(f"/v1/fine_tuning/jobs/{job_id}/events", params = {"limit": 100})).json()
            obs["metric_events"] = [e["data"] for e in events.get("data", []) if e.get("type") == "metrics"]
            checkpoints = (await c.get(f"/v1/fine_tuning/jobs/{job_id}/checkpoints")).json()
            obs["checkpoints"] = [x["fine_tuned_model_checkpoint"] for x in checkpoints.get("data", [])]

    asyncio.run(flow())
    return obs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True, type = Path)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--model", default = "unsloth/gemma-3-270m-it")
    ap.add_argument("--steps", type = int, default = 6)
    ap.add_argument("--budget-s", type = int, default = 1800)
    args = ap.parse_args()
    out = args.out.resolve()
    obs: dict = {"state": args.state}
    try:
        obs.update(run(args.checkout.resolve(), args.model, args.steps, args.budget_s))
    except Exception as e:  # noqa: BLE001
        obs["probe_error"] = f"{type(e).__name__}: {e}"
    out.write_text(json.dumps(obs, indent = 2, default = str), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
