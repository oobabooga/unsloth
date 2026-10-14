"""Drive Studio's real image load path on a Mac up to the unified-memory refusal, and stop there.

usage: python decision_ab.py <studio/backend dir> <label> <reporter|runner>

``reporter`` replays the reporter's machine (24 GiB MacBook Air, Metal working set 16 GiB, about 16.8 GiB available,
which is what "needs about 15 GB, only about 12 GB usable" implies under main's arithmetic). ``runner`` uses this VM's
own live readings. Each model is loaded with DiffusionBackend.load_pipeline, which downloads and sizes the real files;
the refusal either raises (refused) or is wrapped so the load stops right after it (passed), because this 7 GB VM cannot
hold the weights.
"""
import json
import os
import sys
import time

tree, label, machine = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, tree)
os.chdir(tree)

import torch  # noqa: E402

from core.inference import diffusion as d  # noqa: E402
from core.inference import diffusion_memory as dm  # noqa: E402

MiB = 1024 * 1024
if machine == "reporter":
    _total_mib, _avail_mib, _ws = 24 * 1024, int(16.8 * 1024), 16 * 1024**3
    dm._system_memory_mib = lambda: (_total_mib, _avail_mib)
    torch.mps.recommended_max_memory = lambda: _ws
    torch.mps.driver_allocated_memory = lambda: 0


class GuardPassed(Exception):
    pass


_real_guard = d.raise_on_unified_memory_shortfall
seen = {}


def _guard(plan, **kwargs):
    seen["plan"] = plan
    try:
        _real_guard(plan, **kwargs)
    except RuntimeError as exc:
        seen["refused"] = str(exc)
        raise
    finally:
        seen["reached"] = True
    raise GuardPassed("stopped by the A/B harness after the unified-memory guard passed")


d.raise_on_unified_memory_shortfall = _guard

MODELS = [
    ("Z-Image Turbo Q4_K_M", "unsloth/Z-Image-Turbo-GGUF", "z-image-turbo-Q4_K_M.gguf"),
    ("Z-Image Turbo Q3_K_S", "unsloth/Z-Image-Turbo-GGUF", "z-image-turbo-Q3_K_S.gguf"),
    ("Z-Image Turbo Q2_K", "unsloth/Z-Image-Turbo-GGUF", "z-image-turbo-Q2_K.gguf"),
    ("FLUX.2 Klein 4B Q4_K_M", "unsloth/FLUX.2-klein-4B-GGUF", "flux-2-klein-4b-Q4_K_M.gguf"),
    ("FLUX.2 Klein 4B Q6_K", "unsloth/FLUX.2-klein-4B-GGUF", "flux-2-klein-4b-Q6_K.gguf"),
    ("FLUX.2 Klein 4B Q8_0", "unsloth/FLUX.2-klein-4B-GGUF", "flux-2-klein-4b-Q8_0.gguf"),
    ("SDXL base 1.0", "stabilityai/stable-diffusion-xl-base-1.0", None),
]
only = os.environ.get("AB_ONLY")
backend = d.DiffusionBackend()
for name, repo, gguf in MODELS:
    if only and only not in name:
        continue
    seen.clear()
    row = {"tree": label, "machine": machine, "model": name}
    # begin_load is the route's entry point: it stages the companions (prefetch) and then runs load_pipeline on a
    # worker thread, so the guard sees the text encoder and VAE on disk exactly as it does in the app.
    backend.begin_load(repo, gguf_filename = gguf)
    deadline = time.time() + 3600
    while time.time() < deadline and not seen.get("reached"):
        if backend.load_progress().get("phase") == "error":
            break
        time.sleep(2)
    if "refused" in seen:
        row["verdict"] = "refused"
        row["message"] = seen["refused"]
    elif seen.get("reached"):
        row["verdict"] = "passed the guard"
    else:
        row["verdict"] = "error before the guard"
        row["message"] = str(backend.load_progress())[:800]
    settle = time.time() + 300
    while time.time() < settle and backend._loading is not None and backend.load_progress().get("phase") != "error":
        time.sleep(1)
    plan = seen.get("plan")
    if plan is not None:
        est = plan.estimates
        mem = plan.device_memory
        row.update(
            free_mib = mem.free_mib,
            total_mib = mem.total_mib,
            budget_mib = est.get("safe_device_budget_mib"),
            weights_mib = est.get("model_dense_mib"),
            companions_mib = est.get("companion_dense_mib"),
            overhead_mib = est.get("base_overhead_mib"),
        )
    print("AB " + json.dumps(row), flush = True)
