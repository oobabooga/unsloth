"""Profile Wan2.2-TI2V-5B LoRA training steps through Studio's own trainer (run_dit_lora_training), Studio UI defaults
(base_precision auto, compile auto, rank 32 / alpha = rank, lr 1e-4, constant_with_warmup 20, 704 px stills).

Per-step wall time comes from the trainer's progress events (each follows a host sync: loss .item() and the grad-norm
float). A torch.profiler window over steps (wait+warmup+1 .. wait+warmup+active) classifies device kernel time into
attention backward / attention forward / the rest, so the attention-backward share of a step is measured, not guessed.

usage: python wan_step_profile.py --backend <tree>/studio/backend --out obs.json [--res 704] [--steps 20]
"""

import argparse
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--backend", required = True)
ap.add_argument("--out", required = True)
ap.add_argument("--work", required = True)
ap.add_argument("--res", type = int, default = 704)
ap.add_argument("--steps", type = int, default = 20)
ap.add_argument("--prof-wait", type = int, default = 10)
ap.add_argument("--prof-active", type = int, default = 3)
ap.add_argument("--no-profile", action = "store_true")
args = ap.parse_args()

os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")  # Studio's main.py sets it for the server
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("UNSLOTH_NF4_DEQUANT_CACHE", "0")  # the training worker's default
for k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
    os.environ.pop(k, None)
sys.path.insert(0, args.backend)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

work = Path(args.work)
data = work / "data"
data.mkdir(parents = True, exist_ok = True)
rng = np.random.default_rng(0)
for i in range(8):
    yy, xx = np.mgrid[0:args.res, 0:args.res]
    cx, cy = rng.integers(args.res // 4, 3 * args.res // 4, size = 2)
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    ring = ((r // 24) % 2).astype(np.uint8)
    img = np.stack([ring * 240 + 10, ring * 40 + 20, (1 - ring) * 200 + 30], -1).astype(np.uint8)
    Image.fromarray(img).save(data / f"img{i}.png")
    (data / f"img{i}.txt").write_text("a ringstyle pattern", encoding = "utf-8")

from core.training.diffusion_train_common import DiffusionLoraConfig  # noqa: E402
from core.training.diffusion_dit_trainer import run_dit_lora_training  # noqa: E402

obs: dict = {
    "backend": args.backend,
    "gpu": torch.cuda.get_device_name(0),
    "torch": torch.__version__,
    "hip": torch.version.hip,
    "res": args.res,
    "steps": args.steps,
}
events: list = []
step_t: dict = {}

prof = None
WAIT, WARM, ACTIVE = args.prof_wait, 1, args.prof_active
if not args.no_profile:
    from torch.profiler import ProfilerActivity, profile, schedule

    prof = profile(
        activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule = schedule(wait = WAIT, warmup = WARM, active = ACTIVE, repeat = 1),
    )
    prof.__enter__()


def on_event(ev):
    flat = {k: v for k, v in ev.items() if isinstance(v, (int, float, str, type(None)))}
    events.append(flat)
    if ev.get("type") == "progress":
        torch.cuda.synchronize()
        step_t[int(ev["step"])] = time.perf_counter()
        if prof is not None:
            prof.step()


cfg = DiffusionLoraConfig(
    base_model = "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
    data_dir = str(data),
    output_dir = str(work / "out"),
    resolution = args.res,
    train_steps = args.steps,
    lora_rank = 32,
    learning_rate = 1e-4,
    lr_scheduler = "constant_with_warmup",
    lr_warmup_steps = 20,
    model_family = "wan2.2-ti2v-5b",
    base_precision = "auto",
    compile_transformer = "auto",
)
torch.cuda.reset_peak_memory_stats()
t0 = time.time()
try:
    run_dit_lora_training(cfg, on_event = on_event)
    obs["train_ok"] = True
except Exception as exc:  # noqa: BLE001
    import traceback

    obs["train_ok"] = False
    obs["train_error"] = f"{type(exc).__name__}: {exc}"[:1500]
    obs["traceback"] = traceback.format_exc()[-4000:]
obs["train_wall_s"] = round(time.time() - t0, 1)
obs["peak_alloc_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
obs["peak_reserved_gib"] = round(torch.cuda.max_memory_reserved() / 2**30, 2)
prog = [e for e in events if e.get("type") == "progress"]
obs["losses"] = [e.get("loss") for e in prog]
obs["peak_memory_gb_events"] = [e.get("peak_memory_gb") for e in prog][-1:] or None
obs["non_progress_events"] = [e for e in events if e.get("type") not in ("progress", "preparing")][:60]
steps = sorted(step_t)
dts = {s: step_t[s] - step_t[s - 1] for s in steps if s - 1 in step_t}
obs["step_dt_s"] = {str(s): round(v, 4) for s, v in dts.items()}
prof_steps = set(range(WAIT + WARM + 1, WAIT + WARM + ACTIVE + 1)) if prof is not None else set()
clean = [v for s, v in dts.items() if s >= 3 and s not in prof_steps and s - 1 not in prof_steps]
obs["s_per_step_median_clean"] = round(statistics.median(clean), 4) if clean else None
obs["s_per_step_clean_n"] = len(clean)

ATTN_BWD = re.compile(r"bwd_kernel|attn_bwd|bwd_preprocess|flash.*bwd|fmha.*bwd|_bwd_kernel|feather", re.I)
ATTN_FWD = re.compile(r"attn_fwd|fwd_kernel|flash.*fwd|fmha.*fwd", re.I)
ATTN_OPS_BWD = (
    "aten::_scaled_dot_product_flash_attention_backward",
    "aten::_scaled_dot_product_efficient_attention_backward",
    "aten::_flash_attention_backward",
    "aten::_efficient_attention_backward",
)
if prof is not None:
    try:
        prof.__exit__(None, None, None)
        from torch.autograd import DeviceType

        kern: dict = {}
        for e in prof.events():
            if e.device_type == DeviceType.CUDA:
                kern[e.name] = kern.get(e.name, 0.0) + e.device_time_total
        total = sum(kern.values())
        bwd = sum(v for k, v in kern.items() if ATTN_BWD.search(k))
        fwd = sum(v for k, v in kern.items() if ATTN_FWD.search(k) and not ATTN_BWD.search(k))
        win = [s for s in range(WAIT + WARM + 1, WAIT + WARM + ACTIVE + 1)]
        win_wall = (step_t[win[-1]] - step_t[win[0] - 1]) if all(s in step_t for s in win + [win[0] - 1]) else None
        op_bwd = {}
        for ka in prof.key_averages():
            if ka.key in ATTN_OPS_BWD or "attention_backward" in ka.key:
                op_bwd[ka.key] = {"count": ka.count, "device_time_total_us": ka.device_time_total}
        obs["profile"] = {
            "window_steps": win,
            "window_wall_s": round(win_wall, 4) if win_wall else None,
            "kernel_total_s": round(total / 1e6, 4),
            "attn_bwd_kernel_s": round(bwd / 1e6, 4),
            "attn_fwd_kernel_s": round(fwd / 1e6, 4),
            "attn_bwd_share_of_wall": round(bwd / 1e6 / win_wall, 4) if win_wall else None,
            "attn_bwd_share_of_kernel_time": round(bwd / total, 4) if total else None,
            "attn_fwd_share_of_wall": round(fwd / 1e6 / win_wall, 4) if win_wall else None,
            "attn_bwd_ops": op_bwd,
            "attn_bwd_kernels": {k: round(v / 1e3, 2) for k, v in kern.items() if ATTN_BWD.search(k)},
            "top_kernels_ms": [[k[:160], round(v / 1e3, 2)] for k, v in sorted(kern.items(), key = lambda kv: -kv[1])[:60]],
        }
    except Exception as exc:  # noqa: BLE001
        obs["profile_error"] = f"{type(exc).__name__}: {exc}"[:800]

Path(args.out).write_text(json.dumps(obs, indent = 1), encoding = "utf-8")
print("WROTE", args.out, "train_ok", obs.get("train_ok"), "s/step", obs.get("s_per_step_median_clean"),
      "share", (obs.get("profile") or {}).get("attn_bwd_share_of_wall"))
