"""Studio's own DiffusionBackend: Qwen-Image-2.1 GGUF Q4_K_M, Auto, optionally under a simulated 20 GB discrete card."""
import argparse
import json
import os
import sys
import threading
import time
import traceback
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--case", required=True)
p.add_argument("--output", required=True)
p.add_argument("--free-gib", type=float, default=0)
p.add_argument("--te", default="auto")
p.add_argument("--steps", type=int, default=4)
p.add_argument("--repeats", type=int, default=2)
p.add_argument("--size", type=int, default=1024)
p.add_argument("--serialize", action="store_true")
p.add_argument("--disable-fused-sdpa", action="store_true")
p.add_argument("--memory-mode", default="auto")
p.add_argument("--env", action="append", default=[])
args = p.parse_args()
out = Path(args.output)
out.mkdir(parents=True, exist_ok=True)
lock = threading.Lock()


def emit(event, **fields):
    line = json.dumps(dict(event=event, case=args.case, t=time.time(), **fields), default=str)
    with lock:
        print("PROBE " + line, flush=True)
        with (out / (args.case + ".jsonl")).open("a", encoding="utf-8") as f:
            f.write(line + "\n")


for kv in args.env:
    k, _, v = kv.partition("=")
    os.environ[k] = v
if args.serialize:
    os.environ["AMD_SERIALIZE_KERNEL"] = "1"
    os.environ["HIP_LAUNCH_BLOCKING"] = "1"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import main as studio_main  # Studio's own startup env (AOTriton opt-in, etc.)

import torch
import psutil

emit("env", main=studio_main.__file__, torch=torch.__version__, hip=torch.version.hip,
     props=str(torch.cuda.get_device_properties(0)),
     env={k: os.environ.get(k) for k in ("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "MIOPEN_SEARCH_CUTOFF",
                                         "AMD_SERIALIZE_KERNEL", "PYTORCH_ALLOC_CONF", "PYTORCH_HIP_ALLOC_CONF")},
     sdp=dict(flash=torch.backends.cuda.flash_sdp_enabled(), mem=torch.backends.cuda.mem_efficient_sdp_enabled(),
              math=torch.backends.cuda.math_sdp_enabled()),
     blas=str(torch.backends.cuda.preferred_blas_library()))
if args.disable_fused_sdpa:
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
from utils.hardware import ensure_hardware_detected

ensure_hardware_detected()
emit("after_hw_detect", env={k: os.environ.get(k) for k in ("MIOPEN_SEARCH_CUTOFF", "MIOPEN_FIND_MODE")})
from core.inference import diffusion as d
from core.inference import diffusion_memory as dm
from core.inference.diffusion_attention import available_sdpa_kernels
from core.inference.diffusion_device import resolve_diffusion_device_target

target = resolve_diffusion_device_target()
emit("target", target=target.as_public_dict(), studio_sdpa_probe=available_sdpa_kernels(target))
props = torch.cuda.get_device_properties(0)
if args.free_gib:
    cap = int(args.free_gib * 2**30)
    torch.cuda.set_per_process_memory_fraction(cap / props.total_memory, 0)

    def virtual_memory(backend):
        physical_free, _ = torch.cuda.mem_get_info()
        return min(physical_free, max(0, cap - torch.cuda.memory_reserved())) // 2**20, 20 * 1024, "discrete_vram"

    dm._cuda_memory = virtual_memory
    emit("synthetic_capacity", total_gib=20, free_gib=args.free_gib)
orig_plan = d.plan_diffusion_memory


def record_plan(**kw):
    plan = orig_plan(**kw)
    emit("plan", plan=plan.as_public_dict())
    return plan


d.plan_diffusion_memory = record_plan
proc = psutil.Process()


def mem():
    return dict(alloc=torch.cuda.memory_allocated(), reserved=torch.cuda.memory_reserved(),
                peak=torch.cuda.max_memory_allocated(), rss=proc.memory_info().rss)


backend = d.DiffusionBackend()
try:
    t0 = time.perf_counter()
    backend.begin_load("unsloth/Qwen-Image-2.1-GGUF", gguf_filename="qwen-image-2.1-Q4_K_M.gguf",
                       family_override="qwen-image-2.1", memory_mode=args.memory_mode,
                       **({} if args.te == "auto" else {"text_encoder_quant": args.te}))
    deadline = time.monotonic() + 3000
    while True:
        pr = backend.load_progress()
        if pr.get("error"):
            raise RuntimeError(pr["error"])
        if pr.get("phase") == "ready":
            break
        if time.monotonic() > deadline:
            raise TimeoutError("load > 50 min")
        time.sleep(10)
    emit("loaded", seconds=time.perf_counter() - t0, status=backend.status(), mem=mem())
    prompts = ["A red ceramic teapot on a wooden table beside a window, soft daylight, photograph.",
               "A blue ceramic teapot on a wooden table beside a window, soft daylight, photograph.",
               "A chalkboard menu that reads FRESH COFFEE 2.50 in a small cafe."]
    for i in range(args.repeats):
        t1 = time.perf_counter()
        res = backend.generate(prompt=prompts[i % len(prompts)], width=args.size, height=args.size,
                               steps=args.steps, guidance=1.0, seed=42)
        torch.cuda.synchronize()
        for j, im in enumerate(res["images"]):
            im.save(out / f"{args.case}-{i}-{j}.png")
        emit("render", index=i, seconds=time.perf_counter() - t1, mem=mem(), status=backend.status())
    emit("complete", success=True)
except Exception as exc:  # noqa: BLE001
    emit("failure", error=repr(exc)[:2000], traceback=traceback.format_exc()[-8000:], mem=None)
    raise
finally:
    try:
        backend.unload()
    except Exception:  # noqa: BLE001
        pass
