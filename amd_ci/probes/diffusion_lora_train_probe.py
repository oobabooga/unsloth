#!/usr/bin/env python3
"""Probe: diffusion LoRA training end to end through Studio's own code on this GPU.

Observes only (judged by criteria/diffusion_lora_train.py). Per state, every stage runs in its own subprocess so
device memory starts clean and a crash is contained:

  host     torch / HIP / arch / memory / disk facts, ``train_precision_modes()``, whether the target family is
           accepted for training (``DiffusionLoraConfig.normalized()``) and the start route's per-precision preflight
           (``training_precision_preflight_error``) for nf4 / bf16 / int8 / fp8 / mxfp8 / auto
  train    one run per offered dense/nf4 precision through ``run_diffusion_training_process`` (the training worker's
           entry point: config dict -> normalized() -> get_trainer(family) -> trainer). Records every event, the
           trainer's own samples/s and peak, process max allocated / reserved, the auto pick, and the saved adapter's
           tensor norms
  render   ``DiffusionBackend.load_pipeline`` on the deploy repo in a fresh process, then ``generate`` with no LoRA
           twice (the A/A noise arm), with each trained LoRA by its catalog id, and with the copy written by the LoRA
           export route's function (``export_local_lora``) into a scanned folder, rediscovered by id
           (the round trip). LPIPS (alex) and mean abs pixel difference between renders.

The control family (an already trainable one) runs only in ``--control-state`` so a failure can be attributed to the
new trainer or to the ROCm stack. Results go to ``--out`` as JSON, never stdout.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

ALL_PRECISIONS = ("nf4", "bf16", "int8", "fp8", "mxfp8", "auto")

# Each suite: the family under test, plus an already-trainable control. Repos are public (no token on the runner):
# the trainer and loader map a gated upstream to its ungated unsloth mirror themselves.
SUITES = {
    "flux3": {
        "target": {
            "key": "flux.3-klein-base",
            "base_model": "black-forest-labs/flux-3-image-klein-base",
            "deploy": "black-forest-labs/flux-3-image-klein",
            "render_steps": 16,
            "guidance": 1.0,
        },
        "control": {
            "key": "flux.2-klein",
            "base_model": "black-forest-labs/FLUX.2-klein-base-4B",
            "deploy": "black-forest-labs/FLUX.2-klein-base-4B",
            "render_steps": 20,
            "guidance": 5.0,
        },
    },
    "qwen21": {
        "target": {
            "key": "qwen-image-2.1",
            "base_model": "Qwen/Qwen-Image-2.1",
            "deploy": "Qwen/Qwen-Image-2.1",
            "render_steps": 20,
            "guidance": 1.0,
        },
        "control": None,
    },
    "h3image": {
        "target": {
            "key": "minimax-h3-image",
            "base_model": "MiniMaxAI/MiniMax-H3-Image",
            "deploy": "MiniMaxAI/MiniMax-H3-Image",
            "render_steps": 6,
            "guidance": 1.0,
            # Studio merges H3-Image adapters into the DiT at load, so each LoRA render is its own load.
            "load_time_lora": True,
        },
        "control": None,
    },
}

DATASET = "multimodalart/1920-raider-waite-tarot-public-domain"
TRIGGER_PROMPT = 'a trtcrd of a cat wearing a crown sitting on a throne, holding a golden cup, "queen of cups"'
SEED = 1234


def _write(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_text(json.dumps(obj, indent = 2, default = str), encoding = "utf-8")


def _read(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding = "utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {"_read_error": f"{type(exc).__name__}: {exc}"}


def _meminfo() -> dict:
    out = {}
    try:
        for line in Path("/proc/meminfo").read_text(encoding = "utf-8").splitlines():
            k, v = line.split(":", 1)
            if k in ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree"):
                out[k] = round(int(v.split()[0]) / 1024 / 1024, 2)  # GiB
    except Exception:  # noqa: BLE001
        pass
    return out


class _HostMemWatch:
    """Lowest MemAvailable seen during a stage: on a unified-memory APU the host pool IS the GPU pool."""

    def __init__(self):
        self.min_avail = None
        self._stop = threading.Event()
        self._t = threading.Thread(target = self._run, daemon = True)

    def _run(self):
        while not self._stop.is_set():
            a = _meminfo().get("MemAvailable")
            if a is not None and (self.min_avail is None or a < self.min_avail):
                self.min_avail = a
            self._stop.wait(1.0)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join(timeout = 3)


def _studio_env(backend: Path) -> None:
    """What Studio's main.py sets before any GPU use; the training / inference workers inherit it."""
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    if not os.environ.get("AMD_CI_KEEP_AOTRITON_ENV"):
        os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    sys.path.insert(0, str(backend))
    os.chdir(backend)
    try:
        from utils.rocm_flash_attn import default_triton_amd_env

        default_triton_amd_env()
    except Exception:  # noqa: BLE001 -- absent at older commits
        pass


# ----------------------------------------------------------------------------------------------------------- stages


def stage_host(args) -> dict:
    res: dict = {"meminfo": _meminfo()}
    try:
        res["disk_free_gb"] = round(shutil.disk_usage(os.environ.get("HF_HOME", "/")).free / 1e9, 1)
    except Exception:  # noqa: BLE001
        pass
    import torch

    res["torch"] = torch.__version__
    res["hip"] = getattr(torch.version, "hip", None)
    res["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        res["device_name"] = torch.cuda.get_device_name(0)
        res["arch"] = getattr(p, "gcnArchName", None)
        res["total_memory_gb"] = round(p.total_memory / 1e9, 2)
        free, total = torch.cuda.mem_get_info()
        res["mem_get_info_gb"] = [round(free / 1e9, 2), round(total / 1e9, 2)]
        res["bf16_supported"] = torch.cuda.is_bf16_supported()
    for mod in ("triton", "bitsandbytes", "torchao", "diffusers", "transformers", "peft", "accelerate"):
        try:
            m = __import__(mod)
            res[f"{mod}_version"] = getattr(m, "__version__", "?")
        except Exception as exc:  # noqa: BLE001
            res[f"{mod}_version"] = f"IMPORT FAILED: {type(exc).__name__}: {str(exc)[:200]}"
    if torch.cuda.is_available():
        try:
            res["kernel_check"] = _kernel_check(torch)
        except Exception as exc:  # noqa: BLE001
            res["kernel_check"] = {"error": f"{type(exc).__name__}: {exc}"}
    if torch.cuda.is_available() and args.vae_check:
        try:
            res["vae_check"] = _vae_check(torch, args)
        except Exception as exc:  # noqa: BLE001
            res["vae_check"] = {"error": f"{type(exc).__name__}: {str(exc)[:500]}",
                                "traceback": traceback.format_exc()[-2000:]}
    from core.training import diffusion_train_common as dtc

    modes, rec = dtc.train_precision_modes()
    res["train_precision_modes"] = modes
    res["train_precision_recommended"] = rec
    try:
        res["has_functional_torchao"] = dtc.has_functional_torchao()
    except Exception as exc:  # noqa: BLE001
        res["has_functional_torchao"] = f"error: {exc}"
    try:
        from utils.hardware import trusted_mem_get_info

        f, t = trusted_mem_get_info()
        res["trusted_mem_get_info_gb"] = [round(f / 1e9, 2), round(t / 1e9, 2)]
    except Exception as exc:  # noqa: BLE001
        res["trusted_mem_get_info_gb"] = f"error: {type(exc).__name__}: {exc}"

    fams = {}
    for role in ("target", "control"):
        spec = SUITES[args.suite][role]
        if not spec:
            continue
        f: dict = {"base_model": spec["base_model"]}
        try:
            cfg = dtc.DiffusionLoraConfig(
                base_model = spec["base_model"], data_dir = "d", output_dir = "o", resolution = args.resolution
            ).normalized()
            f["accepted"] = True
            f["resolved_family"] = cfg.resolved_family
            f["fetch_base_model"] = cfg.fetch_base_model
        except Exception as exc:  # noqa: BLE001
            f["accepted"] = False
            f["refusal"] = f"{type(exc).__name__}: {str(exc)[:600]}"
        pre = {}
        fam_name = f.get("resolved_family") or spec["key"]
        fn = getattr(dtc, "training_precision_preflight_error", None)
        for p in ALL_PRECISIONS:
            try:
                pre[p] = fn(fam_name, p) if fn else "n/a (no preflight at this commit)"
            except Exception as exc:  # noqa: BLE001
                pre[p] = f"preflight raised {type(exc).__name__}: {exc}"
        f["preflight"] = pre
        fams[role] = f
    res["families"] = fams
    return res


def _kernel_check(torch) -> dict:
    """Attention and GEMM at DiT shapes against an fp32 reference, per SDPA backend, twice each (determinism). Observes
    whether this torch build computes the two ops every DiT forward is made of correctly on this GPU."""
    import torch.nn.functional as F

    out: dict = {"aotriton_experimental": os.environ.get("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"),
                 "sdpa_flags": {n: bool(getattr(torch.backends.cuda, f"{n}_sdp_enabled")())
                                for n in ("flash", "mem_efficient", "math")}}
    g = torch.Generator(device = "cuda").manual_seed(0)
    q, k, v = (torch.randn(1, 24, 1536, 128, device = "cuda", generator = g) for _ in range(3))
    ref = F.scaled_dot_product_attention(q, k, v)  # fp32 inputs
    qb, kb, vb = q.bfloat16(), k.bfloat16(), v.bfloat16()

    def err(x):
        x = x.float()
        return {"finite": bool(torch.isfinite(x).all()),
                "max_abs": round(float((x - ref).abs().max()), 5),
                "rel_l2": round(float((x - ref).norm() / ref.norm()), 6)}

    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
    except Exception:  # noqa: BLE001
        SDPBackend = sdpa_kernel = None
    arms = [("default", None)]
    if SDPBackend is not None:
        arms += [("flash", SDPBackend.FLASH_ATTENTION), ("efficient", SDPBackend.EFFICIENT_ATTENTION),
                 ("math", SDPBackend.MATH)]
    for name, be in arms:
        try:
            ctx = sdpa_kernel([be]) if be is not None else _null()
            with ctx:
                a = F.scaled_dot_product_attention(qb, kb, vb)
                b = F.scaled_dot_product_attention(qb, kb, vb)
            torch.cuda.synchronize()
            out[f"sdpa_{name}"] = {**err(a), "repeat_identical": bool(torch.equal(a, b))}
        except Exception as exc:  # noqa: BLE001
            out[f"sdpa_{name}"] = {"error": f"{type(exc).__name__}: {str(exc)[:200]}"}
    x = torch.randn(4096, 3072, device = "cuda", generator = g)
    w = torch.randn(3072, 3072, device = "cuda", generator = g) / 55.0
    r = x @ w
    y1 = x.bfloat16() @ w.bfloat16()
    y2 = x.bfloat16() @ w.bfloat16()
    torch.cuda.synchronize()
    out["gemm_bf16"] = {"finite": bool(torch.isfinite(y1).all()),
                        "rel_l2": round(float((y1.float() - r).norm() / r.norm()), 6),
                        "repeat_identical": bool(torch.equal(y1, y2))}
    return out


def _vae_check(torch, args) -> dict:
    """Stock diffusers FLUX.2 VAE (the FLUX.2 / FLUX.3 latent space) encode + decode of a training image: CPU fp32 is
    the reference; GPU fp32 and bf16 each with MIOpen immediate mode off and on (Studio's inference turns it on).
    Observes whether the convolution stack this torch build runs on this GPU reproduces the CPU answer."""
    import numpy as np
    from diffusers import AutoencoderKLFlux2
    from PIL import Image

    img_path = next(p for p in sorted(Path(args.data_dir).iterdir()) if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    im = Image.open(img_path).convert("RGB").resize((512, 512))
    x = torch.from_numpy(np.asarray(im, dtype = "float32")).permute(2, 0, 1)[None].div(127.5).sub(1)
    vae = AutoencoderKLFlux2.from_pretrained("black-forest-labs/FLUX.2-klein-base-4B", subfolder = "vae",
                                             torch_dtype = torch.float32, token = False).eval()

    def run(device, dtype):
        v = vae.to(device = device, dtype = dtype)
        with torch.no_grad():
            lat = v.encode(x.to(device, dtype)).latent_dist.mode()
            rec = v.decode(lat).sample
        if device != "cpu":
            torch.cuda.synchronize()
        return lat.float().cpu(), rec.float().cpu().clamp(-1, 1)

    def psnr(a, b):
        mse = float(((a - b) ** 2).mean()) / 4.0  # [-1, 1] -> [0, 1]
        return round(10 * math.log10(1.0 / max(mse, 1e-12)), 2)

    out: dict = {"image": img_path.name}
    ref_lat, ref_rec = run("cpu", torch.float32)
    out["cpu_fp32"] = {"psnr_vs_input": psnr(ref_rec, x)}
    miopen = getattr(torch.backends, "miopen", None)
    for imm in (False, True):
        if miopen is not None and hasattr(miopen, "immediate"):
            miopen.immediate = imm
        for name, dt in (("fp32", torch.float32), ("bf16", torch.bfloat16)):
            try:
                lat, rec = run("cuda", dt)
                out[f"gpu_{name}_immediate{int(imm)}"] = {
                    "finite": bool(torch.isfinite(rec).all()),
                    "latent_rel_l2_vs_cpu": round(float((lat - ref_lat).norm() / ref_lat.norm()), 5),
                    "psnr_vs_input": psnr(rec, x),
                    "psnr_vs_cpu_recon": psnr(rec, ref_rec),
                }
            except Exception as exc:  # noqa: BLE001
                out[f"gpu_{name}_immediate{int(imm)}"] = {"error": f"{type(exc).__name__}: {str(exc)[:200]}"}
    vae.to("cpu")
    torch.cuda.empty_cache()
    if miopen is not None and hasattr(miopen, "immediate"):
        miopen.immediate = False
    out["ops"] = _vae_ops_check(torch)
    return out


def _vae_ops_check(torch) -> dict:
    """The VAE's building blocks at its own shapes (512 px image), GPU vs CPU fp32: which op disagrees."""
    import torch.nn.functional as F

    g = torch.Generator().manual_seed(0)
    res: dict = {}

    def rel(a, b):
        a, b = a.float().cpu(), b.float().cpu()
        return round(float((a - b).norm() / max(float(b.norm()), 1e-12)), 6)

    def case(name, fn, *tensors):
        try:
            ref = fn(*[t.float() for t in tensors])
            for dt in (torch.float32, torch.bfloat16):
                got = fn(*[t.to("cuda", dt) for t in tensors])
                torch.cuda.synchronize()
                res[f"{name}_{str(dt).split('.')[-1]}"] = rel(got, ref)
        except Exception as exc:  # noqa: BLE001
            res[name] = f"error: {type(exc).__name__}: {str(exc)[:160]}"

    for c, hw in ((128, 512), (256, 256), (512, 128), (512, 64)):
        x = torch.randn(1, c, hw, hw, generator = g)
        w = torch.randn(c, c, 3, 3, generator = g) / (3 * c ** 0.5)
        case(f"conv3x3_c{c}_{hw}", lambda a, b: F.conv2d(a, b, padding = 1), x, w)
    x = torch.randn(1, 3, 512, 512, generator = g)
    w = torch.randn(128, 3, 3, 3, generator = g) / 5
    case("conv_in_3to128_512", lambda a, b: F.conv2d(a, b, padding = 1), x, w)
    x = torch.randn(1, 128, 513, 513, generator = g)
    w = torch.randn(128, 128, 3, 3, generator = g) / 34
    case("conv_down_s2_c128", lambda a, b: F.conv2d(a, b, stride = 2), x, w)
    x = torch.randn(1, 512, 64, 64, generator = g)
    w = torch.randn(32, 512, 1, 1, generator = g) / 22
    case("conv1x1_512to32_64", lambda a, b: F.conv2d(a, b), x, w)
    x = torch.randn(1, 512, 64, 64, generator = g)
    case("group_norm32_c512_64", lambda a: F.group_norm(a, 32), x)
    x = torch.randn(1, 128, 256, 256, generator = g)
    case("group_norm32_c128_256", lambda a: F.group_norm(a, 32), x)
    q, k, v = (torch.randn(1, 1, 4096, 512, generator = g) for _ in range(3))
    case("sdpa_1head_d512_4096", lambda a, b, c: F.scaled_dot_product_attention(a, b, c), q, k, v)
    x = torch.randn(1, 512, 64, 64, generator = g)
    case("upsample_nearest2x_c512", lambda a: F.interpolate(a, scale_factor = 2.0, mode = "nearest"), x)
    return res


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _Q:
    """The two multiprocessing queues run_diffusion_training_process talks to, in-process."""

    def __init__(self, sink = None):
        self.items: list = []
        self.sink = sink

    def put(self, item):
        self.items.append(item)
        if self.sink:
            self.sink(item)

    def empty(self):
        return True

    def get_nowait(self):
        raise RuntimeError("empty")


def _lora_norms(path: str) -> dict:
    from safetensors.torch import load_file

    sd = load_file(path)
    a = [v.float().norm().item() for k, v in sd.items() if "lora_A" in k or "lora_down" in k]
    b = [v.float().norm().item() for k, v in sd.items() if "lora_B" in k or "lora_up" in k]
    finite = all(math.isfinite(x) for x in a + b)
    return {
        "tensors": len(sd),
        "lora_A_count": len(a),
        "lora_B_count": len(b),
        "lora_B_norm_sum": round(sum(b), 6) if b else None,
        "lora_B_nonzero": sum(1 for x in b if x > 0),
        "finite": finite,
        "bytes": os.path.getsize(path),
    }


def stage_train(args) -> dict:
    spec = SUITES[args.suite][args.role]
    import torch

    from core.training import diffusion_lora_trainer as dlt
    from core.training import diffusion_train_common as dtc

    res: dict = {"precision": args.precision, "role": args.role, "base_model": spec["base_model"]}
    events_path = Path(args.stage_out).with_suffix(".events.jsonl")
    ev_fh = open(events_path, "w", encoding = "utf-8")

    def sink(ev):
        ev_fh.write(json.dumps(ev, default = str) + "\n")
        ev_fh.flush()
        t = ev.get("type")
        if t == "progress":
            if ev.get("step") in (1, 2, 3) or ev.get("step", 0) % 10 == 0:
                print("progress", {k: ev.get(k) for k in ("step", "loss", "samples_per_second", "peak_memory_gb")},
                      flush = True)
        elif t not in ("preparing", "download_progress"):
            print("event", json.dumps(ev, default = str)[:600], flush = True)

    out_dir = Path(args.work) / "runs" / f"{args.role}_{args.precision}"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    config = {
        "base_model": spec["base_model"],
        "data_dir": args.data_dir,
        "output_dir": str(out_dir),
        "resolution": args.resolution,
        "train_steps": args.steps,
        "learning_rate": args.lr,
        "lora_rank": args.rank,
        "lr_scheduler": "constant_with_warmup",
        "lr_warmup_steps": 5,
        "base_precision": args.precision.split("-")[0],
        "compile_transformer": "off" if args.precision.endswith("-eager") else "auto",
        "seed": 42,
        "adapter_name": f"amdci_{args.suite}_{args.role}_{args.precision}",
        "log_every": 1,
        "hf_token": None,
    }
    res["config"] = config
    # What "auto" would pick on this host right now (pure policy over free memory), without a second run.
    try:
        from core.training import diffusion_dit_trainer as ddt

        cfg = dtc._config_from_dict({**config, "base_precision": "auto"}).normalized()
        sp = ddt._SPECS.get(cfg.resolved_family)
        if sp is not None:
            res["auto_pick"] = ddt._resolve_base_precision(cfg, sp, "cuda")
    except Exception as exc:  # noqa: BLE001
        res["auto_pick"] = f"n/a: {type(exc).__name__}: {str(exc)[:200]}"

    gpu = torch.cuda.is_available()
    if gpu:
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    q_ev, q_stop = _Q(sink), _Q()
    with _HostMemWatch() as watch:
        try:
            dlt.run_diffusion_training_process(event_queue = q_ev, stop_queue = q_stop, config = config)
        except Exception as exc:  # noqa: BLE001 -- the worker catches its own; this is belt and braces
            res["raised"] = f"{type(exc).__name__}: {exc}"
            res["traceback"] = traceback.format_exc()[-4000:]
    ev_fh.close()
    res["wall_s"] = round(time.time() - t0, 1)
    res["host_min_available_gib"] = watch.min_avail
    if gpu:
        res["max_allocated_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
        res["max_reserved_gb"] = round(torch.cuda.max_memory_reserved() / 1e9, 2)
    events = q_ev.items
    prog = [e for e in events if e.get("type") == "progress"]
    res["losses"] = [e.get("loss") for e in prog]
    res["grad_norms"] = [e.get("grad_norm") for e in prog]
    res["steps_seen"] = len(prog)
    if prog:
        res["samples_per_second"] = prog[-1].get("samples_per_second")
        res["trainer_peak_memory_gb"] = max((e.get("peak_memory_gb") or 0) for e in prog)
    res["errors"] = [e.get("message") or e.get("error") for e in events if e.get("type") == "error"]
    res["warnings"] = [str(e.get("message"))[:400] for e in events if e.get("type") == "warning"]
    loaded = [e for e in events if e.get("type") == "model_load_completed"]
    if loaded:
        res["compiled"] = loaded[-1].get("compiled")
        res["base_precision_engaged"] = loaded[-1].get("base_precision")
    done = [e for e in events if e.get("type") == "complete"]
    if done:
        c = done[-1]
        res["complete"] = {k: c.get(k) for k in ("lora_path", "catalog_path", "stopped", "steps_run", "wall_seconds",
                                                 "family")}
        lp = c.get("lora_path")
        if lp and os.path.isfile(lp):
            res["lora_norms"] = _lora_norms(lp)
        cp = c.get("catalog_path")
        if cp:
            res["catalog_id"] = Path(cp).stem
            sidecar = Path(cp).with_suffix(".json")
            if sidecar.is_file():
                res["catalog_sidecar"] = _read(sidecar)
    return res


def _lpips_fn():
    try:
        import lpips  # noqa: F401
    except Exception:  # noqa: BLE001
        # The Studio venv may be uv-made without pip: try pip, then ensurepip + pip, then uv.
        attempts = [[sys.executable, "-m", "pip", "install", "-q", "--no-deps", "lpips"],
                    [sys.executable, "-m", "ensurepip", "-q"],
                    [sys.executable, "-m", "pip", "install", "-q", "--no-deps", "lpips"]]
        uv = shutil.which("uv") or next((str(c) for c in (Path.home() / ".local" / "bin" / "uv",
                                                         Path.home() / ".cargo" / "bin" / "uv") if c.is_file()), None)
        if uv:
            attempts.append([uv, "pip", "install", "-q", "--no-deps", "--python", sys.executable, "lpips"])
        for cmd in attempts:
            subprocess.run(cmd, check = False, stdout = subprocess.DEVNULL, stderr = subprocess.DEVNULL)
            try:
                import importlib

                importlib.invalidate_caches()
                import lpips  # noqa: F401

                break
            except Exception:  # noqa: BLE001
                continue
    import lpips
    import torch

    net = lpips.LPIPS(net = "alex", verbose = False).eval()

    def t(img):
        import numpy as np

        a = np.asarray(img.convert("RGB"), dtype = "float32")
        return torch.from_numpy(a).permute(2, 0, 1)[None].div(127.5).sub(1)

    def score(x, y):
        with torch.no_grad():
            return round(float(net(t(x), t(y))), 5)

    return score


def _mad(x, y) -> float:
    import numpy as np

    a = np.asarray(x.convert("RGB"), dtype = "float32")
    b = np.asarray(y.convert("RGB"), dtype = "float32")
    return round(float(abs(a - b).mean()), 4)


def stage_render(args) -> dict:
    spec = SUITES[args.suite][args.role]
    import torch

    from core.inference.diffusion import DiffusionBackend
    from core.inference import diffusion_lora as dl

    lora_ids = [x for x in (args.lora_ids or "").split(",") if x]
    res: dict = {"deploy": spec["deploy"], "loras": {}, "renders": {}}
    img_dir = Path(args.stage_out).parent / f"img_{args.role}"
    img_dir.mkdir(parents = True, exist_ok = True)
    gpu = torch.cuda.is_available()
    if gpu:
        torch.cuda.reset_peak_memory_stats()
    be = DiffusionBackend()
    t0 = time.time()
    with _HostMemWatch() as watch:
        st = be.load_pipeline(spec["deploy"])
        res["load_s"] = round(time.time() - t0, 1)
        res["load_status"] = {k: st.get(k) for k in ("loaded", "family", "supports_lora", "transformer_quant",
                                                     "text_encoder_quant", "memory_mode", "offload_policy",
                                                     "attention_backend", "speed_mode") if k in st}
        kw = dict(prompt = TRIGGER_PROMPT, width = args.resolution, height = args.resolution,
                  steps = spec["render_steps"], guidance = spec["guidance"], seed = SEED)
        imgs = {}

        def reload(loras):
            be.unload()
            t2 = time.time()
            st2 = be.load_pipeline(spec["deploy"], loras = loras)
            res.setdefault("reload_s", []).append(round(time.time() - t2, 1))
            res.setdefault("reload_status", []).append({k: st2.get(k) for k in ("transformer_quant", "text_encoder_quant")})

        def render(tag, loras = None):
            t1 = time.time()
            try:
                try:
                    r = be.generate(**kw, loras = loras) if loras else be.generate(**kw)
                except ValueError as exc:
                    # A quantized (int8 / fp8) load bakes adapters in at load time, as Deploy does: reload with them.
                    if not loras or "Reload the model with" not in str(exc):
                        raise
                    res.setdefault("baked_reloads", []).append(tag)
                    reload(loras)
                    t1 = time.time()
                    r = be.generate(**kw, loras = loras)
                imgs[tag] = r["images"][0]
                imgs[tag].save(img_dir / f"{tag}.png")
                res["renders"][tag] = {"s": round(time.time() - t1, 1), "ok": True,
                                       "transformer_quant": r.get("transformer_quant")}
            except Exception as exc:  # noqa: BLE001
                res["renders"][tag] = {"ok": False, "error": f"{type(exc).__name__}: {str(exc)[:800]}",
                                       "traceback": traceback.format_exc()[-3000:]}
            print("render", tag, res["renders"][tag], flush = True)

        render("none")
        render("none_repeat")
        export_root = None
        try:
            from utils.hf_cache_settings import get_hf_cache_paths

            export_root = Path(get_hf_cache_paths().cache_home) / "amd_ci_lora_exports"
        except Exception as exc:  # noqa: BLE001
            res["export_root_error"] = f"{type(exc).__name__}: {exc}"
        load_time = bool(spec.get("load_time_lora"))

        for lid in lora_ids:
            info: dict = {}
            if load_time:
                reload([(lid, 1.0)])
            render(f"lora_{lid}", [(lid, 1.0)])
            # The export route's function: copy out of the catalog into a folder Studio scans, then find it by id.
            try:
                src = next((e for e in dl._scan_local() if e.id == lid), None)
                info["catalog_entry"] = {"local_path": getattr(src, "local_path", None),
                                         "families": list(getattr(src, "families", ()) or ())}
                out = dl.export_local_lora(lid, export_root)
                info["export_path"] = str(out)
                same = Path(src.local_path).read_bytes() == Path(out).read_bytes()
                info["export_bytes_identical"] = same
                info["export_sidecar"] = _read(Path(out).with_suffix(".json"))
                again = next((e for e in dl._scan_local()
                              if e.local_path and os.path.realpath(e.local_path) == os.path.realpath(out)), None)
                info["rediscovered_id"] = getattr(again, "id", None)
                info["rediscovered_copy_of"] = getattr(again, "copy_of", None)
                info["rediscovered_families"] = list(getattr(again, "families", ()) or ())
                if again is not None:
                    if load_time:
                        reload([(again.id, 1.0)])
                    render(f"export_{lid}", [(again.id, 1.0)])
            except Exception as exc:  # noqa: BLE001
                info["export_error"] = f"{type(exc).__name__}: {str(exc)[:800]}"
                info["traceback"] = traceback.format_exc()[-3000:]
            res["loras"][lid] = info
    res["host_min_available_gib"] = watch.min_avail
    if gpu:
        res["max_allocated_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
        res["max_reserved_gb"] = round(torch.cuda.max_memory_reserved() / 1e9, 2)
    # The exported copies sit in the runner's HF cache, which can be persistent and shared: leave nothing behind.
    for info in res["loras"].values():
        for key in ("export_path",):
            pth = info.get(key)
            if pth:
                for f in (Path(pth), Path(pth).with_suffix(".json")):
                    try:
                        f.unlink()
                    except OSError:
                        pass
    try:
        be.unload()
    except Exception:  # noqa: BLE001
        pass
    # Scores on CPU, after the pipeline is gone.
    try:
        score = _lpips_fn()
    except Exception as exc:  # noqa: BLE001
        score = None
        res["lpips_error"] = f"{type(exc).__name__}: {exc}"
    cmp = {}

    def pair(name, a, b):
        if a in imgs and b in imgs:
            cmp[name] = {"mad": _mad(imgs[a], imgs[b])}
            if score:
                cmp[name]["lpips"] = score(imgs[a], imgs[b])

    pair("aa_none_vs_repeat", "none", "none_repeat")
    for lid in lora_ids:
        pair(f"lora_vs_none::{lid}", f"lora_{lid}", "none")
        pair(f"export_vs_lora::{lid}", f"export_{lid}", f"lora_{lid}")
    res["compare"] = cmp
    return res


def stage_sdpa(args) -> dict:
    """SDPA per backend and head_dim against a CPU fp32 reference, in a fresh process so the AOTriton experimental
    switch (read once by torch) is what this process's environment says."""
    import torch
    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel

    res: dict = {"aotriton_experimental": os.environ.get("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL")}
    g = torch.Generator().manual_seed(0)
    for heads, tokens, dim in ((24, 1536, 128), (1, 4096, 256), (1, 4096, 512), (1, 1024, 512)):
        q, k, v = (torch.randn(1, heads, tokens, dim, generator = g) for _ in range(3))
        ref = F.scaled_dot_product_attention(q, k, v)
        for dt in (torch.float32, torch.bfloat16):
            for name, be in (("default", None), ("flash", SDPBackend.FLASH_ATTENTION),
                             ("efficient", SDPBackend.EFFICIENT_ATTENTION), ("math", SDPBackend.MATH)):
                key = f"h{heads}_n{tokens}_d{dim}_{str(dt).split('.')[-1]}_{name}"
                try:
                    qc, kc, vc = (t.to("cuda", dt) for t in (q, k, v))
                    with (sdpa_kernel([be]) if be is not None else _null()):
                        o = F.scaled_dot_product_attention(qc, kc, vc)
                    torch.cuda.synchronize()
                    o = o.float().cpu()
                    res[key] = round(float((o - ref).norm() / ref.norm()), 5) if torch.isfinite(o).all() else "nonfinite"
                except Exception as exc:  # noqa: BLE001
                    res[key] = f"refused: {type(exc).__name__}"
    return res


STAGES = {"host": stage_host, "train": stage_train, "render": stage_render, "sdpa": stage_sdpa}


def child_main(args) -> int:
    _studio_env(Path(args.checkout) / "studio" / "backend")
    out = Path(args.stage_out)
    try:
        res = STAGES[args.stage](args)
    except Exception as exc:  # noqa: BLE001
        res = {"stage_error": f"{type(exc).__name__}: {str(exc)[:1500]}", "traceback": traceback.format_exc()[-5000:]}
    _write(out, res)
    return 0


# ----------------------------------------------------------------------------------------------------------- parent


def fetch_dataset(dest: Path, n: int) -> dict:
    if (dest / "done").is_file():
        return _read(dest / "done")
    from huggingface_hub import hf_hub_download

    meta = hf_hub_download(DATASET, "output_file.jsonl", repo_type = "dataset", token = False)
    rows = [json.loads(line) for line in open(meta, encoding = "utf-8") if line.strip()]
    rows = sorted(rows, key = lambda r: r["file_name"])[:n]
    dest.mkdir(parents = True, exist_ok = True)
    for r in rows:
        p = hf_hub_download(DATASET, "train/" + r["file_name"], repo_type = "dataset", token = False)
        stem = Path(r["file_name"]).stem
        shutil.copy(p, dest / (stem + Path(r["file_name"]).suffix.lower()))
        (dest / (stem + ".txt")).write_text(r["text"], encoding = "utf-8")
    info = {"dataset": DATASET, "images": len(rows)}
    _write(dest / "done", info)
    return info


def run_child(args, stage: str, stage_out: Path, timeout_s: int, extra: list[str], env = None) -> dict:
    cmd = [sys.executable, "-u", __file__, "--child", "--stage", stage, "--state", args.state,
           "--checkout", args.checkout, "--out", args.out, "--stage-out", str(stage_out), "--suite", args.suite,
           "--work", args.work, "--data-dir", args.data_dir, "--steps", str(args.steps),
           "--resolution", str(args.resolution), "--lr", str(args.lr), "--rank", str(args.rank),
           *(["--vae-check"] if args.vae_check else []), *extra]
    log = stage_out.with_suffix(".log")
    t0 = time.time()
    with open(log, "wb") as fh:
        try:
            rc = subprocess.run(cmd, stdout = fh, stderr = subprocess.STDOUT, timeout = timeout_s, env = env).returncode
        except subprocess.TimeoutExpired:
            rc = "timeout"
    res = _read(stage_out) if stage_out.is_file() else {"missing_output": True}
    res["_rc"] = rc
    res["_wall_s"] = round(time.time() - t0, 1)
    if rc != 0 or "stage_error" in res or "missing_output" in res:
        try:
            res["_log_tail"] = log.read_text(encoding = "utf-8", errors = "replace")[-3000:]
        except Exception:  # noqa: BLE001
            pass
    print(f"[{args.state}] {stage} {extra} rc={rc} {res['_wall_s']}s", flush = True)
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True)
    ap.add_argument("--suite", choices = sorted(SUITES), default = "flux3")
    ap.add_argument("--control-state", default = "head", help = "state that also trains the control family")
    ap.add_argument("--precisions", default = "bf16,nf4", help = "trained when offered, in this order")
    ap.add_argument("--steps", type = int, default = 40)
    ap.add_argument("--resolution", type = int, default = 512)
    ap.add_argument("--lr", type = float, default = 2e-4)
    ap.add_argument("--rank", type = int, default = 16)
    ap.add_argument("--images", type = int, default = 8)
    ap.add_argument("--budget-min", type = float, default = 85.0, help = "whole-probe budget across all states")
    ap.add_argument("--work", default = "")
    ap.add_argument("--data-dir", default = "")
    ap.add_argument("--vae-check", action = "store_true", help = "host stage: stock FLUX.2 VAE round trip, CPU vs GPU")
    # child-only
    ap.add_argument("--child", action = "store_true")
    ap.add_argument("--stage", default = "")
    ap.add_argument("--stage-out", default = "")
    ap.add_argument("--role", default = "target")
    ap.add_argument("--precision", default = "")
    ap.add_argument("--lora-ids", default = "")
    args = ap.parse_args()
    if args.child:
        return child_main(args)

    out = Path(args.out).resolve()
    args.out = str(out)
    args.checkout = str(Path(args.checkout).resolve())
    work = Path(args.work or (Path(os.environ.get("AMD_CI_WORK", out.parent)) / "lora_probe")).resolve()
    args.work = str(work)
    work.mkdir(parents = True, exist_ok = True)
    # Downloads under the job's wiped temp root, never the runner's home.
    os.environ.setdefault("HF_HOME", str(work / "hf"))
    os.environ.setdefault("UNSLOTH_DISABLE_AUTO_UPDATES", "1")
    # One budget for every state: the first state's start is stamped in a file the next one reads.
    stamp = work / "budget_start"
    if not stamp.is_file():
        stamp.write_text(str(time.time()), encoding = "utf-8")
    deadline = float(stamp.read_text(encoding = "utf-8")) + args.budget_min * 60

    obs: dict = {"suite": args.suite, "state": args.state, "checkout": args.checkout, "steps": args.steps,
                 "resolution": args.resolution, "lr": args.lr, "rank": args.rank, "families": {}}
    stage_dir = out.parent / f"stages_{args.state}"
    stage_dir.mkdir(parents = True, exist_ok = True)
    try:
        info = fetch_dataset(work / "data", args.images)
        obs["dataset"] = info
        args.data_dir = str(work / "data")
    except Exception as exc:  # noqa: BLE001
        obs["dataset_error"] = f"{type(exc).__name__}: {exc}"
        args.data_dir = str(work / "data")
    _write(out, obs)

    host = run_child(args, "host", stage_dir / "host.json", 900, [])
    obs["host"] = host
    _write(out, obs)
    if args.vae_check:
        for tag, value in (("exp1", "1"), ("exp_unset", None)):
            env = dict(os.environ)
            env.pop("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", None)
            if value is not None:
                env["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = value
            env["AMD_CI_KEEP_AOTRITON_ENV"] = "1"
            obs.setdefault("sdpa_matrix", {})[tag] = run_child(args, "sdpa", stage_dir / f"sdpa_{tag}.json", 600, [],
                                                               env = env)
        _write(out, obs)

    roles = ["target"] + (["control"] if SUITES[args.suite]["control"] and args.state == args.control_state else [])
    offered = host.get("train_precision_modes") or []
    for role in roles:
        fam = (host.get("families") or {}).get(role) or {}
        entry: dict = {"accepted": fam.get("accepted"), "refusal": fam.get("refusal"),
                       "resolved_family": fam.get("resolved_family"), "preflight": fam.get("preflight"),
                       "cells": {}, "skipped": []}
        obs["families"][role] = entry
        if not fam.get("accepted"):
            _write(out, obs)
            continue
        ids = []
        for p in [x for x in args.precisions.split(",") if x]:
            if p.split("-")[0] not in offered:
                entry["skipped"].append({"precision": p, "why": "not offered by train_precision_modes() here"})
                continue
            left = deadline - time.time()
            if left < 10 * 60:
                entry["skipped"].append({"precision": p, "why": f"probe budget exhausted ({left / 60:.1f} min left)"})
                continue
            r = run_child(args, "train", stage_dir / f"train_{role}_{p}.json", int(min(left - 300, 50 * 60)),
                          ["--role", role, "--precision", p])
            entry["cells"][p] = r
            if r.get("catalog_id"):
                ids.append(r["catalog_id"])
            _write(out, obs)
        if ids:
            left = deadline - time.time()
            if left < 4 * 60:
                entry["render"] = {"skipped": f"probe budget exhausted ({left / 60:.1f} min left)"}
            else:
                entry["render"] = run_child(args, "render", stage_dir / f"render_{role}.json",
                                            int(min(left - 60, 40 * 60)),
                                            ["--role", role, "--lora-ids", ",".join(ids)])
        _write(out, obs)
    obs["finished"] = True
    _write(out, obs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
