#!/usr/bin/env python3
"""Probe: does Studio's own DiffusionBackend load and render Qwen-Image-2.1 with an explicit INT8 / FP8
transformer on this AMD GPU, and how close is the render to the same load at bf16?

Observes only; criteria/studio_weight_only.py judges. Every state loads bf16 first (the reference),
then each explicit scheme, through ``DiffusionBackend.load_pipeline`` exactly as the Images page does,
and records the status Studio reports, the transformer's weight bytes and the render, or the verbatim
refusal. Public, ungated weights only; HF_TOKEN is dropped before the hub is touched.

``PROBE_FAKE_ROCM=1`` pretends the host is ROCm, for a dry run of this harness on an NVIDIA box.
"""

from __future__ import annotations

import argparse
import faulthandler
import gc
import json
import os
import sys
import time
import traceback
from pathlib import Path

BASE_REPO = "Qwen/Qwen-Image-2.1"
PROMPTS = (
    "A photorealistic portrait of an elderly fisherman with a weathered face, golden hour light, 85mm lens",
    'A cozy bakery storefront with a hand-painted wooden sign that reads "UNSLOTH BAKERY", morning light',
)


def err(exc: BaseException) -> dict:
    return {
        "type": type(exc).__name__,
        "message": str(exc)[:1500],
        "tail": traceback.format_exc().strip().splitlines()[-6:],
    }


def write(out: Path, obs: dict) -> None:
    out.parent.mkdir(parents = True, exist_ok = True)
    out.write_text(json.dumps(obs, indent = 2, default = str), encoding = "utf-8")


def weight_gib(module) -> float:
    seen, total = set(), 0
    for t in list(module.parameters()) + list(module.buffers()):
        try:
            key = t.untyped_storage().data_ptr()
        except Exception:  # noqa: BLE001
            continue
        if key in seen or t.device.type != "cuda":
            continue
        seen.add(key)
        total += t.untyped_storage().nbytes()
    return round(total / 2**30, 2)


def memory() -> dict:
    import torch

    out: dict = {}
    try:
        free, total = torch.cuda.mem_get_info()
        out["device_free_gib"], out["device_total_gib"] = round(free / 2**30, 2), round(total / 2**30, 2)
        out["allocated_gib"] = round(torch.cuda.memory_allocated() / 2**30, 2)
    except Exception as exc:  # noqa: BLE001
        out["error"] = str(exc)[:200]
    try:
        import psutil

        out["process_rss_gib"] = round(psutil.Process().memory_info().rss / 2**30, 2)
        vm = psutil.virtual_memory()
        out["system_available_gib"] = round(vm.available / 2**30, 2)
    except Exception:  # noqa: BLE001
        pass
    return out


def orchestrate(args) -> int:
    """One process per scheme, then score every quantised render against the bf16 render."""
    import subprocess

    # "none": PowerShell drops an empty-string argument, so an empty list needs a word.
    schemes = [s for s in args.schemes.split(",") if s and s != "none"]
    # The base only has to show the refusal; its bf16 render would be the head's bf16 render again.
    order = schemes if args.state == "base" and not args.reference_at_base else ["off"] + schemes
    obs: dict = {"state": args.state, "args": vars(args) | {"out": str(args.out)}, "arms": {}}
    write(args.out, obs)
    for scheme in order:
        part = args.out.with_name(f"{args.out.stem}.{scheme}.json")
        cmd = [sys.executable, "-u", __file__, "--state", args.state, "--checkout", args.checkout,
               "--out", str(part), "--models", args.models, "--schemes", args.schemes,
               "--size", str(args.size), "--steps", str(args.steps), "--prompts", str(args.prompts),
               "--single", scheme]
        if args.reference_at_base:
            cmd.append("--reference-at-base")
        rc = subprocess.call(cmd)
        sub = json.loads(part.read_text(encoding = "utf-8")) if part.is_file() else {}
        for key in ("torch", "hip", "device", "arch", "download_seconds", "download_retries", "stub_error"):
            if key in sub and key not in obs:
                obs[key] = sub[key]
        arm = (sub.get("arms") or {}).get(scheme) or {"loaded": False, "error": {"message": f"probe exited {rc}"}}
        arm["exit_code"] = rc
        obs["arms"][scheme] = arm
        write(args.out, obs)
    score(args, obs)
    obs["done"] = True
    write(args.out, obs)
    return 0


def score(args, obs: dict) -> None:
    import numpy as np
    from PIL import Image

    img_dir = args.out.parent / f"images_{args.state}"
    scorer = None
    try:
        import lpips
        import torch

        scorer = lpips.LPIPS(net = "alex", verbose = False).eval()
    except Exception as exc:  # noqa: BLE001
        obs["lpips_error"] = str(exc)[:200]
    for scheme, rec in obs["arms"].items():
        if scheme == "off":
            continue
        for item in rec.get("images") or []:
            ref_path = img_dir / f"off_p{item.get('prompt_index')}.png"
            if "error" in item or not ref_path.is_file():
                continue
            a = np.asarray(Image.open(ref_path).convert("RGB"), dtype = np.float32) / 255.0
            b = np.asarray(Image.open(img_dir / item["path"]).convert("RGB"), dtype = np.float32) / 255.0
            item["psnr"] = round(float(10 * np.log10(1.0 / max(float(((a - b) ** 2).mean()), 1e-12))), 3)
            if scorer is not None:
                ta = torch.from_numpy(a).permute(2, 0, 1)[None] * 2 - 1
                tb = torch.from_numpy(b).permute(2, 0, 1)[None] * 2 - 1
                with torch.no_grad():
                    item["lpips"] = round(float(scorer(ta, tb).item()), 4)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--models", default = "")
    ap.add_argument("--schemes", default = "int8,fp8")
    ap.add_argument("--size", type = int, default = 1024)
    ap.add_argument("--steps", type = int, default = 20)
    ap.add_argument("--prompts", type = int, default = len(PROMPTS))
    ap.add_argument("--reference-at-base", action = "store_true")
    ap.add_argument("--speed", default = "off", help = "Studio speed_mode for every load")
    ap.add_argument("--cache", default = "off", help = "Studio transformer_cache for every load (off / fbcache)")
    # Internal: run one scheme in this process and write only its arm. The parent runs every scheme
    # in a fresh process, as a user starting Studio would: on a unified-memory APU the pages an
    # earlier arm read stay charged to the system, so a second load in the same process is judged
    # against memory the first one left behind, not against what the scheme needs.
    ap.add_argument("--single", default = "")
    args = ap.parse_args()
    if not args.single:
        return orchestrate(args)
    # A stuck render still leaves evidence: stacks every 10 minutes into the probe log.
    faulthandler.dump_traceback_later(int(os.environ.get("PROBE_STACK_EVERY", "600")), repeat = True)

    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        os.environ.pop(key, None)
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")

    obs: dict = {"state": args.state, "args": vars(args) | {"out": str(args.out)}}
    write(args.out, obs)
    import torch

    obs["torch"] = torch.__version__
    obs["hip"] = getattr(torch.version, "hip", None)
    obs["device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    obs["arch"] = (
        getattr(torch.cuda.get_device_properties(0), "gcnArchName", None)
        if torch.cuda.is_available()
        else None
    )

    models = Path(args.models or (args.out.parent / "models"))
    base_dir = models / "qwen_image_21"
    from huggingface_hub import snapshot_download

    t0 = time.time()
    for attempt in range(6):
        try:
            snapshot_download(
                BASE_REPO,
                local_dir = str(base_dir),
                allow_patterns = [
                    "model_index.json",
                    "processor/*",
                    "scheduler/*",
                    "text_encoder/*",
                    "transformer/*",
                    "vae/*",
                ],
                token = False,
                max_workers = 4,
            )
            break
        except Exception as exc:  # noqa: BLE001 - the runner mirror times out; resume
            obs.setdefault("download_retries", []).append(str(exc)[:200])
            time.sleep(20)
    obs["download_seconds"] = round(time.time() - t0, 1)
    write(args.out, obs)

    sys.path.insert(0, str(Path(args.checkout) / "studio" / "backend"))
    # What run.py does first, so Windows ROCm sees the same stubs a real Studio process does.
    try:
        from core._torchao_stub import (
            install_torchao_windows_rocm_stub,
            install_xformers_windows_rocm_stub,
        )

        install_xformers_windows_rocm_stub()
        install_torchao_windows_rocm_stub()
    except Exception as exc:  # noqa: BLE001 - an older checkout without the stubs
        obs["stub_error"] = str(exc)[:200]
    if os.environ.get("PROBE_FAKE_ROCM") == "1":
        import core._torchao_stub as stub

        stub.torch_is_rocm = lambda: True
        from core.inference import diffusion_transformer_quant as tq

        tq.torch_is_rocm = lambda: True
    if os.environ.get("PROBE_FORCE_COMPILE") == "1":
        # Studio does not compile on ROCm by default (supports_default_torch_compile = not is_rocm). Measuring what
        # compile would buy means lifting only that gate: the runtime and bf16 checks still apply.
        import core.inference.diffusion_speed as ds

        ds.compile_eligible = lambda target, *, is_gguf, family: (
            ds.torch_compile_runtime_available()
            and bool(getattr(family, "supports_torch_compile", True))
            and ds._is_bfloat16(getattr(target, "dtype", None))
        )
        obs["forced_compile"] = True
    from core.inference.diffusion import DiffusionBackend

    backend = DiffusionBackend()
    import numpy as np

    img_dir = args.out.parent / f"images_{args.state}"
    img_dir.mkdir(parents = True, exist_ok = True)
    arms: dict = {}
    obs["arms"] = arms
    for scheme in [args.single]:
        rec: dict = {"memory_before_load": memory()}
        arms[scheme] = rec
        t1 = time.time()
        try:
            status = backend.load_pipeline(
                str(base_dir),
                model_kind = "pipeline",
                family_override = "qwen-image-2.1",
                local_files_only = True,
                transformer_quant = scheme,
                text_encoder_quant = "none",
                speed_mode = args.speed,
                transformer_cache = args.cache,
            )
        except Exception as exc:  # noqa: BLE001 - a refusal is an observation
            rec["loaded"] = False
            rec["error"] = err(exc)
            write(args.out, obs)
            continue
        rec["loaded"] = True
        rec["load_seconds"] = round(time.time() - t1, 1)
        rec["transformer_quant"] = status.get("transformer_quant")
        rec["resolved"] = (status.get("resolved") or {}).get("transformer_quant")
        rec["speed"] = {k: status.get(k) for k in ("speed_mode", "speed_optims", "compiled") if k in status}
        rec["transformer_cache"] = status.get("transformer_cache")
        rec["cache_resolved"] = (status.get("resolved") or {}).get("transformer_cache")
        # No local reference to the pipeline: one would keep this arm's weights alive through the next load.
        rec["transformer_gib"] = weight_gib(backend._state.pipe.transformer)
        rec["images"] = []
        for i, prompt in enumerate(PROMPTS[: args.prompts]):
            torch.cuda.reset_peak_memory_stats()
            t2 = time.time()
            try:
                result = backend.generate(
                    prompt = prompt,
                    width = args.size,
                    height = args.size,
                    steps = args.steps,
                    guidance = 4.0,
                    seed = 42 + i,
                )
            except Exception as exc:  # noqa: BLE001
                rec["images"].append({"prompt_index": i, "error": err(exc)})
                write(args.out, obs)
                continue
            image = result["images"][0]
            if isinstance(image, str):
                import base64
                import io

                from PIL import Image

                image = Image.open(io.BytesIO(base64.b64decode(image.split(",", 1)[-1])))
            image = image.convert("RGB")
            path = img_dir / f"{scheme}_p{i}.png"
            image.save(path)
            item = {
                "prompt_index": i,
                "seconds": round(time.time() - t2, 1),
                "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                "path": path.name,
                "mean_luma": round(float(np.asarray(image.convert("L"), dtype = np.float32).mean()), 2),
            }
            rec["images"].append(item)
            write(args.out, obs)
        try:
            backend.unload()
        except Exception as exc:  # noqa: BLE001
            rec["unload_error"] = err(exc)
        gc.collect()
        torch.cuda.empty_cache()
        rec["memory_after_unload"] = memory()
        if scheme == "off":
            # Diagnostic only: the same bf16 load again in this process. Refused here means a second
            # load in one session is judged against what the first left charged, whatever the scheme.
            try:
                backend.load_pipeline(
                    str(base_dir),
                    model_kind = "pipeline",
                    family_override = "qwen-image-2.1",
                    local_files_only = True,
                    transformer_quant = "off",
                    text_encoder_quant = "none",
                    speed_mode = "off",
                )
                rec["same_process_reload"] = {"loaded": True}
                backend.unload()
            except Exception as exc:  # noqa: BLE001
                rec["same_process_reload"] = {"loaded": False, "error": err(exc)}
            gc.collect()
            torch.cuda.empty_cache()
        write(args.out, obs)
    obs["done"] = True
    write(args.out, obs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
