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
    import torch

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
    args = ap.parse_args()
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
    from core.inference.diffusion import DiffusionBackend

    backend = DiffusionBackend()
    import numpy as np

    scorer = None
    try:
        import lpips

        scorer = lpips.LPIPS(net = "alex", verbose = False).eval()
    except Exception as exc:  # noqa: BLE001
        obs["lpips_error"] = str(exc)[:200]

    img_dir = args.out.parent / f"images_{args.state}"
    img_dir.mkdir(parents = True, exist_ok = True)
    refs: dict = {}
    arms: dict = {}
    obs["arms"] = arms
    # "none": PowerShell drops an empty-string argument, so an empty list needs a word.
    schemes = [s for s in args.schemes.split(",") if s and s != "none"]
    # The base only has to show the refusal; its bf16 render would be the head's bf16 render again.
    for scheme in schemes if args.state == "base" and not args.reference_at_base else ["off"] + schemes:
        rec: dict = {}
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
                speed_mode = "off",
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
            if scheme == "off":
                refs[i] = image
            elif i in refs:
                a = np.asarray(refs[i], dtype = np.float32) / 255.0
                b = np.asarray(image, dtype = np.float32) / 255.0
                item["psnr"] = round(
                    float(10 * np.log10(1.0 / max(float(((a - b) ** 2).mean()), 1e-12))), 3
                )
                if scorer is not None:
                    ta = torch.from_numpy(a).permute(2, 0, 1)[None] * 2 - 1
                    tb = torch.from_numpy(b).permute(2, 0, 1)[None] * 2 - 1
                    with torch.no_grad():
                        item["lpips"] = round(float(scorer(ta, tb).item()), 4)
            rec["images"].append(item)
            write(args.out, obs)
        try:
            backend.unload()
        except Exception as exc:  # noqa: BLE001
            rec["unload_error"] = err(exc)
        gc.collect()
        torch.cuda.empty_cache()
        write(args.out, obs)
    obs["done"] = True
    write(args.out, obs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
