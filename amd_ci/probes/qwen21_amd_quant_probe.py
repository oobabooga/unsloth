#!/usr/bin/env python3
"""Probe: which FP8 / INT8 paths run Qwen-Image-2.1 correctly on this AMD GPU, and at what cost?

Observes only; criteria/qwen21_amd_quant.py judges. Every arm renders the SAME prompts and seeds
as the bf16 arm, so each image is scored against its bf16 twin (LPIPS, PSNR) and every arm records
its load / quantise time, per-step time, weight footprint and peak memory, or the verbatim error.

Arms (transformer unless named; the text encoder stays bf16 unless the arm is about it):
  bf16            the reference
  studio_int8     Studio's own int8 W8A8 torchao path, the ROCm gate bypassed
  studio_fp8      Studio's own fp8 path (fp8 x fp8 _scaled_mm), the ROCm gate bypassed
  fp8_layerwise   fp8 weight STORAGE, bf16 compute (diffusers layerwise casting), no fp8 GEMM
  int8_weight     torchao int8 weight-only, bf16 activations
  fp8_weight      torchao fp8 weight-only, bf16 activations
  te_fp8          bf16 transformer, text encoder through Studio's fp8 layerwise cast
  gguf_q4km       the public Q4_K_M GGUF through diffusers (today's GPU GGUF route)

Only the head state renders; the base state records Studio's selector answers, which is cheap and
is what a user on this box meets today. Public, ungated weights only: no credential of any kind is
read or needed, and HF_TOKEN is dropped from the environment before the hub is touched.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

BASE_REPO = "Qwen/Qwen-Image-2.1"
GGUF_REPO = "unsloth/Qwen-Image-2.1-GGUF"
GGUF_FILE = "qwen-image-2.1-Q4_K_M.gguf"
PROMPTS = (
    "A photorealistic portrait of an elderly fisherman with a weathered face, golden hour light, 85mm lens",
    'A cozy bakery storefront with a hand-painted wooden sign that reads "UNSLOTH BAKERY", morning light',
)
ARMS = (
    "bf16",
    "studio_int8",
    "fp8_layerwise",
    "studio_fp8",
    "int8_weight",
    "fp8_weight",
    "te_fp8",
    "gguf_q4km",
)


def err(exc: BaseException) -> dict:
    return {
        "type": type(exc).__name__,
        "message": str(exc)[:2000],
        "tail": traceback.format_exc().strip().splitlines()[-6:],
    }


def attempt(fn) -> dict:
    try:
        return {"ok": True, "value": fn()}
    except BaseException as exc:  # noqa: BLE001 - recording, not handling
        return {"ok": False, "error": err(exc)}


def write(out: Path, obs: dict) -> None:
    out.parent.mkdir(parents = True, exist_ok = True)
    out.write_text(json.dumps(obs, indent = 2, default = str), encoding = "utf-8")


def studio_modules(checkout: str):
    backend = Path(checkout) / "studio" / "backend"
    if str(backend) not in sys.path:
        sys.path.insert(0, str(backend))
    from core.inference import diffusion_transformer_quant as tq

    return tq


def selector_answers(checkout: str) -> dict:
    """What Studio answers on this box today, per scheme."""
    import torch

    tq = studio_modules(checkout)
    target = SimpleNamespace(device = "cuda", dtype = torch.bfloat16)
    rec: dict = {
        "dense_transformer_supported": attempt(lambda: tq.dense_transformer_supported(target)),
        "unsupported_reason": attempt(lambda: tq.dense_transformer_unsupported_reason(target)),
    }
    for mode in ("auto", "int8", "fp8"):
        rec[f"select[{mode}]"] = attempt(
            lambda mode = mode: tq.select_transformer_quant_scheme(
                target, mode, family = "qwen-image-2.1"
            )
        )
    try:
        from core.inference import diffusion_precision as dp

        for mode in ("fp8", "int8"):
            rec[f"te_supported[{mode}]"] = attempt(
                lambda mode = mode: dp.te_quant_supported(target, mode)
            )
    except Exception as exc:  # noqa: BLE001
        rec["te_supported"] = {"ok": False, "error": err(exc)}
    return rec


def device_identity() -> dict:
    import torch

    props = torch.cuda.get_device_properties(0)
    return {
        "torch": torch.__version__,
        "hip": getattr(torch.version, "hip", None),
        "cuda": torch.version.cuda,
        "name": torch.cuda.get_device_name(0),
        "arch": getattr(props, "gcnArchName", None),
        "capability": list(torch.cuda.get_device_capability(0)),
        "total_gib": round(props.total_memory / 2**30, 2),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
    }


def package_versions() -> dict:
    import importlib.metadata as md

    out = {}
    for name in ("torch", "diffusers", "transformers", "torchao", "accelerate", "gguf", "lpips"):
        try:
            out[name] = md.version(name)
        except Exception:  # noqa: BLE001
            out[name] = None
    return out


class Scorer:
    """LPIPS (alex) when lpips imports, PSNR always."""

    def __init__(self):
        self.lpips = None
        self.lpips_error = None
        try:
            import lpips

            self.lpips = lpips.LPIPS(net = "alex", verbose = False).eval()
        except Exception as exc:  # noqa: BLE001
            self.lpips_error = f"{type(exc).__name__}: {exc}"[:300]

    def score(self, ref, img) -> dict:
        import numpy as np
        import torch

        a = np.asarray(ref.convert("RGB"), dtype = np.float32) / 255.0
        b = np.asarray(img.convert("RGB"), dtype = np.float32) / 255.0
        mse = float(((a - b) ** 2).mean())
        out = {"psnr": round(10 * np.log10(1.0 / max(mse, 1e-12)), 3)}
        if self.lpips is not None:
            ta = torch.from_numpy(a).permute(2, 0, 1)[None] * 2 - 1
            tb = torch.from_numpy(b).permute(2, 0, 1)[None] * 2 - 1
            with torch.no_grad():
                out["lpips"] = round(float(self.lpips(ta, tb).item()), 4)
        return out


def weight_bytes(module) -> int:
    """Bytes the module's parameters and buffers hold, counting tensor subclasses' inner storage."""
    import torch

    seen = set()
    total = 0

    def _add(t):
        nonlocal total
        inner = getattr(t, "__tensor_flatten__", None)
        if callable(inner) and type(t) not in (torch.Tensor, torch.nn.Parameter):
            try:
                names, _ = t.__tensor_flatten__()
                for n in names:
                    _add(getattr(t, n))
                return
            except Exception:  # noqa: BLE001
                pass
        try:
            key = t.untyped_storage().data_ptr()
            if key in seen:
                return
            seen.add(key)
            total += t.untyped_storage().nbytes()
        except Exception:  # noqa: BLE001
            total += t.numel() * t.element_size()

    for p in module.parameters():
        _add(p.data if isinstance(p, torch.nn.Parameter) else p)
    for b in module.buffers():
        _add(b)
    return total


def linear_filter(min_features: int = 1024):
    import torch

    def fn(module, fqn: str = "") -> bool:
        return (
            isinstance(module, torch.nn.Linear)
            and module.in_features >= min_features
            and module.out_features >= min_features
            and module.in_features % 16 == 0
            and module.out_features % 16 == 0
        )

    return fn


def count_converted(module) -> dict:
    import torch

    dense = converted = 0
    for m in module.modules():
        if isinstance(m, torch.nn.Linear):
            w = m.weight
            data = getattr(w, "data", w)
            if type(data) is torch.Tensor and data.dtype in (torch.bfloat16, torch.float16, torch.float32):
                dense += 1
            else:
                converted += 1
    return {"dense_linear": dense, "converted_linear": converted}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--models", default = os.environ.get("Q21_MODELS", ""))
    ap.add_argument("--arms", default = ",".join(ARMS))
    ap.add_argument("--size", type = int, default = 1024)
    ap.add_argument("--steps", type = int, default = 20)
    ap.add_argument("--cfg", type = float, default = 4.0)
    ap.add_argument("--prompts", type = int, default = len(PROMPTS))
    ap.add_argument("--budget-min", type = float, default = 150.0)
    ap.add_argument("--render-states", default = "head")
    args = ap.parse_args()

    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        os.environ.pop(key, None)
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    # The runners reach the hub through a caching mirror whose large-file reads time out at the
    # default 10 s; read before huggingface_hub is first imported, which the selector below does.
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")

    t_start = time.time()
    obs: dict = {"state": args.state, "args": vars(args) | {"out": str(args.out)}}
    write(args.out, obs)

    imported = attempt(lambda: __import__("torch").__version__)
    obs["torch_import"] = imported
    if not imported["ok"]:
        write(args.out, obs)
        return 0
    import torch

    obs["cuda_available"] = torch.cuda.is_available()
    if obs["cuda_available"]:
        obs["device"] = attempt(device_identity)
    obs["packages"] = package_versions()
    obs["selector"] = attempt(lambda: selector_answers(args.checkout))
    write(args.out, obs)

    if args.state not in args.render_states.split(",") or not obs["cuda_available"]:
        obs["rendered"] = False
        write(args.out, obs)
        return 0

    models = Path(args.models or (Path(args.out).parent / "models"))
    models.mkdir(parents = True, exist_ok = True)
    img_dir = Path(args.out).parent / f"images_{args.state}"
    img_dir.mkdir(parents = True, exist_ok = True)

    from huggingface_hub import hf_hub_download, snapshot_download

    obs["hf_endpoint"] = os.environ.get("HF_ENDPOINT")
    t0 = time.time()
    tries: list = []
    base_dl: dict = {"ok": False}
    for i in range(6):
        base_dl = attempt(
            lambda: snapshot_download(
                BASE_REPO,
                local_dir = str(models / "qwen_image_21"),
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
        )
        tries.append(base_dl.get("ok") or (base_dl.get("error") or {}).get("message", "")[:200])
        if base_dl["ok"]:
            break
        time.sleep(20)
    obs["download_base"] = {**base_dl, "seconds": round(time.time() - t0, 1), "tries": tries}
    write(args.out, obs)
    if not base_dl["ok"]:
        return 0
    base = base_dl["value"]

    import diffusers
    from diffusers import QwenImage21Pipeline, QwenImage21Transformer2DModel

    scorer = Scorer()
    obs["lpips_error"] = scorer.lpips_error
    target = SimpleNamespace(device = "cuda", dtype = torch.bfloat16)
    prompts = PROMPTS[: max(1, args.prompts)]
    refs: dict = {}
    arms: dict = {}
    obs["arms"] = arms

    t0 = time.time()
    loaded = attempt(
        lambda: QwenImage21Pipeline.from_pretrained(base, torch_dtype = torch.bfloat16)
    )
    obs["pipeline_load"] = {
        "ok": loaded["ok"],
        "seconds": round(time.time() - t0, 1),
        **({} if loaded["ok"] else {"error": loaded["error"]}),
    }
    write(args.out, obs)
    if not loaded["ok"]:
        return 0
    pipe = loaded["value"]
    pipe.to("cuda")
    bf16_transformer_state = None  # reload from disk per arm; keeps every arm starting from the same bytes

    def fresh_transformer():
        nonlocal bf16_transformer_state
        old = getattr(pipe, "transformer", None)
        pipe.transformer = None
        del old
        gc.collect()
        torch.cuda.empty_cache()
        t = QwenImage21Transformer2DModel.from_pretrained(
            base, subfolder = "transformer", torch_dtype = torch.bfloat16
        )
        return t.to("cuda")

    def fresh_text_encoder():
        from transformers import AutoModel

        cls = type(pipe.text_encoder)
        old = pipe.text_encoder
        pipe.text_encoder = None
        del old
        gc.collect()
        torch.cuda.empty_cache()
        te = cls.from_pretrained(base, subfolder = "text_encoder", torch_dtype = torch.bfloat16)
        return te.to("cuda")

    def render(arm: str) -> dict:
        rec: dict = {"images": []}
        for i, prompt in enumerate(prompts):
            steps_t: list = []

            def _cb(_pipe, step, _t, kwargs):
                torch.cuda.synchronize()
                steps_t.append(time.perf_counter())
                return kwargs

            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            t1 = time.perf_counter()
            image = pipe(
                prompt = prompt,
                negative_prompt = " ",
                width = args.size,
                height = args.size,
                num_inference_steps = args.steps,
                true_cfg_scale = args.cfg,
                generator = torch.Generator("cuda").manual_seed(42 + i),
                callback_on_step_end = _cb,
            ).images[0]
            torch.cuda.synchronize()
            total = time.perf_counter() - t1
            gaps = [b - a for a, b in zip(steps_t, steps_t[1:])]
            gaps.sort()
            path = img_dir / f"{arm}_p{i}.png"
            image.save(path)
            import numpy as np

            item = {
                "prompt_index": i,
                "seconds": round(total, 2),
                "median_step_s": round(gaps[len(gaps) // 2], 3) if gaps else None,
                "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                "luma": round(float(np.asarray(image.convert("L"), dtype = np.float32).mean()), 2),
                "path": path.name,
            }
            if arm == "bf16":
                refs[i] = image
            elif i in refs:
                item.update(scorer.score(refs[i], image))
            rec["images"].append(item)
        return rec

    def run_arm(arm: str) -> dict:
        rec: dict = {}
        t0 = time.time()
        if arm == "te_fp8":
            if not isinstance(pipe.transformer, QwenImage21Transformer2DModel) or (
                count_converted(pipe.transformer)["converted_linear"]
                or getattr(pipe.transformer, "_unsloth_gguf", False)
            ):
                pipe.transformer = fresh_transformer()
            from core.inference import diffusion_precision as dp

            dp._cast_fp8(pipe.text_encoder, target)
            rec["te_weight_gib"] = round(weight_bytes(pipe.text_encoder) / 2**30, 2)
        elif arm == "gguf_q4km":
            from diffusers import GGUFQuantizationConfig

            path = None
            for _ in range(4):
                try:
                    path = hf_hub_download(
                        GGUF_REPO, GGUF_FILE, local_dir = str(models / "gguf"), token = False
                    )
                    break
                except Exception:  # noqa: BLE001 - the mirror times out; resumed below
                    time.sleep(20)
            if path is None:
                path = hf_hub_download(
                    GGUF_REPO, GGUF_FILE, local_dir = str(models / "gguf"), token = False
                )
            from core.inference import diffusion as studio
            import logging

            log = logging.getLogger("probe")
            studio._register_unregistered_single_file_classes(log)
            studio._install_gguf_prefix_strip(QwenImage21Transformer2DModel, log)
            studio._install_gguf_dim_restore(log)
            old = pipe.transformer
            pipe.transformer = None
            del old
            gc.collect()
            torch.cuda.empty_cache()
            t = QwenImage21Transformer2DModel.from_single_file(
                path,
                quantization_config = GGUFQuantizationConfig(compute_dtype = torch.bfloat16),
                torch_dtype = torch.bfloat16,
                config = base,
                subfolder = "transformer",
                local_files_only = True,
            )
            t._unsloth_gguf = True
            pipe.transformer = t.to("cuda")
        else:
            pipe.transformer = fresh_transformer()
            tr = pipe.transformer
            if arm in ("studio_int8", "studio_fp8"):
                tq = studio_modules(args.checkout)
                # The gate under test: bypass it so the kernels answer, not the policy.
                tq.dense_transformer_supported = lambda _t: True
                if hasattr(tq, "_scheme_supported"):
                    rec["scheme_supported"] = attempt(
                        lambda: tq._scheme_supported(arm.split("_")[1], "cuda")
                    )
                engaged = tq.quantize_transformer(
                    pipe, target, mode = arm.split("_")[1], family = "qwen-image-2.1"
                )
                rec["engaged"] = engaged
                if engaged is None:
                    raise RuntimeError(f"quantize_transformer returned None for {arm}")
            elif arm == "fp8_layerwise":
                tr.enable_layerwise_casting(
                    storage_dtype = torch.float8_e4m3fn, compute_dtype = torch.bfloat16
                )
            elif arm in ("int8_weight", "fp8_weight"):
                from torchao.quantization import quantize_

                if arm == "int8_weight":
                    from torchao.quantization import Int8WeightOnlyConfig as Cfg
                else:
                    from torchao.quantization import Float8WeightOnlyConfig as Cfg
                quantize_(tr, Cfg(), filter_fn = linear_filter())
        torch.cuda.synchronize()
        rec["prepare_seconds"] = round(time.time() - t0, 1)
        rec["transformer_weight_gib"] = round(weight_bytes(pipe.transformer) / 2**30, 2)
        rec["linears"] = count_converted(pipe.transformer)
        rec["allocated_gib"] = round(torch.cuda.memory_allocated() / 2**30, 2)
        rec.update(render(arm))
        return rec

    wanted = [a for a in args.arms.split(",") if a]
    for arm in wanted:
        elapsed = (time.time() - t_start) / 60
        if elapsed > args.budget_min:
            arms[arm] = {"ok": False, "skipped": f"budget: {elapsed:.0f} min elapsed"}
            continue
        print(f"== arm {arm} ({elapsed:.1f} min in)", flush = True)
        res = attempt(lambda arm = arm: run_arm(arm))
        arms[arm] = res
        if arm == "te_fp8":
            # Put the encoder back so a later arm is not quietly scored with an fp8 encoder.
            try:
                pipe.text_encoder = fresh_text_encoder()
            except Exception as exc:  # noqa: BLE001
                arms[arm]["te_restore_error"] = err(exc)
        write(args.out, obs)
        print(json.dumps({arm: res}, default = str)[:1500], flush = True)

    obs["rendered"] = True
    obs["total_minutes"] = round((time.time() - t_start) / 60, 1)
    write(args.out, obs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
