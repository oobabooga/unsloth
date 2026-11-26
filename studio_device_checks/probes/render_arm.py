"""One render arm in a fresh process: load FLUX.2 klein 4B through Studio's DiffusionBackend, then render a fixed
list of (prompt, seed) pairs. Images go to --imgdir as <label>__<gen>__<i>.png, everything else to --out.
usage: render_arm.py --out FILE --label L --imgdir D --tq none|fp8|nvfp4 [--speed eager|default|off]
                     [--load-lora W] --gens name:w|-,... [--n 2] [--steps 4] [--size 1024] [--warmup 0]"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time

import _dc

PROMPTS = [
    "a red fox sitting in a snowy forest at dawn, soft golden light",
    "a lighthouse on a rocky coast under a stormy sky, waves crashing",
    "a small cafe on a rainy city street at night, neon signs reflecting on wet pavement",
    "a bowl of ramen with steam rising, close up, warm kitchen light",
]
SEEDS = (1234, 4321)
PAIRS = [(p, s) for s in SEEDS for p in PROMPTS]
STATUS_KEYS = (
    "family", "transformer_quant", "transformer_quant_backend", "transformer_quant_backend_reason",
    "transformer_quant_status", "text_encoder_quant", "lora_unavailable_reason", "baked_loras", "supports_lora",
    "speed_mode", "speed_optims", "offload_policy", "fp8_gemm",
)


def parse(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required = True)
    ap.add_argument("--label", required = True)
    ap.add_argument("--imgdir", required = True)
    ap.add_argument("--tq", required = True)
    ap.add_argument("--speed", default = None)
    ap.add_argument("--load-lora", type = float, default = None)
    ap.add_argument("--gens", default = "base:-")
    ap.add_argument("--repo", default = "black-forest-labs/FLUX.2-klein-4B")
    ap.add_argument("--lora", default = "adirik/pixel-art-lora-flux.2-klein-4B")
    ap.add_argument("--steps", type = int, default = 4)
    ap.add_argument("--size", type = int, default = 1024)
    ap.add_argument("--n", type = int, default = 2)
    ap.add_argument("--warmup", type = int, default = 0, help = "untimed renders before the timed ones")
    return ap.parse_args(argv)


def to_pil(img):
    import base64
    import io

    from PIL import Image

    if hasattr(img, "save"):
        return img.convert("RGB")
    return Image.open(io.BytesIO(base64.b64decode(str(img).split(",", 1)[-1]))).convert("RGB")


def status_fields(eng) -> dict:
    try:
        st = eng.status() or {}
    except Exception as exc:  # noqa: BLE001
        return {"status_error": _dc.err(exc, 300)}
    keep = {k: st.get(k) for k in STATUS_KEYS if k in st}
    for k, v in st.items():
        if k not in keep and any(w in k for w in ("fp8", "nvfp4", "compile")):
            keep[k] = v
    return keep


def main() -> None:
    a = parse(sys.argv[1:])
    rec = {"label": a.label, "tq": a.tq, "speed": a.speed, "load_lora": a.load_lora, "renders": {}, "timings": {},
           "tree": _dc.enter_tree(), **_dc.cuda_info(),
           "env": {k: v for k, v in os.environ.items() if k.startswith("UNSLOTH_")}}
    os.makedirs(a.imgdir, exist_ok = True)
    try:
        if not rec.get("cuda_available"):
            raise SystemExit("no CUDA device")
        import torch

        _dc.windows_compile_gate(rec)
        import core.inference.diffusion as D
        from core.inference.diffusion import DiffusionBackend

        timings = rec["timings"]

        def timed(name, fn):
            def wrapper(*args, **kwargs):
                torch.cuda.synchronize()
                t = time.perf_counter()
                try:
                    return fn(*args, **kwargs)
                finally:
                    torch.cuda.synchronize()
                    timings.setdefault(name, []).append(round(time.perf_counter() - t, 3))
            return wrapper

        if hasattr(D, "quantize_transformer"):
            D.quantize_transformer = timed("quantize_transformer", D.quantize_transformer)
        if hasattr(D.DiffusionBackend, "_pipeline_quant_from_cache"):
            D.DiffusionBackend._pipeline_quant_from_cache = staticmethod(
                timed("pipeline_quant_from_cache", D.DiffusionBackend._pipeline_quant_from_cache))
        eng = DiffusionBackend()
        kw = {"transformer_quant": a.tq, "hf_token": os.environ.get("HF_TOKEN") or None}
        if a.speed:
            kw["speed_mode"] = a.speed
        if a.load_lora is not None:
            kw["loras"] = [(a.lora, a.load_lora)]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        eng.begin_load(a.repo, **kw)
        last = None
        while True:
            p = eng.load_progress() or {}
            if p.get("phase") == "error" or p.get("error"):
                raise SystemExit(f"load error: {p.get('error')}")
            if p.get("phase") == "ready":
                break
            if p.get("phase") != last:
                last = p.get("phase")
                print("PHASE", a.label, last, round(time.perf_counter() - t0, 1), flush = True)
            time.sleep(0.1)
        rec["load_s"] = round(time.perf_counter() - t0, 2)
        rec["status"] = status_fields(eng)
        print("LOADED", a.label, rec["load_s"], "s", flush = True)
        for i in range(a.warmup):
            prompt, seed = PAIRS[i % len(PAIRS)]
            t = time.perf_counter()
            eng.generate(prompt = prompt, width = a.size, height = a.size, steps = a.steps, guidance = 1.0, seed = seed)
            rec.setdefault("warmup_s", []).append(round(time.perf_counter() - t, 2))
        for spec in a.gens.split(","):
            name, w = spec.split(":")
            gkw = {}
            if w != "-":
                gkw["loras"] = [(a.lora, float(w))]
            shas, times = [], []
            for i, (prompt, seed) in enumerate(PAIRS[: a.n]):
                torch.cuda.synchronize()
                t = time.perf_counter()
                try:
                    out = eng.generate(prompt = prompt, width = a.size, height = a.size, steps = a.steps,
                                       guidance = 1.0, seed = seed, **gkw)
                except Exception as exc:  # noqa: BLE001
                    rec["renders"][name] = {"error": _dc.err(exc, 1500), "tb": _dc.tb(2000)}
                    print("GEN_ERROR", a.label, name, _dc.err(exc, 600), flush = True)
                    break
                torch.cuda.synchronize()
                times.append(round(time.perf_counter() - t, 3))
                im = to_pil(out["images"][0])
                im.save(os.path.join(a.imgdir, f"{a.label}__{name}__{i}.png"))
                shas.append(hashlib.sha256(im.tobytes()).hexdigest()[:16])
            else:
                rec["renders"][name] = {"shas": shas, "s": times, "steps": a.steps, "status": status_fields(eng)}
                print("GEN", a.label, name, times, flush = True)
        try:
            from torch._dynamo.utils import counters

            rec["dynamo_unique_graphs"] = int(counters["stats"].get("unique_graphs", 0))
        except Exception:  # noqa: BLE001
            pass
        try:
            from core.inference import diffusion_prequant_cache as pc

            if hasattr(pc, "finish_background_writes"):
                pc.finish_background_writes(timeout = 900)
        except Exception as exc:  # noqa: BLE001
            rec["cache_flush_error"] = _dc.err(exc, 300)
        rec["peak_vram_gb"] = round(torch.cuda.max_memory_allocated(0) / 2**30, 2)
    except SystemExit as exc:
        rec["fatal"] = str(exc)[:2000]
    except Exception as exc:  # noqa: BLE001
        rec["fatal"] = _dc.err(exc, 1500)
        rec["tb"] = _dc.tb()
    _dc.write(a.out, rec)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
