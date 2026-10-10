"""Load a GGUF image model through Studio's diffusion backend and generate once, as the Images tab does.

Observes only: writes JSON to --out (never stdout). Usage:
  python gguf_gen_probe.py --backend-dir <tree>/studio/backend --out result.json [--steps 2] [--size 512]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import sys
import time
import traceback


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend-dir", required = True)
    ap.add_argument("--out", required = True)
    ap.add_argument("--repo", default = "unsloth/FLUX.2-klein-4B-GGUF")
    ap.add_argument("--gguf", default = "flux-2-klein-4b-Q2_K.gguf")
    ap.add_argument("--steps", type = int, default = 2)
    ap.add_argument("--size", type = int, default = 512)
    ap.add_argument("--image-out", default = None)
    # Raise where #9897's traceback did (inductor codegen asking Triton for its backend hash) while leaving every
    # directly launched Triton kernel working: isolates the compiled GGUF dequant on hosts whose torch uses Triton eagerly.
    ap.add_argument("--inject-inductor-driver-failure", action = "store_true")
    args = ap.parse_args()

    sys.path.insert(0, os.path.abspath(args.backend_dir))
    os.chdir(os.path.abspath(args.backend_dir))
    records: list = []

    class _Keep(logging.Handler):
        def emit(self, rec):
            msg = rec.getMessage()
            if any(k in msg for k in ("compile", "gguf", "eager", "Triton", "toolchain")):
                records.append(f"{rec.levelname} {rec.name}: {msg[:600]}")

    logging.getLogger().addHandler(_Keep())
    logging.getLogger().setLevel(logging.INFO)

    res: dict = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "env": {k: os.environ.get(k) for k in ("CC", "TRITON_CACHE_DIR", "TORCHDYNAMO_DISABLE")},
    }
    try:
        import torch

        res["torch"] = torch.__version__
        res["hip"] = getattr(torch.version, "hip", None)
        res["cuda"] = torch.version.cuda
        res["device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        try:
            import triton

            res["triton"] = triton.__version__
        except Exception as exc:  # noqa: BLE001
            res["triton"] = f"unimportable: {exc!r}"[:300]
        from core.inference import diffusion_speed

        res["torch_compile_runtime_available"] = diffusion_speed.torch_compile_runtime_available()
        if sys.platform == "win32":
            from core import _msvc_env

            res["crt_headers_reachable"] = _msvc_env.crt_headers_reachable()
            res["toolchain"] = _msvc_env._toolchain_summary()

        if args.inject_inductor_driver_failure:
            import subprocess
            import torch.utils._triton as tu

            def _fail(*a, **k):
                raise subprocess.CalledProcessError(1, ["clang-cl.exe", "hip_utils.c"])

            tu.triton_hash_with_backend = _fail
            res["injected"] = "torch.utils._triton.triton_hash_with_backend"
        from core.inference.diffusion import get_diffusion_backend
        from core.inference import diffusion_gguf_compile

        backend = get_diffusion_backend()
        t0 = time.time()
        backend.load_pipeline(args.repo, gguf_filename = args.gguf)
        res["load_s"] = round(time.time() - t0, 1)
        res["compiled_dequant_installed"] = diffusion_gguf_compile.is_compiled_dequant_installed()
        t0 = time.time()
        try:
            out = backend.generate(
                prompt = "a red apple on a wooden table",
                width = args.size,
                height = args.size,
                steps = args.steps,
                seed = 0,
            )
            res["generate_ok"] = True
            res["generate_s"] = round(time.time() - t0, 1)
            res["generate_keys"] = sorted(out.keys()) if isinstance(out, dict) else str(type(out))
            if args.image_out and isinstance(out, dict):
                import base64

                imgs = out.get("images") or out.get("image_b64") or []
                if isinstance(imgs, str):
                    imgs = [imgs]
                if imgs:
                    data = imgs[0]
                    data = data.get("b64", data) if isinstance(data, dict) else data
                    if isinstance(data, str):
                        data = data.split(",", 1)[-1]
                        with open(args.image_out, "wb") as fh:
                            fh.write(base64.b64decode(data))
                        res["image_out"] = args.image_out
        except Exception as exc:  # noqa: BLE001 - the observation
            res["generate_ok"] = False
            res["generate_error_type"] = type(exc).__name__
            res["generate_error"] = str(exc)[:1500]
            res["generate_tb_tail"] = traceback.format_exc()[-12000:]
    except Exception as exc:  # noqa: BLE001
        res["setup_error"] = f"{type(exc).__name__}: {exc}"[:1500]
        res["setup_tb_tail"] = traceback.format_exc()[-12000:]
    res["log_records"] = records[-40:]
    with open(args.out, "w", encoding = "utf-8") as fh:
        json.dump(res, fh, indent = 1, default = str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
