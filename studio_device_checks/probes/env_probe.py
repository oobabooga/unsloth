"""Machine and install facts, plus the torch / Triton sanity check (#19569): import torch and triton, then compile a
tiny function on the GPU. usage: env_probe.py --out FILE [--no-compile]"""

from __future__ import annotations

import importlib.metadata as md
import importlib.util
import os
import platform
import sys
import time

import _dc


def dist_version(name: str):
    try:
        return md.version(name)
    except Exception:  # noqa: BLE001
        return None


def main() -> None:
    out = _dc.out_path()
    rec = {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "prefix": sys.prefix,
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }
    spec = importlib.util.find_spec("studio")
    rec["installed_studio_pkg"] = list(spec.submodule_search_locations or []) if spec else None
    rec["dists"] = {
        name: dist_version(name)
        for name in (
            "unsloth",
            "unsloth_zoo",
            "torch",
            "torchvision",
            "triton",
            "triton-windows",
            "pytorch-triton-rocm",
            "pytorch-triton-xpu",
            "torchao",
            "diffusers",
            "transformers",
            "huggingface_hub",
            "safetensors",
            "numpy",
            "pillow",
            "scipy",
            "lpips",
            "pytest",
            "packaging",
        )
    }
    rec.update(_dc.cuda_info())
    try:
        import triton

        rec["triton_import"] = getattr(triton, "__version__", "?")
    except Exception as exc:  # noqa: BLE001
        rec["triton_import_error"] = _dc.err(exc)
    # The Hugging Face cache this Studio install downloads into (its own settings, read only).
    try:
        from utils.hf_cache_settings import get_hf_cache_paths

        paths = get_hf_cache_paths()
        rec["studio_hf_hub_cache"] = str(paths.hub_cache)
        rec["studio_hf_xet_cache"] = str(getattr(paths, "xet_cache", "") or "")
    except Exception as exc:  # noqa: BLE001
        rec["studio_hf_cache_error"] = _dc.err(exc, 300)
    if "--no-compile" not in sys.argv and rec.get("cuda_available") and not rec.get("torch_hip"):
        _dc.windows_compile_gate(rec)
        try:
            import torch

            def f(x, y):
                return torch.nn.functional.gelu(x * 2.0 + y).sum(dim = -1)

            x = torch.randn(64, 256, device = "cuda")
            y = torch.randn(64, 256, device = "cuda")
            t = time.perf_counter()
            cf = torch.compile(f)
            got = cf(x, y)
            torch.cuda.synchronize()
            rec["compile_s"] = round(time.perf_counter() - t, 2)
            ref = f(x, y)
            rec["compile_max_abs_err"] = float((got - ref).abs().max())
            rec["compile_ok"] = bool(torch.allclose(got, ref, atol = 1e-3, rtol = 1e-3))
        except Exception as exc:  # noqa: BLE001
            rec["compile_ok"] = False
            rec["compile_error"] = _dc.err(exc, 1500)
            rec["compile_tb"] = _dc.tb(2500)
    rec["env_flags"] = {k: v for k, v in os.environ.items() if k.startswith(("UNSLOTH_", "CUDA_VISIBLE"))}
    _dc.write(out, rec)


if __name__ == "__main__":
    main()
