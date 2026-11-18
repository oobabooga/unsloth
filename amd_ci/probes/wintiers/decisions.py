#!/usr/bin/env python3
"""The RDNA tier decisions of one Studio tree on this GPU, in a fresh process (observes only).

Order matters: the tier reasons first (the head's Windows path runs the regional compile probe inside them; the base
never does), then the probe itself, then the reasons again (a probe result is cached per card).

  python decisions.py --root STUDIO_BACKEND --out decisions.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
import types
from pathlib import Path

FAMILIES = ("flux.3-klein", "qwen-image-2.1", "qwen-image", "z-image")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required = True)
    ap.add_argument("--out", required = True)
    a = ap.parse_args()
    sys.path.insert(0, str(Path(a.root).resolve()))
    res: dict = {"platform": sys.platform}

    def save():
        Path(a.out).write_text(json.dumps(res, indent = 1, default = str), encoding = "utf-8")

    try:
        import logging

        logging.basicConfig(level = logging.INFO, stream = sys.stdout, format = "LOG %(name)s %(message)s")
        import torch

        res["torch"] = torch.__version__
        res["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        from core.inference import diffusion_native_quant as nq
        from core.inference import diffusion_rocm_compile as rc
        from core.inference import diffusion_transformer_quant as tq
        from core.inference.diffusion_speed import torch_compile_runtime_available

        target = types.SimpleNamespace(device = "cuda", dtype = torch.bfloat16, ordinal = 0, backend = "rocm",
                                       supports_default_torch_compile = True)
        res["torch_compile_runtime_available"] = bool(torch_compile_runtime_available())
        res["rocm_rdna_w8a8_ready"] = bool(nq.rocm_rdna_w8a8_ready(0))
        res["native_quant_host"] = bool(tq.native_quant_host(target))
        res["probe_known_before"] = rc.rocm_block_compile_known(0)

        def reasons() -> dict:
            out = {}
            for fam in FAMILIES:
                out[fam] = {"auto_w8a8": tq.rdna_auto_w8a8_reason(target, fam),
                            "explicit_int8": tq.rdna_explicit_int8_reason(target, fam),
                            "explicit_w8a8": tq.rdna_explicit_w8a8_reason(target, fam)}
            return out

        t0 = time.time()
        res["reasons_first"] = reasons()
        res["reasons_first_s"] = round(time.time() - t0, 2)
        res["probe_known_after_reasons"] = rc.rocm_block_compile_known(0)
        save()
        t0 = time.time()
        ok, why = rc.rocm_block_compile_probe(0)
        res["compile_probe"] = {"ok": ok, "why": why, "seconds": round(time.time() - t0, 2)}
        res["reasons_after_probe"] = reasons()
        res["verdict"] = "ok"
    except Exception as exc:  # noqa: BLE001
        res["verdict"] = "error"
        res["error"] = repr(exc)[:1000]
        res["traceback"] = traceback.format_exc()[-4000:]
    save()
    return 0


if __name__ == "__main__":
    sys.exit(main())
