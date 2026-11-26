#!/usr/bin/env python3
"""R3b-c: MiniMax-H3's Qwen3-VL vision tower on gfx1151: Studio's float32 scope (``h3_vision_float32``, the shipped
default) vs the bf16 tower, one fresh process per arm.

  h3vis.py --backend B --dir MODEL --arm fp32|bf16 --out F --save FEATS.pt

The text encoder is built from its real config with ONE language layer (the tower is untouched) and the real
``model.visual.*`` weights (bf16, as Studio loads it). Input: one 832x480 reference image through the repo's image
processor. 2 warm calls, 5 timed; features saved for the fp32-vs-bf16 comparison."""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics as st
import sys
import time


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required = True)
    ap.add_argument("--dir", required = True)
    ap.add_argument("--arm", required = True)
    ap.add_argument("--out", required = True)
    ap.add_argument("--save")
    a = ap.parse_args()
    sys.path.insert(0, a.backend)
    import numpy as np
    import torch
    import transformers
    from safetensors import safe_open
    from transformers import AutoConfig, AutoImageProcessor, Qwen3VLForConditionalGeneration

    obs = {"arm": a.arm, "transformers": transformers.__version__}
    try:
        cfg = AutoConfig.from_pretrained(os.path.join(a.dir, "text_encoder"))
        cfg.text_config.num_hidden_layers = 1
        with torch.device("cuda"):
            te = Qwen3VLForConditionalGeneration._from_config(cfg, torch_dtype = torch.bfloat16)
        te.eval()
        sd = {}
        for f in glob.glob(os.path.join(a.dir, "text_encoder", "*.safetensors")):
            with safe_open(f, framework = "pt") as fh:
                for k in fh.keys():
                    if ".visual." in k or k.startswith("visual."):
                        sd[k] = fh.get_tensor(k)
        own = te.state_dict()
        loaded = 0
        for k, v in sd.items():
            for cand in (k, k.replace("model.visual.", "visual."), k.replace("visual.", "model.visual.", 1)):
                if cand in own:
                    own[cand].copy_(v.to(own[cand].dtype))
                    loaded += 1
                    break
        obs["visual_tensors"], obs["visual_loaded"] = len(sd), loaded
        proc = AutoImageProcessor.from_pretrained(os.path.join(a.dir, "processor"))
        yy, xx = np.mgrid[0:480, 0:832]
        rng = np.random.default_rng(3)
        img = np.stack([(xx * 255 // 831), (yy * 255 // 479), ((xx + yy) % 256)], -1).astype(np.int32)
        img = np.clip(img + rng.integers(-20, 21, img.shape), 0, 255).astype(np.uint8)
        enc = proc(images = [img], return_tensors = "pt")
        pv = enc["pixel_values"].to("cuda")
        grid = enc["image_grid_thw"].to("cuda")
        obs["pixel_values"], obs["grid"] = list(pv.shape), grid.tolist()
        model = te.model
        from core.inference import h3_vision_fp32 as V

        def call():
            if a.arm == "fp32":
                with V.float32_pixel_sources(), V.h3_vision_float32(te) as m:
                    return m.get_image_features(pv.float(), grid)
            return model.get_image_features(pv.to(torch.bfloat16), grid)
        with torch.no_grad():
            for _ in range(2):
                out = call()
            torch.cuda.synchronize()
            ts = []
            for _ in range(5):
                torch.cuda.synchronize()
                t = time.perf_counter()
                out = call()
                torch.cuda.synchronize()
                ts.append(time.perf_counter() - t)
        obs["s"], obs["med_s"] = ts, st.median(ts)
        obs["fp32_calls"] = V.STATS["calls"]
        feats = out[0] if isinstance(out, (tuple, list)) else getattr(out, "pooler_output", out)
        if isinstance(feats, (tuple, list)):
            feats = torch.cat(list(feats), 0)
        obs["feat_shape"] = list(feats.shape)
        obs["nonfinite"] = int((~torch.isfinite(feats)).sum())
        if a.save:
            torch.save(feats.float().cpu(), a.save)
    except Exception as exc:  # noqa: BLE001
        import traceback
        obs["error"] = repr(exc)[:400]
        obs["tb"] = traceback.format_exc()[-2000:]
    open(a.out, "w", encoding = "utf-8").write(json.dumps(obs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
