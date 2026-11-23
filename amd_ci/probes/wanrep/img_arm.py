#!/usr/bin/env python3
"""One image arm in a fresh process: load through Studio's DiffusionBackend (Studio default load unless --load says
otherwise), render one prompt + seed ``--renders`` times, hash the final pixels (md5 + sha1) so processes compare bit for
bit. Records the reduction-config pin calls, inductor best_config picks and the load's resolved status.
Same record.json shape as arm.py (renders[].frames_sha / md5 / wall_s, reduction_pins, verdict)."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

PROMPT = "a red fox sitting in fresh snow at golden hour, detailed fur, shallow depth of field"
SEED = 20261123


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required = True)
    ap.add_argument("--model", required = True)
    ap.add_argument("--tag", required = True)
    ap.add_argument("--out", required = True)
    ap.add_argument("--tensors", required = True)
    ap.add_argument("--load", default = "{}")
    ap.add_argument("--gen", default = "{}")
    ap.add_argument("--renders", type = int, default = 2)
    ap.add_argument("--block-hooks", action = "store_true")
    ap.add_argument("--deterministic-algos", action = "store_true")
    a = ap.parse_args()
    out = Path(a.out) / a.tag
    out.mkdir(parents = True, exist_ok = True)
    rec: dict = {"tag": a.tag, "verdict": "running", "renders": [], "model": a.model,
                 "env": {k: v for k, v in os.environ.items()
                         if k.startswith(("UNSLOTH_", "PYTORCH_", "TORCHINDUCTOR_", "FLASH_", "HIP_", "MIOPEN_"))}}
    recp = out / "record.json"

    def save():
        recp.write_text(json.dumps(rec, indent = 1, default = str), encoding = "utf-8")

    sys.path.insert(0, str(Path(a.root).resolve()))
    try:
        import numpy as np
        import torch

        rec["torch"] = torch.__version__
        rec["hip"] = getattr(torch.version, "hip", None)
        rec["gpu"] = torch.cuda.get_device_name(0)
        from core.inference import diffusion_speed as DS
        pins: list = []
        pin0 = getattr(DS, "pin_reduction_configs", None)
        if pin0 is not None:
            def pin(kwargs, logger = None):
                r = pin0(kwargs, logger)
                pins.append(bool(r))
                return r
            DS.pin_reduction_configs = pin
        rec["reduction_pins"] = pins
        from core.inference.diffusion import DiffusionBackend
        import core.inference.diffusion as D
        rec["backend_file"] = D.__file__
        eng = DiffusionBackend()
        t0 = time.perf_counter()
        eng.begin_load(a.model, **json.loads(a.load))
        deadline = time.time() + 5400
        while True:
            p = eng.load_progress() or {}
            if p.get("phase") == "ready":
                break
            if p.get("phase") == "error" or p.get("error"):
                raise RuntimeError(f"load failed: {p.get('error')}")
            if time.time() > deadline:
                raise TimeoutError(str(p))
            time.sleep(0.5)
        rec["load_s"] = round(time.perf_counter() - t0, 2)
        st = eng.status() or {}
        rec["status"] = {k: st.get(k) for k in ("attention_backend", "transformer_quant", "text_encoder_quant",
                                                 "speed_mode", "speed_optims", "offload_policy", "family")}
        try:
            tr = eng._state.pipe.transformer
            rec["dit_compiled_whole"] = [getattr(tr, "_compiled_call_impl", None) is not None]
            blocks = [m for m in tr.modules() if getattr(m, "_compiled_call_impl", None) is not None]
            rec["blocks_compiled"] = len(blocks)
        except Exception as exc:  # noqa: BLE001
            rec["blocks_compiled"] = repr(exc)[:200]
        gen = json.loads(a.gen)
        params = dict(width = gen.pop("width", 1024), height = gen.pop("height", 1024), steps = gen.pop("steps", 8),
                      guidance = gen.pop("guidance", 0.0))
        params.update(gen)
        rec["params"] = params
        save()
        for r in range(a.renders):
            torch.cuda.synchronize()
            t = time.perf_counter()
            res = eng.generate(prompt = PROMPT, seed = SEED, **params)
            torch.cuda.synchronize()
            wall = time.perf_counter() - t
            img = res["images"][0].convert("RGB")
            arr = np.asarray(img, dtype = np.uint8)
            row = {"r": r, "wall_s": round(wall, 3), "frames_sha": hashlib.sha1(arr.tobytes()).hexdigest()[:16],
                   "md5": hashlib.md5(arr.tobytes()).hexdigest(), "frames_shape": list(arr.shape)}
            if r == 0:
                img.save(out / "r0.png")
            rec["renders"].append(row)
            print("RENDER", json.dumps(row), flush = True)
            save()
        ind = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
        rec["inductor_dir"] = ind
        if ind and Path(ind).is_dir():
            picks = {}
            for p in Path(ind).rglob("*.best_config"):
                try:
                    picks[p.name] = json.loads(p.read_text(encoding = "utf-8"))
                except Exception:  # noqa: BLE001
                    picks[p.name] = "unreadable"
            rec["inductor_best_configs"] = picks
        try:
            tun = torch.cuda.tunable
            rec["tunableop"] = {"enabled": tun.is_enabled(), "tuning": tun.tuning_is_enabled(),
                                "n_results": len(tun.get_results() or [])}
        except Exception as exc:  # noqa: BLE001
            rec["tunableop"] = repr(exc)[:200]
        rec["log"] = []
        rec["verdict"] = "ok"
    except Exception as exc:  # noqa: BLE001
        rec["verdict"] = "error"
        rec["error"] = repr(exc)[:2000]
        rec["traceback"] = traceback.format_exc()[-6000:]
        print(rec["traceback"], flush = True)
    save()
    print("ARM_DONE", a.tag, rec["verdict"], flush = True)
    os._exit(0 if rec["verdict"] == "ok" else 3)


if __name__ == "__main__":
    main()
