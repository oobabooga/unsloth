#!/usr/bin/env python3
"""C18-6A: one Qwen-Image-2.1 LoRA training run through Studio's training worker entry point
(``diffusion_lora_trainer.run_diffusion_training_process``: config dict -> normalized() -> trainer), fresh process.
Observes only.

Records every progress event with a wall timestamp (steady s/step = median gap over steps > --steady-from), losses,
grad norms, the trainer's peak, process max allocated / reserved, the engaged base precision, whether it compiled,
and the W8A8 engagement line (``int8 W8A8: N frozen linears``). The dataset is synthetic (--data, made by the
orchestrator), so nothing but the public base weights is fetched. The int8 arm's ROCm opt-in
(UNSLOTH_DIFFUSION_ROCM_W8A8_TRAIN=1, UNSLOTH_DIFFUSION_INT8_TRAIN=w8a8) comes from the caller's environment.

  run_train.py --backend B --precision bf16|nf4|int8 --seed S --data D --work W --out F [--steps 30] [--resolution 512]
  run_train.py --make-data D [--n 8] [--resolution 512]      (no GPU: writes the synthetic image + caption set)
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics as st
import sys
import time
import traceback
from pathlib import Path

CAPTIONS = ["a red fox in fresh snow", "a lighthouse on a rocky coast at dusk", "a bowl of ripe oranges on a table",
            "a vintage bicycle against a brick wall", "a mountain lake with pine trees", "a cup of coffee on a desk",
            "a small sailboat on calm water", "a city street at night with neon signs"]


def make_data(d: Path, n: int, res: int) -> None:
    import numpy as np
    from PIL import Image
    d.mkdir(parents = True, exist_ok = True)
    rng = np.random.default_rng(1234)
    yy, xx = np.mgrid[0:res, 0:res] / res
    for i in range(n):
        f1, f2, ph = rng.uniform(1, 6), rng.uniform(1, 6), rng.uniform(0, 6.28)
        img = np.stack([np.sin(f1 * 6.28 * xx + ph), np.cos(f2 * 6.28 * yy - ph), np.sin(6.28 * (xx + yy) * (i + 1) / n)],
                       -1)
        img = ((img * 0.5 + 0.5) * 255).astype("uint8")
        Image.fromarray(img).save(d / f"img_{i:02d}.png")
        (d / f"img_{i:02d}.txt").write_text("trtcrd " + CAPTIONS[i % len(CAPTIONS)], encoding = "utf-8")


class _Q:
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend")
    ap.add_argument("--precision", choices = ("bf16", "nf4", "int8"))
    ap.add_argument("--seed", type = int, default = 42)
    ap.add_argument("--data")
    ap.add_argument("--work")
    ap.add_argument("--out")
    ap.add_argument("--base-model", default = "unsloth/Qwen-Image-2.1")
    ap.add_argument("--steps", type = int, default = 30)
    ap.add_argument("--resolution", type = int, default = 512)
    ap.add_argument("--rank", type = int, default = 16)
    ap.add_argument("--steady-from", type = int, default = 10)
    ap.add_argument("--make-data")
    ap.add_argument("--n", type = int, default = 8)
    a = ap.parse_args()
    if a.make_data:
        make_data(Path(a.make_data), a.n, a.resolution)
        return 0
    out = Path(a.out)
    res: dict = {"precision": a.precision, "seed": a.seed, "base_model": a.base_model, "steps": a.steps,
                 "resolution": a.resolution, "backend": a.backend,
                 "env": {k: os.environ.get(k) for k in ("UNSLOTH_DIFFUSION_ROCM_W8A8_TRAIN",
                                                        "UNSLOTH_DIFFUSION_INT8_TRAIN")}}

    def save():
        out.write_text(json.dumps(res, default = str), encoding = "utf-8")
    try:
        os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        sys.path.insert(0, a.backend)
        os.chdir(a.backend)
        try:
            from utils.rocm_flash_attn import default_triton_amd_env
            default_triton_amd_env()
        except Exception:  # noqa: BLE001
            pass
        import torch

        from core.training import diffusion_lora_trainer as dlt
        res["torch"], res["hip"] = torch.__version__, torch.version.hip
        res["arch"] = getattr(torch.cuda.get_device_properties(0), "gcnArchName", None)
        run_dir = Path(a.work) / f"run_{a.precision}_{a.seed}"
        shutil.rmtree(run_dir, ignore_errors = True)
        config = {"base_model": a.base_model, "data_dir": a.data, "output_dir": str(run_dir),
                  "resolution": a.resolution, "train_steps": a.steps, "learning_rate": 1e-4, "lora_rank": a.rank,
                  "lr_scheduler": "constant_with_warmup", "lr_warmup_steps": 5, "base_precision": a.precision,
                  "compile_transformer": "auto", "seed": a.seed, "adapter_name": f"r8_{a.precision}_{a.seed}",
                  "log_every": 1, "hf_token": None}
        res["config"] = config
        prog = []
        t0 = time.perf_counter()

        def sink(ev):
            if ev.get("type") == "progress":
                prog.append({"t": round(time.perf_counter() - t0, 3), **{k: ev.get(k) for k in (
                    "step", "loss", "grad_norm", "samples_per_second", "peak_memory_gb")}})
        torch.cuda.reset_peak_memory_stats()
        q_ev, q_stop = _Q(sink), _Q()
        try:
            dlt.run_diffusion_training_process(event_queue = q_ev, stop_queue = q_stop, config = config)
        except Exception as exc:  # noqa: BLE001
            res["raised"] = f"{type(exc).__name__}: {exc}"
            res["traceback"] = traceback.format_exc()[-3000:]
        res["wall_s"] = round(time.perf_counter() - t0, 1)
        res["max_allocated_gib"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
        res["max_reserved_gib"] = round(torch.cuda.max_memory_reserved() / 2 ** 30, 2)
        ev = q_ev.items
        res["progress"] = prog
        res["losses"] = [p.get("loss") for p in prog]
        res["steps_seen"] = len(prog)
        gaps = [prog[i]["t"] - prog[i - 1]["t"] for i in range(1, len(prog))
                if (prog[i].get("step") or 0) > a.steady_from]
        res["steady_s_per_step"] = round(st.median(gaps), 4) if gaps else None
        res["steady_gaps_n"] = len(gaps)
        if prog:
            res["trainer_peak_memory_gb"] = max((p.get("peak_memory_gb") or 0) for p in prog)
            res["samples_per_second_last"] = prog[-1].get("samples_per_second")
        res["errors"] = [str(e.get("message") or e.get("error"))[:600] for e in ev if e.get("type") == "error"]
        res["warnings"] = [str(e.get("message"))[:400] for e in ev if e.get("type") == "warning"]
        res["w8a8_line"] = next((w for w in res["warnings"] if w.startswith("int8 W8A8:")), None)
        res["w8a8_fallback"] = next((w for w in res["warnings"] if "W8A8 unavailable" in w), None)
        loaded = [e for e in ev if e.get("type") == "model_load_completed"]
        if loaded:
            res["compiled"] = loaded[-1].get("compiled")
            res["base_precision_engaged"] = loaded[-1].get("base_precision")
        done = [e for e in ev if e.get("type") == "complete"]
        res["complete"] = bool(done)
        shutil.rmtree(run_dir, ignore_errors = True)
    except Exception as exc:  # noqa: BLE001
        res["error"] = repr(exc)[:400]
        res["traceback"] = traceback.format_exc()[-3000:]
    save()
    return 0


if __name__ == "__main__":
    sys.exit(main())
