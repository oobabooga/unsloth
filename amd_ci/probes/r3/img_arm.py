#!/usr/bin/env python3
"""One image arm in a fresh process: load through Studio's DiffusionBackend (``--load`` kwargs, {} = Studio default),
render one prompt + seed ``--renders`` times at Studio's default steps / guidance (only ``--gen`` keys are passed).
Records load s, wall s per render, peak allocated / reserved GiB, host RAM low-water mark, md5 and pixel mean / std of
the image (black-frame check), the load's resolved status, and a PNG of the last render. Observes only."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
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
    ap.add_argument("--load", default = "{}")
    ap.add_argument("--gen", default = "{}")
    ap.add_argument("--renders", type = int, default = 2)
    a = ap.parse_args()
    out = Path(a.out) / a.tag
    out.mkdir(parents = True, exist_ok = True)
    rec: dict = {"tag": a.tag, "verdict": "running", "renders": [], "model": a.model, "load": json.loads(a.load),
                 "prompt": PROMPT, "seed": SEED}
    recp = out / "record.json"

    def save():
        recp.write_text(json.dumps(rec, indent = 1, default = str), encoding = "utf-8")

    low = {"avail": None}

    def watch():
        try:
            import psutil
        except Exception:  # noqa: BLE001
            return
        while True:
            av = psutil.virtual_memory().available
            low["avail"] = av if low["avail"] is None else min(low["avail"], av)
            time.sleep(1.0)
    threading.Thread(target = watch, daemon = True).start()
    sys.path.insert(0, str(Path(a.root).resolve()))
    try:
        import numpy as np
        import torch

        rec["torch"], rec["hip"] = torch.__version__, getattr(torch.version, "hip", None)
        rec["gpu"], rec["platform"] = torch.cuda.get_device_name(0), sys.platform
        from core.inference.diffusion import DiffusionBackend
        eng = DiffusionBackend()
        t0 = time.perf_counter()
        eng.begin_load(a.model, **json.loads(a.load))
        deadline = time.time() + float(os.environ.get("MQ_LOAD_DEADLINE_S", "14400"))
        last = None
        last_print = 0.0
        while True:
            p = eng.load_progress() or {}
            if p.get("phase") != last or time.time() - last_print > 60:
                last = p.get("phase")
                last_print = time.time()
                print("LOAD_PHASE", round(time.perf_counter() - t0, 1), json.dumps(p, default = str)[:400], flush = True)
            if p.get("phase") == "ready":
                break
            if p.get("phase") == "error" or p.get("error"):
                raise RuntimeError(f"load failed: {p.get('error')}")
            if time.time() > deadline:
                raise TimeoutError(str(p))
            time.sleep(0.5)
        rec["load_s"] = round(time.perf_counter() - t0, 2)
        st = eng.status() or {}
        rec["status"] = {k: (v if isinstance(v, (str, int, float, bool, type(None), list)) else str(v))
                         for k, v in st.items() if not isinstance(v, dict)}
        gen = json.loads(a.gen)
        rec["params"] = gen
        save()
        for r in range(a.renders):
            low["avail"] = None
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            t = time.perf_counter()
            res = eng.generate(prompt = PROMPT, seed = SEED, **gen)
            torch.cuda.synchronize()
            wall = time.perf_counter() - t
            img = res["images"][0].convert("RGB")
            arr = np.asarray(img, dtype = np.uint8)
            row = {"r": r, "wall_s": round(wall, 3), "md5": hashlib.md5(arr.tobytes()).hexdigest(),
                   "shape": list(arr.shape), "pix_mean": float(arr.mean()), "pix_std": float(arr.std()),
                   "peak_alloc_gib": round(torch.cuda.max_memory_allocated() / 2 ** 30, 3),
                   "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 2 ** 30, 3),
                   "min_host_avail_gib": round((low["avail"] or 0) / 2 ** 30, 2),
                   "steps": res.get("steps") if isinstance(res, dict) else None}
            img.save(out / f"r{r}.png")
            rec["renders"].append(row)
            print("RENDER", json.dumps(row), flush = True)
            save()
        rec["verdict"] = "ok"
    except BaseException as exc:  # noqa: BLE001
        rec["verdict"] = "error"
        rec["error"] = repr(exc)[:2000]
        rec["traceback"] = traceback.format_exc()[-6000:]
        print(rec["traceback"], flush = True)
    save()
    print("ARM_DONE", a.tag, rec["verdict"], flush = True)
    os._exit(0 if rec["verdict"] == "ok" else 3)


if __name__ == "__main__":
    main()
