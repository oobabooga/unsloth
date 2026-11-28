#!/usr/bin/env python3
"""carbon12#6: Studio's RDNA W8A8 int8 GEMM (core/inference/diffusion_native_w8a8_rdna.py, the tree under test) at the
real Qwen-Image-2.1 / Wan 2.2 5B / Krea-2 Linear (N, K) shapes, M buckets 16..8192, current tiles (``_mm_cfg``) vs
M-bucketed configs:
  p1150   aiter#7282 gfx1150 gemm_a8w8_preshuffled DEFAULT.json buckets (d3ef238a9f), BLOCK / GROUP / warps / stages /
          waves_per_eu mapped onto Studio's kernel (no preshuffle; cache_modifier / nonkdim have no Studio knob)
  p1151   aiter#7270 gfx1151 gemm_a8w8_preshuffled DEFAULT.json buckets (643393ba3b), same mapping
  c_*     a fixed candidate grid (the ceiling an M-bucketed table could reach; picked per bucket from the same data)
Every arm launches Studio's own ``w8a8_mm`` Triton kernel object with only the tile constexprs / launch options changed;
``cur`` is checked bitwise against Studio's ``_mm_launch`` (the production entry). Bit identity: torch.equal of every
arm vs ``cur``. Timing: CUDA events, per (shape, M) 3 warm-ups then 5 rounds in rotated order, ITERS launches each,
``cur`` timed twice per round (cur, cur_aa) for the A/A spread. Engagement: torch.profiler kernel names (full) for one
launch per arm at one shape, plus Studio's own parity ``prepare()`` result.

  w8a8_bench.py --root <tree>/studio/backend --out R.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as stt
import sys
import time
import traceback
from pathlib import Path

SHAPES = {  # model: [(name, N, K, bias)]
    "qwen21": [("qkv_out", 4096, 4096, False), ("ffn_up_gate", 12288, 4096, False), ("ffn_down", 4096, 12288, False),
               ("modulation", 16384, 4096, False)],
    "wan22_5b": [("attn", 3072, 3072, True), ("ffn_up", 14336, 3072, True), ("ffn_down", 3072, 14336, True)],
    "krea2": [("q_out", 6144, 6144, False), ("kv", 1536, 6144, False), ("ffn_up", 16384, 6144, False),
              ("ffn_down", 6144, 16384, False), ("txt_attn", 2560, 2560, False), ("txt_ffn_up", 6912, 2560, False),
              ("txt_ffn_down", 2560, 6912, False)],
}
MS = (16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)
# (BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M, warps, stages, waves_per_eu or None)
P1150 = {8: (16, 64, 256, 1, 4, 1, 2), 16: (16, 32, 256, 1, 2, 1, 2), 32: (32, 32, 256, 1, 2, 1, 2),
         64: (32, 32, 256, 8, 1, 1, 0), 128: (64, 64, 256, 8, 4, 2, 1), 256: (32, 64, 256, 8, 4, 2, 1),
         512: (32, 64, 256, 8, 4, 2, 1), 1024: (128, 64, 128, 1, 4, 2, 1), 2048: (128, 64, 128, 1, 4, 2, 1),
         None: (128, 64, 128, 1, 4, 2, 1)}
P1151 = {8: (16, 64, 256, 1, 4, 1, 2), 16: (16, 64, 256, 1, 4, 1, 2), 32: (32, 64, 256, 1, 4, 2, 1),
         64: (16, 32, 256, 1, 2, 2, 1), 128: (32, 64, 256, 1, 4, 2, 1), 256: (64, 128, 128, 1, 4, 2, 0),
         512: (64, 128, 128, 1, 4, 2, 0), 1024: (64, 128, 128, 1, 4, 2, 0), 2048: (64, 128, 128, 1, 4, 2, 0),
         None: (64, 128, 128, 1, 4, 2, 0)}
GRID_SMALL = [(16, 64, 256, 1, 4, 1, None), (16, 128, 128, 1, 4, 2, None), (32, 64, 256, 1, 4, 2, None),
              (32, 128, 128, 8, 4, 2, None), (64, 128, 128, 8, 4, 2, None), (64, 64, 128, 8, 4, 2, None),
              (32, 64, 128, 8, 4, 2, None), (64, 64, 256, 8, 4, 2, None)]
GRID_LARGE = [(128, 64, 128, 1, 4, 2, None), (64, 128, 128, 8, 4, 2, None), (128, 128, 64, 8, 8, 2, None),
              (128, 128, 128, 8, 8, 2, None), (64, 128, 64, 8, 4, 2, None)]
ITERS = 10
ROUNDS = 5


def bucket(table: dict, m: int) -> tuple:
    for b in sorted(k for k in table if k is not None):
        if m <= b:
            return table[b]
    return table[None]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required = True)
    ap.add_argument("--out", required = True)
    ap.add_argument("--quick", action = "store_true", help = "smoke: one shape, two M, no grid")
    a = ap.parse_args()
    rec: dict = {"verdict": "running", "rows": [], "errors": []}
    outp = Path(a.out)

    def save():
        outp.write_text(json.dumps(rec, indent = 1, default = str), encoding = "utf-8")
    sys.path.insert(0, str(Path(a.root).resolve()))
    try:
        import torch
        import core.inference  # noqa: F401  Studio's process env (TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL)
        from core.inference import diffusion_native_w8a8_rdna as R
        rec["module_file"] = R.__file__
        rec["env"] = {k: os.environ.get(k) for k in ("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "MIOPEN_SEARCH_CUTOFF",
                                                     "UNSLOTH_NATIVE_INT8_RDNA_TILES", "UNSLOTH_NATIVE_INT8_RDNA_TRITON")}
        import triton
        rec["torch"], rec["triton"] = torch.__version__, triton.__version__
        rec["arch"] = torch.cuda.get_device_properties(0).gcnArchName
        rec["studio_cfg"] = {k: list(v) for k, v in R._CFG.items()}
        dev = torch.device("cuda", 0)
        rec["prepare"] = R.prepare(dev)
        rec["prepare_state"] = R.state()
        kern = R._kernels()
        save()

        def launch(cfg, xq, xs, wq, ws, bias, out):
            bm, bn, bk, gm, warps, stages, wpe = cfg
            m, k = xq.shape
            n = wq.shape[0]
            extra = {} if wpe is None else {"waves_per_eu": int(wpe)}
            kern.w8a8_mm[(triton.cdiv(m, bm) * triton.cdiv(n, bn),)](
                xq, wq, xs, ws, bias if bias is not None else ws, out, m, n, k,
                HAS_BIAS = bias is not None, EVEN_K = k % bk == 0, BLOCK_M = bm, BLOCK_N = bn, BLOCK_K = bk,
                GROUP_M = gm, num_warps = warps, num_stages = stages, enable_fp_fusion = False, **extra)

        gen = torch.Generator(device = "cpu").manual_seed(1234)
        shapes = [(mdl, nm, n, k, b) for mdl, lst in SHAPES.items() for nm, n, k, b in lst]
        ms = MS
        if a.quick:
            shapes, ms = shapes[:1], (64, 4096)
        prof_done = False
        for mdl, nm, n, k, has_b in shapes:
            wq = torch.randint(-127, 128, (n, k), generator = gen, dtype = torch.int8).to(dev)
            ws = (torch.rand(n, generator = gen) * 1e-2 + 1e-4).to(dev)
            bias = torch.randn(n, generator = gen).to(torch.bfloat16).to(dev) if has_b else None
            for m in ms:
                row = {"model": mdl, "layer": nm, "N": n, "K": k, "M": m, "bias": has_b}
                try:
                    xq = torch.randint(-127, 128, (m, k), generator = gen, dtype = torch.int8).to(dev)
                    xs = (torch.rand(m, 1, generator = gen) * 1e-2 + 1e-4).to(dev)
                    cur = tuple(R._mm_cfg(m, k)) + (None,)
                    arms = {"cur": cur, "cur_aa": cur, "p1150": bucket(P1150, m), "p1151": bucket(P1151, m)}
                    if not a.quick:
                        for c in (GRID_SMALL if m < 1024 else GRID_LARGE):
                            arms["c_" + "x".join(map(str, c[:6]))] = c
                    row["cfg"] = {k_: list(v) for k_, v in arms.items()}
                    outs = {}
                    for an, cfg in arms.items():
                        o = torch.empty((m, n), device = dev, dtype = torch.bfloat16)
                        try:
                            launch(cfg, xq, xs, wq, ws, bias, o)
                            torch.cuda.synchronize()
                            outs[an] = o
                        except Exception as exc:  # noqa: BLE001
                            row.setdefault("build_errors", {})[an] = f"{type(exc).__name__}: {str(exc)[:300]}"
                    prod = R._mm_launch(xq, xs, wq, ws, bias, torch.bfloat16)
                    torch.cuda.synchronize()
                    row["cur_equals_production"] = bool(torch.equal(prod, outs["cur"]))
                    row["broken"] = R._STATE.get("broken")
                    row["bitwise_vs_cur"] = {an: bool(torch.equal(o, outs["cur"])) for an, o in outs.items()}
                    if m <= 1024:
                        ref = R.reference_mm(xq, xs, wq, ws, bias, torch.bfloat16)
                        row["maxabs_vs_fp64_ref"] = float((outs["cur"].float() - ref.float()).abs().max())
                    live = [an for an in arms if an in outs]
                    if not prof_done and m == 64:
                        from torch.profiler import ProfilerActivity, profile
                        names = {}
                        for an in ("cur", "p1150", "p1151"):
                            if an not in outs:
                                continue
                            with profile(activities = [ProfilerActivity.CUDA]) as pr:
                                launch(arms[an], xq, xs, wq, ws, bias, outs[an])
                                torch.cuda.synchronize()
                            names[an] = sorted({e.key for e in pr.key_averages()
                                                if (getattr(e, "self_device_time_total", 0) or 0) > 0})
                        rec["profiler_kernels"] = {"shape": f"{mdl}.{nm} M={m}", "by_arm": names}
                        prof_done = True
                    for _ in range(3):
                        for an in live:
                            launch(arms[an], xq, xs, wq, ws, bias, outs[an])
                    torch.cuda.synchronize()
                    times: dict = {an: [] for an in live}
                    for r in range(ROUNDS):
                        order = live[r % len(live):] + live[:r % len(live)]
                        for an in order:
                            e0 = torch.cuda.Event(enable_timing = True)
                            e1 = torch.cuda.Event(enable_timing = True)
                            e0.record()
                            for _ in range(ITERS):
                                launch(arms[an], xq, xs, wq, ws, bias, outs[an])
                            e1.record()
                            e1.synchronize()
                            times[an].append(e0.elapsed_time(e1) / ITERS)
                    row["ms"] = {an: round(stt.median(v), 5) for an, v in times.items()}
                    row["ms_all"] = {an: [round(x, 5) for x in v] for an, v in times.items()}
                    row["tops_cur"] = round(2 * m * n * k / (row["ms"]["cur"] * 1e-3) / 1e12, 2)
                    del xq, xs, outs, prod
                except Exception as exc:  # noqa: BLE001
                    row["error"] = f"{type(exc).__name__}: {str(exc)[:500]}"
                rec["rows"].append(row)
                print("W8A8", json.dumps({k_: row.get(k_) for k_ in ("model", "layer", "M", "ms")}), flush = True)
                save()
            del wq, ws, bias
            torch.cuda.empty_cache()
        rec["verdict"] = "ok"
    except BaseException as exc:  # noqa: BLE001
        rec["verdict"] = "error"
        rec["error"] = repr(exc)[:2000]
        rec["traceback"] = traceback.format_exc()[-4000:]
        print(rec["traceback"], flush = True)
    save()
    os._exit(0 if rec["verdict"] == "ok" else 3)


if __name__ == "__main__":
    t0 = time.time()
    sys.exit(main())
