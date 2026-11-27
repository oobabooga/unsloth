#!/usr/bin/env python3
"""A7 item 6: FlyDSL RDNA3 INT8 WMMA GEMM (kernels/gemm/rdna3_int8_gemm.py, autotuned entry) vs Studio's RDNA W8A8
Triton GEMM (diffusion_native_w8a8_rdna._launch_mm, its own tile), on gfx1151. Import first: the kill line includes
"does not import on ROCm 7.1x gfx1151".

  flydsl_bench.py --backend STUDIO_BACKEND --flydsl-src DIR --out F   (flydsl wheel on PYTHONPATH by the caller)

Exactness: FlyDSL i32 output vs torch._int_mm (torch.equal). Timing: bf16 output with per-row / per-column scales
(scale_mode row_col) vs Studio's kernel with the same scales and no bias, interleaved rounds (a7common.interleave)."""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# (name, K, N): Qwen-Image-2.1 (4096 / 12288), FLUX.3 klein (3072 / 18432 / 9216), Z-Image (3840 / 10240)
SHAPES = [("q21_qkv", 4096, 12288), ("q21_down", 12288, 4096), ("kl_fused", 3072, 18432), ("kl_down", 9216, 3072),
          ("z_up", 3840, 10240), ("z_down", 10240, 3840)]
MS = [4096, 9728]


def graph_times(fns, rounds, launches = 10):
    """name -> fn; each captured ``launches`` times in one CUDA graph; ABBA rounds of one replay each; per-launch ms."""
    import statistics as st

    import torch
    graphs = {}
    for k, f in fns.items():
        f()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(launches):
                f()
        g.replay()
        torch.cuda.synchronize()
        graphs[k] = g
    res = {k: {"ms": []} for k in fns}
    names = list(fns)
    for r in range(rounds):
        for k in (names if r % 2 == 0 else names[::-1]):
            a, b = torch.cuda.Event(enable_timing = True), torch.cuda.Event(enable_timing = True)
            a.record()
            graphs[k].replay()
            b.record()
            b.synchronize()
            res[k]["ms"].append(a.elapsed_time(b) / launches)
    for k in names:
        res[k]["med"] = st.median(res[k]["ms"])
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required = True)
    ap.add_argument("--flydsl-src", required = True)
    ap.add_argument("--out", required = True)
    ap.add_argument("--rounds", type = int, default = 7)
    ap.add_argument("--graph", action = "store_true", help = "time both kernels by CUDA-graph replay (device time)")
    a = ap.parse_args()
    obs: dict = {"rows": []}
    t0 = time.time()
    try:
        try:
            import flydsl  # noqa: F401 -- before torch, as FlyDSL's own tests (comgr preload)
            obs["flydsl_version"] = getattr(flydsl, "__version__", "?")
            obs["flydsl_file"] = getattr(flydsl, "__file__", None)
        except Exception:  # noqa: BLE001
            obs["import_error"] = traceback.format_exc()[-1500:]
            raise
        import torch
        sys.path.insert(0, a.flydsl_src)
        try:
            from flydsl.runtime.device import get_rocm_arch
            obs["flydsl_arch"] = str(get_rocm_arch())
            from kernels.gemm.rdna3_int8_gemm_autotune import rdna3_int8_gemm_autotuned
        except Exception:  # noqa: BLE001
            obs["import_error"] = traceback.format_exc()[-1500:]
            raise
        obs["imported"] = True
        from a7common import env_info, interleave
        obs["env"] = env_info()
        sys.path.insert(0, a.backend)
        from core.inference import diffusion_native_w8a8_rdna as R
        obs["studio_prepare"] = R.prepare(torch.device("cuda", 0))
        for name, k, n in SHAPES:
            for m in MS:
                row = {"shape": name, "m": m, "k": k, "n": n}
                try:
                    g = torch.Generator(device = "cuda").manual_seed(m + k + n)
                    xq = torch.randint(-127, 128, (m, k), generator = g, device = "cuda", dtype = torch.int8)
                    wq = torch.randint(-127, 128, (n, k), generator = g, device = "cuda", dtype = torch.int8)
                    xs = torch.rand(m, generator = g, device = "cuda") * 1e-3 + 1e-4
                    ws = torch.rand(n, generator = g, device = "cuda") * 1e-3 + 1e-4
                    ci = torch.empty((m, n), device = "cuda", dtype = torch.int32)
                    rdna3_int8_gemm_autotuned(ci, xq, wq)
                    torch.cuda.synchronize()
                    ref = torch._int_mm(xq, wq.t())
                    row["i32_equal"] = bool(torch.equal(ci, ref))
                    ob_f = torch.empty((m, n), device = "cuda", dtype = torch.bfloat16)
                    ob_s = torch.empty((m, n), device = "cuda", dtype = torch.bfloat16)

                    def fly():
                        rdna3_int8_gemm_autotuned(ob_f, xq, wq, scale_a = xs, scale_b = ws, out_dtype = "bf16",
                                                  scale_mode = "row_col")

                    def studio():
                        R._launch_mm(xq, xs, wq, ws, None, ob_s, m, n, k)
                    fly()
                    studio()
                    torch.cuda.synchronize()
                    refb = (ref.double() * xs.double()[:, None] * ws.double()[None, :])
                    row["bf16_rel_fly"] = float((ob_f.double() - refb).norm() / refb.norm())
                    row["bf16_rel_studio"] = float((ob_s.double() - refb).norm() / refb.norm())
                    row["bf16_equal_fly_studio"] = bool(torch.equal(ob_f, ob_s))
                    if a.graph:
                        res = graph_times({"fly": fly, "studio": studio}, a.rounds)
                        row["graph"] = True
                    else:
                        res = interleave({"fly": fly, "studio": studio}, a.rounds)
                    row["med_ms"] = {x: res[x]["med"] for x in ("fly", "studio")}
                    row["ms"] = {x: res[x]["ms"] for x in ("fly", "studio")}
                    row["speedup_fly"] = row["med_ms"]["studio"] / row["med_ms"]["fly"]
                    del xq, wq, ci, ref, ob_f, ob_s
                    torch.cuda.empty_cache()
                except Exception:  # noqa: BLE001
                    row["error"] = traceback.format_exc()[-800:]
                obs["rows"].append(row)
        sp = [r["speedup_fly"] for r in obs["rows"] if "speedup_fly" in r]
        if sp:
            import math
            obs["geomean_speedup_fly"] = math.exp(sum(math.log(x) for x in sp) / len(sp))
    except Exception:  # noqa: BLE001
        obs.setdefault("error", traceback.format_exc()[-1500:])
    obs["seconds"] = round(time.time() - t0, 1)
    from a7common import write
    write(a.out, obs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
