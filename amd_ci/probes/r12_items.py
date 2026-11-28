#!/usr/bin/env python3
"""Round-12 gfx1151 Linux items (observe only; judged by criteria/r12_judge.py). Tree = origin/main (head of the
--merged state). No token is read or passed; only public unsloth/* repos are prefetched (token=False).

  smoke   gfx1151 + one bf16 GEMM; quick W8A8 bench (1 shape, 2 M) and quick conv bench (2 shapes, no decode).
  w8a8    carbon12#6: probes/r12/w8a8_bench.py (Studio's RDNA W8A8 kernel, current vs M-bucketed tiles).
  conv    carbon6#3: probes/r12/conv_bench.py (aiter Triton conv2d vs MIOpen, VAE decoder shapes + random-weight decodes).
  adaln   carbon16#2: Qwen-Image-2.1 at Studio defaults 1024^2, fresh processes, A = Auto (env unset) vs
          B = UNSLOTH_DIFFUSION_ADALN_PRECOMPUTE=1 + UNSLOTH_DIFFUSION_ADALN_VERIFY=1; 1 warm-up + 1 measured render
          each. Order A B first; the remaining B A A B only run when B actually installed the table (otherwise the
          arms run identical code and more processes measure nothing). The first A also runs --mod-probe (cost of the
          per-call modulation Linears = the most a table could save).
  convflux  carbon6#3 decode share: ONLY if the conv item's FLUX.2 random-weight decode is >= 1.08x faster with aiter:
          unsloth/FLUX.2-klein-4B at Studio defaults 1024^2, 1 warm-up + 1 measured render, VAE share of the render.
"""
from __future__ import annotations

import json
from pathlib import Path

from r12_common import R9, e2e, prefetch, run_code, run_py

QWEN = "unsloth/Qwen-Image-2.1"
KLEIN = "unsloth/FLUX.2-klein-4B"
STUDIO_ENV = {"TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL": "1"}
CONV_GATE = 1.08

SMOKE = r"""
import json, time, torch
p = torch.cuda.get_device_properties(0)
out = dict(torch = torch.__version__, hip = getattr(torch.version, "hip", None), cuda = torch.cuda.is_available(),
           arch = getattr(p, "gcnArchName", None), total_gib = round(p.total_memory / 2**30, 3))
x = torch.randn(4096, 4096, device = "cuda", dtype = torch.bfloat16)
for _ in range(3): x @ x
torch.cuda.synchronize(); t = time.perf_counter()
for _ in range(20): x @ x
torch.cuda.synchronize(); out["gemm_tflops"] = round(20 * 2 * 4096**3 / (time.perf_counter() - t) / 1e12, 2)
print("R9JSON " + json.dumps(out), flush = True)
"""


def _bench(ctx, script, tag, extra_args, timeout):
    od = Path(ctx["work"]) / "bench"
    od.mkdir(parents = True, exist_ok = True)
    out = od / f"{tag}.json"
    return run_py(ctx, R9 / script, ["--root", ctx["main_backend"], "--out", out, *extra_args], out,
                  timeout = timeout, extra_env = STUDIO_ENV, tag = tag)


def item_smoke(ctx, obs):
    r = run_code(ctx, SMOKE, [], timeout = 600, tag = "smoke")
    obs["smoke"] = r
    obs["smoke_ok"] = bool(r.get("cuda")) and str(r.get("arch") or "").startswith("gfx1151")
    ctx["save"]()
    if not obs["smoke_ok"]:
        raise RuntimeError("smoke failed: no gfx1151 device")
    w = _bench(ctx, "w8a8_bench.py", "smoke_w8a8", ["--quick"], 900)
    c = _bench(ctx, "conv_bench.py", "smoke_conv", ["--quick", "--no-decode"], 1200)
    obs["smoke_w8a8"] = {k: w.get(k) for k in ("rc", "verdict", "error", "prepare", "rows", "profiler_kernels")}
    obs["smoke_conv"] = {k: c.get(k) for k in ("rc", "verdict", "error", "rows", "aiter_file")}
    ctx["save"]()
    if w.get("verdict") != "ok" or c.get("verdict") != "ok":
        raise RuntimeError(f"smoke benches failed: w8a8 {w.get('error')} conv {c.get('error')}")


def item_w8a8(ctx, obs):
    obs["w8a8"] = _bench(ctx, "w8a8_bench.py", "w8a8", [], 3600)
    ctx["save"]()


def item_conv(ctx, obs):
    obs["conv"] = _bench(ctx, "conv_bench.py", "conv", [], 3600)
    ctx["save"]()


def _keep(r):
    keep = {k: r.get(k) for k in ("verdict", "error", "load_s", "rc", "wall_s", "adaln_install_calls", "adaln_env",
                                  "adaln_mode", "adaln_counts", "adaln_installed", "adaln_summary", "transformer_class",
                                  "vae_class", "vae_dtype", "proc_env", "mod_probe", "params", "pipeline_class")}
    keep["renders"] = [{k: x.get(k) for k in ("r", "wall_s", "s_per_step", "n_steps", "md5", "vae_s", "encode_s",
                                              "peak_alloc_gib", "rdna_mm", "transformer_quant", "speed_optims",
                                              "first_step_s")} for x in r.get("renders") or []]
    st_ = r.get("status") or {}
    keep["status"] = {k: st_.get(k) for k in ("transformer_quant", "transformer_quant_backend", "speed_mode",
                                              "attention_backend", "offload_policy", "cpu_offload", "denoiser_offload",
                                              "memory_mode", "family")}
    keep["load_vmhwm_gib"], keep["end_vmhwm_gib"] = r.get("load_vmhwm_gib"), r.get("end_vmhwm_gib")
    return keep


def item_adaln(ctx, obs):
    res: dict = {"arms": []}
    obs["adaln"] = res
    res["prefetch"] = prefetch(ctx, QWEN)
    ctx["save"]()
    gen = json.dumps({"width": 1024, "height": 1024})

    def arm(i, which, mod_probe = False):
        env = {"UNSLOTH_DIFFUSION_ADALN_PRECOMPUTE": None, "UNSLOTH_DIFFUSION_ADALN_VERIFY": None}
        if which == "B":
            env = {"UNSLOTH_DIFFUSION_ADALN_PRECOMPUTE": "1", "UNSLOTH_DIFFUSION_ADALN_VERIFY": "1"}
        tag = f"adaln_{i}_{which}"
        args = ["--renders", "2", "--gen", gen] + (["--mod-probe"] if mod_probe else [])
        r = e2e(ctx, ctx["main_backend"], QWEN, tag, args, extra_env = env)
        k = _keep(r)
        k.update(arm = which, i = i)
        res["arms"].append(k)
        ctx["save"]()
        return k
    arm(0, "A", mod_probe = True)
    b = arm(1, "B")
    engaged = bool(b.get("adaln_installed")) or int((b.get("adaln_counts") or {}).get("installs") or 0) > 0
    res["b_engaged"] = engaged
    if not engaged:
        res["not_run"] = ("B did not install the AdaLN table (decline recorded): A and B run identical code, so the "
                          "remaining B A A B processes were not run")
        ctx["save"]()
        return
    for i, w in enumerate("BAAB", start = 2):
        arm(i, w)


def item_convflux(ctx, obs):
    res: dict = {}
    obs["convflux"] = res
    d = ((obs.get("conv") or {}).get("decode") or {}).get("flux2") or {}
    sp = d.get("speedup")
    res["gate_speedup"] = sp
    if not isinstance(sp, (int, float)) or sp < CONV_GATE:
        res["not_run"] = f"FLUX.2 random-weight decode speedup {sp} under {CONV_GATE}: no decode-share render (kill line)"
        ctx["save"]()
        return
    res["prefetch"] = prefetch(ctx, KLEIN)
    r = e2e(ctx, ctx["main_backend"], KLEIN, "convflux_0_A", ["--renders", "2", "--gen",
                                                               json.dumps({"width": 1024, "height": 1024})])
    res["arm"] = _keep(r)
    ctx["save"]()


ITEMS = {"smoke": item_smoke, "w8a8": item_w8a8, "conv": item_conv, "adaln": item_adaln, "convflux": item_convflux}
