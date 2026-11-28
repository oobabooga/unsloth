#!/usr/bin/env python3
"""gfx1151 items for the FLUX.2 VAE Triton conv2d route (observe only; judged by criteria/c2d_judge.py).
base tree = origin/main (the --patch-base), head tree = base + the branch patch. No token is read or passed; only the
public unsloth/FLUX.2-klein-4B repo is prefetched (token=False).

  smoke  gfx1151; the head tree's vendored kernels on the GPU: router self-check, then each of the 11 routed klein
         1024^2 decoder shapes, MIOpen (F.conv2d) vs Triton interleaved (5 rounds x 20 iters, median ms), error vs an
         fp32 F.conv2d of the same bf16 operands for both, top profiler kernel per arm.
  e2e    klein 4B 1024^2 4 steps, Studio defaults. Fresh processes M B M B M B (M = main tree, B = branch tree), each
         with its own EMPTY Triton cache (every B process is a cold start: the prebuild child races the first decode),
         5 renders each + an fp32-VAE decode of the last latent. Then B_restart (the last B's warm cache and manifest:
         a restarted Studio), and B_kill (branch tree, UNSLOTH_DIFFUSION_ROCM_TRITON_CONV2D=0: must be md5-identical
         to M).
"""
from __future__ import annotations

import json
from pathlib import Path

from c2d_common import R9, prefetch, run_py

KLEIN = "unsloth/FLUX.2-klein-4B"
STUDIO_ENV = {"TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL": "1"}
GEN = {"width": 1024, "height": 1024, "steps": 4, "guidance": 1.0}

SMOKE = r'''
import json, statistics, sys, time
sys.path.insert(0, sys.argv[1])
import torch, torch.nn.functional as F
out = {}
p = torch.cuda.get_device_properties(0)
out.update(torch = torch.__version__, hip = getattr(torch.version, "hip", None), arch = getattr(p, "gcnArchName", None))
import triton
out["triton"] = triton.__version__
from core.inference import diffusion_rocm_triton_conv2d as R
mod = R._kernels()
out["import_error"] = R._STATE.get("import_error")
dev = torch.device("cuda", 0)
out["selfcheck"] = R._selfcheck(torch, mod, dev, torch.bfloat16)
out["selfcheck_error"] = R._STATE.get("selfcheck_error")
SH = [((1,512,128,128),(512,512,3,3)), ((1,512,256,256),(512,512,3,3)), ((1,512,512,512),(512,512,3,3)),
      ((1,512,512,512),(256,512,3,3)), ((1,256,512,512),(256,256,3,3)), ((1,512,512,512),(256,512,1,1)),
      ((1,256,1024,1024),(256,256,3,3)), ((1,256,1024,1024),(128,256,3,3)), ((1,128,1024,1024),(128,128,3,3)),
      ((1,256,1024,1024),(128,256,1,1)), ((1,128,1024,1024),(3,128,3,3))]
rows = []
g = torch.Generator(device = "cpu").manual_seed(0)
for xs, ws in SH:
    route = R.route_for(xs, ws, (1, 1), (ws[2] // 2,) * 2, (1, 1), "bfloat16")
    x = torch.randn(xs, generator = g).to(dev, torch.bfloat16)
    w = (torch.randn(ws, generator = g) * (1.0 / (ws[1] * ws[2] * ws[3]) ** 0.5)).to(dev, torch.bfloat16)
    b = (torch.randn(ws[0], generator = g) * 0.1).to(dev, torch.bfloat16)
    pad = ws[2] // 2
    m = lambda: F.conv2d(x, w, b, padding = pad)
    t = lambda: R._run(mod, route, x, w, b)
    with torch.no_grad():
        ref = F.conv2d(x.float(), w.float(), b.float(), padding = pad)
        ym, yt = m(), t()
        em = float((ym.float() - ref).norm() / ref.norm()); et = float((yt.float() - ref).norm() / ref.norm())
        times = {"M": [], "T": []}
        for _ in range(5):
            for arm, fn in (("M", m), ("T", t)):
                for _ in range(3): fn()
                torch.cuda.synchronize()
                e0, e1 = torch.cuda.Event(enable_timing = True), torch.cuda.Event(enable_timing = True)
                e0.record()
                for _ in range(20): fn()
                e1.record(); e1.synchronize()
                times[arm].append(e0.elapsed_time(e1) / 20)
        kern = {}
        from torch.profiler import ProfilerActivity, profile
        for arm, fn in (("M", m), ("T", t)):
            with profile(activities = [ProfilerActivity.CUDA]) as prof:
                fn(); torch.cuda.synchronize()
            ev = sorted([e for e in prof.key_averages() if (getattr(e, "self_device_time_total", 0) or 0) > 0],
                        key = lambda e: -e.self_device_time_total)
            kern[arm] = [e.key[:90] for e in ev[:2]]
    mm, mt = statistics.median(times["M"]), statistics.median(times["T"])
    rows.append(dict(x = xs, w = ws, route = route, miopen_ms = round(mm, 4), triton_ms = round(mt, 4),
                     speedup = round(mm / mt, 3), aa_spread = round((max(times["M"]) - min(times["M"])) / mm, 4),
                     err_miopen = em, err_triton = et, err_ratio = round(et / em, 3), kernels = kern))
    del x, w, b, ref, ym, yt
    torch.cuda.empty_cache()
out["shapes"] = rows
vc = {}
for k in ("_conv2d_1x1_kernel", "_conv2d_3x3_cblocked_kernel", "_ncx_to_cblocked_kernel"):
    try:
        vc[k] = sum(len(v[0]) for v in getattr(mod, k).device_caches.values())
    except Exception as exc:
        vc[k] = repr(exc)[:100]
out["variants_compiled"] = vc
print("R9JSON " + json.dumps(out), flush = True)
'''


def item_smoke(ctx, obs):
    from c2d_common import run_code
    obs["smoke"] = run_code(ctx, SMOKE, [ctx["backend"]], timeout = 1800, extra_env = dict(
        STUDIO_ENV, TRITON_CACHE_DIR = str(Path(ctx["scratch"]) / "triton_smoke")), tag = "smoke")
    ctx["save"]()
    s = obs["smoke"]
    if not s.get("selfcheck") or not s.get("shapes"):
        raise RuntimeError(f"smoke failed: {json.dumps(s)[:600]}")


def _arm(ctx, tree, tag, cache, extra = None, renders = 5):
    od = Path(ctx["work"]) / "e2e"
    od.mkdir(parents = True, exist_ok = True)
    env = dict(STUDIO_ENV, TRITON_CACHE_DIR = str(cache))
    env.update(extra or {})
    rec = run_py(ctx, R9 / "e2e_arm.py", ["--root", tree, "--model", KLEIN, "--tag", tag, "--out", od,
                                          "--gen", json.dumps(GEN), "--renders", str(renders), "--fp32-check"],
                 None, timeout = 3600, extra_env = env, tag = tag)
    rp = od / f"{tag}.json"
    if rp.is_file():
        try:
            rec.update(json.loads(rp.read_text(encoding = "utf-8")))
        except Exception as exc:  # noqa: BLE001
            rec["parse_error"] = repr(exc)
    else:
        rec["verdict"] = rec.get("verdict") or "missing"
    return rec


def item_e2e(ctx, obs):
    res: dict = {"arms": []}
    obs["e2e"] = res
    res["prefetch"] = prefetch(ctx, KLEIN)
    ctx["save"]()
    scratch = Path(ctx["scratch"])
    last_b = None
    for i, which in enumerate("MBMBMB"):
        tree = ctx["base_backend"] if which == "M" else ctx["backend"]
        cache = scratch / f"triton_{i}_{which}"
        r = _arm(ctx, tree, f"e2e_{i}_{which}", cache)
        r.update(arm = which, i = i)
        res["arms"].append(r)
        ctx["save"]()
        if which == "B":
            last_b = cache
    r = _arm(ctx, ctx["backend"], "e2e_6_Brestart", last_b, renders = 3)
    r.update(arm = "B_restart", i = 6)
    res["arms"].append(r)
    ctx["save"]()
    r = _arm(ctx, ctx["backend"], "e2e_7_Bkill", scratch / "triton_7_kill",
             extra = {"UNSLOTH_DIFFUSION_ROCM_TRITON_CONV2D": "0"}, renders = 3)
    r.update(arm = "B_kill", i = 7)
    res["arms"].append(r)
    ctx["save"]()


ITEMS = {"smoke": item_smoke, "e2e": item_e2e}
