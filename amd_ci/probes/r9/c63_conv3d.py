#!/usr/bin/env python3
"""C6-3 patch-embed Conv3d probe (observes only). Two modes, each run in a fresh process by r9_items.py so a native
crash (segfault / AqlPacket abort) shows up as the child's return code, not as a lost job:

  crash  --n N : fp32 nn.Conv3d(3 -> C, k = stride = (2, 16, 16)) on (N, 3, 2, 16, 16), 3 calls; records each call's
                 wall, finiteness, and max |conv3d - F.linear| (the geometry-gated swap the item would ship), plus a
                 CPU fp64 reference on the first 64 rows. Writes JSON after every call, so a crash keeps what ran.
  share  --n N : Qwen3-VL vision tower built from its public config (random weights: timing does not depend on values),
                 bf16 on the GPU; times a whole vision forward on N patches with CUDA events, and the patch_embed
                 module inside it through forward hooks (engagement: hook call count), plus the fp32 conv3d alone.

  python c63_conv3d.py crash --n 113 --out o.json
  python c63_conv3d.py share --n 4032 --config DIR --out o.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time


def _save(path, obj):
    with open(path, "w", encoding = "utf-8") as fh:
        json.dump(obj, fh, indent = 1, default = str)


def crash(a, out):
    import torch
    import torch.nn.functional as F
    torch.manual_seed(0)
    c = a.channels
    dt = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[a.dtype]
    conv = torch.nn.Conv3d(3, c, kernel_size = (2, 16, 16), stride = (2, 16, 16), bias = True).cuda().to(dt)
    x = torch.randn(a.n, 3, 2, 16, 16, device = "cuda", dtype = torch.float32).to(dt)
    out.update(n = a.n, channels = c, dtype = a.dtype, input_bytes = x.numel() * x.element_size(), calls = [])
    _save(a.out, out)
    w2 = conv.weight.reshape(c, -1)
    for i in range(3):
        rec = {"i": i, "started": True}
        out["calls"].append(rec)
        _save(a.out, out)  # a crash inside the next call leaves "started" with no wall
        torch.cuda.synchronize()
        t = time.perf_counter()
        y = conv(x)
        torch.cuda.synchronize()
        rec["finite_first"] = bool(torch.isfinite(y).all())
        rec["wall_ms"] = round((time.perf_counter() - t) * 1e3, 3)
        rec["shape"] = list(y.shape)
        rec["finite"] = bool(torch.isfinite(y).all())
        lin = F.linear(x.reshape(a.n, -1), w2, conv.bias)
        rec["maxabs_vs_linear"] = float((y.reshape(a.n, c) - lin).abs().max())
        tol = 1e-4 if a.dtype == "fp32" else 3e-2
        rec["allclose_linear"] = bool(torch.allclose(y.reshape(a.n, c).float(), lin.float(), rtol = tol, atol = tol))
        if i == 0:
            k = min(64, a.n)
            ref = F.linear(x[:k].double().cpu().reshape(k, -1), w2.double().cpu(), conv.bias.double().cpu()).to(y.dtype).double()
            rec["maxabs_conv_vs_cpu64"] = float((y[:k].reshape(k, c).double().cpu() - ref).abs().max())
            rec["maxabs_linear_vs_cpu64"] = float((lin[:k].double().cpu() - ref).abs().max())
        t = time.perf_counter()
        for _ in range(5):
            F.linear(x.reshape(a.n, -1), w2, conv.bias)
        torch.cuda.synchronize()
        rec["linear_ms"] = round((time.perf_counter() - t) * 1e3 / 5, 3)
        _save(a.out, out)
    out["verdict"] = "ok"


def share(a, out):
    import torch
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(a.config)
    vcfg = getattr(cfg, "vision_config", cfg)
    out["vision_config"] = {k: getattr(vcfg, k, None) for k in ("hidden_size", "depth", "patch_size",
                                                               "temporal_patch_size", "spatial_merge_size",
                                                               "in_channels", "model_type")}
    from transformers.models.qwen3_vl import modeling_qwen3_vl as M
    torch.manual_seed(0)
    with torch.device("cuda"):
        model = M.Qwen3VLVisionModel._from_config(vcfg, torch_dtype = torch.bfloat16)
    model.eval()
    ps, tp = vcfg.patch_size, vcfg.temporal_patch_size
    side = {112: (8, 14), 113: None, 4032: (56, 72), 5600: (70, 80), 16384: (128, 128)}.get(a.n)
    if side is None:
        out["verdict"] = "skipped"
        out["why"] = f"N={a.n} has no even h x w grid (spatial merge 2)"
        return
    h, w = side
    grid = torch.tensor([[1, h, w]], device = "cuda")
    px = torch.randn(h * w, 3 * tp * ps * ps, device = "cuda", dtype = torch.bfloat16)
    pe = model.patch_embed
    ev: list = []

    def pre(_m, _i):
        e = torch.cuda.Event(enable_timing = True)
        e.record()
        ev.append([e, None])

    def post(_m, _i, _o):
        e = torch.cuda.Event(enable_timing = True)
        e.record()
        ev[-1][1] = e
    pe.register_forward_pre_hook(pre)
    pe.register_forward_hook(post)
    out.update(n = a.n, grid = [1, h, w], patch_embed_class = type(pe).__name__,
               patch_embed_proj = type(getattr(pe, "proj", None)).__name__)
    walls, pes = [], []
    with torch.inference_mode():
        for i in range(4):
            ev.clear()
            s, e = torch.cuda.Event(enable_timing = True), torch.cuda.Event(enable_timing = True)
            s.record()
            model(px, grid_thw = grid)
            e.record()
            e.synchronize()
            if i:  # first call = warm-up
                walls.append(s.elapsed_time(e))
                pes.append(sum(x.elapsed_time(y) for x, y in ev if y is not None))
        out["hook_calls_last"] = len(ev)
        # the item's case: the same patch embed in fp32 (h3_vision_fp32-style upcast), standalone
        conv = torch.nn.Conv3d(3, vcfg.hidden_size, kernel_size = (tp, ps, ps), stride = (tp, ps, ps)).cuda().float()
        x = px.float().reshape(-1, 3, tp, ps, ps)
        conv(x)
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(5):
            conv(x)
        torch.cuda.synchronize()
        out["fp32_conv3d_ms"] = round((time.perf_counter() - t) * 1e3 / 5, 3)
    import statistics as st
    out["encode_ms"] = [round(x, 3) for x in walls]
    out["patch_embed_ms"] = [round(x, 3) for x in pes]
    out["encode_ms_med"] = round(st.median(walls), 3)
    out["patch_embed_ms_med"] = round(st.median(pes), 3)
    out["share_bf16"] = round(st.median(pes) / st.median(walls), 5)
    out["share_fp32_conv_over_encode"] = round(out["fp32_conv3d_ms"] / st.median(walls), 5)
    out["verdict"] = "ok"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices = ("crash", "share"))
    ap.add_argument("--n", type = int, required = True)
    ap.add_argument("--channels", type = int, default = 1152)
    ap.add_argument("--dtype", default = "fp32", choices = ("fp32", "bf16", "fp16"))
    ap.add_argument("--config", default = "")
    ap.add_argument("--out", required = True)
    a = ap.parse_args()
    out: dict = {"mode": a.mode, "verdict": "running"}
    try:
        import torch
        p = torch.cuda.get_device_properties(0)
        out.update(torch = torch.__version__, hip = getattr(torch.version, "hip", None),
                   arch = getattr(p, "gcnArchName", None), platform = sys.platform)
        (crash if a.mode == "crash" else share)(a, out)
    except Exception:  # noqa: BLE001 - a Python-level failure is recorded; a native crash ends the process
        import traceback
        out["verdict"] = "error"
        out["error"] = traceback.format_exc()[-2000:]
    _save(a.out, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
