#!/usr/bin/env python3
"""Probe (PR 19208): tiny random Krea-2 transformer on the real card, this checkout's Studio modules.

Observes only. Per case (padded prompt layouts): fp32 stock masked reference, bf16 stock masked (base path, no trim),
bf16 text trim alone (base Studio path), and at a checkout carrying diffusion_krea2_prefix_kv the bf16/fp32 prefix
path. Also: the INT8 route Auto installs (and forced), per-block torch.compile graph breaks / unique graphs over two
prompts. One child process (killed on timeout: Windows arms that never exit).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

CHILD = r'''
import copy, json, os, sys, time, traceback, types
sys.path.insert(0, sys.argv[1])
os.environ.pop("UNSLOTH_DIFFUSION_INT8_ATTENTION_KERNEL", None)
import torch, diffusers
from diffusers.models.transformers import transformer_krea2
from core.inference import diffusion_krea2_text_trim as trim
from core.inference import diffusion_int8_attention_dispatch as D
try:
    from core.inference import diffusion_krea2_prefix_kv as pkv
except ImportError:
    pkv = None
out = {"trim_file": trim.__file__, "has_prefix": pkv is not None, "torch": torch.__version__,
       "hip": getattr(torch.version, "hip", None), "diffusers": diffusers.__version__,
       "cuda_available": torch.cuda.is_available(), "platform": sys.platform, "errors": {}}
dev = torch.device(os.environ.get("KREA_PROBE_DEVICE") or "cuda:0")
out["device"] = str(dev)
out["arch"] = getattr(torch.cuda.get_device_properties(0), "gcnArchName", None) if dev.type == "cuda" else None
TEXT, GRID, LAYERS, DIM, BLOCKS, CIN = 512, 16, 2, 64, 2, 16
CASES = {"b1_v37": (37,), "b1_v200": (200,), "b2_v37_150": (37, 150)}

def build():
    torch.manual_seed(0)
    m = diffusers.Krea2Transformer2DModel(
        in_channels=CIN, num_layers=BLOCKS, attention_head_dim=64, num_attention_heads=4, num_key_value_heads=2,
        intermediate_size=512, timestep_embed_dim=64, text_hidden_dim=DIM, num_text_layers=LAYERS,
        text_num_attention_heads=2, text_num_key_value_heads=2, text_intermediate_size=128,
        num_layerwise_text_blocks=1, num_refiner_text_blocks=1, axes_dims_rope=(16, 24, 24)).eval()
    g = torch.Generator().manual_seed(123)
    with torch.no_grad():
        for p in m.parameters():  # no zero-init projections: every path moves the output
            p.copy_(torch.randn(p.shape, generator=g) * 0.05)
    return m

BASE = build()

def posids():
    # diffusers.Krea2Pipeline.prepare_position_ids (0.42.1), inlined: the pipeline module import is not needed
    img = torch.zeros(GRID, GRID, 3, device=dev)
    img[..., 1] = torch.arange(GRID, device=dev)[:, None]
    img[..., 2] = torch.arange(GRID, device=dev)[None, :]
    return torch.cat([torch.zeros(TEXT, 3, device=dev), img.reshape(GRID * GRID, 3)], dim=0)

def inputs(valid, suffix=3):
    g = torch.Generator().manual_seed(1 + sum(valid))
    b = len(valid)
    mask = torch.zeros(b, TEXT, dtype=torch.bool)
    for i, n in enumerate(valid):
        mask[i, :n] = True
        mask[i, TEXT - suffix:] = True
    pos = posids()
    return {"hidden_states": torch.randn(b, GRID * GRID, CIN, generator=g).to(dev),
            "encoder_hidden_states": torch.randn(b, TEXT, LAYERS, DIM, generator=g).to(dev),
            "timestep": torch.full((b,), 0.5).to(dev), "position_ids": pos,
            "encoder_attention_mask": mask.to(dev), "return_dict": False}

class Pipe:
    def __init__(self, t): self.transformer = t

def model(dtype, prefix=False, use_trim=False):
    m = copy.deepcopy(BASE).to(dev, dtype)
    n = pkv.install(m) if prefix else 0
    if use_trim:
        assert trim.install_for_pipe(Pipe(m))
    return m, n

def fwd(m, kw, dtype):
    k = {a: (v.to(dtype) if torch.is_tensor(v) and v.is_floating_point() and a != "timestep" else v) for a, v in kw.items()}
    with torch.no_grad():
        r = m(**k)[0].float()
    if dev.type == "cuda":
        torch.cuda.synchronize()
    return r

def d(a, b):
    return {"max_abs": float((a - b).abs().max()), "rel_l2": float((a - b).norm() / b.norm().clamp_min(1e-12))}

calls = []
real = transformer_krea2.dispatch_attention_fn
def spy(q, k, v, attn_mask=None, *a, **kw):
    if q.shape[1] > GRID * GRID:
        calls.append([attn_mask is not None, int(k.shape[1])])
    return real(q, k, v, attn_mask, *a, **kw)
transformer_krea2.dispatch_attention_fn = spy

out["cases"] = {}
for name, valid in CASES.items():
    c = {}
    try:
        kw = inputs(valid)
        m32, _ = model(torch.float32); ref32 = fwd(m32, kw, torch.float32); del m32
        c["ref32_absmean"] = float(ref32.abs().mean()); c["finite_ref"] = bool(torch.isfinite(ref32).all())
        m16, _ = model(torch.bfloat16); calls.clear(); masked16 = fwd(m16, kw, torch.bfloat16)
        c["masked16_calls"] = list(calls); del m16
        c["masked16_vs_ref32"] = d(masked16, ref32)
        t16, _ = model(torch.bfloat16, use_trim=True); calls.clear(); trim16 = fwd(t16, kw, torch.bfloat16)
        c["trim16_calls"] = list(calls); trim.uninstall(t16); del t16
        c["trim16_vs_ref32"] = d(trim16, ref32); c["trim16_vs_masked16"] = d(trim16, masked16)
        if pkv is not None:
            p16, n = model(torch.bfloat16, prefix=True, use_trim=True)
            before = dict(pkv.STATS); calls.clear(); pre16 = fwd(p16, kw, torch.bfloat16)
            c["prefix_installed"] = n
            c["prefix16_stats"] = {k: pkv.STATS[k] - before[k] for k in before}
            c["prefix16_calls"] = list(calls); trim.uninstall(p16); del p16
            c["prefix16_vs_ref32"] = d(pre16, ref32); c["prefix16_vs_masked16"] = d(pre16, masked16)
            c["prefix16_vs_trim16"] = d(pre16, trim16)
            p32, _ = model(torch.float32, prefix=True, use_trim=True)
            pre32 = fwd(p32, kw, torch.float32); trim.uninstall(p32); del p32
            c["prefix32_vs_ref32"] = d(pre32, ref32)
    except Exception:
        c["error"] = traceback.format_exc()[-3000:]
    out["cases"][name] = c
transformer_krea2.dispatch_attention_fn = real

# INT8 route: Auto (env unset) and forced, on the Studio load order (prefix install first, then the route)
target = types.SimpleNamespace(torch_device="cuda:0", device="cuda:0", dtype=torch.bfloat16)
out["route"] = {}
for mode in ("auto", "force"):
    r = {}
    try:
        if mode == "force":
            os.environ["UNSLOTH_DIFFUSION_INT8_ATTENTION_KERNEL"] = "force"
        else:
            os.environ.pop("UNSLOTH_DIFFUSION_INT8_ATTENTION_KERNEL", None)
        m, _ = model(torch.bfloat16, prefix=pkv is not None, use_trim=True)
        pipe = Pipe(m)
        t0 = time.time(); r["routed"] = int(D.install(pipe, target)); r["install_s"] = round(time.time() - t0, 2)
        r["backends"] = sorted({type(getattr(b.attn.processor, "_attention_backend", None)).__name__ + ":" +
                                str(getattr(getattr(b.attn.processor, "_attention_backend", None), "value",
                                            getattr(b.attn.processor, "_attention_backend", None)))
                                for b in m.transformer_blocks})
        stats0 = dict(D.ROUTE_STATS)
        kw = inputs(CASES["b2_v37_150"])
        o = fwd(m, kw, torch.bfloat16)
        r["route_calls"] = {k: D.ROUTE_STATS.get(k, 0) - stats0.get(k, 0) for k in D.ROUTE_STATS}
        ref = None
        m32, _ = model(torch.float32); ref = fwd(m32, kw, torch.float32); del m32
        r["vs_ref32"] = d(o, ref)
        D.remove(pipe); trim.uninstall(m); del m
    except Exception:
        r["error"] = traceback.format_exc()[-3000:]
    out["route"][mode] = r
os.environ.pop("UNSLOTH_DIFFUSION_INT8_ATTENTION_KERNEL", None)

# per-block compile over two prompts: graph breaks and unique graphs
comp = {}
try:
    import torch._dynamo as dyn
    from torch._dynamo.utils import counters
    dyn.reset(); counters.clear()
    m, _ = model(torch.bfloat16, prefix=pkv is not None, use_trim=True)
    for b in m.transformer_blocks:
        b.compile()
    for name in ("b1_v37", "b1_v200", "b1_v37"):
        fwd(m, inputs(CASES[name]), torch.bfloat16)
    comp["graph_breaks"] = int(sum(counters["graph_break"].values()))
    comp["graph_break_reasons"] = [str(k)[:200] for k in list(counters["graph_break"])[:10]]
    comp["unique_graphs"] = int(counters["stats"].get("unique_graphs", 0))
    comp["frames_ok"] = int(counters["frames"].get("ok", 0))
    comp["recompiles"] = int(sum(v for k, v in counters["recompiles"].items())) if "recompiles" in counters else None
    if pkv is not None:
        comp["prefix_stats_total"] = dict(pkv.STATS)
    trim.uninstall(m)
except Exception:
    comp["error"] = traceback.format_exc()[-3000:]
out["compile"] = comp
print("PROBE_JSON " + json.dumps(out, default=str))
'''


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True, type = Path)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--timeout", type = int, default = 1500)
    a = ap.parse_args()
    backend = str(a.checkout / "studio" / "backend")
    obs: dict = {"state": a.state}
    try:
        p = subprocess.run([sys.executable, "-c", CHILD, backend], cwd = backend, capture_output = True,
                           text = True, encoding = "utf-8", errors = "replace", timeout = a.timeout)
        line = next((ln for ln in p.stdout.splitlines() if ln.startswith("PROBE_JSON ")), None)
        if line:
            obs.update(json.loads(line[len("PROBE_JSON "):]))
        else:
            obs["error"] = (p.stderr or p.stdout)[-4000:]
        obs["rc"] = p.returncode
        obs["stderr_tail"] = (p.stderr or "")[-1500:]
    except subprocess.TimeoutExpired as e:
        obs["error"] = f"timeout after {a.timeout}s (killed)"
        obs["stderr_tail"] = str(e.stderr or "")[-1500:]
    a.out.write_text(json.dumps(obs, indent = 1, default = str), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
