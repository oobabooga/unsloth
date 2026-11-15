#!/usr/bin/env python3
"""Probe: does this unsloth_zoo checkout route the Qwen3.5 GatedDeltaNet causal conv through its Triton
kernel on this GPU, does that match the torch path, and what does a training step cost either way.
Observes only; criteria/gdn_conv1d_engages.py judges."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path

KILL = "UNSLOTH_DISABLE_TRITON_CAUSAL_CONV1D"
DEV = os.environ.get("GDN_PROBE_DEVICE", "cuda")  # "cpu" only for a local dry run


def _sync(torch):
    if DEV == "cuda":
        torch.cuda.synchronize()


def _stats(gcc):
    return dict(gcc.GDN_CAUSAL_CONV1D_STATS)


def _reset(gcc):
    for k in gcc.GDN_CAUSAL_CONV1D_STATS:
        gcc.GDN_CAUSAL_CONV1D_STATS[k] = 0


def _cfg(Cfg, small):
    if small:
        dims = dict(vocab_size = 512, hidden_size = 128, intermediate_size = 256, num_hidden_layers = 4,
                    num_attention_heads = 4, num_key_value_heads = 2, head_dim = 32,
                    linear_num_key_heads = 2, linear_num_value_heads = 4,
                    linear_key_head_dim = 32, linear_value_head_dim = 32,
                    layer_types = ["linear_attention"] * 3 + ["full_attention"])
    else:
        # Qwen3.5-4B's GatedDeltaNet widths (conv_dim 8192), two GDN layers and one attention layer.
        dims = dict(vocab_size = 1024, hidden_size = 2560, intermediate_size = 9216, num_hidden_layers = 3,
                    num_attention_heads = 16, num_key_value_heads = 4, head_dim = 256,
                    linear_num_key_heads = 16, linear_num_value_heads = 32,
                    linear_key_head_dim = 128, linear_value_head_dim = 128,
                    layer_types = ["linear_attention"] * 2 + ["full_attention"])
    return Cfg(linear_conv_kernel_dim = 4, **dims)


def _step(model, ids):
    model.zero_grad(set_to_none = True)
    out = model(input_ids = ids, labels = ids)
    out.loss.backward()
    return out.loss.detach().float().item()


def _parity(torch, gcc, Cfg, Model):
    """fp32 tiny model: same weights and batch through Triton, then with the kill switch (torch path)."""
    torch.manual_seed(0)
    model = Model(_cfg(Cfg, True)).to(DEV).float()
    ids = torch.randint(0, 512, (2, 256), device = DEV)
    res = {}
    for arm, kill in (("default", "0"), ("kill_switch", "1")):
        os.environ[KILL] = kill
        _reset(gcc)
        loss = _step(model, ids)
        _sync(torch)
        grads = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}
        res[arm] = {"loss": loss, "stats": _stats(gcc), "grads": grads}
    os.environ[KILL] = "0"
    worst = 0.0
    for n, g in res["default"]["grads"].items():
        r = res["kill_switch"]["grads"][n]
        worst = max(worst, ((g - r).norm() / r.norm().clamp_min(1e-12)).item())
    conv = [n for n in res["default"]["grads"] if "conv1d" in n]
    return {
        "loss_default": res["default"]["loss"], "loss_kill_switch": res["kill_switch"]["loss"],
        "stats_default": res["default"]["stats"], "stats_kill_switch": res["kill_switch"]["stats"],
        "max_grad_rel_diff": worst, "n_conv_params": len(conv),
    }


def _bench_model(torch, gcc, Cfg, Model, iters):
    """bf16, Qwen3.5-4B GDN widths, B=1 T=2048: median fwd+bwd step time and the counters of one step."""
    torch.manual_seed(0)
    model = Model(_cfg(Cfg, False)).to(DEV).to(torch.bfloat16)
    ids = torch.randint(0, 1024, (1, 2048), device = DEV)
    _reset(gcc)
    loss = _step(model, ids)
    _sync(torch)
    stats = _stats(gcc)
    for _ in range(2):
        _step(model, ids)
    _sync(torch)
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        _step(model, ids)
        _sync(torch)
        times.append(time.perf_counter() - t0)
    times.sort()
    del model
    if DEV == "cuda":
        torch.cuda.empty_cache()
    return {"loss": loss, "stats": stats, "step_s": times, "median_s": times[len(times) // 2]}


def _bench_op(torch, gcc, iters):
    """The conv alone at the reporter's shape: zoo Triton vs the transformers torch path (MIOpen)."""
    import torch.nn.functional as F
    B, D, T, W = 1, 8192, 2048, 4
    g = torch.Generator(device = DEV).manual_seed(0)
    x0 = torch.randn(B, D, T, device = DEV, generator = g)
    w0 = torch.randn(D, W, device = DEV, generator = g) * 0.5
    b0 = torch.randn(D, device = DEV, generator = g) * 0.1
    dy = torch.randn(B, D, T, device = DEV, generator = g)

    def torch_path(x, w, b):
        return F.silu(F.conv1d(x, w.unsqueeze(1), b, padding = W - 1, groups = D)[..., :T])

    def triton_path(x, w, b):
        return gcc.triton_causal_conv1d(x, w, b, "silu")

    def grads(fn, dtype):
        x, w, b = (t.to(dtype).requires_grad_() for t in (x0, w0, b0))
        y = fn(x, w, b)
        y.backward(dy.to(dtype))
        return [y.float()] + [t.grad.float() for t in (x, w, b)]

    ref = grads(lambda x, w, b: torch_path(x.double(), w.double(), b.double()), torch.float32)
    out = {}
    for dtype in (torch.float32, torch.bfloat16):
        tri, tor = grads(triton_path, dtype), grads(torch_path, dtype)
        names = ("y", "dx", "dw", "db")
        out[str(dtype)] = {
            "triton_rel": {n: ((a - r).norm() / r.norm()).item() for n, a, r in zip(names, tri, ref)},
            "torch_rel": {n: ((a - r).norm() / r.norm()).item() for n, a, r in zip(names, tor, ref)},
        }
    timing = {}
    for name, fn in (("triton", triton_path), ("torch", torch_path)):
        x, w, b = (t.to(torch.bfloat16).requires_grad_() for t in (x0, w0, b0))
        d = dy.to(torch.bfloat16)
        for _ in range(3):
            fn(x, w, b).backward(d)
        _sync(torch)
        ts = []
        for _ in range(iters):
            t0 = time.perf_counter()
            fn(x, w, b).backward(d)
            _sync(torch)
            ts.append(time.perf_counter() - t0)
        ts.sort()
        timing[name] = {"median_ms": ts[len(ts) // 2] * 1e3, "min_ms": ts[0] * 1e3, "max_ms": ts[-1] * 1e3}
    out["timing_bf16_fwd_bwd"] = timing
    return out


def _pytest(checkout, python, tests, out_dir, state):
    present = [t for t in tests if (Path(checkout) / t).exists()]
    if not present:
        return {"selected": [], "rc": None}
    xml = out_dir / f"{state}_junit.xml"
    env = dict(os.environ, PYTHONPATH = checkout + os.pathsep + os.environ.get("PYTHONPATH", ""),
               UNSLOTH_IS_PRESENT = "1")
    p = subprocess.run([python, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-o", "addopts=",
                        f"--junitxml={xml}", *present], cwd = checkout, env = env,
                       capture_output = True, text = True, timeout = 3600)
    res = {"selected": present, "rc": p.returncode, "tail": p.stdout[-3000:], "stderr_tail": p.stderr[-1500:],
           "passed": 0, "failed": [], "skipped": 0}
    try:
        for case in ET.parse(xml).getroot().iter("testcase"):
            tid = f"{case.get('classname')}::{case.get('name')}"
            if case.find("failure") is not None or case.find("error") is not None:
                res["failed"].append(tid)
            elif case.find("skipped") is not None:
                res["skipped"] += 1
            else:
                res["passed"] += 1
    except Exception as e:
        res["junit_error"] = repr(e)
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--iters", type = int, default = 10)
    ap.add_argument("--tests", nargs = "*", default = [])
    args = ap.parse_args()
    checkout = str(Path(args.checkout).resolve())
    obs: dict = {"state": args.state, "checkout": checkout, "errors": {}}

    def save():
        args.out.write_text(json.dumps(obs, indent = 2, default = str), encoding = "utf-8")

    os.environ["UNSLOTH_IS_PRESENT"] = "1"
    os.environ[KILL] = "0"
    sys.path.insert(0, checkout)
    try:
        import torch
        obs["torch"] = torch.__version__
        obs["hip"] = torch.version.hip
        obs["cuda_available"] = torch.cuda.is_available()
        if obs["cuda_available"]:
            props = torch.cuda.get_device_properties(0)
            obs["device"] = props.name
            obs["arch"] = getattr(props, "gcnArchName", None)
        import triton
        obs["triton"] = triton.__version__
        import transformers
        obs["transformers"] = transformers.__version__
        from unsloth_zoo.temporary_patches import gdn_causal_conv1d as gcc
        obs["zoo_file"] = gcc.__file__
        obs["zoo_from_checkout"] = Path(gcc.__file__).resolve().is_relative_to(checkout)
        obs["real_causal_conv1d"] = gcc._real_causal_conv1d_available()
        gcc.patch_gdn_causal_conv1d()
        import transformers.models.qwen3_5.modeling_qwen3_5 as mq
        from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig as Cfg
        obs["modeling_conv_patched"] = bool(mq.causal_conv1d_fn is not None and gcc._is_marked(mq.causal_conv1d_fn))
    except Exception:
        obs["errors"]["setup"] = traceback.format_exc()
        save()
        return 0
    save()
    for name, fn in (("parity", lambda: _parity(torch, gcc, Cfg, mq.Qwen3_5ForCausalLM)),
                     ("model_bench", lambda: _bench_model(torch, gcc, Cfg, mq.Qwen3_5ForCausalLM, args.iters)),
                     ("op", lambda: _bench_op(torch, gcc, args.iters * 2))):
        try:
            obs[name] = fn()
        except Exception:
            obs["errors"][name] = traceback.format_exc()
        save()
    if args.tests:
        try:
            obs["pytest"] = _pytest(checkout, sys.executable, args.tests, args.out.parent, args.state)
        except Exception:
            obs["errors"]["pytest"] = traceback.format_exc()
    save()
    return 0


if __name__ == "__main__":
    sys.exit(main())
