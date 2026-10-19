"""Observe one torch build on this AMD GPU. Never judges: writes cells to --out JSON.

Runs inside the arm's venv. Cells: device info, bf16 matmul vs CPU fp32, SDPA fwd+bwd per
backend vs fp32 math, a matmul after SDPA (ROCm/TheRock#7992 broke kernels after SDPA),
torch._grouped_mm fwd+bwd vs a per-group loop (the bug behind Unsloth's torch 2.11 floor),
torch.compile, then Unsloth LoRA SFT in a subprocess (16-bit, then 4-bit).
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time
import traceback

ap = argparse.ArgumentParser()
ap.add_argument("--out", required = True)
ap.add_argument("--train-script", required = True)
ap.add_argument("--work", required = True)
ap.add_argument("--train-timeout", type = int, default = 1800)
args = ap.parse_args()

obs = {"cells": {}, "python": sys.version.split()[0], "platform": platform.platform()}


def flush():
    with open(args.out, "w", encoding = "utf-8") as f:
        json.dump(obs, f, indent = 1)


def cell(name, fn):
    t = time.time()
    try:
        r = fn() or {}
        obs["cells"][name] = {"ok": True, **r}
    except BaseException as e:  # noqa: BLE001 - a crash is the observation
        obs["cells"][name] = {
            "ok": False,
            "error": f"{type(e).__name__}: {e}"[:1500],
            "tb": traceback.format_exc()[-2500:],
        }
    obs["cells"][name]["s"] = round(time.time() - t, 2)
    print(f"[probe] {name}: {'ok' if obs['cells'][name]['ok'] else 'FAIL'} {obs['cells'][name].get('error', '')[:200]}", flush = True)
    flush()


def info():
    import torch

    d = {
        "torch": torch.__version__,
        "hip": getattr(torch.version, "hip", None),
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
    }
    try:
        d["arch_list"] = torch.cuda.get_arch_list()
    except Exception as e:  # noqa: BLE001
        d["arch_list_error"] = str(e)
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        d["name"] = torch.cuda.get_device_name(0)
        d["gcnArchName"] = getattr(p, "gcnArchName", None)
        d["total_gib"] = round(p.total_memory / 2**30, 1)
    try:
        import triton

        d["triton"] = triton.__version__
    except Exception as e:  # noqa: BLE001
        d["triton"] = f"unavailable: {type(e).__name__}: {e}"[:300]
    obs["info"] = d
    if not d["cuda_available"]:
        raise RuntimeError("torch.cuda.is_available() is False")
    arch = (d.get("gcnArchName") or "").split(":")[0]
    if arch and d.get("arch_list") is not None and arch not in d["arch_list"]:
        raise RuntimeError(f"{arch} not in torch.cuda.get_arch_list() {d['arch_list']}")
    return d


def rel(a, b):
    return float((a.float().cpu() - b.float().cpu()).abs().max() / (b.float().abs().max() + 1e-6))


def matmul_check():
    import torch

    g = torch.Generator().manual_seed(0)
    a = torch.randn(1024, 1024, generator = g)
    b = torch.randn(1024, 1024, generator = g)
    c = (a.cuda().bfloat16() @ b.cuda().bfloat16())
    torch.cuda.synchronize()
    e = rel(c, a @ b)
    if not e < 2e-2:
        raise AssertionError(f"rel err {e}")
    return {"rel_err": e}


def sdpa_check(backend_name):
    def run():
        import torch
        import torch.nn.functional as F
        from torch.nn.attention import SDPBackend, sdpa_kernel

        g = torch.Generator().manual_seed(1)
        shape = (2, 8, 512, 64)
        q0, k0, v0 = (torch.randn(shape, generator = g) for _ in range(3))
        go = torch.randn(shape, generator = g)

        def go_fn(dev, dtype, backend):
            q, k, v = (t.to(dev, dtype).requires_grad_() for t in (q0, k0, v0))
            ctx = sdpa_kernel([backend]) if backend is not None else None
            if ctx is not None:
                with ctx:
                    o = F.scaled_dot_product_attention(q, k, v, is_causal = True)
            else:
                o = F.scaled_dot_product_attention(q, k, v, is_causal = True)
            o.backward(go.to(dev, dtype))
            return o, q.grad, k.grad, v.grad

        ref = go_fn("cpu", torch.float32, SDPBackend.MATH)
        backend = None if backend_name == "default" else getattr(SDPBackend, backend_name)
        out = go_fn("cuda", torch.bfloat16, backend)
        torch.cuda.synchronize()
        errs = {n: rel(o, r) for n, o, r in zip(("out", "dq", "dk", "dv"), out, ref)}
        bad = {n: e for n, e in errs.items() if not e < 5e-2}
        if bad:
            raise AssertionError(f"rel err {bad}")
        return {"rel_err": errs}

    return run


def after_sdpa_matmul():
    # Same matmul as before; #7992 made every kernel after SDPA fail with hipErrorInvalidValue.
    return matmul_check()


def grouped_mm_check():
    import torch

    if not hasattr(torch, "_grouped_mm"):
        raise RuntimeError("torch._grouped_mm missing")
    g = torch.Generator().manual_seed(2)
    sizes = [64, 32, 96, 64]
    K, N, G = 256, 128, len(sizes)
    A0 = torch.randn(sum(sizes), K, generator = g)
    B0 = torch.randn(G, K, N, generator = g) / K**0.5
    go = torch.randn(sum(sizes), N, generator = g)
    A = A0.cuda().bfloat16().requires_grad_()
    B = B0.cuda().bfloat16().requires_grad_()
    offs = torch.tensor(sizes, device = "cuda").cumsum(0).to(torch.int32)
    out = torch._grouped_mm(A, B, offs = offs)
    out.backward(go.cuda().bfloat16())
    torch.cuda.synchronize()
    Ar, Br = A0.clone().requires_grad_(), B0.clone().requires_grad_()
    parts, s = [], 0
    for i, n in enumerate(sizes):
        parts.append(Ar[s : s + n] @ Br[i])
        s += n
    ref = torch.cat(parts)
    ref.backward(go)
    errs = {"out": rel(out, ref), "dA": rel(A.grad, Ar.grad), "dB": rel(B.grad, Br.grad)}
    bad = {n: e for n, e in errs.items() if not e < 3e-2}
    if bad:
        raise AssertionError(f"rel err {bad}")
    return {"rel_err": errs}


def compile_check():
    import torch

    def f(x):
        return torch.sin(x) * x + torch.cos(x).sum(-1, keepdim = True)

    x = torch.randn(256, 512, device = "cuda")
    y = torch.compile(f)(x)
    torch.cuda.synchronize()
    e = rel(y, f(x))
    if not e < 1e-4:
        raise AssertionError(f"rel err {e}")
    return {"rel_err": e}


def train(mode):
    def run():
        out = os.path.join(args.work, f"train_{mode}.json")
        if os.path.exists(out):
            os.remove(out)
        log = os.path.join(args.work, f"train_{mode}.log")
        with open(log, "w", encoding = "utf-8") as lf:
            p = subprocess.run(
                [sys.executable, "-u", args.train_script, "--mode", mode, "--out", out, "--work", args.work],
                stdout = lf, stderr = subprocess.STDOUT, timeout = args.train_timeout,
            )
        with open(log, encoding = "utf-8", errors = "replace") as lf:
            tail = lf.read()[-6000:]
        r = {}
        if os.path.exists(out):
            with open(out, encoding = "utf-8") as f:
                r = json.load(f)
        r["rc"] = p.returncode
        r["log_tail"] = tail
        if p.returncode != 0 or not r.get("passed"):
            obs["cells"][f"train_{mode}_detail"] = r
            raise RuntimeError(f"train {mode} rc={p.returncode} error={r.get('error')}")
        return r

    return run


cell("info", info)
if obs["cells"]["info"]["ok"]:
    cell("matmul", matmul_check)
    for b in ("default", "FLASH_ATTENTION", "EFFICIENT_ATTENTION"):
        cell(f"sdpa_{b.lower()}", sdpa_check(b))
    cell("matmul_after_sdpa", after_sdpa_matmul)
    cell("grouped_mm", grouped_mm_check)
    cell("compile", compile_check)
    cell("train_16bit", train("16bit"))
    cell("train_4bit", train("4bit"))
flush()
