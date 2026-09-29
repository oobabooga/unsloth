#!/usr/bin/env python3
"""Probe: the Windows ROCm torch route the checkout's installer picks, installed and run.

Observes only. Asks the checkout's studio/install_python_stack.py which index and which
torch / torchvision / torchaudio specs it would install for --gfx, installs exactly that
into a fresh venv, then runs a GPU smoke in that venv: device identity, fp16 matmul
error vs fp32 CPU, bf16 torch._grouped_mm vs bmm, torchvision nms, and a tiny training
loop. The criteria module decides.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROUTE = r"""
import json, os, sys
sys.path.insert(0, os.path.join(sys.argv[1], "studio"))
for k in ("UNSLOTH_ROCM_WINDOWS_MIRROR", "UNSLOTH_ROCM_WINDOWS_MULTIARCH_MIRROR", "UNSLOTH_TORCH_INDEX_URL"):
    os.environ.pop(k, None)
import install_python_stack as m
gfx = sys.argv[2]
url = m._windows_rocm_index_url(gfx)
if hasattr(m, "_windows_rocm_torch_pkg_specs_for"):
    specs = m._windows_rocm_torch_pkg_specs_for(url, gfx)
else:
    specs = m._WINDOWS_ROCM_TORCH_PKG_SPECS.get(gfx, ("torch", "torchvision", "torchaudio"))
print("ROUTE_JSON " + json.dumps({"index_url": url, "specs": list(specs)}))
"""

SMOKE = r"""
import json, sys, time, traceback
res = {}
def step(name, fn):
    try:
        res[name] = fn()
    except Exception as e:
        res[name] = {"error": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()[-1500:]}
import torch
res["torch"] = torch.__version__
res["hip"] = getattr(torch.version, "hip", None)
res["cuda_available"] = torch.cuda.is_available()
if res["cuda_available"]:
    p = torch.cuda.get_device_properties(0)
    res["device"] = torch.cuda.get_device_name(0)
    res["arch"] = getattr(p, "gcnArchName", None)
    def matmul():
        torch.manual_seed(0)
        a = torch.randn(512, 512); b = torch.randn(512, 512)
        ref = a @ b
        out = (a.half().cuda() @ b.half().cuda()).float().cpu()
        x = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
        torch.cuda.synchronize(); t = time.perf_counter()
        for _ in range(10): x @ x
        torch.cuda.synchronize(); dt = (time.perf_counter() - t) / 10
        return {"max_abs_err": float((out - ref).abs().max()), "tflops_4096_fp16": 2 * 4096**3 / dt / 1e12}
    step("matmul", matmul)
    def grouped():
        if not hasattr(torch, "_grouped_mm"):
            return {"absent": True}
        torch.manual_seed(0)
        G, M, K, N = 4, 64, 128, 64
        a = torch.randn(G * M, K, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(G, K, N, device="cuda", dtype=torch.bfloat16)
        offs = torch.arange(M, G * M + 1, M, device="cuda", dtype=torch.int32)
        out = torch._grouped_mm(a, b.transpose(-2, -1).contiguous().transpose(-2, -1), offs=offs)
        ref = torch.bmm(a.view(G, M, K), b).reshape(G * M, N)
        torch.cuda.synchronize()
        return {"max_abs_err_vs_bmm": float((out.float() - ref.float()).abs().max())}
    step("grouped_mm", grouped)
    def vision():
        import torchvision
        boxes = torch.tensor([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], dtype=torch.float, device="cuda")
        keep = torchvision.ops.nms(boxes, torch.tensor([0.9, 0.8, 0.7], device="cuda"), 0.5)
        return {"torchvision": torchvision.__version__, "nms_keep": keep.tolist()}
    step("torchvision", vision)
    def audio():
        import torchaudio
        return {"torchaudio": torchaudio.__version__}
    step("torchaudio", audio)
    def train():
        torch.manual_seed(3407)
        model = torch.nn.Sequential(torch.nn.Linear(64, 256), torch.nn.GELU(), torch.nn.Linear(256, 1)).cuda()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
        x = torch.randn(256, 64, device="cuda"); y = x[:, :1] * 2 + 1
        losses = []
        for _ in range(30):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = torch.nn.functional.mse_loss(model(x).float(), y)
            opt.zero_grad(); loss.backward(); opt.step(); losses.append(float(loss))
        return {"first": losses[0], "last": losses[-1]}
    step("train", train)
print("SMOKE_JSON " + json.dumps(res))
"""


def _run(cmd, timeout, **kw):
    t = time.time()
    try:
        p = subprocess.run(cmd, capture_output = True, text = True, encoding = "utf-8",
                           errors = "replace", timeout = timeout, **kw)
        return p.returncode, (p.stdout or "") + (p.stderr or ""), time.time() - t
    except subprocess.TimeoutExpired:
        return -1, "TimeoutExpired", time.time() - t


def _tagged(text: str, tag: str):
    for line in text.splitlines():
        if line.startswith(tag + " "):
            return json.loads(line[len(tag) + 1:])
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--python", default = sys.executable)
    ap.add_argument("--gfx", default = "gfx1151")
    args = ap.parse_args()
    obs: dict = {"state": args.state, "gfx": args.gfx}

    def done() -> int:
        args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
        return 0

    if args.state not in ("base", "head"):
        obs["note"] = "only base and head are installed (the merge state equals head here)"
        return done()

    rc, out, _ = _run([args.python, "-c", ROUTE, args.checkout, args.gfx], 300)
    route = _tagged(out, "ROUTE_JSON")
    obs["route_rc"] = rc
    if route is None:
        obs["error"] = "route resolution failed: " + out[-2000:]
        return done()
    obs.update(route)
    if not route.get("index_url"):
        obs["error"] = "installer resolves no index for this arch"
        return done()

    work = Path(os.environ.get("AMD_CI_WORK") or os.environ.get("RUNNER_TEMP") or ".")
    venv = work / f"route_venv_{args.state}"
    rc, out, _ = _run([args.python, "-m", "venv", str(venv)], 600)
    py = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if rc != 0 or not py.exists():
        obs["error"] = "venv failed: " + out[-2000:]
        return done()
    _run([str(py), "-m", "pip", "install", "-q", "--upgrade", "pip"], 600)
    cmd = [str(py), "-m", "pip", "install", "--no-cache-dir", *route["specs"],
           "--index-url", route["index_url"]]
    obs["install_cmd"] = " ".join(cmd)
    rc, out, dt = _run(cmd, 3600)
    obs["install_rc"] = rc
    obs["install_s"] = round(dt, 1)
    obs["install_tail"] = out[-3000:]
    if rc != 0:
        return done()
    _, frz, _ = _run([str(py), "-m", "pip", "freeze"], 300)
    obs["freeze"] = [l for l in frz.splitlines()
                     if l.lower().startswith(("torch", "amd-torch", "rocm"))]

    rc, out, _ = _run([str(py), "-c", SMOKE], 1800)
    obs["smoke_rc"] = rc
    smoke = _tagged(out, "SMOKE_JSON")
    if smoke is None:
        obs["smoke_error"] = out[-3000:]
    else:
        obs["smoke"] = smoke
    return done()


if __name__ == "__main__":
    sys.exit(main())
