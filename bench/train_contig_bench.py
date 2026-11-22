#!/usr/bin/env python3
"""gfx1151 training: contiguous q/k/v (head) vs Studio's strided native attention (base), measured, not judged.

Stages, each a fresh process:
  micro  attention fwd+bwd, 24 heads x 128, bf16: strided (base) vs k/v-only vs q/k/v contiguous, eager raw SDPA and a
         torch.compile(dynamic=True) projection block, over a token sweep. Errors vs strided (expected exactly 0).
  train  one LoRA run through run_diffusion_training_process (the training worker's entry) from --checkout:
         per-step wall times, losses, grad norms, peak memory, dynamo counters, adapter tensor hash, and how many
         attention calls / traced sites took the contiguous copy (comptime counter: no graph break).
Parent: --plan "cell:arm,arm;cell:arm" runs arms interleaved; results to --out JSON only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path

DATASET = "multimodalart/1920-raider-waite-tarot-public-domain"
MODEL = "black-forest-labs/FLUX.2-klein-base-4B"
CELLS = {"flux2k_1024": 1024, "flux2k_512": 512}
KILL_ENV = "UNSLOTH_DIFFUSION_ROCM_TRAIN_CONTIGUOUS_QKV"


def _write(path, obj):
    Path(path).parent.mkdir(parents = True, exist_ok = True)
    Path(path).write_text(json.dumps(obj, indent = 2, default = str), encoding = "utf-8")


def _read(path):
    try:
        return json.loads(Path(path).read_text(encoding = "utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {"_read_error": f"{type(exc).__name__}: {exc}"}


# ------------------------------------------------------------------------------------------------------------ micro


def stage_micro(args) -> dict:
    import torch
    import torch.nn.functional as F

    B, H, D, dt = 1, 24, 128, torch.bfloat16
    dev = "cuda"
    res = {"torch": torch.__version__, "hip": torch.version.hip,
           "arch": torch.cuda.get_device_properties(0).gcnArchName,
           "aotriton_exp": os.environ.get("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"), "raw": {}, "block": {}}

    def hm(t):
        return t.permute(0, 2, 1, 3).contiguous().permute(0, 2, 1, 3)

    def attn(mode):
        def f(q, k, v):
            if mode == "qkv":
                q, k, v = hm(q), hm(k), hm(v)
            elif mode == "kv":
                k, v = hm(k), hm(v)
            return F.scaled_dot_product_attention(q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3),
                                                  v.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
        return f

    def timed(fn, reps):
        torch.cuda.synchronize()
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1e3)
        return ts

    modes = ("strided", "kv", "qkv")
    for S in [int(s) for s in args.tokens.split(",")]:
        g = torch.Generator(device = dev).manual_seed(S)
        leaves = [torch.randn(B, S, H, D, device = dev, dtype = dt, generator = g) for _ in range(4)]
        reps = 5 if S >= 16384 else 12
        outs, times = {}, {m: [] for m in modes}
        fns = {m: attn(m) for m in modes}
        for m in modes:  # warm + reference
            q, k, v = (t.clone().requires_grad_() for t in leaves[:3])
            o = fns[m](q, k, v)
            o.backward(leaves[3])
            outs[m] = [o.detach(), q.grad, k.grad, v.grad]
        for r in range(3):  # interleaved rounds
            for m in (modes[r % 3:] + modes[:r % 3]):
                q, k, v = (t.clone().requires_grad_() for t in leaves[:3])

                def step():
                    q.grad = k.grad = v.grad = None
                    fns[m](q, k, v).backward(leaves[3])
                times[m] += timed(step, reps)
        row = {m: {"median_ms": round(statistics.median(times[m]), 3),
                   "min_ms": round(min(times[m]), 3), "n": len(times[m])} for m in modes}
        row["bitwise_equal_vs_strided"] = {m: all(torch.equal(a, b) for a, b in zip(outs[m], outs["strided"]))
                                           for m in ("kv", "qkv")}
        res["raw"][str(S)] = row
        print("raw", S, {m: row[m]["median_ms"] for m in modes}, row["bitwise_equal_vs_strided"], flush = True)
        del outs, leaves
        torch.cuda.empty_cache()

    # compiled projection block (what the trainer's regional compile sees: linear -> unflatten -> native permute)
    torch._dynamo.config.cache_size_limit = 64
    C = H * D

    class Block(torch.nn.Module):
        def __init__(self, mode):
            super().__init__()
            self.qkv = torch.nn.Linear(C, 3 * C, bias = False, device = dev, dtype = dt)
            self.o = torch.nn.Linear(C, C, bias = False, device = dev, dtype = dt)
            self.f = attn(mode)

        def forward(self, x):
            q, k, v = self.qkv(x).unflatten(-1, (3, H, D)).unbind(2)
            q, k = q * 1.0001, k * 0.9999  # stand-in for the norm / RoPE producer kernels
            return self.o(self.f(q, k, v).flatten(2))

    torch.manual_seed(0)
    base_blk = Block("strided")
    blocks = {}
    for m in modes:
        b = Block(m)
        b.load_state_dict(base_blk.state_dict())
        blocks[m] = torch.compile(b, dynamic = True)
    from torch._dynamo.utils import counters
    for S in [int(s) for s in args.block_tokens.split(",")]:
        g = torch.Generator(device = dev).manual_seed(S + 1)
        x0 = torch.randn(B, S, C, device = dev, dtype = dt, generator = g)
        times, outs = {m: [] for m in modes}, {}
        for m in modes:
            x = x0.clone().requires_grad_()
            for p in blocks[m].parameters():
                p.grad = None
            y = blocks[m](x)
            y.float().square().mean().backward()
            outs[m] = [y.detach(), x.grad] + [p.grad for p in blocks[m].parameters()]
        for r in range(3):
            for m in (modes[r % 3:] + modes[:r % 3]):
                x = x0.clone().requires_grad_()

                def step():
                    x.grad = None
                    blocks[m](x).float().square().mean().backward()
                times[m] += timed(step, 8)
        row = {m: {"median_ms": round(statistics.median(times[m]), 3), "min_ms": round(min(times[m]), 3)}
               for m in modes}
        row["bitwise_equal_vs_strided"] = {m: all(torch.equal(a, b) for a, b in zip(outs[m], outs["strided"]))
                                           for m in ("kv", "qkv")}
        row["max_abs_vs_strided"] = {m: max(float((a.float() - b.float()).abs().max()) for a, b in
                                            zip(outs[m], outs["strided"])) for m in ("kv", "qkv")}
        res["block"][str(S)] = row
        print("block", S, {m: row[m]["median_ms"] for m in modes}, row["bitwise_equal_vs_strided"], flush = True)
        torch.cuda.empty_cache()
    res["dynamo"] = {"graph_break": sum(counters["graph_break"].values()), "stats": dict(counters["stats"])}
    return res


# ------------------------------------------------------------------------------------------------------------ train


def _studio_env(backend: Path) -> None:
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    sys.path.insert(0, str(backend))
    os.chdir(backend)
    try:
        from utils.rocm_flash_attn import default_triton_amd_env

        default_triton_amd_env()
    except Exception:  # noqa: BLE001
        pass


class _Q:
    def __init__(self, sink = None):
        self.items, self.sink = [], sink

    def put(self, item):
        self.items.append(item)
        if self.sink:
            self.sink(item)

    def empty(self):
        return True

    def get_nowait(self):
        raise RuntimeError("empty")


def _counters():
    try:
        from torch._dynamo.utils import counters

        return {k: dict(v) for k, v in counters.items()}
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


def stage_train(args) -> dict:
    import torch
    from torch._dynamo.comptime import comptime

    backend = Path(args.checkout) / "studio" / "backend"
    res = {"arm": args.arm, "cell": args.cell, "checkout": args.checkout, "kill_env": os.environ.get(KILL_ENV)}
    assert str(backend) == sys.path[0]
    from core.inference import diffusion_attention as da

    res["attention_module"] = da.__file__
    assert da.__file__.startswith(str(backend)), "arm imports another tree"
    # Engagement: _heads_major is the one copy the wrapper makes, in every tree (inference k/v hook on base too).
    eng = {"eager": 0, "traced": 0}
    orig = da._heads_major

    def counted(t):
        if torch.compiler.is_compiling():
            comptime(lambda ctx: eng.__setitem__("traced", eng["traced"] + 1))
        else:
            eng["eager"] += 1
        return orig(t)

    da._heads_major = counted
    installs = []
    if hasattr(da, "install_rocm_contiguous_qkv_training"):
        real_install = da.install_rocm_contiguous_qkv_training

        def rec_install():
            out = real_install()
            installs.append({"cards": list(out), "floors": dict(da._ROCM_CONTIGUOUS_QKV_TRAIN)})
            return out

        da.install_rocm_contiguous_qkv_training = rec_install
    from core.training import diffusion_lora_trainer as dlt

    times, events = [], []
    out_dir = Path(args.work) / "runs" / f"{args.cell}_{args.arm}"
    if out_dir.exists():
        shutil.rmtree(out_dir)

    def sink(ev):
        events.append(ev)
        if ev.get("type") == "progress":
            times.append(time.perf_counter())
            print("progress", ev.get("step"), ev.get("loss"), ev.get("peak_memory_gb"), flush = True)
        elif ev.get("type") in ("warning", "error", "model_load_completed", "complete"):
            print("event", json.dumps(ev, default = str)[:500], flush = True)

    config = {
        "base_model": MODEL, "data_dir": args.data_dir, "output_dir": str(out_dir),
        "resolution": CELLS[args.cell], "train_steps": args.steps, "learning_rate": 2e-4, "lora_rank": 16,
        "lr_scheduler": "constant_with_warmup", "lr_warmup_steps": 5, "base_precision": "bf16",
        "compile_transformer": "auto", "seed": 42, "adapter_name": f"rtc_{args.cell}_{args.arm}", "log_every": 1,
        "hf_token": None,
    }
    res["config"] = config
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    try:
        dlt.run_diffusion_training_process(event_queue = _Q(sink), stop_queue = _Q(), config = config)
    except Exception as exc:  # noqa: BLE001
        res["raised"] = f"{type(exc).__name__}: {exc}"
        res["traceback"] = traceback.format_exc()[-4000:]
    res["wall_s"] = round(time.time() - t0, 1)
    res["max_allocated_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 3)
    res["max_reserved_gb"] = round(torch.cuda.max_memory_reserved() / 1e9, 3)
    prog = [e for e in events if e.get("type") == "progress"]
    res["steps"] = [e.get("step") for e in prog]
    res["losses"] = [e.get("loss") for e in prog]
    res["grad_norms"] = [e.get("grad_norm") for e in prog]
    res["trainer_peak_gb"] = max([(e.get("peak_memory_gb") or 0) for e in prog] or [0])
    res["samples_per_second_last"] = prog[-1].get("samples_per_second") if prog else None
    # step k's wall = gap between the progress events of k-1 and k (the loop syncs on loss.item() every step)
    res["step_s"] = [round(b - a, 4) for a, b in zip(times, times[1:])]  # index 0 = step 2
    tail = res["step_s"][1:]  # from step 3
    res["median_step_s_from3"] = round(statistics.median(tail), 4) if tail else None
    res["engaged"] = dict(eng)
    res["installs"] = installs
    res["dynamo"] = _counters()
    res["warnings"] = [str(e.get("message"))[:400] for e in events if e.get("type") == "warning"]
    res["errors"] = [e.get("message") or e.get("error") for e in events if e.get("type") == "error"]
    loaded = [e for e in events if e.get("type") == "model_load_completed"]
    if loaded:
        res["compiled"] = loaded[-1].get("compiled")
    done = [e for e in events if e.get("type") == "complete"]
    if done and done[-1].get("lora_path") and os.path.isfile(done[-1]["lora_path"]):
        from safetensors.torch import load_file

        sd = load_file(done[-1]["lora_path"])
        h = hashlib.sha256()
        for k in sorted(sd):
            h.update(k.encode())
            h.update(sd[k].float().cpu().numpy().tobytes())
        res["lora_sha256"] = h.hexdigest()
        res["lora_tensors"] = len(sd)
    return res


STAGES = {"micro": stage_micro, "train": stage_train}


def fetch_dataset(dest: Path, n: int) -> dict:
    if (dest / "done").is_file():
        return _read(dest / "done")
    from huggingface_hub import hf_hub_download

    meta = hf_hub_download(DATASET, "output_file.jsonl", repo_type = "dataset", token = False)
    rows = [json.loads(line) for line in open(meta, encoding = "utf-8") if line.strip()]
    rows = sorted(rows, key = lambda r: r["file_name"])[:n]
    dest.mkdir(parents = True, exist_ok = True)
    for r in rows:
        p = hf_hub_download(DATASET, "train/" + r["file_name"], repo_type = "dataset", token = False)
        stem = Path(r["file_name"]).stem
        shutil.copy(p, dest / (stem + Path(r["file_name"]).suffix.lower()))
        (dest / (stem + ".txt")).write_text(r["text"], encoding = "utf-8")
    info = {"dataset": DATASET, "images": len(rows)}
    _write(dest / "done", info)
    return info


def child(args) -> int:
    if args.stage == "train":
        _studio_env(Path(args.checkout) / "studio" / "backend")
    else:
        os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")
    try:
        res = STAGES[args.stage](args)
    except Exception as exc:  # noqa: BLE001
        res = {"stage_error": f"{type(exc).__name__}: {str(exc)[:1500]}", "traceback": traceback.format_exc()[-5000:]}
    _write(args.stage_out, res)
    return 0


def run_child(args, stage, stage_out: Path, extra, env, timeout_s) -> dict:
    cmd = [sys.executable, "-u", __file__, "--child", "--stage", stage, "--stage-out", str(stage_out),
           "--work", args.work, "--data-dir", args.data_dir, "--steps", str(args.steps),
           "--tokens", args.tokens, "--block-tokens", args.block_tokens, *extra]
    log = stage_out.with_suffix(".log")
    t0 = time.time()
    with open(log, "wb") as fh:
        try:
            rc = subprocess.run(cmd, stdout = fh, stderr = subprocess.STDOUT, timeout = timeout_s, env = env).returncode
        except subprocess.TimeoutExpired:
            rc = "timeout"
    res = _read(stage_out) if stage_out.is_file() else {"missing_output": True}
    res["_rc"], res["_wall_s"] = rc, round(time.time() - t0, 1)
    if rc != 0 or "stage_error" in res or "missing_output" in res or res.get("raised"):
        res["_log_tail"] = log.read_text(encoding = "utf-8", errors = "replace")[-3000:]
    print(f"{stage} {extra} rc={rc} {res['_wall_s']}s median_from3={res.get('median_step_s_from3')} "
          f"engaged={res.get('engaged')} loss_last={(res.get('losses') or [None])[-1]}", flush = True)
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--child", action = "store_true")
    ap.add_argument("--stage", default = "")
    ap.add_argument("--stage-out", default = "")
    ap.add_argument("--arm", default = "")
    ap.add_argument("--cell", default = "")
    ap.add_argument("--checkout", default = "")
    ap.add_argument("--base", default = "")
    ap.add_argument("--head", default = "")
    ap.add_argument("--work", default = "")
    ap.add_argument("--data-dir", default = "")
    ap.add_argument("--steps", type = int, default = 20)
    ap.add_argument("--images", type = int, default = 8)
    ap.add_argument("--tokens", default = "1024,1536,2048,2560,3072,4096,4608,8192")
    ap.add_argument("--block-tokens", default = "1024,1536,2048,3072,4096,4608")
    ap.add_argument("--micro", action = "store_true")
    ap.add_argument("--plan", default = "", help = "cell:arm,arm,...;cell:... (arms base|head|kill, run in order)")
    ap.add_argument("--out", default = "")
    args = ap.parse_args()
    if args.child:
        return child(args)
    work = Path(args.work)
    out_dir = Path(args.out).parent / "stages"
    out_dir.mkdir(parents = True, exist_ok = True)
    report = {"plan": args.plan, "steps": args.steps, "model": MODEL, "micro": None, "cells": {}}
    if args.micro:
        report["micro"] = run_child(args, "micro", out_dir / "micro.json", [], None, 3600)
        _write(args.out, report)
    if args.plan:
        if not args.data_dir:
            args.data_dir = str(work / "data")
        report["dataset"] = fetch_dataset(Path(args.data_dir), args.images)
        for spec in args.plan.split(";"):
            cell, arms = spec.split(":")
            seen = {}
            for arm in arms.split(","):
                seen[arm] = seen.get(arm, 0) + 1
                tag = f"{arm}{seen[arm]}"
                env = dict(os.environ)
                env.pop(KILL_ENV, None)
                if arm == "kill":
                    env[KILL_ENV] = "0"
                checkout = args.base if arm == "base" else args.head
                res = run_child(args, "train", out_dir / f"{cell}_{tag}.json",
                                ["--arm", tag, "--cell", cell, "--checkout", checkout], env, 3600)
                report["cells"].setdefault(cell, {})[tag] = res
                _write(args.out, report)
    _write(args.out, report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
