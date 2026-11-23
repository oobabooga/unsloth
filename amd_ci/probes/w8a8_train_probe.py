#!/usr/bin/env python3
"""Probe: the frozen int8 W8A8 training base (#18782), its backward weight cache (#18860) and block swap over a
frozen low-precision base (#18869) on this GPU, through Studio's own trainer (run_diffusion_training_process).

Observes only (judged by criteria/w8a8_train.py). Results go to --out as JSON, never stdout. Each stage runs in its
own subprocess (fresh process per arm, clean device memory, contained crash):

  host     torch / HIP / arch facts, train_precision_modes(), the start route's preflight
           (training_precision_preflight_error, block_swap_start_error) and the trainer's _resolve_base_precision for
           every requested arm, env on and off
  kernels  torch._int_mm and torch._scaled_mm (fp8) availability; FrozenInt8Linear forward / grad_x vs the dense
           bf16 layer on FLUX.2-klein shapes, backward cache on vs off bit-equality, fullgraph compile, and
           int8 vs bf16 GEMM timing
  train    one arm per process. "Requested" arms use the shipped gates as they are. "Forced" arms (name starts F_)
           make only the ROCm precision gates (_resolve_base_precision / training_precision_preflight_error /
           train_precision_modes) and, for swap arms, the integrated-GPU swap gate answer as on NVIDIA, so the code
           those PRs added actually runs here; every other ROCm branch keeps its real answer.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import shutil
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path

DATASET = "multimodalart/1920-raider-waite-tarot-public-domain"
BASE_MODEL = "black-forest-labs/FLUX.2-klein-base-4B"

# name: (base_precision, compile_transformer, env, block_swap, forced)
W8 = {"UNSLOTH_DIFFUSION_INT8_TRAIN": "w8a8"}
ARMS = {
    # shipped gates, as a user gets them
    "bf16_c_a": ("bf16", "auto", {}, "", False),
    "R_int8": ("int8", "auto", {}, "", False),
    "R_w8a8": ("int8", "auto", W8, "", False),
    "R_fp8": ("fp8", "auto", {}, "", False),
    "R_w8a8_swap4": ("int8", "auto", W8, "4", False),
    "R_bf16_swap4": ("bf16", "auto", {}, "4", False),
    # gates forced open: does the merged code itself work on gfx1151?
    "F_w8a8_c": ("int8", "auto", W8, "", True),
    "F_int8wo_e": ("int8", "off", {}, "", True),
    "F_w8a8_e": ("int8", "off", W8, "", True),
    "F_w8a8_nocache_c": ("int8", "auto", {**W8, "UNSLOTH_DIFFUSION_INT8_BWD_CACHE": "0"}, "", True),
    "F_w8a8_swap4_c": ("int8", "auto", W8, "4", True),
    "F_fp8_c": ("fp8", "auto", {}, "", True),
    "F_bf16_swap4_c": ("bf16", "auto", {}, "4", True),
    "F_w8a8_swap4_lowpoff_c": ("int8", "auto", {**W8, "UNSLOTH_DIT_BLOCK_SWAP_LOWP": "0"}, "4", True),
    "bf16_c_b": ("bf16", "auto", {}, "", False),
    "bf16_e": ("bf16", "off", {}, "", False),
}
HEAD_ORDER = ["bf16_c_a", "R_int8", "R_w8a8", "R_fp8", "R_w8a8_swap4", "R_bf16_swap4", "F_w8a8_c", "F_int8wo_e",
              "F_w8a8_e", "bf16_c_b", "F_w8a8_nocache_c", "F_w8a8_swap4_c", "F_bf16_swap4_c",
              "F_w8a8_swap4_lowpoff_c", "F_fp8_c", "bf16_e"]
BASE_ORDER = ["bf16_c_a", "F_w8a8_swap4_c", "F_bf16_swap4_c", "F_w8a8_c"]
GATED_CALLERS = {"_resolve_base_precision", "training_precision_preflight_error", "train_precision_modes"}


def _write(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_text(json.dumps(obj, indent = 2, default = str), encoding = "utf-8")


def _read(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding = "utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {"_read_error": f"{type(exc).__name__}: {exc}"}


def _studio_env(backend: Path) -> None:
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    # Studio's main.py sets this before any worker starts; unsloth_zoo refuses to import without it.
    os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")
    sys.path.insert(0, str(backend))
    os.chdir(backend)
    try:
        from utils.rocm_flash_attn import default_triton_amd_env

        default_triton_amd_env()
    except Exception:  # noqa: BLE001
        pass


def _err(exc) -> str:
    return f"{type(exc).__name__}: {str(exc)[:600]}"


def _force_gates(swap: bool) -> dict:
    """Make only the ROCm precision gates (and, for swap, the integrated-GPU swap gate) answer False."""
    import core._torchao_stub as stub
    from core.training import diffusion_dit_trainer as ddt
    from core.training import diffusion_train_common as dtc

    real = stub.torch_is_rocm
    seen: dict = {}

    def fake():
        name = sys._getframe(1).f_code.co_name
        if name in GATED_CALLERS:
            seen[name] = seen.get(name, 0) + 1
            return False
        return real()

    for mod in (ddt, dtc):
        if getattr(mod, "torch_is_rocm", None) is real:
            mod.torch_is_rocm = fake
    if swap:
        real_int = ddt._device_is_integrated

        def not_integrated(device):
            if sys._getframe(1).f_code.co_name == "_auto_block_swap_wanted":
                return real_int(device)
            seen["_device_is_integrated"] = seen.get("_device_is_integrated", 0) + 1
            return False

        ddt._device_is_integrated = not_integrated
    return seen


# ----------------------------------------------------------------------------------------------------------- stages


def stage_host(args) -> dict:
    import torch

    res: dict = {"torch": torch.__version__, "hip": getattr(torch.version, "hip", None),
                 "cuda_available": torch.cuda.is_available()}
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        res.update(device = torch.cuda.get_device_name(0), arch = getattr(p, "gcnArchName", None),
                   capability = list(torch.cuda.get_device_capability(0)),
                   total_gib = round(p.total_memory / (1 << 30), 2))
    from core.training import diffusion_block_swap as dbs
    from core.training import diffusion_dit_trainer as ddt
    from core.training import diffusion_train_common as dtc

    import importlib.metadata as md

    for pkg in ("unsloth_zoo", "unsloth", "torchao", "diffusers", "peft", "triton"):
        try:
            res.setdefault("versions", {})[pkg] = md.version(pkg)
        except Exception as exc:  # noqa: BLE001
            res.setdefault("versions", {})[pkg] = _err(exc)
    try:
        import unsloth_zoo.block_swap  # noqa: F401

        res["zoo_block_swap_import"] = "ok"
    except Exception as exc:  # noqa: BLE001
        res["zoo_block_swap_import"] = _err(exc)
    try:
        res["train_precision_modes"] = dtc.train_precision_modes()
    except Exception as exc:  # noqa: BLE001
        res["train_precision_modes"] = _err(exc)
    try:
        from core.inference.host_pinning import device_is_integrated

        res["device_is_integrated"] = device_is_integrated("cuda")
    except Exception as exc:  # noqa: BLE001
        res["device_is_integrated"] = _err(exc)
    gates: dict = {}
    for name, (prec, comp, env, swap, forced) in ARMS.items():
        if forced:
            continue
        saved = {k: os.environ.get(k) for k in ("UNSLOTH_DIFFUSION_INT8_TRAIN", "UNSLOTH_DIFFUSION_INT8_BWD_CACHE")}
        for k in saved:
            os.environ.pop(k, None)
        os.environ.update(env)
        row: dict = {}
        try:
            cfg = dtc._config_from_dict({"base_model": BASE_MODEL, "data_dir": "d", "output_dir": "o",
                                         "resolution": 512, "base_precision": prec, "block_swap": swap,
                                         "compile_transformer": comp}).normalized()
            row["resolved_family"] = cfg.resolved_family
            row["preflight"] = dtc.training_precision_preflight_error(cfg.resolved_family, prec)
            row["block_swap_start_error"] = dbs.block_swap_start_error(cfg)
            try:
                row["resolve_base_precision"] = ddt._resolve_base_precision(cfg, ddt._SPECS[cfg.resolved_family],
                                                                            "cuda")
            except Exception as exc:  # noqa: BLE001
                row["resolve_base_precision"] = "raised " + _err(exc)
        except Exception as exc:  # noqa: BLE001
            row["error"] = _err(exc)
        for k, v in saved.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v
        gates[name] = row
    res["gates"] = gates
    return res


def _sync(torch):
    torch.cuda.synchronize()


def _timeit(torch, fn, n = 20) -> float:
    for _ in range(3):
        fn()
    _sync(torch)
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        _sync(torch)
        ts.append(time.perf_counter() - t)
    return statistics.median(ts) * 1e3


def stage_kernels(args) -> dict:
    import torch
    import torch.nn as nn

    dev = "cuda"
    res: dict = {}
    try:
        a = torch.ones((32, 32), dtype = torch.int8, device = dev)
        r = torch._int_mm(a, a.t())
        _sync(torch)
        res["int_mm_probe"] = {"ok": True, "value": int(r[0, 0].item())}
    except Exception as exc:  # noqa: BLE001
        res["int_mm_probe"] = {"ok": False, "error": _err(exc)}
    try:
        x = torch.randn(64, 64, device = dev).to(torch.float8_e4m3fn)
        w = torch.randn(64, 64, device = dev).to(torch.float8_e4m3fn)
        one = torch.ones((), device = dev)
        y = torch._scaled_mm(x, w.t(), scale_a = one, scale_b = one, out_dtype = torch.bfloat16)
        _sync(torch)
        res["scaled_mm_fp8_e4m3fn"] = {"ok": bool(torch.isfinite(y).all().item())}
    except Exception as exc:  # noqa: BLE001
        res["scaled_mm_fp8_e4m3fn"] = {"ok": False, "error": _err(exc)}
    if not res["int_mm_probe"]["ok"]:
        return res
    from core.training import diffusion_frozen_int8 as dfi

    torch.manual_seed(0)
    shapes = [(4608, 3072, 3072), (4608, 3072, 9216), (4608, 9216, 3072), (512, 3072, 3072), (8, 3072, 3072)]
    rows = []
    for m, k, n in shapes:
        row: dict = {"M": m, "K": k, "N": n}
        try:
            lin = nn.Linear(k, n, bias = True, device = dev, dtype = torch.bfloat16).requires_grad_(False)
            nn.init.normal_(lin.weight, std = k ** -0.5)
            x = (torch.randn(m, k, device = dev, dtype = torch.bfloat16)).requires_grad_(True)
            ref = (x.double() @ lin.weight.double().t() + lin.bias.double())
            q = dfi.FrozenInt8Linear(lin)
            y = q(x)
            gy = torch.randn_like(y)
            y.backward(gy)
            gx_inline = x.grad.clone()
            x.grad = None
            ref_gx = gy.double() @ lin.weight.double()
            dense = (x @ lin.weight.t() + lin.bias)

            def rel(a, b):
                return float((a.double() - b).norm() / b.norm().clamp_min(1e-30))

            def cos(a, b):
                return float(torch.nn.functional.cosine_similarity(a.double().flatten(), b.flatten(), dim = 0))

            row.update(fwd_rel = rel(y, ref), fwd_cos = cos(y, ref), dense_bf16_fwd_rel = rel(dense, ref),
                       gx_rel = rel(gx_inline, ref_gx), gx_cos = cos(gx_inline, ref_gx),
                       finite = bool(torch.isfinite(y).all() and torch.isfinite(gx_inline).all()))
            # backward cache on vs inline copy: must be bit-identical (int32 accumulation is exact)
            bw = dfi.BACKWARD_WEIGHTS
            bw.clear()
            bw.active, bw.budget = True, 1 << 34
            for _ in range(2):
                y2 = q(x)
                y2.backward(gy)
            gx_cached = x.grad.clone()
            x.grad = None
            row["cache"] = {**bw.stats(), "grad_equal_inline": bool(torch.equal(gx_cached / 2, gx_inline))
                            if m > 16 else None}
            bw.clear()
            bw.active = False
            if m > 16:
                xq = torch.randint(-127, 127, (m, k), dtype = torch.int8, device = dev)
                wq = torch.randint(-127, 127, (n, k), dtype = torch.int8, device = dev)
                xb, wb = xq.to(torch.bfloat16), wq.to(torch.bfloat16)
                row["ms_int_mm"] = round(_timeit(torch, lambda: torch._int_mm(xq, wq.t())), 4)
                row["ms_bf16_mm"] = round(_timeit(torch, lambda: xb @ wb.t()), 4)
                row["ms_w8a8_fwd_module"] = round(_timeit(torch, lambda: q(x.detach())), 4)
                row["ms_bf16_fwd_module"] = round(_timeit(torch, lambda: lin(x.detach())), 4)
            del lin, q, x, y, gy
        except Exception as exc:  # noqa: BLE001
            row["error"] = _err(exc)
            row["traceback"] = traceback.format_exc()[-1500:]
        rows.append(row)
        torch.cuda.empty_cache()
    res["frozen_int8_linear"] = rows
    # fullgraph compile of a block of FrozenInt8Linear layers, forward + backward
    try:
        import torch._dynamo

        torch._dynamo.reset()
        layers = nn.Sequential(nn.Linear(3072, 3072, bias = False), nn.GELU(), nn.Linear(3072, 3072, bias = False))
        layers = layers.to(dev, torch.bfloat16).requires_grad_(False)
        layers[0] = dfi.FrozenInt8Linear(layers[0])
        layers[2] = dfi.FrozenInt8Linear(layers[2])
        x = torch.randn(1024, 3072, device = dev, dtype = torch.bfloat16, requires_grad = True)
        eager_y = layers(x)
        eager_y.float().sum().backward()
        eg = x.grad.clone()
        x.grad = None
        comp = torch.compile(layers, fullgraph = True)
        cy = comp(x)
        cy.float().sum().backward()
        from torch._dynamo.utils import counters

        res["compile_fullgraph"] = {"ok": True, "graph_breaks": sum(counters["graph_break"].values()),
                                    "fwd_rel_vs_eager": float((cy.float() - eager_y.float()).norm()
                                                              / eager_y.float().norm()),
                                    "gx_rel_vs_eager": float((x.grad.float() - eg.float()).norm()
                                                             / eg.float().norm())}
    except Exception as exc:  # noqa: BLE001
        res["compile_fullgraph"] = {"ok": False, "error": _err(exc), "traceback": traceback.format_exc()[-2000:]}
    return res


class _Q:
    def __init__(self, sink = None):
        self.items: list = []
        self.sink = sink

    def put(self, item):
        self.items.append(item)
        if self.sink:
            self.sink(item)

    def empty(self):
        return True

    def get_nowait(self):
        raise RuntimeError("empty")


class _LogGrab(logging.Handler):
    KEYS = ("W8A8", "w8a8", "int8", "fp8", "frozen backward", "block_swap", "block swap", "swap", "compile",
            "graph", "recompil", "fallback", "Fp8", "Int8")

    def __init__(self):
        super().__init__(level = logging.INFO)
        self.lines: list = []

    def emit(self, record):
        try:
            msg = record.getMessage()
        except Exception:  # noqa: BLE001
            return
        if any(k in msg for k in self.KEYS) and len(self.lines) < 200:
            self.lines.append(f"{record.name}: {msg[:500]}")


def stage_train(args) -> dict:
    prec, comp, env, swap, forced = ARMS[args.arm]
    import torch

    grab = _LogGrab()
    logging.getLogger().addHandler(grab)
    logging.getLogger().setLevel(logging.INFO)
    from core.training import diffusion_lora_trainer as dlt

    res: dict = {"arm": args.arm, "base_precision": prec, "compile_transformer": comp, "env": env,
                 "block_swap": swap, "forced": forced}
    if forced:
        res["forced_gate_calls"] = _force_gates(bool(swap))
    events_path = Path(args.stage_out).with_suffix(".events.jsonl")
    ev_fh = open(events_path, "w", encoding = "utf-8")
    stamps: dict = {}

    def sink(ev):
        t = time.time()
        ev_fh.write(json.dumps({**ev, "_t": t}, default = str) + "\n")
        ev_fh.flush()
        if ev.get("type") == "progress" and ev.get("step") is not None:
            stamps.setdefault(int(ev["step"]), t)
        if ev.get("type") not in ("preparing", "download_progress", "progress"):
            print("event", json.dumps(ev, default = str)[:600], flush = True)

    out_dir = Path(args.work) / "runs" / f"{args.state}_{args.arm}"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    config = {
        "base_model": BASE_MODEL, "data_dir": args.data_dir, "output_dir": str(out_dir), "resolution": 512,
        "train_steps": args.steps, "learning_rate": 1e-4, "lora_rank": 16, "lr_scheduler": "constant_with_warmup",
        "lr_warmup_steps": 3, "base_precision": prec, "compile_transformer": comp, "seed": 42,
        "adapter_name": f"amdci_w8a8_{args.state}_{args.arm}", "log_every": 1, "hf_token": None,
    }
    if swap:
        config["block_swap"] = swap
    res["config"] = config
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    q_ev = _Q(sink)
    try:
        dlt.run_diffusion_training_process(event_queue = q_ev, stop_queue = _Q(), config = config)
    except Exception as exc:  # noqa: BLE001
        res["raised"] = _err(exc)
        res["traceback"] = traceback.format_exc()[-4000:]
    ev_fh.close()
    res["wall_s"] = round(time.time() - t0, 1)
    res["max_allocated_gib"] = round(torch.cuda.max_memory_allocated() / (1 << 30), 3)
    res["max_reserved_gib"] = round(torch.cuda.max_memory_reserved() / (1 << 30), 3)
    events = q_ev.items
    prog = [e for e in events if e.get("type") == "progress"]
    res["losses"] = [e.get("loss") for e in prog]
    res["steps_seen"] = len(prog)
    if prog:
        res["trainer_peak_memory_gb"] = max((e.get("peak_memory_gb") or 0) for e in prog)
        res["samples_per_second_last"] = prog[-1].get("samples_per_second")
    steps = sorted(stamps)
    dts = {s: stamps[s] - stamps[s - 1] for s in steps if s - 1 in stamps}
    res["step_s"] = {s: round(v, 4) for s, v in dts.items()}
    tail = [v for s, v in dts.items() if s >= 3]
    res["s_per_step_median_from3"] = round(statistics.median(tail), 4) if tail else None
    res["errors"] = [str(e.get("message") or e.get("error"))[:1500] for e in events if e.get("type") == "error"]
    res["warnings"] = [str(e.get("message"))[:600] for e in events if e.get("type") == "warning"]
    loaded = [e for e in events if e.get("type") == "model_load_completed"]
    if loaded:
        res["model_load_completed"] = {k: v for k, v in loaded[-1].items() if k not in ("type",)}
    res["block_swap_events"] = [e for e in events if e.get("type") == "block_swap"]
    res["fast_lora_stats"] = [e for e in events if e.get("type") == "fast_lora_stats"]
    done = [e for e in events if e.get("type") == "complete"]
    res["complete"] = bool(done)
    if done:
        res["complete_info"] = {k: done[-1].get(k) for k in ("steps_run", "wall_seconds", "family", "stopped")}
    try:
        from torch._dynamo.utils import counters

        res["dynamo"] = {"graph_breaks": sum(counters["graph_break"].values()),
                         "graph_break_reasons": [str(k)[:300] for k in list(counters["graph_break"])[:10]],
                         "unique_graphs": counters["stats"].get("unique_graphs"),
                         "frames_ok": counters["frames"].get("ok"), "frames_total": counters["frames"].get("total"),
                         "recompiles": sum(counters["recompiles"].values()) if "recompiles" in counters else None}
    except Exception as exc:  # noqa: BLE001
        res["dynamo"] = _err(exc)
    try:
        from core.training import diffusion_frozen_int8 as dfi

        res["int8_bwd_stats_after"] = dfi.BACKWARD_WEIGHTS.stats()
    except Exception as exc:  # noqa: BLE001
        res["int8_bwd_stats_after"] = _err(exc)
    try:
        from core.training import diffusion_block_swap as dbs

        res["swap_stats"] = dbs.swap_stats()
    except Exception as exc:  # noqa: BLE001
        res["swap_stats"] = _err(exc)
    res["log_lines"] = grab.lines
    return res


STAGES = {"host": stage_host, "kernels": stage_kernels, "train": stage_train}


def child_main(args) -> int:
    _studio_env(Path(args.checkout) / "studio" / "backend")
    out = Path(args.stage_out)
    try:
        res = STAGES[args.stage](args)
    except Exception as exc:  # noqa: BLE001
        res = {"stage_error": _err(exc), "traceback": traceback.format_exc()[-5000:]}
    _write(out, res)
    return 0


# ----------------------------------------------------------------------------------------------------------- parent


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


def _count(text: str, needle: str) -> int:
    return text.count(needle)


def run_child(args, stage: str, stage_out: Path, timeout_s: int, extra: list, env_add = None) -> dict:
    cmd = [sys.executable, "-u", __file__, "--child", "--stage", stage, "--state", args.state, "--checkout",
           args.checkout, "--out", args.out, "--stage-out", str(stage_out), "--work", args.work, "--data-dir",
           args.data_dir, "--steps", str(args.steps), *extra]
    env = dict(os.environ)
    for k in ("UNSLOTH_DIFFUSION_INT8_TRAIN", "UNSLOTH_DIFFUSION_INT8_BWD_CACHE", "UNSLOTH_DIT_BLOCK_SWAP",
              "UNSLOTH_DIT_BLOCK_SWAP_LOWP"):
        env.pop(k, None)
    env.update(env_add or {})
    env["TORCH_LOGS"] = "recompiles,graph_breaks"
    log = stage_out.with_suffix(".log")
    t0 = time.time()
    with open(log, "wb") as fh:
        try:
            rc = subprocess.run(cmd, stdout = fh, stderr = subprocess.STDOUT, timeout = timeout_s, env = env).returncode
        except subprocess.TimeoutExpired:
            rc = "timeout"
    res = _read(stage_out) if stage_out.is_file() else {"missing_output": True}
    res["_rc"] = rc
    res["_wall_s"] = round(time.time() - t0, 1)
    try:
        text = log.read_text(encoding = "utf-8", errors = "replace")
        res["_log_recompile_lines"] = _count(text, "Recompiling function")
        res["_log_graph_break_lines"] = _count(text, "Graph break")
        if rc != 0 or "stage_error" in res or "missing_output" in res or res.get("errors") or res.get("raised"):
            res["_log_tail"] = text[-3000:]
    except Exception:  # noqa: BLE001
        pass
    print(f"[{args.state}] {stage} {extra} rc={rc} {res['_wall_s']}s", flush = True)
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True)
    ap.add_argument("--steps", type = int, default = 20)
    ap.add_argument("--images", type = int, default = 8)
    ap.add_argument("--budget-min", type = float, default = 200.0, help = "whole-probe budget across all states")
    ap.add_argument("--work", default = "")
    ap.add_argument("--data-dir", default = "")
    ap.add_argument("--child", action = "store_true")
    ap.add_argument("--stage", default = "")
    ap.add_argument("--stage-out", default = "")
    ap.add_argument("--arm", default = "")
    args = ap.parse_args()
    if args.child:
        return child_main(args)

    out = Path(args.out).resolve()
    args.out = str(out)
    args.checkout = str(Path(args.checkout).resolve())
    work = Path(args.work or (Path(os.environ.get("AMD_CI_WORK", out.parent)) / "w8a8_probe")).resolve()
    args.work = str(work)
    work.mkdir(parents = True, exist_ok = True)
    os.environ.setdefault("HF_HOME", str(work / "hf"))
    os.environ.setdefault("UNSLOTH_DISABLE_AUTO_UPDATES", "1")
    os.environ.pop("HF_TOKEN", None)
    stamp = work / "budget_start"
    if not stamp.is_file():
        stamp.write_text(str(time.time()), encoding = "utf-8")
    deadline = float(stamp.read_text(encoding = "utf-8")) + args.budget_min * 60

    order = HEAD_ORDER if args.state == "head" else BASE_ORDER
    obs: dict = {"state": args.state, "checkout": args.checkout, "steps": args.steps, "base_model": BASE_MODEL,
                 "arms": {}, "skipped_arms": [a for a in HEAD_ORDER if a not in order]}
    stage_dir = out.parent / f"stages_{args.state}"
    stage_dir.mkdir(parents = True, exist_ok = True)
    try:
        obs["dataset"] = fetch_dataset(work / "data", args.images)
    except Exception as exc:  # noqa: BLE001
        obs["dataset_error"] = _err(exc)
    args.data_dir = str(work / "data")
    _write(out, obs)
    obs["host"] = run_child(args, "host", stage_dir / "host.json", 900, [])
    _write(out, obs)
    obs["kernels"] = run_child(args, "kernels", stage_dir / "kernels.json", 1200, [])
    _write(out, obs)
    for arm in order:
        left = deadline - time.time()
        if left < 8 * 60:
            obs["arms"][arm] = {"skipped": f"probe budget exhausted ({left / 60:.1f} min left)"}
            continue
        env = ARMS[arm][2]
        obs["arms"][arm] = run_child(args, "train", stage_dir / f"train_{arm}.json", int(min(left - 120, 35 * 60)),
                                     ["--arm", arm], env_add = env)
        _write(out, obs)
    obs["finished"] = True
    _write(out, obs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
