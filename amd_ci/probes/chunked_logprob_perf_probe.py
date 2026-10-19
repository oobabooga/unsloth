#!/usr/bin/env python3
"""Probe: where the time goes in unsloth-zoo#1803's chunked log-prob head on this GPU.

base state -> unsloth-zoo main (`chunked_hidden_states_selective_log_softmax`), one timing row.
head state -> unsloth-zoo#1803, timed per variant in a fresh process each (the env knobs are read at import):
default, fp32 logits off, torch.compile off, both off; plus raw GEMM timings for the head's shapes and a profiler
table of the slowest kernels for the default variant. Judging is criteria/chunked_logprob_perf.py's job.
"""

from __future__ import annotations

import argparse, json, os, subprocess, sys, time
from pathlib import Path

ZOO = "https://github.com/unslothai/unsloth-zoo"
REF = {"base": "main", "head": "chunked-logprob-recompute"}
VARIANTS = {
    "default": {},
    "fp32_off": {"UNSLOTH_LOGPROB_FP32_LOGITS": "0"},
    "compile_off": {"UNSLOTH_COMPILE_DISABLE": "1"},
    "both_off": {"UNSLOTH_LOGPROB_FP32_LOGITS": "0", "UNSLOTH_COMPILE_DISABLE": "1"},
}
H, V = 2048, 151936


def zoo_checkout(state: str) -> Path:
    root = Path(os.environ["RUNNER_TEMP"]) / f"zoo_perf_{state}"
    if not (root / "unsloth_zoo").is_dir():
        subprocess.run(["git", "clone", "-q", "--depth", "1", "-b", REF[state], ZOO, str(root)], check = True)
    return root


def worker(state: str, zoo: str, profile: bool) -> dict:
    sys.path.insert(0, zoo)
    os.environ["UNSLOTH_IS_PRESENT"] = "1"
    import torch
    import torch.nn.functional as F

    if state == "head":
        from unsloth_zoo.fused_losses.chunked_logprob import unsloth_masked_logprobs as fn
        logps = lambda h, W, t: fn(h, W, t)
    else:
        from unsloth_zoo.rl_replacements import chunked_hidden_states_selective_log_softmax as fn
        logps = lambda h, W, t: fn(h, W, t.clamp(min = 0), 32) * (t != -100)
    g = torch.Generator().manual_seed(3407)
    W = (torch.randn(V, H, generator = g) / H**0.5).cuda().bfloat16()
    out = {}
    for S in (1024, 4096):
        h0 = torch.randn(2, S, H, generator = g).cuda().bfloat16()
        ref_h = (h0.float() + 0.05 * torch.randn(2, S, H, generator = g).cuda()).bfloat16()
        t = torch.randint(0, V, (2, S), generator = g).cuda()
        t[:, : S // 10] = -100
        fwd, full = [], []
        for _ in range(5):
            h = h0.clone().requires_grad_(True)
            torch.cuda.synchronize(); t0 = time.perf_counter()
            with torch.no_grad():
                r = logps(ref_h, W, t).sum(-1)
            torch.cuda.synchronize(); t1 = time.perf_counter()
            pol = logps(h, W, t).sum(-1)
            loss = -F.logsigmoid(0.1 * ((pol[0] - r[0]) - (pol[1] - r[1])))
            loss.backward(); torch.cuda.synchronize(); t2 = time.perf_counter()
            fwd.append(round((t1 - t0) * 1e3, 2)); full.append(round((t2 - t0) * 1e3, 2))
        out[str(S)] = {"nograd_ms": fwd, "total_ms": full}
        if profile and S == 4096:
            from torch.profiler import ProfilerActivity, profile as prof
            with prof(activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
                h = h0.clone().requires_grad_(True)
                logps(h, W, t).sum().backward(); torch.cuda.synchronize()
            rows = sorted(p.key_averages(), key = lambda e: -getattr(e, "self_device_time_total", 0))[:15]
            out["profile_4096"] = [
                {"name": e.key[:90], "self_device_ms": round(getattr(e, "self_device_time_total", 0) / 1e3, 2),
                 "count": e.count, "cpu_ms": round(e.cpu_time_total / 1e3, 2)} for e in rows]
        del h0, ref_h
        torch.cuda.empty_cache()
    if state == "head":
        out["gemm"] = gemm_times(torch)
    return out


def gemm_times(torch) -> dict:
    res = {}
    for rows in (512, 921):
        a = torch.randn(rows, H, device = "cuda", dtype = torch.bfloat16)
        w = torch.randn(V, H, device = "cuda", dtype = torch.bfloat16)
        dz = torch.randn(rows, V, device = "cuda", dtype = torch.bfloat16)
        cases = {
            "logits_bf16": lambda: torch.mm(a, w.t()),
            "logits_bf16_to_fp32": lambda: torch.mm(a, w.t()).float(),
            "dz_w_bf16": lambda: torch.mm(dz, w),
        }
        for name, f in (("logits_out_fp32", lambda: torch.mm(a, w.t(), out_dtype = torch.float32)),
                        ("dz_w_out_fp32", lambda: torch.mm(dz, w, out_dtype = torch.float32))):
            try:
                f(); cases[name] = f
            except Exception as e:  # noqa: BLE001
                res[f"{rows}:{name}"] = f"unsupported: {type(e).__name__}: {e}"[:200]
        for name, f in cases.items():
            for _ in range(3):
                f()
            torch.cuda.synchronize(); t0 = time.perf_counter()
            for _ in range(10):
                f()
            torch.cuda.synchronize()
            res[f"{rows}:{name}"] = round((time.perf_counter() - t0) * 100, 3)
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state")
    ap.add_argument("--checkout", type = Path)
    ap.add_argument("--out", type = Path)
    ap.add_argument("--worker")
    ap.add_argument("--zoo")
    ap.add_argument("--profile", action = "store_true")
    args = ap.parse_args()
    if args.worker:
        args.out.write_text(json.dumps(worker(args.worker, args.zoo, args.profile)), encoding = "utf-8")
        return 0
    state = "head" if "head" in args.state else "base"
    obs: dict = {"state": args.state, "zoo_ref": REF[state], "variants": {}}
    try:
        zoo = str(zoo_checkout(state))
        for name, extra in (VARIANTS.items() if state == "head" else [("default", {})]):
            tmp = Path(os.environ["RUNNER_TEMP"]) / f"perf_{args.state}_{name}.json"
            env = {**os.environ, **extra, "TORCHINDUCTOR_CACHE_DIR": str(tmp) + ".inductor"}
            cmd = [sys.executable, __file__, "--worker", state, "--zoo", zoo, "--out", str(tmp)]
            if name == "default":
                cmd.append("--profile")
            p = subprocess.run(cmd, env = env, capture_output = True, text = True, timeout = 1800)
            obs["variants"][name] = json.loads(tmp.read_text(encoding = "utf-8")) if p.returncode == 0 and tmp.exists() \
                else {"error": f"rc={p.returncode}: {p.stderr[-1500:]}"}
    except Exception as e:  # noqa: BLE001
        obs["error"] = f"{type(e).__name__}: {e}"
    args.out.write_text(json.dumps(obs, indent = 1), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
