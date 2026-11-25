#!/usr/bin/env python3
"""Cell for unsloth-zoo #2767 (W8A16 grouped GEMM for block-FP8 routed experts, CUDA sm89+ only, never ROCm).
Builds a minimal block-FP8 experts module (the attributes forward_moe_backend_fp8 reads; no transformers FP8Experts
needed) with deterministic codes / scales (CPU generator, then .to(device)), runs forward_moe_backend_fp8 on bf16
x (64 tokens, top-2) with x.requires_grad_(), backward on out.float().pow(2).sum(), and records sha256 of out and
x.grad bytes, finiteness, the backend message (spy on _log_moe_fp8_backend_once), device / arch, and on the head
FP8_FUSED_STATS deltas and fp8_experts_as_packed(module, x). Runs: auto twice (A/A), then UNSLOTH_MOE_FP8_FUSED=0.
--device cpu only for a local Python smoke (no evidence). Observes only; JSON to --out."""
import argparse
import hashlib
import importlib.util
import json
import os
import traceback

ap = argparse.ArgumentParser()
ap.add_argument("--out", required = True)
ap.add_argument("--device", default = "cuda")
a = ap.parse_args()
res = {"stage": "start", "device_arg": a.device, "runs": {}}

E, HIDDEN, INTER, TOKENS, TOPK, BLOCK = 8, 256, 256, 64, 2, (128, 128)


def dump():
    with open(a.out, "w", encoding = "utf-8") as f:
        json.dump(res, f, indent = 1, default = str)


def sha(t):
    return hashlib.sha256(t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()


def summary(t):
    f = t.detach().float().cpu()
    return {"sha256": sha(t), "dtype": str(t.dtype), "shape": list(t.shape), "finite": bool(torch.isfinite(f).all()),
            "sum": float(f.sum()), "abs_sum": float(f.abs().sum()), "first": [float(v) for v in f.flatten()[:8]]}


def quantize(R, C, seed):
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(E, R, C, generator = g) * 0.02
    br, bc = BLOCK
    pr, pc = -(-R // br), -(-C // bc)
    wp = w.view(E, pr, br, pc, bc)
    s = wp.abs().amax(dim = (2, 4)).clamp(min = 1e-12) / 448.0
    q = (wp / s[:, :, None, :, None]).to(torch.float8_e4m3fn).view(E, R, C).contiguous()
    return q, s.float().contiguous()


def build(device):
    class Fp8Experts(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.num_experts = E
            self.hidden_dim = HIDDEN
            self.intermediate_dim = INTER
            self.block_size = BLOCK
            self.act_fn = torch.nn.SiLU()
            self.has_gate = True
            self.has_bias = False
            self.is_transposed = False
            gq, gs = quantize(2 * INTER, HIDDEN, 1)
            dq, ds = quantize(HIDDEN, INTER, 2)
            self.gate_up_proj = torch.nn.Parameter(gq.to(device), requires_grad = False)
            self.gate_up_proj_scale_inv = torch.nn.Parameter(gs.to(device), requires_grad = False)
            self.down_proj = torch.nn.Parameter(dq.to(device), requires_grad = False)
            self.down_proj_scale_inv = torch.nn.Parameter(ds.to(device), requires_grad = False)
    return Fp8Experts()


def inputs(device):
    g = torch.Generator().manual_seed(7)
    x = torch.randn(TOKENS, HIDDEN, generator = g).to(torch.bfloat16)
    idx = torch.stack([torch.randperm(E, generator = g)[:TOPK] for _ in range(TOKENS)])
    w = torch.rand(TOKENS, TOPK, generator = g)
    w = w / w.sum(-1, keepdim = True)
    return x.to(device), idx.to(device), w.to(device)


def run(tag, fused_env):
    if fused_env is None:
        os.environ.pop("UNSLOTH_MOE_FP8_FUSED", None)
    else:
        os.environ["UNSLOTH_MOE_FP8_FUSED"] = fused_env
    r = {"fused_env": fused_env, "stage": "start"}
    res["runs"][tag] = r
    try:
        dev = torch.device(a.device)
        module = build(dev)
        x0, idx, w = inputs(dev)
        x = x0.clone().requires_grad_()
        if G is not None:
            r["fused_mode"] = G.fp8_fused_mode()
            r["fused_enabled"] = bool(G.fp8_fused_enabled(dev))
            packed = G.fp8_experts_as_packed(module, x)
            r["as_packed_is_none"] = packed is None
            r["as_packed_type"] = type(packed).__name__
            before = dict(G.FP8_FUSED_STATS)
        msgs = []
        deq = {"unsloth_triton": 0, "vectorized": 0}
        orig = M._log_moe_fp8_backend_once
        orig_u, orig_v = M._dequantize_full_expert_weights_unsloth, M._dequantize_full_expert_weights_vectorized

        def spy_u(*args, **kw):
            # --device cpu (local smoke only): unsloth's Triton dequant cannot launch without a GPU driver.
            got = None if dev.type == "cpu" else orig_u(*args, **kw)
            deq["unsloth_triton"] += got is not None
            return got

        def spy_v(*args, **kw):
            got = orig_v(*args, **kw)
            deq["vectorized"] += got is not None
            return got

        def spy(mod, message):
            msgs.append(message)
            return orig(mod, message)
        M._log_moe_fp8_backend_once = spy
        M._dequantize_full_expert_weights_unsloth, M._dequantize_full_expert_weights_vectorized = spy_u, spy_v
        try:
            out = M.forward_moe_backend_fp8(module, x, idx, w)
            out.float().pow(2).sum().backward()
        finally:
            M._log_moe_fp8_backend_once = orig
            M._dequantize_full_expert_weights_unsloth, M._dequantize_full_expert_weights_vectorized = orig_u, orig_v
        if dev.type == "cuda":
            torch.cuda.synchronize()
        r["backend_messages"] = msgs
        r["dequant_calls"] = deq
        # Head: _stack_view built the Fp8BlockExpertParam views, so a refusal came from fp8_fused_enabled (HIP).
        r["view_cache_len"] = len(module.__dict__.get("_unsloth_fp8_block_views") or {})
        r["logged_attr"] = getattr(module, "_unsloth_logged_fp8_backend", None)
        r["out"] = summary(out)
        r["dx"] = summary(x.grad) if x.grad is not None else None
        if G is not None:
            r["fused_stats_delta"] = {k: G.FP8_FUSED_STATS[k] - before.get(k, 0) for k in G.FP8_FUSED_STATS}
        r["stage"] = "done"
    except Exception as e:  # noqa: BLE001
        r["error"] = f"{type(e).__name__}: {str(e)[:800]}"
        r["traceback"] = traceback.format_exc()[-3000:]
    dump()


try:
    os.environ.setdefault("UNSLOTH_MOE_BACKEND", "grouped_mm")
    import torch
    import unsloth_zoo
    import unsloth_zoo.temporary_patches.moe_utils as MU
    import unsloth_zoo.temporary_patches.moe_utils_fp8 as M
    has_g = importlib.util.find_spec("unsloth_zoo.temporary_patches.moe_fp8_grouped") is not None
    G = None
    if has_g:
        import unsloth_zoo.temporary_patches.moe_fp8_grouped as G
    res["files"] = {"unsloth_zoo": unsloth_zoo.__file__, "moe_utils_fp8": M.__file__,
                    "moe_fp8_grouped": getattr(G, "__file__", None)}
    res["has_fused_module"] = has_g
    props = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
    res["versions"] = {"torch": torch.__version__, "hip": torch.version.hip, "cuda": torch.version.cuda,
                       "cuda_available": torch.cuda.is_available(),
                       "device": getattr(props, "name", None), "arch": str(getattr(props, "gcnArchName", "")),
                       "capability": list(torch.cuda.get_device_capability(0)) if props is not None else None}
    res["env"] = {"UNSLOTH_MOE_BACKEND": os.environ.get("UNSLOTH_MOE_BACKEND")}
    try:
        res["backend"] = MU.select_moe_backend_without_sonic(torch.device(a.device))
    except Exception as e:  # noqa: BLE001
        res["backend"] = f"error: {type(e).__name__}: {e}"
    res["stage"] = "imported"
    dump()
    run("auto", None)
    run("auto_rep", None)
    run("fused0", "0")
    import sys
    res["files"]["unsloth"] = getattr(sys.modules.get("unsloth"), "__file__", None)
    res["stage"] = "done" if all(v.get("stage") == "done" for v in res["runs"].values()) else "run_failed"
except Exception as e:  # noqa: BLE001
    res["error"] = f"{type(e).__name__}: {str(e)[:800]}"
    res["traceback"] = traceback.format_exc()[-3000:]
dump()
