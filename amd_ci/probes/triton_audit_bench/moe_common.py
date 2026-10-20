"""Shared MoE helpers for b_moe_*.py / b_mxfp4.py (routing, references)."""
import torch

FP4_VALUES = [+0.0, +0.5, +1.0, +1.5, +2.0, +3.0, +4.0, +6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]

# name -> (hidden, experts, top_k, intermediate)
MODELS = {
    "qwen3_30b_a3b": (2048, 128, 8, 768),
    "gpt_oss_20b": (2880, 32, 4, 2880),
}


def route(T, E, k, device="cuda", seed=0):
    """Realistic imbalance: per-token top-k of softmax(randn logits + per-expert bias)."""
    g = torch.Generator(device=device).manual_seed(seed)
    bias = torch.randn(E, device=device, generator=g) * 1.0  # popular / unpopular experts
    logits = torch.randn(T, E, device=device, generator=g) + bias
    probs = logits.softmax(-1)
    w, idx = probs.topk(k, dim=-1)
    w = w / w.sum(-1, keepdim=True)
    return idx, w


def sort_routing(idx, E):
    """-> sorted_indices (T*k), token_indices, counts (E,) int32, offsets (cumsum int32)."""
    flat = idx.reshape(-1)
    counts = torch.bincount(flat, minlength=E).to(torch.int32)
    sorted_indices = torch.argsort(flat, stable=True)
    token_indices = sorted_indices // idx.shape[-1]
    offsets = torch.cumsum(counts, 0, dtype=torch.int32)
    return sorted_indices, token_indices, counts, offsets


def has_grouped_mm():
    return hasattr(torch, "_grouped_mm")


def grouped_mm_ref(x, w, offsets):
    """torch._grouped_mm(x [M,K], w [E,K,N], offs)."""
    return torch._grouped_mm(x, w, offs=offsets)


def loop_mm(x, w, counts_list):
    """per-expert cuBLAS loop, w [E,K,N]."""
    outs = []
    s = 0
    for e, c in enumerate(counts_list):
        if c:
            outs.append(x[s:s + c] @ w[e])
        s += c
    return torch.cat(outs, 0)


def loop_wgrad(x, g, counts_list, E):
    """dW[e] = x_e^T g_e -> [E,K,N]."""
    K, N = x.shape[1], g.shape[1]
    out = torch.zeros(E, K, N, dtype=x.dtype, device=x.device)
    s = 0
    for e, c in enumerate(counts_list):
        if c:
            out[e] = x[s:s + c].t() @ g[s:s + c]
        s += c
    return out


def mxfp4_dequant_plain(blocks, scales, dtype=torch.bfloat16):
    """Compile-friendly reference dequant: blocks (..., G, 16) uint8, scales (..., G) -> (..., G*32)."""
    lut = torch.tensor(FP4_VALUES, dtype=torch.float32, device=blocks.device)
    lo = lut[(blocks & 0x0F).long()]
    hi = lut[(blocks >> 4).long()]
    v = torch.stack((lo, hi), dim=-1).flatten(-2)  # (..., G, 32)
    exp = scales.to(torch.int32) - 127
    v = torch.ldexp(v, exp.unsqueeze(-1).to(torch.float32))
    return v.flatten(-2).to(dtype)


def small_gpu():
    """< 20 GB cards (T4 / Kaggle T4): scaled-down cases, fewer compiles (T4 is ~10x slower per case)."""
    import os
    if os.environ.get("AUDIT_FORCE_SMALL") == "1":
        return True
    return torch.cuda.get_device_properties(0).total_memory < 20e9


def tune_budget(H, ms=100):
    """On small GPUs cap the per-arm timing budget (median of >= 10 timed calls still) to fit the T4 time budget."""
    if not small_gpu() or getattr(H, "_moe_budget", False):
        return
    orig = H.measure

    def measure(fn, iters=None, warmup=None, flush_l2=False, budget_ms=None):
        return orig(fn, iters=iters, warmup=warmup, flush_l2=flush_l2, budget_ms=min(budget_ms or ms, ms))
    H.measure = measure
    H._moe_budget = True
