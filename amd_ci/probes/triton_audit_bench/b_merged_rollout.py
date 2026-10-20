"""zoo merged_rollout_exact.py: exact in-place LoRA fold / unfold of the base weights for vLLM rollouts
(GRPO / RL with fast_inference, 16-bit base): W <- round(W + s*B@A) before generation, bit-exact restore
after, without keeping a pristine copy of W (fix-up store of the non-invertible elements on the host).
One kernel (_exact_merge_kernel, BLOCK_N=32, BLOCK_K=128, 4 warps, fp fusion off) in 3 modes (count / fold / unfold).
Arms (one transformer layer of Llama-3-8B / Qwen3-8B, 7 projections, LoRA r=16 / 64):
  ours          exact_fold(targets) + exact_unfold(targets)   (full round trip, incl. host stores)
  ours_fold     exact_fold only
  torch_backup  W_bak = W.clone(); W.copy_((W.float() + s*(B.float()@A.float())).to(dt)); W.copy_(W_bak)
                (the straightforward exact alternative; costs a full weight copy of GPU memory)
  torch_addmm   W.addmm_(B, A, alpha=s) then W.addmm_(B, A, alpha=-s): fast, NOT exact (reference for speed only)
Also reports the extra GPU memory each arm needs.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import *

# (out, in) per projection of one layer
LLAMA8B = [(4096, 4096), (1024, 4096), (1024, 4096), (4096, 4096), (14336, 4096), (14336, 4096), (4096, 14336)]
QWEN3_8B = [(4096, 4096), (1024, 4096), (1024, 4096), (4096, 4096), (12288, 4096), (12288, 4096), (4096, 12288)]
CASES = [
    dict(name="llama8b_layer_r16", shapes=LLAMA8B, r=16),
    dict(name="qwen3_8b_layer_r64", shapes=QWEN3_8B, r=64, big=True),
    dict(name="moe_experts_E128_r16", moe=(128, 768, 2048), r=16),  # Qwen3-30B-A3B experts gate_up-like [E, N, K]
]


def _small_gpu():
    import torch
    return os.environ.get("AUDIT_FORCE_SMALL") == "1" or torch.cuda.get_device_properties(0).total_memory < 20 * 2**30


# < 20 GB cards (Colab T4: ~10x B200 per case, 14.5 GB): drop the big=True shapes, keep the rest
if _small_gpu():
    CASES = [c for c in CASES if not c.get("big")]


def run(case, dtype, H):
    global torch
    import torch
    m = import_zoo("unsloth_zoo.merged_rollout_exact")
    dt = H.dtype
    r = case["r"]
    s = 2.0
    if case.get("moe"):
        E, N, Kd = case["moe"]
        Ws = [(torch.randn(E, N, Kd, device="cuda") * 0.02).to(dt)]
        As = [(torch.randn(E, r, Kd, device="cuda") * 0.02).to(dt)]
        Bs = [(torch.randn(E, N, r, device="cuda") * 0.02).to(dt)]
    else:
        Ws = [(torch.randn(1, o, i, device="cuda") * 0.02).to(dt) for o, i in case["shapes"]]
        As = [(torch.randn(1, r, i, device="cuda") * 0.02).to(dt) for o, i in case["shapes"]]
        Bs = [(torch.randn(1, o, r, device="cuda") * 0.02).to(dt) for o, i in case["shapes"]]
    shape = [sum(w.numel() for w in Ws), r]
    W0 = [w.clone() for w in Ws]
    H.emit(kind="info", what="exact_merge_available", ok=m.exact_merge_available(Ws[0].device, dt),
           failure=m.exact_merge_failure(Ws[0].device, dt))
    targets = [m.ExactFoldTarget(w, a, b, s) for w, a, b in zip(Ws, As, Bs)]

    def ours():
        st = m.exact_fold(targets)
        m.exact_unfold(targets)
        return st

    def ours_fold_only():
        st = m.exact_fold(targets)
        return st

    def torch_backup():
        baks = [w.clone() for w in Ws]
        for w, a, b in zip(Ws, As, Bs):
            w.copy_((w.float() + s * torch.bmm(b.float(), a.float())).to(dt))
        for w, bk in zip(Ws, baks):
            w.copy_(bk)

    Wc = [w.clone() for w in Ws]

    def torch_addmm():
        for w, a, b in zip(Wc, As, Bs):
            w.baddbmm_(b, a, alpha=s)
        for w, a, b in zip(Wc, As, Bs):
            w.baddbmm_(b, a, alpha=-s)

    st = ours()
    H.emit(kind="info", what="fold_stats", **st, frac_bad=st["bad"] / max(st["elements"], 1))
    exact = all(torch.equal(w.view(torch.int16), w0.view(torch.int16)) for w, w0 in zip(Ws, W0))
    # fold correctness: within 1 ulp of fp64 reference
    ours_fold_only()
    ulps = 0
    for w, w0, a, b in zip(Ws, W0, As, Bs):
        want = (w0.double() + s * torch.bmm(b.double(), a.double())).to(dt)
        # ulps in units of the bf16/fp16 spacing at |want| (int16 bit distance is wrong across a sign flip)
        sp = torch.where(want == 0, torch.full_like(want.float(), torch.finfo(dt).tiny),
                         (torch.nextafter(want.abs(), torch.full_like(want, float("inf"))).float() - want.abs().float()))
        ulps = max(ulps, ((w.float() - want.float()).abs() / sp).max().item())
    m.exact_unfold(targets)
    H.emit(kind="parity_exact", what="roundtrip_bit_exact", exact=exact, fold_max_ulps_vs_fp64=ulps)
    # memory: peak extra over steady state per arm
    for nm, f in (("ours", ours), ("torch_backup", torch_backup)):
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        f(); torch.cuda.synchronize()
        H.emit(kind="memory", arm=nm, extra_peak_mib=(torch.cuda.max_memory_allocated() - base) / 2**20,
               weights_mib=sum(w.numel() * w.element_size() for w in Ws) / 2**20)
    H.interleave(dict(ours=ours, ours_fold=lambda: (ours_fold_only(), m.exact_unfold(targets)),
                      torch_backup=torch_backup, torch_addmm=torch_addmm), "fold_unfold", shape=shape,
                 budget_ms=200 if not H.quick else 50)
    # fold-only timing needs an unfold between calls; time fold alone via events around the fold call
    times = []
    for _ in range(5 if H.quick else 15):
        e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
        e0.record(); ours_fold_only(); e1.record(); torch.cuda.synchronize()
        times.append(e0.elapsed_time(e1))
        m.exact_unfold(targets)
    times.sort()
    H.emit(kind="time", arm="ours_fold_only", phase="fold", shape=shape, ms=times[len(times) // 2], p10=times[1], p90=times[-2], n=len(times))
    final_exact = all(torch.equal(w.view(torch.int16), w0.view(torch.int16)) for w, w0 in zip(Ws, W0))
    H.emit(kind="parity_exact", what="after_bench_bit_exact", exact=final_exact)


def _run_with_peak(case, dtype, H):
    import torch
    torch.cuda.reset_peak_memory_stats()
    try:
        run(case, dtype, H)
    finally:
        H.emit(kind="memory", what="case_peak", peak_gib=torch.cuda.max_memory_allocated() / 2**30,
               total_gib=torch.cuda.get_device_properties(0).total_memory / 2**30)


if __name__ == "__main__":
    main(CASES, _run_with_peak)
