"""zoo _vendored/vllm/gdn_replayssm_decode.py: GDN ReplaySSM cached decode (vLLM fast_inference rollouts for
Qwen3.5 / Qwen3.6 / Qwen3-Next; backport of vllm-project/vllm#48792 wired by vllm_gdn_replayssm.py).
The state checkpoint is written only every `buffer_len` (16) steps; in between the per-token update vectors
(d, k, g) go to small ring caches and the state is rebuilt on the fly.
Arms (one decode token per sequence, gate + beta + qk-l2norm all in kernel where the arm supports it):
  ours        fused_recurrent_gated_delta_rule_replayssm, mid-ring step (write_pos=8, non-flush) and flush step
  ours_flush  same kernel on a flush step (write_pos=15)
  fla_vendored / fla_latest  fused_recurrent_gated_delta_rule(T=1, use_gate_in_kernel, use_beta_sigmoid_in_kernel,
              initial_state, output_final_state=True): the conventional per-token state read + write
  eager / compiled           transformers torch_recurrent_gated_delta_rule (+ gate math), T=1
Shapes: H=16 k-heads, HV=32 v-heads, K=V=128 (Qwen3-Next / Qwen3.5), bf16 ring caches -> fp16 (as zoo does).
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import *

H_, HV, K, V, L = 16, 32, 128, 128, 16
CASES = [
    dict(name="replay_B1", B=1, host=True, parity=True),
    dict(name="replay_B8", B=8, host=True, parity=True),
    dict(name="replay_B64", B=64),
    dict(name="replay_B256", B=256, big=True),
]


def _small_gpu():
    import torch
    return os.environ.get("AUDIT_FORCE_SMALL") == "1" or torch.cuda.get_device_properties(0).total_memory < 20 * 2**30


# < 20 GB cards (Colab T4: ~10x B200 per case, 14.5 GB): drop the big=True shapes, keep the rest
if _small_gpu():
    CASES = [c for c in CASES if not c.get("big")]


def gates(a, b, A_log, dt_bias):
    g = -A_log.float().exp() * torch.nn.functional.softplus(a.float() + dt_bias.float())
    return g, b.float().sigmoid()


def ref_step(q, k, v, g, beta, h):
    """fp math of one token: q,k [B,HV,K] (already repeated), v [B,HV,V], g/beta [B,HV], h [B,HV,K,V]"""
    q = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + 1e-6) * (K ** -0.5)
    k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    h = h * g.exp()[..., None, None]
    kv = (h * k[..., None]).sum(-2)
    d = (v - kv) * beta[..., None]
    h = h + k[..., None] * d[..., None, :]
    o = (h * q[..., None]).sum(-2)
    return o, h


def split(mixed, B):
    q = mixed[:, : H_ * K].view(B, H_, K)
    k = mixed[:, H_ * K: 2 * H_ * K].view(B, H_, K)
    v = mixed[:, 2 * H_ * K:].view(B, HV, V)
    return q, k, v


def run(case, dtype, H):
    global torch
    import torch
    rmod = import_zoo("unsloth_zoo._vendored.vllm.gdn_replayssm_decode", stub=("unsloth_zoo", "unsloth_zoo._vendored", "unsloth_zoo._vendored.vllm"))
    rep = rmod.fused_recurrent_gated_delta_rule_replayssm
    dt = H.dtype
    cdt = torch.float16 if dt == torch.bfloat16 else dt
    B = case["B"]
    S = B + 1  # slot 0 = padding slot (state_idx <= 0 bails)
    shape = [B, HV, K]
    mixed = torch.randn(B, 2 * H_ * K + HV * V, device="cuda", dtype=dt)
    a = torch.randn(B, HV, device="cuda", dtype=dt)
    b = torch.randn(B, HV, device="cuda", dtype=dt)
    A_log = torch.log(torch.empty(HV, device="cuda").uniform_(1, 16))
    dt_bias = torch.ones(HV, device="cuda")
    h0 = (torch.randn(S, HV, V, K, device="cuda") * 0.1).to(dt)
    d_cache = (torch.randn(S, HV, L, V, device="cuda") * 0.1).to(cdt)
    k_cache = (torch.randn(S, H_, L, K, device="cuda") * 0.1).to(cdt)
    g_cache = -torch.rand(S, HV, L, device="cuda") * 0.1
    idx = torch.arange(1, B + 1, device="cuda", dtype=torch.int32)
    out = torch.empty(B, HV, V, device="cuda", dtype=dt)
    wp_mid = torch.full((B,), 8, device="cuda", dtype=torch.int32)
    wp_fl = torch.full((B,), L - 1, device="cuda", dtype=torch.int32)
    nf = torch.zeros(B, device="cuda", dtype=torch.int8)
    fl = torch.ones(B, device="cuda", dtype=torch.int8)
    h0_f = h0.clone()

    def ours():
        return rep(mixed, a, b, A_log, dt_bias, K ** -0.5, h0, d_cache, k_cache, g_cache, out, idx, wp_mid, nf,
                   use_qk_l2norm_in_kernel=True)

    def ours_flush():
        return rep(mixed, a, b, A_log, dt_bias, K ** -0.5, h0_f, d_cache, k_cache, g_cache, out, idx, wp_fl, fl,
                   use_qk_l2norm_in_kernel=True)

    arms = dict(ours=ours, ours_flush=ours_flush)
    q, k, v = split(mixed, B)
    q4, k4, v4 = q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1)
    hf = torch.randn(B, HV, K, V, device="cuda") * 0.1
    try:
        import _fla_dual
        ns = _fla_dual.load_both()
        for impl, nm in (("vendored", "fla_vendored"), ("latest", "fla_latest")):
            if impl not in ns:
                continue
            f = _fla_dual.wrap(impl, ns[impl]["fused_recurrent_gated_delta_rule"])
            a4, b4 = a.unsqueeze(1), b.unsqueeze(1)
            arms[nm] = (lambda f=f, a4=a4, b4=b4: f(q4, k4, v4, g=a4, beta=b4, initial_state=hf, output_final_state=True,
                                                    use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True, A_log=A_log,
                                                    dt_bias=dt_bias, use_beta_sigmoid_in_kernel=True))
    except Exception as ex:
        H.emit(kind="info", what="fla_unavailable", err=repr(ex)[:300])

    def eager():
        g, beta = gates(a, b, A_log, dt_bias)
        qr = q.float().repeat_interleave(HV // H_, 1); kr = k.float().repeat_interleave(HV // H_, 1)
        return ref_step(qr, kr, v.float(), g, beta, hf)
    cref = torch.compile(eager, dynamic=False)
    arms["eager"] = eager
    arms["compiled"] = cref
    with torch.no_grad():
        for nm, f in list(arms.items()):
            try:
                c0 = time.time(); f(); torch.cuda.synchronize()
                H.emit(kind="cold", arm=nm, phase="fwd", shape=shape, secs=time.time() - c0)
            except Exception as ex:
                H.emit(kind="error", arm=nm, phase="fwd", err=repr(ex)[:400])
                arms.pop(nm)
        H.interleave(arms, "fwd", shape=shape, host=case.get("host", False))
        if case.get("parity"):
            # 40 chained decode steps through the ring (flush every 16th) vs fp64 recurrence
            hs = torch.zeros(S, HV, V, K, device="cuda", dtype=dt)
            dc = torch.zeros(S, HV, L, V, device="cuda", dtype=cdt)
            kc = torch.zeros(S, H_, L, K, device="cuda", dtype=cdt)
            gc = torch.zeros(S, HV, L, device="cuda")
            h64 = torch.zeros(B, HV, K, V, device="cuda", dtype=torch.float64)
            o_ours, o_64, o_32 = [], [], []
            h32 = torch.zeros(B, HV, K, V, device="cuda")
            for t in range(40):
                mx = torch.randn_like(mixed); at = torch.randn_like(a); bt = torch.randn_like(b)
                wp = torch.full((B,), t % L, device="cuda", dtype=torch.int32)
                isf = torch.full((B,), int(t % L == L - 1), device="cuda", dtype=torch.int8)
                ot = torch.empty_like(out)
                rep(mx, at, bt, A_log, dt_bias, K ** -0.5, hs, dc, kc, gc, ot, idx, wp, isf, use_qk_l2norm_in_kernel=True)
                o_ours.append(ot.clone())
                qq, kk, vv = split(mx, B)
                g, beta = gates(at, bt, A_log, dt_bias)
                r = HV // H_
                o, h64 = ref_step(qq.double().repeat_interleave(r, 1), kk.double().repeat_interleave(r, 1), vv.double(),
                                  g.double(), beta.double(), h64)
                o_64.append(o)
                o3, h32 = ref_step(qq.float().repeat_interleave(r, 1), kk.float().repeat_interleave(r, 1), vv.float(), g, beta, h32)
                o_32.append(o3.to(dt))
            # the checkpoint written on the last flush (t=31) is not the t=39 state; compare outputs only
            H.parity("ours_o_40step", torch.stack(o_ours), torch.stack(o_64), torch.stack(o_32), shape=[40] + shape)


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
