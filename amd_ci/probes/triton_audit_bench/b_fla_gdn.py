"""Gated DeltaNet core (Qwen3.5 / Qwen3.6 / Qwen3-Next linear-attention layers).

Entry points (what transformers' Qwen3NextGatedDeltaNet / Qwen3_5GatedDeltaNet call):
  chunk_gated_delta_rule(q,k,v,g,beta, use_qk_l2norm_in_kernel=True)   training / prefill, fwd+bwd
  fused_recurrent_gated_delta_rule(..., initial_state, output_final_state=True)  cached decode, T=1
Arms:
  ours        zoo-vendored fla 0.5.1 (unsloth_zoo/_vendored/fla, injected as `fla` by fla_vendor.py)
  fla_latest  upstream fla-org/flash-linear-attention main ($AUDIT_FLA_LATEST)
  eager       transformers torch_chunk_gated_delta_rule / torch_recurrent_gated_delta_rule
              (the fallback a user without fla gets; copied from transformers 4.57.6)
  compiled    torch.compile of the same reference (T<=2048 chunk only: loop unrolls per chunk)
Shapes from Qwen3-Next-80B-A3B / Qwen3.5-9B configs: 16 k-heads repeated to 32 v-heads, dk=dv=128.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import *
import _fla_dual

HV, DK, DV = 32, 128, 128
# torch.compile of the chunk reference unrolls the per-chunk loop: 465 s fwd + 369 s bwd at T=2048 on B200,
# 46 s + 83 s at T=64. Opt in with AUDIT_SLOW=1.
SLOW = os.environ.get("AUDIT_SLOW") == "1"
CASES = [
    dict(name="chunk_B1_T2048", kind="chunk", B=1, T=2048, parity=True, compile_check=True),
    dict(name="chunk_B1_T8192", kind="chunk", B=1, T=8192, parity=False),
    dict(name="chunk_B2_T4096", kind="chunk", B=2, T=4096, big=True, parity=False),
    dict(name="chunk_tiny_T64", kind="chunk", B=1, T=64, host=True, parity=True, compiled=SLOW),
    dict(name="recur_decode_B1", kind="recur", B=1, T=1, host=True, parity=True, compiled=True),
    dict(name="recur_decode_B8", kind="recur", B=8, T=1, host=True, parity=True, compiled=True),
    dict(name="recur_decode_B64", kind="recur", B=64, T=1, parity=False, compiled=True),
]


def _small_gpu():
    import torch
    return os.environ.get("AUDIT_FORCE_SMALL") == "1" or torch.cuda.get_device_properties(0).total_memory < 20 * 2**30


# < 20 GB cards (Colab T4: ~10x B200 per case, 14.5 GB): drop the big=True shapes, keep the rest
if _small_gpu():
    CASES = [c for c in CASES if not c.get("big")]


# ----------------------------------------------------------------- transformers torch reference
def l2norm(x, dim=-1, eps=1e-6):
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


def torch_chunk_gated_delta_rule(query, key, value, g, beta, chunk_size=64, initial_state=None,
                                 output_final_state=False, use_qk_l2norm_in_kernel=False, compute_dtype=None):
    F = torch.nn.functional
    cd = compute_dtype or torch.float32
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [x.transpose(1, 2).contiguous().to(cd) for x in (query, key, value, beta, g)]
    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size)); key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size)); beta = F.pad(beta, (0, pad_size)); g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale
    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, value, k_beta, v_beta = [x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
                                         for x in (query, key, value, k_beta, v_beta)]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)
    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().to(cd)).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    last_recurrent_state = (torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
                            if initial_state is None else initial_state.to(value))
    core_attn_out = torch.zeros_like(value)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1)
    for i in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(mask, 0)
        v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn @ v_new
        last_recurrent_state = (last_recurrent_state * g[:, :, i, -1, None, None].exp()
                                + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new)
    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1])
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype if compute_dtype is None else cd)
    return core_attn_out, last_recurrent_state


def torch_recurrent_gated_delta_rule(query, key, value, g, beta, initial_state, output_final_state,
                                     use_qk_l2norm_in_kernel=False, compute_dtype=None):
    cd = compute_dtype or torch.float32
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [x.transpose(1, 2).contiguous().to(cd) for x in (query, key, value, beta, g)]
    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale
    core_attn_out = torch.zeros(batch_size, num_heads, sequence_length, v_head_dim).to(value)
    last_recurrent_state = (torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
                            if initial_state is None else initial_state.to(value))
    for i in range(sequence_length):
        q_t = query[:, :, i]; k_t = key[:, :, i]; v_t = value[:, :, i]
        g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, :, i].unsqueeze(-1)
        last_recurrent_state = last_recurrent_state * g_t
        kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        core_attn_out[:, :, i] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)
    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype if compute_dtype is None else cd)
    return core_attn_out, last_recurrent_state


# ----------------------------------------------------------------- helpers
def make_inputs(B, T, dt, requires_grad=False):
    q = torch.randn(B, T, HV, DK, device="cuda", dtype=dt)
    k = torch.randn(B, T, HV, DK, device="cuda", dtype=dt)
    v = torch.randn(B, T, HV, DV, device="cuda", dtype=dt)
    # Qwen3-Next gate: g = -exp(A_log) * softplus(a + dt_bias) (fp32), beta = sigmoid(b)
    A_log = torch.log(torch.empty(HV, device="cuda").uniform_(0, 16))
    a = torch.randn(B, T, HV, device="cuda")
    g = (-A_log.exp() * torch.nn.functional.softplus(a + 1.0)).float()
    beta = torch.randn(B, T, HV, device="cuda", dtype=dt).sigmoid()
    ts = [q, k, v, g, beta]
    if requires_grad:
        for t in ts:
            t.requires_grad_(True)
    return ts


def _grads(ts):
    out = tuple(t.grad.clone() if t.grad is not None else None for t in ts)
    for t in ts:
        t.grad = None
    return out


def autotune_summary(H, impl):
    """Per autotuned kernel of a tree: n configs, keys tuned, last bench_time, best vs worst / 2nd."""
    import triton
    seen = 0
    total_bench = 0.0
    for mn, m in _fla_dual._SNAP.get(impl, {}).items():
        if m is None:
            continue
        for an, obj in list(vars(m).items()):
            o = obj
            for _ in range(4):
                if isinstance(o, triton.runtime.autotuner.Autotuner):
                    break
                o = getattr(o, "fn", None)
                if o is None:
                    break
            if not isinstance(o, triton.runtime.autotuner.Autotuner) or not getattr(o, "cache", None):
                continue
            if id(o) in getattr(H, "_seen_at", set()):
                continue
            H._seen_at = getattr(H, "_seen_at", set()); H._seen_at.add(id(o))
            ct = getattr(o, "configs_timings", None) or {}
            vals = sorted(float(t[0] if isinstance(t, (list, tuple)) else t) for t in ct.values())
            vals = [v for v in vals if v == v and v < float("inf")]
            bt = getattr(o, "bench_time", None)
            total_bench += bt or 0.0
            seen += 1
            H.emit(kind="autotune", what="%s:%s.%s" % (impl, mn.replace("fla.", ""), an),
                   n_configs=len(o.configs), n_keys=len(o.cache), bench_time_s=bt,
                   best_ms=vals[0] if vals else None, second_ms=vals[1] if len(vals) > 1 else None,
                   worst_ms=vals[-1] if vals else None,
                   spread_worst_over_best=(vals[-1] / vals[0]) if vals and vals[0] > 0 else None,
                   best=str(list(o.cache.values())[-1]))
    H.emit(kind="autotune_total", what=impl, kernels=seen, bench_time_s_sum=total_bench)


def run(case, dtype, H):
    global torch
    import torch
    t0 = time.time()
    ns = _fla_dual.load_both()
    small = _fla_dual.small_gpu()
    torch.cuda.reset_peak_memory_stats()
    H.emit(kind="info", what="fla_import", autotune_cap=ns.get("autotune_cap"), small_gpu=small, vendored_s=ns["vendored"]["import_s"],
           latest_s=ns.get("latest", {}).get("import_s"), latest_err=ns.get("latest_err"),
           latest_root=ns.get("latest", {}).get("root"))
    impls = ["vendored"] + (["latest"] if "latest" in ns else [])
    arm_name = {"vendored": "ours", "latest": "fla_latest"}
    dt = H.dtype
    B, T = case["B"], case["T"]
    shape = [B, T, HV, DK]

    if case["kind"] == "chunk":
        fns = {i: _fla_dual.wrap(i, ns[i]["chunk_gated_delta_rule"]) for i in impls}
        q, k, v, g, beta = make_inputs(B, T, dt, requires_grad=True)
        do = torch.randn(B, T, HV, DV, device="cuda", dtype=dt)
        # cold first call (JIT + autotune; on-disk caches may already be warm on rep>0)
        for i in impls:
            torch.cuda.synchronize(); c0 = time.time()
            o, _ = fns[i](q, k, v, g, beta, use_qk_l2norm_in_kernel=True)
            o.backward(do)
            torch.cuda.synchronize()
            H.emit(kind="cold", arm=arm_name[i], phase="fwdbwd", shape=shape, secs=time.time() - c0)
            _grads([q, k, v, g, beta])

        def fwd(i):
            def f():
                with torch.no_grad():
                    return fns[i](q, k, v, g, beta, use_qk_l2norm_in_kernel=True)
            return f

        def fb(i):
            def f():
                o, _ = fns[i](q, k, v, g, beta, use_qk_l2norm_in_kernel=True)
                o.backward(do)
            return f

        def ref_fwd():
            with torch.no_grad():
                return torch_chunk_gated_delta_rule(q, k, v, g, beta, use_qk_l2norm_in_kernel=True)

        def ref_fb():
            o, _ = torch_chunk_gated_delta_rule(q, k, v, g, beta, use_qk_l2norm_in_kernel=True)
            o.backward(do)

        arms_f = {arm_name[i]: fwd(i) for i in impls}
        arms_fb = {arm_name[i]: fb(i) for i in impls}
        # eager reference: python loop over chunks + 63-step triangular solve; budget-limited
        # eager reference: python loop over chunks + 63-step triangular solve (0.1-0.4 s on B200, ~10x on T4):
        # timed outside the interleave with 3 iterations; fwd+bwd only where autograd's saved graph fits
        eager_f = ref_fwd
        eager_fb = ref_fb if (B * T <= 2048 or not small) else None
        if case.get("compiled"):
            cref = torch.compile(torch_chunk_gated_delta_rule, dynamic=False)
            c0 = time.time()
            try:
                with torch.no_grad():
                    cref(q, k, v, g, beta, use_qk_l2norm_in_kernel=True)
                H.emit(kind="cold", arm="compiled", phase="fwd", shape=shape, secs=time.time() - c0)

                def cf():
                    with torch.no_grad():
                        return cref(q, k, v, g, beta, use_qk_l2norm_in_kernel=True)
                arms_f["compiled"] = cf
                c0 = time.time()
                o, _ = cref(q, k, v, g, beta, use_qk_l2norm_in_kernel=True); o.backward(do); _grads([q, k, v, g, beta])
                H.emit(kind="cold", arm="compiled", phase="fwdbwd", shape=shape, secs=time.time() - c0)

                def cfb():
                    o, _ = cref(q, k, v, g, beta, use_qk_l2norm_in_kernel=True)
                    o.backward(do)
                arms_fb["compiled"] = cfb
            except Exception as ex:
                H.emit(kind="error", arm="compiled", phase="fwd", shape=shape, err=repr(ex)[:400])
        H.interleave(arms_f, "fwd", shape=shape, host=case.get("host", False), budget_ms=300 if not H.quick else 60)
        H.interleave(arms_fb, "fwdbwd", shape=shape, host=case.get("host", False), budget_ms=300 if not H.quick else 60)
        H.time_arm("eager", "fwd", eager_f, shape=shape, iters=3, warmup=1)
        if eager_fb is not None:
            H.time_arm("eager", "fwdbwd", eager_fb, shape=shape, iters=3, warmup=1)
        _grads([q, k, v, g, beta])
        torch.cuda.empty_cache()

        if case.get("parity"):
            # fp64 oracle (same math, chunk form, fp64 end to end) for o and dq/dk/dv/dg/dbeta
            ins64 = [t.detach().double().requires_grad_(True) for t in (q, k, v, g, beta)]
            o64, _ = torch_chunk_gated_delta_rule(*ins64, use_qk_l2norm_in_kernel=True, compute_dtype=torch.float64)
            o64.backward(do.double())
            g64 = tuple(t.grad for t in ins64)
            o_ref, _ = torch_chunk_gated_delta_rule(q, k, v, g, beta, use_qk_l2norm_in_kernel=True)
            o_ref.backward(do); g_ref = _grads([q, k, v, g, beta])
            for i in impls:
                o, _ = fns[i](q, k, v, g, beta, use_qk_l2norm_in_kernel=True)
                o.backward(do); gi = _grads([q, k, v, g, beta])
                H.parity("%s_o" % arm_name[i], o.detach(), o64.detach(), o_ref.detach(), shape=shape)
                H.parity("%s_grads(q,k,v,g,beta)" % arm_name[i], gi, g64, g_ref, shape=shape)
        if case.get("compile_check"):
            raw = ns["vendored"]["chunk_gated_delta_rule"]

            def entry(q_, k_, v_, g_, b_):
                o, _ = raw(q_, k_, v_, g_, b_, use_qk_l2norm_in_kernel=True)
                return o
            q2, k2, v2, g2, b2 = make_inputs(B, T // 2, dt, requires_grad=True)
            _fla_dual.use("vendored")
            H.compile_check("ours_chunk_gated_delta_rule", entry, [(q, k, v, g, beta), (q2, k2, v2, g2, b2)])
        for i in impls:
            autotune_summary(H, i)

    else:  # recurrent decode, T=1 with an fp32 state carried
        fns = {i: _fla_dual.wrap(i, ns[i]["fused_recurrent_gated_delta_rule"]) for i in impls}
        q, k, v, g, beta = make_inputs(B, T, dt)
        h0 = torch.randn(B, HV, DK, DV, device="cuda", dtype=torch.float32) * 0.1

        def rec(i):
            def f():
                return fns[i](q, k, v, g=g, beta=beta, initial_state=h0, output_final_state=True,
                              use_qk_l2norm_in_kernel=True)
            return f

        def ref():
            return torch_recurrent_gated_delta_rule(q, k, v, g, beta, h0, True, use_qk_l2norm_in_kernel=True)

        for i in impls:
            torch.cuda.synchronize(); c0 = time.time(); rec(i)(); torch.cuda.synchronize()
            H.emit(kind="cold", arm=arm_name[i], phase="fwd", shape=shape, secs=time.time() - c0)
        arms = {arm_name[i]: rec(i) for i in impls}
        arms["eager"] = ref
        if case.get("compiled"):
            cref = torch.compile(torch_recurrent_gated_delta_rule, dynamic=False)
            try:
                c0 = time.time(); cref(q, k, v, g, beta, h0, True, use_qk_l2norm_in_kernel=True); torch.cuda.synchronize()
                H.emit(kind="cold", arm="compiled", phase="fwd", shape=shape, secs=time.time() - c0)
                arms["compiled"] = lambda: cref(q, k, v, g, beta, h0, True, use_qk_l2norm_in_kernel=True)
            except Exception as ex:
                H.emit(kind="error", arm="compiled", phase="fwd", shape=shape, err=repr(ex)[:400])
        H.interleave(arms, "fwd", shape=shape, host=case.get("host", False), budget_ms=200 if not H.quick else 50)
        if case.get("parity"):
            # multi-token: 32 steps of decode chained through the state, vs fp64 recurrence
            Tn = 32
            qs, ks, vs, gs, bs = make_inputs(B, Tn, dt)
            o64, h64 = torch_recurrent_gated_delta_rule(*(t.double() for t in (qs, ks, vs, gs, bs)), h0.double(), True,
                                                        use_qk_l2norm_in_kernel=True, compute_dtype=torch.float64)
            oref, _ = torch_recurrent_gated_delta_rule(qs, ks, vs, gs, bs, h0, True, use_qk_l2norm_in_kernel=True)
            for i in impls:
                h = h0.clone(); outs = []
                for t in range(Tn):
                    sl = slice(t, t + 1)
                    o, h = fns[i](qs[:, sl], ks[:, sl], vs[:, sl], g=gs[:, sl], beta=bs[:, sl], initial_state=h,
                                  output_final_state=True, use_qk_l2norm_in_kernel=True)
                    outs.append(o)
                o = torch.cat(outs, 1)
                H.parity("%s_o_32step" % arm_name[i], (o, h), (o64, h64), (oref, None), shape=[B, Tn, HV, DK])
            raw = ns["vendored"]["fused_recurrent_gated_delta_rule"]

            def entry(q_, k_, v_, g_, b_, h_):
                return raw(q_, k_, v_, g=g_, beta=b_, initial_state=h_, output_final_state=True,
                                       use_qk_l2norm_in_kernel=True)[0]
            _fla_dual.use("vendored")
            q2, k2, v2, g2, b2 = make_inputs(B + 1, T, dt)
            h2 = torch.zeros(B + 1, HV, DK, DV, device="cuda")
            H.compile_check("ours_fused_recurrent_decode", entry, [(q, k, v, g, beta, h0), (q2, k2, v2, g2, b2, h2)])
    _fla_dual.use("vendored")
    H.emit(kind="memory", what="case_peak", peak_gib=torch.cuda.max_memory_allocated() / 2**30,
           total_gib=torch.cuda.get_device_properties(0).total_memory / 2**30)


if __name__ == "__main__":
    main(CASES, run)
