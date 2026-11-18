"""gfx1151: packed-row attention candidates vs fp32 per-document reference, fwd+bwd time and error."""
import json, math, sys, time
import torch
import torch.nn.functional as F

out_path = sys.argv[1]
dev = torch.device("cuda")
res = {"torch": torch.__version__, "hip": torch.version.hip,
       "name": torch.cuda.get_device_name(0), "cap": list(torch.cuda.get_device_capability(0)), "cases": []}
try:
    from torch.nn.attention.varlen import varlen_attn
    import inspect
    res["varlen_params"] = list(inspect.signature(varlen_attn).parameters)
except Exception as e:
    varlen_attn = None
    res["varlen_import"] = repr(e)


def doc_lengths(total, seed):
    g = torch.Generator().manual_seed(seed)
    out, left = [], total
    while left > 0:
        L = int(min(left, max(16, math.exp(5.6 + 0.9 * torch.randn(1, generator = g).item()))))
        out.append(L)
        left -= L
    return out


def ref_attn(q, k, v, lengths, scale, window):
    # q (1,H,T,D) fp32, k/v (1,Hkv,T,D): per-document causal (windowed) attention, explicit softmax.
    H, Hkv = q.shape[1], k.shape[1]
    k = k.repeat_interleave(H // Hkv, 1)
    v = v.repeat_interleave(H // Hkv, 1)
    outs, s = [], 0
    for L in lengths:
        qs, ks, vs = q[:, :, s:s + L], k[:, :, s:s + L], v[:, :, s:s + L]
        att = (qs @ ks.transpose(-1, -2)) * scale
        i = torch.arange(L, device = q.device)
        m = i[None, :] <= i[:, None]
        if window is not None:
            m &= i[None, :] > i[:, None] - window
        att = att.masked_fill(~m, float("-inf")).softmax(-1)
        outs.append(att @ vs)
        s += L
    return torch.cat(outs, 2).transpose(1, 2)  # (1,T,H,D)


def cu_of(lengths):
    return torch.tensor([0] + list(torch.tensor(lengths).cumsum(0).tolist()), dtype = torch.int32, device = dev)


def c_dense(q, k, v, lengths, scale, window):
    T = q.shape[2]
    seg = torch.repeat_interleave(torch.arange(len(lengths), device = dev), torch.tensor(lengths, device = dev))
    i = torch.arange(T, device = dev)
    m = (seg[:, None] == seg[None, :]) & (i[None, :] <= i[:, None])
    if window is not None:
        m &= i[None, :] > i[:, None] - window
    H, Hkv = q.shape[1], k.shape[1]
    kk, vv = k.repeat_interleave(H // Hkv, 1), v.repeat_interleave(H // Hkv, 1)
    return F.scaled_dot_product_attention(q, kk, vv, attn_mask = m[None, None], scale = scale).transpose(1, 2)


def c_varlen(q, k, v, lengths, scale, window, gqa_native):
    cu = cu_of(lengths)
    H, Hkv = q.shape[1], k.shape[1]
    kw = {}
    if Hkv != H:
        if gqa_native:
            kw["enable_gqa"] = True
        else:
            k, v = k.repeat_interleave(H // Hkv, 1), v.repeat_interleave(H // Hkv, 1)
    o = varlen_attn(q[0].transpose(0, 1), k[0].transpose(0, 1), v[0].transpose(0, 1), cu, cu,
                    max(lengths), max(lengths), scale = scale,
                    window_size = (-1, 0) if window is None else (window - 1, 0), **kw)
    return o.unsqueeze(0)


def c_aten_flash(q, k, v, lengths, scale, window):
    cu = cu_of(lengths)
    o = torch.ops.aten._flash_attention_forward(
        q[0].transpose(0, 1), k[0].transpose(0, 1), v[0].transpose(0, 1), cu, cu, max(lengths), max(lengths),
        0.0, False, False, scale = scale,
        window_size_left = -1 if window is None else window - 1, window_size_right = 0)[0]
    return o.unsqueeze(0)


class Eff(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, cu, mx, scale, window):
        o, lse, seed, off, _, _ = torch.ops.aten._efficient_attention_forward(
            q, k, v, None, cu, cu, mx, mx, 0.0, 1, True, scale = scale, window_size = window)
        ctx.save_for_backward(q, k, v, o, lse, cu, seed, off)
        ctx.mx, ctx.scale, ctx.window = mx, scale, window
        return o

    @staticmethod
    def backward(ctx, g):
        q, k, v, o, lse, cu, seed, off = ctx.saved_tensors
        dq, dk, dv, _ = torch.ops.aten._efficient_attention_backward(
            g.contiguous(), q, k, v, None, o, cu, cu, ctx.mx, ctx.mx, lse, 0.0, seed, off, 1, False,
            scale = ctx.scale, window_size = ctx.window)
        return dq, dk, dv, None, None, None, None


def c_efficient(q, k, v, lengths, scale, window):
    H, Hkv = q.shape[1], k.shape[1]
    qq = q[0].transpose(0, 1).unsqueeze(0)
    kk = k.repeat_interleave(H // Hkv, 1)[0].transpose(0, 1).unsqueeze(0)
    vv = v.repeat_interleave(H // Hkv, 1)[0].transpose(0, 1).unsqueeze(0)
    return Eff.apply(qq, kk, vv, cu_of(lengths), max(lengths), scale, window)


def c_loop(q, k, v, lengths, scale, window):
    if window is not None:
        raise RuntimeError("loop: no window")
    H, Hkv = q.shape[1], k.shape[1]
    kk, vv = k.repeat_interleave(H // Hkv, 1), v.repeat_interleave(H // Hkv, 1)
    outs, s = [], 0
    for L in lengths:
        outs.append(F.scaled_dot_product_attention(q[:, :, s:s + L], kk[:, :, s:s + L], vv[:, :, s:s + L],
                                                   is_causal = True, scale = scale))
        s += L
    return torch.cat(outs, 2).transpose(1, 2)


def run_case(name, fn, q0, k0, v0, lengths, scale, window, ref, gref, iters):
    rec = {"cand": name}
    try:
        q, k, v = (x.detach().clone().requires_grad_() for x in (q0, k0, v0))
        o = fn(q, k, v, lengths, scale, window)
        go = torch.ones_like(o) * 0 + gref[0].to(o.dtype)
        dq, dk, dv = torch.autograd.grad(o, (q, k, v), go)
        torch.cuda.synchronize()
        def rel(a, b):
            return float((a.float() - b).abs().max() / b.abs().max().clamp_min(1e-6))
        rec.update(err_out = rel(o, ref), err_dq = rel(dq, gref[1]), err_dk = rel(dk, gref[2]),
                   err_dv = rel(dv, gref[3]))
        for _ in range(2):
            o = fn(q, k, v, lengths, scale, window); torch.autograd.grad(o, (q, k, v), go)
        torch.cuda.synchronize()
        ts = []
        for _ in range(iters):
            s = torch.cuda.Event(enable_timing = True); e = torch.cuda.Event(enable_timing = True)
            s.record()
            o = fn(q, k, v, lengths, scale, window)
            torch.autograd.grad(o, (q, k, v), go)
            e.record(); torch.cuda.synchronize()
            ts.append(s.elapsed_time(e))
        ts.sort()
        rec.update(ms_median = ts[len(ts) // 2], ms_min = ts[0], ms_max = ts[-1])
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {str(e)[:300]}"
        torch.cuda.synchronize()
    return rec


def padded_case(q0, k0, v0, lengths, scale, iters, batch = 4):
    # Same documents padded per batch of 4 to the longest (the reporter's padded path), causal SDPA.
    H, Hkv, D = q0.shape[1], k0.shape[1], q0.shape[3]
    ts_total, s = 0.0, 0
    groups = [lengths[i:i + batch] for i in range(0, len(lengths), batch)]
    try:
        for grp in groups:
            Lm = max(grp)
            q = torch.randn(len(grp), H, Lm, D, device = dev, dtype = q0.dtype, requires_grad = True)
            k = torch.randn(len(grp), Hkv, Lm, D, device = dev, dtype = q0.dtype, requires_grad = True)
            v = torch.randn(len(grp), Hkv, Lm, D, device = dev, dtype = q0.dtype, requires_grad = True)
            def f():
                kk, vv = k.repeat_interleave(H // Hkv, 1), v.repeat_interleave(H // Hkv, 1)
                o = F.scaled_dot_product_attention(q, kk, vv, is_causal = True, scale = scale)
                torch.autograd.grad(o.sum(), (q, k, v))
            f(); torch.cuda.synchronize()
            ts = []
            for _ in range(iters):
                a = torch.cuda.Event(enable_timing = True); b = torch.cuda.Event(enable_timing = True)
                a.record(); f(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b))
            ts.sort(); ts_total += ts[len(ts) // 2]
        return {"cand": "padded_batches_of_4", "ms_median": ts_total, "n_batches": len(groups)}
    except Exception as e:
        return {"cand": "padded_batches_of_4", "error": f"{type(e).__name__}: {str(e)[:300]}"}


configs = [
    # (label, H, Hkv, D, T, window, dtype)
    ("q35_4b_full_T8192", 16, 4, 256, 8192, None, torch.bfloat16),
    ("q35_4b_full_T2048", 16, 4, 256, 2048, None, torch.bfloat16),
    ("hd128_gqa_T8192", 32, 8, 128, 8192, None, torch.bfloat16),
    ("hd128_window512_T4096", 16, 4, 128, 4096, 512, torch.bfloat16),
    ("q35_fp16_T2048", 16, 4, 256, 2048, None, torch.float16),
]
for label, H, Hkv, D, T, window, dtype in configs:
    lengths = doc_lengths(T, seed = T + D)
    torch.manual_seed(0)
    q0 = torch.randn(1, H, T, D, device = dev, dtype = dtype)
    k0 = torch.randn(1, Hkv, T, D, device = dev, dtype = dtype)
    v0 = torch.randn(1, Hkv, T, D, device = dev, dtype = dtype)
    scale = D ** -0.5
    qf, kf, vf = (x.float().requires_grad_() for x in (q0, k0, v0))
    ref = ref_attn(qf, kf, vf, lengths, scale, window)
    go = torch.randn_like(ref)
    gq, gk, gv = torch.autograd.grad(ref, (qf, kf, vf), go)
    gref = (go, gq.detach(), gk.detach(), gv.detach())
    ref = ref.detach()
    iters = 10 if T <= 4096 else 6
    case = {"label": label, "H": H, "Hkv": Hkv, "D": D, "T": T, "window": window, "dtype": str(dtype),
            "n_docs": len(lengths), "max_doc": max(lengths), "results": []}
    cands = [("dense_mask_sdpa", c_dense), ("aten_flash_varlen_gqa", c_aten_flash),
             ("efficient_varlen", c_efficient), ("per_doc_sdpa_loop", c_loop)]
    if varlen_attn is not None:
        cands.insert(1, ("varlen_attn_repeat", lambda *a: c_varlen(*a, gqa_native = False)))
        if "enable_gqa" in res.get("varlen_params", []):
            cands.insert(2, ("varlen_attn_gqa", lambda *a: c_varlen(*a, gqa_native = True)))
    for name, fn in cands:
        case["results"].append(run_case(name, fn, q0, k0, v0, lengths, scale, window, ref, gref, iters))
        print(label, case["results"][-1], flush = True)
    if window is None:
        case["results"].append(padded_case(q0, k0, v0, lengths, scale, iters))
        print(label, case["results"][-1], flush = True)
    res["cases"].append(case)
    del qf, kf, vf, ref, gref
    torch.cuda.empty_cache()
    with open(out_path, "w") as f:
        json.dump(res, f, indent = 1)
print("ATTN_BENCH_DONE", flush = True)
