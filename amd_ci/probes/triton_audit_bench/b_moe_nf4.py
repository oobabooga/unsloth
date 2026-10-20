"""4-bit (bnb NF4) MoE expert kernels + the weighted unpermute epilogue.

Entry points:
  * zoo moe_triton_kernels.nf4_dequant_triton(packed, quant_state)   <- moe_utils._dequantize_bnb4bit_experts
      (stacked NF4 experts [E, N, K] as one Params4bit: Qwen3-MoE / Qwen3.5-MoE / GLM bnb-4bit QLoRA, train + prefill)
  * zoo gpt_oss_grouped_qlora.nf4_dequant_expert_stack(tb, dtype)   <- grouped_qlora_forward (gpt-oss bnb-4bit,
      per-expert Linear4bit ModuleList, QLoRA train)
  * zoo moe_triton_kernels.weighted_unpermute(y, sorted_idx, w, T, k) <- moe_utils.combine_permuted_moe_outputs
      (every grouped MoE forward, fwd + bwd, 4-bit and 16-bit)
References: bitsandbytes.functional.dequantize_4bit (CUDA), torch.compile of the plain NF4 dequant math,
eager / compiled torch index math for the unpermute.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import *
import moe_common as mc

CASES = [
    dict(name="dequant_stacked_qwen3_gate_up", kind="stacked", model="qwen3_30b_a3b", proj="gate_up"),
    dict(name="dequant_stacked_gptoss_down", kind="stacked", model="gpt_oss_20b", proj="down"),
    dict(name="dequant_ptrstack_gptoss_gate_up", kind="ptrstack", model="gpt_oss_20b", proj="gate_up"),
    dict(name="unpermute_qwen3_T4096", kind="unpermute", model="qwen3_30b_a3b", T=4096),
    dict(name="unpermute_gptoss_T4096", kind="unpermute", model="gpt_oss_20b", T=4096),
    dict(name="unpermute_qwen3_T8_host", kind="unpermute", model="qwen3_30b_a3b", T=8, host=True),
]


def nf4_plain(packed_u8, absmax_q, absmax2, code2, offset, lut, blocksize, blocksize2, dtype):
    """bnb NF4 dequant (nested absmax), plain torch; packed hi nibble first."""
    a = code2[absmax_q.long()] * absmax2.repeat_interleave(blocksize2)[: absmax_q.numel()] + offset
    q = packed_u8.reshape(-1)
    hi = lut[(q >> 4).long()]
    lo = lut[(q & 15).long()]
    v = torch.stack((hi, lo), -1).reshape(-1, blocksize)
    return (v * a[:, None]).to(dtype)


def quantize(Wf, dt):
    import bitsandbytes.functional as F
    q, qs = F.quantize_4bit(Wf, blocksize=64, compress_statistics=True, quant_type="nf4", quant_storage=torch.uint8)
    qs.dtype = dt
    return q, qs


def run(case, dtype, H):
    global torch
    import torch
    mk = import_zoo("unsloth_zoo.temporary_patches.moe_triton_kernels")
    dt = H.dtype
    mc.tune_budget(H)
    Hd, E, k, I = mc.MODELS[case["model"]]
    kind = case["kind"]
    if kind in ("stacked", "ptrstack"):
        try:
            import bitsandbytes.functional as BF
        except Exception as ex:
            H.emit(kind="skip", why="bitsandbytes missing: %r" % ex)
            return
        N, K = (2 * I, Hd) if case["proj"] == "gate_up" else (Hd, I)
        g = torch.Generator(device="cuda").manual_seed(0)
        if kind == "stacked":
            Wf = (torch.randn(E * N, K, device="cuda", generator=g) * 0.02).to(dt)
            q, qs = quantize(Wf, dt)
            del Wf
            torch.cuda.empty_cache()
            H.emit(kind="info", what="availability", triton=mk.moe_triton_kernels_available(q.device), nested=qs.nested)
            off = qs.offset if isinstance(qs.offset, torch.Tensor) else torch.tensor(float(qs.offset), device="cuda")
            comp = torch.compile(nf4_plain, dynamic=False)
            lut = qs.code.float()
            args = (q, qs.absmax, qs.state2.absmax.float(), qs.state2.code.float(), off.float(), lut, 64, qs.state2.blocksize, dt)
            shape = [E, N, K]
            H.interleave(dict(
                ours=lambda: mk.nf4_dequant_triton(q, qs, (E, N, K)),
                bnb=lambda: BF.dequantize_4bit(q, qs),
                compiled=lambda: comp(*args),
            ), "dequant", shape=shape)
            o = mk.nf4_dequant_triton(q, qs, (E, N, K))
            r = BF.dequantize_4bit(q, qs).view(E, N, K)
            H.emit(kind="parity_exact", what="ours_vs_bnb_bitexact", equal=bool(torch.equal(o, r)),
                   n_diff=int((o != r).sum().item()))
            if not (mc.small_gpu() and case["model"] != "qwen3_30b_a3b"):
                H.compile_check("nf4_dequant_triton", lambda a: mk.nf4_dequant_triton(a, qs, (E, N, K)), [(q,)])
        else:
            gq = import_zoo("unsloth_zoo.temporary_patches.gpt_oss_grouped_qlora")
            qs_list, qw = [], []
            for e in range(E):
                Wf = (torch.randn(N, K, device="cuda", generator=g) * 0.02).to(dt)
                q, qs = quantize(Wf, dt)
                qw.append(q); qs_list.append(qs)
            i64 = lambda v: torch.tensor(v, dtype=torch.int64, device="cuda")
            tb = dict(w=i64([t.data_ptr() for t in qw]), a=i64([s.absmax.data_ptr() for s in qs_list]),
                      a2=i64([s.state2.absmax.data_ptr() for s in qs_list]), c2=i64([s.state2.code.data_ptr() for s in qs_list]),
                      off=torch.tensor([float(s.offset) for s in qs_list], dtype=torch.float32, device="cuda"),
                      lut=qs_list[0].code.float().contiguous(), N=N, K=K, blocksize=64,
                      blocksize2=int(qs_list[0].state2.blocksize), nested=True)
            H.emit(kind="info", what="availability", triton=gq.stacked_dequant_available(torch.device("cuda", 0)))
            out = torch.empty(E, N, K, dtype=dt, device="cuda")

            def bnb_stack():
                for e in range(E):
                    out[e].copy_(BF.dequantize_4bit(qw[e], qs_list[e]))
                return out
            shape = [E, N, K]
            H.interleave(dict(
                ours=lambda: gq.nf4_dequant_expert_stack(tb, dt),
                bnb_loop=bnb_stack,
            ), "dequant", shape=shape)
            o = gq.nf4_dequant_expert_stack(tb, dt)
            r = bnb_stack()
            H.emit(kind="parity_exact", what="ours_vs_bnb_bitexact", equal=bool(torch.equal(o, r)),
                   n_diff=int((o != r).sum().item()))
    elif kind == "unpermute":
        T = case["T"]
        idx, w = mc.route(T, E, k)
        sorted_idx, tok, counts, offs = mc.sort_routing(idx, E)
        P = T * k
        y = torch.randn(P, Hd, device="cuda", dtype=dt, requires_grad=True)
        w_perm = w.reshape(-1)[sorted_idx].to(dt).detach().requires_grad_(True)

        def ref(y, sorted_idx, w_perm):
            # out[t] = sum_k w * y over token t's slots (fp32 accumulate)
            inv = torch.empty_like(sorted_idx)
            inv[sorted_idx] = torch.arange(sorted_idx.numel(), device=y.device)
            yy = y.float()[inv].view(T, k, -1)
            ww = w_perm.float()[inv].view(T, k, 1)
            return (yy * ww).sum(1).to(y.dtype)
        comp = torch.compile(ref, dynamic=False)
        shape = [T, k, Hd]
        with torch.no_grad():
            H.interleave(dict(
                ours=lambda: mk.weighted_unpermute(y, sorted_idx, w_perm, T, k),
                eager=lambda: ref(y, sorted_idx, w_perm),
                compiled=lambda: comp(y, sorted_idx, w_perm),
            ), "fwd", shape=shape, host=case.get("host", False))
        go = torch.randn(T, Hd, device="cuda", dtype=dt)

        def fb(fn):
            def f():
                out = fn(y, sorted_idx, w_perm)
                torch.autograd.grad(out, (y, w_perm), go)
            return f
        H.interleave(dict(
            ours=fb(lambda a, b, c: mk.weighted_unpermute(a, b, c, T, k)),
            eager=fb(ref),
            compiled=fb(comp),
        ), "fwdbwd", shape=shape, host=case.get("host", False))
        o = mk.weighted_unpermute(y, sorted_idx, w_perm, T, k)
        gy_o, gw_o = torch.autograd.grad(o, (y, w_perm), go)
        y64, w64 = y.detach().double().requires_grad_(True), w_perm.detach().double().requires_grad_(True)
        r64 = ref(y64, sorted_idx, w64)
        gy_r, gw_r = torch.autograd.grad(r64, (y64, w64), go.double())
        rl = ref(y, sorted_idx, w_perm)
        gy_l, gw_l = torch.autograd.grad(rl, (y, w_perm), go)
        H.parity("unpermute", (o, gy_o, gw_o), (r64, gy_r, gw_r), (rl, gy_l, gw_l), shape=shape)
        if not (mc.small_gpu() and T != 4096) and case["model"] == "qwen3_30b_a3b":
            H.compile_check("weighted_unpermute", lambda a, b, c: mk.weighted_unpermute(a, b, c, T, k), [(y, sorted_idx, w_perm)])


if __name__ == "__main__":
    main(CASES, run)
