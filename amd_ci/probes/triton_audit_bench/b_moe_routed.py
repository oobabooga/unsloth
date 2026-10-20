"""Decode-sized (no-grad) routed MoE: zoo moe_routed.py (17 kernels), via moe_utils.forward_moe_backend ->
routed_moe_forward for <= 32 (bf16 / fp16) or <= E (stacked NF4) (token, expert) slots.

Entry points benched:
  * routed_bf16_moe(x, idx, rw, w_gu [E,H,2I], w_dn [E,I,H], ...)  16-bit experts, fused (default) and
    UNSLOTH_MOE_ROUTED_FUSED=0 (torch epilogues)
  * _nf4_routed(state, ...)  stacked NF4 experts (bnb-4bit Qwen3-MoE etc.), state built like prepare_stacked_nf4
References: the path taken when the routed kernels decline (sort + torch._grouped_mm + act + torch._grouped_mm +
weighted sum), gather + bmm eager and torch.compile'd, zoo's own NF4 comparator (selective dequant + grouped_mm).
"""
import os, sys, types
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import *
import moe_common as mc

CASES = [
    dict(name="bf16_qwen3_T1_host", kind="bf16", model="qwen3_30b_a3b", T=1, host=True),
    dict(name="bf16_qwen3_T4", kind="bf16", model="qwen3_30b_a3b", T=4),       # 32 slots = BF16_MAX_SLOTS
    dict(name="bf16_gptoss_T8", kind="bf16", model="gpt_oss_20b", T=8),        # 32 slots (SiLU stand-in act)
    dict(name="nf4_qwen3_T1_host", kind="nf4", model="qwen3_30b_a3b", T=1, host=True),
    dict(name="nf4_qwen3_T16", kind="nf4", model="qwen3_30b_a3b", T=16),       # 128 slots = E
    dict(name="nf4_gptoss_T8", kind="nf4", model="gpt_oss_20b", T=8),          # 32 slots = E
]


def silu_mul(gu):
    g, u = gu.chunk(2, dim=-1)
    return torch.nn.functional.silu(g) * u


def ref_grouped(x, idx, rw, w_gu, w_dn, E):
    """forward_native_grouped_mm without LoRA: what runs when the routed kernels decline."""
    T, k = idx.shape
    flat = idx.reshape(-1)
    counts = torch.bincount(flat, minlength=E).to(torch.int32)
    order = torch.argsort(flat, stable=True)
    offs = torch.cumsum(counts, 0, dtype=torch.int32)
    xs = x[order // k]
    h = silu_mul(torch._grouped_mm(xs, w_gu, offs=offs))
    y = torch._grouped_mm(h, w_dn, offs=offs)
    out = torch.zeros(T * k, y.shape[-1], dtype=torch.float32, device=x.device)
    out[order] = y.float() * rw.reshape(-1)[order].float()[:, None]
    return out.view(T, k, -1).sum(1).to(x.dtype)


def ref_gather(x, idx, rw, w_gu, w_dn):
    T, k = idx.shape
    flat = idx.reshape(-1)
    xs = x.repeat_interleave(k, 0).unsqueeze(1)                 # [P,1,H]
    h = silu_mul(torch.bmm(xs, w_gu[flat]))                       # [P,1,2I] -> [P,1,I]
    y = torch.bmm(h, w_dn[flat]).squeeze(1)                       # [P,H]
    return (y.float().view(T, k, -1) * rw.float()[..., None]).sum(1).to(x.dtype)


def run(case, dtype, H):
    global torch
    import torch
    mr = import_zoo("unsloth_zoo.temporary_patches.moe_routed")
    dt = H.dtype
    mc.tune_budget(H)
    Hd, E, k, I = mc.MODELS[case["model"]]
    T = case["T"]
    idx, rw = mc.route(T, E, k)
    rw = rw.to(dt)
    x = torch.randn(T, Hd, device="cuda", dtype=dt)
    g = torch.Generator(device="cuda").manual_seed(0)
    gu_nk = (torch.randn(E, 2 * I, Hd, device="cuda", generator=g) * 0.02).to(dt)   # HF [E, out, in]
    dn_nk = (torch.randn(E, Hd, I, device="cuda", generator=g) * 0.02).to(dt)
    shape = [T, k, Hd, I, E]
    if case["kind"] == "bf16":
        if dt == torch.bfloat16 and torch.cuda.get_device_capability(0)[0] < 8:
            H.emit(kind="skip", why="bf16 routed needs sm80")
            return
        w_gu, w_dn = gu_nk.transpose(1, 2), dn_nk.transpose(1, 2)   # [E,H,2I], [E,I,H] views (as _bf16_views)
        comp = torch.compile(ref_gather, dynamic=False)

        def unfused():
            os.environ["UNSLOTH_MOE_ROUTED_FUSED"] = "0"
            try:
                return mr.routed_bf16_moe(x, idx, rw, w_gu, w_dn, None, None, mr.ACT_SILU, False, out_dtype=dt)
            finally:
                os.environ.pop("UNSLOTH_MOE_ROUTED_FUSED", None)
        with torch.no_grad():
            arms = dict(
                ours=lambda: mr.routed_bf16_moe(x, idx, rw, w_gu, w_dn, None, None, mr.ACT_SILU, False, out_dtype=dt),
                ours_unfused=unfused,
                gather_eager=lambda: ref_gather(x, idx, rw, w_gu, w_dn),
                gather_compiled=lambda: comp(x, idx, rw, w_gu, w_dn),
            )
            if mc.has_grouped_mm():
                arms["zoo_grouped_mm_path"] = lambda: ref_grouped(x, idx, rw, w_gu, w_dn, E)
            H.interleave(arms, "fwd", shape=shape, host=case.get("host", False))
            o = mr.routed_bf16_moe(x, idx, rw, w_gu, w_dn, None, None, mr.ACT_SILU, False, out_dtype=dt)
            r32 = ref_gather(x.float(), idx, rw.float(), w_gu.float(), w_dn.float())
            H.parity("routed_bf16", o, r32, ref_gather(x, idx, rw, w_gu, w_dn), shape=shape)
            if not case.get("host") and not (mc.small_gpu() and case["model"] != "qwen3_30b_a3b"):
                H.compile_check("routed_bf16_moe", lambda a, i, r: mr.routed_bf16_moe(a, i, r, w_gu, w_dn, None, None, mr.ACT_SILU, False, out_dtype=dt),
                                [(x, idx, rw), (x[: T // 2], idx[: T // 2], rw[: T // 2])])
    else:
        try:
            import bitsandbytes.functional as BF
        except Exception as ex:
            H.emit(kind="skip", why="bitsandbytes missing: %r" % ex)
            return
        state = {"E": E, "act": mr.ACT_SILU, "interleaved": False, "alpha": 1.702, "limit": 7.0}
        deq = {}
        for name, W in (("gate_up", gu_nk), ("down", dn_nk)):
            q, qs = BF.quantize_4bit(W.reshape(-1, W.shape[-1]), blocksize=64, compress_statistics=True, quant_type="nf4",
                                     quant_storage=torch.uint8)
            qs.dtype = dt
            p = types.SimpleNamespace(data=q, quant_state=qs, _original_shape=tuple(W.shape))
            tb = mr._nf4_table(p, q.device)
            if tb is None:
                H.emit(kind="skip", why="_nf4_table declined")
                return
            state[name] = tb
            deq[name] = BF.dequantize_4bit(q, qs).view(W.shape)
            state["_q_" + name] = (q, qs, tuple(W.shape))
        del gu_nk, dn_nk
        H.emit(kind="info", what="nf4_gate", slots=T * k, limit=mr.nf4_slot_limit(E), routed=T * k <= mr.nf4_slot_limit(E))
        w_gu, w_dn = deq["gate_up"].transpose(1, 2), deq["down"].transpose(1, 2)
        with torch.no_grad():
            arms = dict(
                ours=lambda: mr._nf4_routed(state, x, idx, rw, k, None, None, dt),
                zoo_selective_dq_gmm=lambda: mr._nf4_grouped(state, x, idx, rw, k, None, None, dt),
                lb_gather_predequant=lambda: ref_gather(x, idx, rw, w_gu, w_dn),
            )
            mk = import_zoo("unsloth_zoo.temporary_patches.moe_triton_kernels")
            if mc.has_grouped_mm():
                def full_dq():
                    a = mk.nf4_dequant_triton(*state["_q_gate_up"])
                    b = mk.nf4_dequant_triton(*state["_q_down"])
                    return ref_grouped(x, idx, rw, a.transpose(1, 2), b.transpose(1, 2), E)
                arms["zoo_full_dq_gmm_path"] = full_dq
            H.interleave(arms, "fwd", shape=shape, host=case.get("host", False))
            o = mr._nf4_routed(state, x, idx, rw, k, None, None, dt)
            r32 = ref_gather(x.float(), idx, rw.float(), w_gu.float(), w_dn.float())
            H.parity("routed_nf4", o, r32, ref_gather(x, idx, rw, w_gu, w_dn), shape=shape)


if __name__ == "__main__":
    main(CASES, run)
