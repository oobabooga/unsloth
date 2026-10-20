"""MXFP4 (gpt-oss experts / MXFP4 compressed Linear): zoo mxfp4_gemm.py + mxfp4_dequant.py.

Entry points:
  mxfp4_expert_grouped_mm(x, blocks, scales, counts, transpose_b)  <- moe_utils._mxfp4_base_grouped_mm
      (gpt-oss MXFP4 experts, bf16 only, auto below UNSLOTH_MXFP4_FUSED_MAX_ROWS=192 rows/expert; backward = dX only)
  mxfp4_grouped_mm_compiled                                         <- same, inside compiled regions
  mxfp4_matmul(x, blocks, scales)                                   <- unsloth/models/mxfp4_compressed_linear.py
  mxfp4_dequantize(blocks, scales, transpose=True)                  <- gpt_oss.py / mxfp4.py / stacked experts
References: zoo's own fallback (Triton dequant + torch._grouped_mm), torch._grouped_mm / per-expert cuBLAS on
pre-dequantized bf16 weights (no-dequant lower bound), torch.compile of the plain dequant math.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import *
import moe_common as mc

H_, E_, K_, I_ = 2880, 32, 4, 2880
CASES = [
    # grouped (gpt-oss-20b): T tokens, top-4 -> M = 4T rows
    dict(name="grouped_T4096", kind="grouped", T=4096),     # 512 rows/expert: zoo auto -> dequant + cuBLAS
    dict(name="grouped_T512", kind="grouped", T=512),       # 64 rows/expert: fused
    dict(name="grouped_T8", kind="grouped", T=8, skip_small=True),           # 1 row/expert: MMA path
    dict(name="grouped_T1_host", kind="grouped", T=1, host=True),  # gemv path, launch overhead
    dict(name="dequant_gate_up", kind="dequant", R=2 * I_, Kd=H_),
    dict(name="dequant_down", kind="dequant", R=H_, Kd=I_),
    dict(name="dense_M1_4096x4096", kind="dense", M=1, N=4096, K=4096, host=True),
    dict(name="dense_M16_14336x4096", kind="dense", M=16, N=14336, K=4096, skip_small=True),
    dict(name="dense_M64_4096x4096", kind="dense", M=64, N=4096, K=4096),    # zoo/unsloth fused limit (64 rows)
    dict(name="dense_M256_4096x4096", kind="dense", M=256, N=4096, K=4096),  # above the limit: gate check
]


def rand_mx(shape_rows, G, device="cuda", seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    blocks = torch.randint(0, 256, (*shape_rows, G, 16), dtype=torch.uint8, device=device, generator=g)
    scales = torch.randint(118, 128, (*shape_rows, G), dtype=torch.uint8, device=device, generator=g)
    return blocks, scales


def run(case, dtype, H):
    global torch
    import torch
    mg = import_zoo("unsloth_zoo.mxfp4_gemm")
    md = import_zoo("unsloth_zoo.mxfp4_dequant")
    dt = H.dtype
    kind = case["kind"]
    small = mc.small_gpu()
    mc.tune_budget(H)
    if small and case.get("skip_small"):
        H.emit(kind="skip", why="small GPU (<20 GB): case dropped to fit the T4 time budget")
        return
    if kind in ("grouped", "dense"):
        avail = mg.mxfp4_grouped_mm_available()
        dense_ok = mg.mxfp4_gemm_available(dtype=dt) if kind == "dense" else None
        H.emit(kind="info", what="availability", grouped=avail, dense=dense_ok,
               asm=mg._asm_ok(torch.device("cuda", 0)), big_tiles=mg._big_tiles(torch.device("cuda", 0)),
               has_grouped_mm=mc.has_grouped_mm(), zoo_moe_gate_bf16_only=True)
        if not avail:
            H.emit(kind="skip", why="mxfp4_grouped_mm_available()=False")
            return
    if kind == "grouped":
        T = case["T"]
        idx, w = mc.route(T, E_, K_)
        _, tok, counts, offs = mc.sort_routing(idx, E_)
        M = T * K_
        cl = counts.tolist()
        x = torch.randn(T, H_, device="cuda", dtype=dt)[tok].contiguous()
        H.emit(kind="info", what="routing", M=M, rows_per_expert=M / E_, max_count=max(cl), min_count=min(cl),
               zoo_fused_auto=(M < 192 * E_), gemv=mg._gemv_rows(M, E_, True))
        for proj, (R, Kd) in dict(gate_up=(2 * I_, H_), down=(H_, I_)).items():
            blocks, scales = rand_mx((E_, R), Kd // 32, seed=1 if proj == "gate_up" else 2)
            xin = x if proj == "gate_up" else torch.randn(M, Kd, device="cuda", dtype=dt)
            wd = md.mxfp4_dequantize_torch(blocks, scales, dtype=dt, transpose=True)  # (E, K, N)
            shape = [M, Kd, R, E_]
            comp_dq = torch.compile(lambda b, s: mc.mxfp4_dequant_plain(b, s, dt).transpose(-1, -2), dynamic=False)
            arms = dict(
                ours=lambda: mg.mxfp4_grouped_mm(xin, blocks, scales, counts, transpose_b=True),
                zoo_fallback_dq_gmm=lambda: torch._grouped_mm(xin, md.mxfp4_dequantize(blocks, scales, dtype=dt, transpose=True), offs=offs),
                compiled_dq_gmm=lambda: torch._grouped_mm(xin, comp_dq(blocks, scales), offs=offs),
                lb_gmm_predequant=lambda: torch._grouped_mm(xin, wd, offs=offs),
            )
            if small and T != 512:
                arms.pop("compiled_dq_gmm")  # T4 CPU compile time
            if M <= 64:
                arms["lb_loop_predequant"] = lambda: mc.loop_mm(xin, wd, cl)
            H.interleave(arms, "fwd_" + proj, shape=shape, host=case.get("host", False))
            # dX backward (transpose_b=False): gy [M, R] @ P[e] -> [M, Kd]
            if T >= 512:
                gy = torch.randn(M, R, device="cuda", dtype=dt)
                wdt = wd.transpose(-1, -2)  # (E, N, K) view
                wdt_c = wdt.contiguous()
                H.interleave(dict(
                    ours=lambda: mg.mxfp4_grouped_mm(gy, blocks, scales, counts, transpose_b=False),
                    zoo_fallback_dq_gmm=lambda: torch._grouped_mm(gy, md.mxfp4_dequantize(blocks, scales, dtype=dt, transpose=True).transpose(-1, -2), offs=offs),
                    lb_gmm_predequant=lambda: torch._grouped_mm(gy, wdt_c, offs=offs),
                ), "bwd_dX_" + proj, shape=shape)
            torch.cuda.empty_cache()
            # parity
            o = mg.mxfp4_grouped_mm(xin, blocks, scales, counts, transpose_b=True)
            wd32 = md.mxfp4_dequantize_torch(blocks, scales, dtype=torch.float32, transpose=True)
            r32 = mc.loop_mm(xin.float(), wd32, cl)
            try:
                rl = torch._grouped_mm(xin, wd, offs=offs)
            except Exception:
                rl = mc.loop_mm(xin, wd, cl)
            H.parity("fwd_" + proj, o, r32, rl, shape=shape)
            del wd
        if T not in ((512,) if small else (512, 1)):
            return
        # compile check of the compiled-region entry
        blocks, scales = rand_mx((E_, 2 * I_), H_ // 32, seed=1)
        x2 = x[: M // 2].contiguous()
        _, _, counts2, _ = mc.sort_routing(idx[: T // 2] if T > 1 else idx, E_)
        H.compile_check("mxfp4_grouped_mm_compiled",
                        lambda a, c: mg.mxfp4_grouped_mm_compiled(a, blocks, scales, c, True),
                        [(x, counts), (x2 if T > 1 else x, counts2)])
    elif kind == "dequant":
        R, Kd = case["R"], case["Kd"]
        Ed = 8 if small else E_   # fp32 reference + compiled intermediates of all 32 experts exceed a T4
        H.emit(kind="info", what="shape_used", experts=Ed)
        blocks, scales = rand_mx((Ed, R), Kd // 32)
        H.emit(kind="info", what="availability", kernel=md.mxfp4_kernel_available(dtype=dt))
        comp = torch.compile(lambda b, s: mc.mxfp4_dequant_plain(b, s, dt).transpose(-1, -2).contiguous(), dynamic=False)
        comp_nt = torch.compile(lambda b, s: mc.mxfp4_dequant_plain(b, s, dt), dynamic=False)
        shape = [Ed, R, Kd]
        H.interleave(dict(
            ours=lambda: md.mxfp4_dequantize(blocks, scales, dtype=dt, transpose=True),
            eager=lambda: md.mxfp4_dequantize_torch(blocks, scales, dtype=dt, transpose=True),
            compiled=lambda: comp(blocks, scales),
        ), "transpose", shape=shape)
        H.interleave(dict(
            ours=lambda: md.mxfp4_dequantize(blocks, scales, dtype=dt, transpose=False),
            eager=lambda: md.mxfp4_dequantize_torch(blocks, scales, dtype=dt, transpose=False),
            compiled=lambda: comp_nt(blocks, scales),
        ), "plain", shape=shape)
        o = md.mxfp4_dequantize(blocks, scales, dtype=dt, transpose=True)
        r = md.mxfp4_dequantize_torch(blocks, scales, dtype=torch.float32, transpose=True)
        H.parity("dequant_T", o, r, md.mxfp4_dequantize_torch(blocks, scales, dtype=dt, transpose=True), shape=shape)
        H.emit(kind="parity_exact", what="dequant_T_vs_torch_same_dtype",
               equal=bool(torch.equal(o, md.mxfp4_dequantize_torch(blocks, scales, dtype=dt, transpose=True))))
    elif kind == "dense":
        M, N, K = case["M"], case["N"], case["K"]
        if not mg.mxfp4_gemm_available(dtype=dt):
            H.emit(kind="skip", why="mxfp4_gemm_available(%s)=False" % dtype)
            return
        blocks, scales = rand_mx((N,), K // 32)
        x = torch.randn(M, K, device="cuda", dtype=dt)
        wd = md.mxfp4_dequantize_torch(blocks, scales, dtype=dt)  # (N, K)
        comp_dq = torch.compile(lambda b, s: mc.mxfp4_dequant_plain(b, s, dt), dynamic=False)
        shape = [M, N, K]
        H.interleave(dict(
            ours=lambda: mg.mxfp4_matmul(x, blocks, scales),
            dq_triton_linear=lambda: torch.nn.functional.linear(x, md.mxfp4_dequantize(blocks.view(1, N, K // 32, 16), scales.view(1, N, K // 32), dtype=dt).view(N, K)),
            compiled_dq_linear=lambda: torch.nn.functional.linear(x, comp_dq(blocks, scales)),
            lb_cublas_predequant=lambda: torch.nn.functional.linear(x, wd),
        ), "fwd", shape=shape, host=case.get("host", False))
        o = mg.mxfp4_matmul(x, blocks, scales)
        r32 = x.float() @ md.mxfp4_dequantize_torch(blocks, scales, dtype=torch.float32).t()
        H.parity("dense", o, r32, torch.nn.functional.linear(x, wd), shape=shape)
        H.emit(kind="info", what="gate", unsloth_fused_used=(M <= 64))
        if M == 1:
            H.compile_check("mxfp4_matmul", lambda a: mg.mxfp4_matmul(a, blocks, scales), [(x,), (torch.randn(M + 3, K, device="cuda", dtype=dt),)])


if __name__ == "__main__":
    main(CASES, run)
