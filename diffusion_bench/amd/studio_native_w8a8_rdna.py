"""Studio native int8 W8A8 linear on ROCm RDNA: Triton rotate+quant + int8 GEMM with the row-scale epilogue.

    python kernel_oracle.py kernel_oracle_specs/studio_native_w8a8_rdna.py --spec-arg backend=<tree>/studio/backend \\
        --inductor-config emulate_precision_casts=True

ours  = NativeWeightOnlyLinear (act_int8, ConvRot g256) taking the RDNA kernels (two custom ops).
stock = the same layer with UNSLOTH_NATIVE_INT8_RDNA_TRITON=0: eager = the compiled ``_w8a8_rows`` it calls today,
        Inductor = ``_rotate`` + ``_int_mm_rows`` traced, as a compiled model runs it.
ref   = fp64 x @ W.T + b with the unquantised bf16 weight, so both arms carry the same int8 error budget.
"""

import os
import sys


def _layers(torch, cls, k, n, bias, gen):
    lin = torch.nn.Linear(k, n, bias = bias, dtype = torch.bfloat16)
    with torch.no_grad():
        lin.weight.copy_((torch.randn(n, k, generator = gen) * 0.02).to(torch.bfloat16))
        if bias:
            lin.bias.copy_((torch.randn(n, generator = gen) * 0.05).to(torch.bfloat16))
    lin = lin.cuda()
    os.environ["UNSLOTH_NATIVE_INT8_RDNA_TRITON"] = "0"
    stock = cls(lin, "int8", act_int8 = True, rot_group = 256).cuda().eval()
    os.environ.pop("UNSLOTH_NATIVE_INT8_RDNA_TRITON", None)
    ours = cls(lin, "int8", act_int8 = True, rot_group = 256).cuda().eval()
    return lin, stock, ours


def cases(args):
    import torch

    sys.path.insert(0, args["backend"])
    from core.inference import diffusion_native_quant as nq
    from core.inference import diffusion_native_w8a8_rdna as rdna

    if not torch.cuda.is_available() or not rdna.prepare():
        raise SystemExit(f"studio_native_w8a8_rdna: needs a ROCm RDNA card where the probe passes ({rdna.state()})")
    cls = nq.native_linear_class()
    gen = torch.Generator().manual_seed(1234)
    shapes = [("q21_attn_4096x3072x3072", 4096, 3072, 3072, True), ("q21_ff_up_4096x3072x12288", 4096, 3072, 12288, True),
              ("q21_ff_down_4096x12288x3072", 4096, 12288, 3072, True), ("f3_qkv_mlp_4608x3072x9216", 4608, 3072, 9216, False),
              ("odd_rows_4101x3072x3072", 4101, 3072, 3072, True), ("min_rows_17x3072x3072", 17, 3072, 3072, True),
              ("rows_1040x768x48", 1040, 768, 48, True)]  # fmt: skip
    if args.get("small") == "1":
        shapes = [("small_1040x768x256", 1040, 768, 256, True)]
    for name, m, k, n, bias in shapes:
        lin, stock, ours = _layers(torch, cls, k, n, bias, gen)
        assert ours._rdna and not stock._rdna, "ours must take the RDNA kernels, stock must not"
        x = torch.randn(m, k, generator = gen).to(torch.bfloat16)
        x[:: max(1, m // 17)] *= 8  # heavy-tailed rows, as DiT activations
        x = x.cuda()
        ref = x.double() @ lin.weight.double().t()
        if bias:
            ref = ref + lin.bias.double()

        def run_ours(x = x, ours = ours):
            with torch.no_grad():
                return ours(x)

        def run_stock(x = x, stock = stock):
            with torch.no_grad():
                return stock(x)

        yield dict(name = name, ours = run_ours, stock = run_stock, ref = ref, inference = True,
                   note = "int8 error vs the bf16 weight; both arms quantise the same rotated activation")  # fmt: skip
    for dtype in (torch.bfloat16, torch.float16):
        for group in (256, 0):
            x = (torch.randn(4096, 3072, generator = gen) * 2).to(dtype).cuda()
            x[7] = 0
            tag = "bf16" if dtype == torch.bfloat16 else "fp16"

            def rq_ours(x = x, group = group):
                q, s = rdna.rotq(x, group)
                return q.float() * s

            def rq_stock(x = x, group = group):
                q, s = rdna.reference_rotq(x, group)
                return q.float() * s

            xr = x.double()
            if group:
                from core.inference.diffusion_convrot import build_convrot_hadamard

                h = build_convrot_hadamard(group, device = x.device, dtype = torch.float64)
                xr = (xr.reshape(xr.shape[0], -1, group) @ h).reshape(xr.shape)
            yield dict(name = f"rotq_{tag}_g{group}_4096x3072", ours = rq_ours, stock = rq_stock, ref = xr,
                       inference = True, note = "dequantised activation codes vs the fp64 rotated input")  # fmt: skip
