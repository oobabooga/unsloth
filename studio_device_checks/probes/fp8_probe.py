"""#19665 FP8 checks that need no model.
  kernels: does this torch build run the row-wise fp8 torch._scaled_mm, the tensorwise one, and fp32 output?
  smoke:   Studio's own FP8 smoke probe for this tree, the fp8_gemm path it picked, and which _scaled_mm kinds ran.
usage: fp8_probe.py --out FILE kernels|smoke"""

from __future__ import annotations

import sys

import _dc


def _kernels(rec: dict) -> None:
    import torch

    dev = "cuda:0"
    m, k, n = 32, 64, 48
    a = torch.randn(m, k, device = dev).to(torch.float8_e4m3fn)
    b = torch.randn(n, k, device = dev).to(torch.float8_e4m3fn).t()
    cases = {
        "rowwise_bf16": (torch.ones(m, 1, device = dev), torch.ones(1, n, device = dev), torch.bfloat16),
        "tensorwise_bf16": (torch.ones((), device = dev), torch.ones((), device = dev), torch.bfloat16),
        "tensorwise_fp32": (torch.ones((), device = dev), torch.ones((), device = dev), torch.float32),
    }
    res = {}
    for name, (sa, sb, dt) in cases.items():
        try:
            out = torch._scaled_mm(a, b, scale_a = sa, scale_b = sb, out_dtype = dt)
            out = out[0] if isinstance(out, tuple) else out
            torch.cuda.synchronize()
            res[name] = {"ok": bool(torch.isfinite(out.float()).all()), "dtype": str(out.dtype)}
        except Exception as exc:  # noqa: BLE001
            res[name] = {"ok": False, "error": _dc.err(exc, 400)}
    rec["kernels"] = res
    rec["rowwise_missing"] = (not res["rowwise_bf16"]["ok"]) and res["tensorwise_bf16"]["ok"]


def _smoke(rec: dict) -> None:
    import torch

    calls = {"rowwise": 0, "tensorwise": 0}
    real = torch._scaled_mm

    def counting(a, b, *args, **kwargs):
        sa = kwargs.get("scale_a", args[0] if args else None)
        kind = "rowwise" if getattr(sa, "numel", lambda: 1)() > 1 else "tensorwise"
        calls[kind] += 1
        return real(a, b, *args, **kwargs)

    torch._scaled_mm = counting
    from core.inference import diffusion_transformer_quant as tq

    scheme = getattr(tq, "TQ_FP8", "fp8")
    fn = getattr(tq, "_run_smoke_probe", None)
    try:
        if fn is not None:
            rec["smoke_probe"] = fn(scheme, "cuda:0")
            rec["smoke_fn"] = "_run_smoke_probe"
        else:
            rec["smoke_probe"] = tq._smoke_probe(scheme, "cuda:0")
            rec["smoke_fn"] = "_smoke_probe"
    except Exception as exc:  # noqa: BLE001
        rec["smoke_probe"] = False
        rec["smoke_error"] = _dc.err(exc)
    # The same quantized linear, called directly, so a failure shows its message.
    try:
        from torchao.quantization import quantize_

        lin = torch.nn.Linear(512, 512, bias = False).to(device = "cuda:0", dtype = torch.bfloat16)
        quantize_(lin, tq._make_quant_config(scheme), filter_fn = tq.make_filter_fn(0))
        with torch.no_grad():
            y = lin(torch.randn(32, 512, device = "cuda:0", dtype = torch.bfloat16))
        torch.cuda.synchronize()
        rec["linear_ok"] = bool(torch.isfinite(y).all())
    except Exception as exc:  # noqa: BLE001
        rec["linear_ok"] = False
        rec["linear_error"] = _dc.err(exc)
    rec["scaled_mm_calls"] = dict(calls)
    try:
        from core.inference import diffusion_fp8_unit_scale as us

        rec["has_gemm_path"] = hasattr(us, "gemm_path")
        if hasattr(us, "gemm_path"):
            rec["gemm_path"] = us.gemm_path(0)
        if hasattr(us, "fallback_cards"):
            rec["fallback_cards"] = sorted(us.fallback_cards())
        if hasattr(us, "stats"):
            rec["unit_scale_stats"] = us.stats()
    except Exception as exc:  # noqa: BLE001
        rec["unit_scale_error"] = _dc.err(exc)


def main() -> None:
    out = _dc.out_path()
    part = sys.argv[3] if len(sys.argv) > 3 else "smoke"
    rec = {"part": part, "tree": _dc.enter_tree(), **_dc.cuda_info()}
    try:
        if not rec.get("cuda_available"):
            rec["not_run"] = "no CUDA device"
        elif part == "kernels":
            _kernels(rec)
        else:
            _smoke(rec)
    except Exception as exc:  # noqa: BLE001
        rec["fatal"] = _dc.err(exc)
        rec["tb"] = _dc.tb()
    _dc.write(out, rec)


if __name__ == "__main__":
    main()
