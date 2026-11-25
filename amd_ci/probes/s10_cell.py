#!/usr/bin/env python3
"""One isolated cell for the ROCm SDK-wheel detection change (zoo A10-4 / A10-5 + core flex gates). Observes only.

  --cell gates   the real torch build facts (torch.__version__, torch.version.hip, arch, Triton) and what every
                 ROCm gate answers on it: zoo fla / flex / fused-loss / moe gates, core prefix_grouper and
                 Inkling 2 gates (core files loaded by path from --core).
  --cell op      fla chunk_gated_delta_rule (vendored, as zoo injects it) float32 fwd + bwd vs an fp64
                 recurrence, at B:T:H:K:V. A HIP fault kills the process; a hang hits the parent's timeout.
JSON to --out, rewritten after each step so a crash leaves the last step on disk.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
import traceback
from pathlib import Path

RES: dict = {"steps": {}}
OUT: Path | None = None


def save():
    OUT.write_text(json.dumps(RES, indent = 1, default = str), encoding = "utf-8")


def step(name, fn):
    RES["current_step"] = name
    save()
    try:
        val = fn()
        RES["steps"][name] = "ok"
        return val
    except Exception as e:  # noqa: BLE001
        RES["steps"][name] = f"ERR {type(e).__name__}: {str(e)[:400]}"
        RES.setdefault("tracebacks", {})[name] = traceback.format_exc()[-3000:]
        return None
    finally:
        save()


def facts():
    import torch
    d = {"torch": str(torch.__version__), "hip": getattr(torch.version, "hip", None),
         "cuda": getattr(torch.version, "cuda", None), "cuda_available": torch.cuda.is_available()}
    if d["cuda_available"]:
        p = torch.cuda.get_device_properties(0)
        d["name"], d["arch"] = p.name, getattr(p, "gcnArchName", None)
        d["capability"] = list(torch.cuda.get_device_capability(0))
    try:
        import triton
        d["triton"] = triton.__version__
    except Exception as e:  # noqa: BLE001
        d["triton"] = "ERR " + repr(e)[:200]
    RES["facts"] = d
    return d


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def cell_gates(core: Path):
    import torch
    facts()
    import unsloth_zoo
    RES["zoo_file"] = unsloth_zoo.__file__
    g: dict = {}
    RES["gates"] = g

    def fla():
        from unsloth_zoo.temporary_patches import fla_vendor as fv
        g["fla_fp32_gfx11_guard"] = bool(fv._rocm_has_fp32_faulting_gpu())
        g["fla_kda_withheld"] = bool(fv._withhold_kda_on_rocm())
        t = torch.zeros(4, device = "cuda", dtype = torch.float32)
        g["fla_fp32_tensor_routed"] = bool(fv._rocm_fp32_on_faulting_gpu(t))
        g["fla_hopper_dev0"] = fv._device_index_is_hopper(0)

    def flex():
        from unsloth_zoo import rocm_flex_static
        g["zoo_flex_static"] = bool(rocm_flex_static.rocm_flex_static_enabled())

    def losses():
        from unsloth_zoo.fused_losses import chunked_logprob, fused_linear, preference_loss
        g["chunked_logprob_fp32_out_gemm"] = bool(chunked_logprob._FP32_OUT_GEMM)
        g["fused_linear_hip_tiles"] = bool(fused_linear._HIP_TILES)
        g["fused_linear_bwd_rows"] = int(fused_linear._BWD_ROWS)
        g["fused_linear_is_hip"] = bool(fused_linear._IS_HIP)
        preference_loss._MM_OUT_DTYPE_OK.clear()
        h = torch.randn(4, 64, device = "cuda", dtype = torch.bfloat16)
        w = torch.randn(32, 64, device = "cuda", dtype = torch.bfloat16)
        preference_loss._linear_float32(h, w, None)
        g["preference_loss_tried_fp32_out_gemm"] = bool(preference_loss._MM_OUT_DTYPE_OK)

    def moe():
        from unsloth_zoo.temporary_patches import moe_utils
        grad = torch.zeros(4, 4, device = "cuda", dtype = torch.bfloat16)
        g["moe_fp32_out_gemm"] = bool(moe_utils._mm_fp32_out_ok(grad))
        g["moe_torch_is_cuda_build"] = bool(moe_utils._TORCH_IS_CUDA_BUILD)

    def core_gates():
        pg = _load(core / "unsloth" / "utils" / "prefix_grouper_kernel.py", "_s10_prefix_grouper_kernel")
        RES["core_prefix_file"] = pg.__file__
        g["core_prefix_flex_static"] = bool(pg._ROCM_FLEX_STATIC)
        ink = _load(core / "unsloth" / "models" / "inkling2_flex.py", "_s10_inkling2_flex")
        RES["core_inkling_file"] = ink.__file__
        import os
        os.environ.pop(ink.ENV, None)
        os.environ.pop(ink.MIN_LEN_ENV, None)
        g["core_inkling_auto_takes_flex"] = bool(ink._wants_flex(1024, 1024, 0, 0, "cuda"))

    for name, fn in (("fla", fla), ("flex", flex), ("losses", losses), ("moe", moe), ("core", core_gates)):
        step(name, fn)


def ref_gdr(q, k, v, g, beta, scale):
    import torch
    q, k, v, g, beta = (x.double() for x in (q, k, v, g, beta))
    q = q / q.norm(dim = -1, keepdim = True).clamp_min(1e-6)
    k = k / k.norm(dim = -1, keepdim = True).clamp_min(1e-6)
    B, T, H, K = q.shape
    S = torch.zeros(B, H, K, v.shape[-1], dtype = torch.float64, device = q.device)
    outs = []
    for t in range(T):
        S = S * g[:, t].exp()[..., None, None]
        u = beta[:, t][..., None] * (v[:, t] - torch.einsum("bhk,bhkv->bhv", k[:, t], S))
        S = S + k[:, t][..., :, None] * u[..., None, :]
        outs.append(scale * torch.einsum("bhk,bhkv->bhv", q[:, t], S))
    return torch.stack(outs, 1)


def cell_op(shape: str):
    import torch
    facts()
    B, T, H, K, V = (int(x) for x in shape.split(":"))
    import unsloth_zoo
    RES["zoo_file"] = unsloth_zoo.__file__
    from unsloth_zoo.temporary_patches import fla_vendor as fv
    fv.patch_vendor_fla()
    import fla
    import fla.ops.gated_delta_rule as gdr
    RES["fla_file"] = fla.__file__
    fn = gdr.chunk_gated_delta_rule
    RES["torch_path_mark"] = getattr(fn, fv._ROCM_FP32_MARK, None)
    gen = torch.Generator().manual_seed(0)
    mk = lambda *s: torch.randn(*s, generator = gen)  # noqa: E731
    q, k, v = mk(B, T, H, K), mk(B, T, H, K), mk(B, T, H, V)
    g = torch.nn.functional.logsigmoid(mk(B, T, H)) * 0.5
    beta = torch.rand(B, T, H, generator = gen)
    do = mk(B, T, H, V) * 0.1
    scale = K ** -0.5
    leaves = [x.to("cuda", torch.float32).requires_grad_() for x in (q, k, v, g, beta)]
    RES["stage"] = "fwd"
    save()
    t0 = time.time()
    o, _ = fn(*leaves, scale = scale, use_qk_l2norm_in_kernel = True, output_final_state = False)
    torch.cuda.synchronize()
    RES["stage"] = "bwd"
    save()
    o.backward(do.to("cuda"))
    torch.cuda.synchronize()
    RES["seconds"] = round(time.time() - t0, 2)
    ref = [x.double().cuda().requires_grad_() for x in (q, k, v, g, beta)]
    o_ref = ref_gdr(*ref, scale)
    o_ref.backward(do.double().cuda())
    err = {"o": {"max_abs": float((o.double() - o_ref).abs().max()), "finite": bool(torch.isfinite(o).all())}}
    for name, a, b in zip(("dq", "dk", "dv", "dg", "dbeta"), leaves, ref):
        err[name] = {"max_abs": float((a.grad.double() - b.grad).abs().max()), "finite": bool(torch.isfinite(a.grad).all())}
    RES["err_vs_fp64"] = err
    RES["torch_path_calls"] = fv._rocm_fp32_state.get("calls")
    RES["stage"] = "done"
    save()


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required = True)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--core", type = Path)
    ap.add_argument("--shape", default = "3:6:32:128:128")
    a = ap.parse_args()
    OUT = a.out
    save()
    if a.cell == "gates":
        cell_gates(a.core)
        RES["stage"] = "done"
    else:
        cell_op(a.shape)
    save()
    try:
        import torch
        torch.cuda.synchronize()  # Windows HIP: drain queued work before exit or the process hangs in ntdll.
    except Exception:  # noqa: BLE001
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
