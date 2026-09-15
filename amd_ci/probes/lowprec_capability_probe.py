#!/usr/bin/env python3
"""Probe: which low-precision paths exist on this GPU, and what does Studio pick?

Observes only. Every question below is recorded as what happened, including the
exact exception text, and nothing here decides whether an answer is good. The
difference between "the dtype does not exist", "the dtype exists and the kernel
refuses it" and "it ran and returned garbage" is the whole point of the exercise,
so a bare boolean would destroy the result; each attempt records its outcome and
its error verbatim.

Needs no model weights and no Hugging Face credential: every tensor here is
randn. That is deliberate. These are third-party machines, so the probe is
written to have nothing worth stealing on it.

Pairs with criteria/lowprec_capability.py.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path

# GEMM shapes we actually care about: a square reference, and a Z-Image/Qwen-DiT
# shaped feed-forward projection. Not a benchmark study; enough to say what the
# fast path on this hardware is.
GEMM_SHAPES = ((4096, 4096, 4096), (4096, 3072, 12288))


def err(exc: BaseException) -> dict:
    """An exception as an observation. The text matters: 'not supported' and
    'not compiled in' are different answers and only the string tells them apart."""
    return {
        "type": type(exc).__name__,
        "message": str(exc),
        "tail": traceback.format_exc().strip().splitlines()[-3:],
    }


def attempt(fn) -> dict:
    """Run fn, record either its value or its exception. Never raises."""
    try:
        return {"ok": True, "value": fn()}
    except BaseException as exc:  # noqa: BLE001 - recording, not handling
        return {"ok": False, "error": err(exc)}


# --------------------------------------------------------------------------
# 1. device identity
# --------------------------------------------------------------------------

def kfd_gfx_target_versions() -> list[int]:
    """gfx_target_version straight off the KFD topology. Expected 110501 on
    gfx1151 (11 * 10000 + 5 * 100 + 1). Read from sysfs rather than from torch
    so it is independent of whatever HSA override may be in play."""
    out: list[int] = []
    for path in sorted(glob.glob("/sys/class/kfd/kfd/topology/nodes/*/properties")):
        try:
            text = Path(path).read_text(encoding = "utf-8", errors = "replace")
        except OSError:
            continue
        for line in text.splitlines():
            if line.startswith("gfx_target_version"):
                try:
                    v = int(line.split()[1])
                except (IndexError, ValueError):
                    continue
                if v:
                    out.append(v)
    return out


def rocm_version_files() -> dict:
    found = {}
    for path in ("/opt/rocm/.info/version", "/opt/rocm/.info/version-dev"):
        try:
            found[path] = Path(path).read_text(encoding = "utf-8").strip()
        except OSError:
            pass
    return found


def device_identity() -> dict:
    obs: dict = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        # Any HSA_OVERRIDE here means torch is being told it is on a different
        # part, and every arch-keyed answer below would be about that lie.
        "hsa_env": {k: v for k, v in os.environ.items()
                    if k.startswith(("HSA_", "HIP_", "ROCR_", "PYTORCH_ROCM"))},
        "kfd_gfx_target_version": kfd_gfx_target_versions(),
        "rocm_version_files": rocm_version_files(),
    }
    obs["rocminfo_gfx"] = attempt(lambda: sorted({
        line.split(":")[-1].strip()
        for line in subprocess.run(["rocminfo"], capture_output = True, text = True,
                                   timeout = 120).stdout.splitlines()
        if "gfx" in line and "Name:" in line}))
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        obs["torch_import"] = err(exc)
        return obs

    obs["torch_version"] = torch.__version__
    obs["torch_version_hip"] = torch.version.hip
    obs["torch_version_cuda"] = torch.version.cuda
    obs["cuda_available"] = attempt(torch.cuda.is_available)
    obs["device_count"] = attempt(torch.cuda.device_count)
    # get_device_capability is a CUDA compute capability on NVIDIA. On ROCm torch
    # synthesises one from the gfx target, and Studio's ladder compares it against
    # sm_100 / sm_89 / sm_80 floors, so what it returns here decides which tier
    # gfx1151 lands in. Recorded raw; the criteria module says what it implies.
    obs["device_capability"] = attempt(lambda: list(torch.cuda.get_device_capability(0)))
    obs["device_name"] = attempt(lambda: torch.cuda.get_device_name(0))

    def props() -> dict:
        p = torch.cuda.get_device_properties(0)
        keep = ("name", "gcnArchName", "major", "minor", "total_memory",
                "multi_processor_count", "warp_size", "is_integrated",
                "is_multi_gpu_board", "L2_cache_size", "max_threads_per_multi_processor")
        return {k: getattr(p, k) for k in keep if hasattr(p, k)}

    obs["device_properties"] = attempt(props)
    return obs


# --------------------------------------------------------------------------
# 2. float8: which dtypes exist, and which survive a matmul
# --------------------------------------------------------------------------

FP8_DTYPES = ("float8_e4m3fn", "float8_e5m2", "float8_e4m3fnuz", "float8_e5m2fnuz")


def probe_float8() -> dict:
    """For each fp8 dtype: does it exist, does a tensor CONSTRUCT, and does
    torch._scaled_mm accept it, per-tensor and per-row.

    Per-row matters on its own: Studio's fp8 config REQUIRES PerRow granularity
    (a per-tensor scale collapses Z-Image's outlier rows), so a build with
    per-tensor scaled_mm and no per-row support is an fp8 that Studio cannot use.
    """
    import torch
    out: dict = {}
    for name in FP8_DTYPES:
        entry: dict = {"exists": hasattr(torch, name)}
        if not entry["exists"]:
            out[name] = entry
            continue
        dt = getattr(torch, name)
        entry["dtype_repr"] = str(dt)
        # Construction on the GPU is a separate question from the matmul: a dtype
        # can exist in the python bindings with no device-side storage support.
        entry["construct_cuda"] = attempt(
            lambda dt = dt: str(torch.zeros(32, 32, device = "cuda").to(dt).dtype))
        entry["construct_cpu"] = attempt(
            lambda dt = dt: str(torch.zeros(32, 32).to(dt).dtype))

        # M,K x K,N with N-major B, which is what _scaled_mm demands.
        def scaled_mm(dt = dt, per_row = False):
            m, k, n = 256, 256, 256
            a = torch.randn(m, k, device = "cuda", dtype = torch.bfloat16).to(dt)
            b = torch.randn(n, k, device = "cuda", dtype = torch.bfloat16).to(dt).t()
            if per_row:
                sa = torch.ones(m, 1, device = "cuda", dtype = torch.float32)
                sb = torch.ones(1, n, device = "cuda", dtype = torch.float32)
            else:
                sa = torch.ones((), device = "cuda", dtype = torch.float32)
                sb = torch.ones((), device = "cuda", dtype = torch.float32)
            r = torch._scaled_mm(a, b, scale_a = sa, scale_b = sb,
                                 out_dtype = torch.bfloat16)
            torch.cuda.synchronize()
            return {"shape": list(r.shape), "dtype": str(r.dtype),
                    "finite": bool(torch.isfinite(r).all().item())}

        entry["scaled_mm_per_tensor"] = attempt(lambda dt = dt: scaled_mm(dt, False))
        entry["scaled_mm_per_row"] = attempt(lambda dt = dt: scaled_mm(dt, True))
        out[name] = entry
    return out


# --------------------------------------------------------------------------
# 3. any 4-bit path at all
# --------------------------------------------------------------------------

FP4_DTYPE_NAMES = ("float4_e2m1fn_x2", "float4_e2m1fn", "uint4", "int4")


def probe_fp4() -> dict:
    """Does a 4-bit float exist in this torch, does torchao import on ROCm, and
    what do the mx_formats entry points DO when asked for NVFP4 or MXFP4?

    Failure is expected. The failure MODE is the result: a clean refusal, a
    silent fallback to a slow dense path, or a crash are three different answers
    and only the third makes Studio unsafe here. So the quantise attempt records
    whether the module type actually CHANGED, which is how a silent no-op is
    told apart from a real quantisation.
    """
    import torch
    out: dict = {"torch_fp4_dtypes": {n: hasattr(torch, n) for n in FP4_DTYPE_NAMES}}

    try:
        import torchao
        out["torchao_version"] = getattr(torchao, "__version__", "unknown")
        out["torchao_file"] = getattr(torchao, "__file__", None)
    except Exception as exc:  # noqa: BLE001
        out["torchao_import"] = err(exc)
        return out

    out["quantize_import"] = attempt(
        lambda: str(__import__("torchao.quantization", fromlist = ["quantize_"]).quantize_))

    def mx_entry(attr: str):
        mod = __import__("torchao.prototype.mx_formats", fromlist = [attr])
        return str(getattr(mod, attr))

    for attr in ("NVFP4DynamicActivationNVFP4WeightConfig",
                 "MXDynamicActivationMXWeightConfig",
                 "MXFP4InferenceConfig"):
        out[f"entrypoint_{attr}"] = attempt(lambda a = attr: mx_entry(a))

    def run_scheme(kind: str) -> dict:
        """Build the config Studio would build, apply it to a real Linear, and
        try a forward. Records the type before and after so a no-op is visible."""
        from torchao.quantization import quantize_
        mx = __import__("torchao.prototype.mx_formats",
                        fromlist = ["NVFP4DynamicActivationNVFP4WeightConfig",
                                    "MXDynamicActivationMXWeightConfig"])
        if kind == "nvfp4":
            cls = mx.NVFP4DynamicActivationNVFP4WeightConfig
            try:
                cfg = cls(use_triton_kernel = False)
            except TypeError:
                cfg = cls()
        else:
            cls = mx.MXDynamicActivationMXWeightConfig
            try:
                cfg = cls(activation_dtype = torch.float4_e2m1fn_x2,
                          weight_dtype = torch.float4_e2m1fn_x2)
            except (TypeError, AttributeError) as exc:
                return {"config_error": err(exc)}
        lin = torch.nn.Linear(256, 256, bias = False).to("cuda", torch.bfloat16)
        before = type(lin.weight.data).__name__
        rec: dict = {"config": str(cfg), "weight_type_before": before}
        quantize_(lin, cfg)
        after = type(lin.weight.data).__name__
        rec["weight_type_after"] = after
        # The load-bearing distinction: quantize_ can return cleanly having done
        # NOTHING, which reads as success and leaves the model dense.
        rec["weight_type_changed"] = after != before
        x = torch.randn(64, 256, device = "cuda", dtype = torch.bfloat16)
        y = lin(x)
        torch.cuda.synchronize()
        rec["forward_shape"] = list(y.shape)
        rec["forward_finite"] = bool(torch.isfinite(y).all().item())
        return rec

    out["apply_nvfp4"] = attempt(lambda: run_scheme("nvfp4"))
    out["apply_mxfp4"] = attempt(lambda: run_scheme("mxfp4"))
    return out


# --------------------------------------------------------------------------
# 4. what DOES work
# --------------------------------------------------------------------------

def timed_gemm(fn, warmup: int = 3, iters: int = 10) -> dict:
    import torch
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return {"seconds_per_iter": (time.perf_counter() - t0) / iters}


def probe_working_paths() -> dict:
    """int8 and the 16-bit paths. Short on purpose: this states what the fast
    path on this hardware IS, it does not characterise it."""
    import torch
    out: dict = {}

    def int_mm():
        a = torch.randint(-8, 8, (256, 256), device = "cuda", dtype = torch.int8)
        b = torch.randint(-8, 8, (256, 256), device = "cuda", dtype = torch.int8)
        r = torch._int_mm(a, b)
        torch.cuda.synchronize()
        return {"dtype": str(r.dtype), "shape": list(r.shape),
                "sum": int(r.sum().item())}

    out["torch_int_mm"] = attempt(int_mm)

    def safe_int_mm():
        from torchao.kernel import safe_int_mm
        a = torch.randint(-8, 8, (256, 256), device = "cuda", dtype = torch.int8)
        b = torch.randint(-8, 8, (256, 256), device = "cuda", dtype = torch.int8)
        r = safe_int_mm(a, b)
        torch.cuda.synchronize()
        return {"dtype": str(r.dtype), "shape": list(r.shape)}

    out["torchao_safe_int_mm"] = attempt(safe_int_mm)

    gemms: dict = {}
    for m, k, n in GEMM_SHAPES:
        shape_key = f"{m}x{k}x{n}"
        gemms[shape_key] = {}
        for dtype_name in ("float16", "bfloat16"):
            dt = getattr(torch, dtype_name)

            def one(dt = dt, m = m, k = k, n = n):
                a = torch.randn(m, k, device = "cuda", dtype = dt)
                b = torch.randn(k, n, device = "cuda", dtype = dt)
                r = timed_gemm(lambda: a @ b)
                # 2*M*N*K flops; TFLOP/s is the only number worth quoting.
                r["tflops"] = (2.0 * m * n * k) / r["seconds_per_iter"] / 1e12
                del a, b
                torch.cuda.empty_cache()
                return r

            gemms[shape_key][dtype_name] = attempt(one)
    out["gemm"] = gemms

    def int8_gemm():
        # int8 at a real shape, to put the int8 path on the same axis as fp16.
        m, k, n = 4096, 4096, 4096
        a = torch.randint(-8, 8, (m, k), device = "cuda", dtype = torch.int8)
        b = torch.randint(-8, 8, (k, n), device = "cuda", dtype = torch.int8)
        r = timed_gemm(lambda: torch._int_mm(a, b))
        r["tflops"] = (2.0 * m * n * k) / r["seconds_per_iter"] / 1e12
        del a, b
        torch.cuda.empty_cache()
        return r

    out["gemm_int8_4096"] = attempt(int8_gemm)
    return out


# --------------------------------------------------------------------------
# 5. Studio's own quant selection, imported from THIS checkout
# --------------------------------------------------------------------------

def probe_studio(checkout: Path) -> dict:
    """What does this checkout's selector resolve to on this device?

    Imported the way the app imports it, from the state's own worktree, so base
    and head answer for their own code and not for whichever was imported first.
    """
    import torch
    out: dict = {"checkout": str(checkout)}
    backend = checkout / "studio" / "backend"
    if not backend.is_dir():
        out["error"] = f"no backend at {backend}"
        return out

    sys.path.insert(0, str(backend))
    for stale in [m for m in sys.modules
                  if m.startswith(("core.", "utils.")) or m in ("core", "utils")]:
        del sys.modules[stale]

    def load():
        import core.inference.diffusion_transformer_quant as q  # noqa: PLC0415
        return q

    loaded = attempt(load)
    out["module_import"] = {"ok": loaded["ok"]}
    if not loaded["ok"]:
        out["module_import"]["error"] = loaded["error"]
        return out
    q = loaded["value"]
    out["module_file"] = q.__file__

    # The target must be SHAPED LIKE THE CALLER'S or the answer is about the
    # probe. dense_transformer_supported does `getattr(target, "device", None)
    # != "cuda"`, a comparison against the STRING "cuda": a real tensor's
    # .device is a torch.device, which is never string-equal, so a tensor target
    # fails at the first line and every downstream None would be an artefact of
    # the probe rather than a fact about the GPU. The loader passes an object
    # carrying a device string and a dtype, so that is what is passed here. The
    # tensor is kept alongside precisely to show the two answers differ.
    from types import SimpleNamespace
    target = SimpleNamespace(device = "cuda", dtype = torch.bfloat16)
    tensor_target = torch.zeros(8, 8, device = "cuda", dtype = torch.bfloat16)

    out["dense_transformer_supported"] = attempt(
        lambda: bool(q.dense_transformer_supported(target)))
    out["dense_transformer_supported[tensor target]"] = attempt(
        lambda: bool(q.dense_transformer_supported(tensor_target)))
    # Which of the three early returns fired. This is the whole explanation for
    # every None below, and without it the inventory says "nothing works" without
    # saying whether that is the hardware, the platform policy or a stub.
    out["torch_is_rocm"] = attempt(lambda: bool(q.torch_is_rocm()))
    out["is_stubbed_torchao"] = attempt(lambda: bool(q.is_stubbed("torchao")))
    # The specific, correct AMD message. Recorded because it is what a user
    # SHOULD see; whether any caller actually surfaces it is a separate question.
    out["dense_transformer_unsupported_reason"] = attempt(
        lambda: q.dense_transformer_unsupported_reason(target))
    out["torchao_unavailable_reason"] = attempt(q.torchao_unavailable_reason)
    out["capability_tuple"] = attempt(lambda: list(q._capability() or []))
    out["is_consumer_gpu"] = attempt(lambda: bool(q._is_consumer_gpu("cuda")))
    out["auto_ladder"] = attempt(
        lambda: [[list(f), list(s)] for f, s in q._AUTO_LADDER])

    # The headline: what does `auto` resolve to here, and what would it have
    # taken next. Recorded per family because the deny list is family-keyed.
    for family in (None, "qwen-image", "z-image"):
        key = family or "no-family"
        out[f"auto_scheme[{key}]"] = attempt(
            lambda f = family: q.select_transformer_quant_scheme(target, "auto", family = f))
        out[f"auto_candidates[{key}]"] = attempt(
            lambda f = family: list(q.auto_scheme_candidates(target, family = f)))

    # The single most valuable result: an explicit request a user can type. It
    # must produce a clean answer, not a traceback. `None` plus an explanation
    # is the clean decline; an exception escaping here is the unsafe case.
    for scheme in ("nvfp4", "mxfp8", "fp8", "int8"):
        out[f"explicit[{scheme}]"] = attempt(
            lambda s = scheme: q.select_transformer_quant_scheme(target, s, family = None))
        out[f"explain[{scheme}]"] = attempt(
            lambda s = scheme: q.explain_unusable_scheme(None, s))
        out[f"scheme_supported[{scheme}]"] = attempt(
            lambda s = scheme: bool(q._scheme_supported(s, "cuda")))

    # Host classification: does Studio call this box rocm?
    def hardware_accelerator():
        import utils.hardware.hardware as hw  # noqa: PLC0415
        rec = {"file": hw.__file__}
        for attr in ("IS_ROCM", "IS_CUDA"):
            if hasattr(hw, attr):
                rec[attr] = bool(getattr(hw, attr))
        for fn in ("get_accelerator", "accelerator_kind", "detect_hardware"):
            if hasattr(hw, fn):
                rec[fn] = attempt(lambda f = fn: getattr(hw, f)())
        rec["gpu_summary"] = attempt(hw.get_gpu_summary)
        return rec

    out["hardware"] = attempt(hardware_accelerator)
    # Named in the brief; it does not exist in the tree, and recording that is
    # more useful than an AttributeError buried in a traceback.
    out["dense_quant_host_capable_present"] = hasattr(q, "dense_quant_host_capable")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True, type = Path)
    ap.add_argument("--out", required = True, type = Path)
    args = ap.parse_args()

    obs: dict = {"state": args.state}
    obs["identity"] = attempt(device_identity)
    obs["float8"] = attempt(probe_float8)
    obs["fp4"] = attempt(probe_fp4)
    obs["working"] = attempt(probe_working_paths)
    obs["studio"] = attempt(lambda: probe_studio(args.checkout))

    args.out.parent.mkdir(parents = True, exist_ok = True)
    args.out.write_text(json.dumps(obs, indent = 2, default = str), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
