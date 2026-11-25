#!/usr/bin/env python3
"""Item 6: bf16 non-causal attention race on gfx1151, one fresh process per (shape, arm).

  aiter_rdna.py --aiter-dir <A12_WORK>/aiter_rdna --arm aotriton|ck|rdna --shape wan5b|ltx|ltx_long|h3|qwen21|warm --out F

aotriton  torch SDPA as Studio runs it on ROCm: TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1, [B, S, H, D] tensors handed
          over as transpose(1, 2) views (diffusers' layout), FLASH backend forced (so the AOTriton kernel provably ran)
ck        the same call after torch.backends.cuda.preferred_rocm_fa_library("ck"); "unavailable" when the wheel lacks it
rdna      aiter.ops.rdna_attn.encoder_forward (PR #8605): q / k / v as [S, H, D] (B = 1), read in place (no repack:
          each token's heads are contiguous), one segment; JIT build time reported separately (build_s)
Times: 2 warm calls, then --reps calls timed with CUDA events, median. Accuracy: 256 sampled query rows x 3 heads vs
an fp64 softmax reference (max abs, rel L2), plus a sha256 of the whole output. Writes one JSON; exits 0."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics as st
import sys
import time

os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")
SHAPES = {"wan5b": (1, 27280, 24, 128), "ltx": (1, 6144, 32, 128), "ltx_long": (1, 24576, 32, 128),
          "h3": (1, 37720, 56, 128), "qwen21": (1, 4096, 24, 128), "warm": (1, 256, 24, 128)}


def _rocm_env(aiter_dir: str) -> None:
    """pydeps (pybind11, ninja) and the toolchain aiter_setup.sh found, so the harness passes no extra env."""
    work = os.path.dirname(os.path.abspath(aiter_dir))
    deps = os.path.join(work, "pydeps")
    if os.path.isdir(deps):
        sys.path.insert(0, deps)
        os.environ["PATH"] = os.path.join(deps, "bin") + os.pathsep + os.environ.get("PATH", "")
    try:
        s = json.load(open(os.path.join(work, "aiter_setup.json"), encoding = "utf-8"))
        home = s.get("rocm_home")
    except Exception:  # noqa: BLE001 -- during aiter_setup.sh's own warm build the env carries ROCM_HOME
        home = None
    if home and not os.environ.get("ROCM_HOME"):
        os.environ["ROCM_HOME"] = os.environ["ROCM_PATH"] = home
        os.environ["PATH"] = os.path.join(home, "bin") + os.pathsep + os.environ["PATH"]
    os.environ.setdefault("GPU_ARCHS", "gfx1151")


def _import_rdna(aiter_dir: str):
    """aiter.ops.rdna_attn through stub ``aiter`` / ``aiter.ops`` packages: skips aiter/__init__.py, which imports
    every CK / HIP op family; only the jit core and this module load."""
    import importlib.machinery
    import importlib.util
    root = os.path.abspath(aiter_dir)
    for name, sub in (("aiter", "aiter"), ("aiter.ops", "aiter/ops")):
        spec = importlib.machinery.ModuleSpec(name, None, is_package = True)
        spec.submodule_search_locations = [os.path.join(root, sub)]
        sys.modules[name] = importlib.util.module_from_spec(spec)
    from aiter.ops import rdna_attn
    return rdna_attn


def inputs(B, S, H, D, seed = 0):
    import torch
    g = torch.Generator(device = "cuda").manual_seed(1234 + seed)
    # DiT-like: per-channel scales on Q / K, an offset on K and V
    wq = torch.exp(0.4 * torch.randn(1, 1, H, D, generator = g, device = "cuda"))
    wk = torch.exp(0.4 * torch.randn(1, 1, H, D, generator = g, device = "cuda"))
    out = []
    for i, (w, ofs) in enumerate(((wq, 0.0), (wk, 0.6), (None, 0.5))):
        x = torch.empty(B, S, H, D, device = "cuda", dtype = torch.bfloat16)
        for j in range(0, S, 8192):
            n = min(8192, S - j)
            r = torch.randn(B, n, H, D, generator = g, device = "cuda")
            x[:, j:j + n] = (r * (w * 1.6 if w is not None else 1.0) + ofs).to(torch.bfloat16)
        out.append(x)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aiter-dir", required = True)
    ap.add_argument("--arm", required = True, choices = ("aotriton", "ck", "rdna"))
    ap.add_argument("--shape", required = True, choices = tuple(SHAPES))
    ap.add_argument("--out", required = True)
    ap.add_argument("--reps", type = int, default = 7)
    a = ap.parse_args()
    obs: dict = {"arm": a.arm, "shape": a.shape, "bshd": list(SHAPES[a.shape])}

    def save():
        open(a.out, "w", encoding = "utf-8").write(json.dumps(obs, default = str))
    t0 = time.time()
    try:
        import torch
        import torch.nn.functional as F
        obs.update(torch = torch.__version__, hip = torch.version.hip,
                   arch = getattr(torch.cuda.get_device_properties(0), "gcnArchName", None))
        B, S, H, D = SHAPES[a.shape]
        scale = D ** -0.5
        q, k, v = inputs(B, S, H, D)
        if a.arm in ("aotriton", "ck"):
            from torch.nn.attention import SDPBackend, sdpa_kernel
            if a.arm == "ck":
                try:
                    torch.backends.cuda.preferred_rocm_fa_library("ck")
                except Exception as exc:  # noqa: BLE001 -- the expected answer on a wheel built without CK SDPA
                    obs["unavailable"] = repr(exc)[:400]
                    save()
                    return 0
            obs["fa_library"] = str(torch.backends.cuda.preferred_rocm_fa_library())

            def call():
                with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
                    return F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                                          scale = scale).transpose(1, 2)
        else:
            _rocm_env(a.aiter_dir)
            tb = time.time()
            R = _import_rdna(a.aiter_dir)
            obs["import_s"] = round(time.time() - tb, 2)
            why = R.encoder_unsupported(q[0], H)
            if why is not None:
                obs["unavailable"] = f"encoder_unsupported: {why}"
                save()
                return 0
            q3, k3, v3 = q[0], k[0], v[0]
            obs["repack"] = "none: [S, H, D] contiguous heads are read in place"
            tb = time.time()
            R.encoder_forward(q3, k3, v3, [S], scale)  # first launch builds module_rdna_attn + the d128 bf16 unit
            torch.cuda.synchronize()
            obs["build_s"] = round(time.time() - tb, 2)

            def call():
                return R.encoder_forward(q3, k3, v3, [S], scale).unsqueeze(0)
        with torch.no_grad():
            for _ in range(2):
                out = call()
            torch.cuda.synchronize()
            ms = []
            for _ in range(a.reps):
                e0, e1 = torch.cuda.Event(enable_timing = True), torch.cuda.Event(enable_timing = True)
                e0.record()
                out = call()
                e1.record()
                e1.synchronize()
                ms.append(e0.elapsed_time(e1))
        obs["ms"], obs["med_ms"] = ms, st.median(ms)
        obs["spread_ms"] = max(ms) - min(ms)
        heads = sorted({0, H // 2, H - 1})
        rows = torch.linspace(0, S - 1, min(256, S), device = "cuda").long()
        qs = q[0, rows][:, heads].permute(1, 0, 2).double()
        ks = k[0][:, heads].permute(1, 0, 2).double()
        vs = v[0][:, heads].permute(1, 0, 2).double()
        ref = torch.matmul(torch.softmax(torch.matmul(qs, ks.transpose(1, 2)) * scale, -1), vs)
        got = out[0, rows][:, heads].permute(1, 0, 2).double()
        obs["max_abs"] = float((got - ref).abs().max())
        obs["rel_l2"] = float(torch.linalg.vector_norm(got - ref) / torch.linalg.vector_norm(ref))
        obs["finite"] = bool(torch.isfinite(out).all())
        obs["out_shape"] = list(out.shape)
        obs["out_sha"] = hashlib.sha256(out.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()[:16]
    except Exception as exc:  # noqa: BLE001
        import traceback
        obs["error"] = repr(exc)[:400]
        obs["tb"] = traceback.format_exc()[-2000:]
    obs["seconds"] = round(time.time() - t0, 1)
    save()
    return 0


if __name__ == "__main__":
    sys.exit(main())
