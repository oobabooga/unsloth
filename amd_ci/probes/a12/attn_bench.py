#!/usr/bin/env python3
"""Items 17.2 (K loaded [D, BN]) and LLC head-group split: one (impl, shape, arm) per process.

  attn_bench.py --backend B --impl int8|sdpa --shape S --arm A --out F [--save P] [--reps 5]

int8 arms (Studio's gfx1151 table after select_config, overrides on top):
  studio | kt | kt_s2 | bn64_s2 | kt_bn64_s2 | hgG (studio config, heads run in groups of G, G in 2,4,8,16)
sdpa arms (torch SDPA, TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 as Studio's main.py): all | hgG
Timing: median of reps single calls (CUDA events, 256 MB flush before each). Accuracy: sampled rows vs fp64 softmax.
Exactness between arms/processes: sha of the full output (deterministic kernels) + saved sampled rows (--save)."""
from __future__ import annotations

import argparse
import os
import sys
import time

os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")
os.environ.setdefault("UNSLOTH_DIFFUSION_INT8_ATTENTION_IMPL", "triton")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a12common import env_info, sha, time_calls, write  # noqa: E402

SHAPES = {"h3_37720": (1, 37720, 56, 128), "wan5b_27280": (1, 27280, 24, 128), "klein_4608": (1, 4608, 24, 128)}
INT8_ARMS = {"studio": {}, "kt": {"k_t": True}, "kt_s2": {"k_t": True, "stages": 2}, "bn64_s2": {"bn": 64, "stages": 2},
             "kt_bn64_s2": {"k_t": True, "bn": 64, "stages": 2}, "s2": {"stages": 2}}


def inputs(B, S, H, D, seed = 0):
    import torch
    dtype = torch.bfloat16
    g = torch.Generator(device = "cuda").manual_seed(1234 + seed)

    def rn(*s):
        return torch.randn(*s, generator = g, device = "cuda", dtype = torch.float32)
    wq, wk = torch.exp(0.4 * rn(1, 1, H, D)), torch.exp(0.4 * rn(1, 1, H, D))
    kofs, vofs, vsc = 0.6 * rn(1, 1, H, D), 0.5 * rn(1, 1, H, D), 0.5 + rn(1, 1, H, D).abs()
    buf = torch.empty(B, S, 3, H, D, device = "cuda", dtype = dtype)
    for i in range(0, S, 8192):
        n = min(8192, S - i)
        buf[:, i:i + n, 0] = (rn(B, n, H, D) * wq * 1.6).to(dtype)
        buf[:, i:i + n, 1] = (rn(B, n, H, D) * wk * 1.6 + kofs).to(dtype)
        buf[:, i:i + n, 2] = (rn(B, n, H, D) * vsc + vofs).to(dtype)
    buf = buf.view(B, S, 3 * H * D)
    hd = H * D
    return (buf[:, :, :hd].view(B, S, H, D), buf[:, :, hd:2 * hd].view(B, S, H, D),
            buf[:, :, 2 * hd:].view(B, S, H, D))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required = True)
    ap.add_argument("--impl", required = True, choices = ("int8", "sdpa"))
    ap.add_argument("--shape", required = True)
    ap.add_argument("--arm", required = True)
    ap.add_argument("--out", required = True)
    ap.add_argument("--save")
    ap.add_argument("--reps", type = int, default = 5)
    a = ap.parse_args()
    sys.path.insert(0, a.backend)
    obs = {"impl": a.impl, "shape": a.shape, "arm": a.arm, "backend": a.backend}
    t0 = time.time()
    try:
        import torch
        import torch.nn.functional as F
        obs["env"] = env_info()
        B, S, H, D = SHAPES[a.shape]
        q, k, v = inputs(B, S, H, D)
        out = torch.empty(B, S, H, D, device = "cuda", dtype = torch.bfloat16)
        G = int(a.arm[2:]) if a.arm.startswith("hg") else H
        groups = [(h0, min(H, h0 + G)) for h0 in range(0, H, G)]
        obs["groups"] = len(groups)
        if a.impl == "int8":
            from core.inference import diffusion_int8_attention as M
            from core.inference import diffusion_int8_attention_triton as T
            import inspect
            obs["kt_in_tree"] = "K_T" in inspect.getsource(T)
            table = M.rocm_kernel_config(M._gfx_arch(0), D)
            cfg = T.select_config(table, S)
            cfg = {kk: vv for kk, vv in cfg.items() if kk not in ("long", "long_keys")}
            cfg.update(INT8_ARMS.get(a.arm, {}))
            obs["cfg"] = cfg
            if cfg.get("k_t") and not obs["kt_in_tree"]:
                raise RuntimeError("k_t arm on a tree without K_T")

            def call():
                if len(groups) == 1:
                    out.copy_(T.int8_attention(q, k, v, cfg))
                else:
                    for h0, h1 in groups:
                        out[:, :, h0:h1].copy_(T.int8_attention(q[:, :, h0:h1], k[:, :, h0:h1], v[:, :, h0:h1], cfg))
        else:
            def one(h0, h1):
                return F.scaled_dot_product_attention(q[:, :, h0:h1].transpose(1, 2), k[:, :, h0:h1].transpose(1, 2),
                                                      v[:, :, h0:h1].transpose(1, 2)).transpose(1, 2)

            def call():
                for h0, h1 in groups:
                    out[:, :, h0:h1].copy_(one(h0, h1))
            # which backend the default call takes: the forced-flash result must equal it
            from torch.nn.attention import SDPBackend, sdpa_kernel
            with torch.no_grad():
                call()
                ref_default = out.clone()
                try:
                    with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
                        call()
                    obs["default_is_flash"] = bool(torch.equal(out, ref_default))
                except Exception as exc:  # noqa: BLE001
                    obs["default_is_flash"] = f"flash refused: {exc!r}"[:300]
                del ref_default
        flush = torch.empty(int(256e6), dtype = torch.int8, device = "cuda")
        with torch.no_grad():
            obs.update(time_calls(call, a.reps, flush))
            heads = sorted({0, H // 2, H - 1})
            rows = torch.linspace(0, S - 1, 256, device = "cuda").long()
            qs = q[0, rows][:, heads].permute(1, 0, 2).double()
            ks = k[0][:, heads].permute(1, 0, 2).double()
            vs = v[0][:, heads].permute(1, 0, 2).double()
            ref = torch.matmul(torch.softmax(torch.matmul(qs, ks.transpose(1, 2)) / (D ** 0.5), -1), vs)
            got = out[0, rows][:, heads].permute(1, 0, 2).double()
            obs["rel_l2_fp64"] = float(torch.linalg.vector_norm(got - ref) / torch.linalg.vector_norm(ref))
            obs["max_abs_fp64"] = float((got - ref).abs().max())
            obs["finite"] = bool(torch.isfinite(out).all())
            obs["out_sha"] = sha(out)
            if a.save:
                torch.save(out[0, rows].float().cpu(), a.save)
    except Exception as exc:  # noqa: BLE001
        import traceback
        obs["error"] = traceback.format_exc()[-1500:]
    obs["seconds"] = round(time.time() - t0, 1)
    write(a.out, obs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
