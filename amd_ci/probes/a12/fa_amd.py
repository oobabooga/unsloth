#!/usr/bin/env python3
"""R6-A item 2 (n6#1): flash-attn Triton AMD backend configs vs Studio's gfx1151 SDPA, one (shape, arm) per process.

  fa_amd.py --shape S --arm sdpa_studio|fa_default|fa_autotune|fa_json --out F [--config-json JSON] [--reps 5]

Studio installs flash-attn from Dao-AILab/flash-attention bbe25bacff (studio/backend/utils/rocm_flash_attn.py) with
FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE; diffusers' FLASH backend calls flash_attn_func on [B, S, H, D].
Env names read at import by flash_attn/flash_attn_triton_amd/utils.py (that commit):
  FLASH_ATTENTION_TRITON_AMD_AUTOTUNE (1/true/yes) -> 288-config autotune of attn_fwd (fwd_prefill.py)
  FLASH_ATTENTION_FWD_TRITON_AMD_CONFIG_JSON       -> one fixed config (ignored while AUTOTUNE is on)
  default RDNA config: BLOCK_M 32, BLOCK_N 32, PRE_LOAD_V False, num_stages 1, num_warps 4
sdpa_studio = Studio's own gfx1151 path: separate [B, S, H, D] projections, q as the permuted view, k / v copied
heads-major inside the call (diffusion_attention._heads_major), output copied back to [B, S, H, D].
Timing: 2 warm calls then --reps calls (CUDA events, 256 MB flush before each), median. The autotune arm reports the
first call (the tuning) separately and the winning config as JSON (``best_config``) for fa_json.
Accuracy: 256 sampled query rows x 3 heads vs fp64 softmax (rel L2, max abs), sha of the whole output."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
SHAPES = {"wan5b_27280": (1, 27280, 24, 128), "wan5b_5070": (1, 5070, 24, 128), "qwen21_4096": (1, 4096, 24, 128),
          "tiny": (1, 300, 2, 64)}


def _cfg_json(cfg) -> dict:
    d = dict(getattr(cfg, "kwargs", {}) or {})
    d["num_warps"] = getattr(cfg, "num_warps", None)
    d["num_stages"] = getattr(cfg, "num_stages", None)
    return d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", required = True, choices = tuple(SHAPES))
    ap.add_argument("--arm", required = True, choices = ("sdpa_studio", "fa_default", "fa_autotune", "fa_json"))
    ap.add_argument("--out", required = True)
    ap.add_argument("--config-json", default = "")
    ap.add_argument("--reps", type = int, default = 5)
    a = ap.parse_args()
    os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")
    os.environ["FLASH_ATTENTION_TRITON_AMD_ENABLE"] = "TRUE"
    os.environ.pop("FLASH_ATTENTION_TRITON_AMD_AUTOTUNE", None)
    os.environ.pop("FLASH_ATTENTION_FWD_TRITON_AMD_CONFIG_JSON", None)
    if a.arm == "fa_autotune":
        os.environ["FLASH_ATTENTION_TRITON_AMD_AUTOTUNE"] = "TRUE"
    if a.arm == "fa_json":
        os.environ["FLASH_ATTENTION_FWD_TRITON_AMD_CONFIG_JSON"] = a.config_json
    from a12common import env_info, sha, time_calls, write
    obs = {"shape": a.shape, "arm": a.arm, "config_json": a.config_json or None}
    t0 = time.time()
    try:
        import torch
        import torch.nn.functional as F
        obs["env"] = env_info()
        from attn_bench import inputs
        B, S, H, D = SHAPES[a.shape]
        q, k, v = (t.contiguous() for t in inputs(B, S, H, D))
        torch.cuda.empty_cache()
        out = torch.empty(B, S, H, D, device = "cuda", dtype = torch.bfloat16)
        if a.arm == "sdpa_studio":
            def call():
                kh, vh = k.permute(0, 2, 1, 3).contiguous(), v.permute(0, 2, 1, 3).contiguous()
                out.copy_(F.scaled_dot_product_attention(q.transpose(1, 2), kh, vh).transpose(1, 2))
                return out
        else:
            try:
                import flash_attn
                from flash_attn import flash_attn_func
                from flash_attn import flash_attn_interface as FI
            except Exception as exc:  # noqa: BLE001
                obs["unavailable"] = f"flash_attn import: {exc!r}"[:400]
                raise
            obs["flash_attn"] = {"version": getattr(flash_attn, "__version__", None),
                                 "path": os.path.dirname(os.path.realpath(flash_attn.__file__)),
                                 "use_triton_rocm": getattr(FI, "USE_TRITON_ROCM", None)}
            from flash_attn.flash_attn_triton_amd import fwd_prefill as FP
            from flash_attn.flash_attn_triton_amd import utils as FU
            obs["fa_triton"] = {"autotune": bool(FU.AUTOTUNE), "override": _cfg_json(FU.FWD_CONF_OVERRIDE)
                                if FU.FWD_CONF_OVERRIDE else None,
                                "n_configs": len(FP.fwd_prefill_autotune_configs),
                                "configs0": _cfg_json(FP.fwd_prefill_autotune_configs[0])}
            if a.arm == "fa_json" and not FU.FWD_CONF_OVERRIDE:
                raise RuntimeError("config JSON did not take (FWD_CONF_OVERRIDE is None)")

            def call():
                return flash_attn_func(q, k, v, dropout_p = 0.0, causal = False)
            if a.arm == "fa_autotune":
                tt = time.time()
                with torch.no_grad():
                    call()
                torch.cuda.synchronize()
                obs["autotune_s"] = round(time.time() - tt, 1)
                cache = getattr(FP.attn_fwd, "cache", {}) or {}
                obs["autotune_keys"] = len(cache)
                best = getattr(FP.attn_fwd, "best_config", None) or (next(iter(cache.values())) if cache else None)
                obs["best_config"] = _cfg_json(best) if best is not None else None
        flush = torch.empty(int(256e6), dtype = torch.int8, device = "cuda")
        hold = {}
        with torch.no_grad():
            obs.update(time_calls(lambda: hold.__setitem__("o", call()), a.reps, flush))
            o = hold.pop("o")
            heads = sorted({0, H // 2, H - 1})
            rows = torch.linspace(0, S - 1, 256, device = "cuda").long()
            qs = q[0, rows][:, heads].permute(1, 0, 2).double()
            ks = k[0][:, heads].permute(1, 0, 2).double()
            vs = v[0][:, heads].permute(1, 0, 2).double()
            ref = torch.matmul(torch.softmax(torch.matmul(qs, ks.transpose(1, 2)) / (D ** 0.5), -1), vs)
            got = o[0, rows][:, heads].permute(1, 0, 2).double()
            obs["rel_l2_fp64"] = float(torch.linalg.vector_norm(got - ref) / torch.linalg.vector_norm(ref))
            obs["max_abs_fp64"] = float((got - ref).abs().max())
            obs["finite"] = bool(torch.isfinite(o).all())
            obs["out_sha"] = sha(o)
            obs["out_shape"] = list(o.shape)
    except Exception:  # noqa: BLE001
        import traceback
        obs["error"] = traceback.format_exc()[-1500:]
    obs["seconds"] = round(time.time() - t0, 1)
    write(a.out, obs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
