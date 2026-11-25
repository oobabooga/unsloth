#!/usr/bin/env python3
"""R6-A item 1: Studio's exact fused kernels on gfx1151 with the ROCm gate lifted (UNSLOTH_DIFFUSION_ROCM_FUSED_LIFT=1,
patch r6a_lift.diff on the head tree). One part per process:

  fusedk.py --backend B --part f3|h3|wanblk --out F [--arm stock|fused|fused_norope] [--tokens N]

f3   FLUX.3 QK-norm+RoPE: the module's own device check (_check: bit-exact probe + race), then klein's real call
     [1, 4608, 24, 128] bf16 (weights bf16 and fp32): stock reference_qk vs launch, torch.equal, median ms.
h3   MiniMax-H3 QK-norm+RoPE: module _check (eager + once roundings, rotary dims 128 / 64), then the real call
     [1, 37720, 56, 128]: stock vs each passing form, torch.equal, median ms.
wanblk  diffusers WanTransformerBlock at the Wan 2.2 TI2V-5B config (3072, 24 heads, ffn 14336, per-token temb),
     bf16 random weights, eager: arm stock (no install) / fused (install, rotary on) / fused_norope
     (UNSLOTH_DIFFUSION_WAN_FUSED_ROPE=0), median block ms, output sha + sampled rows saved for cross-arm equality,
     engagement = video_wan_fused.counts() and _VERIFIED."""
from __future__ import annotations

import argparse
import os
import sys
import time

os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a12common import env_info, sha, time_calls, write  # noqa: E402


def part_f3(obs):
    import torch
    from core.inference import diffusion_flux3_qk_rope as F3
    dev = torch.device("cuda", 0)
    obs["wanted"] = bool(F3.wanted("cuda"))
    obs["check"] = {}
    for wname, wd in (("bf16", torch.bfloat16), ("fp32", torch.float32)):
        try:
            obs["check"][wname] = F3._check(dev, torch.bfloat16, wd, 128, 1e-6)
        except Exception as exc:  # noqa: BLE001
            obs["check"][wname] = {"error": repr(exc)[:400]}
    g = torch.Generator(device = "cpu").manual_seed(5)
    S, H, D = 4608, 24, 128
    x = torch.randn(1, S, H, D, generator = g).to(torch.bfloat16).cuda()
    ang = (torch.rand(S, D // 2, generator = g) * 60).repeat_interleave(2, -1).cuda()
    cos, sin = ang.cos(), ang.sin()
    real = {}
    for wname, wd in (("bf16", torch.bfloat16), ("fp32", torch.float32)):
        w = (1 + torch.rand(D, generator = g)).to(wd).cuda()
        r = {"eligible": bool(F3.eligible(x, w, cos, sin, torch.bfloat16))}
        try:
            ref = F3.reference_qk(x, w, 1e-6, cos, sin, torch.bfloat16)
            got = F3.launch(x, w, 1e-6, cos, sin, torch.bfloat16)
            r["equal"] = bool(torch.equal(ref, got))
            r["max_abs"] = float((ref.float() - got.float()).abs().max())
            r["stock"] = time_calls(lambda: F3.reference_qk(x, w, 1e-6, cos, sin, torch.bfloat16), 15)["med_ms"]
            r["fused"] = time_calls(lambda: F3.launch(x, w, 1e-6, cos, sin, torch.bfloat16), 15)["med_ms"]
        except Exception as exc:  # noqa: BLE001
            r["error"] = repr(exc)[:400]
        real[wname] = r
    obs["real_klein"] = real


def part_h3(obs):
    import torch
    from core.inference import video_minimax_h3_qknorm as H3
    dev = torch.device("cuda", 0)
    obs["check"] = {}
    for rot in (128, 64):
        for rnd in H3.ROUNDINGS:
            try:
                obs["check"][f"rot{rot}_{rnd}"] = H3._check(dev, torch.bfloat16, 128, rot, rnd)
            except Exception as exc:  # noqa: BLE001
                obs["check"][f"rot{rot}_{rnd}"] = {"error": repr(exc)[:400]}
    g = torch.Generator(device = "cpu").manual_seed(7)
    S, H, D = 37720, 56, 128
    x = torch.randn(1, S, H, D, generator = g).to(torch.bfloat16).cuda()
    w = (1 + 0.2 * torch.randn(D, generator = g)).to(torch.bfloat16).cuda()
    real = {}
    for rot in (128, 64):
        ang = (torch.rand(S, rot // 2, generator = g) * 60).cuda()
        cos, sin = torch.cat((ang, ang), -1).cos(), torch.cat((ang, ang), -1).sin()
        for rnd in H3.ROUNDINGS:
            r = {"triton_ok": bool(H3._triton_ok(x))}
            try:
                ref = H3.reference_qk_norm_rope(x, w, cos, sin, 1e-5, rnd, exact = True)
                r["stock"] = time_calls(lambda: H3.reference_qk_norm_rope(x, w, cos, sin, 1e-5, rnd), 9)["med_ms"]
                for form in H3.MODES:
                    try:
                        got = H3._run_form(form, x, w, cos, sin, 1e-5, rnd)
                        r[f"{form}_equal"] = bool(torch.equal(got, ref))
                        r[form] = time_calls(lambda: H3._run_form(form, x, w, cos, sin, 1e-5, rnd), 9)["med_ms"]
                    except Exception as exc:  # noqa: BLE001
                        r[f"{form}_error"] = repr(exc)[:300]
            except Exception as exc:  # noqa: BLE001
                r["error"] = repr(exc)[:400]
            real[f"rot{rot}_{rnd}"] = r
    obs["real_h3"] = real


def part_wanblk(obs, arm, tokens, save):
    import torch
    from diffusers.models.transformers.transformer_wan import WanTransformerBlock
    from core.inference import video_wan_fused as W
    from core.inference import diffusion_attention as DA
    DA._ROCM_CONTIGUOUS_KV_L2[0] = int(getattr(torch.cuda.get_device_properties(0), "L2_cache_size", 0) or 0)
    obs["studio_kv_wrapper"] = bool(DA._install_contiguous_native())
    if arm == "fused_norope":
        os.environ["UNSLOTH_DIFFUSION_WAN_FUSED_ROPE"] = "0"
    torch.manual_seed(0)
    D, Hh, FF = 3072, 24, 14336
    blk = WanTransformerBlock(D, FF, Hh, qk_norm = "rms_norm_across_heads", cross_attn_norm = True, eps = 1e-6)
    with torch.no_grad():
        for p in blk.parameters():
            if p.dim() == 2:
                p.normal_(0, p.shape[1] ** -0.5)
            elif p.dim() == 3:
                p.normal_(0, 0.02)
    blk = blk.to("cuda", torch.bfloat16).eval()
    blk.scale_shift_table.data = blk.scale_shift_table.data.float()  # diffusers keeps it fp32 (_keep_in_fp32_modules)
    for n in (blk.norm2,):
        if getattr(n, "weight", None) is not None:
            n.float()
    g = torch.Generator(device = "cuda").manual_seed(1)
    x = torch.randn(1, tokens, D, generator = g, device = "cuda").to(torch.bfloat16)
    enc = torch.randn(1, 512, D, generator = g, device = "cuda").to(torch.bfloat16)
    temb = (torch.randn(1, tokens, 6, D, generator = g, device = "cuda") * 0.1).float()
    ang = torch.rand(1, tokens, 1, 64, generator = g, device = "cuda") * 60
    ang = ang.repeat_interleave(2, -1)
    rot = (ang.cos().float(), ang.sin().float())
    if arm != "stock":
        obs["installed"] = bool(W.install(torch.bfloat16, "cuda"))
    with torch.no_grad():
        y = blk(x, enc, temb, rot)
        obs.update(time_calls(lambda: blk(x, enc, temb, rot), 5))
    obs["out_sha"] = sha(y)
    if save:
        torch.save(y[0, ::97].float().cpu(), save)
    obs["counts"] = dict(W.counts()) if hasattr(W, "counts") else None
    obs["verified"] = {str(k): v for k, v in W._VERIFIED.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required = True)
    ap.add_argument("--part", required = True, choices = ("f3", "h3", "wanblk"))
    ap.add_argument("--arm", default = "stock")
    ap.add_argument("--tokens", type = int, default = 27280)
    ap.add_argument("--out", required = True)
    ap.add_argument("--save")
    a = ap.parse_args()
    sys.path.insert(0, a.backend)
    obs = {"part": a.part, "arm": a.arm, "tokens": a.tokens, "lift_env": os.environ.get("UNSLOTH_DIFFUSION_ROCM_FUSED_LIFT")}
    t0 = time.time()
    try:
        obs["env"] = env_info()
        if a.part == "f3":
            part_f3(obs)
        elif a.part == "h3":
            part_h3(obs)
        else:
            part_wanblk(obs, a.arm, a.tokens, a.save)
    except Exception:  # noqa: BLE001
        import traceback
        obs["error"] = traceback.format_exc()[-1500:]
    obs["seconds"] = round(time.time() - t0, 1)
    write(a.out, obs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
