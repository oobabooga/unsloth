#!/usr/bin/env python3
"""R3b-b: Qwen-Image-2.1 VAE decode attention on ROCm: stock diffusers SDPA vs Studio's VAE attention install
(``diffusion_vae_attn_chunked.install``: bf16 math SDPA, i.e. explicit bf16 matmuls), one fresh process per arm.

  qvae_attn.py --backend B --dir MODEL --arm stock|studio --out F [--save OUT.pt]

1024^2 latent [1, 64, 1, 64, 64], bf16. 2 warm decodes, 5 timed (no hooks), then 3 decodes with CUDA-event hooks
around every *AttentionBlock module for the attention time per decode."""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
import time


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required = True)
    ap.add_argument("--dir", required = True)
    ap.add_argument("--arm", required = True)
    ap.add_argument("--out", required = True)
    ap.add_argument("--save")
    a = ap.parse_args()
    sys.path.insert(0, a.backend)
    import torch
    import diffusers

    obs = {"arm": a.arm, "torch": torch.__version__}
    try:
        vae = diffusers.AutoencoderKLQwenImage21.from_pretrained(a.dir, subfolder = "vae",
                                                                  torch_dtype = torch.bfloat16).to("cuda").eval()
        if a.arm == "studio":
            from core.inference.diffusion_vae_attn_chunked import install
            obs["patched"] = install(vae)
        mods = [(n, m) for n, m in vae.named_modules() if type(m).__name__.endswith("AttentionBlock")]
        obs["attn_modules"] = [n for n, _ in mods]
        z = torch.randn((1, 64, 1, 64, 64), generator = torch.Generator(device = "cuda").manual_seed(5),
                        device = "cuda").to(torch.bfloat16)
        with torch.no_grad():
            for _ in range(2):
                vae.decode(z).sample
            torch.cuda.synchronize()
            ts = []
            for _ in range(5):
                torch.cuda.synchronize()
                t = time.perf_counter()
                y = vae.decode(z).sample
                torch.cuda.synchronize()
                ts.append(time.perf_counter() - t)
            obs["s"], obs["med_s"] = ts, st.median(ts)
            ev: list = []
            hooks = []
            for n, m in mods:
                def pre(mod, args, n = n):
                    e = torch.cuda.Event(enable_timing = True)
                    e.record()
                    ev.append([n, e, None])

                def post(mod, args, out, n = n):
                    e = torch.cuda.Event(enable_timing = True)
                    e.record()
                    for rec in reversed(ev):
                        if rec[0] == n and rec[2] is None:
                            rec[2] = e
                            break
                hooks += [m.register_forward_pre_hook(pre), m.register_forward_hook(post)]
            per = []
            for _ in range(3):
                ev.clear()
                vae.decode(z).sample
                torch.cuda.synchronize()
                per.append({"total_ms": sum(s.elapsed_time(e) for _, s, e in ev if e is not None),
                            "by_module": {n: s.elapsed_time(e) for n, s, e in ev if e is not None},
                            "calls": len(ev)})
            for h in hooks:
                h.remove()
        obs["attn"] = per
        obs["attn_ms_med"] = st.median(p["total_ms"] for p in per)
        obs["nonfinite"] = int((~torch.isfinite(y)).sum())
        if a.save:
            torch.save(y.float().clamp(-1, 1).to(torch.float16).cpu(), a.save)
    except Exception as exc:  # noqa: BLE001
        import traceback
        obs["error"] = repr(exc)[:400]
        obs["tb"] = traceback.format_exc()[-2000:]
    open(a.out, "w", encoding = "utf-8").write(json.dumps(obs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
