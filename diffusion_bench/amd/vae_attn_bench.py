"""VAE decode A/B on the AMD runner: Studio's ROCm VAE attention install from two trees (and the head's kill switch),
same weights, same latent. Per arm: decode ms (median of 5, device events), attention-module ms, peak alloc; across
arms: max abs / PSNR of the decoded image and LPIPS vs the base decode.

    python vae_attn_bench.py --repo unsloth/flux-3-image-klein --arm base=<tree> --arm head=<tree> \
        --arm head_off=<tree>:UNSLOTH_VAE_ATTN_HALF_MATH=0 --image x.png --out vae.json
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path


def child(a):
    import types

    import torch

    sys.path.insert(0, os.path.join(a.tree, "studio", "backend"))
    from diffusers import AutoencoderKLFlux2

    from core.inference import diffusion_vae_attn_chunked as C

    vae = AutoencoderKLFlux2.from_pretrained(a.repo, subfolder = "vae", torch_dtype = torch.bfloat16, token = False)
    vae = vae.to("cuda").eval()
    n = C.install(vae, types.SimpleNamespace(backend = "rocm", device = "cuda"))
    if a.image and Path(a.image).exists():
        from PIL import Image

        img = Image.open(a.image).convert("RGB").resize((1024, 1024))
        x = torch.tensor(list(img.getdata()), dtype = torch.float32).reshape(1024, 1024, 3)
        x = (x.permute(2, 0, 1)[None] / 127.5 - 1).to("cuda", torch.bfloat16)
    else:
        g = torch.Generator().manual_seed(0)
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, 1024), torch.linspace(-1, 1, 1024), indexing = "ij")
        x = torch.stack([torch.sin(6 * xx), torch.cos(5 * yy), xx * yy])[None] * 0.8
        x = (x + 0.05 * torch.randn(x.shape, generator = g)).to("cuda", torch.bfloat16)
    with torch.no_grad():
        lat = vae.encode(x).latent_dist.mode()
        attn_ms = []
        mods = [m for m in vae.decoder.modules() if getattr(m, C._PATCHED, False)]
        for m in mods:
            inner = m.forward

            def timed(*args, _inner = inner, **kw):
                s, e = torch.cuda.Event(enable_timing = True), torch.cuda.Event(enable_timing = True)
                s.record()
                out = _inner(*args, **kw)
                e.record()
                e.synchronize()
                attn_ms.append(s.elapsed_time(e))
                return out

            m.forward = timed
        for _ in range(2):
            vae.decode(lat)
        torch.cuda.synchronize()
        attn_ms.clear()
        torch.cuda.reset_peak_memory_stats()
        times = []
        for _ in range(5):
            s, e = torch.cuda.Event(enable_timing = True), torch.cuda.Event(enable_timing = True)
            s.record()
            y = vae.decode(lat).sample
            e.record()
            e.synchronize()
            times.append(s.elapsed_time(e))
    torch.save({"y": y.float().cpu(), "x": x.float().cpu()}, a.dump)
    print(json.dumps({"decode_ms": round(statistics.median(times), 2), "decode_runs": [round(t, 2) for t in times],
                      "attn_ms_per_decode": round(sum(attn_ms) / 5, 2), "attn_modules": len(mods), "patched": n,
                      "peak_alloc_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                      "half_flag_after": bool(getattr(torch.backends.cuda, "fp16_bf16_reduction_math_sdp_allowed", lambda: None)()),
                      "torch": torch.__version__}))  # fmt: skip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default = "unsloth/flux-3-image-klein")
    ap.add_argument("--arm", action = "append", default = [])
    ap.add_argument("--image", default = "")
    ap.add_argument("--out", required = True)
    ap.add_argument("--child", action = "store_true")
    ap.add_argument("--tree", default = "")
    ap.add_argument("--dump", default = "")
    ap.add_argument("--rounds", type = int, default = 2)
    a = ap.parse_args()
    if a.child:
        return child(a)
    out_dir = Path(a.out).parent
    res = {"arms": {}}
    arms = []
    for spec in a.arm:
        name, rest = spec.split("=", 1)
        tree, _, envs = rest.partition(":")
        env = dict(os.environ)
        for kv in filter(None, envs.split(",")):
            k, v = kv.split("=", 1)
            env[k] = v
        arms.append((name, tree, env))
    # Interleaved rounds (A/A spread per arm); a fresh process per run so each imports its own tree.
    for r in range(a.rounds):
        for name, tree, env in arms:
            dump = str(out_dir / f"vae_{name}_{r}.pt")
            p = subprocess.run([sys.executable, __file__, "--child", "--repo", a.repo, "--tree", tree, "--image", a.image,
                                "--dump", dump, "--out", a.out], env = env, capture_output = True, text = True, timeout = 1200)  # fmt: skip
            line = (p.stdout.strip().splitlines() or [""])[-1]
            try:
                got = json.loads(line)
            except Exception:  # noqa: BLE001
                got = {"error": p.stderr[-2000:], "rc": p.returncode}
            res["arms"].setdefault(name, []).append(got)
            print(name, r, got, flush = True)
    import torch

    ref = torch.load(out_dir / f"vae_{arms[0][0]}_0.pt")["y"]
    lp = None
    try:
        import lpips

        lp = lpips.LPIPS(net = "alex")
    except Exception as exc:  # noqa: BLE001
        res["lpips_error"] = repr(exc)
    cmp = {}
    for name, _, _ in arms:
        for r in range(a.rounds):
            f = out_dir / f"vae_{name}_{r}.pt"
            if not f.exists():
                continue
            y = torch.load(f)["y"]
            mse = float(((y.clamp(-1, 1) - ref.clamp(-1, 1)) ** 2).mean())
            row = {"max_abs": float((y - ref).abs().max()), "psnr_db": round(10 * __import__("math").log10(4 / max(mse, 1e-12)), 2)}
            if lp is not None:
                with torch.no_grad():
                    row["lpips_vs_base0"] = round(float(lp(y.clamp(-1, 1), ref.clamp(-1, 1))), 5)
            cmp[f"{name}_{r}"] = row
    res["vs_base0"] = cmp
    Path(a.out).write_text(json.dumps(res, indent = 1))
    print(json.dumps(cmp))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
