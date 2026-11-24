#!/usr/bin/env python3
"""M13 accuracy: one Wan2.2-TI2V-5B VAE arm in a fresh process.
  capture  Studio default load, one render at --steps; saves the exact vae.decode input (after a device sync) to --latent
  decode   Studio default load (its load-time VAE wrappers: bf16 decode, untiled), then pipe.vae.decode(latent) twice,
           timed; frames (uint8, (x+1)/2) to --npy
  cpu      stock diffusers AutoencoderKLWan in fp32 on the CPU (the reference), same latent, same uint8 mapping
Writes a JSON record to --rec. Observes only."""
import argparse
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

REPO = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
PROMPT = "a golden retriever running through a sunny meadow, slow motion, shallow depth of field"
SEED = 20261124


def to_u8(x):
    import numpy as np

    v = x.detach().float().cpu().clamp(-1, 1)          # (1, 3, F, H, W)
    v = ((v + 1) / 2 * 255).round().to(dtype = __import__("torch").uint8)[0].permute(1, 2, 3, 0).contiguous()
    return np.ascontiguousarray(v.numpy())            # (F, H, W, 3)


def load_studio(root):
    sys.path.insert(0, str(Path(root).resolve()))
    from core.inference import video as V

    eng = V.VideoBackend()
    eng.begin_load(REPO, local_files_only = False)
    while True:
        p = eng.load_progress() or {}
        if p.get("phase") == "ready":
            break
        if p.get("phase") == "error" or p.get("error"):
            raise RuntimeError(f"load failed: {p.get('error')}")
        time.sleep(0.5)
    return eng


def engagement(rec):
    import os

    rec["user_db_env_after_load"] = os.environ.get("MIOPEN_USER_DB_PATH")
    try:
        from core.inference import diffusion_rocm_miopen_db as M
        rec["studio_miopen_db_active"] = M.is_active()
        rec["studio_miopen_db"] = M._STATE.get("db")
    except Exception as exc:  # noqa: BLE001 - base tree has no module
        rec["studio_miopen_db_active"] = repr(exc)[:200]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices = ("capture", "decode", "cpu"))
    ap.add_argument("--root", default = "")
    ap.add_argument("--latent", required = True)
    ap.add_argument("--npy", default = "")
    ap.add_argument("--rec", required = True)
    ap.add_argument("--steps", type = int, default = 20)
    a = ap.parse_args()
    import os
    rec = {"mode": a.mode, "env": {k: v for k, v in os.environ.items() if k.startswith("MIOPEN_")}}
    try:
        import numpy as np
        import torch

        rec["torch"] = torch.__version__
        if a.mode == "capture":
            eng = load_studio(a.root)
            engagement(rec)
            pipe = eng._state.pipe
            vdec = pipe.vae.decode
            box = {}

            def decode(*x, **k):
                torch.cuda.synchronize()
                z = x[0].detach().float().cpu().clone()
                box.setdefault("z", z)
                return vdec(*x, **k)
            pipe.vae.decode = decode
            t = time.perf_counter()
            eng.generate(prompt = PROMPT, seed = SEED, steps = a.steps)
            rec["render_s"] = round(time.perf_counter() - t, 1)
            z = box["z"]
            torch.save(z, a.latent)
            rec["latent_shape"] = list(z.shape)
            rec["latent_md5"] = hashlib.md5(z.numpy().tobytes()).hexdigest()
            rec["latent_std"] = float(z.std())
        elif a.mode == "decode":
            eng = load_studio(a.root)
            engagement(rec)
            pipe = eng._state.pipe
            z = torch.load(a.latent)
            rec["latent_md5"] = hashlib.md5(z.numpy().tobytes()).hexdigest()
            zz = z.to("cuda")
            rec["vae_dtype"] = str(pipe.vae.dtype)
            ts = []
            out = None
            with torch.no_grad():
                for _ in range(2):
                    torch.cuda.synchronize()
                    t = time.perf_counter()
                    out = pipe.vae.decode(zz, return_dict = False)[0]
                    torch.cuda.synchronize()
                    ts.append(round(time.perf_counter() - t, 3))
            rec["decode_s"] = ts
            rec["peak_alloc_gib"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 3)
            arr = to_u8(out)
            np.save(a.npy, arr)
            rec["md5"] = hashlib.md5(arr.tobytes()).hexdigest()
        else:
            import diffusers

            torch.set_num_threads(max(1, (os.cpu_count() or 8)))
            vae = diffusers.AutoencoderKLWan.from_pretrained(REPO, subfolder = "vae", torch_dtype = torch.float32,
                                                             token = False).eval()
            z = torch.load(a.latent)
            rec["latent_md5"] = hashlib.md5(z.numpy().tobytes()).hexdigest()
            t = time.perf_counter()
            with torch.no_grad():
                out = vae.decode(z, return_dict = False)[0]
            rec["decode_s"] = [round(time.perf_counter() - t, 1)]
            arr = to_u8(out)
            np.save(a.npy, arr)
            rec["md5"] = hashlib.md5(arr.tobytes()).hexdigest()
        rec["ok"] = True
    except BaseException as exc:  # noqa: BLE001
        rec["ok"] = False
        rec["error"] = repr(exc)[:1500]
        rec["traceback"] = traceback.format_exc()[-4000:]
        print(rec["traceback"], flush = True)
    Path(a.rec).write_text(json.dumps(rec, indent = 1), encoding = "utf-8")
    print("VAE_ARM_DONE", a.mode, rec.get("ok"), flush = True)
    os._exit(0 if rec.get("ok") else 3)


if __name__ == "__main__":
    main()
