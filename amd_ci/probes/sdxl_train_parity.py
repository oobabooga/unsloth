#!/usr/bin/env python3
"""One short Studio SDXL LoRA run -> one metrics JSON (per-step loss / grad_norm, device facts).

Vendored from the workspace parity driver (temp/devmatrix/train_parity.py), SDXL only, plus the
facts sdxl_fp16_probe.py needs to prove the run happened where it claims: the dtype and device of
the UNet's output (the trainer's real weight dtype, not the requested one) and peak GPU memory.
Driven by sdxl_fp16_probe.py; it observes, never judges.

  python sdxl_train_parity.py --wt <checkout> --base <tiny sdxl dir> --mixed fp16 --out x.json
"""

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--wt", required = True)
ap.add_argument("--base", required = True)
ap.add_argument("--mixed", default = "bf16")  # bf16 / fp16 / no
ap.add_argument("--steps", type = int, default = 6)
ap.add_argument("--res", type = int, default = 1024)
ap.add_argument("--images", type = int, default = 4)
ap.add_argument("--seed", type = int, default = 3407)
ap.add_argument("--out", required = True)
a = ap.parse_args()

os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")  # what the Studio server exports to its trainer children
os.environ["UNSLOTH_DIFFUSION_FP32_OPTIM"] = "1"  # plain AdamW, same optimizer on every backend
sys.path.insert(0, str(Path(a.wt) / "studio/backend"))

import torch  # noqa: E402
from PIL import Image  # noqa: E402

work = Path(tempfile.mkdtemp(prefix = "sdxl_parity_"))  # under $TMPDIR, which the preamble roots in $RUNNER_TEMP
data = work / "data"
data.mkdir()
g = torch.Generator().manual_seed(0)
for i in range(a.images):  # deterministic smooth images, one caption each
    base = torch.rand(3, 8, 8, generator = g)
    img = torch.nn.functional.interpolate(base[None], size = (a.res, a.res), mode = "bicubic")[0].clamp(0, 1)
    Image.fromarray((img.permute(1, 2, 0).numpy() * 255).astype("uint8")).save(data / f"img{i}.png")
    (data / f"img{i}.txt").write_text(f"a photo of sks pattern number {i}", encoding = "utf-8")

from core.training.diffusion_train_common import DiffusionLoraConfig, get_trainer  # noqa: E402

# Record what the UNet really computed in: the trainer may downgrade the requested precision
# (bf16 -> fp16 without native bf16, anything -> fp32 on CPU), and a probe trusting the request
# would compare the wrong arms.
unet_seen: dict = {}


def _unet_hook(module, _inp, out):
    if type(module).__name__ != "UNet2DConditionModel" or unet_seen.get("calls", 0) >= 64:
        return
    t = out[0] if isinstance(out, (tuple, list)) else getattr(out, "sample", out)
    if torch.is_tensor(t):
        unet_seen["calls"] = unet_seen.get("calls", 0) + 1
        unet_seen["dtypes"] = sorted(set(unet_seen.get("dtypes", [])) | {str(t.dtype)})
        unet_seen["devices"] = sorted(set(unet_seen.get("devices", [])) | {t.device.type})


_hook = torch.nn.modules.module.register_module_forward_hook(_unet_hook)

events = []
fields = dict(
    base_model = a.base, data_dir = str(data), output_dir = str(work / "out"), resolution = a.res,
    train_steps = a.steps, seed = a.seed, model_family = "sdxl", mixed_precision = a.mixed,
    gradient_checkpointing = True, compile_transformer = "off", enable_tf32 = False, cache_latents = True,
    random_flip = False, log_every = 1, save_steps = 0, lr_scheduler = "constant", lr_warmup_steps = 0,
)
known = set(DiffusionLoraConfig.__dataclass_fields__)
cfg = DiffusionLoraConfig(**{k: v for k, v in fields.items() if k in known})
fam = cfg.normalized().resolved_family
if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()
t0 = time.time()
err = None
try:
    get_trainer(fam)(cfg, on_event = lambda ev: events.append(dict(ev)))
except Exception as e:  # noqa: BLE001
    err = f"{type(e).__name__}: {e}"
_hook.remove()

lora = None
for p in (work / "out").glob("*.safetensors"):
    from safetensors.torch import load_file
    sd = load_file(str(p))
    h = hashlib.sha256()
    for k in sorted(sd):
        h.update(k.encode())
        h.update(sd[k].float().cpu().numpy().tobytes())
    lora = {"sha256": h.hexdigest(), "l2": float(sum(v.float().norm() ** 2 for v in sd.values()) ** 0.5), "n": len(sd)}
cuda = torch.cuda.is_available()
dev = {"cuda": cuda, "hip": torch.version.hip, "cuda_version": torch.version.cuda, "torch": torch.__version__,
       "name": torch.cuda.get_device_name() if cuda else None,
       "arch": getattr(torch.cuda.get_device_properties(0), "gcnArchName", None) if cuda else None,
       "max_mem_allocated": int(torch.cuda.max_memory_allocated()) if cuda else 0}
prog = [e for e in events if e["type"] == "progress"]
losses = [e.get("loss") for e in prog]
res = {"args": vars(a), "family": fam, "device": dev, "error": err, "wall_s": round(time.time() - t0, 1),
       "unet": unet_seen, "warnings": [e.get("message") for e in events if e["type"] == "warning"],
       "loss": losses, "grad_norm": [e.get("grad_norm") for e in prog], "lora": lora,
       "finite": bool(prog) and all(x is not None and x == x and abs(x) != float("inf") for x in losses)}
Path(a.out).parent.mkdir(parents = True, exist_ok = True)
Path(a.out).write_text(json.dumps(res, indent = 1), encoding = "utf-8")
sys.exit(0 if err is None and res["finite"] else 1)
