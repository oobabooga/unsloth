"""One arm: set the flag through the checkout's real apply_speed_optims, then time two VAE decodes."""
import json, os, sys, time, types

src, out = sys.argv[1], sys.argv[2]
sys.path.insert(0, os.path.join(src, "studio", "backend"))
import torch
from diffusers import AutoencoderKLQwenImage
from core.inference.diffusion_speed import apply_speed_optims, SPEED_EAGER

res = {"torch": torch.__version__, "hip": getattr(torch.version, "hip", None),
       "device": torch.cuda.get_device_name(0), "benchmark_before": torch.backends.cudnn.benchmark}
torch.manual_seed(0)
vae = AutoencoderKLQwenImage().to("cuda", torch.bfloat16).eval()
pipe = types.SimpleNamespace(vae = vae)
target = types.SimpleNamespace(device = "cuda", dtype = torch.bfloat16)
family = types.SimpleNamespace(supports_torch_compile = False)
applied = apply_speed_optims(pipe, target, is_gguf = True, family = family, speed_mode = SPEED_EAGER)
res["applied"] = {k: v for k, v in applied.items() if v}
res["benchmark_after"] = torch.backends.cudnn.benchmark
z = torch.randn(1, 16, 1, 736 // 8, 1280 // 8, device = "cuda", dtype = torch.bfloat16)
for name in ("decode1_s", "decode2_s"):
    torch.cuda.synchronize(); t = time.perf_counter()
    with torch.no_grad():
        img = vae.decode(z).sample
    torch.cuda.synchronize(); res[name] = round(time.perf_counter() - t, 2)
    print(name, res[name], flush = True)
res["shape"] = list(img.shape)
res["finite"] = bool(torch.isfinite(img).all())
json.dump(res, open(out, "w"))
print(json.dumps(res), flush = True)
