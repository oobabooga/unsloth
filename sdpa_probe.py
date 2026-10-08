"""One SDPA case per process: a failed fused launch poisons the HIP context, so cases must not share one."""
import json
import os
import sys
import traceback

case = sys.argv[1]
out_path = sys.argv[2]
os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

BACKENDS = {
    "default": None,
    "flash": SDPBackend.FLASH_ATTENTION,
    "efficient": SDPBackend.EFFICIENT_ATTENTION,
    "math": SDPBackend.MATH,
}


def shapes(name):
    g = torch.Generator(device="cuda").manual_seed(0)
    r = lambda *s, dt=torch.bfloat16: torch.randn(*s, device="cuda", dtype=dt, generator=g)
    if name == "studio_probe":  # diffusion_attention._probe_sdpa_kernels
        q = torch.zeros((1, 2, 8, 64), device="cuda", dtype=torch.float16)
        return dict(q=q, k=q, v=q)
    if name == "dit_bhsd":  # Qwen-Image-2.1 joint attention at 1024^2: 4096 image + ~256 text tokens, 32 x 128
        return dict(q=r(1, 32, 4352, 128), k=r(1, 32, 4352, 128), v=r(1, 32, 4352, 128))
    if name == "dit_bshd_view":  # diffusers passes BSHD and transposes: non-contiguous views
        q, k, v = (r(1, 4352, 32, 128).transpose(1, 2) for _ in range(3))
        return dict(q=q, k=k, v=v)
    if name == "te_causal_gqa":  # Qwen3-VL-8B text: 32 q heads, 8 kv heads, 128
        return dict(q=r(1, 32, 300, 128), k=r(1, 8, 300, 128), v=r(1, 8, 300, 128), is_causal=True, enable_gqa=True)
    if name == "te_causal_expanded":  # GQA repeated to 32 heads, causal, no mask (eager repeat_kv style)
        return dict(q=r(1, 32, 300, 128), k=r(1, 32, 300, 128), v=r(1, 32, 300, 128), is_causal=True)
    if name == "te_masked":  # additive mask path (padding)
        m = torch.zeros(1, 1, 300, 300, device="cuda", dtype=torch.bfloat16)
        m[..., 250:] = float("-inf")
        return dict(q=r(1, 32, 300, 128), k=r(1, 32, 300, 128), v=r(1, 32, 300, 128), attn_mask=m)
    if name == "vision_hd72":  # Qwen3-VL vision tower: 16 heads x 72
        return dict(q=r(1, 16, 4096, 72), k=r(1, 16, 4096, 72), v=r(1, 16, 4096, 72))
    raise SystemExit("unknown shape " + name)


shape, backend = case.split(":")
row = dict(case=case, torch=torch.__version__, hip=torch.version.hip,
           arch=getattr(torch.cuda.get_device_properties(0), "gcnArchName", None),
           aotriton_exp=os.environ.get("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"))
try:
    kw = shapes(shape)
    q, k, v = kw.pop("q"), kw.pop("k"), kw.pop("v")
    ctx = sdpa_kernel([BACKENDS[backend]]) if BACKENDS[backend] is not None else torch.no_grad()
    with ctx:
        out = F.scaled_dot_product_attention(q, k, v, **kw)
    # A failed fused launch only surfaces at the next CHECKED launch: run one, then read back.
    y = (out.float() @ torch.ones(out.shape[-1], 8, device="cuda")).contiguous()
    val = y.sum().item()
    ref = None
    with sdpa_kernel([SDPBackend.MATH]):
        ref = F.scaled_dot_product_attention(q, k, v, **kw)
    row.update(ok=True, finite=bool(torch.isfinite(out).all().item()),
               max_abs_vs_math=float((out.float() - ref.float()).abs().max().item()), checksum=val)
except Exception as exc:  # noqa: BLE001
    row.update(ok=False, error=repr(exc)[:600], tb=traceback.format_exc()[-1500:])
print("SDPA " + json.dumps(row), flush=True)
with open(out_path, "a", encoding="utf-8") as f:
    f.write(json.dumps(row) + "\n")
