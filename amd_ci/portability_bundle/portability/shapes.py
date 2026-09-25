"""Real GEMM and conv shapes of Qwen-Image-2.1 and MiniMax-H3.

GEMMs come from ``gemm_shapes.json`` (the profiler's per-shape table) when present, else from the model configs:
  Qwen-Image-2.1 DiT (temp/qwen_image_21_cfg): 32 heads x 128 = 4096 hidden, mlp_ratio 3, single stream over
    [text; image] tokens, unfused SwiGLU (proj, gate_layer 4096->12288, out 12288->4096), no bias.
  MiniMax-H3 DiT (ext_minimax_h3_cfg, enumerated on the meta device): hidden 5376, attn 56x128 = 7168,
    ffn 14336 (gated: net.0.proj 5376->28672, net.2 14336->5376), adaLN
    2688->96768 with bias (M = batch, int8 not applicable), and the VAE decoder ViT (2048 hidden, 8192 ffn, bias).
M: 1024x1024 Qwen image = 64x64 = 4096 image tokens + ~256 text; H3 480x832x17f ~ 1950 tokens/clip-chunk, the VAE
decoder ~1.8k tokens per 256px tile (4 tiles batched).
"""
from __future__ import annotations

import json
import os

Q21_TOK = 4096 + 256
Q21_TOK_QUICK = 1024 + 256
H3_TOK = 8192
H3_TOK_QUICK = 2048
H3_VAE_TOK = 4 * 1800

# (tag, K, N, bias, count per forward)
Q21_LINEARS = [
    ("q21.attn.to_qkv", 4096, 4096, False, 96),
    ("q21.attn.to_out", 4096, 4096, False, 32),
    ("q21.mlp.proj_gate", 4096, 12288, False, 64),
    ("q21.mlp.out", 12288, 4096, False, 32),
]
H3_LINEARS = [
    ("h3.attn.to_qkv", 5376, 7168, False, 150),
    ("h3.attn.to_out", 7168, 5376, False, 50),
    ("h3.ff.proj", 5376, 28672, False, 50),
    ("h3.ff.out", 14336, 5376, False, 50),
]
H3_VAE_LINEARS = [
    ("h3vae.dec.attn", 2048, 2048, True, 144),
    ("h3vae.dec.ff.proj", 2048, 16384, True, 36),
    ("h3vae.dec.ff.out", 8192, 2048, True, 36),
]


def _profiler_table(ws: str | None, bundle: str | None):
    for p in ([os.path.join(bundle, "gemm_shapes.json")] if bundle else []) + (
        [os.path.join(ws, "outputs", "bottlenecks_q21_h3", "gemm_shapes.json")] if ws else []
    ):
        if p and os.path.exists(p):
            try:
                with open(p, encoding = "utf-8") as f:
                    return json.load(f), p
            except Exception:  # noqa: BLE001
                pass
    return None, None


def _parse_profiler(obj) -> list:
    """Tolerant: any list (or dict of lists) of records carrying M/K/N (any case) becomes a shape."""
    recs = []

    def walk(o, model = ""):
        if isinstance(o, dict):
            keys = {k.lower(): k for k in o}
            if all(x in keys for x in ("m", "k", "n")):
                recs.append((o.get("model", model), int(o[keys["m"]]), int(o[keys["k"]]), int(o[keys["n"]]),
                             bool(o.get("bias", False)), float(o.get("time_share", o.get("share", 0)) or 0)))
                return
            for k, v in o.items():
                walk(v, k if isinstance(v, (list, dict)) and not model else model)
        elif isinstance(o, list):
            for v in o:
                walk(v, model)

    walk(obj)
    return recs


def gemm_shapes(quick: bool, ws: str | None = None, bundle: str | None = None) -> list[dict]:
    """[{tag, M, K, N, bias, source}], deduplicated on (M, K, N, bias)."""
    table, src = _profiler_table(ws, bundle)
    out = []
    if table is not None:
        recs = _parse_profiler(table)
        recs.sort(key = lambda r: -r[5])
        for model, m, k, n, b, share in recs[: (6 if quick else 16)]:
            if quick:
                m = min(m, 4352)
            out.append({"tag": f"{model or 'prof'}.{k}x{n}", "M": m, "K": k, "N": n, "bias": b,
                        "source": f"profiler {os.path.basename(src)} share={share:.3f}"})
    if not out:
        qm, hm = (Q21_TOK_QUICK, H3_TOK_QUICK) if quick else (Q21_TOK, H3_TOK)
        for tag, k, n, b, _ in Q21_LINEARS:
            out.append({"tag": tag, "M": qm, "K": k, "N": n, "bias": b, "source": "config"})
        for tag, k, n, b, _ in H3_LINEARS:
            out.append({"tag": tag, "M": hm, "K": k, "N": n, "bias": b, "source": "config"})
        for tag, k, n, b, _ in H3_VAE_LINEARS[: (1 if quick else 3)]:
            out.append({"tag": tag, "M": 2048 if quick else H3_VAE_TOK, "K": k, "N": n, "bias": b, "source": "config"})
        if quick:
            # the Qwen 1024x1024 production size on the two most expensive shapes
            out.append({"tag": "q21.mlp.proj_gate@1024", "M": Q21_TOK, "K": 4096, "N": 12288, "bias": False,
                        "source": "config"})
            out.append({"tag": "q21.attn.to_qkv@1024", "M": Q21_TOK, "K": 4096, "N": 4096, "bias": False,
                        "source": "config"})
    seen, uniq = set(), []
    for s in out:
        key = (s["M"], s["K"], s["N"], s["bias"])
        if key not in seen:
            seen.add(key)
            uniq.append(s)
    return uniq


# Conv3d on an already causally padded, channels-last input (Studio pads outside the conv, as the H3 fused path and
# the Wan-style CausalConv3d both do). (tag, Cin, Cout, T_in, H_in, W_in, kernel, stride)
# H3 encoder at a 256x256 tile, 17-frame clip (block_out_channels 128,256,256,512,512,1024).
# Qwen-Image-2.1 VAE (Wan-style, base 96 / decoder base 144, dim_mult 1,2,4,8,8) decoding one 1024x1024 image: a
# single frame whose causal pad makes T_in = 3.
CONV_SHAPES_FULL = [
    ("h3.enc.L0.128", 128, 128, 19, 258, 258, (3, 3, 3), (1, 1, 1)),
    ("h3.enc.L1.256", 256, 256, 19, 130, 130, (3, 3, 3), (1, 1, 1)),
    ("h3.enc.L3.512", 512, 512, 7, 34, 34, (3, 3, 3), (1, 1, 1)),
    ("h3.enc.conv_in", 3, 128, 19, 258, 258, (3, 3, 3), (1, 1, 1)),
    ("h3.enc.shortcut1x1", 128, 256, 17, 128, 128, (1, 1, 1), (1, 1, 1)),
    ("q21.dec.1152@64", 1152, 1152, 3, 66, 66, (3, 3, 3), (1, 1, 1)),
    ("q21.dec.576@256", 576, 576, 3, 258, 258, (3, 3, 3), (1, 1, 1)),
    ("q21.dec.288@512", 288, 288, 3, 514, 514, (3, 3, 3), (1, 1, 1)),
    ("q21.dec.144@1024", 144, 144, 3, 1026, 1026, (3, 3, 3), (1, 1, 1)),
]
CONV_SHAPES_QUICK = [
    ("h3.enc.L0.128.T5", 128, 128, 7, 130, 130, (3, 3, 3), (1, 1, 1)),
    ("h3.enc.L3.512", 512, 512, 7, 34, 34, (3, 3, 3), (1, 1, 1)),
    ("q21.dec.1152@64", 1152, 1152, 3, 66, 66, (3, 3, 3), (1, 1, 1)),
    ("q21.dec.288@512", 288, 288, 3, 514, 514, (3, 3, 3), (1, 1, 1)),
]


def conv_shapes(quick: bool) -> list:
    return CONV_SHAPES_QUICK if quick else CONV_SHAPES_FULL
