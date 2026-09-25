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


# (model key in gemm_shapes.json, stage, top-k quick, top-k full, M cap in quick)
_PROF_GEMM = [
    ("qwen_image_2.1", "denoiser", 3, 6, 4352),
    ("minimax_h3", "transformer", 3, 5, 8192),
    ("minimax_h3", "vae.decode", 2, 3, 2048),
    ("qwen_image_2.1", "text_encode", 0, 1, 64),
    ("minimax_h3", "block.MiniMaxH3TextEncoderStep", 0, 1, 64),
]
_TAG = {"qwen_image_2.1": "q21", "minimax_h3": "h3"}


def _profiler_gemms(table, quick: bool, src: str) -> list:
    out = []
    models = (table or {}).get("models", {})
    for model, stage, kq, kf, mcap in _PROF_GEMM:
        recs = [r for r in models.get(model, {}).get("gemm_by_stage", {}).get(stage, []) if r.get("batch", 1) == 1]
        recs.sort(key = lambda r: -float(r.get("pct_of_stage", 0)))
        for r in recs[: (kq if quick else kf)]:
            m = int(r["M"])
            out.append({"tag": f"{_TAG.get(model, model)}.{stage}.{r['K']}x{r['N']}", "M": min(m, mcap) if quick else m,
                        "K": int(r["K"]), "N": int(r["N"]), "bias": r.get("op") == "aten::addmm",
                        "source": f"profiler {os.path.basename(src)} {stage} {r.get('pct_of_stage')}% (M={m})"})
    return out


def gemm_shapes(quick: bool, ws: str | None = None, bundle: str | None = None) -> list[dict]:
    """[{tag, M, K, N, bias, source}], deduplicated on (M, K, N, bias)."""
    table, src = _profiler_table(ws, bundle)
    out = _profiler_gemms(table, quick, src) if table is not None else []
    if not out:
        qm, hm = (Q21_TOK_QUICK, H3_TOK_QUICK) if quick else (Q21_TOK, H3_TOK)
        for tag, k, n, b, _ in Q21_LINEARS:
            out.append({"tag": tag, "M": qm, "K": k, "N": n, "bias": b, "source": "config"})
        for tag, k, n, b, _ in H3_LINEARS:
            out.append({"tag": tag, "M": hm, "K": k, "N": n, "bias": b, "source": "config"})
        for tag, k, n, b, _ in H3_VAE_LINEARS[: (1 if quick else 3)]:
            out.append({"tag": tag, "M": 2048 if quick else H3_VAE_TOK, "K": k, "N": n, "bias": b, "source": "config"})
    seen, uniq = set(), []
    for s in out:
        key = (s["M"], s["K"], s["N"], s["bias"])
        if key not in seen:
            seen.add(key)
            uniq.append(s)
    return uniq


# Conv on an already padded, channels-last input (Studio / the VAEs pad outside the conv). Entries are
# (tag, input shape, weight shape, stride). H3 encoder (block_out_channels 128,256,256,512,512,1024) at a 256x256 tile,
# 17-frame clip, from the config; the Qwen-Image-2.1 VAE decode convs come from the profiler (2D 3x3 in image mode).
H3_ENC_CONV = [
    ("h3.enc.L0.128", (1, 128, 19, 258, 258), (128, 128, 3, 3, 3), 1),
    ("h3.enc.L1.256", (1, 256, 19, 130, 130), (256, 256, 3, 3, 3), 1),
    ("h3.enc.L3.512", (1, 512, 7, 34, 34), (512, 512, 3, 3, 3), 1),
    ("h3.enc.conv_in", (1, 3, 19, 258, 258), (128, 3, 3, 3, 3), 1),
]
H3_ENC_CONV_QUICK = [
    ("h3.enc.L0.128.T5", (1, 128, 7, 130, 130), (128, 128, 3, 3, 3), 1),
    ("h3.enc.L3.512", (1, 512, 7, 34, 34), (512, 512, 3, 3, 3), 1),
]
Q21_DEC_CONV = [
    ("q21.vae_decode.144@1024", (1, 144, 1026, 1026), (144, 144, 3, 3), 1),
    ("q21.vae_decode.1152@64", (1, 1152, 66, 66), (1152, 1152, 3, 3), 1),
    ("q21.vae_decode.1152@256", (1, 1152, 256, 256), (1152, 1152, 3, 3), 1),
    ("q21.vae_decode.288@512", (1, 288, 514, 514), (288, 288, 3, 3), 1),
]


def _profiler_convs(table, quick: bool) -> list:
    out = []
    for model, stages in (("qwen_image_2.1", ("vae_decode",)), ("minimax_h3", ("vae.decode",))):
        for st in stages:
            recs = (table or {}).get("models", {}).get(model, {}).get("conv_by_stage", {}).get(st, [])
            recs = sorted(recs, key = lambda r: -float(r.get("device_ms", 0)))
            for r in recs[: (3 if quick else 6)]:
                try:
                    x, w = json.loads(r["shapes"])
                except Exception:  # noqa: BLE001
                    continue
                if len(x) not in (4, 5) or len(w) != len(x) or w[1] != x[1]:
                    continue
                if w[0] * w[1] * (x[-1] * x[-2]) < 1 << 20:
                    continue
                tag = f"{_TAG.get(model, model)}.{st}.{w[1]}->{w[0]}@{x[-2]}"
                out.append((tag, tuple(x), tuple(w), 1))
    return out


def conv_shapes(quick: bool, ws: str | None = None, bundle: str | None = None) -> list:
    table, _ = _profiler_table(ws, bundle)
    prof = _profiler_convs(table, quick) if table is not None else []
    q21 = prof or (Q21_DEC_CONV[:2] if quick else Q21_DEC_CONV)
    return (H3_ENC_CONV_QUICK if quick else H3_ENC_CONV) + q21
