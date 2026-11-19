#!/usr/bin/env python3
"""One SDPA kernel cell for unsloth #18403. Observes only; JSON to --out (written before the tested call,
so a process killed by a HIP fault still leaves its stage behind).

Shapes are the reporter's attention class: H=32 query heads, 8 KV heads, head_dim 128, T=2048.
--backend FLASH | EFFICIENT | MATH: the tested call runs under sdpa_kernel([that backend]).
--layout:
  seg_view_expanded_causal  Unsloth's _sdpa_packed_segments on ROCm: Q (1,T,H,D).transpose(1,2) (strided),
                            K/V repeat_interleave'd, one causal SDPA per segment (1300, 748) on views
                            x[0, :, s:e].unflatten(1, (1, L)).transpose(0, 1)
  gqa_causal_contig         contiguous (1,H,T,D) Q and (1,Hkv,T,D) K/V, enable_gqa=True, is_causal=True
  dense_mask_expanded       contiguous expanded K/V, bool block-diagonal causal (1,1,T,T) mask
Reference: the same layout through MATH in fp32 on the same device, computed first.
Status: ok | inaccurate | nonfinite | no_kernel | error (crash = no "done" stage; the parent records rc).
"""
from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--backend", required = True, choices = ("FLASH", "EFFICIENT", "MATH"))
ap.add_argument("--layout", required = True,
                choices = ("seg_view_expanded_causal", "gqa_causal_contig", "dense_mask_expanded"))
ap.add_argument("--dtype", required = True, choices = ("float16", "bfloat16"))
ap.add_argument("--out", required = True, type = Path)
a = ap.parse_args()
res: dict = {"backend": a.backend, "layout": a.layout, "dtype": a.dtype, "stage": "start"}
H, HKV, D, SEGS = 32, 8, 128, (1300, 748)
T = sum(SEGS)


def dump():
    a.out.write_text(json.dumps(res, indent = 1, default = str), encoding = "utf-8")


def main():
    import torch
    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel

    props = torch.cuda.get_device_properties(0)
    res["versions"] = {"torch": torch.__version__, "hip": torch.version.hip,
                       "arch": getattr(props, "gcnArchName", None)}
    dev = "cuda"
    g = torch.Generator().manual_seed(18403)
    q32 = torch.randn(1, T, H, D, generator = g)
    k32 = torch.randn(1, T, HKV, D, generator = g)
    v32 = torch.randn(1, T, HKV, D, generator = g)
    go32 = torch.randn(1, T, H, D, generator = g)
    groups = H // HKV

    def run(q_bthd, k_bthd, v_bthd, backend):
        """Leaves are (1,T,heads,D); returns (1,T,H,D)."""
        q = q_bthd.transpose(1, 2)
        k = k_bthd.transpose(1, 2)
        v = v_bthd.transpose(1, 2)
        with sdpa_kernel([backend]):
            if a.layout == "seg_view_expanded_causal":
                kx, vx = k.repeat_interleave(groups, dim = 1), v.repeat_interleave(groups, dim = 1)
                outs, s = [], 0
                for L in SEGS:
                    e = s + L

                    def split(x):
                        return x[0, :, s:e].unflatten(1, (1, L)).transpose(0, 1)

                    o = F.scaled_dot_product_attention(split(q), split(kx), split(vx), is_causal = True)
                    outs.append(o.transpose(1, 2).flatten(0, 1))
                    s = e
                return torch.cat(outs, dim = 0).unsqueeze(0)
            if a.layout == "gqa_causal_contig":
                o = F.scaled_dot_product_attention(q.contiguous(), k.contiguous(), v.contiguous(),
                                                   is_causal = True, enable_gqa = True)
                return o.transpose(1, 2)
            kx = k[:, :, None].expand(1, HKV, groups, T, D).reshape(1, H, T, D).contiguous()
            vx = v[:, :, None].expand(1, HKV, groups, T, D).reshape(1, H, T, D).contiguous()
            idx = torch.arange(T, device = q.device)
            seg = torch.repeat_interleave(torch.arange(len(SEGS), device = q.device),
                                          torch.tensor(SEGS, device = q.device))
            mask = (seg[None, :] == seg[:, None]) & (idx[None, :] <= idx[:, None])
            o = F.scaled_dot_product_attention(q.contiguous(), kx, vx, attn_mask = mask[None, None])
            return o.transpose(1, 2)

    def grads(dtype, backend):
        leaves = [x.to(dev, dtype).requires_grad_() for x in (q32, k32, v32)]
        out = run(*leaves, backend)
        dq, dk, dv = torch.autograd.grad(out, leaves, go32.to(dev, dtype))
        return [t.detach().float() for t in (out, dq, dk, dv)]

    res["stage"] = "reference"
    dump()
    want = grads(torch.float32, SDPBackend.MATH)
    torch.cuda.synchronize()
    dtype = getattr(torch, a.dtype)
    backend = getattr(SDPBackend, "EFFICIENT_ATTENTION" if a.backend == "EFFICIENT"
                      else "FLASH_ATTENTION" if a.backend == "FLASH" else "MATH")
    res["stage"] = "tested_call"
    dump()
    try:
        got = grads(dtype, backend)
        torch.cuda.synchronize()
    except RuntimeError as e:
        if "No available kernel" in str(e) or "No viable backend" in str(e):
            res.update({"status": "no_kernel", "error": str(e)[:600], "stage": "done"})
            return
        raise
    names = ("out", "dq", "dk", "dv")
    errs, finite, ok = {}, True, True
    for n, x, w in zip(names, got, want):
        fin = bool(torch.isfinite(x).all())
        mx = float((x - w).abs().max()) if fin else float("inf")
        ref = float(w.abs().max())
        errs[n] = {"max_abs": mx, "max_ref": ref, "rel": mx / ref if ref else mx}
        finite &= fin
        ok &= fin and mx <= 0.02 * ref + 1e-3
    res["errs"] = errs
    res["max_rel"] = max(e["rel"] for e in errs.values())
    res["status"] = "ok" if ok else ("nonfinite" if not finite else "inaccurate")
    res["stage"] = "done"


try:
    main()
except BaseException as e:  # noqa: BLE001
    res.update({"status": "error", "error": f"{type(e).__name__}: {str(e)[:1500]}",
                "traceback": traceback.format_exc()[-3000:], "stage": "done"})
dump()
