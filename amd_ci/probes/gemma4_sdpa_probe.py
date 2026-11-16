#!/usr/bin/env python3
"""Probe: torch SDPA at Gemma 4's global-attention shapes, per backend, against an fp64 reference; and one real Gemma 4
forward with attn_implementation "sdpa" vs "eager". Observes only (judged by criteria/gemma4_sdpa.py).

Kernel arms: head dim 512, bf16, causal, GQA with the real head counts (E2B: 8 q / 1 kv; 26B-A4B: 16 q / 2 kv), seq
128 / 2048 / 8192, causal as ``is_causal=True`` and as an explicit boolean mask, GQA as ``enable_gqa=True`` and as kv
repeated to the q heads (what transformers' sdpa path does when it does not pass enable_gqa), per backend (default
dispatch, flash, efficient, math). Forward output and the q / k / v gradients against fp64 math. JSON to --out only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

HEAD_CFGS = ((8, 1), (16, 2))
SEQS = (128, 2048, 8192)
D = 512
MODEL = "unsloth/gemma-4-E2B-it"


def _write(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent = 2, default = str), encoding = "utf-8")


def kernels(torch, res: dict, out: Path) -> None:
    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel

    backends = (("default", None), ("flash", SDPBackend.FLASH_ATTENTION),
                ("efficient", SDPBackend.EFFICIENT_ATTENTION), ("math", SDPBackend.MATH))
    arms = res.setdefault("arms", {})
    for hq, hkv in HEAD_CFGS:
        for n in SEQS:
            g = torch.Generator(device = "cuda").manual_seed(n * 31 + hq)
            q = torch.randn(1, hq, n, D, device = "cuda", generator = g, dtype = torch.float64)
            k = torch.randn(1, hkv, n, D, device = "cuda", generator = g, dtype = torch.float64)
            v = torch.randn(1, hkv, n, D, device = "cuda", generator = g, dtype = torch.float64)
            go = torch.randn(1, hq, n, D, device = "cuda", generator = g, dtype = torch.float64)
            rep = hq // hkv
            # fp64 reference: explicit causal softmax attention, kv repeated.
            try:
                qr, kr, vr = (t.clone().requires_grad_(True) for t in (q, k, v))
                kk, vv = kr.repeat_interleave(rep, 1), vr.repeat_interleave(rep, 1)
                s = (qr @ kk.transpose(-1, -2)) / (D ** 0.5)
                s = s.masked_fill(torch.ones(n, n, dtype = torch.bool, device = "cuda").triu(1), float("-inf"))
                ref = s.softmax(-1) @ vv
                ref.backward(go)
                ref_o, ref_g = ref.detach(), [t.grad.detach() for t in (qr, kr, vr)]
                del qr, kr, vr, kk, vv, s, ref
            except Exception as exc:  # noqa: BLE001
                arms[f"h{hq}x{hkv}_n{n}"] = {"reference_error": f"{type(exc).__name__}: {str(exc)[:200]}"}
                torch.cuda.empty_cache()
                continue
            mask = torch.ones(n, n, dtype = torch.bool, device = "cuda").tril()

            def rel(a, b):
                a = a.double()
                return round(float((a - b).norm() / b.norm()), 6) if torch.isfinite(a).all() else "nonfinite"

            for mname in ("is_causal", "bool_mask"):
                for gqa in ("enable_gqa", "repeat_kv"):
                    for bname, be in backends:
                        key = f"h{hq}x{hkv}_n{n}_{mname}_{gqa}_{bname}"
                        try:
                            qb, kb, vb = (t.to(torch.bfloat16).requires_grad_(True) for t in (q, k, v))
                            kin, vin = (kb, vb) if gqa == "enable_gqa" else (kb.repeat_interleave(rep, 1),
                                                                               vb.repeat_interleave(rep, 1))
                            kw = {"enable_gqa": True} if gqa == "enable_gqa" else {}
                            if mname == "is_causal":
                                kw["is_causal"] = True
                            else:
                                kw["attn_mask"] = mask
                            ctx = sdpa_kernel([be]) if be is not None else _Null()
                            with ctx:
                                o = F.scaled_dot_product_attention(qb, kin, vin, **kw)
                                o.backward(go.to(torch.bfloat16))
                            torch.cuda.synchronize()
                            arms[key] = {"out": rel(o.detach(), ref_o),
                                         "dq": rel(qb.grad, ref_g[0]), "dk": rel(kb.grad, ref_g[1]),
                                         "dv": rel(vb.grad, ref_g[2])}
                        except Exception as exc:  # noqa: BLE001
                            arms[key] = {"refused": f"{type(exc).__name__}: {str(exc)[:120]}"}
                        finally:
                            torch.cuda.empty_cache()
                        print(key, arms[key], flush = True)
            del q, k, v, go, ref_o, ref_g, mask
            torch.cuda.empty_cache()
            _write(out, res)


class _Null:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def model_forward(torch, res: dict) -> None:
    import transformers

    tok = transformers.AutoTokenizer.from_pretrained(MODEL, token = False)
    text = ("The quick brown fox jumps over the lazy dog while the tarot reader shuffles the deck. " * 400)
    ids = tok(text, return_tensors = "pt").input_ids[:, :2048].cuda()
    out = {"model": MODEL, "tokens": int(ids.shape[1]), "transformers": transformers.__version__}
    logits = {}
    for impl in ("eager", "sdpa"):
        t0 = time.time()
        model = None
        for cls in ("AutoModelForCausalLM", "AutoModelForImageTextToText"):
            try:
                model = getattr(transformers, cls).from_pretrained(MODEL, torch_dtype = torch.bfloat16,
                                                                   attn_implementation = impl, token = False)
                out["class"] = type(model).__name__
                break
            except Exception as exc:  # noqa: BLE001
                out[f"load_{cls}_{impl}"] = f"{type(exc).__name__}: {str(exc)[:200]}"
        if model is None:
            continue
        model = model.cuda().eval()
        out[f"impl_{impl}"] = getattr(model.config, "_attn_implementation", None)
        with torch.no_grad():
            logits[impl] = model(input_ids = ids).logits[0].float().cpu()
        out[f"{impl}_s"] = round(time.time() - t0, 1)
        del model
        torch.cuda.empty_cache()
    if len(logits) == 2:
        e, s = logits["eager"], logits["sdpa"]
        lp_e, lp_s = e.log_softmax(-1), s.log_softmax(-1)
        out["finite"] = bool(torch.isfinite(s).all() and torch.isfinite(e).all())
        out["rel_l2"] = round(float((s - e).norm() / e.norm()), 6)
        out["top1_agree"] = round(float((e.argmax(-1) == s.argmax(-1)).float().mean()), 5)
        out["kl_eager_sdpa_mean"] = round(float((lp_e.exp() * (lp_e - lp_s)).sum(-1).mean()), 6)
        out["kl_eager_sdpa_max"] = round(float((lp_e.exp() * (lp_e - lp_s)).sum(-1).max()), 6)
        # Next-token loss of each on the text itself: a broken attention shows as a loss jump.
        tgt = ids[0, 1:].cpu()
        out["nll_eager"] = round(float(torch.nn.functional.cross_entropy(e[:-1], tgt)), 5)
        out["nll_sdpa"] = round(float(torch.nn.functional.cross_entropy(s[:-1], tgt)), 5)
    res["model_forward"] = out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--model-state", default = "head", help = "state that also runs the real Gemma 4 forward")
    ap.add_argument("--skip-kernels", action = "store_true", help = "only the real model forward")
    ap.add_argument("--studio-guard", action = "store_true",
                    help = "install the checkout's Windows ROCm wide-head SDPA guard when the state has one (as the "
                           "Studio LLM workers do), so base = stock dispatch and head = guarded")
    ap.add_argument("--child-forward", action = "store_true", help = argparse.SUPPRESS)
    ap.add_argument("--seqs", default = "", help = "comma list overriding the kernel arms' sequence lengths")
    args = ap.parse_args()
    os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")  # what Studio's main.py sets
    import torch

    res: dict = {"state": args.state, "torch": torch.__version__, "hip": getattr(torch.version, "hip", None),
                 "platform": sys.platform, "aotriton_experimental": os.environ.get("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL")}
    if torch.cuda.is_available():
        res["arch"] = getattr(torch.cuda.get_device_properties(0), "gcnArchName", None)
    if args.seqs:
        global SEQS
        SEQS = tuple(int(x) for x in args.seqs.split(","))
    if args.studio_guard:
        backend = Path(args.checkout) / "studio" / "backend"
        res["guard_module_present"] = (backend / "core" / "_rocm_windows_sdpa.py").is_file()
        if res["guard_module_present"]:
            sys.path.insert(0, str(backend))
            from core._rocm_windows_sdpa import install_windows_rocm_wide_head_sdpa_guard

            res["guard_installed"] = install_windows_rocm_wide_head_sdpa_guard()
    _write(args.out, res)
    res["kernels_skipped"] = bool(args.skip_kernels)
    try:
        if not args.skip_kernels:
            kernels(torch, res, args.out)
    except Exception as exc:  # noqa: BLE001
        res["kernels_error"] = f"{type(exc).__name__}: {exc}"
        res["kernels_traceback"] = traceback.format_exc()[-3000:]
    _write(args.out, res)
    if args.child_forward:
        try:
            model_forward(torch, res)
        except Exception as exc:  # noqa: BLE001
            res["model_forward"] = {"error": f"{type(exc).__name__}: {str(exc)[:500]}",
                                    "traceback": traceback.format_exc()[-3000:]}
        _write(args.out, res)
        return 0
    if args.model_state in (args.state, "all"):
        # Its own process: a native crash in a broken kernel is recorded (exit code), not lost with the probe.
        import subprocess

        fwd_out = args.out.with_name(args.out.stem + "_forward.json")
        cmd = [sys.executable, __file__, "--state", args.state, "--checkout", args.checkout, "--out", str(fwd_out),
               "--skip-kernels", "--child-forward"] + (["--studio-guard"] if args.studio_guard else [])
        with open(args.out.with_name(args.out.stem + "_forward.log"), "wb") as fh:
            rc = subprocess.run(cmd, stdout = fh, stderr = subprocess.STDOUT).returncode
        try:
            child = json.loads(fwd_out.read_text(encoding = "utf-8"))
        except Exception:  # noqa: BLE001
            child = {}
        res["model_forward"] = child.get("model_forward") or {"error": f"forward process exited {rc} with no result"}
        res["model_forward"]["_rc"] = rc
        res["model_forward"]["_guard_installed"] = child.get("guard_installed")
    _write(args.out, res)
    return 0


if __name__ == "__main__":
    sys.exit(main())
