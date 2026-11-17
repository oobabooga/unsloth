#!/usr/bin/env python3
"""Probe: Windows ROCm SDPA at head dim 512 on whichever torch this venv holds, with and without the state's
wide-head guard (unsloth/import_fixes.py, #17976). Observes only (judged by criteria/wide_head_sdpa_torch.py).

The guard is loaded from the state's own checkout by AST extraction, so no `import unsloth` (and none of its
dependencies) is needed: this runs on nightly torch builds the rest of the stack may not support yet. The base tree
has no guard, so base = stock torch dispatch, head = guarded dispatch.

Tasks, each in its own process (a native crash in a broken kernel is an exit code, not a lost probe):
  kernels   Gemma 4 shaped attention (8 query heads, 1 KV head, bf16) at head dim 256 and 512, seq 256..4096,
            causal / boolean mask, GQA expanded and enable_gqa, default and unit scale. For each: every backend
            forced (flash, efficient, math) and the default dispatch, error vs a float32 math reference, and the
            aten op the default dispatch ran (profiler). Indicative ms per call for the default dispatch at 512.
  hf_fwd    transformers' Gemma 4 E2B forward at 2048 tokens, attn "eager" vs "sdpa": top-1 agreement, KL, NLL.
JSON to --out only.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

FLAG = "_unsloth_windows_rocm_wide_head_guard"
MODEL = "unsloth/gemma-4-E2B-it"
GUARD_NAMES = {
    "_ROCM_WINDOWS_WIDE_HEAD_FLAG", "ROCM_WINDOWS_WIDEST_FUSED_HEAD_DIM", "_ROCM_WINDOWS_WIDE_HEAD_DEVICE_TYPES",
    "_is_windows_rocm_torch", "_sdpa_has_flag", "_sdpa_head_too_wide", "_ROCM_WINDOWS_WIDE_HEAD_ORIGINALS",
    "_ROCM_WINDOWS_WIDE_HEAD_HELD", "_ROCM_WINDOWS_WIDE_HEAD_LOCK", "_rocm_windows_math_on",
    "_rocm_windows_math_off", "_sdpa_math_only", "rocm_windows_wide_head_sdpa", "fix_rocm_windows_wide_head_sdpa",
}
SDPA_OPS = (
    "aten::_scaled_dot_product_flash_attention",
    "aten::_scaled_dot_product_efficient_attention",
    "aten::_scaled_dot_product_attention_math",
    "aten::_scaled_dot_product_cudnn_attention",
    "aten::_scaled_dot_product_fused_attention_overrideable",
    "aten::_scaled_dot_product_flash_attention_for_cpu",
)
PARAGRAPH = (
    "The lighthouse keeper climbed the spiral stairs every evening at dusk, counting one hundred and twelve steps "
    "before he reached the lamp room. He wiped the brass fittings, trimmed the wick, and wrote the weather in a "
    "leather notebook: wind from the north west, a falling barometer, fog banks moving in from the outer islands. "
)


def _write(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent = 2, default = str), encoding = "utf-8")


def install_guard(checkout: str) -> dict:
    """Exec the guard's definitions from the checkout and install it; {} facts when the tree has none."""
    src = (Path(checkout) / "unsloth" / "import_fixes.py").read_text(encoding = "utf-8")
    tree = ast.parse(src)
    keep = []
    for node in tree.body:
        names = set()
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            names = {node.name}
        elif isinstance(node, ast.Assign):
            names = {t.id for t in node.targets if isinstance(t, ast.Name)}
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names = {node.target.id}
        if names & GUARD_NAMES:
            keep.append(node)
    found = sorted({getattr(n, "name", None) or n.targets[0].id for n in keep})
    facts = {"guard_defs_found": found}
    if "fix_rocm_windows_wide_head_sdpa" not in found:
        facts["guard_in_tree"] = False
        return facts
    facts["guard_in_tree"] = True
    import functools
    import logging
    import threading

    ns = {"os": os, "sys": sys, "threading": threading, "functools": functools,
          "logger": logging.getLogger("unsloth_probe"), "__name__": "unsloth_import_fixes_extract"}
    exec(compile(ast.Module(body = keep, type_ignores = []), "import_fixes_extract", "exec"), ns)
    if os.environ.get("WHS_PROBE_FAKE_WINDOWS_ROCM") == "1":  # harness smoke test on a CPU host only
        ns["_is_windows_rocm_torch"] = lambda: True
        ns["_ROCM_WINDOWS_WIDE_HEAD_DEVICE_TYPES"] = ("cuda", "cpu")
        facts["faked_windows_rocm"] = True
    facts["is_windows_rocm_torch"] = bool(ns["_is_windows_rocm_torch"]())
    facts["fix_returned"] = bool(ns["fix_rocm_windows_wide_head_sdpa"]())
    return facts


def guard_installed() -> bool:
    import torch.nn.functional as F

    fn = F.scaled_dot_product_attention
    for _ in range(32):
        if fn is None:
            return False
        if getattr(fn, FLAG, False):
            return True
        fn = getattr(fn, "__wrapped__", None)
    return False


def env_facts(res: dict) -> None:
    import torch

    res["torch"] = torch.__version__
    res["hip"] = getattr(torch.version, "hip", None)
    res["platform"] = sys.platform
    res["python"] = sys.version.split()[0]
    res["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        res["arch"] = getattr(p, "gcnArchName", None)
        res["device"] = p.name
    res["aotriton_experimental"] = os.environ.get("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL")


def _cases():
    for d in (256, 512):
        for s in (256, 1024, 2048, 4096):
            for mask in ("causal", "bool_mask"):
                for gqa in ("expanded", "enable_gqa"):
                    for scale in ("default", "unit"):
                        if s != 2048 and (gqa != "expanded" or scale != "default"):
                            continue  # full grid at 2048, a thinner one elsewhere
                        yield {"d": d, "s": s, "mask": mask, "gqa": gqa, "scale": scale}


def _inputs(torch, c, dev):
    g = torch.Generator(device = "cpu").manual_seed(1000 + c["d"] + c["s"])
    hq, hkv = 8, 1
    q = torch.randn(1, hq, c["s"], c["d"], generator = g)
    k = torch.randn(1, hkv, c["s"], c["d"], generator = g)
    v = torch.randn(1, hkv, c["s"], c["d"], generator = g)
    if c["scale"] == "unit":  # Gemma-style pre-normalised queries with scale 1.0: keep logits in a sane range
        q = q / c["d"] ** 0.5
    q, k, v = (t.to(dev, torch.bfloat16) for t in (q, k, v))
    kw = {}
    if c["mask"] == "causal":
        kw["is_causal"] = True
    else:
        m = torch.ones(c["s"], c["s"], dtype = torch.bool).tril()
        m[:, : c["s"] // 8] = True  # a prefix every query sees, so the mask is not plain causal
        kw["attn_mask"] = m.to(dev)[None, None]
    if c["scale"] == "unit":
        kw["scale"] = 1.0
    if c["gqa"] == "expanded":
        k, v = (t.expand(1, hq, c["s"], c["d"]).contiguous() for t in (k, v))
    else:
        kw["enable_gqa"] = True
    return q, k, v, kw


def _err(torch, out, ref):
    o = out.float()
    finite = bool(torch.isfinite(o).all())
    rel = float((o - ref).norm() / ref.norm()) if finite else float("inf")
    return {"finite": finite, "rel_l2": round(rel, 6), "max_abs": round(float((o - ref).abs().max()), 5) if finite else None}


def _ops(prof) -> list:
    return sorted({ev.name.split("::")[-1] for ev in prof.events() if ev.name in SDPA_OPS})


def task_kernels(res: dict) -> None:
    import torch
    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from torch.profiler import ProfilerActivity, profile

    dev = os.environ.get("WHS_PROBE_DEVICE", "cuda")  # "cpu": harness smoke test only
    sync = torch.cuda.synchronize if dev == "cuda" else (lambda: None)
    raw = torch._C._nn.scaled_dot_product_attention  # the builtin, never the guard: forced rows stay forced
    rows = []
    for c in _cases():
        row = dict(c)
        try:
            q, k, v, kw = _inputs(torch, c, dev)
            ref_kw = dict(kw)
            qf, kf, vf = q.float(), k.float(), v.float()
            with sdpa_kernel([SDPBackend.MATH]):
                ref = raw(qf, kf, vf, **ref_kw).float()
            for name, be in (("flash", SDPBackend.FLASH_ATTENTION), ("efficient", SDPBackend.EFFICIENT_ATTENTION),
                             ("math", SDPBackend.MATH)):
                try:
                    with sdpa_kernel([be]):
                        out = raw(q, k, v, **kw)
                    sync()
                    row[name] = _err(torch, out, ref)
                except Exception as e:  # noqa: BLE001
                    row[name] = {"refused": f"{type(e).__name__}: {str(e)[:160]}"}
            # Default dispatch through F (the guard, when installed, sits on F.scaled_dot_product_attention).
            with profile(activities = [ProfilerActivity.CPU]) as prof:
                out = F.scaled_dot_product_attention(q, k, v, **kw)
                sync()
            row["default"] = _err(torch, out, ref)
            row["default_ops"] = _ops(prof)
            if c["d"] == 512 and c["s"] == 2048 and c["gqa"] == "expanded" and c["scale"] == "default":
                for _ in range(3):
                    F.scaled_dot_product_attention(q, k, v, **kw)
                sync()
                ts = []
                for _ in range(10):
                    t0 = time.perf_counter()
                    F.scaled_dot_product_attention(q, k, v, **kw)
                    sync()
                    ts.append((time.perf_counter() - t0) * 1e3)
                row["default_ms_median"] = round(sorted(ts)[len(ts) // 2], 3)
            del q, k, v, ref, out
            torch.cuda.empty_cache() if dev == "cuda" else None
        except Exception as e:  # noqa: BLE001
            row["error"] = f"{type(e).__name__}: {str(e)[:300]}"
        rows.append(row)
    res["kernels"] = rows


def task_hf_fwd(res: dict) -> None:
    import torch
    import transformers

    out = {"transformers": transformers.__version__}
    tok = transformers.AutoTokenizer.from_pretrained(MODEL, token = False)
    ids = tok(PARAGRAPH * 80, return_tensors = "pt").input_ids[:, :2048].to("cuda")
    out["tokens"] = int(ids.shape[1])
    logits = {}
    for impl in ("eager", "sdpa"):
        model = None
        # transformers 5.23 refuses sdpa for Gemma4AudioModel as a whole-model string; the text tower (the
        # head-512 global layers under test) takes it per sub-config, with the audio / vision towers on eager.
        per_tower = {"text_config": impl, "vision_config": "eager", "audio_config": "eager"}
        for cls in ("AutoModelForCausalLM", "AutoModelForImageTextToText"):
            for label, attn in (("str", impl), ("text_only", per_tower)):
                try:
                    model = getattr(transformers, cls).from_pretrained(
                        MODEL, dtype = torch.bfloat16, attn_implementation = attn, token = False)
                    out["attn_implementation"] = attn
                    break
                except Exception as e:  # noqa: BLE001
                    out[f"load_{cls}_{impl}_{label}"] = f"{type(e).__name__}: {str(e)[:600]}"
            if model is not None:
                break
        if model is None:
            res["hf_fwd"] = out
            raise RuntimeError(f"no class loaded {MODEL} with {impl}: {out}")
        out["class"] = type(model).__name__
        model = model.to("cuda").eval()
        with torch.no_grad():
            logits[impl] = model(input_ids = ids).logits[0].float().cpu()
            if impl == "sdpa":
                from torch.profiler import ProfilerActivity, profile

                with profile(activities = [ProfilerActivity.CPU], record_shapes = True) as prof:
                    model(input_ids = ids[:, :256])
                counts = {}
                for ev in prof.events():
                    if ev.name in SDPA_OPS:
                        shapes = getattr(ev, "input_shapes", None) or []
                        head = shapes[0][-1] if shapes and shapes[0] else None
                        key = f"{ev.name.split('::')[-1]}@{head}"
                        counts[key] = counts.get(key, 0) + 1
                out["sdpa_ops"] = counts
        del model
        torch.cuda.empty_cache()
    e, s = logits["eager"], logits["sdpa"]
    lp_e, lp_s = e.log_softmax(-1), s.log_softmax(-1)
    out["finite"] = bool(torch.isfinite(s).all() and torch.isfinite(e).all())
    out["top1_agree"] = round(float((e.argmax(-1) == s.argmax(-1)).float().mean()), 5)
    out["kl_eager_sdpa_mean"] = round(float((lp_e.exp() * (lp_e - lp_s)).sum(-1).mean()), 6)
    tgt = ids[0, 1:].cpu()
    out["nll_eager"] = round(float(torch.nn.functional.cross_entropy(e[:-1], tgt)), 5)
    out["nll_sdpa"] = round(float(torch.nn.functional.cross_entropy(s[:-1], tgt)), 5)
    res["hf_fwd"] = out


def child(args) -> int:
    res: dict = {"task": args.child}
    try:
        res["guard"] = install_guard(args.checkout)
        res["guard_installed"] = guard_installed()
        env_facts(res)
        {"kernels": task_kernels, "hf_fwd": task_hf_fwd}[args.child](res)
    except Exception as e:  # noqa: BLE001
        res["error"] = f"{type(e).__name__}: {e}"
        res["traceback"] = traceback.format_exc()[-3000:]
    _write(args.out, res)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--tasks", default = "kernels,hf_fwd")
    ap.add_argument("--child", default = None, help = argparse.SUPPRESS)
    args = ap.parse_args()
    args.checkout = str(Path(args.checkout).resolve())
    if args.child:
        return child(args)
    res: dict = {"state": args.state, "checkout": args.checkout, "tasks": {}}
    for task in [t for t in args.tasks.split(",") if t]:
        out = args.out.with_name(f"{args.out.stem}_{task}.json")
        t0 = time.time()
        cmd = [sys.executable, __file__, "--state", args.state, "--checkout", args.checkout, "--out", str(out),
               "--child", task]
        try:
            rc = subprocess.run(cmd, timeout = 5400).returncode
        except subprocess.TimeoutExpired:
            rc = "timeout"
        got = json.loads(out.read_text(encoding = "utf-8")) if out.is_file() else {"missing_output": True}
        got["rc"], got["seconds"] = rc, round(time.time() - t0, 1)
        res["tasks"][task] = got
        for k in ("torch", "hip", "platform", "arch", "device", "python", "guard_installed", "guard"):
            if k in got and k not in res:
                res[k] = got[k]
    _write(args.out, res)
    return 0


if __name__ == "__main__":
    sys.exit(main())
