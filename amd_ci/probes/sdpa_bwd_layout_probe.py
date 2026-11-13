#!/usr/bin/env python3
"""Probe for unsloth#17548: do ROCm SDPA backward kernels return gradient strides other than their
fake kernels', so a compiled backward trips Inductor's stride assert? And what does this checkout's
`import unsloth` change about it?

Per torch env (the Studio venv, plus AMD's multi-arch torch 2.12.0+rocm7.14.1, the reporter's build):
  raw     : real vs FakeTensorMode strides of the flash / efficient backward aten ops over a shape
            matrix (GQA, head_dim 256, transposed vs contiguous inputs); then a torch.compile'd
            HF-layout attention block forward+backward per forced backend. No unsloth.
  unsloth : `import unsloth` from --checkout (path asserted), SDPA flags after import and its cost,
            the same compiled block on the default dispatch, then a tiny Qwen3.5 (GQA, head_dim 256)
            FastModel LoRA forward+backward with compile on.
Each step runs in a fresh child; results go to files, never stdout."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

_CHILD = r'''
import json, os, sys, time, traceback
mode, checkout, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
res = {"mode": mode}
def dump():
    with open(out_path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(res, f, default=str)
    os.replace(out_path + ".tmp", out_path)
def err(e):
    s = f"{type(e).__name__}: {e}"
    return s[:1200]
try:
    if mode == "unsloth":
        sys.path.insert(0, checkout)
        t0 = time.perf_counter()
        import unsloth
        res["import_unsloth_s"] = round(time.perf_counter() - t0, 3)
        res["unsloth_file"] = unsloth.__file__
        res["unsloth_from_checkout"] = os.path.realpath(unsloth.__file__).startswith(os.path.realpath(checkout))
        import unsloth.import_fixes as fx
        res["has_fix"] = hasattr(fx, "_rocm_sdpa_backward_layout_mismatches")
        if res["has_fix"]:
            t0 = time.perf_counter()
            res["fix_rescan"] = fx._rocm_sdpa_backward_layout_mismatches()
            res["fix_rescan_s"] = round(time.perf_counter() - t0, 3)
    import torch
    import torch.nn.functional as F
    from torch._subclasses.fake_tensor import FakeTensorMode
    cuda = torch.backends.cuda
    res["torch"] = torch.__version__
    res["hip"] = getattr(torch.version, "hip", None)
    res["available"] = torch.cuda.is_available()
    p = torch.cuda.get_device_properties(0)
    res["device"] = [p.name, getattr(p, "gcnArchName", None)]
    try:
        res["fa_library"] = str(cuda.preferred_rocm_fa_library())
    except Exception as e:
        res["fa_library"] = err(e)
    res["flags"] = {"flash": cuda.flash_sdp_enabled(), "mem_efficient": cuda.mem_efficient_sdp_enabled(),
                    "math": cuda.math_sdp_enabled()}
    dev = torch.device("cuda", 0)
    dtype = torch.bfloat16
    aten = torch.ops.aten

    def flash_bwd(q, k, v):
        o, lse, cq, ck, mq, mk, seed, off, _ = aten._scaled_dot_product_flash_attention(q, k, v, 0.0, True)
        return aten._scaled_dot_product_flash_attention_backward(
            torch.ones_like(o), q, k, v, o, lse, cq, ck, mq, mk, 0.0, True, seed, off)[:3]

    def eff_bwd(q, k, v):
        if q.shape[1] != k.shape[1]:
            r = q.shape[1] // k.shape[1]
            k, v = (t.repeat_interleave(r, 1) for t in (k, v))
        o, lse, seed, off = aten._scaled_dot_product_efficient_attention(q, k, v, None, True, 0.0, True)
        return aten._scaled_dot_product_efficient_attention_backward(
            torch.ones_like(o), q, k, v, None, o, lse, seed, off, 0.0, [True, True, True, False], True)[:3]

    SHAPES = [(1, 93, 24, 4, 256), (2, 93, 4, 2, 256), (1, 16, 2, 2, 64), (1, 93, 8, 8, 128)]
    if mode == "raw":
        rows = []
        for name, fn in (("flash", flash_bwd), ("mem_efficient", eff_bwd)):
            for (B, S, Hq, Hk, D) in SHAPES:
                for layout in ("bshd", "contiguous"):
                    row = {"backend": name, "shape": [B, S, Hq, Hk, D], "layout": layout}
                    try:
                        def mk(H):
                            if layout == "bshd":
                                return torch.randn(B, S, H, D, device=dev, dtype=dtype).transpose(1, 2)
                            return torch.randn(B, H, S, D, device=dev, dtype=dtype)
                        q, k, v = mk(Hq), mk(Hk), mk(Hk)
                        real = fn(q, k, v)
                        torch.cuda.synchronize()
                        mode_ = FakeTensorMode()
                        fq, fk, fv = (mode_.from_tensor(t) for t in (q, k, v))
                        with mode_:
                            fake = fn(fq, fk, fv)
                        row["real"] = [list(t.stride()) for t in real]
                        row["fake"] = [list(t.stride()) for t in fake]
                        row["mismatch"] = row["real"] != row["fake"]
                    except Exception as e:
                        row["error"] = err(e)
                    rows.append(row)
        res["raw_ops"] = rows

    def block(B, S, Hq, Hk, D):
        hid = 64
        x = torch.randn(B, S, hid, device=dev, dtype=dtype)
        ws = [(torch.randn(hid, h * D, device=dev, dtype=dtype) * 0.05).requires_grad_() for h in (Hq, Hk, Hk)]
        def f(x, wq, wk, wv):
            q = (x @ wq).view(B, S, Hq, D).transpose(1, 2)
            k = (x @ wk).view(B, S, Hk, D).transpose(1, 2)
            v = (x @ wv).view(B, S, Hk, D).transpose(1, 2)
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=Hq != Hk)
            return o.transpose(1, 2).reshape(B, S, Hq * D)
        return f, x, ws

    def compiled_case(B, S, Hq, Hk, D, backend=None):
        from torch.nn.attention import sdpa_kernel, SDPBackend
        torch._dynamo.reset()
        f, x, ws = block(B, S, Hq, Hk, D)
        row = {"shape": [B, S, Hq, Hk, D], "backend": backend or "default"}
        try:
            import contextlib
            ctx = sdpa_kernel([getattr(SDPBackend, backend)]) if backend else contextlib.nullcontext()
            with ctx:
                out = torch.compile(f)(x, *ws)
                out.float().square().mean().backward()
            torch.cuda.synchronize()
            row["ok"] = all(w.grad is not None and bool(torch.isfinite(w.grad).all()) for w in ws)
        except Exception as e:
            row["ok"] = False
            row["error"] = err(e)
            row["stride_assert"] = "stride" in str(e) and "expected size" in str(e)
        return row

    if mode == "raw":
        res["compiled_forced"] = [compiled_case(*s, backend=b) for b in ("FLASH_ATTENTION", "EFFICIENT_ATTENTION", "MATH")
                                  for s in SHAPES[:2]]
        res["compiled_default_no_unsloth"] = [compiled_case(*s) for s in SHAPES[:2]]
    else:
        res["compiled_default"] = [compiled_case(*s) for s in SHAPES[:2]]
        # Tiny Qwen3.5 (4 q heads, 2 kv heads, head_dim 256), Unsloth's compiled path.
        tq = {"steps": []}
        try:
            from unsloth import FastModel
            model, proc = FastModel.from_pretrained(
                "trl-internal-testing/tiny-Qwen3_5ForConditionalGeneration-NoThink",
                max_seq_length=256, load_in_4bit=False, dtype=torch.bfloat16)
            model = FastModel.get_peft_model(model, r=8, lora_alpha=16, lora_dropout=0, bias="none",
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                                             use_gradient_checkpointing="unsloth", random_state=3407)
            tok = getattr(proc, "tokenizer", proc)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
            model.train()
            texts = [" ".join(["hello world"] * 40), "short text " * 7]
            for step, (bs, pad) in enumerate([(1, False), (2, True), (1, False)]):
                batch = tok(texts[:bs], return_tensors="pt", padding=pad).to(dev)
                labels = batch["input_ids"].clone()
                if pad:
                    labels[batch["attention_mask"] == 0] = -100
                try:
                    loss = model(**batch, labels=labels).loss
                    loss.backward()
                    opt.step(); opt.zero_grad()
                    tq["steps"].append({"bs": bs, "pad": pad, "loss": float(loss)})
                except Exception as e:
                    tq["steps"].append({"bs": bs, "pad": pad, "error": err(e),
                                        "stride_assert": "stride" in str(e) and "expected size" in str(e)})
            tq["ok"] = all("loss" in s for s in tq["steps"])
        except Exception as e:
            tq["setup_error"] = err(e)
            tq["trace"] = traceback.format_exc()[-3000:]
        res["tiny_qwen35"] = tq
    res["flags_end"] = {"flash": cuda.flash_sdp_enabled(), "mem_efficient": cuda.mem_efficient_sdp_enabled(),
                        "math": cuda.math_sdp_enabled()}
except BaseException as e:
    res["fatal"] = err(e)
    res["trace"] = traceback.format_exc()[-3000:]
dump()
'''

AMD_INDEX = "https://repo.amd.com/rocm/whl-multi-arch/"
AMD_TORCH = "2.12.0+rocm7.14.1"
SKIP_MIRROR = ("torch", "torchvision", "torchaudio", "triton", "pytorch-triton-rocm", "rocm", "amd-", "unsloth",
               "bitsandbytes", "xformers", "flash-attn", "flash_attn", "torchao", "vllm")


def _log(msg: str) -> None:
    print(msg, flush = True)


def amd_venv(studio_py: str, arch: str) -> str | None:
    """AMD's torch 2.12.0+rocm7.14.1 in its own venv, the Studio venv's other packages mirrored --no-deps.
    Built once per job; later states reuse it."""
    root = Path(os.environ.get("RUNNER_TEMP", tempfile.gettempdir())) / "amd714_venv"
    py = root / "bin" / "python"
    done = root / ".ready"
    if done.is_file():
        return str(py)
    uv = shutil.which("uv") or next((str(c) for c in (Path.home() / ".local/bin/uv", Path.home() / ".cargo/bin/uv")
                                     if c.is_file()), None)
    if uv is None:
        _log("no uv on this runner")
        return None
    ver = subprocess.run([studio_py, "-c", "import sys;print(f'{sys.version_info[0]}.{sys.version_info[1]}')"],
                         capture_output = True, text = True).stdout.strip()
    steps = [
        [uv, "venv", "--python", ver, str(root)],
        [uv, "pip", "install", "--python", str(py), "--index-url", AMD_INDEX, "--index-strategy", "unsafe-best-match",
         f"torch[device-{arch}]=={AMD_TORCH}", "torchvision==0.27.0+rocm7.14.1"],
    ]
    for cmd in steps:
        r = subprocess.run(cmd, capture_output = True, text = True)
        _log(f"$ {' '.join(cmd)} -> {r.returncode}\n{r.stdout[-1500:]}{r.stderr[-3000:]}")
        if r.returncode != 0:
            return None
    freeze = subprocess.run([uv, "pip", "freeze", "--python", studio_py], capture_output = True, text = True).stdout
    reqs = [l for l in freeze.splitlines()
            if l and not l.startswith(("-e", "#")) and " @ " not in l
            and not l.lower().replace("_", "-").startswith(SKIP_MIRROR)]
    req = root / "mirror.txt"
    req.write_text("\n".join(reqs) + "\n", encoding = "utf-8")
    r = subprocess.run([uv, "pip", "install", "--python", str(py), "--no-deps", "--extra-index-url", "https://pypi.org/simple", "-r", str(req)],
                       capture_output = True, text = True)
    _log(f"mirror {len(reqs)} pkgs -> {r.returncode}\n{r.stderr[-3000:]}")
    if r.returncode != 0:
        return None
    done.write_text("ok", encoding = "utf-8")
    return str(py)


def run_child(py: str, mode: str, checkout: str, td: Path, tag: str, env_extra: dict) -> dict:
    res = td / f"{tag}_{mode}.json"
    env = dict(os.environ)
    env.update(env_extra)
    env["UNSLOTH_COMPILE_LOCATION"] = str(td / f"cache_{tag}_{mode}")
    env["TORCHINDUCTOR_CACHE_DIR"] = str(td / f"inductor_{tag}_{mode}")
    env.pop("PYTHONPATH", None)
    t0 = time.time()
    try:
        r = subprocess.run([py, "-c", _CHILD, mode, checkout, str(res)], cwd = str(td), env = env,
                           capture_output = True, text = True, timeout = 1500)
        rc, tail = r.returncode, (r.stdout[-2000:] + "\n" + r.stderr[-4000:])
    except subprocess.TimeoutExpired:
        rc, tail = "timeout", ""
    out = json.loads(res.read_text(encoding = "utf-8")) if res.is_file() else {"missing": True}
    out["_rc"], out["_secs"], out["_tail"] = rc, round(time.time() - t0, 1), tail
    _log(f"[{tag}/{mode}] rc={rc} {out['_secs']}s")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--no-amd-torch", action = "store_true")
    args = ap.parse_args()
    checkout = os.path.realpath(args.checkout)
    arch = subprocess.run([sys.executable, "-c", "import torch;print(torch.cuda.get_device_properties(0).gcnArchName.split(':')[0])"],
                          capture_output = True, text = True).stdout.strip() or "gfx1151"
    envs = {"studio": sys.executable}
    if not args.no_amd_torch:
        py = amd_venv(sys.executable, arch)
        envs["amd_2_12_rocm7_14_1"] = py
    obs: dict = {"state": args.state, "checkout": checkout, "arch": arch, "envs": {}}
    with tempfile.TemporaryDirectory() as td_:
        td = Path(td_)
        for tag, py in envs.items():
            if py is None:
                obs["envs"][tag] = {"unavailable": "venv build failed (see probe log)"}
                continue
            e = {}
            e["raw"] = run_child(py, "raw", checkout, td, tag, {})
            e["unsloth"] = run_child(py, "unsloth", checkout, td, tag, {})
            obs["envs"][tag] = e
    args.out.write_text(json.dumps(obs, default = str), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
