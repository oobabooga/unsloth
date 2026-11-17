#!/usr/bin/env python3
"""Observe-only diagnostic for check C of rand4x_proof.py (unsloth-zoo PR 2241).

Check C: a compiled LoRA-dropout fn run once inside torch.utils.checkpoint(use_reentrant=True) (forward under
no_grad -> Inductor INFERENCE graph; backward recomputes under enable_grad -> Inductor TRAINING graph) and once
plainly in grad mode (TRAINING graph), same seed. This script says WHY the two graphs disagree or agree.

Per arm (each its own process, so a patch / config lands before the first compile; each its own on-disk cache,
disabled in every thread via TORCHINDUCTOR_FORCE_DISABLE_CACHES unless the arm is half of a warm pair):
  stock               the installed torch as is
  patched             unsloth_zoo temporary_patches/inductor_rand4x.py loaded first (only where it applies)
  rand4x_on           TORCHINDUCTOR_USE_RAND4X=1 (torch 2.15+: upstream opt-in back to the 4x path)
  fallback_random     torch._inductor.config.fallback_random = True, assigned in the main thread only (torch
                      2.12+ keeps config per thread, so a compile on autograd's device thread does not see it)
  stock_fill_a -> patched_warm         stock fills cache A (caches on), then the --patch-file patch compiles against A
  stock_fill_b -> patched_fixed_warm   same with cache B and the --patch-file-fixed patch

Observations per arm (no verdicts; the caller judges):
  A, B, C, D           the four rand4x_proof.py statistics, verbatim
  C_cold / C_warm      check C on cold graphs, then again with both graphs already compiled
  C_fixgrad            check C with the checkpoint's no-grad forward routed through the training graph
  mask_*               a mask-exposing variant dropout(x) * w: inference-graph mask vs training-graph mask,
                       cold and warm, plus each mask against eager Triton references computed from the seed
                       the graph should have drawn (aten.randint.low_out on the same RNG state):
                       ref_tl = tl.rand(S, linear index), ref_4x = triton_helpers.rand4x(S, linear index, BLOCK)
  seed_sens            agreement of a compiled no-grad mask under two different manual_seed values
                       (independent = 82 % at p = 0.1; ~100 % = the kernel ignores the seed)
  rng_delta            device RNG offset consumed by a cold call vs a warm call of the same compiled fn
                       (different = compiling consumed or skipped RNG)
  code/*.py            Inductor output code for every captured graph (artifact dir)
Usage: python rand4x_c_diag.py --out OUT.json [--patch-file PATH] [--arms stock,patched,...] [--artifacts DIR]
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

CHILD = r"""
import json, os, sys, importlib.util, traceback
import torch
import torch._inductor.config as ic
from torch._inductor.utils import run_and_get_code
from torch.utils.checkpoint import checkpoint

arm, patch_file, code_dir = sys.argv[1], sys.argv[2], sys.argv[3]
if os.environ.get("DIAG_CACHES") != "1":
    ic.force_disable_caches = True  # this thread; the parent also sets TORCHINDUCTOR_FORCE_DISABLE_CACHES (all threads)
res = {"arm": arm, "torch": torch.__version__, "hip": torch.version.hip, "cuda": torch.version.cuda}
dev = os.environ.get("PROOF_DEVICE", "cuda")
res["device"] = torch.cuda.get_device_name() if dev == "cuda" else dev
try:
    from torch._inductor.codegen.triton import TritonOverrides
    res["has_4x_path"] = "_can_use_4x_random" in TritonOverrides.__dict__
except Exception as e:
    res["has_4x_path"] = repr(e)
res["use_rand4x_config"] = getattr(ic, "use_rand4x", None)
res["patched"] = None
if arm.startswith("patched"):
    spec = importlib.util.spec_from_file_location("rand4x_patch", patch_file)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    res["patched"] = m.patch_inductor_rand4x()
elif arm == "fallback_random":
    ic.fallback_random = True
res["env_use_rand4x"] = os.environ.get("TORCHINDUCTOR_USE_RAND4X")

codes = {}
def cap(tag, fn, *a):
    out, code = run_and_get_code(fn, *a)
    codes[tag] = "\n\n# ======== next module ========\n\n".join(code)
    return out
def compiled(fn):
    torch._dynamo.reset()
    return torch.compile(fn, backend="inductor")
def rel(a, b):
    return ((a - b).float().norm() / b.float().norm()).item()

P = 0.5
def block(x):
    h = x.transpose(1, 2).contiguous()
    h = torch.nn.functional.dropout(h, P, True)
    return h.transpose(1, 2).contiguous()

def guarded(name, fn):
    try:
        fn()
    except Exception:
        res[name + "_error"] = traceback.format_exc()[-3000:]

# ---- A, B, D: verbatim from rand4x_proof.py
def ab_d():
    x = torch.ones(8, 64, 4096, device=dev)
    f = compiled(block)
    torch.manual_seed(0)
    with torch.no_grad():
        y = cap("A_no_grad", f, x)
    mask = (y != 0).float()
    dup = sum((mask[:, 0::4, :] == mask[:, j::4, :]).float().mean().item() for j in (1, 2, 3)) / 3
    keep = mask.mean().item()
    res["A_dup_pct"] = 100 * dup
    res["A_sigma"] = (keep - (1 - P)) / ((P * (1 - P) / mask.numel()) ** 0.5)
    def fn_ckpt(x):
        return checkpoint(block, x, use_reentrant=False, preserve_rng_state=True)
    f = compiled(fn_ckpt)
    x = torch.ones(8, 64, 4096, device=dev, requires_grad=True)
    torch.manual_seed(0)
    y = cap("B_grad", f, x)
    y.backward(torch.ones_like(y))
    res["B_agree_pct"] = 100 * ((y.detach() != 0) == (x.grad != 0)).float().mean().item()
    xd = torch.ones(1 << 22, device=dev)
    fd = compiled(lambda t: torch.nn.functional.dropout(t, P, True))
    torch.manual_seed(3)
    with torch.no_grad():
        md = (cap("D_no_grad", fd, xd) != 0).float()
    res["D_sigma"] = (md.mean().item() - (1 - P)) / ((P * (1 - P) / md.numel()) ** 0.5)
    a, b = md[:-1] - md.mean(), md[1:] - md.mean()
    res["D_lag1_corr"] = ((a * b).mean() / (a.pow(2).mean())).item()
guarded("ABD", ab_d)

# ---- C: verbatim from rand4x_proof.py, then repeated warm
def c_check():
    torch.manual_seed(1)
    Hd, r = 4096, 16
    A = (torch.randn(r, Hd, device=dev) / Hd**0.5).requires_grad_(True)
    B = (torch.randn(Hd, r, device=dev) / r**0.5).requires_grad_(True)
    def lora(x):
        return x + (torch.nn.functional.dropout(x, 0.1, True) @ A.t()) @ B.t()
    cl = compiled(lora)
    def layer(x):
        return cl(x)
    xin = torch.randn(4, 512, Hd, device=dev, requires_grad=True)
    for tag in ("cold", "warm"):
        torch.manual_seed(2)
        yc = checkpoint(layer, xin, use_reentrant=True, preserve_rng_state=True)
        (yc.float().pow(2).mean()).backward()
        g_ckpt = A.grad.detach().clone(); A.grad = None; B.grad = None; xin.grad = None
        torch.manual_seed(2)
        yr = cl(xin)
        (yr.float().pow(2).mean()).backward()
        g_ref = A.grad.detach().clone(); A.grad = None; B.grad = None; xin.grad = None
        res[f"C_{tag}_fwd_rel_err"] = rel(yc, yr)
        res[f"C_{tag}_gradA_rel_err"] = rel(g_ckpt, g_ref)
        if tag == "cold":
            res["C_fwd_rel_err"], res["C_gradA_rel_err"] = res["C_cold_fwd_rel_err"], res["C_cold_gradA_rel_err"]
    # Candidate fix: the checkpoint's no-grad forward runs the TRAINING graph (enable_grad, output detached),
    # so forward and recompute share one compiled graph.
    def layer_grad_graph(x):
        if torch.is_grad_enabled():
            return cl(x)
        with torch.enable_grad():
            return cl(x.detach().requires_grad_(x.requires_grad)).detach()
    torch.manual_seed(2)
    yc = checkpoint(layer_grad_graph, xin, use_reentrant=True, preserve_rng_state=True)
    (yc.float().pow(2).mean()).backward()
    g_ckpt = A.grad.detach().clone(); A.grad = None; B.grad = None; xin.grad = None
    torch.manual_seed(2)
    yr = cl(xin)
    (yr.float().pow(2).mean()).backward()
    g_ref = A.grad.detach().clone(); A.grad = None; B.grad = None; xin.grad = None
    res["C_fixgrad_fwd_rel_err"] = rel(yc, yr)
    res["C_fixgrad_gradA_rel_err"] = rel(g_ckpt, g_ref)
    # Code of both graphs (fresh compile, captured).
    cl2 = compiled(lora)
    with torch.no_grad():
        cap("C_lora_no_grad", cl2, xin)
    cap("C_lora_grad", cl2, xin)
guarded("C", c_check)

# ---- mask-level view: inference graph vs training graph, and vs eager references from the expected seed
def seed_for(s):
    torch.manual_seed(s)
    buf = torch.empty(1, dtype=torch.int64, device=dev)
    torch.ops.aten.randint.low_out(-9223372036854775808, 9223372036854775807, [1], out=buf)
    return buf

def refs(seed_buf, n):
    out = {}
    if dev != "cuda":
        return out
    import triton, triton.language as tl
    try:
        from torch._inductor.runtime import triton_helpers
        has4x = hasattr(triton_helpers, "rand4x")
    except Exception:
        has4x = False
    @triton.jit
    def k_tl(seed_ptr, out_ptr, n, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        seed = tl.load(seed_ptr)
        tl.store(out_ptr + offs, tl.rand(seed, offs.to(tl.uint32)), mask=offs < n)
    o = torch.empty(n, device=dev)
    k_tl[(triton.cdiv(n, 1024),)](seed_buf, o, n, BLOCK=1024)
    out["tl"] = o
    if has4x:
        @triton.jit
        def k_4x(seed_ptr, out_ptr, n, BLOCK: tl.constexpr):
            offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
            seed = tl.load(seed_ptr)
            tl.store(out_ptr + offs, triton_helpers.rand4x(seed, offs.to(tl.uint32), BLOCK), mask=offs < n)
        o4 = torch.empty(n, device=dev)
        k_4x[(triton.cdiv(n, 1024),)](seed_buf, o4, n, BLOCK=1024)
        out["4x"] = o4
    return out

def mask_view():
    pm = 0.1
    w = torch.ones((), device=dev, requires_grad=True)
    def mf(x):
        return torch.nn.functional.dropout(x, pm, True) * w
    x = torch.ones(4, 512, 4096, device=dev)
    n = x.numel()
    S = seed_for(7)
    res["mask_expected_seed"] = int(S.item())
    rf = {k: (v <= pm).logical_not().reshape(x.shape) for k, v in refs(S, n).items()}  # keep = rand > p
    f = compiled(mf)
    masks = {}
    for tag in ("cold", "warm"):
        torch.manual_seed(7)
        with torch.no_grad():
            mi = (f(x) != 0)
        torch.manual_seed(7)
        mt = (f(x) != 0)
        masks[tag] = (mi, mt)
        res[f"mask_{tag}_infer_vs_train_agree_pct"] = 100 * (mi == mt).float().mean().item()
        for rn, r in rf.items():
            res[f"mask_{tag}_infer_eq_ref_{rn}_pct"] = 100 * (mi == r).float().mean().item()
            res[f"mask_{tag}_train_eq_ref_{rn}_pct"] = 100 * (mt == r).float().mean().item()
    # Seed sensitivity of the no-grad graph.
    with torch.no_grad():
        torch.manual_seed(11); m1 = (f(x) != 0)
        torch.manual_seed(12); m2 = (f(x) != 0)
    res["seed_sens_agree_pct"] = 100 * (m1 == m2).float().mean().item()
    # RNG offset consumed by a cold vs a warm call (no_grad graph), read from the device RNG state.
    def call_delta(fn):
        torch.manual_seed(5)
        s0 = torch.cuda.get_rng_state().clone() if dev == "cuda" else torch.get_rng_state().clone()
        with torch.no_grad():
            fn(x)
        s1 = torch.cuda.get_rng_state() if dev == "cuda" else torch.get_rng_state()
        return (s0 != s1).nonzero().flatten().tolist()[:8], s1[-8:].tolist()
    g = compiled(lambda t: torch.nn.functional.dropout(t, pm, True) * 2)
    res["rng_cold_changed_bytes"], res["rng_cold_tail"] = call_delta(g)
    res["rng_warm_changed_bytes"], res["rng_warm_tail"] = call_delta(g)
    # Code of both mask graphs.
    f2 = compiled(mf)
    with torch.no_grad():
        cap("mask_no_grad", f2, x)
    cap("mask_grad", f2, x)
guarded("mask", mask_view)

allcode = "\n".join(codes.values())
res["code_rand4x"] = "rand4x(" in allcode
res["code_tl_rand"] = "tl.rand(" in allcode
res["code_rand_lines"] = sorted({ln.strip() for ln in allcode.splitlines()
                                 if "rand" in ln and ("tl." in ln or "triton_helpers" in ln or "randint" in ln)})[:40]
os.makedirs(code_dir, exist_ok=True)
for tag, c in codes.items():
    with open(os.path.join(code_dir, f"{arm}_{tag}.py"), "w") as fh:
        fh.write(c)
print("RESULT " + json.dumps(res, default=str))
"""

ARMS = ("stock", "patched", "rand4x_on", "fallback_random")
# Warm pairs: the first arm fills a private on-disk cache with caches on, the second (a new process) compiles
# the same graphs against it. Check C's recompute compiles on autograd's device thread, which sees torch's
# default config (config assignments are per thread on torch 2.12+), so this is where a cache key tag that
# only the main thread carries lets a stale rand4x entry through.
WARM_PAIRS = {"stock_fill_a": "patched_warm", "stock_fill_b": "patched_fixed_warm"}


def run_arm(arm, patch_file, code_dir, timeout, cache_dir):
    env = dict(os.environ)
    for k in ("TORCHINDUCTOR_USE_RAND4X", "TORCHINDUCTOR_FALLBACK_RANDOM", "TORCH_COMPILE_CACHE_KEY_TAG",
              "TORCHINDUCTOR_FORCE_DISABLE_CACHES", "DIAG_CACHES"):
        env.pop(k, None)
    env["TORCHINDUCTOR_CACHE_DIR"] = str(cache_dir)
    if arm in WARM_PAIRS or arm in WARM_PAIRS.values():
        env["DIAG_CACHES"] = "1"
    else:
        env["TORCHINDUCTOR_FORCE_DISABLE_CACHES"] = "1"
    if arm == "rand4x_on":
        env["TORCHINDUCTOR_USE_RAND4X"] = "1"
    try:
        out = subprocess.run([sys.executable, "-c", CHILD, arm, str(patch_file), str(code_dir)],
                             capture_output=True, text=True, env=env, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"arm": arm, "error": "timeout"}
    line = [ln for ln in out.stdout.splitlines() if ln.startswith("RESULT ")]
    if out.returncode != 0 or not line:
        return {"arm": arm, "error": f"exit {out.returncode}", "stdout_tail": out.stdout[-2000:],
                "stderr_tail": out.stderr[-5000:]}
    r = json.loads(line[-1][len("RESULT "):])
    r["stderr_tail"] = out.stderr[-1500:]
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--patch-file", default="")
    ap.add_argument("--patch-file-fixed", default="", help="patch for the patched_fixed_warm arm")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--artifacts", type=Path, default=None)
    ap.add_argument("--timeout", type=int, default=1800)
    args = ap.parse_args()
    code_dir = args.artifacts or args.out.parent / (args.out.stem + "_code")
    cache_root = Path(tempfile.mkdtemp(prefix="rand4x_c_diag_"))
    pair_of = {v: k for k, v in WARM_PAIRS.items()}
    results = {}
    for arm in [a for a in args.arms.split(",") if a]:
        pf = args.patch_file_fixed if arm == "patched_fixed_warm" else args.patch_file
        if arm.startswith("patched") and not (pf and Path(pf).is_file()):
            results[arm] = {"arm": arm, "skipped": "no patch file"}
            continue
        cache_dir = cache_root / pair_of.get(arm, arm)
        print(f"== arm {arm} (cache {cache_dir.name})", flush=True)
        results[arm] = run_arm(arm, pf, code_dir, args.timeout, cache_dir)
        args.out.write_text(json.dumps(results, indent=1, default=str), encoding="utf-8")
        brief = {k: v for k, v in results[arm].items() if k.startswith(("C_", "mask_", "seed_sens", "A_dup", "B_agree",
                                                                          "patched", "error", "code_rand4x"))}
        print(json.dumps(brief, default=str), flush=True)
    args.out.write_text(json.dumps(results, indent=1, default=str), encoding="utf-8")
    shutil.rmtree(cache_root, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
