"""GPU proof for the pytorch#198333 backport (unsloth_zoo temporary_patches/inductor_rand4x.py).

Arms, each in its own process so the patch lands before the first compile:
  base          stock torch, caches off
  patched       backport loaded first, caches off
  patched_warm  base compiles into a fresh FX/AOT cache dir, then a NEW process loads the
                backport and compiles the same graphs against that warm cache. Clean only if
                the cache key tag forced a recompile (a stale rand4x kernel would replay).

Cases (all compiled with the inductor backend on CUDA):
  A  issue's symptom A: dropout fed by transpose(1,2).contiguous(), no_grad forward;
     statistic = % of positions where channels 4k+1..3 copy channel 4k (independent = 50%
     at p=0.5) and keep-rate sigma.
  B  issue's symptom B: checkpoint(use_reentrant=False) fwd vs bwd mask agreement.
  C  Unsloth's checkpoint pattern: forward under no_grad (autograd.Function.forward), recompute
     under enable_grad with the RNG state restored (torch.utils.checkpoint use_reentrant=True,
     as unsloth_zoo.gradient_checkpointing does); LoRA-shaped dropout(x) @ A.t() @ B.t().
  D  plain contiguous dropout keep rate + lag-1 correlation (control; should be clean in all arms).
Also records whether the generated code contains triton_helpers.rand4x.

Exit: 0 = base shows the defect and both patched arms are clean; 1 = a patched arm is still
bad; 3 = VOID (base clean, e.g. torch 2.12 / 2.15+, so nothing is proven); 2 = no CUDA GPU.
Usage: python rand4x_dropout_gpu_proof.py [--patch-file PATH]
Default patch file: temp/wt_bp_dropout/... (amd_ci always passes --patch-file).
Copied verbatim from temp/torch_backports/rand4x_dropout_gpu_proof.py for the amd_ci zoo2241 probe.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_PATCH = HERE.parent / "wt_bp_dropout" / "unsloth_zoo" / "temporary_patches" / "inductor_rand4x.py"

CHILD = r"""
import json, os, sys, importlib.util
import torch
import torch._inductor.config as ic
from torch._inductor.utils import run_and_get_code
from torch.utils.checkpoint import checkpoint

arm, patch_file = sys.argv[1], sys.argv[2]
caches = os.environ.get("PROOF_CACHES") == "1"
ic.force_disable_caches = not caches
if caches:
    ic.fx_graph_cache = True
    torch._functorch.config.enable_autograd_cache = True
patched = None
if arm.startswith("patched"):
    spec = importlib.util.spec_from_file_location("rand4x_patch", patch_file)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    patched = m.patch_inductor_rand4x()

P = 0.5
dev = os.environ.get("PROOF_DEVICE", "cuda")  # "cpu" = script smoke test only
codes = []

def compiled(fn):
    torch._dynamo.reset()
    return torch.compile(fn, backend="inductor")

def run_capture(fn, *a):
    out, code = run_and_get_code(fn, *a)
    codes.extend(code)
    return out

def block(x):
    h = x.transpose(1, 2).contiguous()
    h = torch.nn.functional.dropout(h, P, True)
    return h.transpose(1, 2).contiguous()

def dup_stats(mask):
    dup = sum((mask[:, 0::4, :] == mask[:, j::4, :]).float().mean().item() for j in (1, 2, 3)) / 3
    keep = mask.mean().item()
    sigma = (keep - (1 - P)) / ((P * (1 - P) / mask.numel()) ** 0.5)
    return 100 * dup, sigma

res = {"torch": torch.__version__, "device": torch.cuda.get_device_name() if dev == "cuda" else dev,
       "hip": torch.version.hip, "arm": arm, "patched": patched}

# A
x = torch.ones(8, 64, 4096, device=dev)
f = compiled(block)
torch.manual_seed(0)
with torch.no_grad():
    y = run_capture(f, x)
res["A_dup_pct"], res["A_sigma"] = dup_stats((y != 0).float())

# B
def fn_ckpt(x):
    return checkpoint(block, x, use_reentrant=False, preserve_rng_state=True)
f = compiled(fn_ckpt)
x = torch.ones(8, 64, 4096, device=dev, requires_grad=True)
torch.manual_seed(0)
y = run_capture(f, x)
y.backward(torch.ones_like(y))
res["B_agree_pct"] = 100 * ((y.detach() != 0) == (x.grad != 0)).float().mean().item()

# C: LoRA-shaped dropout inside a compiled region, checkpointed the way Unsloth does it.
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
torch.manual_seed(2)
yc = checkpoint(layer, xin, use_reentrant=True, preserve_rng_state=True)
(yc.float().pow(2).mean()).backward()
g_ckpt = A.grad.detach().clone(); A.grad = None; xin.grad = None
torch.manual_seed(2)
yr = cl(xin)  # same RNG state, plain autograd (mask saved, no recompute)
(yr.float().pow(2).mean()).backward()
g_ref = A.grad.detach().clone()
res["C_fwd_rel_err"] = ((yc - yr).norm() / yr.norm()).item()
res["C_gradA_rel_err"] = ((g_ckpt - g_ref).norm() / g_ref.norm()).item()

# D: contiguous control
xd = torch.ones(1 << 22, device=dev)
fd = compiled(lambda t: torch.nn.functional.dropout(t, P, True))
torch.manual_seed(3)
with torch.no_grad():
    md = (run_capture(fd, xd) != 0).float()
res["D_sigma"] = (md.mean().item() - (1 - P)) / ((P * (1 - P) / md.numel()) ** 0.5)
a, b = md[:-1] - md.mean(), md[1:] - md.mean()
res["D_lag1_corr"] = ((a * b).mean() / (a.pow(2).mean())).item()

code = "\n".join(codes)
res["code_rand4x"] = "rand4x(" in code
res["code_tl_rand"] = "tl.rand(" in code
print("RESULT " + json.dumps(res))
"""


def bad(r):
    return {
        "A": r["A_dup_pct"] > 60 or abs(r["A_sigma"]) > 5,
        "B": r["B_agree_pct"] < 99.99,
        "C": r["C_fwd_rel_err"] > 1e-3 or r["C_gradA_rel_err"] > 1e-2,
        "D": abs(r["D_sigma"]) > 5 or abs(r["D_lag1_corr"]) > 0.01,
        "code": r["code_rand4x"],
    }


def run_arm(arm, patch_file, env_extra):
    env = dict(os.environ, **env_extra)
    env.pop("TORCHINDUCTOR_USE_RAND4X", None)
    env.pop("TORCH_COMPILE_CACHE_KEY_TAG", None)
    if env.get("PROOF_CACHES") != "1":
        # Env, not `ic.force_disable_caches`: torch 2.12+ keeps config assignments per thread, and
        # check C's recompute compiles on autograd's device thread, where an assignment made here
        # is invisible and the default on-disk cache (shared by every arm) would be read.
        env["TORCHINDUCTOR_FORCE_DISABLE_CACHES"] = "1"
    out = subprocess.run([sys.executable, "-c", CHILD, arm, str(patch_file)],
                         capture_output=True, text=True, env=env)
    line = [l for l in out.stdout.splitlines() if l.startswith("RESULT ")]
    if out.returncode != 0 or not line:
        print(out.stdout[-3000:], out.stderr[-6000:])
        raise SystemExit(f"arm {arm} crashed (exit {out.returncode})")
    return json.loads(line[-1][len("RESULT "):])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch-file", default=str(DEFAULT_PATCH))
    args = ap.parse_args()
    import torch
    if not torch.cuda.is_available():
        print("no CUDA/ROCm GPU visible")
        return 2
    results = {}
    results["base"] = run_arm("base", args.patch_file, {})
    results["patched"] = run_arm("patched", args.patch_file, {})
    with tempfile.TemporaryDirectory() as cache:
        warm = {"PROOF_CACHES": "1", "TORCHINDUCTOR_CACHE_DIR": cache}
        results["base_fill_cache"] = run_arm("base", args.patch_file, warm)
        results["patched_warm"] = run_arm("patched_warm", args.patch_file, warm)
    verdicts = {k: bad(v) for k, v in results.items()}
    print(json.dumps({"results": results, "bad": verdicts}, indent=1))
    base_bad = any(verdicts["base"][k] for k in ("A", "B", "C", "code"))
    if not base_bad:
        print("VOID: base arm shows no defect on this torch / GPU; nothing proven")
        return 3
    for arm in ("patched", "patched_warm"):
        if not results[arm]["patched"] or any(verdicts[arm].values()):
            print(f"FAIL: {arm} still bad: {verdicts[arm]}")
            return 1
    print("PASS: base shows the defect, patched (cold and warm cache) is clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
