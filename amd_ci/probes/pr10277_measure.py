#!/usr/bin/env python3
"""Runs in the venv install.sh built. Observes the installed environment; judges nothing.

Every GPU step is a child process: the defect in issue 10273 is a SIGSEGV, and a
crash must come back as a return code, not take this script down with it.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys

TORCH_INFO = r"""
import json, os, torch
d = {"torch": torch.__version__, "hip": torch.version.hip, "cuda_available": torch.cuda.is_available(),
     "hsa_override": os.environ.get("HSA_OVERRIDE_GFX_VERSION"), "arch_list": torch.cuda.get_arch_list()}
if d["cuda_available"]:
    p = torch.cuda.get_device_properties(0)
    d["device"] = torch.cuda.get_device_name(0)
    d["arch"] = getattr(p, "gcnArchName", None)
    try:
        a = torch.randn(512, 512, device="cuda"); b = a @ a
        torch.cuda.synchronize()
        d["plain_matmul_ok"] = bool(torch.isfinite(b).all())
    except Exception as e:
        d["plain_matmul_ok"] = False
        d["plain_matmul_error"] = f"{type(e).__name__}: {str(e).splitlines()[0]}"
print("RESULT " + json.dumps(d))
"""

DYNAMO = r"""
import json, torch
f = torch.compile(lambda x: x * 2 + 1, backend="eager")
print("RESULT " + json.dumps({"dynamo_ok": bool((f(torch.ones(3)) == 3).all())}))
"""

BNB_IMPORT = r"""
import json, bitsandbytes as bnb
libs = sorted({l.split()[-1].rsplit("/", 1)[-1] for l in open("/proc/self/maps") if "libbitsandbytes" in l})
print("RESULT " + json.dumps({"bnb": bnb.__version__, "loaded_bnb_libs": libs}))
"""

BNB_4BIT = r"""
import json, torch, bitsandbytes as bnb, bitsandbytes.functional as F
d = {"bnb": bnb.__version__}
torch.manual_seed(0)
W = torch.randn(256, 512, dtype=torch.bfloat16, device="cuda")
q, st = F.quantize_4bit(W, quant_type="nf4", compress_statistics=True)
Wd = F.dequantize_4bit(q, st)
torch.cuda.synchronize()
d["dequant_rel_err"] = float((Wd.float() - W.float()).norm() / W.float().norm())
lin = bnb.nn.Linear4bit(512, 256, bias=False, compute_dtype=torch.bfloat16, quant_type="nf4")
lin.weight = bnb.nn.Params4bit(W.cpu(), requires_grad=False, quant_type="nf4")
lin = lin.cuda()
x = torch.randn(8, 512, dtype=torch.bfloat16, device="cuda", requires_grad=True)
y = lin(x)
loss = y.float().pow(2).mean()
loss.backward()
torch.cuda.synchronize()
ref = x.detach().float() @ F.dequantize_4bit(lin.weight.data, lin.weight.quant_state).float().t()
d["fwd_rel_err"] = float((y.float() - ref).norm() / ref.norm())
d["grad_finite"] = bool(torch.isfinite(x.grad).all())
d["loss_finite"] = bool(torch.isfinite(loss))
libs = sorted({l.split()[-1] for l in open("/proc/self/maps") if "libbitsandbytes" in l})
d["loaded_bnb_libs"] = libs
print("RESULT " + json.dumps(d))
"""

UNSLOTH_QLORA = r"""
import json
from unsloth import FastLanguageModel
import torch
model, tok = FastLanguageModel.from_pretrained("unsloth/Qwen3-0.6B", max_seq_length=256, load_in_4bit=True)
model = FastLanguageModel.get_peft_model(model, r=8, lora_alpha=8, use_gradient_checkpointing="unsloth",
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"], random_state=3407)
model.train()
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)
batch = tok(["The quick brown fox jumps over the lazy dog. " * 10] * 2, return_tensors="pt").to("cuda")
losses = []
for _ in range(3):
    out = model(**batch, labels=batch["input_ids"]); out.loss.backward(); opt.step(); opt.zero_grad()
    losses.append(float(out.loss))
print("RESULT " + json.dumps({"losses": losses}))
"""


def child(py: str, code: str, timeout: int) -> dict:
    try:
        p = subprocess.run([py, "-c", code], capture_output = True, text = True, timeout = timeout)
    except subprocess.TimeoutExpired:
        return {"rc": 124, "timeout": True}
    rec: dict = {"rc": p.returncode, "signal": -p.returncode if p.returncode < 0 else None}
    for line in p.stdout.splitlines():
        if line.startswith("RESULT "):
            rec.update(json.loads(line[7:]))
    rec["stderr_tail"] = (p.stderr or "")[-1500:]
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required = True)
    ap.add_argument("--out", required = True)
    args = ap.parse_args()
    with open(args.env, encoding = "utf-8") as fh:
        d = json.load(fh)
    py = sys.executable
    d["torch_info"] = child(py, TORCH_INFO, 600)
    d["dynamo"] = child(py, DYNAMO, 600)
    d["bnb_import"] = child(py, BNB_IMPORT, 600)
    d["bnb_4bit"] = child(py, BNB_4BIT, 900)
    d["unsloth_qlora"] = child(py, UNSLOTH_QLORA, 2400)
    with open(args.out, "w", encoding = "utf-8") as fh:
        json.dump(d, fh, indent = 2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
