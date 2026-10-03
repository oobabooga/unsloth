#!/usr/bin/env python3
"""One SDPA case per fresh process, so a launch error is never attributed to the wrong call.

A failed AOTriton launch on Windows ROCm is not raised by the SDPA call itself nor by synchronize(); it
surfaces at the next checked kernel launch. So each case runs alone, follows the call with a checked
op, and also compares against the math kernel (a kernel that never ran leaves garbage, not an error).
"""
import json, subprocess, sys

CASE = r'''
import json, sys, torch, torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
name, hd, grad, mask, gqa = sys.argv[1], int(sys.argv[2]), sys.argv[3] == "1", sys.argv[4] == "1", sys.argv[5] == "1"
g = torch.Generator(device = "cuda").manual_seed(0)
B, H, S = 4, 4, 12
q = torch.randn(B, H, S, hd, device = "cuda", dtype = torch.bfloat16, generator = g)
k = torch.randn(B, H // 2 if gqa else H, S, hd, device = "cuda", dtype = torch.bfloat16, generator = g)
v = torch.randn(B, H // 2 if gqa else H, S, hd, device = "cuda", dtype = torch.bfloat16, generator = g)
kw = {"is_causal": not mask}
if mask: kw["attn_mask"] = torch.ones(S, S, device = "cuda", dtype = torch.bool).tril()
if gqa: kw["enable_gqa"] = True
torch.cuda.synchronize()
def run(math):
    qq = q.clone().requires_grad_(grad)
    with (sdpa_kernel([SDPBackend.MATH]) if math else sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH])):
        o = F.scaled_dot_product_attention(qq, k, v, **kw)
    o2 = o.float()  # checked launch: raises a pending launch error from the call above
    if grad:
        o2.sum().backward()
        return o2.detach(), qq.grad.float()
    return o2, None
out = {"case": name}
try:
    o, dq = run(False); torch.cuda.synchronize(); o.sum().item()
    out["fused"] = "ran"
except Exception as e:
    out["fused"] = f"{type(e).__name__}: {str(e).splitlines()[0][:120]}"
    print("CASE " + json.dumps(out)); sys.exit(0)
ro, rdq = run(True)
out["max_diff_out"] = float((o - ro).abs().max())
if grad: out["max_diff_dq"] = float((dq - rdq).abs().max())
print("CASE " + json.dumps(out))
'''

cases = []
for hd in (16, 64):
    for grad in (0, 1):
        for mask, gqa in ((0, 0), (1, 0), (0, 1), (1, 1)):
            cases.append((f"hd{hd}_{'grad' if grad else 'nograd'}_{'mask' if mask else 'causal'}{'_gqa' if gqa else ''}", hd, grad, mask, gqa))
res = []
for name, hd, grad, mask, gqa in cases:
    p = subprocess.run([sys.executable, "-c", CASE, name, str(hd), str(grad), str(mask), str(gqa)],
                       capture_output = True, text = True, encoding = "utf-8", errors = "replace", timeout = 600)
    line = [l for l in p.stdout.splitlines() if l.startswith("CASE ")]
    res.append(json.loads(line[-1][5:]) if line else {"case": name, "crash": (p.stdout + p.stderr)[-300:]})
    print(json.dumps(res[-1]), flush = True)
print("SDPA_CASES " + json.dumps(res))
