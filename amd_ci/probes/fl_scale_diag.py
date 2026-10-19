"""Diagnose test_ce_incoming_gradient_scales_in_fp32 on HIP: determinism of the CE grads and the fp32 scale kernel."""
import json, sys, torch
import unsloth_zoo  # noqa: F401
from unsloth_zoo.fused_losses import fused_linear as fl
from unsloth_zoo.fused_losses.fused_linear import fused_linear_cross_entropy

out = {"zoo": fl.__file__, "hip": torch.version.hip, "triton_broken": [str(d) for d in fl._TRITON_BROKEN]}
dev = "cuda"
for dtype in (torch.float16, torch.bfloat16, torch.float32):
    g = torch.Generator(device = "cpu").manual_seed(0)
    h = (torch.randn(50, 32, generator = g) * 0.5).to(dev, dtype).requires_grad_(True)
    w = (torch.randn(131, 32, generator = g) * 0.5).to(dev, dtype).requires_grad_(True)
    b = (torch.randn(131, generator = g) * 0.1).to(dev, dtype).requires_grad_(True)
    t = torch.randint(0, 131, (50,), generator = g).to(dev)
    runs = []
    for go in (None, None, 0.37):
        loss = fused_linear_cross_entropy(h, w, b, t, divisor = 50)
        runs.append(torch.autograd.grad(loss, (h, w, b), None if go is None else torch.tensor(go, device = dev)))
    rec = {"mode": list(fl._logits_mode(h.device, dtype)) if dtype != torch.float32 else None}
    for i, name in enumerate("hwb"):
        y, y2, x = runs[0][i], runs[1][i], runs[2][i]
        want = (y.float() * torch.tensor(0.37, dtype = torch.float32, device = dev)).to(dtype)
        direct = torch.ops.unsloth_zoo.fused_linear_scale_rows(y, torch.tensor(0.37, dtype = torch.float64, device = dev), dtype)
        rec[name] = {
            "deterministic": bool(torch.equal(y, y2)),
            "n_diff_det": int((y != y2).sum()),
            "scaled_eq_want": bool(torch.equal(x, want)),
            "n_diff_scaled": int((x != want).sum()),
            "scale_op_eq_want": bool(torch.equal(direct, want)),
            "n_diff_scale_op": int((direct != want).sum()),
            "max_abs_diff_scaled": float((x.float() - want.float()).abs().max()),
        }
        if rec[name]["n_diff_scale_op"]:
            k = (direct != want).nonzero()[0].tolist()
            rec[name]["example"] = {"y": float(y[tuple(k)]), "op": float(direct[tuple(k)]), "want": float(want[tuple(k)]),
                                    "y_times_fp32": float(y[tuple(k)].float() * 0.37)}
    out[str(dtype)] = rec
out["calls"] = dict(fl.CALLS)
print("FL_SCALE_DIAG " + json.dumps(out))
