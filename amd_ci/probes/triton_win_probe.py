"""Probe: is Triton installed in this Python, does it import after torch, which backend is active, and does a tiny
@triton.jit kernel and a torch.compile (Inductor) function run on the GPU. Also a small unsloth_zoo fused CE call with
its CALLS delta when unsloth_zoo is importable. Writes JSON to --out (never stdout)."""

import argparse
import json
import sys
import traceback


def _err(e):
    return f"{type(e).__name__}: {e}"[:2000]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required = True)
    a = ap.parse_args()
    rec = {"python": sys.executable}

    def save():
        with open(a.out, "w", encoding = "utf-8") as f:
            json.dump(rec, f, indent = 1)

    try:
        import importlib.metadata as m

        rec["dists"] = sorted(
            f"{d.metadata['Name']}=={d.version}" for d in m.distributions()
            if any(k in (d.metadata["Name"] or "").lower() for k in ("triton", "torch", "unsloth"))
        )
    except Exception as e:
        rec["dists_error"] = _err(e)
    save()
    try:
        import torch

        rec["torch"] = {"version": torch.__version__, "hip": torch.version.hip, "cuda_available": torch.cuda.is_available(),
                        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    except Exception as e:
        rec["torch_error"] = _err(e)
        save()
        return 0
    save()
    try:
        import triton
        import triton.language as tl

        rec["triton"] = {"version": triton.__version__, "file": triton.__file__}
        try:
            rec["triton"]["target"] = str(triton.runtime.driver.active.get_current_target())
        except Exception as e:
            rec["triton"]["target_error"] = _err(e)
    except Exception as e:
        rec["triton"] = {"error": _err(e)}
        save()
        return 0
    save()
    if torch.cuda.is_available():
        try:
            @triton.jit
            def _add(x_ptr, y_ptr, o_ptr, n, BLOCK: tl.constexpr):
                offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
                ok = offs < n
                tl.store(o_ptr + offs, tl.load(x_ptr + offs, mask = ok) + tl.load(y_ptr + offs, mask = ok), mask = ok)

            x = torch.randn(4099, device = "cuda")
            y = torch.randn(4099, device = "cuda")
            o = torch.empty_like(x)
            _add[(triton.cdiv(4099, 1024),)](x, y, o, 4099, BLOCK = 1024)
            torch.cuda.synchronize()
            rec["jit_kernel"] = {"ok": bool(torch.allclose(o, x + y)), "max_err": float((o - x - y).abs().max())}
        except Exception as e:
            rec["jit_kernel"] = {"error": _err(e), "tb": traceback.format_exc()[-3000:]}
        save()
        try:
            f = torch.compile(lambda t: torch.nn.functional.gelu(t) * 2 + 1)
            t = torch.randn(1000, device = "cuda")
            ref = torch.nn.functional.gelu(t) * 2 + 1
            rec["inductor"] = {"ok": bool(torch.allclose(f(t), ref, atol = 1e-5))}
        except Exception as e:
            rec["inductor"] = {"error": _err(e), "tb": traceback.format_exc()[-3000:]}
        save()
        try:
            import unsloth_zoo.fused_losses.fused_linear as fl

            before = dict(fl.CALLS)
            h = torch.randn(256, 64, device = "cuda", dtype = torch.bfloat16, requires_grad = True)
            w = torch.randn(1003, 64, device = "cuda", dtype = torch.bfloat16)
            t = torch.randint(0, 1003, (256,), device = "cuda")
            lp = fl.fused_linear_logprobs(h, w, t)
            lp.sum().backward()
            ref = torch.log_softmax((h.detach().float() @ w.float().T), -1).gather(1, t[:, None])[:, 0]
            rec["fused_logprobs"] = {
                "max_err": float((lp.float() - ref).abs().max()),
                "has_triton": bool(fl._HAS_TRITON), "use_triton_cuda": bool(fl._use_triton(torch.device("cuda", 0))),
                "calls_delta": {k: fl.CALLS[k] - before.get(k, 0) for k in fl.CALLS if fl.CALLS[k] != before.get(k, 0)},
                "triton_broken": [str(x) for x in getattr(fl, "_TRITON_BROKEN", [])],
            }
        except Exception as e:
            rec["fused_logprobs"] = {"error": _err(e), "tb": traceback.format_exc()[-3000:]}
        save()
    return 0


if __name__ == "__main__":
    sys.exit(main())
