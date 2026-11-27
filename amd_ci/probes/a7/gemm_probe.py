#!/usr/bin/env python3
"""A7 item 1: ROCm large-GEMM silent-corruption probe (ComfyUI #17405 shapes) on the installed (pinned) ROCm torch.

  gemm_probe.py --out F

Cases, each against a float64 CPU product of the SAME rounded inputs:
  mm    (524289, 64) @ (64, 64)  bf16 and fp16 (also F.linear, the layout a Linear runs)
  bmm   (65537, 4, 4) @ (65537, 4, 4)  fp32, bf16, fp16
under each BLAS backend torch exposes here (default, hipBLASLt, rocBLAS). The caller runs this once plain and once with
PYTORCH_TUNABLEOP_ENABLED=1 (fresh process each). An element is BAD when |gpu - ref| > 8 * (2 eps |a|.|b| + eps |ref|)
+ 1e-6 (eps of the dtype; fp32 accumulate assumed for 16-bit inputs), or it is not finite. Rows past 2^19 / batches past
2^16 are counted separately (the reported corruption starts past those)."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required = True)
    a = ap.parse_args()
    obs: dict = {"cases": [], "env": {k: os.environ.get(k) for k in ("PYTORCH_TUNABLEOP_ENABLED", "HIPBLASLT_LOG_LEVEL")}}
    t0 = time.time()
    try:
        import torch
        import torch.nn.functional as F
        obs["torch"], obs["hip"] = torch.__version__, getattr(torch.version, "hip", None)
        obs["arch"] = getattr(torch.cuda.get_device_properties(0), "gcnArchName", None)
        obs["platform"] = sys.platform
        g = torch.Generator().manual_seed(17405)

        def bad_count(got, ref, bound, split_dim0):
            err = (got.double() - ref).abs()
            fin = torch.isfinite(got)
            bad = (~fin) | (err > 8 * bound + 1e-6)
            rows = bad.reshape(bad.shape[0], -1).any(1)
            return {"bad_elems": int(bad.sum()), "bad_rows": int(rows.sum()),
                    "bad_rows_past_split": int(rows[split_dim0:].sum()),
                    "first_bad_row": int(rows.nonzero()[0, 0]) if bool(rows.any()) else None,
                    "max_abs_err": float(err[fin].max()) if bool(fin.any()) else None,
                    "max_ratio": float((err / (bound + 1e-6))[fin].max()) if bool(fin.any()) else None,
                    "nonfinite": int((~fin).sum())}

        libs = ["default"]
        for name in ("cublaslt", "cublas"):
            libs.append(name)
        eps = {torch.bfloat16: 2.0 ** -8, torch.float16: 2.0 ** -11, torch.float32: 2.0 ** -24}
        A32 = torch.randn(524289, 64, generator = g)
        B32 = torch.randn(64, 64, generator = g)
        X32 = torch.randn(65537, 4, 4, generator = g)
        Y32 = torch.randn(65537, 4, 4, generator = g)
        # sensitivity: the checker must flag a zeroed row past 2^19 and a garbage batch past 2^16
        As, Bs = A32.to(torch.bfloat16), B32.to(torch.bfloat16)
        refs = As.double() @ Bs.double()
        bnd = 2 * 2.0 ** -8 * (As.double().abs() @ Bs.double().abs()) + 2.0 ** -8 * refs.abs()
        fake = refs.to(torch.bfloat16)
        fake[524288] = 0
        obs["selftest_mm"] = bad_count(fake, refs, bnd, 2 ** 19)
        for lib in libs:
            try:
                if lib != "default":
                    torch.backends.cuda.preferred_blas_library(lib)
                obs.setdefault("blas_lib", {})[lib] = str(torch.backends.cuda.preferred_blas_library())
            except Exception as exc:  # noqa: BLE001
                obs.setdefault("blas_lib", {})[lib] = f"unavailable: {exc!r}"[:200]
                continue
            for dt in (torch.bfloat16, torch.float16):
                A, B = A32.to(dt), B32.to(dt)
                ref = A.double() @ B.double()
                bound = 2 * eps[dt] * (A.double().abs() @ B.double().abs()) + eps[dt] * ref.abs()
                for op in ("mm", "linear"):
                    rec = {"lib": lib, "op": op, "dtype": str(dt).replace("torch.", ""), "shape": "(524289,64)@(64,64)"}
                    try:
                        Ag, Bg = A.cuda(), B.cuda()
                        got = (Ag @ Bg) if op == "mm" else F.linear(Ag, Bg.t().contiguous())
                        torch.cuda.synchronize()
                        rec.update(bad_count(got.cpu(), ref, bound, 2 ** 19))
                    except Exception as exc:  # noqa: BLE001
                        rec["error"] = repr(exc)[:400]
                    obs["cases"].append(rec)
            for dt in (torch.float32, torch.bfloat16, torch.float16):
                X, Y = X32.to(dt), Y32.to(dt)
                ref = torch.bmm(X.double(), Y.double())
                bound = 2 * eps[dt] * torch.bmm(X.double().abs(), Y.double().abs()) + eps[dt] * ref.abs()
                rec = {"lib": lib, "op": "bmm", "dtype": str(dt).replace("torch.", ""), "shape": "(65537,4,4)@(65537,4,4)"}
                try:
                    got = torch.bmm(X.cuda(), Y.cuda())
                    torch.cuda.synchronize()
                    rec.update(bad_count(got.cpu(), ref, bound, 2 ** 16))
                except Exception as exc:  # noqa: BLE001
                    rec["error"] = repr(exc)[:400]
                obs["cases"].append(rec)
        obs["any_bad"] = any((c.get("bad_elems") or 0) > 0 for c in obs["cases"])
        obs["n_cases_ok"] = sum(1 for c in obs["cases"] if "bad_elems" in c)
    except Exception:  # noqa: BLE001
        obs["error"] = traceback.format_exc()[-1500:]
    obs["seconds"] = round(time.time() - t0, 1)
    with open(a.out, "w", encoding = "utf-8") as fh:
        json.dump(obs, fh)
    return 0


if __name__ == "__main__":
    sys.exit(main())
