#!/usr/bin/env python3
"""Items 5.1 / 5.2 (lemonade-sdk/thenoise f3bb2b41f4): bf16 F.linear on gfx1151 at DiT block shapes.

Arms, interleaved in one process (the harness runs 3+ processes for the A/A spread):
  base       W [N, K] row-major, x [M, K] contiguous (what Studio runs)
  colmajor   W stored column-major (W.t().contiguous().t()), x contiguous                         (5.1)
  pad8       x rows padded to K + 8 elements (pitch view, buffer prepared outside the timed call)   (5.2 best case)
  pad8_copy  as pad8 but the copy into the padded buffer is inside the timed call                  (5.2 honest)
  pad64      x rows padded to K + 64
  col_pad8   colmajor W and pad8 x
Every arm vs base: torch.equal and max abs diff (BLAS may pick another kernel; the tolerance is the reader's)."""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a12common import env_info, interleave, write  # noqa: E402

# model -> M (tokens) and the block's linears (name, K, N, count per block)
MODELS = {
    "wan5b_720p": (27280, [("qkvo", 3072, 3072, 5), ("ffn_up", 3072, 14336, 1), ("ffn_down", 14336, 3072, 1)]),
    "krea2_1024": (4608, [("q_o", 6144, 6144, 2), ("kv", 6144, 3072, 1), ("gate_up", 6144, 16384, 2),
                          ("down", 16384, 6144, 1)]),
    "qwen21_1024": (4096, [("qkvo", 4096, 4096, 4), ("ffn_up", 4096, 12288, 1), ("ffn_down", 12288, 4096, 1)]),
    "ltx25": (6144, [("qkvo", 4096, 4096, 5), ("ffn_up", 4096, 16384, 1), ("ffn_down", 16384, 4096, 1)]),
    "klein_1024": (4608, [("o", 3072, 3072, 1), ("fused_in", 3072, 18432, 1), ("down", 9216, 3072, 1)]),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required = True)
    ap.add_argument("--rounds", type = int, default = 7)
    ap.add_argument("--models", default = ",".join(MODELS))
    ap.add_argument("--tunableop", help = "Studio TunableOp csv to READ (tuning off), as diffusion_rocm_tunableop does")
    a = ap.parse_args()
    if a.tunableop:
        import shutil
        tab = os.path.join(os.path.dirname(os.path.abspath(a.out)), "tunableop_read.csv")
        shutil.copyfile(a.tunableop, tab)
        os.environ.update({"PYTORCH_TUNABLEOP_ENABLED": "1", "PYTORCH_TUNABLEOP_TUNING": "0",
                           "PYTORCH_TUNABLEOP_RECORD_UNTUNED": "0", "PYTORCH_TUNABLEOP_FILENAME": tab})
    obs = {"rows": []}
    t0 = time.time()
    try:
        import torch
        import torch.nn.functional as F
        obs["env"] = env_info()
        obs["blas"] = str(torch.backends.cuda.preferred_blas_library())
        obs["tunableop"] = {"enabled": torch.cuda.tunable.is_enabled(), "tuning": torch.cuda.tunable.tuning_is_enabled(),
                            "file": a.tunableop}
        for model in a.models.split(","):
            M, lins = MODELS[model]
            for name, K, N, cnt in lins:
                g = torch.Generator(device = "cuda").manual_seed(K * 7 + N)
                x = (torch.randn(M, K, generator = g, device = "cuda") * 0.5).to(torch.bfloat16)
                W = (torch.randn(N, K, generator = g, device = "cuda") * K ** -0.5).to(torch.bfloat16)
                b = torch.randn(N, generator = g, device = "cuda").to(torch.bfloat16) * 0.1
                Wc = W.t().contiguous().t()
                pads = {}
                for p in (8, 64):
                    buf = torch.empty(M, K + p, device = "cuda", dtype = torch.bfloat16)
                    buf[:, :K].copy_(x)
                    pads[p] = buf
                xp8, xp64 = pads[8][:, :K], pads[64][:, :K]
                wbuf = torch.empty(N, K + 64, device = "cuda", dtype = torch.bfloat16)
                wbuf[:, :K].copy_(W)
                Wp64 = wbuf[:, :K]
                fns = {"base": lambda: F.linear(x, W, b), "colmajor": lambda: F.linear(x, Wc, b),
                       "pad8": lambda: F.linear(xp8, W, b),
                       "pad8_copy": lambda: F.linear(pads[8][:, :K].copy_(x), W, b),
                       "pad64": lambda: F.linear(xp64, W, b), "col_pad8": lambda: F.linear(xp8, Wc, b),
                       "pad64_copy": lambda: F.linear(pads[64][:, :K].copy_(x), W, b),
                       "wpad64": lambda: F.linear(x, Wp64, b), "wpad64_pad64": lambda: F.linear(xp64, Wp64, b)}
                with torch.no_grad():
                    ref = fns["base"]()
                    exact, mad = {}, {}
                    for arm, fn in fns.items():
                        y = fn()
                        exact[arm] = bool(torch.equal(y, ref))
                        mad[arm] = float((y.float() - ref.float()).abs().max())
                    res = interleave(fns, a.rounds)
                obs["rows"].append({"model": model, "lin": name, "M": M, "K": K, "N": N, "count": cnt,
                                    "pitch_bytes": K * 2, "exact": exact, "max_abs": mad,
                                    "med_ms": {k: res[k]["med"] for k in fns}, "ms": {k: res[k]["ms"] for k in fns}})
                del x, W, Wc, pads, ref, fns, wbuf, Wp64
                torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        import traceback
        obs["error"] = traceback.format_exc()[-1500:]
    try:
        import torch
        obs["tunableop_results_n"] = len(torch.cuda.tunable.get_results() or [])
    except Exception as exc:  # noqa: BLE001
        obs["tunableop_results_n"] = repr(exc)[:200]
    obs["seconds"] = round(time.time() - t0, 1)
    write(a.out, obs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
