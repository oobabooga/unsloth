#!/usr/bin/env python3
"""R6-A item 3 (n6#6): TunableOp tables for Wan 2.2 5B, LTX-2.5 and MiniMax-H3 inference GEMMs on gfx1151.

  tunable_r6.py tune  --out F --table T.csv [--models ...] [--atol 1e-2 --rtol 1e-2]
  tunable_r6.py bench --out F --table T.csv [--models ...] [--rounds 7]

tune: TunableOp on, tuning on, rotating buffer 512 MiB (PYTORCH_TUNABLEOP_ROTATING_BUFFER_SIZE=512, the
  pytorch-tunableop-offline blog), Studio's limits (diffusion_rocm_tunableop.py: MAX_TUNING_MS 30, MAX_TUNING_ITERS 20),
  numerical check on (set_numerical_check_tolerances(True, atol, rtol)); each block GEMM is called once (the tuning),
  seconds per GEMM recorded; the table is written in torch's CSV layout (validators + results) as Studio's _write does.
bench: table READ only (tuning off), then per GEMM the off / on arms interleaved in the same process by
  torch.cuda.tunable.enable(False / True) (TunableOp is a runtime switch); outputs compared (on vs off: equal, max abs);
  the table's rows that match each GEMM are listed (engagement). The harness runs 3 bench processes (A/A spread).
Block linear sums per model = sum(count x median ms)."""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# model -> (M tokens, [(name, K, N, count per block)]); linears carry bias (diffusers Wan / LTX / H3 attention + FFN)
MODELS = {
    "wan5b_27280": (27280, [("qkvo", 3072, 3072, 4), ("ffn_up", 3072, 14336, 1), ("ffn_down", 14336, 3072, 1)]),
    "wan5b_5070": (5070, [("qkvo", 3072, 3072, 4), ("ffn_up", 3072, 14336, 1), ("ffn_down", 14336, 3072, 1)]),
    "ltx25_6144": (6144, [("qkvo", 4096, 4096, 4), ("ffn_up", 4096, 16384, 1), ("ffn_down", 16384, 4096, 1)]),
    "h3_37720": (37720, [("qkvo", 5376, 5376, 4), ("ffn_up", 5376, 14336, 1), ("ffn_down", 14336, 5376, 1)]),
}


def _tensors(torch, M, K, N):
    g = torch.Generator(device = "cuda").manual_seed(K * 7 + N + M)
    x = (torch.randn(M, K, generator = g, device = "cuda") * 0.5).to(torch.bfloat16)
    W = (torch.randn(N, K, generator = g, device = "cuda") * K ** -0.5).to(torch.bfloat16)
    b = (torch.randn(N, generator = g, device = "cuda") * 0.1).to(torch.bfloat16)
    return x, W, b


def _write_table(tunable, path):
    validators = list(tunable.get_validators() or ())
    results = list(tunable.get_results() or ())
    lines = [f"Validator,{k},{v}" for k, v in validators] + [",".join(str(x) for x in r) for r in results]
    with open(path, "w", encoding = "utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return {str(k): str(v) for k, v in validators}, len(results)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices = ("tune", "bench"))
    ap.add_argument("--out", required = True)
    ap.add_argument("--table", required = True)
    ap.add_argument("--models", default = ",".join(MODELS))
    ap.add_argument("--rounds", type = int, default = 7)
    ap.add_argument("--atol", type = float, default = 1e-2)
    ap.add_argument("--rtol", type = float, default = 1e-2)
    a = ap.parse_args()
    for k in [k for k in os.environ if k.startswith("PYTORCH_TUNABLEOP_")]:
        os.environ.pop(k)
    if a.mode == "tune":
        os.environ["PYTORCH_TUNABLEOP_ROTATING_BUFFER_SIZE"] = "512"
    from a12common import env_info, interleave, write
    obs = {"mode": a.mode, "table": a.table, "rows": []}
    t0 = time.time()
    try:
        import torch
        import torch.nn.functional as F
        tun = torch.cuda.tunable
        obs["env"] = env_info()
        if a.mode == "tune":
            tun.enable(True)
            tun.tuning_enable(True)
            tun.record_untuned_enable(False)
            tun.set_max_tuning_duration(30)
            tun.set_max_tuning_iterations(20)
            tun.set_rotating_buffer_size(512)
            obs["numerical_check"] = None
            if hasattr(tun, "set_numerical_check_tolerances"):
                tun.set_numerical_check_tolerances(True, a.atol, a.rtol)
                obs["numerical_check"] = {"atol": a.atol, "rtol": a.rtol}
            tun.set_filename(a.table + ".torch", insert_device_ordinal = False)
            obs["rotating_buffer_mb"] = tun.get_rotating_buffer_size()
            for model in a.models.split(","):
                M, lins = MODELS[model]
                for name, K, N, cnt in lins:
                    x, W, b = _tensors(torch, M, K, N)
                    n0 = len(tun.get_results() or ())
                    torch.cuda.synchronize()
                    t = time.time()
                    with torch.no_grad():
                        F.linear(x, W, b)
                    torch.cuda.synchronize()
                    obs["rows"].append({"model": model, "lin": name, "M": M, "K": K, "N": N,
                                        "tune_s": round(time.time() - t, 1),
                                        "new_results": len(tun.get_results() or ()) - n0})
                    del x, W, b
                    torch.cuda.empty_cache()
                    obs["validators"], obs["n_results"] = _write_table(tun, a.table)
                    write(a.out, obs)
            obs["results"] = [list(map(str, r)) for r in (tun.get_results() or ())]
        else:
            tun.enable(True)
            tun.tuning_enable(False)
            tun.record_untuned_enable(False)
            tun.set_filename(os.path.join(os.path.dirname(os.path.abspath(a.out)), f"unused_{os.getpid()}.csv"),
                             insert_device_ordinal = False)
            with torch.no_grad():
                a8 = torch.ones(8, 8, dtype = torch.bfloat16, device = "cuda")
                (a8 @ a8).sum().item()  # validators register on the first tunable GEMM (Studio's _prime)
            obs["read_ok"] = bool(tun.read_file(a.table))
            loaded = [list(map(str, r)) for r in (tun.get_results() or ())]
            obs["n_loaded"] = len(loaded)
            obs["validators_now"] = {str(k): str(v) for k, v in (tun.get_validators() or ())}
            for model in a.models.split(","):
                M, lins = MODELS[model]
                for name, K, N, cnt in lins:
                    x, W, b = _tensors(torch, M, K, N)
                    key = f"_{N}_{M}_{K}_"
                    match = [r[:3] for r in loaded if key in r[1] + "_"]

                    def mk(on):
                        def fn():
                            tun.enable(on)
                            return F.linear(x, W, b)
                        return fn
                    fns = {"off": mk(False), "on": mk(True)}
                    with torch.no_grad():
                        y_off, y_on = fns["off"](), fns["on"]()
                        res = interleave(fns, a.rounds)
                    tun.enable(True)
                    obs["rows"].append({"model": model, "lin": name, "M": M, "K": K, "N": N, "count": cnt,
                                        "table_rows": match, "equal": bool(torch.equal(y_on, y_off)),
                                        "max_abs": float((y_on.float() - y_off.float()).abs().max()),
                                        "med_ms": {k: res[k]["med"] for k in fns},
                                        "ms": {k: res[k]["ms"] for k in fns}})
                    del x, W, b, y_on, y_off
                    torch.cuda.empty_cache()
                    write(a.out, obs)
            blocks = {}
            for r in obs["rows"]:
                d = blocks.setdefault(r["model"], {"off": 0.0, "on": 0.0})
                for k in ("off", "on"):
                    d[k] += r["count"] * r["med_ms"][k]
            obs["block_ms"] = blocks
    except Exception:  # noqa: BLE001
        import traceback
        obs["error"] = traceback.format_exc()[-1500:]
    obs["seconds"] = round(time.time() - t0, 1)
    write(a.out, obs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
