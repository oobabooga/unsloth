#!/usr/bin/env python3
"""Studio INT8 attention on every visible card, the rule a multi-GPU load (CFG replica / layer split) relies on: the
op runs the kernel on a card only after THAT card's own accuracy check passed, and native attention there before.

Per card c, at [1, S, H, D] strided views on cuda:c:
  1. before verdict(c): one int8_attention_call -> must run native (STATS native +1, int8 +0)
  2. verdict(c)
  3. after: one call -> must run the kernel (int8 +1), output on cuda:c, accuracy vs fp32 on that card
  4. cross-check: card 0's verdict alone never enables card 1 (step 1 on card 1 runs after card 0's verdict)
Writes one JSON: python mgpu_check.py --studio BACKEND --out mgpu.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--studio", required = True)
    ap.add_argument("--out", required = True)
    ap.add_argument("--shapes", default = "4096x24x128,16384x24x128,8192x56x64")
    a = ap.parse_args()
    os.environ.setdefault("UNSLOTH_STUDIO_HOME", tempfile.mkdtemp(prefix = "mg_home_"))
    sys.path.insert(0, a.studio)
    import torch

    from core.inference import diffusion_int8_attention as I8

    n = torch.cuda.device_count()
    res = {"cards": n, "names": [torch.cuda.get_device_name(i) for i in range(n)], "torch": torch.__version__,
           "checks": [], "ok": None}
    if n < 2:
        res["ok"] = None
        res["why"] = "NOT_RUN: fewer than 2 cards visible"
    dt = torch.float16 if torch.cuda.get_device_capability(0) < (8, 0) else torch.bfloat16
    for shape in a.shapes.split(","):
        S, H, D = (int(x) for x in shape.split("x"))
        for c in range(n):
            dev = torch.device("cuda", c)
            q, k, v = I8.probe_inputs(dev, dt, S, H, D, seed = c)
            buf = torch.empty((1, S, 3, H, D), device = dev, dtype = dt)
            for i, t in enumerate((q, k, v)):
                buf[:, :, i].copy_(t)
            q, k, v = buf[:, :, 0], buf[:, :, 1], buf[:, :, 2]
            row = {"shape": shape, "card": c, "dtype": str(dt)[6:]}
            key_known = any(kk[0] == c and kk[3] == D and kk[2] == H for kk in I8._VERDICTS)
            s0 = dict(I8.STATS)
            with torch.cuda.device(c):
                o0 = I8.int8_attention_call(q, k, v)
                torch.cuda.synchronize(c)
            row["before_verdict"] = {"int8": I8.STATS["int8"] - s0["int8"], "native": I8.STATS["native"] - s0["native"],
                                     "verdict_already_cached": key_known}
            with torch.cuda.device(c):
                ver = I8.verdict(dev, dt, H, D, timed = True)
            row["verdict"] = {kk: ver.get(kk) for kk in ("ok", "impl", "cos", "rel_l1", "min_tokens", "error")}
            s1 = dict(I8.STATS)
            with torch.cuda.device(c):
                o1 = I8.int8_attention_call(q, k, v)
                torch.cuda.synchronize(c)
            row["after_verdict"] = {"int8": I8.STATS["int8"] - s1["int8"], "native": I8.STATS["native"] - s1["native"]}
            row["out_device"] = str(o1.device)
            ref = I8.reference_attention(q, k, v)
            row["acc"] = {kk: round(x, 6) for kk, x in I8.accuracy(o1, ref).items()}
            row["acc_before(native)"] = {kk: round(x, 6) for kk, x in I8.accuracy(o0, ref).items()}
            gated = key_known or (row["before_verdict"]["native"] == 1 and row["before_verdict"]["int8"] == 0)
            engaged = (not ver.get("ok")) or (row["after_verdict"]["int8"] == 1 and row["out_device"] == str(dev))
            accurate = (not ver.get("ok")) or (row["acc"]["cos"] >= I8.MIN_COSINE and row["acc"]["rel_l1"] <= I8.MAX_REL_L1)
            row["pass"] = {"gated_before_own_verdict": gated, "engaged_after": engaged, "accurate": accurate}
            print("MGPU", json.dumps(row), flush = True)
            res["checks"].append(row)
            del q, k, v, buf, o0, o1, ref
            torch.cuda.empty_cache()
    if n >= 2:
        res["ok"] = all(all(r["pass"].values()) for r in res["checks"])
        res["both_cards_engaged"] = all(any(r["card"] == c and r["after_verdict"]["int8"] == 1 for r in res["checks"])
                                        for c in range(n))
    res["stats"] = dict(I8.STATS)
    json.dump(res, open(a.out, "w"), indent = 1, default = str)
    print("MGPU_DONE", res["ok"], flush = True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
