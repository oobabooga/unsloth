#!/usr/bin/env python3
"""C15-6 follow-up workload for ``rocprofv3 --kernel-trace``: a few named GEMMs and an elementwise op, then sync.
Writes a small JSON via --out (the trace itself is judged by the orchestrator from rocprofv3's output dir)."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required = True)
    a = ap.parse_args()
    obs = {"HSA_TOOLS_DISABLE_REGISTER": os.environ.get("HSA_TOOLS_DISABLE_REGISTER")}
    try:
        import torch
        x = torch.randn(1024, 1024, device = "cuda")
        for _ in range(10):
            x = (x @ x).sin()
        torch.cuda.synchronize()
        obs["ok"] = bool(torch.isfinite(x).all())
    except Exception as exc:  # noqa: BLE001
        obs["error"] = repr(exc)[:400]
    Path(a.out).write_text(json.dumps(obs), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
