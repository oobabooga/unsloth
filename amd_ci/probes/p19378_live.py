#!/usr/bin/env python3
"""Live-host facts for PR 19378 at one state, run inside the state's studio/backend (observes only; JSON via --out).

Records the torch device, where `unsloth` resolves from (find_spec, no import), and, when the state has
core/training/parallel_plan.py: capabilities() and plan() for ddp / fsdp2 / single / layer_split / auto on this host.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import traceback


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend-dir", required = True)
    ap.add_argument("--out", required = True)
    args = ap.parse_args()
    out = os.path.abspath(args.out)
    rec: dict = {"platform": sys.platform, "python": sys.executable}

    def dump():
        with open(out, "w", encoding = "utf-8") as fh:
            json.dump(rec, fh, indent = 2, default = str)

    try:
        import torch

        ok = torch.cuda.is_available()
        p = torch.cuda.get_device_properties(0) if ok else None
        rec["torch"] = {"version": torch.__version__, "hip": torch.version.hip, "cuda_available": ok,
                        "device_count": torch.cuda.device_count() if ok else 0,
                        "arch": getattr(p, "gcnArchName", None) if p else None, "device": p.name if p else None}
        try:
            import torch.distributed as dist

            rec["torch"]["dist_available"] = dist.is_available()
            rec["torch"]["nccl_available"] = dist.is_available() and dist.is_nccl_available()
        except Exception as e:  # noqa: BLE001
            rec["torch"]["dist_error"] = repr(e)
    except Exception as e:  # noqa: BLE001
        rec["torch"] = {"error": repr(e)}
    dump()

    backend = os.path.abspath(args.backend_dir)
    sys.path.insert(0, backend)
    os.chdir(backend)
    spec = importlib.util.find_spec("unsloth")
    rec["unsloth_origin"] = spec.origin if spec else None
    rec["has_parallel_plan"] = os.path.isfile(os.path.join(backend, "core", "training", "parallel_plan.py"))
    if not rec["has_parallel_plan"]:
        dump()
        return 0
    try:
        from core.training import parallel_plan as pp

        host = pp.detect_host()
        rec["host"] = host
        rec["capabilities"] = pp.capabilities(host)
        rec["plans"] = {}
        for mode in ("ddp", "fsdp2", "single", "layer_split", None):
            key = mode or "auto"
            try:
                r = pp.plan({"parallel_mode": mode} if mode else {}, host, None)
                rec["plans"][key] = {"ok": True, "mode": r.get("mode"), "launcher": r.get("launcher"),
                                     "world_size": r.get("world_size"), "reason": r.get("reason"),
                                     "distributed_backend": r.get("distributed_backend")}
            except pp.ParallelPlanError as e:
                rec["plans"][key] = {"ok": False, "parallel_plan_error": True, "payload": e.payload}
            except Exception as e:  # noqa: BLE001
                rec["plans"][key] = {"ok": False, "parallel_plan_error": False, "error": repr(e),
                                     "tb": traceback.format_exc()[-1500:]}
    except Exception as e:  # noqa: BLE001
        rec["error"] = repr(e)
        rec["tb"] = traceback.format_exc()[-3000:]
    dump()
    return 0


if __name__ == "__main__":
    sys.exit(main())
