# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

import argparse
import dataclasses
import inspect
import json
from pathlib import Path
import sys
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", required = True)
    parser.add_argument("--checkout", required = True)
    parser.add_argument("--out", required = True, type = Path)
    args = parser.parse_args()
    checkout = Path(args.checkout).resolve()
    result = {"state": args.state, "backend": "rocm"}
    try:
        sys.path.insert(0, str(checkout / "studio/backend"))
        import torch
        from core.inference import diffusion_memory as memory
        from utils.hardware import trusted_mem_get_info

        result.update(
            torch = torch.__version__,
            hip = torch.version.hip,
            gpu = torch.cuda.get_device_name(),
            source = memory.__file__,
            imports_checkout = Path(memory.__file__).resolve().is_relative_to(checkout),
            sampler_accepts_can_stop = "can_stop"
            in inspect.signature(memory.settled_snapshot_device_memory).parameters,
        )
        target = SimpleNamespace(device = "cuda", backend = "rocm")
        result["trusted_info"] = list(trusted_mem_get_info())
        result["snapshot"] = dataclasses.asdict(
            memory.settled_snapshot_device_memory(target, delay_s = 0)
        )
        before = torch.cuda.memory_allocated()
        held = torch.empty(64 * 2**20, device = "cuda", dtype = torch.uint8)
        held.fill_(1)
        torch.cuda.synchronize()
        result["allocated_delta"] = torch.cuda.memory_allocated() - before
        result["sum"] = int(held.sum().cpu().item())
        result["after"] = dataclasses.asdict(
            memory.settled_snapshot_device_memory(target, delay_s = 0)
        )
        del held
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    except Exception as error:
        result["error"] = repr(error)
    args.out.write_text(json.dumps(result, indent = 2), encoding = "utf-8")


if __name__ == "__main__":
    main()
