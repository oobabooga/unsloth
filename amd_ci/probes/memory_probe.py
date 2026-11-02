import argparse
import dataclasses
import json
from pathlib import Path
import sys
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", required=True)
    parser.add_argument("--checkout", required=True)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    result = {"state": args.state}
    try:
        sys.path.insert(0, str(Path(args.checkout) / "studio/backend"))
        import torch
        from core.inference import diffusion_memory as memory

        result.update(torch=torch.__version__, hip=torch.version.hip, gpu=torch.cuda.get_device_name())
        target = SimpleNamespace(device="cuda", backend="rocm")
        first = memory.settled_snapshot_device_memory(target, delay_s=0)
        result["snapshot"] = dataclasses.asdict(first)
        before = torch.cuda.memory_allocated()
        held = torch.empty(64 * 2**20, device="cuda", dtype=torch.uint8)
        held.fill_(1)
        torch.cuda.synchronize()
        result["allocated_delta"] = torch.cuda.memory_allocated() - before
        result["sum"] = int(held.sum().cpu().item())
        result["after"] = dataclasses.asdict(memory.settled_snapshot_device_memory(target, delay_s=0))
        check = getattr(memory, "settled_memory_unchanged", None)
        if check is not None:
            result["reuse_probe"] = check(target, first)
        result["source"] = memory.__file__
        assert Path(memory.__file__).resolve().is_relative_to(Path(args.checkout).resolve())
    except Exception as error:
        result["error"] = repr(error)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
