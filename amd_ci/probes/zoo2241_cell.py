#!/usr/bin/env python3
"""One cell for the unsloth-zoo PR 2241 probe (Inductor rand4x off on torch 2.13 / 2.14). Observes only; JSON via --out.

--mode state    import torch's TritonOverrides, then the state's unsloth_zoo (first on PYTHONPATH), and record
                whether the 4x path exists, whether zoo replaced it, and the compile cache key tag.
--mode dropout  the same import, then rand4x_proof.py's CHILD statistics (cases A-D, generated-code scan) run
                on the GPU in THIS process, i.e. with whatever the state's unsloth_zoo import did to Inductor.
                Caches are off in every thread (the probe sets TORCHINDUCTOR_FORCE_DISABLE_CACHES; CHILD's
                force_disable_caches assignment reaches only this thread on torch 2.12+, not the autograd thread
                where case C's checkpoint recompute compiles), so each state compiles fresh.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent


def state_info() -> dict:
    import torch
    from torch._inductor.codegen.triton import TritonOverrides
    info: dict = {"torch": torch.__version__, "hip": torch.version.hip, "cuda_available": torch.cuda.is_available()}
    if info["cuda_available"]:
        p = torch.cuda.get_device_properties(0)
        info["device"] = {"name": p.name, "arch": getattr(p, "gcnArchName", None)}
    info["has_4x_path_before_import"] = "_can_use_4x_random" in TritonOverrides.__dict__
    info["upstream_use_rand4x_config"] = hasattr(torch._inductor.config, "use_rand4x")
    import unsloth_zoo
    info["unsloth_zoo_file"] = unsloth_zoo.__file__

    def patched():
        fn = TritonOverrides.__dict__.get("_can_use_4x_random")
        f = getattr(fn, "__func__", fn)
        return bool(fn is not None and getattr(f, "_unsloth_rand4x_off", False))

    # On a GPU host `import unsloth_zoo` imports temporary_patches itself; unsloth (models/_utils.py) also imports
    # it explicitly. Do what unsloth does, and record whether the bare import already did it.
    info["patched_after_zoo_import"] = patched()
    import unsloth_zoo.temporary_patches  # noqa: F401
    info["patched"] = patched()
    cfg = getattr(torch.compiler, "config", None)
    info["cache_key_tag"] = getattr(cfg, "cache_key_tag", None)
    info["tag_present"] = "inductor_rand4x_off" in str(info["cache_key_tag"] or "")
    info["state_has_patch_module"] = (Path.cwd() / "unsloth_zoo/temporary_patches/inductor_rand4x.py").is_file()
    return info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices = ("state", "dropout"), required = True)
    ap.add_argument("--out", required = True)
    args = ap.parse_args()
    rec: dict = {"mode": args.mode, "stage": "state"}
    try:
        rec["state"] = state_info()
        if args.mode == "dropout":
            rec["stage"] = "dropout"
            sys.path.insert(0, str(HERE))
            import rand4x_proof
            ns: dict = {"__name__": "rand4x_child"}
            argv = sys.argv
            sys.argv = ["rand4x_child", "zoo_import", ""]  # arm name: never loads a patch file by path
            try:
                exec(rand4x_proof.CHILD, ns)
            finally:
                sys.argv = argv
            rec["dropout"] = ns.get("res")
            rec["stage"] = "done"
        else:
            rec["stage"] = "done"
    except Exception as e:  # noqa: BLE001
        rec["fatal"] = repr(e)[:1500]
        rec["traceback"] = traceback.format_exc()[-4000:]
    with open(args.out, "w", encoding = "utf-8") as fh:
        json.dump(rec, fh, indent = 2, default = str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
