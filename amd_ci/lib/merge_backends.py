#!/usr/bin/env python3
"""Put two ggml backends in one directory, so one binary sees both.

The upstream Linux releases are built with `GGML_BACKEND_DL=ON`, so each backend
is a separate `libggml-<name>.so` that llama.cpp dlopens from the directory
beside the executable. The Vulkan tarball ships `libggml-vulkan.so`, the ROCm
tarball ships `libggml-hip.so`, and neither ships the other. Copying one across
gives a single build that enumerates `Vulkan0` AND `ROCm0` - two devices backed
by the same physical GPU.

That is worth doing because it is not a trick, it is the configuration the
upstream reports blame. On llama.cpp#24492 @Kononnable narrowed the
"pre-allocated tensor (cache_k_lNN) ... cannot run the operation (NONE)" abort
to exactly this: "I can reproduce it only when I'm building binary with both
backends active at the same time", with the fix being to name the draft device
explicitly (`--spec-draft-device Vulkan0`). Plenty of AMD users run dual-backend
builds without realising it.

VERIFIED on a host with one physical GPU: dropping `libggml-vulkan.so` into a
ROCm build made the Vulkan devices appear in `--list-devices`.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--into", required = True, type = Path,
                    help = "the build that keeps its binaries")
    ap.add_argument("--from-dir", required = True, type = Path,
                    help = "the build to take backend libraries from")
    ap.add_argument("--lib", action = "append", default = [],
                    help = "repeatable; defaults to libggml-vulkan.so")
    ap.add_argument("--dest", required = True, type = Path,
                    help = "where to assemble the merged build")
    ap.add_argument("--out", type = Path, default = None)
    args = ap.parse_args()

    libs = args.lib or ["libggml-vulkan.so"]
    info: dict = {"into": str(args.into), "from": str(args.from_dir),
                  "libs": libs, "dest": str(args.dest)}

    if args.dest.exists():
        shutil.rmtree(args.dest)
    shutil.copytree(args.into, args.dest, symlinks = True)

    copied, missing = [], []
    for name in libs:
        src = args.from_dir / name
        if src.is_file():
            shutil.copy2(src, args.dest / name)
            copied.append(name)
        else:
            missing.append(name)
    info["copied"] = copied
    info["missing"] = missing

    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = str(args.dest)
    try:
        p = subprocess.run([str(args.dest / "llama-server"), "--list-devices"],
                           capture_output = True, text = True, timeout = 300, env = env)
        text = ((p.stdout or "") + (p.stderr or "")).strip()
        info["list_devices"] = text[:2000]
        # The merge is only real if BOTH backends enumerated a device. Reporting
        # a merged directory that quietly loaded one backend would put a
        # single-device run behind a multi-device label.
        info["has_vulkan"] = "Vulkan" in text
        info["has_rocm"] = "ROCm" in text
        info["both_backends"] = info["has_vulkan"] and info["has_rocm"]
    except Exception as e:  # noqa: BLE001
        info["error"] = f"{type(e).__name__}: {e}"
        info["both_backends"] = False

    print(json.dumps(info, indent = 2))
    if args.out:
        args.out.parent.mkdir(parents = True, exist_ok = True)
        args.out.write_text(json.dumps(info, indent = 2))
    return 0 if info.get("both_backends") and not missing else 1


if __name__ == "__main__":
    sys.exit(main())
