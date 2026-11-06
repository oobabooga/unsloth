#!/usr/bin/env python3
"""Render capability.py's 'Not tested here' for what PR 16857 touches. Writes --out (UTF-8)."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))
from capability import detect, untested_section  # noqa: E402

# The PR changes llama-server argv for an oversized load on an integrated GPU: CUDA/HIP prop.integrated
# or a Vulkan integrated device, Linux + Windows (fit-target on Windows only); discrete GPUs must stay untouched.
NEEDS = ["linux", "windows", "rocm", "vulkan", "gpu", "integrated_gpu", "discrete_gpu", "nvidia", "exclusive_gpu"]

ap = argparse.ArgumentParser()
ap.add_argument("--out", required = True)
a = ap.parse_args()
p = detect(require_torch = False)
txt = untested_section(p, NEEDS) + "\n\nhost profile:\n" + p.to_json() + "\n"
Path(a.out).write_text(txt, encoding = "utf-8")
print(txt)
