import os, re, sys
import torch
from importlib import metadata

print("VERSION", torch.__version__, "HIP", getattr(torch.version, "hip", None))
try:
    print("ARCHLIST", sorted(torch.cuda.get_arch_list()))
except Exception as e:
    print("ARCHLIST_ERR", type(e).__name__, e)
tp = os.path.dirname(torch.__file__)
for sub in ("lib/rocblas/library", "lib/hipblaslt/library"):
    p = os.path.join(tp, sub)
    if os.path.isdir(p):
        arch = sorted({m for f in os.listdir(p) for m in re.findall(r"gfx[0-9a-z]+", f)})
        print("BLASLIB", sub, arch)
try:
    print("ROCM_REQUIRES", metadata.requires("rocm"))
except Exception as e:
    print("ROCM_REQUIRES", "none:", type(e).__name__)
print("ROCM_SDK_DISTS", sorted(
    (d.metadata["Name"] or "") for d in metadata.distributions()
    if "rocm-sdk" in (d.metadata["Name"] or "")
))
