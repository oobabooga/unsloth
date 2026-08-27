import sys
from importlib import metadata

for dist in ("torch", "rocm"):
    try:
        print(f"{dist}.requires =", metadata.requires(dist))
    except Exception as e:
        print(f"{dist}.requires = ABSENT ({type(e).__name__})")
try:
    import torch
    print("torch.__version__ =", torch.__version__)
except Exception as e:
    print("import torch failed:", e)
print("rocm-sdk dists =", sorted(
    (d.metadata["Name"] or "") for d in metadata.distributions()
    if "rocm-sdk" in (d.metadata["Name"] or "") or (d.metadata["Name"] or "") == "rocm"
))
