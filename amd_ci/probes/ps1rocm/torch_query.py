"""Run with a venv's python (-I): what torch it holds and whether it reaches the GPU. Writes JSON to --out."""
import argparse
import json
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--out", required = True)
a = ap.parse_args()
res = {"python": sys.version.split()[0]}
try:
    import torch
    res.update({"version": torch.__version__, "hip": getattr(torch.version, "hip", None),
                "cuda": torch.version.cuda, "is_available": bool(torch.cuda.is_available()),
                "device_count": torch.cuda.device_count()})
    if res["is_available"]:
        props = torch.cuda.get_device_properties(0)
        res["name"] = torch.cuda.get_device_name(0)
        res["arch"] = getattr(props, "gcnArchName", None)
        x = torch.randn(256, 256, device = "cuda")
        y = (x @ x).sum().item()
        torch.cuda.synchronize()
        res["matmul_on_gpu"] = bool(y == y)
except Exception as e:  # noqa: BLE001
    res["error"] = f"{type(e).__name__}: {e}"
with open(a.out, "w", encoding = "utf-8") as f:
    json.dump(res, f)
