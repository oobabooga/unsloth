"""The checkout's own ROCm varlen self-check verdicts on this device, and their cost."""
import json, sys, time
import torch

res = {}
try:
    import unsloth.utils.hf_packed_attention as hpa
    res["file"] = hpa.__file__
    dev = torch.device("cuda", 0)
    t = time.perf_counter()
    res["torch_varlen"] = hpa._torch_varlen(dev) is not None
    res["check_s"] = time.perf_counter() - t
    if hasattr(hpa, "_hip_window_ok"):
        t = time.perf_counter()
        res["window_ok"] = hpa._hip_window_ok(dev, 512)
        res["window_check_s"] = time.perf_counter() - t
except Exception as e:
    res["error"] = f"{type(e).__name__}: {e}"
json.dump(res, open(sys.argv[1], "w"))
