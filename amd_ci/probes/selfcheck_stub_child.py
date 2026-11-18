"""The checkout's ROCm varlen self-check, importing only hf_packed_attention (+ flash_impl) from the tree."""
import importlib, json, os, sys, time, types
import torch

checkout, out = sys.argv[1], sys.argv[2]
res = {"torch": torch.__version__, "hip": torch.version.hip}
try:
    for name, rel in (("unsloth", "unsloth"), ("unsloth.utils", os.path.join("unsloth", "utils"))):
        mod = types.ModuleType(name)
        mod.__path__ = [os.path.join(checkout, rel)]
        sys.modules[name] = mod
    hpa = importlib.import_module("unsloth.utils.hf_packed_attention")
    res["file"] = hpa.__file__
    res["gpu"] = torch.cuda.get_device_name(0)
    dev = torch.device("cuda", 0)
    t = time.perf_counter()
    res["torch_varlen"] = hpa._torch_varlen(dev) is not None
    res["check_s"] = time.perf_counter() - t
    if hasattr(hpa, "_hip_window_ok"):
        t = time.perf_counter()
        res["window_ok"] = hpa._hip_window_ok(dev, 512)
        res["window_check_s"] = time.perf_counter() - t
except Exception as e:
    import traceback
    res["error"] = f"{type(e).__name__}: {e}"
    res["tb"] = traceback.format_exc()[-2000:]
with open(out, "w", encoding = "utf-8") as f:
    json.dump(res, f)
