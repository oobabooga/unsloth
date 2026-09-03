"""Deterministic base-vs-head routing / payload identity probe.

Imports ONE tree per process (argv: <backend_root> <label>) and dumps every
hardware-routing answer Studio makes, so the two trees can be diffed byte for byte.
On any non-Apple platform the two dumps MUST be identical: the PR may only move the
Apple unified-memory numbers.
"""
import json, os, platform, sys

root, label, outfile = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, root)
os.chdir(root)

out = {"label": label, "platform": platform.system(), "machine": platform.machine()}


def safe(name, fn):
    try:
        out[name] = fn()
    except Exception as e:
        out[name] = {"__exc__": f"{type(e).__name__}: {e}"}


from utils.hardware import hardware as hw

safe("is_apple_silicon", lambda: hw.is_apple_silicon())
safe("get_device", lambda: hw.get_device().value)
safe("backend_label", lambda: hw._backend_label(hw.get_device()))
safe("chat_only", lambda: bool(getattr(hw, "CHAT_ONLY", None)))
safe("chat_only_reason", lambda: getattr(hw, "CHAT_ONLY_REASON", None))
safe("gpu_memory_info", lambda: hw.get_gpu_memory_info())
safe("gpu_summary", lambda: hw.get_gpu_summary())
safe("backend_visible", lambda: hw.get_backend_visible_gpu_info())
safe("visible_utilization", lambda: hw.get_visible_gpu_utilization())
safe("vulkan_inference", lambda: hw.get_vulkan_inference_gpu_info())
safe("export_capability", lambda: hw.export_capability())
safe("video_capability", lambda: hw.video_capability())
safe("visible_gpu_count", lambda: hw.get_visible_gpu_count())
safe("physical_gpu_count", lambda: hw.get_physical_gpu_count())

open(outfile, "w").write(json.dumps(out, sort_keys=True, default=str))
print(f"wrote {outfile}")
