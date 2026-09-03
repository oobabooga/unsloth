"""macOS: validate the Apple free-memory contract against the REAL mx.device_info().

The contract at head is free = min(host available, max_recommended_working_set_size).
The working set is a per-process budget and the AGX counter is whole-device, so free
must NOT move when device-wide GPU use moves. That invariant is what this checks on
real Apple Silicon rather than against a mock.
"""
import sys
from unittest.mock import patch

sys.path.insert(0, __import__("os").path.abspath(sys.argv[1]))
import psutil
import mlx.core as mx
from utils.hardware import hardware as hw
from utils.hardware.hardware import DeviceType

GB = 1024 ** 3
info = mx.device_info()
rec = int(info.get("max_recommended_working_set_size") or 0)
print("device_info keys:", sorted(info.keys()))
print(f"real recommended working set = {rec / GB:.3f} GB")
if rec <= 0:
    print("FAIL: no positive max_recommended_working_set_size")
    sys.exit(1)

results, fails = [], []
with patch.object(hw, "get_device", return_value=DeviceType.MLX):
    for alloc_gb in (0.0, rec / GB * 0.25, rec / GB * 0.9, rec / GB * 1.5):
        with patch.object(hw, "_read_apple_gpu_stats",
                          return_value={"vram_used_bytes": int(alloc_gb * GB), "utilization_pct": 5}):
            mem = hw.get_gpu_memory_info()
            dev = hw.get_visible_gpu_utilization()["devices"][0]
        avail = psutil.virtual_memory().available / GB
        free = mem["free_gb"]
        expected = min(avail, rec / GB)
        ok = (mem["available"] is True
              and 0 <= free <= avail + 0.5
              and abs(free - expected) < 0.5
              and abs(dev["vram_free_gb"] - round(free, 2)) < 0.05)
        results.append(free)
        print(f"  device-wide in use={alloc_gb:6.2f} GB  host avail={avail:6.2f}"
              f"  -> free={free:6.3f} (expect {expected:6.3f})  dev={dev['vram_free_gb']:6.2f}"
              f"  {'ok' if ok else 'FAIL'}")
        if not ok:
            fails.append(alloc_gb)

spread = max(results) - min(results)
print(f"\nfree spread across a 0 -> 1.5x working-set sweep of device-wide GPU use: {spread:.3f} GB")
if spread > 0.5:
    print("FAIL: free moved with device-wide GPU use; the per-process budget is being charged twice")
    fails.append("spread")
if fails:
    print("MLX CONTRACT FAIL", fails)
    sys.exit(1)
print("MLX CONTRACT OK: free = min(host available, working set) and is invariant to device-wide GPU use")
