"""macOS only: exercise the allocated > 0 branch with the REAL mx.device_info()."""
import sys
from unittest.mock import patch

sys.path.insert(0, sys.argv[1])
import mlx.core as mx
from utils.hardware import hardware as hw
from utils.hardware.hardware import DeviceType

GB = 1024 ** 3
rec = int(mx.device_info().get("max_recommended_working_set_size") or 0)
assert rec > 0, "no positive max_recommended_working_set_size on this host"
print(f"real recommended working set = {rec / GB:.3f} GB")

fails = []
with patch.object(hw, "get_device", return_value=DeviceType.MLX):
    for alloc_gb in (0.0, rec / GB * 0.25, rec / GB * 0.5, rec / GB * 0.9, rec / GB * 1.5):
        with patch.object(hw, "_read_apple_gpu_stats",
                          return_value={"vram_used_bytes": int(alloc_gb * GB), "utilization_pct": 5}):
            mem = hw.get_gpu_memory_info()
            dev = hw.get_visible_gpu_utilization()["devices"][0]
        import psutil
        avail = psutil.virtual_memory().available / GB
        headroom = max(0.0, rec / GB - alloc_gb)
        free = mem["free_gb"]
        ok = (0 <= free <= avail + 0.5) and (free <= headroom + 0.5) \
             and abs(dev["vram_free_gb"] - round(free, 2)) < 0.05 and mem["available"] is True
        print(f"  alloc={alloc_gb:6.2f} GB  avail={avail:6.2f}  headroom={headroom:6.2f}"
              f"  -> free={free:6.3f}  dev={dev['vram_free_gb']:6.2f}  {'ok' if ok else 'FAIL'}")
        if not ok:
            fails.append(alloc_gb)
if fails:
    print("MLX BRANCH FAIL", fails)
    sys.exit(1)
print("MLX BRANCH OK: the allocated > 0 subtraction behaves on real Apple Silicon")
