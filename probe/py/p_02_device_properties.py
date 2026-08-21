import torch
p = torch.cuda.get_device_properties(0)
print("props:", p)
for a in ("name", "total_memory", "gcnArchName", "is_integrated", "integrated",
          "multi_processor_count", "major", "minor", "L2_cache_size", "warp_size", "uuid"):
    print("   ", a, "=", getattr(p, a, "<missing>"))
