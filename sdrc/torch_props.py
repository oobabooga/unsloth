"""J3 probe: the raw torch device properties Studio's two Strix Halo classifiers read, and what each decides.
Mirrors studio/backend/core/inference/diffusion_memory.py:_cuda_memory (kind) and
studio/backend/core/training/worker.py:_rocm_classify_unified_memory at unsloth 22a024f0.  usage: torch_props.py OUT.json"""
import json, sys

res = {}
try:
    import torch
    res["torch"] = torch.__version__
    res["hip"] = getattr(torch.version, "hip", None)
    res["cuda_available"] = torch.cuda.is_available()
    if res["cuda_available"]:
        p = torch.cuda.get_device_properties(torch.cuda.current_device())
        res["props_repr"] = repr(p)
        for a in ("name", "gcnArchName", "is_integrated", "integrated", "total_memory", "multi_processor_count",
                  "major", "minor"):
            v = getattr(p, a, "<absent>")
            res[a] = v if isinstance(v, (int, float, str, bool)) or v is None else repr(v)
        free, total = torch.cuda.mem_get_info()
        res["mem_get_info_MiB"] = [free // 2**20, total // 2**20]
        diff_kind = "unified_memory" if bool(getattr(p, "integrated", False) or getattr(p, "is_integrated", False)) else "discrete_vram"
        gcn = (getattr(p, "gcnArchName", "") or "").split(":")[0].strip()
        if getattr(p, "is_integrated", 0):
            shared = True
        elif gcn:
            shared = gcn.lower() in {"gfx1150", "gfx1151", "gfx1152"}
        else:
            shared = "8060s" in (getattr(p, "name", "") or "").lower()
        res["diffusion_memory_kind"] = diff_kind
        res["shared_classifier_unified"] = shared
        res["disagree"] = (diff_kind == "unified_memory") != shared
except Exception as e:  # noqa: BLE001
    res["error"] = repr(e)
open(sys.argv[1], "w", encoding="utf-8").write(json.dumps(res, indent=1))
