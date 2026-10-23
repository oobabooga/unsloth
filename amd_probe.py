"""Real gfx1151 props, and the same props with is_integrated hidden as HIP before 6.2 leaves it."""
import json, sys, traceback
out_path, label = sys.argv[1], sys.argv[2]
res = {"label": label}
try:
    import torch
    props = torch.cuda.get_device_properties(0)
    res["arch"] = getattr(props, "gcnArchName", None)
    res["is_integrated"] = getattr(props, "is_integrated", None)
    res["torch"] = torch.__version__
    class FlagUnset:
        def __init__(self, p): self._p = p
        def __getattr__(self, n):
            if n in ("is_integrated", "integrated"): return None
            return getattr(self._p, n)
    real_gdp = torch.cuda.get_device_properties
    for view, wrap in (("real", lambda p: p), ("flag_unset", FlagUnset)):
        torch.cuda.get_device_properties = lambda o=0, _w=wrap: _w(real_gdp(o))
        row = {}
        for name, fn in (
            ("training", lambda: __import__("core.training.worker", fromlist=["x"])._rocm_classify_unified_memory(torch.cuda.get_device_properties(0))[1]),
            ("inventory", lambda: __import__("utils.hardware.hardware", fromlist=["x"])._rocm_props_unified_status(torch.cuda.get_device_properties(0))),
            ("llama_cpp", lambda: sorted(__import__("core.inference.llama_cpp", fromlist=["x"]).LlamaCppBackend._rocm_unified_memory_gpu_ids())),
            ("diffusion", lambda: __import__("core.inference.diffusion_memory", fromlist=["x"])._cuda_memory("cuda")[2]),
        ):
            try:
                row[name] = fn()
            except Exception as e:
                row[name] = "ERR " + repr(e)[:200]
        res[view] = row
    torch.cuda.get_device_properties = real_gdp
    try:
        from core.inference.presence_penalty import seeded_generation
        torch.manual_seed(7)
        before = torch.cuda.get_rng_state().clone()
        with seeded_generation(1234):
            a = torch.rand(8, device="cuda")
        restored = bool(torch.equal(torch.cuda.get_rng_state(), before))
        with seeded_generation(1234):
            b = torch.rand(8, device="cuda")
        res["seed_on_rocm"] = {"repeat": bool(torch.equal(a, b)), "rng_restored": restored}
    except ImportError:
        res["seed_on_rocm"] = "n/a on this branch"
    except Exception as e:
        res["seed_on_rocm"] = "ERR " + repr(e)[:200]
except Exception:
    res["fatal"] = traceback.format_exc()[-2000:]
json.dump(res, open(out_path, "w"), indent=1, default=str)
