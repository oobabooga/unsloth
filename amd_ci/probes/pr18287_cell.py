"""One cell for unsloth #18287 (run as its own process): what a torchrun rank > 0 does at `import unsloth`, i.e.
CUDA / HIP already started before unsloth_zoo exports its allocator settings. Records every allocator-setter
call, any FutureWarning about it, an exception if import raised, and whether a fresh 64 MiB segment is expandable.
argv: <out.json> <pre_init 0|1>. JSON to the file, never stdout."""
import json, os, sys, traceback, warnings
out, pre = sys.argv[1], sys.argv[2] == "1"
r = {"pre_init": pre, "env_in": {k: os.environ.get(k) for k in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_HIP_ALLOC_CONF")}}
try:
    import torch
    r["torch"] = torch.__version__; r["hip"] = torch.version.hip; r["gpu"] = torch.cuda.is_available()
    calls = []
    o1 = getattr(torch.cuda.memory, "_set_allocator_settings", None)
    if o1 is not None:
        def c1(v, _o = o1):
            calls.append(v); return _o(v)
        torch.cuda.memory._set_allocator_settings = c1
    if pre and r["gpu"]:
        torch.cuda.set_device(0)
    r["initialized_before_import"] = torch.cuda.is_initialized()
    with warnings.catch_warnings(record = True) as w:
        warnings.simplefilter("always")
        try:
            import unsloth  # noqa: F401
            r["unsloth_file"] = unsloth.__file__
        except Exception as e:  # noqa: BLE001
            r["import_error"] = "".join(traceback.format_exception_only(type(e), e))[-800:]
        r["allocator_warnings"] = [str(x.message)[:300] for x in w if "llocator" in str(x.message) or "expandable" in str(x.message)]
    r["setter_calls"] = calls
    r["env_out"] = {k: os.environ.get(k) for k in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_HIP_ALLOC_CONF")}
    if r["gpu"]:
        x = torch.empty(64 << 20, dtype = torch.uint8, device = "cuda")
        segs = torch.cuda.memory._snapshot()["segments"]
        r["expandable"] = any(s.get("is_expandable") for s in segs)
        r["n_segments"] = len(segs)
except Exception as e:  # noqa: BLE001
    r["error"] = "".join(traceback.format_exception(type(e), e, e.__traceback__))[-1500:]
with open(out, "w", encoding = "utf-8") as f:
    json.dump(r, f)
