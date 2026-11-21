#!/usr/bin/env python3
"""One arm of the PR 2644 real-GPU cell: load unsloth_zoo.gradient_checkpointing from TREE (package __init__ skipped,
so only torch is needed), print what the detector answers for this device and count what the smart GC actually
offloads in a reentrant forward/backward and in the non-reentrant pack hook. JSON to --out. Observes only.

usage: sim_g12_2644_gpu_cell.py TREE --out OUT.json [--layers 6] [--seq 2048] [--hidden 2048] [--steps 3]
"""
import argparse
import hashlib
import json
import os
import sys
import time
import types


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tree")
    ap.add_argument("--out", required = True)
    ap.add_argument("--layers", type = int, default = 6)
    ap.add_argument("--seq", type = int, default = 2048)
    ap.add_argument("--hidden", type = int, default = 2048)
    ap.add_argument("--steps", type = int, default = 3)
    a = ap.parse_args()
    for k in ("UNSLOTH_GC_KEEP_ON_DEVICE", "UNSLOTH_DISABLE_DOUBLE_BUFFER"):
        os.environ.pop(k, None)
    obs = {"tree": a.tree}
    pkg = types.ModuleType("unsloth_zoo")
    pkg.__path__ = [os.path.join(os.path.abspath(a.tree), "unsloth_zoo")]
    sys.modules["unsloth_zoo"] = pkg
    import torch
    import unsloth_zoo.gradient_checkpointing as G
    import unsloth_zoo.integrated_device as D
    obs["gc_file"] = G.__file__
    obs["torch"] = torch.__version__
    obs["hip"] = getattr(torch.version, "hip", None)
    obs["DEVICE_TYPE"] = G.DEVICE_TYPE
    obs["cuda_available"] = torch.cuda.is_available()
    obs["device_count"] = torch.cuda.device_count() if obs["cuda_available"] else 0
    if not obs["cuda_available"]:
        json.dump(obs, open(a.out, "w"), indent = 2, default = str)
        return
    p = torch.cuda.get_device_properties(0)
    obs["props"] = {k: getattr(p, k, None) for k in ("name", "is_integrated", "integrated", "total_memory", "major",
                                                       "minor", "gcnArchName", "pci_bus_id", "multi_processor_count")}
    obs["mem_get_info"] = list(torch.cuda.mem_get_info(0))
    obs["any_integrated"] = bool(D._any_device_integrated())
    # Head-only API; absent at the base (the base has no keep decision at all: it always offloads).
    t0 = time.perf_counter()
    obs["head_api"] = hasattr(D, "unified_memory_device_indices")
    if obs["head_api"]:
        obs["is_integrated_part"] = D.device_is_integrated_part(0)
        obs["memory_pool"] = D.device_memory_pool(0)
        obs["unified_indices"] = sorted(D.unified_memory_device_indices())
        obs["keep_on_device_indices"] = sorted(G._keep_on_device_indices())
    obs["detector_ms"] = (time.perf_counter() - t0) * 1e3
    obs["double_buffer_disabled"] = bool(G._double_buffer_disabled())

    leases = []
    real = G._CpuSlotLease

    class _Counting(real):
        def __init__(self, *x, **k):
            super().__init__(*x, **k)
            leases.append(1)
    G._CpuSlotLease = _Counting
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    G.initialize_unsloth_gradient_checkpointing(dtype)
    obs["cpu_buffers_after_init"] = len(G.CPU_BUFFERS)
    dev = torch.device("cuda", 0)
    g = torch.Generator(device = "cpu").manual_seed(0)
    ws = [(torch.randn(a.hidden, a.hidden, generator = g) / a.hidden ** 0.5).to(dev, dtype).requires_grad_()
          for _ in range(a.layers)]
    x0 = torch.randn(1, a.seq, a.hidden, generator = g).to(dev, dtype)
    obs["layer_input_bytes"] = x0.numel() * x0.element_size()
    obs["MINIMUM_SIZE"] = G.MINIMUM_SIZE
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(0)
    times, lease_per_step = [], []
    for step in range(a.steps):
        for w in ws: w.grad = None
        x = x0.clone().requires_grad_()
        n0 = len(leases)
        torch.cuda.synchronize(); t = time.perf_counter()
        h = x * 1.0
        for w in ws:
            f = (lambda w: lambda h: h + torch.tanh(h @ w))(w)
            h = G.UnslothCheckpointFunction.apply(f, True, h)
        h.float().square().mean().backward()
        torch.cuda.synchronize(); times.append(time.perf_counter() - t)
        lease_per_step.append(len(leases) - n0)
        G.BACKWARD_PASS = True
        G.FIRST_PASS = False
    obs["reentrant_leases_per_step"] = lease_per_step
    obs["reentrant_step_s"] = times
    obs["peak_allocated"] = torch.cuda.max_memory_allocated(0)
    hsh = hashlib.sha256()
    for t in [w.grad for w in ws] + [x.grad]:
        hsh.update(t.detach().float().cpu().numpy().tobytes())
    obs["grad_sha256"] = hsh.hexdigest()
    obs["keep_budget"] = {str(k): v for k, v in getattr(G, "_KEEP_BUDGET", {}).items()}

    off = G.NonReentrantActivationOffloader()
    y = torch.randn(4096, 4096, device = dev, dtype = dtype, requires_grad = True)
    packed = off.pack(y)
    # Kept: the head's keep branch returns (None, x.detach()). Offloaded: (device, cpu tensor) or an
    # _OffloadedActivation handle (streamed D2H copy).
    obs["nonreentrant_packed_type"] = type(packed).__name__
    obs["nonreentrant_pack_device_is_none"] = isinstance(packed, tuple) and packed[0] is None
    obs["nonreentrant_packed_on"] = str(getattr(packed[1], "device", None)) if isinstance(packed, tuple) else type(packed).__name__
    torch.cuda.synchronize()
    obs["nonreentrant_stats"] = {k: v for k, v in off.stats.items() if v}
    json.dump(obs, open(a.out, "w"), indent = 2, default = str)


if __name__ == "__main__":
    main()
