#!/usr/bin/env python3
"""M8 (b) micro: does uploading tensors mapped through Studio's native-cache path (``diffusion_native_cache.
_map_tensor``: mmap ACCESS_COPY = MAP_PRIVATE writable) turn the mapped pages into private anonymous host memory on
this ROCm stack (ComfyUI #17751's ROCm 7.x report: an 8 MiB upload made 8,392,704 B of private anonymous memory)?
Observes only.

Writes a bf16 safetensors file of --mib MiB in --tensor-mib MiB tensors (in --dir, page cache dropped), maps every
tensor through the tree's ``_shard_entries`` + ``_map_tensor`` (arm copy), or the same with ACCESS_READ (arm read: a
patched copy of the function, the only change), then ``.to("cuda", non_blocking = --nb)`` each tensor and syncs.
Reads /proc/self/smaps_rollup (Anonymous, Private_Dirty, Rss) and /proc/self/status (RssAnon) after mapping, after
the upload with the host tensors alive, and after dropping them. Checks the device copy equals the file bytes.

  run_mmap_cow.py --backend B --dir SCRATCH --arm copy|read --out F [--mib 64] [--tensor-mib 8] [--nb 0|1]
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import types
from pathlib import Path


def mem() -> dict:
    out = {}
    try:
        for ln in open("/proc/self/smaps_rollup", encoding = "utf-8"):
            k, _, v = ln.partition(":")
            if k in ("Rss", "Anonymous", "Private_Dirty", "Private_Clean", "Shared_Clean"):
                out[k] = int(v.split()[0]) * 1024
        for ln in open("/proc/self/status", encoding = "utf-8"):
            if ln.startswith(("RssAnon", "RssFile", "VmHWM")):
                k, _, v = ln.partition(":")
                out[k] = int(v.split()[0]) * 1024
    except Exception as exc:  # noqa: BLE001
        out["error"] = repr(exc)[:200]
    return out


def write_file(path: Path, mib: int, tensor_mib: int) -> None:
    import torch
    from safetensors.torch import save_file
    n = max(1, mib // tensor_mib)
    numel = tensor_mib * 2 ** 20 // 2
    g = torch.Generator().manual_seed(3)
    save_file({f"blk.{i}.weight": torch.randn(numel, generator = g).to(torch.bfloat16).reshape(-1, 1024)
               for i in range(n)}, str(path))
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required = True)
    ap.add_argument("--dir", required = True)
    ap.add_argument("--arm", required = True, choices = ("copy", "read"))
    ap.add_argument("--out", required = True)
    ap.add_argument("--mib", type = int, default = 64)
    ap.add_argument("--tensor-mib", type = int, default = 8)
    ap.add_argument("--nb", type = int, default = 0)
    a = ap.parse_args()
    sys.path.insert(0, a.backend)
    obs = {"arm": a.arm, "mib": a.mib, "tensor_mib": a.tensor_mib, "non_blocking": bool(a.nb)}
    try:
        import mmap

        import torch

        from core.inference import diffusion_native_cache as NC
        obs["torch"], obs["hip"] = torch.__version__, torch.version.hip
        path = Path(a.dir) / f"cow_{a.mib}_{a.tensor_mib}.safetensors"
        if not path.is_file():
            write_file(path, a.mib, a.tensor_mib)
        mapper = NC._map_tensor
        if a.arm == "read":
            import inspect
            src = inspect.getsource(NC._map_tensor)
            if src.count("access = mmap.ACCESS_COPY") != 1:
                raise RuntimeError("tree's _map_tensor no longer maps ACCESS_COPY once: read arm not constructible")
            ns = dict(NC.__dict__)
            exec(compile(src.replace("access = mmap.ACCESS_COPY", "access = mmap.ACCESS_READ"), NC.__file__, "exec"),
                 ns)
            mapper = ns["_map_tensor"]
        obs["map_src_copy"] = "ACCESS_COPY" in __import__("inspect").getsource(NC._map_tensor)
        torch.cuda.init()
        torch.empty(1, device = "cuda")
        torch.cuda.synchronize()
        gc.collect()
        obs["m0"] = mem()
        start, header = NC._shard_entries(str(path))
        fd = os.open(path, os.O_RDONLY)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # frombuffer on a read-only map warns: nothing writes it
            host = [mapper(fd, start, e) for _, e in sorted(header.items(), key = lambda kv: kv[1]["data_offsets"][0])]
        os.close(fd)
        obs["mapped_bytes"] = sum(t.numel() * t.element_size() for t in host)
        obs["m1_mapped"] = mem()
        dev = [t.to("cuda", non_blocking = bool(a.nb)) for t in host]
        torch.cuda.synchronize()
        obs["m2_uploaded"] = mem()
        obs["equal"] = all(torch.equal(h, d.cpu()) for h, d in zip(host[:2], dev[:2]))
        del host
        gc.collect()
        obs["m3_host_dropped"] = mem()
        m0, m2 = obs["m0"], obs["m2_uploaded"]
        for k in ("Anonymous", "Private_Dirty", "RssAnon"):
            if k in m0 and k in m2:
                obs[f"delta_{k}"] = m2[k] - m0[k]
                obs[f"frac_{k}"] = round((m2[k] - m0[k]) / max(1, obs["mapped_bytes"]), 4)
        obs["ok"] = True
        del dev
        _ = mmap
    except Exception as exc:  # noqa: BLE001
        import traceback
        obs["error"] = repr(exc)[:400]
        obs["tb"] = traceback.format_exc()[-1500:]
    Path(a.out).write_text(json.dumps(obs), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
_ = types
