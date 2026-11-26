#!/usr/bin/env python3
"""A7 item 3: the native per-tensor build's shard mapping (diffusion_native_cache._map_tensor), ACCESS_COPY (main) vs
ACCESS_READ (head tree knob UNSLOTH_NATIVE_CACHE_MMAP=read), optionally with MIMALLOC_PURGE_DELAY=0 (set by caller).

  mmap_bench.py --backend HEAD_BACKEND --repo R --component transformer --mode micro|build --out F

micro: page cache dropped (fsync + POSIX_FADV_DONTNEED), then every tensor of the component's shards is mapped through
  Studio's _map_tensor and copied to the GPU one at a time (synchronised): cold pass, then a warm pass. GiB/s of the
  copies, RssAnon / RssFile peaks (sampled every 20 ms), and a checksum of the device bytes.
build: page cache dropped, then Studio's build_from_shards (native int8 W8A8, cast_floats) under
  quantise_host_linears_on(cuda), then the module moved to the GPU: build s, RssAnon after the build, bytes still
  host-resident (these are the mapped views), .to(cuda) s and GiB/s, checksum of the state dict.
Both report whether a mimalloc is mapped in the process (MIMALLOC_PURGE_DELAY does nothing without one)."""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import traceback
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a7common import write  # noqa: E402


def status_gib(keys = ("RssAnon", "RssFile", "VmHWM", "VmRSS")):
    out = {}
    try:
        for line in open("/proc/self/status", encoding = "utf-8"):
            k = line.split(":")[0]
            if k in keys:
                out[k] = round(int(line.split()[1]) / 2 ** 20, 3)
    except Exception:  # noqa: BLE001
        pass
    return out


class Sampler:
    def __init__(self):
        self.peak = {}
        self._stop = threading.Event()
        self._t = threading.Thread(target = self._run, daemon = True)
        self._t.start()

    def _run(self):
        while not self._stop.is_set():
            for k, v in status_gib(("RssAnon", "RssFile")).items():
                self.peak[k] = max(self.peak.get(k, 0.0), v)
            time.sleep(0.02)

    def take(self):
        p = dict(self.peak)
        self.peak = {}
        return p

    def stop(self):
        self._stop.set()


def mimalloc_mapped():
    try:
        return any("mimalloc" in ln for ln in open("/proc/self/maps", encoding = "utf-8"))
    except Exception:  # noqa: BLE001
        return None


def drop(files):
    for f in files:
        fd = os.open(f, os.O_RDONLY)
        try:
            os.fsync(fd)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required = True)
    ap.add_argument("--repo", required = True)
    ap.add_argument("--component", default = "transformer")
    ap.add_argument("--family", default = None)
    ap.add_argument("--mode", required = True, choices = ("micro", "build"))
    ap.add_argument("--max-gb", type = float, default = 0, help = "micro: stop after this many GB of shards")
    ap.add_argument("--out", required = True)
    a = ap.parse_args()
    obs: dict = {"mode": a.mode, "repo": a.repo, "access_env": os.environ.get("UNSLOTH_NATIVE_CACHE_MMAP"),
                 "mimalloc_purge_delay": os.environ.get("MIMALLOC_PURGE_DELAY")}
    t0 = time.time()
    sampler = None
    try:
        sys.path.insert(0, a.backend)
        import torch
        from core.inference import diffusion_native_cache as NC
        import inspect
        obs["knob_in_tree"] = "UNSLOTH_NATIVE_CACHE_MMAP" in inspect.getsource(NC._map_tensor)
        hub = os.path.join(os.environ["HF_HOME"], "hub")
        shards = NC._dense_shards(a.repo, a.component, hub)
        obs["shards"] = [os.path.basename(s) for s in shards or []]
        if not shards:
            raise RuntimeError("no local shards")
        obs["shard_gb"] = round(sum(os.path.getsize(s) for s in shards) / 1e9, 2)
        torch.cuda.init()
        torch.empty(1, device = "cuda")
        obs["mimalloc_mapped"] = mimalloc_mapped()
        obs["rss_start"] = status_gib()
        sampler = Sampler()
        if a.mode == "micro":
            files = []
            tot = 0
            for s in shards:
                files.append(s)
                tot += os.path.getsize(s)
                if a.max_gb and tot / 1e9 >= a.max_gb:
                    break
            obs["micro_files"] = len(files)
            for pas in ("cold", "warm"):
                if pas == "cold":
                    drop(files)
                sampler.take()
                acc = torch.zeros((), dtype = torch.int64, device = "cuda")
                nbytes, copy_s = 0, 0.0
                tw = time.perf_counter()
                for s in files:
                    found = NC._shard_entries(s, {**NC._STREAM_DTYPES, **NC._CAST_DTYPES})
                    if found is None:
                        raise RuntimeError(f"unreadable shard header {s}")
                    start, header = found
                    fd = os.open(s, os.O_RDONLY)
                    try:
                        for key, entry in sorted(header.items(), key = lambda kv: int(kv[1]["data_offsets"][0])):
                            t = NC._map_tensor(fd, start, entry)
                            if t.numel() == 0:
                                continue
                            torch.cuda.synchronize()
                            c0 = time.perf_counter()
                            g = t.to("cuda")
                            torch.cuda.synchronize()
                            copy_s += time.perf_counter() - c0
                            nbytes += t.numel() * t.element_size()
                            acc += g.reshape(-1).view(torch.uint8)[:: 4099].to(torch.int64).sum()
                            del t, g
                    finally:
                        os.close(fd)
                wall = time.perf_counter() - tw
                obs[pas] = {"gib": round(nbytes / 2 ** 30, 3), "copy_s": round(copy_s, 3), "wall_s": round(wall, 3),
                            "copy_gibps": round(nbytes / 2 ** 30 / max(copy_s, 1e-9), 3),
                            "wall_gibps": round(nbytes / 2 ** 30 / max(wall, 1e-9), 3), "peak": sampler.take(),
                            "after": status_gib(), "checksum": int(acc)}
                write(a.out, obs)
        else:
            import diffusers
            from core.inference.diffusion_native_cache import build_from_shards, native_spec
            from core.inference.diffusion_native_quant import NATIVE_INT8, quantise_host_linears_on
            cfg = json.loads(Path(NC._cache._local_file(a.repo, f"{a.component}/config.json", hub)).read_text(
                encoding = "utf-8"))
            cls = getattr(diffusers, cfg["_class_name"])
            spec = native_spec(NATIVE_INT8, a.family, act_int8 = True)
            obs["spec"] = spec
            logs = []

            class L:
                def __getattr__(self, name):
                    return lambda msg, *x, **k: logs.append(f"{name}: {msg % x if x else msg}"[:400])
            drop(shards)
            sampler.take()
            tb = time.perf_counter()
            with quantise_host_linears_on(torch.device("cuda", 0)):
                model = build_from_shards(cls, a.repo, spec, family = a.family, dtype = torch.bfloat16,
                                          component = a.component,
                                          config_kwargs = {"cache_dir": hub, "local_files_only": True},
                                          logger = L(), cast_floats = True, reraise_oom = True)
            obs["build_s"] = round(time.perf_counter() - tb, 2)
            obs["build_peak"] = sampler.take()
            obs["after_build"] = status_gib()
            obs["logs"] = logs[-20:]
            if model is None:
                raise RuntimeError("build_from_shards returned None")
            host = [p for p in list(model.parameters()) + list(model.buffers()) if p.device.type == "cpu"]
            obs["host_tensors"] = len(host)
            obs["host_gib"] = round(sum(p.numel() * p.element_size() for p in host) / 2 ** 30, 3)
            torch.cuda.synchronize()
            tm = time.perf_counter()
            model.to("cuda")
            torch.cuda.synchronize()
            obs["to_cuda_s"] = round(time.perf_counter() - tm, 3)
            obs["to_cuda_gibps"] = round(obs["host_gib"] / max(obs["to_cuda_s"], 1e-9), 3)
            obs["after_to_cuda"] = status_gib()
            acc = torch.zeros((), dtype = torch.float64, device = "cuda")
            for i, (name, t) in enumerate(model.state_dict().items()):
                if t.numel():
                    acc += t.detach().reshape(-1)[:: 1013].to(torch.float64).abs().sum() * (1 + (i % 97) / 97.0)
            obs["checksum"] = float(acc)
            obs["n_state"] = len(model.state_dict())
    except Exception:  # noqa: BLE001
        obs["error"] = traceback.format_exc()[-2000:]
    if sampler is not None:
        sampler.stop()
    obs["seconds"] = round(time.time() - t0, 1)
    write(a.out, obs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
