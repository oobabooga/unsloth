#!/usr/bin/env python3
"""Probe: does a released ROCm llama.cpp bundle, packaged the way THIS checkout packages it,
survive GPU embedding requests? (unslothai/unsloth#13663, unslothai/llama.cpp#277)

The checkout is a state of unslothai/llama.cpp. When it carries
scripts/unsloth/relocate_kpack.py (the packaging step under test), the probe applies it to a
copy of the released bundle with the matching TheRock kernel packs; otherwise the bundle is used
as released. It then runs llama-server with Studio's embedder flags on the GPU and on the CPU
(devices hidden) and records per-request status, server liveness and the GPU-vs-CPU cosine.

Observes only; criteria/rocm_kpack_embed.py judges. Stdlib only, Linux and Windows.
Writes JSON to --out, never to stdout.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import socket
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

FAMILY = {"gfx110X": ("gfx110X-all", ["gfx1100", "gfx1101", "gfx1102", "gfx1103"]),
          "gfx1151": ("gfx1151", ["gfx1151"])}
WORD = "The quick brown fox jumps over the lazy dog near the riverbank. "
CASES = [
    ("short_1x20tok", ["hello world, a tiny note of 86 characters about nothing in particular at all."]),
    ("mid_1x73tok", [WORD * 5]),
    ("long_1x500tok", [WORD * 35]),
    ("long_64x500tok", [f"chunk {i}: " + WORD * 35 for i in range(64)]),
]


def fetch(url: str, dest: Path) -> Path:
    if not dest.exists():
        tmp = dest.with_suffix(dest.suffix + ".part")
        with urllib.request.urlopen(url, timeout = 600) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f, 1 << 20)
        tmp.replace(dest)
    return dest


def bundle_dir(cache: Path, osname: str, release: str, gfx: str) -> Path:
    ext = "zip" if osname == "windows" else "tar.gz"
    out = cache / f"bundle_{osname}_{gfx}"
    if (out / ".done").exists():
        return out
    arc = fetch(f"https://github.com/unslothai/llama.cpp/releases/download/{release}/"
                f"app-{release}-{osname}-x64-rocm-{gfx}.{ext}", cache / f"bundle_{osname}_{gfx}.{ext}")
    shutil.rmtree(out, ignore_errors = True)
    out.mkdir(parents = True)
    if ext == "zip":
        with zipfile.ZipFile(arc) as z:
            z.extractall(out)
    else:
        with tarfile.open(arc) as t:
            t.extractall(out, **({"filter": "data"} if hasattr(tarfile, "data_filter") else {}))
    (out / ".done").write_text("", encoding = "utf-8")
    return out


def kpack_dir(cache: Path, osname: str, therock: str, family: str) -> Path:
    out = cache / f"kpack_{osname}_{family}"
    if (out / ".done").exists():
        return out
    out.mkdir(parents = True, exist_ok = True)
    url = f"https://nightly.repo.amd.com/rocm/core/tarball/therock-dist-{osname}-{family}-{therock}.tar.gz"
    # Streamed: only the blas_lib packs are kept from a multi-GB dist.
    with urllib.request.urlopen(url, timeout = 1800) as r, tarfile.open(fileobj = r, mode = "r|gz") as t:
        for m in t:
            name = m.name.replace("\\", "/")
            if m.isfile() and "/.kpack/blas_lib_" in "/" + name:
                with t.extractfile(m) as src, open(out / Path(name).name, "wb") as dst:
                    shutil.copyfileobj(src, dst)
    (out / ".done").write_text("", encoding = "utf-8")
    return out


def kpack_refs(bundle: Path) -> dict:
    refs = {"old": 0, "new": 0}
    for p in bundle.iterdir():
        if p.is_file() and not p.is_symlink() and (p.suffix.lower() == ".dll" or ".so" in p.name):
            data = p.read_bytes()
            refs["old"] += data.count(b"../.kpack/blas_lib_@GFXARCH@")
            refs["new"] += data.count(b"rocblas/k/blas_lib_@GFXARCH@")
    return refs


def post(url: str, payload: dict, timeout: int = 600):
    req = urllib.request.Request(url, data = json.dumps(payload).encode(),
                                 headers = {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout = timeout) as r:
        return r.status, json.loads(r.read().decode())


def serve(bundle: Path, model: Path, log: Path, ngl: str, env_extra: dict) -> dict:
    exe = bundle / ("llama-server.exe" if os.name == "nt" else "llama-server")
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    cmd = [str(exe), "-m", str(model), "--host", "127.0.0.1", "--port", str(port), "--embedding",
           "--pooling", "mean", "--fit", "off", "-ub", "512", "-ngl", ngl,
           # Default verbosity omits the per-device buffer lines the device gate reads.
           "-lv", "4"]
    env = dict(os.environ, LLAMA_SET_ROWS = "1", **env_extra)
    if os.name != "nt":
        env["LD_LIBRARY_PATH"] = f"{bundle}:{env.get('LD_LIBRARY_PATH', '')}"
    res: dict = {"ngl": ngl, "env": env_extra, "cases": [], "ready": False, "vectors": None}
    with open(log, "w", encoding = "utf-8") as lf:
        proc = subprocess.Popen(cmd, stdout = lf, stderr = subprocess.STDOUT, env = env, cwd = str(bundle))
        base = f"http://127.0.0.1:{port}"
        try:
            for _ in range(600):
                if proc.poll() is not None:
                    break
                try:
                    with urllib.request.urlopen(base + "/health", timeout = 2) as r:
                        if r.status == 200:
                            res["ready"] = True
                            break
                except Exception:  # noqa: BLE001
                    pass
                time.sleep(0.5)
            for name, texts in CASES if res["ready"] else []:
                row: dict = {"case": name, "inputs": len(texts)}
                t0 = time.time()
                try:
                    status, body = post(base + "/v1/embeddings",
                                        {"input": texts, "model": "embedding", "encoding_format": "float"})
                    vecs = [d["embedding"] for d in sorted(body.get("data", []), key = lambda d: d.get("index", 0))]
                    row.update(status = status, vectors = len(vecs),
                               finite = all(math.isfinite(x) for v in vecs for x in v))
                    if name == "long_64x500tok":
                        res["vectors"] = vecs
                except urllib.error.HTTPError as e:
                    row.update(status = e.code, error = e.read().decode(errors = "replace")[:200])
                except Exception as e:  # noqa: BLE001
                    row.update(status = "EXC", error = repr(e)[:200])
                row["seconds"] = round(time.time() - t0, 2)
                time.sleep(0.5)
                row["server_alive_after"] = proc.poll() is None
                res["cases"].append(row)
                if not row["server_alive_after"]:
                    break
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout = 15)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
    res["exit_code"] = proc.returncode
    text = log.read_text(encoding = "utf-8", errors = "replace")
    listed = subprocess.run([str(exe), "--list-devices"], capture_output = True, text = True, env = env,
                            cwd = str(bundle), timeout = 120)
    res["rocm_devices"] = len(re.findall(r"^\s*ROCm\d+:", listed.stdout + listed.stderr, re.M))
    res["rocm_model_buffer"] = bool(re.search(r"ROCm\d+ model buffer size", text))
    res["kernel_load_error"] = "hipErrorInvalidKernelFile" in text
    res["log_tail"] = text.splitlines()[-12:]
    return res


def cosine_min(a, b):
    if not a or not b or len(a) != len(b):
        return None
    out = []
    for x, y in zip(a, b):
        dot = sum(i * j for i, j in zip(x, y))
        nx, ny = math.sqrt(sum(i * i for i in x)), math.sqrt(sum(j * j for j in y))
        out.append(dot / (nx * ny) if nx and ny else 0.0)
    return min(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True, type = Path)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--release", default = "b11541-mix-2500203")
    ap.add_argument("--therock", default = "10.2.0a20261003")
    ap.add_argument("--bundles", default = "gfx1151", help = "gfx[@HSA_OVERRIDE_GFX_VERSION],...")
    ap.add_argument("--model-url", default = "https://huggingface.co/unsloth/embeddinggemma-2-GGUF/resolve/main/embeddinggemma-2-F16.gguf")
    ap.add_argument("--cache", type = Path, default = None)
    args = ap.parse_args()

    osname = "windows" if os.name == "nt" else "linux"
    # Bundles and downloads stay out of the uploaded out-dir; only logs and the JSON land there.
    root = Path(os.environ.get("AMD_CI_WORK") or args.out.parent)
    work = root / "kpack_work" / args.state
    cache = args.cache or root / "kpack_cache"
    cache.mkdir(parents = True, exist_ok = True)
    shutil.rmtree(work, ignore_errors = True)
    work.mkdir(parents = True)
    obs: dict = {"state": args.state, "os": osname, "release": args.release, "therock": args.therock,
                 "configs": {}}
    script = args.checkout / "scripts" / "unsloth" / "relocate_kpack.py"
    obs["packaging_step_present"] = script.is_file()
    model = fetch(args.model_url, cache / "model.gguf")

    for spec in [x for x in args.bundles.split(",") if x]:
        gfx, _, override = spec.partition("@")
        family, archs = FAMILY[gfx]
        name = gfx + (f"_as_{override}" if override else "")
        cfg: dict = {"gfx": gfx, "hsa_override": override or None}
        bundle = work / name
        shutil.copytree(bundle_dir(cache, osname, args.release, gfx), bundle, symlinks = True)
        if script.is_file():
            kp = kpack_dir(cache, osname, args.therock, family)
            r = subprocess.run([sys.executable, str(script), "--bundle", str(bundle), "--kpack-src", str(kp),
                                "--archs", ",".join(archs)], capture_output = True, text = True)
            cfg["packaging"] = {"rc": r.returncode, "out": (r.stdout + r.stderr)[-600:]}
        cfg["kpack_refs"] = kpack_refs(bundle)
        cfg["shipped_kpacks"] = sorted(p.name for p in (bundle / "rocblas" / "k").glob("*.kpack")) \
            if (bundle / "rocblas" / "k").is_dir() else []
        gpu_env = {"HSA_OVERRIDE_GFX_VERSION": override} if override else {}
        logs = args.out.parent
        gpu = serve(bundle, model, logs / f"{args.state}_{name}_gpu.log", "-1", gpu_env)
        cpu = serve(bundle, model, logs / f"{args.state}_{name}_cpu.log", "0",
                    {"CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": "-1"})
        cfg["gpu_vs_cpu_min_cos"] = cosine_min(gpu.pop("vectors"), cpu.pop("vectors"))
        cfg["gpu"], cfg["cpu"] = gpu, cpu
        obs["configs"][name] = cfg

    args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
    shutil.rmtree(work, ignore_errors = True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
