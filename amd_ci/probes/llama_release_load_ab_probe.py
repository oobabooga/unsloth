#!/usr/bin/env python3
"""Probe: load + serve a small GGUF with two released unslothai/llama.cpp ROCm bundles
(unslothai/unsloth#18563: b12540-mix-c4017f9 crashes / crawls on multi-GPU load on Windows).

The checkout is ignored (the PR state only picks which release is "code under test"):
  state base -> code = --old release, state head -> code = --new release.
For the state's release it runs two arms:
  asis    the bundle exactly as released
  rtswap  the same bundle with the HIP runtime (amdhip64_7.dll / libamdhip64.so*) and
          amd_comgr from the OTHER release copied over it (separates code from runtime)
and per arm these load configs (Studio-like server flags):
  gpu_none   -ngl 99 --load-mode none
  gpu_mmap   -ngl 99 --load-mode mmap
  split_cpu  -ngl 14 --load-mode none          (model split GPU + CPU)
  split_rpc  -ngl 99 --rpc <local rpc-server on the same GPU> -ts 1,1 --load-mode none
             (model split across two ggml devices: ROCm0 and RPC0, both the one GPU)
Each config: N repeats of (launch, wait /health, one 128-token completion, stop).
Observes only; criteria/llama_release_load_ab.py judges. Stdlib only. JSON to --out.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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

IS_WIN = os.name == "nt"
OSNAME = "windows" if IS_WIN else "linux"
PROMPT = ("Write a short paragraph about the history of the printing press, "
          "its inventor, and how it changed the spread of knowledge in Europe.")
RT_NAMES_WIN = ["amdhip64_7.dll", "amd_comgr.dll"]
RT_PAT_LINUX = [r"^libamdhip64\.so", r"^libamd_comgr\.so"]


def fetch(url: str, dest: Path) -> Path:
    if not dest.exists():
        tmp = dest.with_suffix(dest.suffix + ".part")
        for attempt in range(4):
            try:
                with urllib.request.urlopen(url, timeout = 900) as r, open(tmp, "wb") as f:
                    shutil.copyfileobj(r, f, 1 << 20)
                break
            except Exception:  # noqa: BLE001
                if attempt == 3:
                    raise
                time.sleep(10)
        tmp.replace(dest)
    return dest


def bundle_dir(cache: Path, release: str, gfx: str) -> Path:
    ext = "zip" if IS_WIN else "tar.gz"
    out = cache / f"bundle_{release}_{gfx}"
    if (out / ".done").exists():
        return out
    arc = fetch(f"https://github.com/unslothai/llama.cpp/releases/download/{release}/"
                f"app-{release}-{OSNAME}-x64-rocm-{gfx}.{ext}", cache / f"bundle_{release}_{gfx}.{ext}")
    shutil.rmtree(out, ignore_errors = True)
    out.mkdir(parents = True)
    if ext == "zip":
        with zipfile.ZipFile(arc) as z:
            z.extractall(out)
    else:
        with tarfile.open(arc) as t:
            t.extractall(out, **({"filter": "data"} if hasattr(tarfile, "data_filter") else {}))
    # tarballs may nest one directory
    exe = "llama-server.exe" if IS_WIN else "llama-server"
    if not (out / exe).exists():
        hits = list(out.rglob(exe))
        if hits:
            (out / ".root").write_text(str(hits[0].parent), encoding = "utf-8")
    (out / ".done").write_text("", encoding = "utf-8")
    return out


def bin_root(b: Path) -> Path:
    r = b / ".root"
    return Path(r.read_text(encoding = "utf-8")) if r.exists() else b


def runtime_files(root: Path) -> list[Path]:
    if IS_WIN:
        return [root / n for n in RT_NAMES_WIN if (root / n).exists()]
    out = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and any(re.match(pat, p.name) for pat in RT_PAT_LINUX):
            out.append(p)
    return out


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def post(url: str, payload: dict, timeout: int = 600):
    req = urllib.request.Request(url, data = json.dumps(payload).encode(),
                                 headers = {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout = timeout) as r:
        return r.status, json.loads(r.read().decode())


def stop(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout = 20)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def env_for(root: Path) -> dict:
    env = dict(os.environ)
    if not IS_WIN:
        env["LD_LIBRARY_PATH"] = f"{root}:{root / 'lib'}:{env.get('LD_LIBRARY_PATH', '')}"
    return env


def run_once(root: Path, model: Path, cfg: str, log: Path, health_timeout: int) -> dict:
    exe = root / ("llama-server.exe" if IS_WIN else "llama-server")
    port = free_port()
    args = ["-m", str(model), "--host", "127.0.0.1", "--port", str(port), "-c", "8192",
            "--parallel", "4", "--kv-unified", "--flash-attn", "on", "--fit", "off",
            "--cache-type-k", "q4_0", "--cache-type-v", "q4_0", "--no-warmup", "-lv", "4"]
    rpc_proc = None
    rpc_log = None
    if cfg == "gpu_none":
        args += ["-ngl", "99", "--load-mode", "none"]
    elif cfg == "gpu_mmap":
        args += ["-ngl", "99", "--load-mode", "mmap"]
    elif cfg == "split_cpu":
        args += ["-ngl", "14", "--load-mode", "none"]
    elif cfg == "split_rpc":
        rexe = root / ("ggml-rpc-server.exe" if IS_WIN else "ggml-rpc-server")
        if not rexe.exists():
            rexe = root / ("rpc-server.exe" if IS_WIN else "rpc-server")
        rport = free_port()
        rpc_log = log.with_name(log.stem + "_rpc.log")
        rf = open(rpc_log, "w", encoding = "utf-8")
        rpc_proc = subprocess.Popen([str(rexe), "-H", "127.0.0.1", "-p", str(rport)], stdout = rf,
                                    stderr = subprocess.STDOUT, env = env_for(root), cwd = str(root))
        time.sleep(4)
        args += ["-ngl", "99", "--load-mode", "none", "--rpc", f"127.0.0.1:{rport}", "-ts", "1,1"]
    res: dict = {"cfg": cfg, "ready": False}
    t0 = time.time()
    with open(log, "w", encoding = "utf-8") as lf:
        proc = subprocess.Popen([str(exe), *args], stdout = lf, stderr = subprocess.STDOUT,
                                env = env_for(root), cwd = str(root))
        base = f"http://127.0.0.1:{port}"
        try:
            while time.time() - t0 < health_timeout:
                if proc.poll() is not None:
                    break
                try:
                    with urllib.request.urlopen(base + "/health", timeout = 2) as r:
                        if r.status == 200:
                            res["ready"] = True
                            break
                except Exception:  # noqa: BLE001
                    pass
                time.sleep(0.25)
            res["ready_s"] = round(time.time() - t0, 2)
            if res["ready"]:
                try:
                    status, body = post(base + "/completion", {"prompt": PROMPT, "n_predict": 128,
                                                               "temperature": 0, "seed": 1,
                                                               "cache_prompt": False})
                    tm = body.get("timings", {})
                    res.update(status = status, pp_tps = tm.get("prompt_per_second"),
                               tg_tps = tm.get("predicted_per_second"),
                               n_pred = tm.get("predicted_n"),
                               text_sha = hashlib.sha256(body.get("content", "").encode()).hexdigest()[:12])
                except urllib.error.HTTPError as e:
                    res.update(status = e.code, error = e.read().decode(errors = "replace")[:300])
                except Exception as e:  # noqa: BLE001
                    res.update(status = "EXC", error = repr(e)[:300])
                time.sleep(0.5)
                res["alive_after"] = proc.poll() is None
        finally:
            stop(proc)
            if rpc_proc is not None:
                stop(rpc_proc)
    res["exit_code"] = proc.returncode
    text = log.read_text(encoding = "utf-8", errors = "replace")
    res["rocm_error"] = [ln.strip()[:200] for ln in text.splitlines() if "ROCm error" in ln or "HIP error" in ln][:4]
    res["buffers"] = sorted(set(re.findall(r"(\S+) model buffer size =\s*([\d.]+) MiB", text)))
    res["devices_in_log"] = sorted(set(re.findall(r"using device (\S+)", text)))
    # load_tensors timing (wall seconds the loader took) from the -lv 4 timestamps
    m_start = re.search(r"^(\d+)\.(\d+)\.(\d+)\.(\d+) I load_tensors: loading model tensors", text, re.M)
    m_end = None
    for m in re.finditer(r"^(\d+)\.(\d+)\.(\d+)\.(\d+) I (llama_context|llama_init_from_model|common_init_from_params)", text, re.M):
        m_end = m
        break

    def ts(m):
        h, mi, s, ms = (int(x) for x in m.groups()[:4])
        return h * 3600 + mi * 60 + s + ms / 1000.0
    if m_start and m_end:
        res["load_tensors_s"] = round(ts(m_end) - ts(m_start), 3)
    if rpc_log is not None:
        res["rpc_log_tail"] = rpc_log.read_text(encoding = "utf-8", errors = "replace").splitlines()[-6:]
    if not res["ready"] or res.get("status") != 200:
        res["log_tail"] = text.splitlines()[-25:]
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True, type = Path)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--old", default = "b12347-mix-241a0f3")
    ap.add_argument("--new", default = "b12540-mix-c4017f9")
    ap.add_argument("--gfx", default = "gfx1151")
    ap.add_argument("--model-url", default = "https://huggingface.co/unsloth/Qwen3-1.7B-GGUF/resolve/main/Qwen3-1.7B-Q8_0.gguf")
    ap.add_argument("--repeats", type = int, default = 3)
    ap.add_argument("--configs", default = "gpu_none,gpu_mmap,split_cpu,split_rpc")
    ap.add_argument("--health-timeout", type = int, default = 300)
    args = ap.parse_args()

    code_rel = args.old if args.state == "base" else args.new
    other_rel = args.new if args.state == "base" else args.old
    root_dir = Path(os.environ.get("AMD_CI_WORK") or args.out.parent)
    cache = root_dir / "rel_cache"
    work = root_dir / "rel_work" / args.state
    cache.mkdir(parents = True, exist_ok = True)
    shutil.rmtree(work, ignore_errors = True)
    work.mkdir(parents = True)
    logs = args.out.parent
    obs: dict = {"state": args.state, "os": OSNAME, "code_release": code_rel, "other_release": other_rel,
                 "gfx": args.gfx, "arms": {}}
    t = time.time()
    model = fetch(args.model_url, cache / "model.gguf")
    obs["model_bytes"] = model.stat().st_size
    obs["model_download_s"] = round(time.time() - t, 1)

    code_b = bin_root(bundle_dir(cache, code_rel, args.gfx))
    other_b = bin_root(bundle_dir(cache, other_rel, args.gfx))
    for arm in ("asis", "rtswap"):
        dst = work / arm
        shutil.copytree(code_b, dst, symlinks = True)
        swapped = []
        if arm == "rtswap":
            for src in runtime_files(other_b):
                rel = src.relative_to(other_b)
                tgt = dst / rel
                tgt.parent.mkdir(parents = True, exist_ok = True)
                shutil.copy2(src, tgt)
                swapped.append(str(rel))
        a: dict = {"swapped": swapped, "runtime_sha": {str(p.relative_to(dst)): sha(p) for p in runtime_files(dst)},
                   "runs": {}}
        exe = dst / ("llama-server.exe" if IS_WIN else "llama-server")
        try:
            ld = subprocess.run([str(exe), "--list-devices"], capture_output = True, text = True,
                                env = env_for(dst), cwd = str(dst), timeout = 120)
            a["list_devices"] = (ld.stdout + ld.stderr).strip().splitlines()[-6:]
        except Exception as e:  # noqa: BLE001
            a["list_devices"] = [repr(e)]
        for cfg in [c for c in args.configs.split(",") if c]:
            rows = []
            for i in range(args.repeats):
                rows.append(run_once(dst, model, cfg, logs / f"{args.state}_{arm}_{cfg}_{i}.log", args.health_timeout))
                if not rows[-1]["ready"]:
                    break  # a crash / hang is the finding; do not hammer the driver
            a["runs"][cfg] = rows
        obs["arms"][arm] = a
        args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
    args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
    shutil.rmtree(work, ignore_errors = True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
