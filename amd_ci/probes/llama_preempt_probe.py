#!/usr/bin/env python3
"""Probe: build llama-server for a state and observe the KV-pool overflow on this GPU.

Observes only. The state's checkout is the Unsloth tree the toolkit resolved; the
llama.cpp tree it stands for is mapped by state name (--map state=url@sha). Builds
are cached by sha under --root so a state that maps to the same commit reuses them.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import threading
import time
import urllib.request
from pathlib import Path


def sh(cmd, cwd = None, timeout = 3600, env = None):
    p = subprocess.run(cmd, cwd = cwd, capture_output = True, text = True, timeout = timeout, env = env)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def tools() -> dict:
    found = {}
    for name in ("cmake", "ninja", "hipcc", "glslc", "gcc", "g++", "clang", "git"):
        found[name] = shutil.which(name) or ""
    for cand in ("/opt/rocm/bin/hipcc", "/opt/rocm/llvm/bin/clang"):
        if os.path.exists(cand):
            found[cand] = cand
    found["rocm_dirs"] = sorted(str(p) for p in Path("/opt").glob("rocm*"))
    found["vulkan_icds"] = sorted(p.name for p in Path("/usr/share/vulkan/icd.d").glob("*.json")) \
        if Path("/usr/share/vulkan/icd.d").is_dir() else []
    return found


def pick_backend(t: dict, want: str) -> list[str] | None:
    hip = t.get("/opt/rocm/bin/hipcc") or t.get("hipcc")
    if want in ("auto", "hip") and hip:
        clang = "/opt/rocm/llvm/bin/clang" if os.path.exists("/opt/rocm/llvm/bin/clang") else ""
        args = ["-DGGML_HIP=ON", "-DGPU_TARGETS=gfx1151", "-DAMDGPU_TARGETS=gfx1151"]
        if clang:
            args += [f"-DCMAKE_C_COMPILER={clang}", f"-DCMAKE_CXX_COMPILER={clang}++",
                     f"-DCMAKE_HIP_COMPILER={clang}++"]
        return ["hip", *args]
    if want in ("auto", "vulkan") and t.get("glslc"):
        return ["vulkan", "-DGGML_VULKAN=ON"]
    return None


def build(url: str, sha: str, root: Path, backend_args: list[str], obs: dict) -> Path | None:
    src = root / f"src_{sha[:12]}"
    bld = src / "build"
    exe = bld / "bin" / "llama-server"
    if exe.exists():
        obs["build_cached"] = True
        return exe
    if not src.exists():
        src.mkdir(parents = True)
        for cmd in (["git", "init", "-q"], ["git", "fetch", "-q", "--depth", "1", url, sha],
                    ["git", "checkout", "-q", "FETCH_HEAD"]):
            for attempt in range(4):
                rc, out = sh(cmd, cwd = src, timeout = 1800)
                if not rc:
                    break
                time.sleep(15 * (attempt + 1))
            if rc:
                obs["clone_error"] = out[-2000:]
                return None
    rc, head = sh(["git", "rev-parse", "HEAD"], cwd = src)
    obs["llama_sha"] = head.strip()
    t0 = time.time()
    rc, out = sh(["cmake", "-S", ".", "-B", "build", "-DCMAKE_BUILD_TYPE=Release", "-DLLAMA_CURL=OFF",
                  "-DLLAMA_BUILD_TESTS=OFF", *backend_args[1:]], cwd = src, timeout = 1800)
    obs["configure_tail"] = out[-1500:]
    if rc:
        obs["configure_error"] = True
        return None
    rc, out = sh(["cmake", "--build", "build", "-j", str(os.cpu_count() or 8), "--target", "llama-server"],
                 cwd = src, timeout = 5400)
    obs["build_seconds"] = round(time.time() - t0, 1)
    obs["build_tail"] = out[-2500:]
    if rc or not exe.exists():
        obs["build_error"] = True
        return None
    return exe


def model_path(root: Path, obs: dict) -> Path | None:
    dst = root / "models" / "Qwen3-0.6B-Q8_0.gguf"
    if dst.exists():
        return dst
    dst.parent.mkdir(parents = True, exist_ok = True)
    url = "https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf"
    try:
        tmp = dst.with_suffix(".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(dst)
        return dst
    except Exception as e:  # noqa: BLE001
        obs["model_error"] = f"{type(e).__name__}: {e}"
        return None


def post(port: int, path: str, body: dict, timeout = 900) -> tuple[int, dict]:
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data = json.dumps(body).encode(),
                                 headers = {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout = timeout) as r:
            return r.status, json.loads(r.read().decode() or "{}")
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode() or "{}")
        except Exception:  # noqa: BLE001
            return e.code, {}
    except Exception as e:  # noqa: BLE001
        return -1, {"error": f"{type(e).__name__}: {e}"}


def get(port: int, path: str) -> dict:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout = 30) as r:
            return json.loads(r.read().decode() or "{}")
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}


class Server:
    def __init__(self, exe: Path, model: Path, port: int, args: list[str], log: Path):
        self.port = port
        self.log = log
        env = dict(os.environ, LD_LIBRARY_PATH = str(exe.parent) + ":" + os.environ.get("LD_LIBRARY_PATH", ""))
        self.fh = open(log, "wb")
        self.proc = subprocess.Popen([str(exe), "-m", str(model), "--port", str(port), "-ngl", "99", *args],
                                     stdout = self.fh, stderr = subprocess.STDOUT, env = env,
                                     start_new_session = True)

    def ready(self, timeout = 300) -> bool:
        t = time.time() + timeout
        while time.time() < t:
            if self.proc.poll() is not None:
                return False
            if get(self.port, "/health").get("status") == "ok":
                return True
            time.sleep(1)
        return False

    def stop(self):
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout = 30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.fh.close()

    def text(self) -> str:
        return self.log.read_text(encoding = "utf-8", errors = "replace")


PROMPT = "The history of computing, told one numbered fact at a time:\n" + "".join(
    f"{i}. Fact number {i} about machines, memory and people who built them.\n" for i in range(1, 400))


def one(port, n_predict, prompt = PROMPT[:2000]):
    t0 = time.time()
    st, j = post(port, "/completion", {"prompt": prompt, "n_predict": n_predict, "temperature": 0,
                                       "cache_prompt": False, "seed": 1})
    return {"status": st, "content": j.get("content"), "timings": j.get("timings"),
            "preempt": j.get("preempt"), "error": j.get("error"), "wall": round(time.time() - t0, 3)}


def oversubscribed(port, n = 4, n_predict = 512):
    res = [None] * n
    def go(i):
        prompt = f"Conversation {i}.\n" + PROMPT[: 6000]
        res[i] = one(port, n_predict, prompt)
    th = [threading.Thread(target = go, args = (i,)) for i in range(n)]
    t0 = time.time()
    for t in th:
        t.start()
    for t in th:
        t.join()
    return res, round(time.time() - t0, 3)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--map", action = "append", default = [], help = "state=url@sha")
    ap.add_argument("--root", default = os.environ.get("AMD_CI_WORK", os.getcwd()))
    ap.add_argument("--backend", default = "auto", choices = ["auto", "hip", "vulkan"])
    ap.add_argument("--ctx", type = int, default = 4096)
    ap.add_argument("--exe", action = "append", default = [], help = "state=path: skip the build (local smoke test)")
    args = ap.parse_args()

    root = Path(args.root) / "llama_probe"
    root.mkdir(parents = True, exist_ok = True)
    obs: dict = {"state": args.state}
    mapping = dict(m.split("=", 1) for m in args.map)
    target = mapping.get(args.state) or mapping.get("head")
    url, sha = target.rsplit("@", 1)
    obs["llama_url"], obs["llama_ref"] = url, sha
    obs["tools"] = tools()

    def done():
        args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
        return 0

    given = dict(e.split("=", 1) for e in args.exe)
    if args.state in given:
        obs["backend"] = "given"
        exe = Path(given[args.state])
    else:
        backend = pick_backend(obs["tools"], args.backend)
        if not backend:
            obs["error"] = "no GPU toolchain (hipcc or glslc) on this runner"
            return done()
        obs["backend"] = backend[0]
        exe = build(url, sha, root, backend, obs)
    if not exe:
        return done()
    model = model_path(root, obs)
    if not model:
        return done()
    env = dict(os.environ, LD_LIBRARY_PATH = str(exe.parent))
    obs["help_has_preempt_ram"] = "--preempt-ram" in sh([str(exe), "--help"], env = env)[1]
    # The server no longer logs its buffers at the default verbosity: pin the device instead, so a run
    # that is not on the GPU fails to start rather than silently falling back to the CPU.
    devices = sh([str(exe), "--list-devices"], env = env)[1]
    obs["devices"] = devices[-1500:]
    gpu = next((d for d in ("ROCm0", "Vulkan0", "CUDA0") if d in devices), None)
    obs["device"] = gpu
    dev_args = ["--device", gpu] if gpu else []

    logs = args.out.parent
    port = 18500 + (abs(hash(args.state)) % 200)

    # 1. single requests that are never preempted: text and speed, without and with --preempt-ram
    arms = {"default": [], "preempt_ram": ["--preempt-ram", "2048"]} if obs["help_has_preempt_ram"] \
        else {"default": []}
    obs["single"] = {}
    for arm, extra in arms.items():
        srv = Server(exe, model, port, ["-c", "8192", "--parallel", "1", *dev_args, *extra],
                     logs / f"srv_{args.state}_{arm}.log")
        try:
            if not srv.ready():
                obs["single"][arm] = {"error": "server did not start", "log_tail": srv.text()[-2000:]}
                continue
            runs = [one(port, 128) for _ in range(4)]
            obs["single"][arm] = {"runs": runs[1:], "warmup": runs[0]["status"],
                                  "device_lines": [l for l in srv.text().splitlines()
                                                   if "model buffer size" in l or "using device" in l][:6]}
        finally:
            srv.stop()

    # 2. four chats that do not fit together: base errors, head should park and finish all
    extra = ["--preempt-ram", "2048"] if obs["help_has_preempt_ram"] else []
    srv = Server(exe, model, port, ["-c", str(args.ctx), "--parallel", "4", "--kv-unified", *dev_args, *extra],
                 logs / f"srv_{args.state}_over.log")
    try:
        if srv.ready():
            res, wall = oversubscribed(port)
            text = srv.text()
            obs["over"] = {
                "results": [{k: r[k] for k in ("status", "error", "preempt", "wall")} |
                            {"n_out": (r.get("timings") or {}).get("predicted_n")} for r in res],
                "makespan": wall,
                "completed": sum(1 for r in res if r["status"] == 200 and not r.get("error")),
                "ctx_exceeded": sum(1 for r in res if "exceed" in json.dumps(r.get("error") or "").lower())
                                + text.count("Context size has been exceeded"),
                "parks_logged": text.count("preempted"),
                "props_preempt": get(port, "/props").get("preempt"),
            }
        else:
            obs["over"] = {"error": "server did not start", "log_tail": srv.text()[-2000:]}
    finally:
        srv.stop()
    return done()


if __name__ == "__main__":
    raise SystemExit(main())
