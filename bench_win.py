"""Compare llama.cpp fork Windows builds on one small MTP model. Stdlib only."""
import ctypes
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(os.environ.get("RUNNER_TEMP", ".")) / "ab"
ROOT.mkdir(parents=True, exist_ok=True)
BUILDS = os.environ.get("BUILDS", "b11985-mix-c116c32,b12128-mix-2f23ac9,b12189-mix-725acdc").split(",")
FLAVOR = os.environ.get("FLAVOR", "windows-x64-cpu")
MODEL_URL = os.environ.get(
    "MODEL_URL",
    "https://huggingface.co/unsloth/Qwen3.5-0.8B-MTP-GGUF/resolve/main/Qwen3.5-0.8B-Q4_K_M.gguf",
)
REPS = int(os.environ.get("REPS", "2"))
PROMPT = (
    "Write a detailed explanation of how computer memory works. Start with the differences between RAM and "
    "storage, then explain caches, virtual memory, and memory bandwidth."
)


def fetch(url, dest):
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    req = urllib.request.Request(url, headers={"User-Agent": "bench"})
    with urllib.request.urlopen(req, timeout=600) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f, 1 << 20)
    return dest


def get_build(tag):
    d = ROOT / tag
    if not (d / "llama-server.exe").exists():
        z = fetch(
            f"https://github.com/unslothai/llama.cpp/releases/download/{tag}/app-{tag}-{FLAVOR}.zip",
            ROOT / f"{tag}.zip",
        )
        d.mkdir(exist_ok=True)
        with zipfile.ZipFile(z) as zf:
            zf.extractall(d)
        exe = next(d.rglob("llama-server.exe"))
        if exe.parent != d:
            d = exe.parent
    else:
        d = next(d.rglob("llama-server.exe")).parent
    return d


def strip_manifest(src_dir, dst_dir):
    """Copy a build and delete the embedded RT_MANIFEST (type 24) from every exe."""
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    shutil.copytree(src_dir, dst_dir)
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.BeginUpdateResourceW.restype = ctypes.c_void_p
    k32.BeginUpdateResourceW.argtypes = [ctypes.c_wchar_p, ctypes.c_int]
    k32.UpdateResourceW.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ushort, ctypes.c_void_p, ctypes.c_uint]
    k32.EndUpdateResourceW.argtypes = [ctypes.c_void_p, ctypes.c_int]
    done = []
    for exe in dst_dir.glob("*.exe"):
        for lang in (0x0409, 0x0000):
            h = k32.BeginUpdateResourceW(str(exe), 0)
            if not h:
                continue
            ok = k32.UpdateResourceW(h, ctypes.c_void_p(24), ctypes.c_void_p(1), lang, None, 0)
            k32.EndUpdateResourceW(h, 0 if ok else 1)
            if ok:
                done.append(exe.name)
                break
    return done


def has_manifest(exe):
    data = Path(exe).read_bytes()
    return b"activeCodePage" in data


def server(build_dir, args, env_extra=None):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    env = os.environ.copy()
    env.update(env_extra or {})
    log = open(ROOT / f"log-{port}.txt", "w", encoding="utf-8", errors="replace")
    cmd = [str(build_dir / "llama-server.exe"), "--host", "127.0.0.1", "--port", str(port), *args]
    proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=str(build_dir))
    base = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 300
    while True:
        if proc.poll() is not None:
            log.close()
            tail = (ROOT / f"log-{port}.txt").read_text(encoding="utf-8", errors="replace")[-1500:]
            raise RuntimeError(f"server exited rc={proc.returncode}: {tail}")
        if time.monotonic() > deadline:
            proc.kill()
            raise TimeoutError("startup")
        try:
            with urllib.request.urlopen(base + "/health", timeout=1) as r:
                if r.status == 200:
                    break
        except Exception:
            time.sleep(0.5)
    return proc, base, log


def post(base, path, data, stream=False):
    req = urllib.request.Request(base + path, json.dumps(data).encode(), {"Content-Type": "application/json"})
    if not stream:
        with urllib.request.urlopen(req, timeout=900) as r:
            return json.load(r)
    last = None
    n = 0
    t0 = time.monotonic()
    first = None
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            n += 1
            if first is None:
                first = time.monotonic() - t0
            try:
                obj = json.loads(line[6:])
            except Exception:
                continue
            if obj.get("timings"):
                last = obj["timings"]
    return {"timings": last, "chunks": n, "first_chunk_s": first, "wall": time.monotonic() - t0}


def measure(build_dir, label, args, mode, tokens, env_extra=None):
    row = {"label": label, "mode": mode}
    try:
        proc, base, log = server(build_dir, args, env_extra)
    except Exception as e:  # noqa: BLE001
        row["error"] = repr(e)[:600]
        return row
    try:
        runs = []
        for i in range(REPS):
            t0 = time.monotonic()
            if mode == "raw":
                d = post(base, "/completion", {"prompt": PROMPT + "\n", "n_predict": tokens, "temperature": 0.6, "top_k": 20,
                                               "top_p": 0.95, "seed": 1 + i, "cache_prompt": False, "ignore_eos": True})
            else:
                d = post(base, "/v1/chat/completions", {"messages": [{"role": "user", "content": PROMPT}], "max_tokens": tokens,
                                                         "temperature": 0.6, "top_k": 20, "top_p": 0.95, "seed": 1 + i,
                                                         "stream": True, "timings_per_token": True}, stream=True)
            t = d.get("timings") or {}
            runs.append({
                "tok_s": round(t.get("predicted_per_second") or 0, 2),
                "pred_n": t.get("predicted_n"),
                "prompt_tok_s": round(t.get("prompt_per_second") or 0, 1),
                "draft_n": t.get("draft_n"),
                "draft_acc": t.get("draft_n_accepted"),
                "wall": round(time.monotonic() - t0, 2),
            })
        row["runs"] = runs
    except Exception as e:  # noqa: BLE001
        row["error"] = repr(e)[:400]
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except Exception:
            proc.kill()
        log.close()
    return row


def main():
    model = fetch(MODEL_URL, ROOT / "model.gguf")
    print("model bytes", model.stat().st_size, flush=True)
    common = ["-m", str(model), "--parallel", "4", "--kv-unified", "--no-context-shift", "-c", "8192", "--jinja"]
    mtp = ["--spec-type", "draft-mtp", "--spec-draft-n-max", "2"]
    cases = [
        ("raw_nospec", common, "raw", 200, None),
        ("raw_mtp", common + mtp, "raw", 200, None),
        ("chat_mtp", common + mtp + ["--reasoning", "on"], "chat", 400, None),
        ("chat_mtp_nockpt", common + mtp + ["--reasoning", "on", "--cache-ram", "0", "--ctx-checkpoints", "0"], "chat", 400, None),
        ("raw_mtp_threads2_passive", common + mtp + ["--threads", "2", "--cache-ram", "0", "--ctx-checkpoints", "0"], "raw", 200,
         {"OMP_WAIT_POLICY": "PASSIVE", "OMP_NUM_THREADS": "2"}),
    ]
    only = os.environ.get("CASES")
    if only:
        cases = [c for c in cases if c[0] in only.split(",")]
    results = []
    dirs = {}
    for tag in BUILDS:
        try:
            dirs[tag] = get_build(tag)
            print(tag, "manifest:", has_manifest(dirs[tag] / "llama-server.exe"), flush=True)
        except Exception as e:  # noqa: BLE001
            print(tag, "DOWNLOAD FAILED", repr(e)[:300], flush=True)
    if os.environ.get("STRIP_TAG") in dirs:
        tag = os.environ["STRIP_TAG"]
        stripped = ROOT / (tag + "-nomanifest")
        done = strip_manifest(dirs[tag], stripped)
        print("stripped manifest from", len(done), "exes; server still has manifest:", has_manifest(stripped / "llama-server.exe"), flush=True)
        dirs[tag + "-nomanifest"] = stripped
    order = list(dirs)
    for rnd in range(int(os.environ.get("ROUNDS", "1"))):
        for tag in order:
            for name, args, mode, tokens, env_extra in cases:
                row = measure(dirs[tag], f"{tag}/{name}", args, mode, tokens, env_extra)
                row["round"] = rnd
                results.append(row)
                print(json.dumps(row), flush=True)
    (ROOT / "results.json").write_text(json.dumps(results, indent=1))
    print("\nSUMMARY (tok/s per rep)")
    for r in results:
        print(f"{r['label']:<60} {[x['tok_s'] for x in r.get('runs', [])]} {r.get('error', '')[:200]}")


if __name__ == "__main__":
    main()
