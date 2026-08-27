#!/usr/bin/env python3
"""Greedy A/B of llama-server under two child environments.

Runs the same model, the same prompts and the same sampler twice, changing only
the environment, and compares the emitted token ids. GGML_CUDA_ENABLE_UNIFIED_MEMORY
is read with getenv() != nullptr, so the only honest control is the name being
ABSENT: pass it as `--env=unset:GGML_CUDA_ENABLE_UNIFIED_MEMORY`.

Always use the `--env=...` / `--extra=...` spelling: a value that starts with a
dash is read as another option in the two-token form.
"""
import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

PROMPTS = [
    ("short", "The capital of France is"),
    (
        "long",
        "Below is a short technical note.\n\n"
        + ("The memory subsystem of a modern accelerator exposes several distinct "
           "allocation kinds, each with its own coherence guarantees, and a program "
           "that mixes them without care will observe stale data. ") * 12
        + "\n\nSummarise the note in one paragraph:",
    ),
]


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def build_env(spec):
    env = dict(os.environ)
    for item in spec:
        if item.startswith("unset:"):
            env.pop(item[len("unset:"):], None)
        elif item.startswith("-"):
            env.pop(item[1:], None)
        else:
            k, _, v = item.partition("=")
            env[k] = v
    return env


def wait_ready(port, proc, timeout):
    t0 = time.time()
    url = "http://127.0.0.1:%d/health" % port
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            return False, "server exited rc=%s after %.0fs" % (proc.returncode, time.time() - t0)
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if r.status == 200:
                    return True, "ready in %.0fs" % (time.time() - t0)
        except Exception:
            pass
        time.sleep(2)
    return False, "not ready after %ds" % timeout


def complete(port, prompt, n_predict, timeout):
    body = json.dumps({
        "prompt": prompt,
        "temperature": 0.0,
        "top_k": 1,
        "top_p": 1.0,
        "min_p": 0.0,
        "n_predict": n_predict,
        "cache_prompt": False,
        "return_tokens": True,
        "seed": 0,
    }).encode()
    req = urllib.request.Request(
        "http://127.0.0.1:%d/completion" % port,
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin-dir", required=True)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--env", action="append", default=[])
    ap.add_argument("--ngl", default="999")
    ap.add_argument("--ctx-size", default="8192")
    ap.add_argument("--n-predict", type=int, default=64)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--load-timeout", type=int, default=1800)
    ap.add_argument("--request-timeout", type=int, default=900)
    ap.add_argument("--extra", action="append", default=[])
    args = ap.parse_args()

    port = free_port()
    binary = os.path.join(args.bin_dir, "llama-server")
    cmd = [
        binary, "-m", args.gguf, "-ngl", args.ngl, "-c", args.ctx_size,
        "--parallel", "1", "-fa", "off", "--host", "127.0.0.1",
        "--port", str(port), "--no-warmup",
    ] + list(args.extra)

    env = build_env(args.env)
    env["LD_LIBRARY_PATH"] = args.bin_dir + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    env.setdefault("GGML_LOG_COLORS", "0")

    out = {
        "label": args.label,
        "cmd": cmd,
        "env_spec": args.env,
        "uma": env.get("GGML_CUDA_ENABLE_UNIFIED_MEMORY", "ABSENT"),
        "gguf": args.gguf,
        "gguf_size": os.path.getsize(args.gguf) if os.path.exists(args.gguf) else None,
        "results": [],
    }

    log_path = args.out + ".server.log"
    with open(log_path, "wb") as log:
        try:
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env,
                                    start_new_session=True)
        except Exception as exc:
            out["server_ready"] = False
            out["server_note"] = "could not spawn llama-server: %r" % (exc,)
            with open(args.out, "w") as fh:
                json.dump(out, fh, indent=1)
            print(out["server_note"])
            return 1
        ok, why = wait_ready(port, proc, args.load_timeout)
        out["server_ready"] = ok
        out["server_note"] = why
        if ok:
            for name, prompt in PROMPTS:
                for rep in range(args.repeats):
                    rec = {"prompt": name, "repeat": rep}
                    try:
                        r = complete(port, prompt, args.n_predict, args.request_timeout)
                        rec["content"] = r.get("content", "")
                        rec["tokens"] = r.get("tokens") or []
                        tim = r.get("timings") or {}
                        rec["predicted_per_second"] = tim.get("predicted_per_second")
                        rec["prompt_n"] = tim.get("prompt_n")
                    except Exception as exc:
                        rec["error"] = repr(exc)
                    out["results"].append(rec)
        if proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except Exception:
                proc.terminate()
            try:
                proc.wait(timeout=60)
            except Exception:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except Exception:
                    proc.kill()

    try:
        with open(log_path, "r", errors="replace") as fh:
            text = fh.read()
        out["log_tail"] = text[-8000:]
        out["layers_line"] = [ln for ln in text.splitlines()
                              if "offloaded" in ln or "buffer size" in ln][:40]
        out["build"] = next((ln for ln in text.splitlines() if "build:" in ln), "")
    except Exception as exc:
        out["log_tail"] = repr(exc)

    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print("wrote %s (ready=%s, %s)" % (args.out, out.get("server_ready"), out.get("server_note")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
