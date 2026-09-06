#!/usr/bin/env python3
"""How long does a prompt of N tokens take to process, and does the server say so
while it is happening?

Two questions, one run. The first is arithmetic a deadline depends on: a 250K
prefill that takes 40 minutes cannot be served under a fixed 20 minute
first-token timeout, and the way to know is to measure the shorter lengths and
extrapolate rather than to wait 40 minutes in CI. The second is the mechanism a
renewable deadline needs: llama-server only emits `prompt_progress` when the
request asks for it, and a client that renews on progress it never receives has
a 20 minute timeout with extra steps.

Prompt lengths are exact, not estimated: the text is tokenized once through
/tokenize and each length is cut from those ids and turned back into text
through /detokenize, so "32768 tokens" is 32768 tokens and not a guess about
characters per token.

Writes JSON: per length, the wall time to the first generated token, every
progress event with its timestamp and processed count, and the server's own
prompt_ms. Standard library only.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path


def post(port: int, path: str, payload: dict, timeout: int = 3600) -> dict:
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}",
                                 data = json.dumps(payload).encode("utf-8"),
                                 headers = {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout = timeout) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def stream_completion(port: int, prompt: str, n_predict: int, timeout: int) -> dict:
    """One streamed completion, timestamping every event the server sends."""
    payload = {"prompt": prompt, "n_predict": n_predict, "stream": True,
               "return_progress": True, "cache_prompt": False, "temperature": 0}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/completion",
                                 data = json.dumps(payload).encode("utf-8"),
                                 headers = {"Content-Type": "application/json"})
    t0 = time.monotonic()
    out: dict = {"progress_events": [], "first_token_s": None, "total_s": None,
                 "error": None, "text": "", "timings": {}}
    try:
        with urllib.request.urlopen(req, timeout = timeout) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                now = round(time.monotonic() - t0, 3)
                try:
                    ev = json.loads(line[5:].strip())
                except json.JSONDecodeError:
                    continue
                prog = ev.get("prompt_progress")
                if isinstance(prog, dict):
                    out["progress_events"].append({"t": now, **{
                        k: prog.get(k) for k in ("total", "cache", "processed", "time_ms")}})
                content = ev.get("content") or ""
                if content:
                    if out["first_token_s"] is None:
                        out["first_token_s"] = now
                    out["text"] += content
                if ev.get("timings"):
                    out["timings"] = ev["timings"]
                if ev.get("stop"):
                    break
    except Exception as e:  # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {e}"
    out["total_s"] = round(time.monotonic() - t0, 3)
    processed = [e.get("processed") or 0 for e in out["progress_events"]]
    out["progress_count"] = len(processed)
    out["progress_monotonic"] = all(b >= a for a, b in zip(processed, processed[1:]))
    out["progress_strictly_increasing_count"] = sum(
        1 for a, b in zip(processed, processed[1:]) if b > a)
    return out


class Server:
    def __init__(self, exe: str, args: list[str], env: dict, log: Path, port: int,
                 load_timeout: int):
        self.exe, self.args, self.env, self.log = exe, args, env, log
        self.port, self.load_timeout = port, load_timeout
        self.p = None
        self.fh = None

    def __enter__(self):
        cmd = [self.exe, "--host", "127.0.0.1", "--port", str(self.port), *self.args]
        self.fh = open(self.log, "w", encoding = "utf-8", errors = "replace")
        self.fh.write("CMD: " + " ".join(cmd) + "\n")
        self.fh.flush()
        self.p = subprocess.Popen(cmd, stdout = self.fh, stderr = subprocess.STDOUT,
                                  env = self.env)
        deadline = time.monotonic() + self.load_timeout
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health",
                                            timeout = 2) as r:
                    if b'"ok"' in r.read():
                        return self
            except Exception:  # noqa: BLE001
                pass
            if self.p.poll() is not None:
                raise RuntimeError(f"server died rc={self.p.returncode}; see {self.log}")
            time.sleep(2)
        raise RuntimeError(f"server did not become ready in {self.load_timeout}s")

    def __exit__(self, *_):
        # By PID. `pkill -f` on this box would match the harness running it.
        if self.p and self.p.poll() is None:
            self.p.send_signal(signal.SIGTERM if os.name == "nt" else signal.SIGINT)
            try:
                self.p.wait(60)
            except subprocess.TimeoutExpired:
                self.p.kill()
                self.p.wait()
        if self.fh:
            self.fh.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required = True, help = "directory holding llama-server")
    ap.add_argument("--model", required = True)
    ap.add_argument("--text", required = True, help = "long UTF-8 source text")
    ap.add_argument("--lengths", default = "8192,32768,65536,131072")
    ap.add_argument("--ctx", type = int, default = 140000)
    ap.add_argument("--n-predict", type = int, default = 8)
    ap.add_argument("--port", type = int, default = 8127)
    ap.add_argument("--load-timeout", type = int, default = 1800)
    ap.add_argument("--request-timeout", type = int, default = 5400)
    ap.add_argument("--server-arg", action = "append", default = [])
    ap.add_argument("--env", action = "append", default = [])
    ap.add_argument("--out", required = True)
    ap.add_argument("--log-dir", default = "")
    a = ap.parse_args()

    lengths = [int(x) for x in a.lengths.split(",") if x.strip()]
    log_dir = Path(a.log_dir or Path(a.out).parent)
    log_dir.mkdir(parents = True, exist_ok = True)
    exe = str(Path(a.bin) / ("llama-server.exe" if os.name == "nt" else "llama-server"))

    env = dict(os.environ)
    if os.name != "nt":
        env["LD_LIBRARY_PATH"] = f"{a.bin}{os.pathsep}" + env.get("LD_LIBRARY_PATH", "")
    for kv in a.env:
        k, _, v = kv.partition("=")
        env[k] = v

    args = ["-m", a.model, "-c", str(a.ctx), "-np", "1", "-ngl", "999", "--fit", "off",
            "--no-warmup", *a.server_arg]
    res: dict = {"bin": a.bin, "model": a.model, "ctx": a.ctx, "lengths": lengths,
                 "server_args": args, "runs": []}

    # UTF-8 explicitly: the Windows default is cp1252, and a mis-decoded prompt
    # is a different prompt, which this project has already paid for once.
    text = Path(a.text).read_text(encoding = "utf-8", errors = "replace")
    with Server(exe, args, env, log_dir / "prefill_server.log", a.port, a.load_timeout) as s:
        ids = post(a.port, "/tokenize", {"content": text})["tokens"]
        res["source_tokens"] = len(ids)
        for n in lengths:
            if n > len(ids):
                res["runs"].append({"tokens_requested": n, "skipped":
                                    f"the source text holds {len(ids)} tokens"})
                continue
            prompt = post(a.port, "/detokenize", {"tokens": ids[:n]})["content"]
            run = stream_completion(a.port, prompt, a.n_predict, a.request_timeout)
            run["tokens_requested"] = n
            run["prompt_n"] = (run.get("timings") or {}).get("prompt_n")
            run["prompt_ms"] = (run.get("timings") or {}).get("prompt_ms")
            if run["prompt_ms"]:
                run["prompt_tokens_per_s"] = round(
                    (run["prompt_n"] or n) / (run["prompt_ms"] / 1000.0), 2)
            res["runs"].append(run)
            print(f"{n:>7} tokens: first token {run['first_token_s']}s, "
                  f"{run.get('prompt_tokens_per_s')} tok/s, "
                  f"{run['progress_count']} progress events", flush = True)
        res["server_rc"] = s.p.poll()

    ok = [r for r in res["runs"] if r.get("first_token_s")]
    if len(ok) >= 2:
        # Linear in prompt length is the right first model for prefill; the
        # residual is reported so a bad fit is visible rather than assumed away.
        xs = [r["prompt_n"] or r["tokens_requested"] for r in ok]
        ys = [r["first_token_s"] for r in ok]
        n = len(xs)
        mx, my = sum(xs) / n, sum(ys) / n
        denom = sum((x - mx) ** 2 for x in xs)
        slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom if denom else 0.0
        intercept = my - slope * mx
        res["fit"] = {"seconds_per_token": slope, "intercept_s": round(intercept, 2),
                      "residuals_s": [round(y - (slope * x + intercept), 2)
                                      for x, y in zip(xs, ys)]}
        for target in (250000, 262144):
            res["fit"][f"predicted_s_at_{target}"] = round(slope * target + intercept, 1)
        res["fit"]["exceeds_1200s_at_tokens"] = (
            round((1200 - intercept) / slope) if slope > 0 else None)
    res["progress_events_seen"] = sum(r.get("progress_count", 0) for r in res["runs"])
    Path(a.out).write_text(json.dumps(res, indent = 2), encoding = "utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "runs"}, indent = 2))
    # No progress events means a renewable first-token deadline has nothing to
    # renew on: a finding, and one that must not pass quietly.
    return 0 if res["progress_events_seen"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
