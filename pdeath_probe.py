"""Owner for a PR_SET_PDEATHSIG probe: spawn the installed lemond with the given death signal,
load an NPU model, print "READY <lemond pid>", then sleep until killed.

Usage: pdeath_probe.py STUDIO_LEMONADE_DIR TERM|KILL
"""

import ctypes
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request

root, which = sys.argv[1], sys.argv[2]
sig = signal.SIGTERM if which == "TERM" else signal.SIGKILL


def arm():
    ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, sig)


with socket.socket() as s:
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
env = dict(os.environ, LEMONADE_API_KEY="probe", FLM_MODEL_PATH=f"{root}/flm", FLM_DISABLE_UPDATE_CHECK="1")
proc = subprocess.Popen(
    [f"{root}/11.9.0/lemond", f"{root}/cache", f"{root}/config", "--port", str(port), "--host", "127.0.0.1", "--no-broadcast", "--log-file", "disabled"],
    cwd=f"{root}/11.9.0", env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, preexec_fn=arm,
)


def call(path, body=None, timeout=120):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=None if body is None else json.dumps(body).encode(), headers={"Authorization": "Bearer probe", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read() or b"{}")


for _ in range(120):
    try:
        call("/v1/health", timeout=2)
        break
    except Exception:
        time.sleep(0.5)
print(call("/v1/load", {"model_name": "qwen3-0.6b-FLM", "ctx_size": 8192, "save_options": False}), flush=True)
loaded = call("/v1/health").get("all_models_loaded")
print(f"READY {proc.pid} loaded={[m.get('model_name') for m in loaded or []]}", flush=True)
time.sleep(3600)
