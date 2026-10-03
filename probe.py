"""Print exactly what FastFlowLM, behind Studio's own Lemonade, sends on a chat.

Run with the Studio venv's python from studio/backend. Usage: probe.py MODEL_ID
"""

import json
import sys
import time

import httpx

sys.path.insert(0, ".")
from core.inference.npu_backend import get_npu_backend  # noqa: E402

MODEL = sys.argv[1]
b = get_npu_backend()
print("STATUS", json.dumps(b.status())[:800], flush = True)
t0 = time.time()
print("ENABLE", json.dumps(b.enable())[:800], f"{time.time() - t0:.1f}s", flush = True)
last = None
for ev in b.download(MODEL):
    last = ev
print("DOWNLOAD", last, flush = True)
b.load(MODEL, None)
up = b.upstream()
root = up.base_url[: -len("/v1")]
H = {"Authorization": f"Bearer {up.api_key}"}
client = httpx.Client(timeout = 600, trust_env = False)
print("HEALTH", client.get(f"{root}/v1/health", headers = H).text[:1500], flush = True)

MSG = [{"role": "user", "content": "Write two sentences about the moon."}]


def show_stream(label, body):
    print(f"===== {label}", flush = True)
    with client.stream("POST", f"{up.base_url}/chat/completions", json = body, headers = H) as r:
        lines = [l for l in r.iter_lines() if l.strip()]
    print(f"{len(lines)} lines", flush = True)
    for i, line in enumerate(lines):
        if i < 2 or i >= len(lines) - 4 or "usage" in line or "timings" in line:
            print(f"[{i}] {line}", flush = True)


show_stream(
    "stream, include_usage",
    {"model": up.model, "messages": MSG, "stream": True, "max_tokens": 200,
     "stream_options": {"include_usage": True}},
)
for path in ("/v1/stats", "/api/v1/stats"):
    r = client.get(f"{root}{path}", headers = H)
    print("STATS", path, r.status_code, r.text[:1500], flush = True)
show_stream("stream, no stream_options", {"model": up.model, "messages": MSG, "stream": True, "max_tokens": 200})
show_stream(
    "stream, thinking on",
    {"model": up.model, "messages": MSG, "stream": True, "max_tokens": 300, "enable_thinking": True,
     "stream_options": {"include_usage": True}},
)
r = client.post(f"{up.base_url}/chat/completions", json = {"model": up.model, "messages": MSG, "max_tokens": 200}, headers = H)
print("===== non-stream", r.status_code, r.text[:2500], flush = True)
r = client.get(f"{root}/v1/stats", headers = H)
print("STATS after non-stream", r.status_code, r.text[:1500], flush = True)
b.unload()
b.shutdown()
