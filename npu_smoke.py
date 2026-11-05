"""Drive Studio's LemonadeNpuBackend: enable, then (phase 'chat') download, load and chat."""

import json
import sys
import time
from pathlib import Path

repo, root, phase = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
sys.path.insert(0, str(repo / "studio" / "backend"))

import httpx  # noqa: E402
from core.inference.npu_backend import LemonadeNpuBackend  # noqa: E402

backend = LemonadeNpuBackend(root = root)
print("hardware:", json.dumps(backend.hardware()))
status = backend.status()
print("status before:", json.dumps({k: status[k] for k in ("state", "ready", "runtime_installed", "versions")}))


def lemond_tail(n = 80):
    server = backend._server
    lines = list(getattr(server, "_tail", []) or [])[-n:]
    print("---- lemond output (last %d lines) ----" % len(lines))
    for line in lines:
        print("  ", line)


if phase == "enable":
    start = time.time()
    result = backend.enable()
    print(f"enable took {time.time() - start:.0f}s")
    print("versions:", json.dumps(backend.status()["versions"]))
    backend.shutdown()
    sys.exit(0)


def chat(model_id, context_length):
    print(f"==== {model_id} (context_length={context_length})")
    last = -10
    for event in backend.download(model_id):
        pct = event.get("percent")
        if isinstance(pct, (int, float)) and pct >= last + 25:
            last = pct
            print("download:", event.get("event"), pct)
    backend.load(model_id, context_length = context_length)
    up = backend.upstream()
    print("loaded:", up.model, "ctx", up.context_length)
    start = time.time()
    response = httpx.post(
        f"{up.base_url}/chat/completions",
        headers = {"Authorization": f"Bearer {up.api_key}"},
        json = {
            "model": up.model,
            "messages": [{"role": "user", "content": "What is the capital of France? One word."}],
            "max_tokens": 512,
            "stream": False,
        },
        timeout = 600,
    )
    print("chat status:", response.status_code, f"{time.time() - start:.1f}s")
    body = response.json()
    print("chat body:", json.dumps(body)[:1200])
    message = body["choices"][0]["message"]
    text = (message.get("content") or "") + (message.get("reasoning_content") or "")
    backend.unload()
    return response.status_code == 200 and "paris" in text.lower()


# The first catalog call runs the pending pin upgrade, like Studio's first NPU use after an update.
catalog = backend.catalog()
print("versions after upgrade:", json.dumps(backend.status()["versions"]))
print("catalog:", len(catalog), "chat models:", ", ".join(m.id for m in catalog))
plain = [m for m in catalog if m.size_gb and "decision" not in m.labels]
primary = min(plain, key = lambda m: m.size_gb)
ok = False
try:
    ok = chat(primary.id, 2048)
except Exception as exc:  # noqa: BLE001
    print("PRIMARY FAILED:", exc)
    lemond_tail()
print("PRIMARY NPU CHAT", "OK" if ok else "FAILED", primary.id)

# Informational: a Decision 3.0 model that is new in FastFlowLM 1.0.12.
d3 = next((m for m in catalog if m.id.startswith("d3-edge")), None)
if d3 is not None:
    try:
        print("D3 RESULT:", chat(d3.id, None))
    except Exception as exc:  # noqa: BLE001
        print("D3 FAILED:", exc)
        lemond_tail()
backend.shutdown()
sys.exit(0 if ok else 1)
