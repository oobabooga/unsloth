"""After the UI walkthrough: load the same Studio's vLLM through the API in every precision mode
it offers, plus a vision checkpoint, and chat each time. Records what works on this AMD GPU and how
the rest fails.

Usage: vllm_matrix.py BASE_URL PASSWORD OUT_JSON CASE...  (CASE = model[:precision])
"""

import json
import sys
import time

import httpx

BASE, PASSWORD, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
CASES = sys.argv[4:]
ROWS = []
client = httpx.Client(base_url = BASE, timeout = 3600, trust_env = False)
token = client.post("/api/auth/login", json = {"username": "unsloth", "password": PASSWORD}).json()["access_token"]
client.headers["Authorization"] = f"Bearer {token}"


def save():
    with open(OUT, "w", encoding = "utf-8") as handle:
        json.dump(ROWS, handle, indent = 1)


def chat(model, image = None):
    content = "What is the capital of France? Answer in one word."
    if image:
        content = [
            {"type": "text", "text": "What color is this image? Answer in one word."},
            {"type": "image_url", "image_url": {"url": image}},
        ]
    response = client.post(
        "/v1/chat/completions",
        json = {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 16,
            "temperature": 0,
        },
    )
    body = response.json()
    try:
        return response.status_code, body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return response.status_code, json.dumps(body)[:600]


def _red_png() -> str:
    import base64, struct, zlib

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    width = height = 64
    raw = b"".join(b"\x00" + b"\xff\x00\x00" * width for _ in range(height))
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")
    return "data:image/png;base64," + base64.b64encode(png).decode()


RED = _red_png()

for case in CASES:
    model, _, precision = case.partition(":")
    precision = precision or "auto"
    row = {"model": model, "precision": precision}
    started = time.monotonic()
    try:
        response = client.post(
            "/api/inference/load",
            json = {"model_path": model, "engine": "vllm", "engine_precision": precision, "max_seq_length": 4096},
        )
        row["load_status"] = response.status_code
        text = response.text.strip()
        # A long load pads the response and reports its outcome in the final JSON object.
        try:
            body = json.loads(text[text.rfind("\n{") + 1 :] if "\n{" in text else text)
        except ValueError:
            body = {"raw": text[-800:]}
        row["load_s"] = round(time.monotonic() - started)
        status = client.get("/api/inference/status").json()
        row["engine"] = status.get("engine")
        row["active_model"] = status.get("active_model")
        row["engine_precision"] = status.get("engine_precision")
        if status.get("active_model") and status.get("engine") == "vllm":
            row["chat"] = chat(status["active_model"], RED if ("VL" in model or "gemma-3" in model) else None)
            row["ok"] = row["chat"][0] == 200
        else:
            row["ok"] = False
            row["error"] = str(body.get("detail") or body.get("error") or body)[-1500:]
    except Exception as exc:
        row["ok"] = False
        row["error"] = f"{type(exc).__name__}: {exc}"[-1500:]
    print(json.dumps(row)[:2000], flush = True)
    ROWS.append(row)
    save()

print(f"{sum(r['ok'] for r in ROWS)}/{len(ROWS)} engine loads answered")
