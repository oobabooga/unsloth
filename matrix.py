"""Studio's API on an installed vLLM: precisions, a refused 4-bit, a vision model, or one reload.

usage: matrix.py BASE OUTDIR [all|reload]   (auth as ui_vllm.py: NEW_PASSWORD / BOOTSTRAP_FILE)
"""

import base64, json, os, struct, sys, time, urllib.error, urllib.request, zlib

BASE, OUT = sys.argv[1], sys.argv[2]
MODE = sys.argv[3] if len(sys.argv) > 3 else "all"
TEXT = "unsloth/Qwen2.5-0.5B-Instruct"
VISION = "unsloth/Qwen2.5-VL-3B-Instruct"
os.makedirs(OUT, exist_ok = True)
results = {}


def call(path, body = None, token = None, timeout = 2400):
    req = urllib.request.Request(
        BASE + path,
        data = json.dumps(body).encode() if body is not None else None,
        headers = {"Content-Type": "application/json", **({"Authorization": f"Bearer {token}"} if token else {})},
    )
    try:
        with urllib.request.urlopen(req, timeout = timeout) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode()


def token():
    new = os.environ.get("NEW_PASSWORD", "AmdVllmTest-2026!")
    for password in (new, open(os.environ["BOOTSTRAP_FILE"]).read().strip() if os.path.exists(os.environ.get("BOOTSTRAP_FILE", "")) else None):
        if not password:
            continue
        status, body = call("/api/auth/login", {"username": "unsloth", "password": password})
        if status == 200:
            data = json.loads(body)
            if data.get("must_change_password"):
                _, body = call("/api/auth/change-password", {"current_password": password, "new_password": new}, data["access_token"])
                data = json.loads(body)
            return data["access_token"]
    raise SystemExit("could not log in")


def red_png() -> str:
    width = height = 64
    rows = b"".join(b"\x00" + b"\xff\x00\x00" * width for _ in range(height))
    chunk = lambda kind, data: struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b"")
    return "data:image/png;base64," + base64.b64encode(png).decode()


def load(tok, model, precision = "auto"):
    t0 = time.time()
    status, body = call("/api/inference/load", {"model_path": model, "engine": "vllm", "engine_precision": precision, "max_seq_length": 4096}, tok)
    return status, body[-1500:], round(time.time() - t0)


def chat(tok, model, content):
    status, body = call("/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": content}], "max_tokens": 16, "temperature": 0}, tok, timeout = 600)
    try:
        return json.loads(body)["choices"][0]["message"]["content"]
    except Exception:
        return f"HTTP {status}: {body[:400]}"


def engine(tok):
    status = json.loads(call("/api/inference/status", token = tok)[1])
    return {k: status.get(k) for k in ("active_model", "engine", "engine_precision")}


def record(name, **data):
    results[name] = data
    print("[matrix]", name, json.dumps(data)[:600], flush = True)
    json.dump(results, open(os.path.join(OUT, "matrix.json"), "w"), indent = 1)


tok = token()
question = "What is the capital of France? Answer in one word."
if MODE == "reload":
    status, body, seconds = load(tok, TEXT)
    record("reload_after_restart", status = status, seconds = seconds, engine = engine(tok), reply = chat(tok, TEXT, question))
else:
    for precision in ("int8", "fp8"):
        status, body, seconds = load(tok, TEXT, precision)
        record(precision, status = status, seconds = seconds, engine = engine(tok), reply = chat(tok, TEXT, question), error = body if status != 200 else None)
    before = engine(tok)
    status, body, seconds = load(tok, TEXT, "int4")
    record("int4_refused", status = status, body = body[:400], resident_kept = engine(tok) == before)
    status, body, seconds = load(tok, VISION)
    content = [{"type": "text", "text": "What color is this image? Answer in one word."}, {"type": "image_url", "image_url": {"url": red_png()}}]
    record("vision", status = status, seconds = seconds, engine = engine(tok), reply = chat(tok, VISION, content), error = body if status != 200 else None)
ok = all(
    (r.get("status") == 400 and r.get("resident_kept")) if name == "int4_refused" else (r.get("status") == 200 and r["engine"].get("engine") == "vllm")
    for name, r in results.items()
)
print("[matrix] ok" if ok else "[matrix] FAILED", flush = True)
sys.exit(0 if ok else 1)
