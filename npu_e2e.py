"""End-to-end check of Studio's NPU support against a running Studio backend.

Usage: npu_e2e.py BASE_URL PASSWORD OUT_JSON [--phase main|after-restart]
Every check is recorded (name, ok, detail) to OUT_JSON; the log goes to stdout.
"""

import base64
import json
import struct
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
import zlib

BASE, PASSWORD, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
PHASE = sys.argv[5] if len(sys.argv) > 5 and sys.argv[4] == "--phase" else "main"
TOKEN = None
RESULTS = []


def log(*a):
    print(*a, flush=True)


def record(name, ok, detail=""):
    RESULTS.append({"name": name, "ok": bool(ok), "detail": str(detail)[:1500]})
    log(("PASS " if ok else "FAIL ") + name + (f" :: {str(detail)[:600]}" if detail else ""))
    with open(OUT, "w", encoding="utf-8") as h:
        json.dump(RESULTS, h, indent=1)


def call(method, path, body=None, timeout=900, raw=False, headers=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data=data, method=method)
    if TOKEN:
        req.add_header("Authorization", "Bearer " + TOKEN)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    t0 = time.time()
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
        if raw:
            return resp.status, resp, time.time() - t0
        txt = resp.read().decode("utf-8", "replace")
        code = resp.status
    except urllib.error.HTTPError as e:
        txt = e.read().decode("utf-8", "replace")
        code = e.code
    except Exception as e:
        return 0, repr(e), time.time() - t0
    try:
        return code, json.loads(txt), time.time() - t0
    except ValueError:
        return code, txt, time.time() - t0


def sse_lines(resp, limit=None):
    lines = []
    for raw in resp:
        line = raw.decode("utf-8", "replace").rstrip("\n")
        if line.strip():
            lines.append(line)
            if limit and len(lines) >= limit:
                break
    return lines


def data_chunks(lines):
    out = []
    for line in lines:
        if line.startswith("data:") and "[DONE]" not in line:
            try:
                out.append(json.loads(line[5:]))
            except ValueError:
                pass
    return out


def content_of(chunks):
    return "".join(
        (c.get("choices") or [{}])[0].get("delta", {}).get("content") or ""
        for c in chunks
        if c.get("choices")
    )


def red_png():
    w = h = 64
    raw = b"".join(b"\x00" + b"\xff\x00\x00" * w for _ in range(h))

    def chunk(t, d):
        c = struct.pack(">I", len(d)) + t + d
        return c + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)

    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )
    return base64.b64encode(png).decode()


def login():
    global TOKEN
    code, body, _ = call("POST", "/api/auth/login", {"username": "unsloth", "password": PASSWORD})
    if code != 200:
        # The first phase may have had to rotate it.
        code, body, _ = call("POST", "/api/auth/login", {"username": "unsloth", "password": PASSWORD + "-x9"})
    if code != 200 or not isinstance(body, dict):
        record("login", False, body)
        return False
    TOKEN = body.get("access_token")
    if body.get("must_change_password"):
        code, body, _ = call(
            "POST",
            "/api/auth/change-password",
            {"current_password": PASSWORD, "new_password": PASSWORD + "-x9"},
        )
        TOKEN = body.get("access_token", TOKEN) if isinstance(body, dict) else TOKEN
    record("login", TOKEN is not None)
    return TOKEN is not None


def download(model_id):
    code, resp, _ = call("POST", f"/api/npu/models/{model_id}/download", raw=True, timeout=3600)
    if code != 200:
        record(f"download {model_id}", False, resp)
        return False
    t0 = time.time()
    events = [json.loads(l[5:]) for l in sse_lines(resp) if l.startswith("data:")]
    last = events[-1] if events else {}
    percents = [e.get("percent") for e in events if isinstance(e.get("percent"), (int, float))]
    record(
        f"download {model_id}",
        last.get("event") == "complete",
        f"{len(events)} events, percents {percents[:3]}..{percents[-2:]}, {time.time()-t0:.1f}s, last={last}",
    )
    return last.get("event") == "complete"


def load(model_path, **extra):
    code, body, dt = call("POST", "/api/inference/load", {"model_path": model_path, **extra}, timeout=1800)
    ok = code == 200 and isinstance(body, dict) and body.get("status") in ("loaded", "already_loaded")
    record(f"load {model_path}", ok, f"HTTP {code} {dt:.1f}s {json.dumps(body)[:700] if isinstance(body, dict) else body}")
    return body if ok else None


def chat(model, messages, path="/v1/chat/completions", timeout=900, **fields):
    return call("POST", path, {"model": model, "messages": messages, **fields}, timeout=timeout)


def main_phase():
    code, status, _ = call("GET", "/api/npu/status")
    record(
        "npu status: supported XDNA2",
        code == 200 and status.get("supported") and status["hardware"].get("family") == "XDNA2",
        status,
    )
    code, status, dt = call("POST", "/api/npu/enable", timeout=1800)
    record(
        "enable: runtime installed, FastFlowLM validated",
        code == 200 and status.get("state") == "ready" and (status.get("validation") or {}).get("ready"),
        f"{dt:.1f}s {status}",
    )
    code, status2, dt = call("POST", "/api/npu/enable", timeout=600)
    record("enable again is quick and idempotent", code == 200 and dt < 30, f"{dt:.1f}s")

    code, body, _ = call("GET", "/api/npu/models")
    ids = [m["id"] for m in body.get("models", [])] if isinstance(body, dict) else []
    record(
        "catalog: FastFlowLM chat models only",
        {"qwen3-0.6b-FLM", "qwen3-it-4b-FLM", "qwen3.5-0.8b-FLM"} <= set(ids)
        and not any("embed" in i or "whisper" in i for i in ids),
        f"{len(ids)} models: {ids}",
    )

    for model_id in ("qwen3-0.6b-FLM", "qwen3.5-0.8b-FLM", "qwen3-it-4b-FLM"):
        download(model_id)

    # ---- qwen3-0.6b: basics
    loaded = load("lemonade:qwen3-0.6b-FLM")
    record(
        "load response reports the NPU and the context started with",
        bool(loaded) and loaded.get("is_npu") is True and loaded.get("context_length") == 8192,
        loaded,
    )
    code, st, _ = call("GET", "/api/inference/status")
    record(
        "status reports the NPU model",
        code == 200
        and st.get("is_npu") is True
        and st.get("active_model") == "qwen3-0.6b-FLM"
        and st.get("model_identifier") == "lemonade:qwen3-0.6b-FLM"
        and st.get("is_gguf") is False,
        {k: st.get(k) for k in ("active_model", "model_identifier", "is_npu", "is_gguf", "context_length", "supports_reasoning", "supports_tools", "is_vision")},
    )
    code, models, _ = call("GET", "/v1/models")
    record("/v1/models lists it", code == 200 and any(m.get("id") == "lemonade:qwen3-0.6b-FLM" for m in models.get("data", [])), models)

    q = [{"role": "user", "content": "What is 17*23? Answer with just the number."}]
    code, body, dt = chat("qwen3-0.6b-FLM", q, stream=False, max_tokens=300, enable_thinking=False)
    msg = body["choices"][0]["message"] if isinstance(body, dict) and body.get("choices") else {}
    record(
        "non-streaming chat",
        code == 200 and "391" in (msg.get("content") or "") and body.get("model") == "lemonade:qwen3-0.6b-FLM" and body.get("usage", {}).get("prompt_tokens", 0) > 0,
        f"{dt:.2f}s {json.dumps(body)[:600]}",
    )
    code, resp, _ = call("POST", "/v1/chat/completions", {"model": "whatever", "messages": q, "stream": True, "max_tokens": 300, "enable_thinking": False, "stream_options": {"include_usage": True}}, raw=True)
    lines = sse_lines(resp) if code == 200 else []
    chunks = data_chunks(lines)
    usage = next((c.get("usage") for c in reversed(chunks) if c.get("usage")), None)
    record(
        "streaming chat",
        code == 200 and lines and lines[-1] == "data: [DONE]" and "391" in content_of(chunks) and {c.get("model") for c in chunks if c.get("model")} == {"lemonade:qwen3-0.6b-FLM"},
        f"{len(chunks)} chunks, usage={usage}",
    )
    counting = [{"role": "user", "content": "Count from 1 to 20 separated by single spaces, nothing else."}]
    code, body, _ = chat("m", counting, stream=False, max_tokens=120, stop=["7"], enable_thinking=False)
    text = body["choices"][0]["message"]["content"] if code == 200 else body
    record("stop enforced (non-streaming)", code == 200 and "7" not in text and body["choices"][0]["finish_reason"] == "stop" and "6" in text, text)
    code, resp, _ = call("POST", "/v1/chat/completions", {"model": "m", "messages": counting, "stream": True, "max_tokens": 120, "stop": ["7"], "enable_thinking": False}, raw=True)
    lines = sse_lines(resp) if code == 200 else []
    chunks = data_chunks(lines)
    text = content_of(chunks)
    finishes = [c["choices"][0].get("finish_reason") for c in chunks if c.get("choices") and c["choices"][0].get("finish_reason")]
    record("stop enforced (streaming)", code == 200 and "7" not in text and finishes[-1:] == ["stop"] and lines[-1] == "data: [DONE]", f"{text!r} finishes={finishes}")
    code, body, _ = chat("m", counting, stream=False, max_tokens=5, enable_thinking=False)
    record("a length cutoff reports finish_reason length", code == 200 and body["choices"][0]["finish_reason"] == "length", json.dumps(body)[:400])

    img = [{"role": "user", "content": [{"type": "text", "text": "What color is this?"}, {"type": "image_url", "image_url": {"url": "data:image/png;base64," + red_png()}}]}]
    code, body, _ = chat("m", img, stream=False)
    record("image refused by a text-only NPU model", code == 400, body)
    code, body, _ = chat("m", q, stream=False, response_format={"type": "json_object"})
    record("response_format refused", code == 400, body)
    code, body, _ = chat("m", q, stream=False, seed=3)
    record("seed refused", code == 400, body)
    code, body, _ = call("POST", "/v1/completions", {"model": "m", "prompt": "Hello", "max_tokens": 5})
    record("/v1/completions says the NPU model serves chat only", code == 400 and "NPU" in json.dumps(body), body)
    code, body, _ = call("POST", "/api/inference/chat/count_tokens", {"messages": q})
    record("token counting reports unavailable", code == 503, body)

    # Stop button: a cancel_id cancel ends the stream promptly, and the next chat works.
    cancel_id = str(uuid.uuid4())
    story = [{"role": "user", "content": "Write a 600 word story about a lighthouse keeper."}]
    code, resp, _ = call("POST", "/api/inference/chat/completions", {"model": "m", "messages": story, "stream": True, "max_tokens": 900, "cancel_id": cancel_id, "enable_thinking": False}, raw=True)
    first = sse_lines(resp, limit=5) if code == 200 else []
    t0 = time.time()
    ccode, cbody, _ = call("POST", "/api/inference/cancel", {"cancel_id": cancel_id})
    rest = sse_lines(resp) if code == 200 else []
    record("Stop (cancel_id) ends the stream", ccode == 200 and cbody.get("cancelled", 0) >= 1 and time.time() - t0 < 10, f"cancelled={cbody} {len(first)}+{len(rest)} lines in {time.time()-t0:.2f}s")
    code, body, dt = chat("m", [{"role": "user", "content": "Say OK."}], stream=False, max_tokens=10, enable_thinking=False)
    record("chat after a stop", code == 200, f"{dt:.2f}s")

    # Client disconnect mid-stream, then chat again.
    code, resp, _ = call("POST", "/v1/chat/completions", {"model": "m", "messages": story, "stream": True, "max_tokens": 900, "enable_thinking": False}, raw=True)
    if code == 200:
        sse_lines(resp, limit=4)
        resp.close()
    code, body, dt = chat("m", [{"role": "user", "content": "Say OK."}], stream=False, max_tokens=10, enable_thinking=False)
    record("chat after a client disconnect", code == 200 and dt < 30, f"{dt:.2f}s")

    # ---- qwen3.5-0.8b: reasoning + vision
    load("lemonade:qwen3.5-0.8b-FLM")
    code, st, _ = call("GET", "/api/inference/status")
    record("switching NPU models", st.get("active_model") == "qwen3.5-0.8b-FLM" and st.get("is_vision") is True and st.get("supports_reasoning") is True, {k: st.get(k) for k in ("active_model", "is_vision", "supports_reasoning")})
    code, resp, _ = call("POST", "/v1/chat/completions", {"model": "m", "messages": q, "stream": True, "max_tokens": 1200, "enable_thinking": True}, raw=True)
    chunks = data_chunks(sse_lines(resp)) if code == 200 else []
    reasoning = "".join((c.get("choices") or [{}])[0].get("delta", {}).get("reasoning_content") or "" for c in chunks if c.get("choices"))
    record("thinking on streams reasoning_content apart from content", code == 200 and len(reasoning) > 20 and "<think>" not in content_of(chunks), f"reasoning {len(reasoning)} chars, content {content_of(chunks)[-120:]!r}")
    code, body, _ = chat("m", q, stream=False, max_tokens=300, enable_thinking=False)
    m = body["choices"][0]["message"] if code == 200 else {}
    record("thinking off", code == 200 and not m.get("reasoning_content") and "<think>" not in (m.get("content") or ""), json.dumps(body)[:400])
    code, body, _ = chat("m", q, stream=False, max_tokens=40, enable_thinking=True)
    m = body["choices"][0]["message"] if code == 200 else {}
    record("a thinking reply cut off by max_tokens keeps no raw <think> in content", code == 200 and "<think>" not in (m.get("content") or "") and body["choices"][0]["finish_reason"] == "length", json.dumps(body)[:500])
    code, body, _ = chat("m", [{"role": "user", "content": [{"type": "text", "text": "What single color fills this image? Answer with one word."}, {"type": "image_url", "image_url": {"url": "data:image/png;base64," + red_png()}}]}], stream=False, max_tokens=300, enable_thinking=False)
    text = body["choices"][0]["message"]["content"] if code == 200 else body
    record("vision on a vision NPU model", code == 200 and "red" in str(text).lower(), text)

    # ---- qwen3-it-4b: tools
    load("lemonade:qwen3-it-4b-FLM")
    tools = [{"type": "function", "function": {"name": "get_weather", "description": "Get current weather for a city", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
    wq = [{"role": "user", "content": "What's the weather in Paris right now? Use the tool."}]
    code, body, _ = chat("m", wq, stream=False, tools=tools, max_tokens=300)
    calls = body["choices"][0]["message"].get("tool_calls") if code == 200 else None
    record("tool call (non-streaming)", code == 200 and calls and calls[0]["function"]["name"] == "get_weather" and body["choices"][0]["finish_reason"] == "tool_calls", json.dumps(body)[:500])
    code, resp, _ = call("POST", "/v1/chat/completions", {"model": "m", "messages": wq, "tools": tools, "stream": True, "max_tokens": 300}, raw=True)
    chunks = data_chunks(sse_lines(resp)) if code == 200 else []
    streamed = [tc for c in chunks if c.get("choices") for tc in (c["choices"][0].get("delta", {}).get("tool_calls") or [])]
    record("tool call (streaming)", code == 200 and streamed and streamed[0]["function"]["name"] == "get_weather", json.dumps(streamed)[:400])
    if calls:
        follow = wq + [{"role": "assistant", "content": "", "tool_calls": calls}, {"role": "tool", "tool_call_id": calls[0]["id"], "content": json.dumps({"temp_c": 31, "condition": "thunderstorm"})}]
        code, body, _ = chat("m", follow, stream=False, tools=tools, max_tokens=200)
        text = body["choices"][0]["message"].get("content") if code == 200 else body
        record("tool result round trip", code == 200 and ("31" in str(text) or "thunder" in str(text).lower()), text)
    code, body, _ = chat("m", wq, stream=False, tools=tools, tool_choice="none", max_tokens=150)
    record("tool_choice none withholds the tools", code == 200 and not body["choices"][0]["message"].get("tool_calls"), json.dumps(body)[:400])
    code, body, _ = chat("m", q, stream=False, max_tokens=50, temperature=0.2)
    record("sampler values reach FastFlowLM and it answers", code == 200, json.dumps(body)[:300])

    # ---- A GGUF load replaces the NPU model, and an NPU load replaces the GGUF.
    gguf = load("unsloth/Qwen3-0.6B-GGUF", gguf_variant="Q4_K_M", max_seq_length=4096)
    code, npu, _ = call("GET", "/api/npu/status")
    code2, st, _ = call("GET", "/api/inference/status")
    record("a GGUF load unloads the NPU model", bool(gguf) and npu.get("loaded_model") is None and st.get("is_gguf") is True and not st.get("is_npu"), {"npu_loaded": npu.get("loaded_model"), "active": st.get("active_model"), "is_gguf": st.get("is_gguf")})
    code, body, _ = chat("m", q, stream=False, max_tokens=200, enable_thinking=False)
    record("chat on the GGUF", code == 200 and body.get("model") != "lemonade:qwen3-it-4b-FLM", json.dumps(body)[:300])
    load("lemonade:qwen3-0.6b-FLM")
    code, st, _ = call("GET", "/api/inference/status")
    record("an NPU load unloads the GGUF", st.get("is_npu") is True and st.get("is_gguf") is False, {k: st.get(k) for k in ("active_model", "is_gguf", "is_npu")})

    # ---- Eject.
    code, body, _ = call("POST", "/api/inference/unload", {"model_path": "lemonade:qwen3-0.6b-FLM"})
    code2, st, _ = call("GET", "/api/inference/status")
    code3, npu, _ = call("GET", "/api/npu/status")
    record("unload", code == 200 and not st.get("active_model") and npu.get("loaded_model") is None, {"unload": body, "active": st.get("active_model"), "npu": npu.get("loaded_model")})
    code, body, _ = chat("m", q, stream=False, max_tokens=20)
    record("chat with nothing loaded fails cleanly", code in (400, 503), body)
    code, body, _ = call("DELETE", "/api/npu/models/qwen3-it-4b-FLM")
    code2, cat, _ = call("GET", "/api/npu/models")
    gone = {m["id"]: m["downloaded"] for m in cat.get("models", [])}.get("qwen3-it-4b-FLM") is False
    record("delete a downloaded model", code == 200 and gone, body)
    # Leave one model loaded for the shutdown check.
    load("lemonade:qwen3-0.6b-FLM")


def after_restart_phase():
    code, st, _ = call("GET", "/api/npu/status")
    record("after restart: nothing loaded, runtime installed", code == 200 and st.get("runtime_installed") and st.get("loaded_model") is None, st)
    code, cat, _ = call("GET", "/api/npu/models")
    downloaded = sorted(m["id"] for m in cat.get("models", []) if m["downloaded"]) if isinstance(cat, dict) else cat
    record("after restart: downloads kept", code == 200 and "qwen3-0.6b-FLM" in downloaded and "qwen3-it-4b-FLM" not in downloaded, downloaded)
    loaded = load("lemonade:qwen3-0.6b-FLM")
    code, body, _ = chat("m", [{"role": "user", "content": "Say OK."}], stream=False, max_tokens=10, enable_thinking=False)
    record("after restart: load and chat", bool(loaded) and code == 200, json.dumps(body)[:300])


def shutdown_phase():
    code, body, _ = call("POST", "/api/shutdown")
    record("graceful shutdown requested", code == 200, body)


PHASES = {"main": main_phase, "after-restart": after_restart_phase, "shutdown": shutdown_phase}

if login():
    try:
        PHASES[PHASE]()
    except Exception as exc:  # keep what was recorded
        import traceback

        record("harness exception", False, traceback.format_exc())
failed = [r["name"] for r in RESULTS if not r["ok"]]
log(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} passed" + (f"; failed: {failed}" if failed else ""))
