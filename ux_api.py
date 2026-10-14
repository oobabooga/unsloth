"""An app developer pointing an OpenAI/Anthropic client at Studio while an NPU model is loaded."""
import json, os, re, shutil, subprocess, sys, time

sys.path.insert(0, os.path.dirname(__file__))
import uxlib as U
from uxlib import step, note, api, poll, run

PASSWORD = (U.ROOT / "password.txt").read_text().strip()
EXE = shutil.which("unsloth")
proc = {}


def start():
    log = open(U.OUT / "studio-api.log", "w", encoding="utf-8")
    kw = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if U.WIN else {"start_new_session": True}
    t = time.monotonic()
    proc["p"] = subprocess.Popen([EXE, "studio", "-H", "127.0.0.1", "-p", str(U.PORT)], stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, **kw)
    poll(lambda: U.requests.get(U.BASE + "/api/health", timeout=5).status_code == 200, 600)
    r = U.requests.post(U.BASE + "/api/auth/login", json={"username": "unsloth", "password": PASSWORD}, timeout=30)
    r.raise_for_status()
    U.session.headers["Authorization"] = "Bearer " + r.json()["access_token"]
    return round(time.monotonic() - t, 1)


def stop():
    p = proc.get("p")
    if not p:
        return
    if U.WIN:
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(p.pid)], capture_output=True)
    else:
        os.killpg(p.pid, 15)
        try:
            p.wait(60)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, 9)
    time.sleep(5)


def chat(body, *, key=None, path="/v1/chat/completions", stream=False, ok=None):
    h = {"Authorization": "Bearer " + key} if key else {}
    s = U.requests if key else U.session
    t = time.monotonic()
    r = s.post(U.BASE + path, json=body, headers=h, timeout=600, stream=stream)
    if stream:
        chunks, text, usage, first = [], "", None, None
        for line in r.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                ev = json.loads(data)
            except ValueError:
                continue
            chunks.append(ev)
            for c in ev.get("choices") or []:
                d = c.get("delta") or {}
                piece = d.get("content") or ""
                if piece and first is None:
                    first = round(time.monotonic() - t, 2)
                text += piece
            usage = ev.get("usage") or usage
        return {"status": r.status_code, "chunks": len(chunks), "first_content_s": first, "text": text[-800:], "usage": usage,
                "seconds": round(time.monotonic() - t, 1)}
    try:
        b = r.json()
    except ValueError:
        b = r.text[:2000]
    return {"status": r.status_code, "seconds": round(time.monotonic() - t, 1), "body": b}


def main():
    step("start_studio_for_api", start, shoot=False)
    st = step("npu_status_after_restart", lambda: api("GET", "/api/npu/status")["body"], shoot=False) or {}
    cat = step("catalog", lambda: api("GET", "/api/npu/models")["body"]["models"], shoot=False) or []
    (U.ROOT / "catalog.json").write_text(json.dumps(cat))
    small = next((m for m in cat if m["checkpoint"] == "qwen3:0.6b"), None)
    vision = next((m for m in cat if m["checkpoint"] == "qwen3.5:2b"), None)
    if not small:
        return
    if not small["downloaded"]:
        step("download_small_via_api", lambda: [l for l in U.session.post(U.BASE + f"/api/npu/models/{small['id']}/download", stream=True, timeout=3600).iter_lines()][-3:], shoot=False)

    key = step("create_api_key", lambda: api("POST", "/api/auth/api-keys", json={"name": "ux-app"})["body"], shoot=False) or {}
    raw_key = next((v for v in key.values() if isinstance(v, str) and len(v) > 20), None) if isinstance(key, dict) else None

    # What does an app see before anything is loaded?
    step("v1_models_nothing_loaded", lambda: api("GET", "/v1/models")["body"], shoot=False)
    def by_name(model):
        return chat({"model": model, "messages": [{"role": "user", "content": "Say ok."}], "max_tokens": 20}, key=raw_key)
    for label, model in (("model_path", small["model_path"]), ("flm_name", "qwen3:0.6b"), ("id", small["id"])):
        r = step(f"chat_by_name_unloaded_{label}", lambda model=model: by_name(model), shoot=False)
        if r and r["status"] != 200:
            note(f"An API client asking for NPU model '{model}' while nothing is loaded gets HTTP {r['status']}: {json.dumps(r['body'])[:300]}")

    step("load_small_via_api", lambda: api("POST", "/api/inference/load", json={"model_path": small["model_path"]}, timeout=900)["body"], shoot=False)
    step("v1_models_loaded", lambda: api("GET", "/v1/models")["body"], shoot=False)

    base_msg = [{"role": "user", "content": "What is 17 + 25? Reply with just the number."}]
    step("chat_nonstream_api_key", lambda: chat({"model": small["model_path"], "messages": base_msg, "max_tokens": 400}, key=raw_key), shoot=False)
    step("chat_stream", lambda: chat({"model": small["model_path"], "messages": base_msg, "max_tokens": 400, "stream": True,
                                      "stream_options": {"include_usage": True}}, key=raw_key, stream=True), shoot=False)
    step("chat_model_field_default", lambda: chat({"model": "default", "messages": base_msg, "max_tokens": 400}, key=raw_key), shoot=False)
    step("chat_model_field_omitted", lambda: chat({"messages": base_msg, "max_tokens": 400}, key=raw_key), shoot=False)
    step("chat_reasoning_off", lambda: chat({"model": small["model_path"], "messages": base_msg, "max_tokens": 400,
                                            "chat_template_kwargs": {"enable_thinking": False}}, key=raw_key), shoot=False)
    step("chat_max_tokens_5", lambda: chat({"model": small["model_path"], "messages": [{"role": "user", "content": "Count from 1 to 50."}],
                                            "max_tokens": 5, "chat_template_kwargs": {"enable_thinking": False}}, key=raw_key), shoot=False)
    # Common client parameters (the OpenAI SDK, LangChain, Open WebUI send some of these by default).
    for label, extra in (("seed", {"seed": 1}), ("response_format_json", {"response_format": {"type": "json_object"}}),
                         ("frequency_penalty", {"frequency_penalty": 0.5}), ("n1", {"n": 1}), ("logprobs_false", {"logprobs": False}),
                         ("tools_on_non_tool_model", {"tools": [{"type": "function", "function": {"name": "get_time", "parameters": {"type": "object", "properties": {}}}}]})):
        r = step(f"param_{label}", lambda extra=extra: chat({"model": small["model_path"], "messages": base_msg, "max_tokens": 200, **extra}, key=raw_key), shoot=False)
        if r and r["status"] != 200:
            note(f"NPU chat rejects `{label}` with HTTP {r['status']}: {json.dumps(r['body'])[:240]}")

    # R1: does the small reasoning model finish a trivial question when nothing caps it?
    def r1(model_path, n):
        rows = []
        for _ in range(n):
            r = chat({"model": model_path, "messages": base_msg}, key=raw_key)
            b = r["body"] if isinstance(r["body"], dict) else {}
            ch = (b.get("choices") or [{}])[0]
            msg = ch.get("message") or {}
            rows.append({"status": r["status"], "seconds": r["seconds"], "finish": ch.get("finish_reason"),
                         "completion_tokens": (b.get("usage") or {}).get("completion_tokens"),
                         "content": (msg.get("content") or "")[-200:], "reasoning_tail": (msg.get("reasoning_content") or "")[-300:]})
        return rows
    step("r1_small_uncapped_x3", lambda: r1(small["model_path"], 3), shoot=False)
    def r1_stream_end():
        r = U.session.post(U.BASE + "/v1/chat/completions", json={"model": small["model_path"], "messages": [{"role": "user", "content": "Think very carefully, step by step, and list every prime number below 3000 with a one-line justification for each."}], "stream": True}, timeout=900, stream=True)
        lines = [l for l in r.iter_lines(decode_unicode=True) if l]
        return {"status": r.status_code, "n": len(lines), "last_lines": lines[-6:]}
    step("r1_stream_until_context_full", r1_stream_end, shoot=False)

    # Other API shapes a user's tools speak.
    anth = {"model": small["model_path"], "max_tokens": 200, "messages": [{"role": "user", "content": "Say ok."}]}
    for label, path, body in (("anthropic_messages", "/v1/messages", anth),
                              ("responses", "/v1/responses", {"model": small["model_path"], "input": "Say ok."}),
                              ("completions", "/v1/completions", {"model": small["model_path"], "prompt": "1, 2, 3,", "max_tokens": 10}),
                              ("embeddings", "/v1/embeddings", {"model": small["model_path"], "input": "hello"})):
        r = step(f"api_{label}", lambda path=path, body=body: chat(body, key=raw_key, path=path), shoot=False)
        if r and r["status"] != 200:
            note(f"With an NPU model loaded, {path} returns HTTP {r['status']}: {json.dumps(r['body'])[:240]}")

    # Ask for the other (downloaded) NPU model by name while the small one is loaded.
    if vision and vision["downloaded"]:
        r = step("chat_switch_to_other_npu_model_by_name", lambda: by_name(vision["model_path"]), shoot=False)
        if r and r["status"] != 200:
            note(f"Requesting another downloaded NPU model by name does not switch to it: HTTP {r['status']} {json.dumps(r['body'])[:240]}")
        step("load_vision_via_api", lambda: api("POST", "/api/inference/load", json={"model_path": vision["model_path"]}, timeout=900)["body"], shoot=False)
        tools = [{"type": "function", "function": {"name": "get_weather", "description": "Weather for a city",
                  "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
        step("tools_call", lambda: chat({"model": vision["model_path"], "messages": [{"role": "user", "content": "What's the weather in Paris? Use the tool."}],
                                         "tools": tools, "max_tokens": 600}, key=raw_key), shoot=False)
        import base64
        red = base64.b64encode((U.ROOT / "red.png").read_bytes()).decode() if (U.ROOT / "red.png").exists() else None
        if red:
            step("vision_api", lambda: chat({"model": vision["model_path"], "max_tokens": 600, "messages": [{"role": "user", "content": [
                {"type": "text", "text": "What color is this image? One word."},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + red}}]}]}, key=raw_key), shoot=False)
        step("unload_vision", lambda: api("POST", "/api/inference/unload", json={"model_path": vision["model_path"]}, ok=None)["body"], shoot=False)
        step("delete_vision_model", lambda: api("DELETE", f"/api/npu/models/{vision['id']}", ok=None)["status"], shoot=False)
    else:
        step("unload", lambda: api("POST", "/api/inference/unload", json={"model_path": small["model_path"]}, ok=None)["body"], shoot=False)
    step("npu_status_end", lambda: api("GET", "/api/npu/status")["body"], shoot=False)
    stop()
    print("NOTES:", json.dumps(U.NOTES, indent=1))


if __name__ == "__main__":
    try:
        main()
    finally:
        stop()
