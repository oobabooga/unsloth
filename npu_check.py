"""NPU stack check against a running Studio: download, Auto/pinned loads with flm_args, reuse, chat.

usage: python3 npu_check.py <base_url> <password> <out.json> <tag>
"""

import json
import sys
import time
import urllib.error
import urllib.request

BASE, PW, OUT, TAG = sys.argv[1:5]
MODEL = "qwen3-0.6b-FLM"
result = {"tag": TAG, "steps": []}


def call(path, body = None, token = None, method = None, timeout = 1800):
    req = urllib.request.Request(
        BASE + path,
        data = None if body is None else json.dumps(body).encode(),
        method = method or ("GET" if body is None else "POST"),
    )
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", "Bearer " + token)
    try:
        with urllib.request.urlopen(req, timeout = timeout) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")


def step(name, **fields):
    fields["name"] = name
    result["steps"].append(fields)
    print(json.dumps(fields)[:1500], flush = True)
    json.dump(result, open(OUT, "w"), indent = 1)


code, body = call("/api/auth/login", {"username": "unsloth", "password": PW})
tok = json.loads(body)["access_token"]
code, body = call("/api/npu/enable", {}, tok)
step("enable", code = code, body = body[:400])
code, body = call(f"/api/npu/models/{MODEL}", token = tok, method = "DELETE")
step("delete-before", code = code, body = body[:200])
t0 = time.time()
code, body = call(f"/api/npu/models/{MODEL}/download", {}, tok)
events = [json.loads(l[5:]) for l in body.splitlines() if l.startswith("data:")]
step(
    "download",
    code = code,
    seconds = round(time.time() - t0),
    n_events = len(events),
    complete_events = sum(e.get("event") == "complete" for e in events),
    tail = events[-4:],
)


def load(**fields):
    t = time.time()
    code, body = call(
        "/api/inference/load", {"model_path": "lemonade:" + MODEL, **fields}, tok
    )
    try:
        r = json.loads(body)
    except ValueError:
        r = {"raw": body[:400]}
    keep = {
        k: r.get(k)
        for k in (
            "status",
            "context_length",
            "max_context_length",
            "native_context_length",
            "npu_power_mode",
            "npu_prefill_chunk_len",
            "npu_img_pre_resize",
            "detail",
        )
    }
    code2, st = call("/api/inference/status", token = tok)
    try:
        s = json.loads(st)
        s = {k: v for k, v in s.items() if k.startswith("npu_") or k in ("context_length", "max_context_length", "active_model")}
    except ValueError:
        s = st[:300]
    code3, chat = call(
        "/v1/chat/completions",
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": "Say hello in five words."}],
            "max_tokens": 32,
            "stream": False,
        },
        tok,
    )
    try:
        text = json.loads(chat)["choices"][0]["message"]["content"][:120]
    except Exception:
        text = chat[:300]
    step(
        "load " + json.dumps(fields),
        code = code,
        seconds = round(time.time() - t),
        response = keep,
        status = s,
        chat_code = code3,
        chat = text,
    )


load(max_seq_length = 0, npu_power_mode = "turbo", npu_prefill_chunk_len = 16384)
load(max_seq_length = 0, npu_power_mode = "turbo", npu_prefill_chunk_len = 16384)
load(max_seq_length = 4096, npu_power_mode = "turbo", npu_prefill_chunk_len = 16384)
load(max_seq_length = 4096, npu_power_mode = "turbo", npu_prefill_chunk_len = 16384)
load(max_seq_length = 0, npu_prefill_chunk_len = 2048)
load(max_seq_length = 0)
print("DONE", TAG)
