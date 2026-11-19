"""NPU speed through Unsloth Studio: decode tok/s and TTFT per FastFlowLM model, measured at the client."""
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE, OUT = os.environ["BASE"], Path(os.environ["OUT"])
MODELS = os.environ["MODELS"].split()
RUNS = int(os.environ.get("RUNS", "3"))
OUT.mkdir(parents = True, exist_ok = True)

SHORT = "Explain in about 300 words how a heat pump warms a house in winter. Plain prose, no lists."
LONG_BODY = (
    "The river town kept its records in a single ledger. Each spring the clerk copied the flood marks, the price of "
    "grain, the names of new families and the boats that came up from the coast. "
)
LONG = LONG_BODY * 60 + "\n\nIn two sentences, what kinds of things did the ledger record?"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush = True)


def raw(method, path, body = None, token = None, timeout = 60):
    req = urllib.request.Request(BASE + path, json.dumps(body).encode() if body is not None else None,
                                 {"Content-Type": "application/json"}, method = method)
    if token:
        req.add_header("Authorization", "Bearer " + token)
    try:
        resp = urllib.request.urlopen(req, timeout = timeout)
        code, text = resp.status, resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        code, text = exc.code, exc.read().decode("utf-8", "replace")
    try:
        data = json.loads(text)
    except ValueError:
        data = {"raw": text[:500]}
    if isinstance(data, dict):
        for v in data.values():  # /load pads past 15 s; the real status is in the body
            if isinstance(v, dict) and set(v) >= {"status_code", "detail"}:
                return v["status_code"], {"detail": v["detail"]}
    return code, data


TOKEN = raw("POST", "/api/auth/login", {"username": "unsloth", "password": os.environ["PW"]})[1]["access_token"]


def pull(mid):
    req = urllib.request.Request(BASE + f"/api/npu/models/{mid}/download", None,
                                 {"Authorization": "Bearer " + TOKEN}, method = "POST")
    for line in urllib.request.urlopen(req, timeout = 7200):
        line = line.decode("utf-8", "replace").strip()
        if line.startswith("data:"):
            ev = json.loads(line[5:])
            if ev.get("event") in ("error", "cancelled"):
                raise RuntimeError(json.dumps(ev)[:300])
            if ev.get("event") == "complete":
                return


def chat(mid, content, max_tokens):
    body = {"model": mid, "messages": [{"role": "user", "content": content}], "max_tokens": max_tokens,
            "temperature": 0, "stream": True, "stream_options": {"include_usage": True}}
    req = urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json", "Authorization": "Bearer " + TOKEN}, method = "POST")
    t0 = time.time()
    first = last = usage = timings = None
    pieces = 0
    try:
        resp = urllib.request.urlopen(req, timeout = 1200)
    except urllib.error.HTTPError as exc:
        return {"ok": False, "error": f"{exc.code} " + exc.read().decode("utf-8", "replace")[:300]}
    for line in resp:
        line = line.decode("utf-8", "replace").strip()
        if not line.startswith("data:") or line.endswith("[DONE]"):
            continue
        try:
            ev = json.loads(line[5:])
        except ValueError:
            continue
        if ev.get("error"):
            return {"ok": False, "error": json.dumps(ev["error"])[:300]}
        usage = ev.get("usage") or usage
        timings = ev.get("timings") or timings
        for ch in ev.get("choices") or []:
            d = ch.get("delta") or {}
            if (d.get("content") or "") + (d.get("reasoning_content") or ""):
                last = time.time()
                first = first or last
                pieces += 1
    r = {"ok": first is not None, "ttft_s": round(first - t0, 3) if first else None, "pieces": pieces}
    if usage:
        r["prompt_tokens"], r["completion_tokens"] = usage.get("prompt_tokens"), usage.get("completion_tokens")
    n = r.get("completion_tokens") or pieces
    if first and last and last > first and n > 1:
        r["decode_tok_s"] = round((n - 1) / (last - first), 2)
    if isinstance(timings, dict):
        r["server"] = {k: timings[k] for k in ("predicted_per_second", "prompt_per_second", "prompt_ms", "predicted_n", "prompt_n") if k in timings}
    return r


st = raw("GET", "/api/npu/status", token = TOKEN)[1]
log("npu status", json.dumps({k: st.get(k) for k in ("supported", "ready", "state", "hardware", "versions")}))
if not st.get("ready"):
    log("enable", json.dumps(raw("POST", "/api/npu/enable", {}, TOKEN, timeout = 1800)[1])[:400])
    st = raw("GET", "/api/npu/status", token = TOKEN)[1]
catalog = {m["id"]: m for m in raw("GET", "/api/npu/models", token = TOKEN, timeout = 300)[1]["models"]}
results = {"status": {k: st.get(k) for k in ("hardware", "versions")}, "runs": RUNS, "models": {}}

for mid in MODELS:
    row = catalog.get(mid)
    rec = results["models"][mid] = {"size_gb": row and row.get("size_gb")}
    if row is None:
        rec["error"] = "not in catalog"
        continue
    try:
        if not row.get("downloaded"):
            t = time.time()
            pull(mid)
            rec["download_s"] = round(time.time() - t)
        t = time.time()
        code, resp = raw("POST", "/api/inference/load", {"model_path": "lemonade:" + mid, "max_seq_length": 8192}, TOKEN, timeout = 1500)
        rec["load_s"], rec["load_http"] = round(time.time() - t, 1), code
        if code != 200:
            rec["error"] = json.dumps(resp)[:300]
            log(mid, "load failed", rec["error"])
            continue
        rec["warmup"] = chat("lemonade:" + mid, "Say hi in five words.", 32)
        rec["short"] = [chat("lemonade:" + mid, SHORT, 400) for _ in range(RUNS)]
        rec["long"] = [chat("lemonade:" + mid, LONG, 64) for _ in range(2)]
        ok = [x["decode_tok_s"] for x in rec["short"] if x.get("decode_tok_s")]
        rec["decode_tok_s_median"] = sorted(ok)[len(ok) // 2] if ok else None
        rec["ttft_short_s"] = sorted(x["ttft_s"] for x in rec["short"] if x.get("ttft_s"))[len(rec["short"]) // 2] if ok else None
        lt = [x["ttft_s"] for x in rec["long"] if x.get("ttft_s")]
        rec["ttft_long_s"] = min(lt) if lt else None
        rec["long_prompt_tokens"] = rec["long"][0].get("prompt_tokens")
        log(mid, "decode", rec["decode_tok_s_median"], "tok/s, ttft short", rec["ttft_short_s"], "long",
            rec["ttft_long_s"], f"({rec['long_prompt_tokens']} tok)")
    except Exception as exc:  # keep going; one model failing shouldn't sink the chart
        rec["error"] = repr(exc)[:300]
        log(mid, "failed", rec["error"])
    finally:
        raw("POST", "/api/inference/unload", {"model_path": "lemonade:" + mid}, TOKEN, timeout = 300)
        (OUT / "bench.json").write_text(json.dumps(results, indent = 2))

log("done")
print(json.dumps({m: {k: r.get(k) for k in ("decode_tok_s_median", "ttft_short_s", "ttft_long_s", "long_prompt_tokens", "error")}
                  for m, r in results["models"].items()}, indent = 1))
sys.exit(0 if any(r.get("decode_tok_s_median") for r in results["models"].values()) else 1)
