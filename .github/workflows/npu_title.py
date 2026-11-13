"""PR 17532 real-NPU check. Was: real-NPU A/B of the NPU runtime optimizations: tools on vs off (#17530), a title request between
turns (#17532), power mode and prefill chunk (#17510). Times come from Lemonade's own telemetry."""
import glob, json, os, re, statistics, time, urllib.error, urllib.request

W, B, PW = os.environ["W"], f"http://127.0.0.1:{os.environ['PORT']}", os.environ["PW"]
OUT = os.environ["OUT"]
PROMPT = "Explain how a bicycle stays upright while moving, in about 200 words."
PARA = ("The history of the bicycle spans two centuries of small changes: the draisine had no pedals, "
        "the velocipede put cranks on the front wheel, the penny-farthing grew that wheel for speed, and the "
        "safety bicycle brought chain drive, equal wheels and pneumatic tyres. ")
TOOLS = [
    {"type": "function", "function": {"name": "web_search", "description": "Search the web.",
     "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "python", "description": "Run Python code and return stdout.",
     "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}}},
]
results = []


def call(method, path, body=None, tok=None, timeout=3000):
    req = urllib.request.Request(B + path, method=method, data=json.dumps(body).encode() if body is not None else None)
    req.add_header("content-type", "application/json")
    if tok:
        req.add_header("Authorization", "Bearer " + tok)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")
    except Exception as e:
        return 0, repr(e)


def telemetry():
    out = []
    for f in sorted(glob.glob(f"{W}/studio/logs/lemonade/lemonade-*.log")):
        for l in open(f, errors="replace"):
            m = re.search(r"Inference completed: model=(\S+), tokens=(\d+) \(in=(\d+), out=(\d+)\), ttft=([\d.]+)s, tps=([\d.]+)", l)
            if m:
                out.append({"in": int(m.group(3)), "out": int(m.group(4)), "ttft": float(m.group(5)), "tps": float(m.group(6))})
    return out


def login():
    return json.loads(call("POST", "/api/auth/login", {"username": "unsloth", "password": PW})[1])["access_token"]


def chat(tok, mid, messages, max_tokens, tools=None):
    body = {"model": "lemonade:" + mid, "max_tokens": max_tokens, "stream": True, "temperature": 0, "messages": messages}
    req = urllib.request.Request(B + "/v1/chat/completions", method="POST", data=json.dumps(body).encode())
    req.add_header("content-type", "application/json")
    req.add_header("Authorization", "Bearer " + tok)
    t0 = time.time()
    ttft, parts, status = None, [], 0
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            status = r.status
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:") or line == "data: [DONE]":
                    continue
                try:
                    delta = json.loads(line[5:])["choices"][0].get("delta") or {}
                except (ValueError, KeyError, IndexError):
                    continue
                piece = delta.get("content") or delta.get("reasoning_content") or ""
                if piece and ttft is None:
                    ttft = time.time() - t0
                parts.append(delta.get("content") or "")
    except urllib.error.HTTPError as e:
        status = e.code
        print("CHAT FAILED", mid, e.code, e.read()[:200], flush=True)
    except Exception as e:
        print("CHAT FAILED", mid, repr(e)[:200], flush=True)
    wall = time.time() - t0
    return {"status": status, "wall": round(wall, 2), "tele": {"ttft": round(ttft, 3)} if ttft else None,
            "reply": "".join(parts)}


def load(tok, mid, **extra):
    body = {"model_path": "lemonade:" + mid, "max_seq_length": 16384, "force_reload": True, **extra}
    t0 = time.time()
    c, b = call("POST", "/api/inference/load", body, tok)
    try:
        d = json.loads(b)
    except ValueError:
        d = {}
    ok = c == 200 and not d.get("_deferred_error") and d.get("is_npu")
    print("LOAD", mid, extra, c, "ok" if ok else b[:300], f"{time.time() - t0:.1f}s ctx={d.get('context_length')}", flush=True)
    return ok, round(time.time() - t0, 1), d.get("context_length")


def rec(model, test, variant, rep, r):
    row = {"model": model, "test": test, "variant": variant, "rep": rep, **{k: r[k] for k in ("status", "wall")},
           **({f"lemonade_{k}": v for k, v in r["tele"].items()} if r["tele"] else {})}
    results.append(row)
    print("ROW", json.dumps(row), flush=True)
    json.dump(results, open(OUT, "w"), indent=1)



def chat_fresh(mid, messages, max_tokens):
    return chat(login(), mid, messages, max_tokens)


tok = login()
print("enable:", call("POST", "/api/npu/enable", {}, tok)[0], flush=True)
st = json.loads(call("GET", "/api/npu/status", tok=tok)[1])
print("VERSIONS:", st.get("versions"), "hardware:", st.get("hardware"), flush=True)
models = {m["checkpoint"]: m for m in json.loads(call("GET", "/api/npu/models", tok=tok)[1])["models"]}

for cp in os.environ["CHECKPOINTS"].split():
    m = models.get(cp)
    if not m:
        print("NOT IN CATALOG:", cp, flush=True)
        continue
    mid = m["id"]
    tok = login()
    c, b = call("POST", f"/api/npu/models/{mid}/download", {}, tok)
    if c != 200 or '"complete"' not in b[-400:]:
        print("DOWNLOAD FAILED", mid, c, b[-200:], flush=True)
        continue
    ok, secs, ctx = load(login(), mid)
    if not ok:
        continue
    chat_fresh(mid, [{"role": "user", "content": "Hi"}], 8)
    for rep in range(int(os.environ.get("REPS", "4"))):
        for variant in ("no request", "title request", "other long chat"):
            nonce = f"[chat {rep}-{variant[:5]}-{time.time():.0f}] "
            long_msg = nonce + PARA * 40 + "\nSummarize the history above in three bullet points."
            msgs = [{"role": "user", "content": long_msg}]
            r1 = chat_fresh(mid, msgs, 96)
            rec(mid, "between", variant + ": turn 1", rep, r1)
            if variant == "title request":
                rec(mid, "between", "title request itself", rep, chat_fresh(
                    mid, [{"role": "user", "content": "Give a 4 word title for this chat: " + long_msg[:300]}], 16))
            elif variant == "other long chat":
                chat_fresh(mid, [{"role": "user", "content": "[other] " + PARA * 40 + "\nOne sentence."}], 16)
            msgs += [{"role": "assistant", "content": r1["reply"] or "Done."},
                     {"role": "user", "content": "Now say which change mattered most, in one sentence."}]
            rec(mid, "between", variant + ": turn 2", rep, chat_fresh(mid, msgs, 64))
    call("POST", "/api/inference/unload", {"model_path": "lemonade:" + mid}, login())
    time.sleep(3)
    print("delete", mid, call("DELETE", f"/api/npu/models/{mid}", tok=login())[0], flush=True)

print("\n| model | variant | median ttft s | median wall s | n |\n|---|---|---|---|---|")
groups = {}
for r in results:
    if "lemonade_ttft" in r:
        groups.setdefault((r["model"], r["variant"]), []).append(r)
for (mo, va), rs in groups.items():
    print(f"| {mo} | {va} | {statistics.median(x['lemonade_ttft'] for x in rs):.2f} | "
          f"{statistics.median(x['wall'] for x in rs):.2f} | {len(rs)} |")
