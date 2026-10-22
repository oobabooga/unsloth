"""Does FastFlowLM (behind Studio's lemond) reuse the KV cache across turns? Sends a captured Hermes
turn 1 and several follow-ups, direct to lemond or through Studio's /v1, and records FLM's
prompt_tokens, cached_tokens and prefill time for each.

Run from studio/backend with the Studio venv python.
"""

import argparse
import copy
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx

p = argparse.ArgumentParser()
p.add_argument("--model", required = True)
p.add_argument("--ctx", type = int, default = 32768)
p.add_argument("--hermes", required = True)
p.add_argument("--out", required = True)
p.add_argument("--phase", required = True, choices = ["direct", "studio"])
p.add_argument("--setup", action = "store_true")
p.add_argument("--port", type = int, default = 8897)
args = p.parse_args()

OUT = Path(args.out)
OUT.mkdir(parents = True, exist_ok = True)
TAG = args.phase
client = httpx.Client(timeout = httpx.Timeout(1800, connect = 30), trust_env = False)
RESULTS = []


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush = True)


TOOL_RESULT = json.dumps(
    {"bytes_written": 21, "dirs_created": True, "lint": {"status": "ok", "output": ""},
     "resolved_path": "C:\\Users\\Admin\\project\\hello.py",
     "files_modified": ["C:\\Users\\Admin\\project\\hello.py"]}
)


def send(name, url, headers, body):
    """Stream one request; return the assistant message it produced and FLM's numbers."""
    t0 = time.time()
    content, reasoning, calls = "", "", {}
    usage, timings, err, status, t_first = None, None, None, None, None
    raw = (OUT / f"{TAG}-{name}.sse.txt").open("w", encoding = "utf-8")
    try:
        with client.stream("POST", url, json = body, headers = headers) as r:
            status = r.status_code
            if status != 200:
                err = r.read().decode("utf-8", "replace")[:3000]
            for line in r.iter_lines() if status == 200 else []:
                raw.write(f"{time.time() - t0:.3f} {line}\n")
                if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                    continue
                try:
                    obj = json.loads(line[5:])
                except ValueError:
                    continue
                if "error" in obj:
                    err = json.dumps(obj["error"])[:2000]
                for ch in obj.get("choices") or []:
                    d = ch.get("delta") or {}
                    if d.get("content"):
                        content += d["content"]
                    rc = d.get("reasoning_content") or d.get("reasoning")
                    if rc:
                        reasoning += rc
                    for tc in d.get("tool_calls") or []:
                        slot = calls.setdefault(tc.get("index", 0), {"id": None, "name": "", "arguments": ""})
                        slot["id"] = tc.get("id") or slot["id"]
                        fn = tc.get("function") or {}
                        slot["name"] += fn.get("name") or ""
                        slot["arguments"] += fn.get("arguments") or ""
                    if (d.get("content") or rc or d.get("tool_calls")) and t_first is None:
                        t_first = time.time()
                if obj.get("usage"):
                    usage = obj["usage"]
                if obj.get("timings"):
                    timings = obj["timings"]
    except Exception as exc:  # noqa: BLE001
        err = repr(exc)
    raw.close()
    u = usage or {}
    res = {
        "phase": TAG,
        "leg": name,
        "status": status,
        "error": err,
        "wall_s": round(time.time() - t0, 2),
        "ttft_s": round(t_first - t0, 2) if t_first else None,
        "prompt_tokens": u.get("prompt_tokens"),
        "cached_tokens": (u.get("prompt_tokens_details") or {}).get("cached_tokens"),
        "prefill_s": u.get("prefill_duration_ttft"),
        "active_kv_tokens": u.get("active_kv_tokens"),
        "completion_tokens": u.get("completion_tokens"),
        "timings_cache_n": (timings or {}).get("cache_n"),
        "n_messages": len(body["messages"]),
        "reasoning_chars": len(reasoning),
        "tool_calls": [c["name"] for c in calls.values()],
    }
    RESULTS.append(res)
    (OUT / f"{TAG}-results.json").write_text(json.dumps(RESULTS, indent = 1), encoding = "utf-8")
    log(f"[{name}] RESULT", json.dumps(res))
    calls_out = [
        {"id": c["id"] or f"call_{i}", "type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}}
        for i, c in sorted(calls.items())
    ]
    return {"content": content, "reasoning": reasoning, "tool_calls": calls_out}


def hermes_echo(gen):
    """The assistant turn as Hermes sends it back: whitespace-only content dropped to "", no
    reasoning, tool-call arguments re-encoded as compact sorted JSON (seen in a captured turn 2)."""
    msg = {"role": "assistant", "content": gen["content"].strip()}
    calls = []
    for c in gen["tool_calls"]:
        try:
            a = json.dumps(json.loads(c["function"]["arguments"]), separators = (",", ":"), sort_keys = True, ensure_ascii = False)
        except ValueError:
            a = c["function"]["arguments"]
        calls.append({"id": c["id"], "type": "function", "function": {"name": c["function"]["name"], "arguments": a}})
    if calls:
        msg["tool_calls"] = calls
    return msg


def exact_echo(gen):
    """The assistant turn exactly as generated, reasoning included."""
    msg = {"role": "assistant", "content": gen["content"]}
    if gen["reasoning"]:
        msg["reasoning_content"] = gen["reasoning"]
    if gen["tool_calls"]:
        msg["tool_calls"] = gen["tool_calls"]
    return msg


def tool_results(gen):
    return [
        {"role": "tool", "name": c["function"]["name"], "content": TOOL_RESULT, "tool_call_id": c["id"]}
        for c in gen["tool_calls"]
    ]


def scenario(url, headers, model):
    t1 = json.loads(Path(args.hermes).read_text(encoding = "utf-8"))
    t1["model"] = model

    def follow(gen, echo):
        body = copy.deepcopy(t1)
        body["messages"] += [echo(gen)] + (
            tool_results(gen) if gen["tool_calls"] else [{"role": "user", "content": "Thanks. Now explain it in one sentence."}]
        )
        return body

    log("warmup")
    send("warmup", url, headers, {"model": model, "messages": [{"role": "user", "content": "Say hi."}],
                                  "max_tokens": 16, "stream": True, "stream_options": {"include_usage": True}})
    gen = send("t1-cold", url, headers, t1)
    send("t2-hermes-echo", url, headers, follow(gen, hermes_echo))
    gen = send("t1-after-t2", url, headers, t1)
    send("t2-exact-echo", url, headers, follow(gen, exact_echo))
    send("t1-again", url, headers, t1)
    other = copy.deepcopy(t1)
    other["messages"][-1]["content"] = "List the files in the current directory."
    send("t1-other-user-msg", url, headers, other)

    # Plain chat with the same long prefix, no tools: turn 2 echoes the reply exactly.
    plain = copy.deepcopy(t1)
    tools = plain.pop("tools")
    plain["messages"][0]["content"] += "\n\n# Tool schemas (reference only)\n" + json.dumps(tools)
    plain["messages"][-1]["content"] = "In two sentences, what can you help me with?"
    gen = send("plain-t1", url, headers, plain)
    p2 = copy.deepcopy(plain)
    p2["messages"] += [exact_echo(gen), {"role": "user", "content": "Thanks. Now say it in one sentence."}]
    send("plain-t2-exact-echo", url, headers, p2)


def phase_direct():
    sys.path.insert(0, ".")
    from core.inference.npu_backend import get_npu_backend

    b = get_npu_backend()
    if args.setup:
        t = time.time()
        log("ENABLE", json.dumps(b.enable())[:1500], f"{time.time() - t:.0f}s")
        last = None
        t = time.time()
        for ev in b.download(args.model):
            if last is None or time.time() - t > 60:
                log("DOWNLOAD", ev)
                t = time.time()
            last = ev
        log("DOWNLOAD done", last)
    log("STATUS", json.dumps(b.status())[:1500])
    b.load(args.model, args.ctx)
    up = b.upstream()
    log("LOADED ctx", up.context_length)
    scenario(f"{up.base_url}/chat/completions", {"Authorization": f"Bearer {up.api_key}"}, up.model)
    b.unload()
    b.shutdown()


def phase_studio():
    home = Path(os.environ["UNSLOTH_STUDIO_HOME"])
    exe = Path(sys.executable).parent / ("unsloth.exe" if os.name == "nt" else "unsloth")
    base = f"http://127.0.0.1:{args.port}"
    srv_log = (OUT / "studio-server.log").open("w", encoding = "utf-8")
    proc = subprocess.Popen(
        [str(exe), "studio", "--api-only", "-H", "127.0.0.1", "-p", str(args.port)],
        stdout = srv_log, stderr = subprocess.STDOUT, cwd = str(OUT),
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
    )
    try:
        deadline = time.time() + 600
        while time.time() < deadline:
            try:
                if client.get(f"{base}/api/health", timeout = 5).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            if proc.poll() is not None:
                raise SystemExit(f"studio exited {proc.returncode}")
            time.sleep(2)
        password = (home / "auth" / ".bootstrap_password").read_text().strip()
        tok = client.post(f"{base}/api/auth/login", json = {"username": "unsloth", "password": password}).json()
        token = tok["access_token"]
        if tok.get("must_change_password"):
            r = client.post(
                f"{base}/api/auth/change-password",
                json = {"current_password": password, "new_password": "ProbePass-" + str(int(time.time()))},
                headers = {"Authorization": f"Bearer {token}"},
            )
            token = r.json()["access_token"]
        A = {"Authorization": f"Bearer {token}"}
        r = client.post(f"{base}/api/auth/api-keys", json = {"name": "cache-probe"}, headers = A)
        key = r.json().get("key") if r.status_code == 200 else None
        log("api-key", r.status_code, bool(key))
        K = {"Authorization": f"Bearer {key}"} if key else A
        t = time.time()
        r = client.post(f"{base}/api/inference/load",
                        json = {"model_path": f"lemonade:{args.model}", "max_seq_length": args.ctx}, headers = A)
        log("load", r.status_code, f"{time.time() - t:.0f}s", r.text.strip()[-300:])
        scenario(f"{base}/v1/chat/completions", K, f"lemonade:{args.model}")
    finally:
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output = True)
        time.sleep(10)


phase_direct() if args.phase == "direct" else phase_studio()
log("SUMMARY")
for r in RESULTS:
    print(f"{r['phase']:7s} {r['leg']:22s} prompt={r['prompt_tokens']} cached={r['cached_tokens']} "
          f"prefill={r['prefill_s']} wall={r['wall_s']} kv={r['active_kv_tokens']} reasoning={r['reasoning_chars']} "
          f"calls={r['tool_calls']} err={r['error']}", flush = True)
