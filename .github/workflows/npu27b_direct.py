"""Qwen3.8 27B on the NPU: flm serve driven directly, FLM #1045 (spec-sampling-qwen38) vs FLM main vs 1.0.12, identical requests.

Per request it records FLM's own usage (decoding_speed_tps = generated / decode time, prefill_duration_ttft:
the numbers Lemonade's telemetry line copies), the client-side wall clock, and the flm serve log lines the
request produced (speculation signal). Reps run in passes against a deadline so every cell gets n>=1."""
import http.client, json, os, random, re, subprocess, sys, time, urllib.request

MODEL = os.environ["MODEL"]                      # qwen3.8-mtp:27b | qwen3.8-dflash2:27b
TOOLS_ON = os.environ["TOOLS"] == "on"
OUT = os.environ["OUT"]
DEADLINE = float(os.environ["DEADLINE"])
BUILDS = [("pr1045", os.environ["FLM_SPEC"], os.environ["FLM_SPEC_COMMIT"]),
          ("main", os.environ["FLM_MAIN"], os.environ["FLM_MAIN_COMMIT"]),
          ("1.0.12", os.environ["FLM_1012"], os.environ["FLM_1012_COMMIT"])]
LENGTHS = [256, 512]
MAX_REPS = {256: 3, 512: 3}
PROMPT = ("Write a long, detailed technical essay of at least 2,000 words, with numbered sections and worked "
          "examples, explaining how a bicycle stays upright while moving: gyroscopic effects, trail and caster, "
          "steering geometry, the rider's own corrections, and the history of the experiments that tested each "
          "idea. Answer directly in plain text; do not call any tools.")
# What Unsloth Studio sends for a lemonade:* chat by default (thinking on, Qwen thinking table), minus min_p,
# which Studio drops for Lemonade. "temp0" is the same body with temperature 0.
STUDIO_DEFAULT = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5,
                  "repetition_penalty": 1.0, "think": True}
SAMPLINGS = [("temp0", {**STUDIO_DEFAULT, "temperature": 0}), ("default", STUDIO_DEFAULT)]
TOOLS = [
    {"type": "function", "function": {"name": "web_search", "description": "Search the web.",
     "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "python", "description": "Run Python code and return stdout.",
     "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}}},
]
SPEC_RE = re.compile(r"specul|accept|draft|dflash|Request done|verify|hit rate", re.I)
rows, seq = [], [0]
srv = {"build": None, "p": None, "port": None, "log": None, "starts": 0}
host = os.uname().nodename


def save():
    json.dump(rows, open(f"{OUT}/results.json", "w"), indent=1)


def flm_version(path):
    try:
        return subprocess.run([path, "--version"], capture_output=True, text=True, timeout=60).stdout.strip()[-300:]
    except Exception as e:
        return repr(e)


def stop():
    p = srv["p"]
    if p and p.poll() is None:
        os.killpg(p.pid, 15)
        try:
            p.wait(30)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, 9)
            p.wait(10)
    srv.update(build=None, p=None)
    time.sleep(3)


def start(build, path):
    stop()
    srv["starts"] += 1
    port = random.randint(18900, 18999)
    log = f"{OUT}/flm-serve-{build}-{srv['starts']:02d}.log"
    cmd = [path, "serve", MODEL, "--ctx-len", "8192", "--port", str(port), "--host", "127.0.0.1"]
    print("START", build, " ".join(cmd), flush=True)
    p = subprocess.Popen(cmd, stdout=open(log, "w"), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                         start_new_session=True, env=os.environ.copy())
    t0 = time.time()
    while time.time() - t0 < 900:
        if p.poll() is not None:
            break
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/api/tags", timeout=5).read()
            srv.update(build=build, p=p, port=port, log=log)
            print("READY", build, f"{time.time() - t0:.1f}s", flush=True)
            r = chat(32 // 2, "temp0", dict(SAMPLINGS[0][1]), False, warm=True)
            print("WARMUP", build, r.get("out"), r.get("tps_decode"), r.get("error"), flush=True)
            return True
        except Exception:
            time.sleep(2)
    print("SERVER FAILED", build, "rc", p.poll(), open(log, errors="replace").read()[-3000:], flush=True)
    srv.update(build=None, p=None)
    return False


def chat(max_tokens, sname, sampling, tools, warm=False):
    seq[0] += 1
    body = {"model": MODEL, "stream": True, "stream_options": {"include_usage": True}, "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": f"[request {seq[0]}] " + PROMPT}], **sampling}
    if tools:
        body["tools"] = TOOLS
    # No ignore_eos: FLM decodes one token at a time while it is set (eos_suppressed), so speculation never runs.
    # Thinking is on, so replies reach max_tokens anyway; out_ok records it.
    off = os.path.getsize(srv["log"])
    r = {"max_tokens": max_tokens, "ignore_eos": body.get("ignore_eos", False)}
    t0 = time.time()
    first = last = None
    usage, finish, text, content, reasoning = None, None, [], [], []
    try:
        c = http.client.HTTPConnection("127.0.0.1", srv["port"], timeout=3600)
        c.request("POST", "/v1/chat/completions", json.dumps(body), {"content-type": "application/json"})
        resp = c.getresponse()
        r["status"] = resp.status
        if resp.status != 200:
            r["error"] = resp.read().decode("utf-8", "replace")[:500]
        else:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    ch = json.loads(data)
                except ValueError:
                    continue
                if ch.get("usage"):
                    usage = ch["usage"]
                for choice in ch.get("choices") or []:
                    d = choice.get("delta") or {}
                    content.append(d.get("content") or "")
                    reasoning.append((d.get("reasoning_content") or "") + (d.get("reasoning") or ""))
                    piece = (d.get("content") or "") + (d.get("reasoning_content") or "") + (d.get("reasoning") or "")
                    if piece or d.get("tool_calls"):
                        now = time.time()
                        first = first or now
                        last = now
                        if len(text) < 400:
                            text.append(piece or json.dumps(d.get("tool_calls"))[:80])
                    finish = choice.get("finish_reason") or finish
        c.close()
    except Exception as e:
        r["status"] = 0
        r["error"] = repr(e)[:300]
    t_end = time.time()
    time.sleep(1.5)
    new = open(srv["log"], errors="replace").read()[off:]
    spec_lines = [l.strip()[:400] for l in new.splitlines() if SPEC_RE.search(l)]
    u = usage or {}
    out = u.get("completion_tokens")
    wall = (last - t0) if last else None
    r.update({
        "out": out, "out_ok": out == max_tokens, "prompt_tokens": u.get("prompt_tokens"),
        "cached_tokens": (u.get("prompt_tokens_details") or {}).get("cached_tokens"),
        "tps_decode": u.get("decoding_speed_tps"), "decoding_duration": u.get("decoding_duration"),
        "ttft_server": u.get("prefill_duration_ttft"), "ttft_client": (first - t0) if first else None,
        "wall": wall, "wall_to_done": t_end - t0, "tps_wall": (out / wall) if out and wall else None,
        "finish_reason": finish, "reply_head": "".join(text)[:300],
        "spec_enabled_line": any("Speculative decoding enabled" in l for l in spec_lines),
        "spec_lines": spec_lines[-12:],
        "reply_full": {"reasoning": "".join(reasoning), "content": "".join(content)},
    })
    if warm:
        r.update(warmup=True, build=srv["build"], sampling_name=sname)
        rows.append(r)
        save()
    return r


def cell(build, path, commit, length, sname, sampling, rep, pas):
    est = length / ((guess.get((build, length, sname)) or 0.25)) + 40
    if time.time() + est > DEADLINE:
        print("SKIP (deadline)", build, length, sname, rep, f"est {est:.0f}s", flush=True)
        rows.append({"skipped": "deadline", "build": build, "max_tokens": length, "sampling_name": sname, "rep": rep})
        save()
        return
    if srv["build"] != build or srv["p"] is None or srv["p"].poll() is not None:
        if not start(build, path):
            rows.append({"skipped": "server failed", "build": build, "max_tokens": length, "sampling_name": sname, "rep": rep})
            save()
            return
    r = chat(length, sname, dict(sampling), TOOLS_ON)
    if srv["p"].poll() is not None:
        r["server_died"] = True
        srv.update(build=None, p=None)
    r.update({"model": MODEL, "build": build, "flm_commit": commit, "flm_version": versions[build], "host": host,
              "sampling_name": sname, "sampling": sampling, "tools": TOOLS_ON, "rep": rep, "pass": pas,
              "path": "flm serve direct", "t": time.time()})
    if r.get("tps_wall"):
        guess[(build, length, sname)] = r["tps_wall"]
        if r.get("tps_decode"):
            rate[(build, length, sname)] = r["tps_decode"]
    rows.append(r)
    save()
    print("ROW", json.dumps({k: r.get(k) for k in ("build", "sampling_name", "max_tokens", "rep", "status", "out",
          "tps_decode", "tps_wall", "ttft_server", "wall", "spec_enabled_line", "error")}), flush=True)
    for l in r["spec_lines"][-3:]:
        print("   ", l, flush=True)


versions = {b: flm_version(p) for b, p, _ in BUILDS}
print("VERSIONS", versions, flush=True)
guess, rate = {}, {}
# Pass A: one rep of every cell, then reps 2 and 3 of every cell, against the deadline.
for b, p, c in BUILDS:
    for L in LENGTHS:
        for sn, s in SAMPLINGS:
            cell(b, p, c, L, sn, s, 0, "A")
for rep in (1, 2):
    for b, p, c in BUILDS:
        for L in LENGTHS:
            if rep >= MAX_REPS[L]:
                continue
            for sn, s in SAMPLINGS:
                cell(b, p, c, L, sn, s, rep, "B")
stop()
print("DONE rows", len(rows), "time left", int(DEADLINE - time.time()), "s", flush=True)
