"""Drive one Studio backend over its API: log in, load a GGUF, chat, record what happened."""
import argparse, glob, json, os, re, time, urllib.request, urllib.error

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--home", required=True)
ap.add_argument("--repo", required=True)
ap.add_argument("--variant", required=True)
ap.add_argument("--model-path", default=None, help="Load this local GGUF instead of repo + variant")
ap.add_argument("--out", required=True)
ap.add_argument("--pwfile", required=True)
a = ap.parse_args()
res = {"repo": a.repo, "variant": a.variant}


def call(method, path, body=None, token=None, timeout=60):
    req = urllib.request.Request(a.base + path, method=method,
                                 data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", **({"Authorization": f"Bearer {token}"} if token else {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode("utf-8", "replace")
            return r.status, (json.loads(raw) if raw.strip().startswith(("{", "[")) else raw)
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", "replace")
        try:
            return e.code, json.loads(raw)
        except Exception:
            return e.code, raw


def login():
    pw = open(a.pwfile).read().strip() if os.path.exists(a.pwfile) else None
    boot = os.path.join(a.home, "auth", ".bootstrap_password")
    for cand in [pw, open(boot).read().strip() if os.path.exists(boot) else None]:
        if not cand:
            continue
        st, r = call("POST", "/api/auth/login", {"username": "unsloth", "password": cand})
        if st == 200 and isinstance(r, dict) and r.get("access_token"):
            tok = r["access_token"]
            if r.get("must_change_password"):
                new = "DevlabPass-12345!"
                st2, r2 = call("POST", "/api/auth/change-password", {"current_password": cand, "new_password": new}, tok)
                res["password_change"] = st2
                open(a.pwfile, "w").write(new)
                tok = (r2 or {}).get("access_token", tok) if isinstance(r2, dict) else tok
            return tok
    raise SystemExit(f"login failed: {st} {r}")


tok = login()
logs_before = set(glob.glob(os.path.join(a.home, "logs", "llama-server", "*.log")))
body = {"model_path": a.model_path or a.repo, "max_seq_length": 4096,
        "speculative_type": "off", "disable_vision": True}
if not a.model_path:
    body["gguf_variant"] = a.variant
t0 = time.time()
st, r = call("POST", "/api/inference/load", body, tok, timeout=3600)
res["load_seconds"] = round(time.time() - t0, 1)
res["load_status"] = st
res["load_response"] = r if isinstance(r, dict) else str(r)[:4000]
st, status = call("GET", "/api/inference/status", None, tok)
res["status"] = status
model_id = (status or {}).get("active_model") if isinstance(status, dict) else None
if res["load_status"] == 200 and model_id:
    prompt = "Summarize the history of the printing press. " * 60
    for n, (msg, max_tokens) in enumerate([("Say hello in five words.", 32), (prompt, 128)]):
        t0 = time.time()
        st, r = call("POST", "/v1/chat/completions", {"model": model_id, "messages": [{"role": "user", "content": msg}],
                     "max_tokens": max_tokens, "temperature": 0, "stream": False}, tok, timeout=1800)
        dt = time.time() - t0
        res[f"chat{n}"] = {"status": st, "seconds": round(dt, 2),
                           "usage": (r or {}).get("usage") if isinstance(r, dict) else None,
                           "timings": (r or {}).get("timings") if isinstance(r, dict) else None,
                           "text": ((r or {}).get("choices") or [{}])[0].get("message", {}).get("content", "")[:200] if isinstance(r, dict) else str(r)[:500]}
new_logs = sorted(set(glob.glob(os.path.join(a.home, "logs", "llama-server", "*.log"))) - logs_before)
res["llama_logs"] = []
for p in new_logs:
    txt = open(p, errors="replace").read()
    keep = [l for l in txt.splitlines() if re.search(r"load_mode =|lazy read|failed to allocate|unable to allocate|error loading|out of memory|model buffer size|CPU_Mapped|Host model|ROCm0 model|Vulkan0 model|file size|memory breakdown|projected to use", l)]
    res["llama_logs"].append({"path": os.path.basename(p), "lines": keep[:60]})
call("POST", "/api/inference/unload", {"model_path": model_id or a.model_path or a.repo}, tok, timeout=600)
json.dump(res, open(a.out, "w"), indent=1, default=str)
print(json.dumps({k: res.get(k) for k in ("load_status", "load_seconds")}))
