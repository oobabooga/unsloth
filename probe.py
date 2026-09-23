"""Throwaway DevLab probe: portable lemond + FLM recipe on the Ryzen AI NPU."""
import json, os, platform, shutil, subprocess, sys, tarfile, time, urllib.request, urllib.error, zipfile

VER = os.environ.get("LEMONADE_VER", "11.9.0")
WIN = platform.system() == "Windows"
ROOT = os.path.join(os.environ.get("RUNNER_TEMP", os.getcwd()), "lemo")
PORT = int(os.environ.get("LEMO_PORT", "13399"))
KEY = "probe-key-123"
BASE = f"http://127.0.0.1:{PORT}"
OUT = {}


def rec(k, v):
    OUT[k] = v
    print(f"=== {k}\n{json.dumps(v, indent=1)[:6000]}", flush=True)


def req(method, path, body=None, timeout=1800, auth=True, raw=False):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(BASE + path, data=data, method=method)
    if auth:
        r.add_header("Authorization", f"Bearer {KEY}")
    if data:
        r.add_header("Content-Type", "application/json")
    t = time.time()
    try:
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            txt = resp.read().decode("utf-8", "replace")
            st = resp.status
    except urllib.error.HTTPError as e:
        txt = e.read().decode("utf-8", "replace")
        st = e.code
    except Exception as e:
        return {"error": repr(e), "secs": round(time.time() - t, 2)}
    dt = round(time.time() - t, 2)
    if raw:
        return {"status": st, "secs": dt, "text": txt[:4000]}
    try:
        j = json.loads(txt)
    except Exception:
        j = txt[:4000]
    return {"status": st, "secs": dt, "json": j}


def stream_chat(model, body):
    body = dict(body, model=model, stream=True, stream_options={"include_usage": True})
    r = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(), method="POST")
    r.add_header("Authorization", f"Bearer {KEY}")
    r.add_header("Content-Type", "application/json")
    t0 = time.time(); ttft = None; chunks = 0; text = ""; reasoning = ""; lines = []; usage = None; finish = None
    try:
        with urllib.request.urlopen(r, timeout=600) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line:
                    continue
                if len(lines) < 6:
                    lines.append(line[:400])
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    lines.append("[DONE]")
                    break
                j = json.loads(payload)
                if j.get("usage"):
                    usage = j["usage"]
                for ch in j.get("choices", []):
                    d = ch.get("delta", {})
                    c = d.get("content") or ""
                    rc = d.get("reasoning_content") or ""
                    if (c or rc) and ttft is None:
                        ttft = time.time() - t0
                    text += c; reasoning += rc; chunks += 1
                    finish = ch.get("finish_reason") or finish
    except Exception as e:
        return {"error": repr(e), "first_lines": lines}
    return {"ttft": ttft and round(ttft, 3), "secs": round(time.time() - t0, 2), "chunks": chunks,
            "text": text[:1500], "reasoning": reasoning[:800], "usage": usage, "finish": finish, "first_lines": lines}


def fetch_lemond():
    os.makedirs(ROOT, exist_ok=True)
    name = f"lemonade-embeddable-{VER}-" + ("windows-x64.zip" if WIN else "ubuntu-x64.tar.gz")
    url = f"https://github.com/lemonade-sdk/lemonade/releases/download/v{VER}/{name}"
    dst = os.path.join(ROOT, name)
    t = time.time()
    urllib.request.urlretrieve(url, dst)
    if WIN:
        zipfile.ZipFile(dst).extractall(ROOT)
    else:
        tarfile.open(dst).extractall(ROOT)
    exe = "lemond.exe" if WIN else "lemond"
    for d, _, files in os.walk(ROOT):
        if exe in files:
            rec("fetch", {"url": url, "secs": round(time.time() - t, 1), "size_mb": os.path.getsize(dst) >> 20,
                          "dir": d, "files": sorted(os.listdir(d))})
            return d, os.path.join(d, exe)
    raise SystemExit("lemond not found")


def run(cmd, timeout=300, env=None):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
        return {"rc": p.returncode, "out": p.stdout[-4000:], "err": p.stderr[-2000:]}
    except Exception as e:
        return {"error": repr(e)}


def main():
    d, exe = fetch_lemond()
    env = dict(os.environ, LEMONADE_API_KEY=KEY)
    log = open(os.path.join(ROOT, "lemond.log"), "w")
    t = time.time()
    proc = subprocess.Popen([exe, d, "--port", str(PORT)], cwd=d, env=env, stdout=log, stderr=subprocess.STDOUT)
    up = None
    for _ in range(240):
        r = req("GET", "/live", auth=False, timeout=5)
        if r.get("status") == 200:
            up = round(time.time() - t, 2); break
        if proc.poll() is not None:
            break
        time.sleep(0.5)
    rec("startup", {"secs_to_live": up, "exit": proc.poll()})
    try:
        rec("unauth_models", req("GET", "/v1/models", auth=False, raw=True))
        rec("health0", req("GET", "/v1/health"))
        si = req("GET", "/v1/system-info")
        rec("system_info", si)
        models = req("GET", "/v1/models?show_all=true")
        allm = models.get("json", {}).get("data", []) if isinstance(models.get("json"), dict) else []
        flm = [m for m in allm if m.get("recipe") == "flm"]
        rec("recipe_counts", {r: sum(1 for m in allm if m.get("recipe") == r) for r in sorted({m.get("recipe") for m in allm})})
        rec("flm_models", [{k: m.get(k) for k in ("id", "checkpoint", "size", "labels", "downloaded", "max_context_window")} for m in flm])
        rec("sample_model_entry", flm[0] if flm else (allm[0] if allm else None))
        rec("install_dry", req("POST", "/v1/install/dry-run", {"recipe": "flm", "backend": "npu"}))
        rec("install_flm", req("POST", "/v1/install", {"recipe": "flm", "backend": "npu", "stream": False}))
        flm_exe = None
        for dd, _, files in os.walk(os.path.join(d, "bin")):
            for f in files:
                if f in ("flm", "flm.exe"):
                    flm_exe = os.path.join(dd, f)
        rec("flm_exe", {"path": flm_exe, "tree": [os.path.relpath(os.path.join(a, f), d) for a, _, fs in os.walk(os.path.join(d, "bin")) for f in fs][:80]})
        if flm_exe:
            rec("flm_version", run([flm_exe, "--version"]))
            rec("flm_validate", run([flm_exe, "validate", "--json"]))
            rec("flm_validate_text", run([flm_exe, "validate"]))
            rec("flm_list", run([flm_exe, "list", "--json", "--quiet"]))
        rec("system_info_after", req("GET", "/v1/system-info").get("json", {}).get("recipes", {}).get("flm"))
        want = os.environ.get("PROBE_MODELS", "")
        names = [m["id"] for m in flm]
        pick = [n for n in want.split(",") if n in names] if want else []
        if not pick and flm:
            pick = [min(flm, key=lambda m: m.get("size") or 1e9)["id"]]
        for m in pick:
            res = {}
            res["pull"] = req("POST", "/v1/pull", {"model_name": m})
            res["load"] = req("POST", "/v1/load", {"model_name": m, "ctx_size": 4096})
            res["health"] = req("GET", "/v1/health")
            res["chat"] = req("POST", "/v1/chat/completions", {"model": m, "messages": [{"role": "user", "content": "Count from 1 to 30, comma separated."}], "max_tokens": 200, "temperature": 0})
            res["stats1"] = req("GET", "/v1/stats")
            res["stream"] = stream_chat(m, {"messages": [{"role": "user", "content": "Write a 150-word story about a lighthouse."}], "max_tokens": 300, "temperature": 0.7, "top_p": 0.9})
            res["stats2"] = req("GET", "/v1/stats")
            res["stop"] = req("POST", "/v1/chat/completions", {"model": m, "messages": [{"role": "user", "content": "Count from 1 to 30, comma separated."}], "max_tokens": 200, "stop": ["7"]})
            res["tools"] = req("POST", "/v1/chat/completions", {"model": m, "messages": [{"role": "user", "content": "What's the weather in Paris? Use the tool."}], "max_tokens": 300,
                                "tools": [{"type": "function", "function": {"name": "get_weather", "description": "Get weather for a city", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]})
            res["completions"] = req("POST", "/v1/completions", {"model": m, "prompt": "The capital of France is", "max_tokens": 16, "temperature": 0})
            res["anthropic"] = req("POST", "/v1/messages", {"model": m, "max_tokens": 64, "messages": [{"role": "user", "content": "Say hi."}]})
            res["system_stats"] = req("GET", "/v1/system-stats")
            res["unload"] = req("POST", "/v1/unload", {"model_name": m})
            res["health_after"] = req("GET", "/v1/health")
            rec(f"model:{m}", res)
    finally:
        proc.terminate()
        try:
            proc.wait(20)
        except Exception:
            proc.kill()
        log.close()
        txt = open(os.path.join(ROOT, "lemond.log"), errors="replace").read()
        print("=== lemond.log (tail)\n" + txt[-15000:])
        with open(os.path.join(ROOT, "probe.json"), "w") as f:
            json.dump(OUT, f, indent=1)


if __name__ == "__main__":
    main()
