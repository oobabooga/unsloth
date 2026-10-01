"""PR 11591 end-to-end probe: a real Studio install serving two GGUFs at once.

Usage: python multi_model_probe.py --bin <home>/unsloth_studio/bin/unsloth --home <home> [--port 8931]
Prints one `PROBE_RESULT {...}` line; exit 0 only when every check passed.
Checks: default load, alongside load keeps the first, each model answers by name, /status lists both,
unloading one leaves the other answering, a plain (non-alongside) load replaces what is loaded.
"""

import argparse
import json
import re
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

A = ("unsloth/Qwen3-0.6B-GGUF", "Q8_0")
B = ("unsloth/gemma-3-270m-it-GGUF", "Q8_0")


def call(base, path, body = None, token = None, timeout = 900, method = None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(base + path, data = data, method = method or ("POST" if data else "GET"))
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout = timeout) as r:
            return r.status, json.loads(r.read() or b"null")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors = "replace")[:600]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required = True)
    ap.add_argument("--home", required = True)
    ap.add_argument("--port", type = int, default = 8931)
    ap.add_argument("--out", default = None)
    a = ap.parse_args()
    base = f"http://127.0.0.1:{a.port}"
    log = open(os.path.join(a.home, "probe_studio.log"), "w")
    env = {**os.environ, "UNSLOTH_STUDIO_HOME": a.home}
    proc = subprocess.Popen(
        [a.bin, "studio", "-p", str(a.port)], stdout = log, stderr = subprocess.STDOUT,
        env = env, start_new_session = True,
    )
    checks, info = {}, {}

    def check(name, ok, detail = ""):
        checks[name] = {"ok": bool(ok), "detail": str(detail)[:400]}
        print(("PASS " if ok else "FAIL ") + name + (f": {detail}" if detail else ""), flush = True)
        return ok

    try:
        deadline = time.time() + 600
        while time.time() < deadline:
            try:
                if call(base, "/api/health", timeout = 5)[0] == 200:
                    break
            except Exception:
                pass
            time.sleep(2)
        else:
            raise RuntimeError("Studio never answered /api/health")
        boot = os.path.join(a.home, "auth", ".bootstrap_password")
        t_boot = time.time() + 30
        while not os.path.exists(boot) and time.time() < t_boot:
            time.sleep(1)
        new_pw = "probe-11591-pass"
        if os.path.exists(boot):
            pw = open(boot).read().strip()
            st, tok = call(base, "/api/auth/login", {"username": "unsloth", "password": pw})
            assert st == 200, (st, tok)
            call(base, "/api/auth/change-password", {"current_password": pw, "new_password": new_pw},
                 tok["access_token"])
        st, tok = call(base, "/api/auth/login", {"username": "unsloth", "password": new_pw})
        assert st == 200, (st, tok)
        tok = tok["access_token"]

        studio_log = os.path.join(a.home, "probe_studio.log")

        def log_size():
            try:
                return os.path.getsize(studio_log)
            except OSError:
                return 0

        def load(m, alongside):
            # Studio's own placement line for this load: "GPUs free: [...], selected: [...]".
            before = log_size()
            t = time.time()
            st, r = call(base, "/api/inference/load",
                         {"model_path": m[0], "gguf_variant": m[1], "alongside": alongside}, tok, timeout = 1800)
            info[f"load_{m[0]}_{alongside}_s"] = round(time.time() - t, 1)
            with open(studio_log, encoding = "utf-8", errors = "replace") as fh:
                fh.seek(before)
                new = fh.read()
            info.setdefault("placements", []).append({
                "model": m[0], "alongside": alongside,
                "lines": re.findall(r"GPUs free: .*?selected: (?:\[[^\]]*\]|None)", new)[-3:],
            })
            return st, r

        def chat(m):
            t = time.time()
            st, r = call(base, "/v1/chat/completions", {
                "model": m[0], "messages": [{"role": "user", "content": "Say hello in five words."}],
                "max_tokens": 32, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False},
            }, tok, timeout = 600)
            ok = st == 200 and isinstance(r, dict) and r["choices"][0]["message"].get("content")
            return ok, (r.get("model") if isinstance(r, dict) else r), round(time.time() - t, 2)

        def status(m = None):
            q = "" if m is None else "?model=" + urllib.request.quote(m[0])
            return call(base, "/api/inference/status" + q, token = tok)[1]

        st, r = load(A, False)
        check("load_A", st == 200, r if st != 200 else "")
        st, r = load(B, True)
        check("load_B_alongside", st == 200, r if st != 200 else r.get("evicted"))
        s = status()
        info["status_after_B"] = {k: s.get(k) for k in ("active_model", "loaded")} if isinstance(s, dict) else s
        names = json.dumps(info["status_after_B"])
        check("status_lists_both", A[0].split("/")[1].split("-GGUF")[0] in names and "gemma-3-270m" in names, names)
        for m, key in ((A, "chat_A"), (B, "chat_B")):
            ok, served, secs = chat(m)
            check(key, ok, f"served={served} {secs}s")
        st, r = call(base, "/api/inference/unload", {"model_path": B[0]}, tok)
        check("unload_B", st == 200, r if st != 200 else "")
        ok, served, _ = chat(A)
        check("A_still_serves_after_unloading_B", ok, served)
        st, r = load(B, False)
        check("plain_load_B", st == 200, r if st != 200 else "")
        s = status()
        info["status_after_plain"] = {k: s.get(k) for k in ("active_model", "loaded")} if isinstance(s, dict) else s
        check("plain_load_replaces", isinstance(s, dict) and len(s.get("loaded") or []) <= 1
              and "qwen3" not in json.dumps(s.get("loaded")).lower() and "gemma" in str(s.get("active_model")).lower(), info["status_after_plain"])
    except Exception as e:
        check("probe_ran", False, repr(e))
    finally:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(60)
        except Exception:
            pass
    passed = bool(checks) and all(c["ok"] for c in checks.values())
    result = {"passed": passed, "checks": checks, "info": info}
    if a.out:
        with open(a.out, "w", encoding = "utf-8") as fh:
            json.dump(result, fh, indent = 1)
    print("PROBE_RESULT " + json.dumps(result), flush = True)
    if not passed:
        print("---- studio log tail ----")
        print(open(os.path.join(a.home, "probe_studio.log")).read()[-4000:])
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
