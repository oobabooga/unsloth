"""#8442 at the llama-server layer, no Studio: does an earlier request change a later, unrelated one?

For each config: fresh server -> Pi request alone (cold reference). Fresh server -> OpenCode title+main
concurrently, then the same Pi request (warm). Greedy decoding, so a clean server gives cold == warm.
Repeats the warm sequence N times on the same server, since the leak is reported as sticky.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prompts import OC_USERS, OPENCODE_SYSTEM, PI_SYSTEM, PI_USER, PROMPT_TITLE, TOOLS  # noqa: E402

CONFIGS = {
    "studio_default": ["-np", "4", "--kv-unified", "--spec-type", "draft-mtp", "--spec-draft-n-max", "3"],
    "mtp_np1": ["-np", "1", "--spec-type", "draft-mtp", "--spec-draft-n-max", "3"],
    "no_mtp_np4": ["-np", "4", "--kv-unified", "--spec-type", "none"],
    "no_mtp_np4_split": ["-np", "4", "--no-kv-unified", "--spec-type", "none"],
    "no_mtp_np1": ["-np", "1", "--spec-type", "none"],
}


def post(port, body, timeout=900):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read())
    m = out["choices"][0]["message"]
    return {"content": m.get("content") or "", "reasoning": m.get("reasoning_content") or "",
            "tool_calls": m.get("tool_calls"), "timings": out.get("timings")}


def title_like(r):
    c = (r["content"] or "").strip()
    return (bool(c) and len(c) <= 60 and "\n" not in c and not r["tool_calls"]) or "title" in (r["reasoning"] or "").lower()


def garbage(r):
    t = (r["content"] or "") + (r["reasoning"] or "")
    return len(t) >= 20 and t.count("!") > len(t) // 2


def body(messages, tools=None, max_tokens=400):
    b = {"messages": messages, "temperature": 0.0, "top_k": 1, "max_tokens": max_tokens, "stream": False}
    if tools:
        b["tools"] = tools
    return b


PI = body([{"role": "system", "content": PI_SYSTEM}, {"role": "user", "content": PI_USER}], TOOLS)


def oc(port, user):
    t = body([{"role": "system", "content": PROMPT_TITLE}, {"role": "user", "content": "Generate a title for this conversation:\n"},
              {"role": "user", "content": user}], None, 200)
    m = body([{"role": "system", "content": OPENCODE_SYSTEM}, {"role": "user", "content": user}], TOOLS, 400)
    with cf.ThreadPoolExecutor(2) as ex:
        a = ex.submit(post, port, t)
        time.sleep(0.05)
        b = ex.submit(post, port, m)
        return a.result(), b.result()


class Server:
    def __init__(self, binary, model, args, port, log):
        self.port = port
        cmd = [binary, "-m", model, "--port", str(port), "--host", "127.0.0.1", "-c", "65536", "-ngl", "999",
               "-fa", "on", "--jinja", "--metrics", *args]
        self.p = subprocess.Popen(cmd, stdout=open(log, "w"), stderr=subprocess.STDOUT, preexec_fn=os.setsid)
        deadline = time.time() + 600
        while time.time() < deadline:
            if self.p.poll() is not None:
                raise RuntimeError(f"llama-server exited {self.p.returncode}, see {log}")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                    if r.status == 200:
                        return
            except Exception:
                time.sleep(1)
        raise TimeoutError(log)

    def stop(self):
        try:
            os.killpg(os.getpgid(self.p.pid), signal.SIGTERM)
            self.p.wait(30)
        except Exception:
            os.killpg(os.getpgid(self.p.pid), signal.SIGKILL)


IDA = body([{"role": "system", "content": "You are Machine 1. Whenever anyone asks who you are, answer exactly: I am Machine 1."},
            {"role": "user", "content": "Hello, who are you?"}], None, 300)
IDB = body([{"role": "system", "content": "You are Bob, a helpful assistant from Acme Corp."},
            {"role": "user", "content": "Who are you? Answer in one sentence."}], None, 300)


def identity_rounds(port, n, no_think=False):
    """Upstream ggml-org/llama.cpp#27422 shape: B must never call itself Machine 1."""
    ida, idb = dict(IDA), dict(IDB)
    if no_think:
        for b in (ida, idb):
            b["chat_template_kwargs"] = {"enable_thinking": False}
            b["max_tokens"] = 80
    hits = []
    for i in range(n):
        if i % 2 == 0:
            post(port, ida)
            b = post(port, idb)
        else:
            with cf.ThreadPoolExecutor(2) as ex:
                fa = ex.submit(post, port, ida)
                time.sleep(0.05)
                fb = ex.submit(post, port, idb)
                fa.result()
                b = fb.result()
        leak = "machine 1" in (b["content"] + b["reasoning"]).lower()
        bob = "bob" in b["content"].lower()
        hits.append({"i": i, "concurrent": i % 2 == 1, "leak": leak, "bob": bob, "content": b["content"][:200]})
    return hits


def run_config(a, name, results):
    args = CONFIGS[name]
    s = Server(a.binary, a.model, args, a.port, f"{a.logdir}/{name}_cold.log")
    try:
        cold = post(a.port, PI)
        cold2 = post(a.port, PI)
    finally:
        s.stop()
    s = Server(a.binary, a.model, args, a.port, f"{a.logdir}/{name}_warm.log")
    warm = []
    try:
        for rep in range(a.repeats):
            ocs = [oc(a.port, u) for u in OC_USERS]
            w = post(a.port, PI)
            warm.append({"oc": ocs, "pi": w})
            oc_bad = [x[1]["content"][:60] for x in ocs[1:] if title_like(x[1])]
            print(f"[{name}] rep{rep} oc_main_title_like={oc_bad} pi_title_like={title_like(w)} oc_main={[x[1]['content'][:60] for x in ocs]} "
                  f"pi_same_as_cold={w['content'] == cold['content']} pi={w['content'][:120]!r}", flush=True)
        idr = identity_rounds(a.port, a.identity_rounds, a.no_think)
        print(f"[{name}] identity leaks={sum(h['leak'] for h in idr)}/{len(idr)} answered_as_bob={sum(h['bob'] for h in idr)} "
              f"sample={idr[0]['content'][:100]!r}", flush=True)
    finally:
        s.stop()
    results[name] = {"identity": idr, "cold": cold, "cold_repeat_same": cold2["content"] == cold["content"], "warm": warm}
    print(f"[{name}] cold={cold['content'][:120]!r} cold_repeat_same={results[name]['cold_repeat_same']}", flush=True)
    Path(a.out).write_text(json.dumps(results, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--configs", default=",".join(CONFIGS))
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--port", type=int, default=18542)
    ap.add_argument("--identity-rounds", type=int, default=10)
    ap.add_argument("--no-think", action="store_true", help="identity rounds with enable_thinking=false")
    ap.add_argument("--out", required=True)
    ap.add_argument("--logdir", required=True)
    a = ap.parse_args()
    results = {}
    for name in a.configs.split(","):
        try:
            run_config(a, name, results)
        except Exception as exc:
            results[name] = {"error": repr(exc)[:500]}
            print(f"[{name}] ERROR {exc!r}"[:600], flush=True)
            Path(a.out).write_text(json.dumps(results, indent=1))
    print("SUMMARY", json.dumps({k: v if "error" in v else {"cold_repeat_same": v["cold_repeat_same"],
                                     "warm_same": [w["pi"]["content"] == v["cold"]["content"] for w in v["warm"]],
                                     "pi_title_like": [title_like(w["pi"]) for w in v["warm"]],
                                     "oc_main_title_like": sum(title_like(x[1]) for w in v["warm"] for x in w["oc"][1:]),
                                     "identity_leaks": sum(h["leak"] for h in v["identity"]),
                                     "identity_bob": sum(h.get("bob", False) for h in v["identity"]),
                                     "identity_n": len(v["identity"]),
                                     "cold_garbage": garbage(v["cold"]),
                                     "warm_garbage": sum(garbage(x) for w in v["warm"] for pair in w["oc"] for x in pair)
                                     + sum(garbage(w["pi"]) for w in v["warm"]),
                                     "warm_total": sum(2 * len(w["oc"]) + 1 for w in v["warm"])}
                                 for k, v in results.items()}))


if __name__ == "__main__":
    main()
