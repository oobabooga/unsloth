"""PR 321 probe (observes only): llama-server --fit on a MoE GGUF, per build x case.
Records the n_ubatch the fit chose (head prints "MoE model, n_ubatch = N fits"; llama_context prints n_ubatch under -v),
the fit trail, device model / KV / compute buffers, n_ctx, whether one ~8k-token request succeeds, and OOM / HIP / Vulkan
errors. Writes rows to OUT/rows.jsonl and logs to OUT/logs/. Timings are recorded but this is one launch: not a speed claim."""
import argparse, json, os, re, signal, socket, subprocess, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--bin", required=True, action="append", help="LABEL=DIR")
ap.add_argument("--model", required=True)
ap.add_argument("--corpus", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--case", action="append", required=True, help="NAME=extra args")
a = ap.parse_args()
os.makedirs(a.out + "/logs", exist_ok=True)
text = open(a.corpus, encoding="utf-8", errors="ignore").read()[:30000]
req = json.dumps({"prompt": text, "n_predict": 32, "cache_prompt": False, "temperature": 0}).encode()

PAT = {
    "grow": re.compile(r"MoE model, n_ubatch = (\d+) fits"),
    "ub_ctx": re.compile(r"llama_context: n_ubatch\s+=\s+(\d+)"),
    "n_ctx": re.compile(r"llama_context: n_ctx\s+=\s+(\d+)"),
    "trail": re.compile(r"common_fit_n_ubatch: (n_ubatch=\d+, id=\d+: -?\d+ MiB free vs\. target of \d+ MiB)"),
    "model_buf": re.compile(r"load_tensors:\s+(\S+) model buffer size =\s+([\d.]+) MiB"),
    "kv_buf": re.compile(r"llama_kv_cache:\s+(\S+) KV buffer size =\s+([\d.]+) MiB"),
    "compute_buf": re.compile(r"sched_reserve:\s+(\S+) compute buffer size =\s+([\d.]+) MiB"),
    "fit": re.compile(r"common_params_fit_impl: (will leave.*|cannot meet.*)"),
    "err": re.compile(r"(out of memory|hipErrorOutOfMemory|ErrorOutOfDeviceMemory|CUDA error|HIP error|GGML_ASSERT|Segmentation fault)", re.I),
    "dev": re.compile(r"(ggml_vulkan: \d+ = [^\n]*|Device \d+: [^\n]*)"),
}


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


with open(a.out + "/rows.jsonl", "a", encoding="utf-8") as rows:
    for case in a.case:
        cname, extra = case.split("=", 1)
        for b in a.bin:
            label, d = b.split("=", 1)
            exe = os.path.join(d, "llama-server")
            row = {"case": cname, "build": label, "extra": extra}
            if not os.path.exists(exe):
                row["error"] = "no binary"
                rows.write(json.dumps(row) + "\n")
                rows.flush()
                continue
            port = free_port()
            log = "%s/logs/%s_%s.log" % (a.out, cname, label)
            cmd = [exe, "-m", a.model, "--port", str(port), "-np", "1", "-fa", "on", "-v"] + extra.split()
            row["cmd"] = " ".join(cmd)
            with open(log, "w", encoding="utf-8") as lf:
                p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)
            ok = False
            for _ in range(900):
                if p.poll() is not None:
                    break
                try:
                    urllib.request.urlopen("http://127.0.0.1:%d/health" % port, timeout=2).read()
                    ok = True
                    break
                except Exception:
                    time.sleep(1)
            row["started"] = ok
            if ok:
                try:
                    t0 = time.time()
                    r = json.loads(urllib.request.urlopen(urllib.request.Request(
                        "http://127.0.0.1:%d/completion" % port, data=req,
                        headers={"Content-Type": "application/json"}), timeout=1800).read())
                    t = r.get("timings", {})
                    row.update(req_ok="error" not in r, prompt_n=t.get("prompt_n"), pp=t.get("prompt_per_second"),
                               tg=t.get("predicted_per_second"), wall=round(time.time() - t0, 2), resp_err=r.get("error"))
                except Exception as e:
                    row.update(req_ok=False, resp_err=repr(e)[:300])
            try:
                os.killpg(p.pid, signal.SIGTERM)
            except Exception:
                pass
            try:
                p.wait(30)
            except Exception:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()
            row["exit"] = p.returncode
            L = open(log, encoding="utf-8", errors="ignore").read()
            g = PAT["grow"].findall(L)
            row["grown_to"] = int(g[-1]) if g else None
            u = PAT["ub_ctx"].findall(L)
            row["n_ubatch_ctx_last"] = int(u[-1]) if u else None
            c = PAT["n_ctx"].findall(L)
            row["n_ctx_last"] = int(c[-1]) if c else None
            row["trail"] = PAT["trail"].findall(L)
            row["fit"] = PAT["fit"].findall(L)[:3]
            for k in ("model_buf", "kv_buf", "compute_buf"):
                row[k] = PAT[k].findall(L)[-4:]
            row["errors"] = sorted(set(m.group(0) for m in PAT["err"].finditer(L)))[:8]
            row["devices"] = sorted(set(PAT["dev"].findall(L)))[:6]
            rows.write(json.dumps(row) + "\n")
            rows.flush()
            print(json.dumps({k: row.get(k) for k in ("case", "build", "started", "req_ok", "grown_to", "n_ubatch_ctx_last",
                                                      "n_ctx_last", "trail", "errors", "pp", "tg")}), flush=True)
