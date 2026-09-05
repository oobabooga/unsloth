#!/usr/bin/env python3
"""Backend corruption cells for qwen4exp against any llama.cpp binary dir.

Observes only; the corruption signatures are recorded per cell so a reader
(or a criteria module) can judge. JSON to --out, never stdout.

Cells:
  c1  single segment, -np 1
  c2  the three #27797 multi-segment prompts, -np 1
  c3  the #10330 replica: -np 4 --kv-unified, four concurrent prompts of
      6k..12k chars, -b 2048 -ub 512 so every prompt spans several ubatches
"""
import argparse, json, os, re, shutil, signal, subprocess, sys, threading, time, urllib.request
from pathlib import Path

def server_exe(bin_dir):
    for n in ("llama-server", "llama-server.exe"):
        for d in (Path(bin_dir), Path(bin_dir) / "build" / "bin"):
            if (d / n).is_file(): return str(d / n)
    return str(Path(bin_dir) / "llama-server")

SIGS = {
    "slash_run":   re.compile(r"/{6,}"),
    "bang_run":    re.compile(r"!{6,}"),
    "word_x4":     re.compile(r"(\b\S{2,}\b)(?:[ _]\1){3,}"),
    "replacement": re.compile("�"),
    "char_run":    re.compile(r"(.)\1{24,}"),
}
LOG_SIGS = ["inconsistent sequence positions", "GGML_ASSERT", "failed to decode", "nan", "NaN", "out of memory"]

P27797 = [
    "<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n",
    "<|im_start|>user\nMy name is Bob.<|im_end|>\n<|im_start|>assistant\nNoted!<|im_end|>\n<|im_start|>user\nWhat is my name?<|im_end|>\n<|im_start|>assistant\n",
    "Previous conversation: user said hi, assistant replied hello.\n<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n",
]

def post(port, payload, timeout):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/completion",
                                 data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())

def completion(port, prompt, n, timeout=1800):
    t0 = time.time()
    try:
        r = post(port, {"prompt": prompt, "n_predict": n, "temperature": 0, "seed": 1,
                        "cache_prompt": False, "top_k": 1}, timeout)
        out = {"ok": True, "text": r.get("content", ""), "n_tokens": (r.get("tokens_predicted")),
               "prompt_tokens": r.get("tokens_evaluated"), "stop": r.get("stop_type") or r.get("stopped_eos"),
               "draft_n": (r.get("timings") or {}).get("draft_n"),
               "draft_n_accepted": (r.get("timings") or {}).get("draft_n_accepted"),
               "tg_tps": (r.get("timings") or {}).get("predicted_per_second")}
    except Exception as e:  # noqa: BLE001
        out = {"ok": False, "error": f"{type(e).__name__}: {e}", "text": ""}
    out["secs"] = round(time.time() - t0, 1)
    out["signatures"] = sorted(k for k, rx in SIGS.items() if rx.search(out["text"]))
    return out

def fingerprint(bin_dir):
    import hashlib
    fp = {}
    exe = Path(server_exe(bin_dir))
    if exe.is_file():
        fp["llama-server_sha256"] = hashlib.sha256(exe.read_bytes()).hexdigest()
    for lib in ("libggml-hip.so", "libggml-cuda.so", "libggml-vulkan.so", "ggml-hip.dll", "ggml-cuda.dll", "ggml-vulkan.dll"):
        for cand in (Path(bin_dir) / lib, Path(bin_dir) / "build" / "bin" / lib):
            if cand.is_file():
                fp.setdefault("backend_libs", []).append(lib)
                if not shutil.which("ldd"): continue
                r = subprocess.run(["ldd", str(cand)], capture_output=True, text=True)
                fp[lib] = [l.strip() for l in r.stdout.splitlines() if any(k in l for k in ("hip", "roc", "cuda", "vulkan", "not found"))][:12]
    return fp

class Server:
    def __init__(self, a, label, extra, log):
        self.a, self.label, self.extra, self.log = a, label, extra, log
        self.p = None
    def __enter__(self):
        env = dict(os.environ)
        env["LD_LIBRARY_PATH"] = f"{self.a.bin}:{self.a.bin}/build/bin:" + env.get("LD_LIBRARY_PATH", "")
        for var in self.a.gpu_var:
            if var != "NONE": env[var] = str(self.a.gpu)
        for kv in self.a.env:
            k, _, v = kv.partition("="); env[k] = v
        for k in self.a.unset_env:
            env.pop(k, None)
        cmd = [server_exe(self.a.bin), "-m", self.a.model, "--host", "127.0.0.1", "--port", str(self.a.port),
               "-ngl", "999", "--fit", "off", "-fa", "on", "--no-warmup", *self.extra, *self.a.server_arg]
        self.fh = open(self.log, "w")
        self.fh.write("CMD: " + " ".join(cmd) + "\nENV_UMA: " + repr(env.get("GGML_CUDA_ENABLE_UNIFIED_MEMORY")) + "\n"); self.fh.flush()
        self.p = subprocess.Popen(cmd, stdout=self.fh, stderr=subprocess.STDOUT, env=env)
        for _ in range(self.a.load_timeout // 2):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{self.a.port}/health", timeout=2) as r:
                    if b'"ok"' in r.read(): return self
            except Exception:
                pass
            if self.p.poll() is not None:
                raise RuntimeError(f"server died rc={self.p.returncode}")
            time.sleep(2)
        raise RuntimeError("server load timeout")
    def __exit__(self, *_):
        if self.p and self.p.poll() is None:
            self.p.send_signal(signal.SIGINT if os.name != "nt" else signal.SIGTERM)
            try: self.p.wait(30)
            except subprocess.TimeoutExpired: self.p.kill(); self.p.wait()
        self.fh.close()
    def log_hits(self):
        txt = Path(self.log).read_text(errors="replace")
        hits = {s: txt.count(s) for s in LOG_SIGS}
        dev = [l.strip() for l in txt.splitlines() if "model buffer size" in l][:6]
        ver = [l.strip() for l in txt.splitlines() if "build:" in l or "version:" in l][:2]
        return {"log_sig_counts": {k: v for k, v in hits.items() if v}, "buffers": dev, "version_lines": ver}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True); ap.add_argument("--label", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="temp/model_verify/Qwen3.8-Flash-Next-UD-IQ1_S-00001-of-00003.gguf")
    ap.add_argument("--gpu", default="7"); ap.add_argument("--port", type=int, default=8600)
    ap.add_argument("--cells", default="c1,c2,c3"); ap.add_argument("--env", action="append", default=[])
    ap.add_argument("--unset-env", action="append", default=[])
    ap.add_argument("--server-arg", action="append", default=[])
    ap.add_argument("--load-timeout", type=int, default=1800)
    ap.add_argument("--wikitext", default="data/wikitext2_test.txt")
    ap.add_argument("--n-predict", type=int, default=128)
    ap.add_argument("--gpu-var", action="append", default=None, help="env var(s) that pin the GPU; NONE to skip")
    ap.add_argument("--sentinel-model", default="", help="small known-good GGUF run before and after the cells")
    ap.add_argument("--c3-order", default="", help="arrival order of the four c3 prompts, e.g. 1,0,2,3")
    ap.add_argument("--c3-stagger", type=float, default=0.0, help="seconds between c3 request starts")
    a = ap.parse_args()
    if a.gpu_var is None: a.gpu_var = ["CUDA_VISIBLE_DEVICES"]
    a.bin = str(Path(a.bin).resolve()); a.model = str(Path(a.model).resolve())
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    res = {"label": a.label, "bin": a.bin, "gpu": a.gpu, "env": a.env, "unset_env": a.unset_env, "cells": {}}
    ver = subprocess.run([server_exe(a.bin), "--version"], capture_output=True, text=True,
                         env={**os.environ, "LD_LIBRARY_PATH": f"{a.bin}:{a.bin}/build/bin"})
    res["version"] = (ver.stdout + ver.stderr).strip().splitlines()[:2]
    res["fingerprint"] = fingerprint(a.bin)
    cells = a.cells.split(",")
    def sentinel(tag):
        if not a.sentinel_model: return
        m = a.model; a.model = str(Path(a.sentinel_model).resolve())
        try:
            with Server(a, "sentinel", ["-c", "2048", "-np", "1"], out / f"srv_{a.label}_sentinel_{tag}.log"):
                r = completion(a.port, "The capital of France is", 8, timeout=300)
            r["clean"] = "Paris" in r["text"] and not r["signatures"]
        except Exception as e:  # noqa: BLE001
            r = {"ok": False, "error": f"{type(e).__name__}: {e}", "text": "", "signatures": [], "clean": False}
        finally:
            a.model = m
        res.setdefault("sentinel", {})[tag] = r
    sentinel("pre")
    try:
        if "c1" in cells or "c2" in cells:
            with Server(a, "np1", ["-c", "8192", "-np", "1"], out / f"srv_{a.label}_np1.log") as s:
                if "c1" in cells:
                    res["cells"]["c1"] = {"prompts": [completion(a.port, "The capital of France is", 64)]}
                if "c2" in cells:
                    res["cells"]["c2"] = {"prompts": [completion(a.port, p, 64) for p in P27797]}
                hits = s.log_hits()
            for c in ("c1", "c2"):
                if c in res["cells"]: res["cells"][c].update(hits)
        if "c3" in cells:
            text = Path(a.wikitext).read_text(errors="replace")
            prompts = [text[i * 15000: i * 15000 + n] + "\n\nSummarize the passage above in three sentences." for i, n in enumerate((6000, 8000, 10000, 12000))]
            with Server(a, "np4", ["-c", "32768", "-np", "4", "--kv-unified", "-b", "2048", "-ub", "512"], out / f"srv_{a.label}_np4.log") as s:
                results = [None] * 4
                def run(i):
                    results[i] = completion(a.port, prompts[i], a.n_predict)
                # Arrival order decides which slot each prompt lands on (llama-server
                # hands the first arrival slot 3 by LRU) and so how the prompts'
                # prefill chunks are packed into batches. The Windows Strix Halo run
                # saw the short prompt arrive second and return EOS as its first
                # token; --c3-order replays a given arrival order, --c3-stagger
                # spaces the starts so the order is the one asked for.
                order = [int(x) for x in a.c3_order.split(",")] if a.c3_order else list(range(4))
                th = {i: threading.Thread(target=run, args=(i,)) for i in range(4)}
                for k, i in enumerate(order):
                    if k and a.c3_stagger > 0: time.sleep(a.c3_stagger)
                    th[i].start()
                [th[i].join() for i in range(4)]
                cross = []
                for i in range(4):
                    for j in range(i + 1, 4):
                        ti, tj = results[i]["text"], results[j]["text"]
                        if len(ti) > 80 and len(tj) > 80:
                            for k in range(0, len(ti) - 60, 20):
                                if ti[k:k + 60] in tj: cross.append([i, j, ti[k:k + 60]]); break
                res["cells"]["c3"] = {"prompts": results, "cross_slot_shared_60char": cross, "order": order, "stagger": a.c3_stagger, **s.log_hits()}
    except Exception as e:  # noqa: BLE001
        res["error"] = f"{type(e).__name__}: {e}"
    sentinel("post")
    res["any_signature"] = sorted({s for c in res["cells"].values() for p in c["prompts"] for s in p["signatures"]})
    (out / f"cells_{a.label}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({"label": a.label, "version": res["version"], "any_signature": res["any_signature"],
                      "cells": {k: [(p.get("n_tokens"), p["signatures"], p.get("error")) for p in v["prompts"]] for k, v in res["cells"].items()},
                      "error": res.get("error")}, indent=1))

if __name__ == "__main__":
    main()
