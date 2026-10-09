#!/usr/bin/env python3
"""Does the rebuilt libhsa-runtime64 change anything on a GPU that already works?

Two llama.cpp bundle directories that differ only in libhsa-runtime64.so.1:
  base  TheRock's shipped runtime
  head  the same nightly's rocm-systems pin rebuilt with the KFD wave count
Real topology, no shim. Function first, then speed:
  greedy    llama-completion, 256 tokens, temp 0: text must be identical
  parallel  llama-server --parallel 4, 8 concurrent /completion requests: all must succeed
  bench     llama-bench pp512/tg128, arms interleaved ABBA, fresh process each,
            one warm-up per arm; noise = spread of the base arm's own runs
Writes observations.json, VERDICT.md and verdict.json (regression mode) for announce.py.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import statistics
import subprocess
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PROMPT = "Write a short story about a lighthouse keeper who finds a message in a bottle."
MIN_TOL = 0.03  # a speed drop must exceed max(3%, base spread) to count


def arm_env(bundle: Path, extra: dict | None = None) -> dict:
    env = {k: v for k, v in os.environ.items()
           if k not in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES",
                        "HSA_OVERRIDE_GFX_VERSION", "HSA_USE_SVM", "LD_PRELOAD")}
    env["LD_LIBRARY_PATH"] = str(bundle)
    env.update(extra or {})
    return env


def placement(text: str) -> dict:
    return {
        "layers_rocm": len(re.findall(r"layer\s+\d+ assigned to device ROCm\d+", text)),
        "layers_cpu": len(re.findall(r"layer\s+\d+ assigned to device CPU", text)),
        "rocm_model_buffer_mib": sum(float(x) for x in re.findall(
            r"ROCm\d+ model buffer size\s*=\s*([\d.]+) MiB", text)),
        "rocm_init_failed": "failed to initialize ROCm" in text,
        "rocm_errors": re.findall(r"ROCm error: [^\n]+", text)[:3],
    }


def on_gpu(p: dict) -> bool:
    return (p["layers_rocm"] > 0 and p["layers_cpu"] == 0 and p["rocm_model_buffer_mib"] > 0
            and not p["rocm_init_failed"])


def loaded_hsa(prefix: Path) -> list[str]:
    out = set()
    for f in prefix.parent.glob(prefix.name + ".*"):
        for line in f.read_text(encoding = "utf-8", errors = "replace").splitlines():
            if "calling init:" in line and "libhsa-runtime64" in line:
                out.add(line.split("calling init:", 1)[1].strip())
    return sorted(out)


def greedy(bundle: Path, model: Path, work: Path, tag: str) -> dict:
    dbg = work / f"lddebug_greedy_{tag}"
    cmd = [str(bundle / "llama-completion"), "-m", str(model), "-p", PROMPT, "-n", "256",
           "-ngl", "99", "--temp", "0", "--seed", "1", "-c", "2048", "-v"]
    p = subprocess.run(cmd, env = arm_env(bundle, {"LD_DEBUG": "libs", "LD_DEBUG_OUTPUT": str(dbg)}),
                       stdin = subprocess.DEVNULL, capture_output = True, text = True,
                       errors = "replace", timeout = 600)
    both = p.stdout + "\n" + p.stderr
    gen = p.stdout.split(PROMPT, 1)[1] if PROMPT in p.stdout else ""
    return {"rc": p.returncode, "generated": gen.strip(), "hsa_loaded": loaded_hsa(dbg),
            **placement(both), "stderr_tail": p.stderr[-2000:]}


def parallel(bundle: Path, model: Path, work: Path, tag: str, port: int) -> dict:
    log = work / f"server_{tag}.log"
    dbg = work / f"lddebug_server_{tag}"
    cmd = [str(bundle / "llama-server"), "-m", str(model), "-ngl", "99", "--parallel", "4",
           "-c", "8192", "--host", "127.0.0.1", "--port", str(port), "-v"]
    fh = open(log, "wb")
    srv = subprocess.Popen(cmd, env = arm_env(bundle, {"LD_DEBUG": "libs", "LD_DEBUG_OUTPUT": str(dbg)}),
                           stdin = subprocess.DEVNULL, stdout = fh, stderr = subprocess.STDOUT,
                           start_new_session = True)
    res: dict = {"requests": []}
    try:
        ready = False
        for _ in range(240):
            if srv.poll() is not None:
                break
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout = 2) as r:
                    if r.status == 200:
                        ready = True
                        break
            except Exception:  # noqa: BLE001
                pass
            time.sleep(0.5)
        res["ready"] = ready

        def one(i: int) -> dict:
            body = json.dumps({"prompt": f"{PROMPT} Story number {i}.", "n_predict": 128,
                               "temperature": 0, "seed": 1, "cache_prompt": False}).encode()
            req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", data = body,
                                         headers = {"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout = 300) as r:
                    d = json.loads(r.read())
                    return {"i": i, "status": r.status, "tokens": d.get("tokens_predicted"),
                            "tps": (d.get("timings") or {}).get("predicted_per_second"),
                            "content": (d.get("content") or "")[:200]}
            except Exception as e:  # noqa: BLE001
                return {"i": i, "status": None, "error": f"{type(e).__name__}: {e}"}

        if ready:
            with ThreadPoolExecutor(8) as ex:
                res["requests"] = list(ex.map(one, range(8)))
    finally:
        try:
            os.killpg(srv.pid, signal.SIGTERM)
            srv.wait(timeout = 30)
        except Exception:  # noqa: BLE001
            os.killpg(srv.pid, signal.SIGKILL)
        fh.close()
    text = log.read_text(encoding = "utf-8", errors = "replace")
    res.update(placement(text))
    res["hsa_loaded"] = loaded_hsa(dbg)
    res["exit_signal_clean"] = srv.returncode in (0, -signal.SIGTERM)
    res["all_ok"] = (res["ready"] and len(res["requests"]) == 8 and
                     all(r.get("status") == 200 and (r.get("tokens") or 0) > 0 for r in res["requests"]))
    return res


def bench(bundle: Path, model: Path) -> dict:
    cmd = [str(bundle / "llama-bench"), "-m", str(model), "-ngl", "99", "-p", "512", "-n", "128",
           "-r", "3", "-o", "json"]
    load = os.getloadavg()[0]
    p = subprocess.run(cmd, env = arm_env(bundle), stdin = subprocess.DEVNULL, capture_output = True,
                       text = True, errors = "replace", timeout = 900)
    out: dict = {"rc": p.returncode, "loadavg": load}
    try:
        rows = json.loads(p.stdout[p.stdout.index("["):])
        for r in rows:
            key = "pp512" if r.get("n_prompt") else "tg128"
            out[key] = r.get("avg_ts")
            out[key + "_sd"] = r.get("stddev_ts")
    except Exception as e:  # noqa: BLE001
        out["parse_error"] = f"{type(e).__name__}: {e}"
        out["stderr_tail"] = p.stderr[-1500:]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required = True, type = Path)
    ap.add_argument("--head", required = True, type = Path)
    ap.add_argument("--model", required = True, type = Path)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--abba", type = int, default = 3, help = "ABBA blocks (2 runs per arm each)")
    ap.add_argument("--port", type = int, default = 39123)
    args = ap.parse_args()
    args.out.mkdir(parents = True, exist_ok = True)
    work = args.out / "work"
    work.mkdir(exist_ok = True)
    arms = {"base": args.base.resolve(), "head": args.head.resolve()}
    obs: dict = {"arms": {k: str(v) for k, v in arms.items()}, "nproc": os.cpu_count()}

    obs["greedy"] = {k: greedy(b, args.model, work, k) for k, b in arms.items()}
    obs["parallel"] = {k: parallel(b, args.model, work, k, args.port + i)
                       for i, (k, b) in enumerate(arms.items())}
    obs["warmup"] = {k: bench(b, args.model) for k, b in arms.items()}
    seq = ["base", "head", "head", "base"] * args.abba
    obs["bench_order"] = seq
    obs["bench"] = [{"arm": a, **bench(arms[a], args.model)} for a in seq]
    (args.out / "observations.json").write_text(json.dumps(obs, indent = 2), encoding = "utf-8")

    # ---- judge
    gates, problems = [], []
    for k, b in arms.items():
        g, s = obs["greedy"][k], obs["parallel"][k]
        want = f"{b}/libhsa-runtime64.so.1"
        gates.append((f"{k}: greedy ran on the GPU", g["rc"] == 0 and on_gpu(g),
                      f"rc={g['rc']} rocm={g['layers_rocm']} cpu={g['layers_cpu']} buf={g['rocm_model_buffer_mib']:.1f}MiB"))
        gates.append((f"{k}: loaded its own libhsa-runtime64",
                      g["hsa_loaded"] == [want] and s["hsa_loaded"] == [want],
                      f"greedy={g['hsa_loaded']} server={s['hsa_loaded']}"))
    for k in arms:
        runs = [r for r in obs["bench"] if r["arm"] == k]
        gates.append((f"{k}: every bench run parsed", all("parse_error" not in r and r["rc"] == 0 for r in runs),
                      f"{len(runs)} runs"))
    busy = [r["loadavg"] for r in obs["bench"] if r["loadavg"] > (os.cpu_count() or 1)]
    gates.append(("host not busy during timing (loadavg <= nproc)", not busy, f"max loadavg {max(r['loadavg'] for r in obs['bench']):.1f}"))
    base_ok = obs["parallel"]["base"]["all_ok"]
    gates.append(("base served all 8 parallel requests", base_ok,
                  f"{sum(1 for r in obs['parallel']['base']['requests'] if r.get('status') == 200)}/8"))

    same_text = obs["greedy"]["base"]["generated"] == obs["greedy"]["head"]["generated"] \
        and bool(obs["greedy"]["base"]["generated"])
    if not same_text:
        problems.append("greedy generation differs between base and head")
    hp = obs["parallel"]["head"]
    if not (hp["all_ok"] and on_gpu(hp)):
        problems.append(f"head parallel serving failed: ok={hp['all_ok']} gpu={on_gpu(hp)} errors={hp['rocm_errors']}")

    perf_rows = []
    for metric in ("pp512", "tg128"):
        b = [r[metric] for r in obs["bench"] if r["arm"] == "base" and r.get(metric)]
        h = [r[metric] for r in obs["bench"] if r["arm"] == "head" and r.get(metric)]
        if not b or not h:
            continue
        bm, hm = statistics.median(b), statistics.median(h)
        spread = (max(b) - min(b)) / bm
        tol = max(MIN_TOL, spread)
        delta = (hm - bm) / bm
        worse = delta < -tol
        perf_rows.append((metric, bm, min(b), max(b), hm, min(h), max(h), delta, tol, worse))
        if worse:
            problems.append(f"{metric} median {hm:.1f} vs base {bm:.1f} ({delta:+.1%}, tolerance {tol:.1%})")

    if any(not ok for _, ok, _ in gates):
        verdict, why = "INCONCLUSIVE", "a gate failed"
    elif problems:
        verdict, why = "REGRESSION", "; ".join(problems)
    else:
        verdict, why = "NO_REGRESSION", ("identical greedy text, all parallel requests served on the GPU, "
                                         "speed within the base arm's own spread")

    md = ["## Stock vs rebuilt libhsa-runtime64 on gfx1151 (real topology)", "",
          "| gate | ok | evidence |", "|---|---|---|"]
    md += [f"| {n} | {'yes' if ok else 'NO'} | {ev} |" for n, ok, ev in gates]
    md += ["", "| check | base | head |", "|---|---|---|",
           f"| greedy 256 tokens | rc={obs['greedy']['base']['rc']} | rc={obs['greedy']['head']['rc']}, text identical: {same_text} |"]
    for k in ("base", "head"):
        s = obs["parallel"][k]
        ok = sum(1 for r in s["requests"] if r.get("status") == 200)
        tps = [r["tps"] for r in s["requests"] if r.get("tps")]
        md.append(f"| parallel 4 slots, 8 requests ({k}) | {ok}/8 OK, GPU={on_gpu(s)} | "
                  f"median {statistics.median(tps) if tps else 0:.1f} tok/s per request |")
    md += ["", f"ABBA order, {args.abba} blocks, llama-bench -r 3 per run, fresh process each:", "",
           "| metric | base median (min-max) | head median (min-max) | delta | tolerance | worse |",
           "|---|---|---|---|---|---|"]
    md += [f"| {m} t/s | {bm:.1f} ({bl:.1f}-{bh:.1f}) | {hm:.1f} ({hl:.1f}-{hh:.1f}) | {d:+.1%} | {t:.1%} | {'YES' if w else 'no'} |"
           for m, bm, bl, bh, hm, hl, hh, d, t, w in perf_rows]
    md += ["", f"**{verdict}** - {why}"]
    (args.out / "VERDICT.md").write_text("\n".join(md) + "\n", encoding = "utf-8")
    (args.out / "verdict.json").write_text(json.dumps({"verdict": verdict, "why": why, "mode": "regression"},
                                                      indent = 2), encoding = "utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
