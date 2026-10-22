"""Short chat vs a captured Hermes Agent request on the NPU, through Studio's /v1 proxy and straight
to Studio's lemond, while sampling the Windows NPU utilization counters.

Run from studio/backend with the Studio venv python.
  --phase direct : drive core.inference.npu_backend in-process (no Studio server), hit lemond directly
  --phase studio : start `unsloth studio --api-only`, load through /api/inference/load, hit /v1
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import httpx

p = argparse.ArgumentParser()
p.add_argument("--model", required = True)
p.add_argument("--ctx", type = int, default = 32768)
p.add_argument("--hermes", required = True)
p.add_argument("--sampler", required = True)
p.add_argument("--out", required = True)
p.add_argument("--phase", required = True, choices = ["direct", "studio"])
p.add_argument("--tag", default = "")
p.add_argument("--setup", action = "store_true")
p.add_argument("--port", type = int, default = 8897)
args = p.parse_args()

OUT = Path(args.out)
OUT.mkdir(parents = True, exist_ok = True)
TAG = args.tag or args.phase
client = httpx.Client(timeout = httpx.Timeout(1800, connect = 30), trust_env = False)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush = True)


SHORT_MESSAGES = [
    {"role": "user", "content": "Write a 300-word story about a lighthouse keeper who finds a message in a bottle."}
]


def hermes_body():
    return json.loads(Path(args.hermes).read_text(encoding = "utf-8"))


def hermes_notools_body():
    """Same characters as the Hermes request, but the tool schemas ride in the system prompt as text."""
    body = hermes_body()
    tools = body.pop("tools")
    body["messages"][0]["content"] += "\n\n# Tool schemas (reference only)\n" + json.dumps(tools)
    return body


def flm_pids():
    out = subprocess.run(["tasklist", "/FO", "CSV", "/NH"], capture_output = True, text = True).stdout
    pids = {}
    for line in out.splitlines():
        cols = [c.strip('"') for c in line.split('","')]
        if len(cols) >= 2 and re.match(r"(flm|lemond|lemonade)", cols[0], re.I):
            pids[int(cols[1])] = cols[0]
    return pids


ENGINE_RE = re.compile(
    r"gpu engine\(pid_(\d+)_luid_(0x[0-9a-f]+_0x[0-9a-f]+)_phys_(\d+)_eng_(\d+)_engtype_([^)]*)\)", re.I
)


def analyze(csv_path, pids, t0, t_first, t_end):
    ticks = []
    engine = {}  # ts -> {(luid, phys, eng): sum over pids}
    by_pid_luid = {}  # (pid, luid) -> total
    other = {}  # ts -> {path: value}
    for raw in Path(csv_path).read_text(encoding = "ascii", errors = "replace").splitlines()[1:]:
        if raw.startswith("#"):
            continue
        try:
            ts_s, rest = raw.split(",", 1)
            path, val = rest.rsplit(",", 1)
            ts, val = int(ts_s), float(val)
        except ValueError:
            continue
        path = path.strip('"')
        if path == "tick":
            ticks.append(ts)
            continue
        m = ENGINE_RE.search(path)
        if m:
            pid, luid, phys, eng, etype = int(m[1]), m[2].lower(), m[3], m[4], m[5]
            key = (luid, phys, eng, etype)
            engine.setdefault(ts, {}).setdefault(key, 0.0)
            engine[ts][key] += val
            by_pid_luid[(pid, luid)] = by_pid_luid.get((pid, luid), 0.0) + val
        else:
            other.setdefault(ts, {})[path.lower()] = val
    luids_by_flm = sorted({l for (pid, l) in by_pid_luid if pid in pids})
    luid_totals = {}
    for (pid, l), v in by_pid_luid.items():
        luid_totals[l] = luid_totals.get(l, 0.0) + v
    npu_luid = luids_by_flm[0] if luids_by_flm else (max(luid_totals, key = luid_totals.get) if luid_totals else None)

    def util_at(ts):
        if npu_luid is None:
            return None
        per_engine = [v for (l, *_), v in engine.get(ts, {}).items() if l == npu_luid]
        return max(per_engine) if per_engine else 0.0

    def cpu_at(ts, name):
        return sum(v for k, v in other.get(ts, {}).items() if f"process({name}" in k)

    def window(a, b):
        sel = [ts for ts in ticks if a + 1000 <= ts <= b]
        if not sel:
            return None
        us = [util_at(ts) for ts in sel]
        return {
            "samples": len(sel),
            "npu_util_mean": round(sum(us) / len(us), 1) if us and us[0] is not None else None,
            "npu_util_p50": sorted(us)[len(us) // 2] if us and us[0] is not None else None,
            "flm_cpu_mean": round(sum(cpu_at(ts, "flm") for ts in sel) / len(sel), 1),
            "lemond_cpu_mean": round(sum(cpu_at(ts, "lemond") for ts in sel) / len(sel), 1),
            "python_cpu_mean": round(sum(cpu_at(ts, "python") for ts in sel) / len(sel), 1),
        }

    t0m, t1m, tem = int(t0 * 1000), int((t_first or t_end) * 1000), int(t_end * 1000)
    series = [(ts - t0m, None if util_at(ts) is None else round(util_at(ts), 1)) for ts in ticks]
    return {
        "npu_luid": npu_luid,
        "luids_seen_with_flm_pid": luids_by_flm,
        "luid_totals": {k: round(v, 1) for k, v in luid_totals.items()},
        "engines_on_npu_luid": sorted({f"{k[1]}/{k[2]}/{k[3]}" for d in engine.values() for k in d if k[0] == npu_luid}),
        "idle_before": window(t0m - 5000, t0m),
        "prefill": window(t0m, t1m),
        "decode": window(t1m, tem) if t_first else None,
        "whole": window(t0m, tem),
        "series_ms_util": series,
    }


def run_leg(name, url, headers, body):
    name = f"{TAG}-{name}"
    stop = OUT / f"{name}.stop"
    csv_path = OUT / f"{name}.npu.csv"
    stop.unlink(missing_ok = True)
    sampler = subprocess.Popen(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", args.sampler,
         "-Out", str(csv_path), "-StopFile", str(stop)]
    )
    time.sleep(7)
    pids = flm_pids()
    log(f"[{name}] runtime pids {pids}")
    t0 = time.time()
    t_first = None
    n_chunks = n_content = n_reasoning = n_tool = 0
    usage = timings = None
    err = None
    status = None
    raw_path = OUT / f"{name}.sse.txt"
    with raw_path.open("w", encoding = "utf-8") as raw:
        try:
            with client.stream("POST", url, json = body, headers = headers) as r:
                status = r.status_code
                if status != 200:
                    err = r.read().decode("utf-8", "replace")[:3000]
                else:
                    for line in r.iter_lines():
                        now = time.time()
                        raw.write(f"{now - t0:.3f} {line}\n")
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            break
                        try:
                            obj = json.loads(data)
                        except ValueError:
                            continue
                        if "error" in obj:
                            err = json.dumps(obj["error"])[:2000]
                        n_chunks += 1
                        for ch in obj.get("choices") or []:
                            d = ch.get("delta") or {}
                            got = False
                            if d.get("content"):
                                n_content += 1
                                got = True
                            if d.get("reasoning_content") or d.get("reasoning"):
                                n_reasoning += 1
                                got = True
                            if d.get("tool_calls"):
                                n_tool += 1
                                got = True
                            if got and t_first is None:
                                t_first = now
                        if obj.get("usage"):
                            usage = obj["usage"]
                        if obj.get("timings"):
                            timings = obj["timings"]
        except Exception as exc:  # noqa: BLE001
            err = repr(exc)
    t_end = time.time()
    time.sleep(5)
    stop.touch()
    try:
        sampler.wait(timeout = 30)
    except subprocess.TimeoutExpired:
        sampler.kill()
    res = {
        "leg": name,
        "status": status,
        "error": err,
        "wall_s": round(t_end - t0, 2),
        "ttft_s": round(t_first - t0, 2) if t_first else None,
        "decode_s": round(t_end - t_first, 2) if t_first else None,
        "chunks": n_chunks,
        "content_chunks": n_content,
        "reasoning_chunks": n_reasoning,
        "tool_call_chunks": n_tool,
        "usage": usage,
        "timings": timings,
    }
    try:
        res["npu"] = analyze(csv_path, pids, t0, t_first, t_end)
    except Exception as exc:  # noqa: BLE001 -- the raw CSV is uploaded; re-analyze offline
        res["npu"] = {"error": repr(exc), "series_ms_util": []}
    (OUT / f"{name}.json").write_text(json.dumps(res, indent = 1), encoding = "utf-8")
    brief = {k: v for k, v in res.items() if k != "npu"}
    brief["npu"] = {k: v for k, v in res["npu"].items() if k != "series_ms_util"}
    log(f"[{name}] RESULT", json.dumps(brief))
    log(f"[{name}] SERIES", res["npu"]["series_ms_util"])
    return res


def stream_body(body):
    body = dict(body)
    body["stream"] = True
    body["stream_options"] = {"include_usage": True}
    return body


def legs(url, headers, model, include_notools):
    out = []
    short = stream_body({"model": model, "messages": SHORT_MESSAGES, "max_tokens": 600})
    warm = stream_body({"model": model, "messages": [{"role": "user", "content": "Say hi."}], "max_tokens": 16})
    log("warmup")
    r = client.post(url, json = warm, headers = headers)
    log("warmup status", r.status_code, r.text[-400:])
    out.append(run_leg("short", url, headers, short))
    hb = hermes_body()
    hb["model"] = model
    out.append(run_leg("hermes", url, headers, hb))
    if out[-1]["status"] != 200 or out[-1]["error"]:
        hb["max_tokens"] = 4096
        out.append(run_leg("hermes-max4096", url, headers, hb))
    if include_notools:
        nb = hermes_notools_body()
        nb["model"] = model
        out.append(run_leg("hermes-notools", url, headers, nb))
    return out


def phase_direct():
    sys.path.insert(0, ".")
    from core.inference.npu_backend import get_npu_backend

    b = get_npu_backend()
    log("STATUS", json.dumps(b.status())[:1500])
    if args.setup:
        t = time.time()
        log("ENABLE", json.dumps(b.enable())[:1500], f"{time.time() - t:.0f}s")
        cat = b.catalog()
        ids = [m.id for m in cat]
        log("CATALOG qwen3.6*", [m.to_json() for m in cat if "qwen3.6" in m.id.lower()])
        if args.model not in ids:
            log("model not in catalog; ids:", ids)
            raise SystemExit(2)
        last = None
        t = time.time()
        for ev in b.download(args.model):
            if last is None or time.time() - t > 30:
                log("DOWNLOAD", ev)
                t = time.time()
            last = ev
        log("DOWNLOAD done", last)
    t = time.time()
    b.load(args.model, args.ctx)
    up = b.upstream()
    log("LOADED", args.model, f"{time.time() - t:.0f}s", "ctx", up.context_length, "base", up.base_url)
    root = up.base_url[: -len("/v1")]
    H = {"Authorization": f"Bearer {up.api_key}"}
    log("HEALTH", client.get(f"{root}/v1/health", headers = H).text[:2000])
    res = legs(f"{up.base_url}/chat/completions", H, up.model, include_notools = True)
    b.unload()
    b.shutdown()
    return res


def phase_studio():
    home = Path(os.environ["UNSLOTH_STUDIO_HOME"])
    exe = Path(sys.executable).parent / ("unsloth.exe" if os.name == "nt" else "unsloth")
    base = f"http://127.0.0.1:{args.port}"
    srv_log = (OUT / f"{TAG}-studio-server.log").open("w", encoding = "utf-8")
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
        log("studio up")
        pw_file = home / "auth" / ".bootstrap_password"
        saved = OUT / "studio_pw.txt"
        password = saved.read_text().strip() if saved.exists() else pw_file.read_text().strip()
        r = client.post(f"{base}/api/auth/login", json = {"username": "unsloth", "password": password})
        log("login", r.status_code, r.text[:300])
        tok = r.json()
        token = tok["access_token"]
        if tok.get("must_change_password"):
            newpw = "ProbePass-" + str(int(time.time()))
            r = client.post(
                f"{base}/api/auth/change-password",
                json = {"current_password": password, "new_password": newpw},
                headers = {"Authorization": f"Bearer {token}"},
            )
            log("change-password", r.status_code, r.text[:200])
            token = r.json()["access_token"]
            saved.write_text(newpw)
        A = {"Authorization": f"Bearer {token}"}
        r = client.post(f"{base}/api/auth/api-keys", json = {"name": f"hermes-probe-{TAG}"}, headers = A)
        log("api-key", r.status_code, sorted(r.json()) if r.status_code == 200 else r.text[:300])
        key = None
        if r.status_code == 200:
            j = r.json()
            key = j.get("key") or j.get("api_key") or j.get("raw_key")
        K = {"Authorization": f"Bearer {key}"} if key else A
        log("NPU status", client.get(f"{base}/api/npu/status", headers = A).text[:1500])
        t = time.time()
        r = client.post(
            f"{base}/api/inference/load",
            json = {"model_path": f"lemonade:{args.model}", "max_seq_length": args.ctx},
            headers = A,
        )
        log("load", r.status_code, f"{time.time() - t:.0f}s", r.text.strip()[-800:])
        log("status", client.get(f"{base}/api/inference/status", headers = A).text[:1500])
        res = legs(f"{base}/v1/chat/completions", K, f"lemonade:{args.model}", include_notools = False)
        return res
    finally:
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output = True)
        time.sleep(10)
        log("runtime pids after stop", flm_pids())


results = phase_direct() if args.phase == "direct" else phase_studio()
(OUT / f"{TAG}-summary.json").write_text(json.dumps(results, indent = 1), encoding = "utf-8")
log("DONE", TAG)
