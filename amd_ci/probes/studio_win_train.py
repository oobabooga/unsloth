"""Windows AMD: install Studio from a git ref with install.ps1 --local, start it, run a short LoRA
job through POST /api/train/start (what the Train button sends), record the outcome. Leg "as_is"
runs Studio untouched; leg "rocm_home" restarts it with ROCM_HOME at the torch wheel's own
_rocm_sdk_core (the candidate fix: Triton reads ROCM_HOME for the HIP headers). Observes only."""

import argparse
import glob
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--ref", required = True)
ap.add_argument("--work", required = True)
ap.add_argument("--out", required = True)
ap.add_argument("--legs", default = "as_is,rocm_home")
a = ap.parse_args()
os.makedirs(a.out, exist_ok = True)
home = os.path.join(a.work, "studio_home")
src = os.path.join(a.work, "unsloth_src")
obs = {"ref": a.ref, "legs": {}}
PW = "RocmRegress-9932-pw"
PORT = 8931
BASE = f"http://127.0.0.1:{PORT}"


def save():
    with open(os.path.join(a.out, "studio_obs.json"), "w", encoding = "utf-8") as f:
        json.dump(obs, f, indent = 1)


def run(cmd, log, timeout, env = None, cwd = None):
    print("+", " ".join(cmd), flush = True)
    t = time.time()
    with open(log, "a", encoding = "utf-8") as lf:
        try:
            rc = subprocess.run(cmd, stdout = lf, stderr = subprocess.STDOUT, timeout = timeout,
                                env = env, cwd = cwd).returncode
        except subprocess.TimeoutExpired:
            rc = -999
    print(f"  rc={rc} {time.time() - t:.0f}s", flush = True)
    return rc


def tail(p, n = 3000):
    try:
        with open(p, encoding = "utf-8", errors = "replace") as f:
            return f.read()[-n:]
    except OSError:
        return ""


def http(method, path, body = None, token = None, timeout = 60):
    req = urllib.request.Request(BASE + path, method = method,
                                 data = None if body is None else json.dumps(body).encode())
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout = timeout) as r:
            return r.status, json.loads(r.read().decode() or "null")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors = "replace")[:2000]
    except Exception as e:  # noqa: BLE001
        return -1, f"{type(e).__name__}: {e}"


# 1. Source + install.
log_i = os.path.join(a.out, "install.log")
if run(["git", "clone", "-q", "https://github.com/unslothai/unsloth", src], log_i, 900) != 0 or \
        run(["git", "-C", src, "checkout", "-q", a.ref], log_i, 300) != 0:
    obs["install"] = {"ok": False, "why": "clone", "tail": tail(log_i)}
    save(); sys.exit(1)
env = dict(os.environ, UNSLOTH_STUDIO_HOME = home, UNSLOTH_SKIP_AUTOSTART = "1")
env.pop("ROCM_HOME", None)
obs["host_env"] = {k: os.environ.get(k) for k in ("HIP_PATH", "HIP_PATH_57", "ROCM_PATH", "ROCM_HOME", "CC")}
rc = run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", os.path.join(src, "install.ps1"),
          "--local", "--skip-autostart"], log_i, 5400, env = env, cwd = src)
py = next(iter(glob.glob(os.path.join(home, "unsloth_studio", "Scripts", "python.exe"))), None)
exe = next(iter(glob.glob(os.path.join(home, "unsloth_studio", "Scripts", "unsloth.exe"))), None)
obs["install"] = {"rc": rc, "python": py, "exe": exe, "tail": tail(log_i, 4000)}
if py:
    fr = subprocess.run([py, "-m", "pip", "list", "--format=freeze"], capture_output = True, text = True)
    obs["install"]["pkgs"] = [l for l in fr.stdout.splitlines()
                              if l.lower().startswith(("torch", "rocm", "triton", "amd-", "unsloth", "bitsandbytes", "transformers"))]
    q = subprocess.run([py, "-c", "import sysconfig,os;p=os.path.join(sysconfig.get_path('platlib'),'_rocm_sdk_core');"
                        "print(p if os.path.isfile(os.path.join(p,'include','hip','hip_runtime.h')) else '')"],
                       capture_output = True, text = True)
    obs["install"]["rocm_sdk_core"] = q.stdout.strip()
save()
if rc != 0 or not exe:
    sys.exit(1)

# 2. Dataset (chat, 16 rows).
ds = os.path.join(home, "assets", "datasets", "uploads", "rocm9932_chat.jsonl")
os.makedirs(os.path.dirname(ds), exist_ok = True)
with open(ds, "w", encoding = "utf-8") as f:
    for i in range(16):
        f.write(json.dumps({"messages": [
            {"role": "user", "content": f"What is {i} plus {i + 1}?"},
            {"role": "assistant", "content": f"{i} plus {i + 1} is {2 * i + 1}."}]}) + "\n")

for leg in a.legs.split(","):
    rec = {}
    obs["legs"][leg] = rec
    lenv = dict(env)
    if leg == "rocm_home":
        if not obs["install"].get("rocm_sdk_core"):
            rec["skipped"] = "no _rocm_sdk_core with HIP headers in the venv"
            save(); continue
        lenv["ROCM_HOME"] = obs["install"]["rocm_sdk_core"]
    slog = os.path.join(a.out, f"studio_{leg}.log")
    lf = open(slog, "w", encoding = "utf-8")
    p = subprocess.Popen([exe, "studio", "-p", str(PORT)], stdout = lf, stderr = subprocess.STDOUT, env = lenv,
                         creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
    try:
        t0 = time.time()
        while time.time() - t0 < 600:
            c, _ = http("GET", "/api/health", timeout = 10)
            if c == 200:
                break
            if p.poll() is not None:
                break
            time.sleep(3)
        rec["health_s"] = round(time.time() - t0, 1)
        boot = os.path.join(home, "auth", ".bootstrap_password")
        tok = None
        for _ in range(60):
            if os.path.exists(boot):
                old = open(boot, encoding = "utf-8").read().strip()
                c, d = http("POST", "/api/auth/login", {"username": "unsloth", "password": old})
                if c == 200:
                    tok = d["access_token"]
                    c, d = http("POST", "/api/auth/change-password",
                                {"current_password": old, "new_password": PW}, tok)
                    tok = (d.get("access_token") if isinstance(d, dict) else None) or tok
                    break
            c, d = http("POST", "/api/auth/login", {"username": "unsloth", "password": PW})
            if c == 200:
                tok = d["access_token"]
                break
            time.sleep(3)
        if not tok:
            rec["error"] = "login failed"
            continue
        c, hw = http("GET", "/api/system/hardware?include_details=true", token = tok)
        rec["hardware"] = hw if c == 200 else f"{c} {hw}"
        body = {"model_name": "unsloth/Qwen3-0.6B", "training_type": "LoRA/QLoRA", "format_type": "chatml",
                "local_datasets": [ds], "project_name": f"rocm9932-{leg}", "load_in_4bit": False,
                "max_seq_length": 512, "max_steps": 5, "num_epochs": 0, "batch_size": 2,
                "gradient_accumulation_steps": 1, "learning_rate": "2e-4", "lora_r": 8, "lora_alpha": 16,
                "save_steps": 10000, "random_seed": 3407, "warmup_steps": 0}
        c, d = http("POST", "/api/train/start", body, tok, timeout = 300)
        rec["start"] = [c, d]
        st = None
        t0 = time.time()
        while time.time() - t0 < 2400:
            c, st = http("GET", "/api/train/status", token = tok)
            if c == 200 and isinstance(st, dict) and st.get("phase") in ("completed", "error", "stopped"):
                break
            time.sleep(10)
        st = st if isinstance(st, dict) else {"raw": st}
        rec["phase"] = st.get("phase")
        rec["error"] = st.get("error") or st.get("message")
        rec["losses"] = (st.get("metric_history") or {}).get("loss")
        rec["train_s"] = round(time.time() - t0, 1)
    finally:
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(p.pid)], capture_output = True)
        lf.close()
        time.sleep(5)
    logs = sorted(glob.glob(os.path.join(home, "logs", "**", "*.log"), recursive = True), key = os.path.getmtime)
    blob = "".join(tail(x, 200000) for x in logs[-6:]) + tail(slog, 200000)
    rec["hip_header_missing"] = "hip/hip_runtime.h' file not found" in blob or "hip_runtime.h" in blob and "not found" in blob
    rec["log_tail"] = tail(slog, 3000)
    rec["worker_log_hits"] = [l[:400] for l in blob.splitlines()
                              if any(k in l for k in ("hip_runtime.h", "Traceback", "Error", "TORCHDYNAMO", "Triton"))][-40:]
    save()
print(json.dumps({k: {x: v.get(x) for x in ("phase", "error", "losses", "hip_header_missing")}
                  for k, v in obs["legs"].items()}, indent = 1))
