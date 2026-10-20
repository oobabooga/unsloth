"""Linux AMD: real install.sh --local / unsloth studio update across base and head trees, recording
the torch build after each step and training through POST /api/train/start. Observes only.

Scenarios (one Studio home per chain, run in order):
  keep:   base fresh (pin rocm7.2) -> head installer re-run -> head studio update -> head Repair
  fresh:  head fresh (pin rocm7.2) -> head studio update
  strix:  head fresh, no pin (the default per-arch route must be unchanged)
"""

import argparse
import glob
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--base", required = True)
ap.add_argument("--head", required = True)
ap.add_argument("--work", required = True)
ap.add_argument("--out", required = True)
ap.add_argument("--only", default = "")
a = ap.parse_args()
os.makedirs(a.out, exist_ok = True)
obs = {"base": a.base, "head": a.head, "steps": []}
PW = "RocmRegress-9932-pw"
PORT = 8931
BASE_URL = f"http://127.0.0.1:{PORT}"


def save():
    with open(os.path.join(a.out, "e2e_obs.json"), "w", encoding = "utf-8") as f:
        json.dump(obs, f, indent = 1)


def run(cmd, log, timeout, env = None, cwd = None):
    print("+", " ".join(cmd), flush = True)
    t = time.time()
    with open(log, "a", encoding = "utf-8") as lf:
        lf.write("+ " + " ".join(cmd) + "\n"); lf.flush()
        try:
            rc = subprocess.run(cmd, stdout = lf, stderr = subprocess.STDOUT, timeout = timeout,
                                env = env, cwd = cwd, stdin = subprocess.DEVNULL).returncode
        except subprocess.TimeoutExpired:
            rc = -999
    print(f"  rc={rc} {time.time() - t:.0f}s", flush = True)
    return rc, round(time.time() - t, 1)


def tail(p, n = 3000):
    try:
        with open(p, encoding = "utf-8", errors = "replace") as f:
            return f.read()[-n:]
    except OSError:
        return ""


def grep(p, keys, n = 30):
    try:
        with open(p, encoding = "utf-8", errors = "replace") as f:
            return [l.rstrip()[:300] for l in f if any(k in l for k in keys)][-n:]
    except OSError:
        return []


def checkout(ref, name):
    d = os.path.join(a.work, name)
    if not os.path.isdir(d):
        log = os.path.join(a.out, "clone.log")
        run(["git", "clone", "-q", "https://github.com/unslothai/unsloth", d], log, 900)
        if ref.startswith("pr:"):
            run(["git", "-C", d, "fetch", "-q", "origin", f"pull/{ref[3:]}/head:prhead"], log, 600)
            run(["git", "-C", d, "checkout", "-q", "prhead"], log, 300)
        else:
            run(["git", "-C", d, "checkout", "-q", ref], log, 300)
    sha = subprocess.run(["git", "-C", d, "rev-parse", "HEAD"], capture_output = True, text = True).stdout.strip()
    return d, sha


def venv_py(home):
    p = os.path.join(home, "unsloth_studio", "bin", "python")
    return p if os.path.exists(p) else None


def torch_state(home):
    py = venv_py(home)
    if not py:
        return {"python": None}
    code = ("import importlib.metadata as m, json\n"
            "out = {}\n"
            "for n in ('torch','torchvision','torchaudio','bitsandbytes','unsloth','unsloth_zoo','transformers'):\n"
            "    try: out[n] = m.version(n)\n"
            "    except Exception: out[n] = None\n"
            "print(json.dumps(out))")
    r = subprocess.run([py, "-I", "-c", code], capture_output = True, text = True, timeout = 120)
    st = json.loads(r.stdout.strip().splitlines()[-1]) if r.returncode == 0 and r.stdout.strip() else {"err": r.stderr[-500:]}
    tf = os.path.join(home, "unsloth_studio")
    st["torch_mtime"] = max((os.path.getmtime(x) for x in glob.glob(os.path.join(tf, "lib", "python*", "site-packages", "torch", "version.py"))), default = None)
    return st


def http(method, path, body = None, token = None, timeout = 60):
    req = urllib.request.Request(BASE_URL + path, method = method,
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


def train(home, tag, load_4bit = False):
    exe = os.path.join(home, "bin", "unsloth")
    if not os.path.exists(exe):
        exe = os.path.join(home, "unsloth_studio", "bin", "unsloth")
    rec = {}
    ds = os.path.join(home, "assets", "datasets", "uploads", "rocm9932_chat.jsonl")
    os.makedirs(os.path.dirname(ds), exist_ok = True)
    with open(ds, "w", encoding = "utf-8") as f:
        for i in range(16):
            f.write(json.dumps({"messages": [
                {"role": "user", "content": f"What is {i} plus {i + 1}?"},
                {"role": "assistant", "content": f"{i} plus {i + 1} is {2 * i + 1}."}]}) + "\n")
    slog = os.path.join(a.out, f"studio_{tag}.log")
    lf = open(slog, "w", encoding = "utf-8")
    env = dict(os.environ, UNSLOTH_STUDIO_HOME = home)
    p = subprocess.Popen([exe, "studio", "-p", str(PORT)], stdout = lf, stderr = subprocess.STDOUT,
                         env = env, stdin = subprocess.DEVNULL, start_new_session = True)
    try:
        t0 = time.time()
        while time.time() - t0 < 600 and p.poll() is None:
            if http("GET", "/api/health", timeout = 10)[0] == 200:
                break
            time.sleep(3)
        tok = None
        boot = os.path.join(home, "auth", ".bootstrap_password")
        for _ in range(60):
            if os.path.exists(boot):
                old = open(boot, encoding = "utf-8").read().strip()
                c, d = http("POST", "/api/auth/login", {"username": "unsloth", "password": old})
                if c == 200:
                    tok = d["access_token"]
                    c, d = http("POST", "/api/auth/change-password", {"current_password": old, "new_password": PW}, tok)
                    tok = (d.get("access_token") if isinstance(d, dict) else None) or tok
                    break
            c, d = http("POST", "/api/auth/login", {"username": "unsloth", "password": PW})
            if c == 200:
                tok = d["access_token"]; break
            time.sleep(3)
        if not tok:
            rec["error"] = "login failed"; return rec
        c, hw = http("GET", "/api/system/hardware?include_details=true", token = tok)
        rec["hw_versions"] = hw.get("versions") if isinstance(hw, dict) else f"{c} {hw}"[:300]
        body = {"model_name": "unsloth/Qwen3-0.6B", "training_type": "LoRA/QLoRA", "format_type": "chatml",
                "local_datasets": [ds], "project_name": f"rocm9932-{tag}", "load_in_4bit": load_4bit,
                "max_seq_length": 512, "max_steps": 5, "num_epochs": 0, "batch_size": 2,
                "gradient_accumulation_steps": 1, "learning_rate": "2e-4", "lora_r": 8, "lora_alpha": 16,
                "save_steps": 10000, "random_seed": 3407, "warmup_steps": 0}
        rec["start"] = http("POST", "/api/train/start", body, tok, timeout = 300)
        st, t0 = None, time.time()
        while time.time() - t0 < 2400:
            c, st = http("GET", "/api/train/status", token = tok)
            if c == 200 and isinstance(st, dict) and st.get("phase") in ("completed", "error", "stopped"):
                break
            time.sleep(10)
        st = st if isinstance(st, dict) else {"raw": st}
        rec.update(phase = st.get("phase"), error = st.get("error") or st.get("message"),
                   losses = (st.get("metric_history") or {}).get("loss"), train_s = round(time.time() - t0, 1))
    finally:
        try:
            os.killpg(p.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            p.wait(30)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
        lf.close()
    return rec


def audio_check(home):
    code = ("import torch, torchaudio, torchaudio.transforms as T\n"
            "x = torch.randn(1, 16000, device='cuda')\n"
            "y = T.Resample(16000, 24000).to('cuda')(x)\n"
            "torch.cuda.synchronize()\n"
            "print(torchaudio.__version__, tuple(y.shape), float(y.abs().mean()))")
    r = subprocess.run([venv_py(home), "-I", "-c", code], capture_output = True, text = True, timeout = 600)
    return {"rc": r.returncode, "out": r.stdout.strip()[-300:], "err": r.stderr.strip()[-800:]}


def step(chain, name, fn):
    print(f"\n===== {chain}/{name} =====", flush = True)
    rec = {"chain": chain, "step": name}
    try:
        rec.update(fn() or {})
    except Exception as e:  # noqa: BLE001
        rec["exception"] = f"{type(e).__name__}: {e}"
    obs["steps"].append(rec)
    save()
    return rec


def installer(tree, home, label, extra_env):
    log = os.path.join(a.out, f"install_{label}.log")
    env = dict(os.environ, UNSLOTH_STUDIO_HOME = home, UNSLOTH_SKIP_AUTOSTART = "1", **extra_env)
    for k in ("UNSLOTH_TORCH_UPGRADE", "UNSLOTH_TORCH_INDEX_FAMILY"):
        if k not in extra_env:
            env.pop(k, None)
    rc, s = run(["bash", os.path.join(tree, "install.sh"), "--local", "--skip-autostart"], log, 5400, env = env, cwd = tree)
    return {"rc": rc, "s": s, "torch": torch_state(home),
            "torch_lines": grep(log, ("torch", "PyTorch", "ROCm", "rocm"), 25), "tail": tail(log, 1500)}


def studio_update(tree, home, label, extra_env):
    log = os.path.join(a.out, f"update_{label}.log")
    exe = os.path.join(home, "bin", "unsloth")
    env = dict(os.environ, UNSLOTH_STUDIO_HOME = home, STUDIO_LOCAL_REPO = tree, **extra_env)
    rc, s = run([exe, "studio", "update", "--local"], log, 5400, env = env, cwd = tree)
    return {"rc": rc, "s": s, "torch": torch_state(home),
            "torch_lines": grep(log, ("torch", "PyTorch", "ROCm", "rocm", "reinstall"), 25), "tail": tail(log, 1500)}


base_tree, base_sha = checkout(a.base, "src_base")
head_tree, head_sha = checkout(a.head, "src_head")
obs["base_sha"], obs["head_sha"] = base_sha, head_sha
save()
PIN = {"UNSLOTH_TORCH_INDEX_FAMILY": "rocm7.2"}
chains = a.only.split(",") if a.only else ["keep", "fresh", "strix"]

if "keep" in chains:
    h = os.path.join(a.work, "home_keep")
    step("keep", "base_fresh_install", lambda: installer(base_tree, h, "keep_base_fresh", PIN))
    step("keep", "base_train", lambda: train(h, "keep_base"))
    step("keep", "head_installer_rerun", lambda: installer(head_tree, h, "keep_head_rerun", PIN))
    step("keep", "head_studio_update", lambda: studio_update(head_tree, h, "keep_head_update", PIN))
    step("keep", "head_train_kept", lambda: train(h, "keep_head_kept"))
    step("keep", "head_repair_upgrade", lambda: installer(head_tree, h, "keep_head_repair", dict(PIN, UNSLOTH_TORCH_UPGRADE = "1")))
    step("keep", "head_train_upgraded", lambda: train(h, "keep_head_upgraded"))
    step("keep", "head_train_upgraded_4bit", lambda: train(h, "keep_head_upgraded_4bit", load_4bit = True))
    step("keep", "head_torchaudio_upgraded", lambda: audio_check(h))

if "fresh" in chains:
    h = os.path.join(a.work, "home_fresh")
    step("fresh", "head_fresh_install", lambda: installer(head_tree, h, "fresh_head", PIN))
    step("fresh", "head_studio_update", lambda: studio_update(head_tree, h, "fresh_head_update", PIN))
    step("fresh", "head_train", lambda: train(h, "fresh_head"))

if "strix" in chains:
    h = os.path.join(a.work, "home_strix")
    step("strix", "head_fresh_install_unpinned", lambda: installer(head_tree, h, "strix_head", {}))

print(json.dumps([{k: (v if k != "tail" else "...") for k, v in s.items() if k not in ("tail", "torch_lines")} for s in obs["steps"]], indent = 1)[:6000])
