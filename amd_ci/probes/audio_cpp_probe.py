#!/usr/bin/env python3
"""Probe for unslothai/unsloth#13298 (issue #13042): observe, never judge.

At one state (checkout): run the REAL audio.cpp installer with no detection
patching (--print-asset, install, idempotent re-run), read its install record,
list the installed server's devices, and time MOSS-Transcribe-Diarize on a
15.9 s clip through that server. Writes JSON to --out only.

Stdlib only; Linux and Windows. No token is ever passed to the installer.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "i13042"
MODEL_REL = "MOSS-Transcribe-Diarize-GGUF/moss-transcribe-diarize-q4_k.gguf"
MODEL_URL = "https://huggingface.co/audio-cpp/audio.cpp-gguf/resolve/main/" + MODEL_REL
RECORD = "UNSLOTH_AUDIO_CPP_PREBUILT_INFO.json"
SERVER = "audiocpp_server.exe" if sys.platform == "win32" else "audiocpp_server"


def clean_env() -> dict:
    env = dict(os.environ)
    for k in list(env):
        if k.upper() in ("GH_TOKEN", "GITHUB_TOKEN") or k.upper().startswith("GH_TOKEN_"):
            env.pop(k)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    return env


def run(cmd, cwd, timeout, env = None) -> dict:
    t0 = time.monotonic()
    print(f"$ {' '.join(map(str, cmd))}", flush = True)
    try:
        p = subprocess.run([str(c) for c in cmd], cwd = str(cwd), env = env or clean_env(),
                           capture_output = True, timeout = timeout)
        out = p.stdout.decode("utf-8", "replace")
        err = p.stderr.decode("utf-8", "replace")
        rc = p.returncode
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or b"").decode("utf-8", "replace")
        err = (e.stderr or b"").decode("utf-8", "replace") + "\n[TIMEOUT]"
        rc = None
    except OSError as e:
        out, err, rc = "", f"OSError: {e}", None
    print(out, flush = True)
    print(err, flush = True)
    return {"rc": rc, "stdout": out[-4000:], "stderr": err[-4000:],
            "secs": round(time.monotonic() - t0, 2)}


def fetch_model(dest: Path) -> dict:
    if dest.is_file() and dest.stat().st_size > 100_000_000:
        return {"path": str(dest), "bytes": dest.stat().st_size, "cached": True}
    dest.parent.mkdir(parents = True, exist_ok = True)
    part = dest.with_suffix(".part")
    last = ""
    for attempt in range(1, 4):
        try:
            t0 = time.monotonic()
            with urllib.request.urlopen(MODEL_URL, timeout = 60) as r, open(part, "wb") as fh:
                shutil.copyfileobj(r, fh, 1 << 20)
            part.replace(dest)
            return {"path": str(dest), "bytes": dest.stat().st_size, "cached": False,
                    "secs": round(time.monotonic() - t0, 1)}
        except Exception as e:  # noqa: BLE001
            last = f"{type(e).__name__}: {e}"
            print(f"model download attempt {attempt} failed: {last}", flush = True)
            time.sleep(10 * attempt)
    return {"error": last}


def host_facts() -> dict:
    f: dict = {"node": platform.node(), "system": platform.system(), "release": platform.release(),
               "python": sys.version.split()[0], "nvidia_smi": shutil.which("nvidia-smi")}
    if sys.platform.startswith("linux"):
        vend = {}
        for p in sorted(glob.glob("/sys/class/drm/card*/device/vendor")):
            try:
                vend[p] = Path(p).read_text(encoding = "utf-8").strip()
            except OSError as e:
                vend[p] = f"err {e}"
        f["drm_vendors"] = vend
    return f


def gpu_busy() -> list:
    out = []
    for p in sorted(glob.glob("/sys/class/drm/card*/device/gpu_busy_percent")):
        try:
            out.append(int(Path(p).read_text(encoding = "utf-8").strip()))
        except (OSError, ValueError):
            pass
    return out


def transcript(text: str | None) -> str | None:
    if not text:
        return None
    m = re.search(r'"text"\s*:\s*"(.*?)(?<!\\)"', text)
    return m.group(1) if m else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--work", default = None)
    ap.add_argument("--repeats", type = int, default = 2)
    ap.add_argument("--bench-states", default = "base,head")
    a = ap.parse_args()
    a.out = a.out.resolve()

    work = (Path(a.work or os.environ.get("AMD_CI_WORK") or os.environ.get("RUNNER_TEMP") or ".") / "acpp").resolve()
    work.mkdir(parents = True, exist_ok = True)
    co = Path(a.checkout)
    inst = work / f"{a.state}_audio"
    py = sys.executable
    script = "studio/install_audio_cpp_prebuilt.py"
    obs: dict = {"state": a.state, "host": host_facts(), "install_dir": str(inst)}

    def save():
        a.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")

    # detection as the installer's own function sees it (no patching)
    obs["detect_direct"] = run([py, "-c",
        "import sys; sys.path.insert(0, 'studio'); import install_audio_cpp_prebuilt as m; "
        "print('DETECTED=' + m.detect_accelerator())"], co, 120)
    m = re.search(r"DETECTED=(\w+)", obs["detect_direct"]["stdout"])
    obs["detected_direct"] = m.group(1) if m else None

    obs["print_asset"] = run([py, script, "--install-dir", inst, "--print-asset"], co, 600)
    lines = [l for l in obs["print_asset"]["stdout"].splitlines() if l.strip()]
    obs["printed_asset"] = lines[-1].split()[-1] if lines else None
    save()

    obs["install"] = run([py, script, "--install-dir", inst], co, 1800)
    rec_path = inst / RECORD
    rec = {}
    if rec_path.is_file():
        try:
            rec = json.loads(rec_path.read_text(encoding = "utf-8"))
        except Exception as e:  # noqa: BLE001
            obs["record_error"] = str(e)
    obs["record"] = {k: rec.get(k) for k in
                     ("asset", "backend", "accelerator", "accelerator_request", "detected_accelerator",
                      "repo", "tag", "release_tag")}
    obs["record_full"] = rec
    save()

    obs["rerun"] = run([py, script, "--install-dir", inst], co, 1800)
    obs["rerun_already_matches"] = "already matches" in (obs["rerun"]["stdout"] + obs["rerun"]["stderr"])

    servers = sorted(inst.rglob(SERVER))
    server = servers[0] if servers else None
    obs["server"] = str(server) if server else None
    if server:
        obs["list_devices"] = run([server, "--list-devices"], server.parent, 120)
        obs["vulkan_rows"] = re.findall(r"^Vulkan:\d+\s.*$", obs["list_devices"]["stdout"], re.MULTILINE)
    else:
        obs["vulkan_rows"] = []
    save()

    if a.state in a.bench_states.split(",") and server:
        mdl = fetch_model(work.parent / "acpp_models" / MODEL_REL)
        obs["model"] = mdl
        if mdl.get("path"):
            backend = (rec.get("backend") or "cpu").lower()
            obs["bench_backend"] = backend
            obs["bench"] = []
            for i in range(a.repeats):
                jo = a.out.parent / f"bench_{a.state}_{i}.json"
                busy_before = gpu_busy()
                r = run([py, ASSETS / "audio_bench.py", "--server", server, "--backend", backend,
                         "--family", "moss_transcribe_diarize", "--model", mdl["path"], "--task", "asr",
                         "--audio", ASSETS / "speech.wav", "--port", "18790", "--timeout", "900",
                         "--json", jo], ASSETS, 1500)
                res = {}
                if jo.is_file():
                    res = json.loads(jo.read_text(encoding = "utf-8"))
                res["transcript"] = transcript(res.get("text"))
                res["transcript_sha"] = (hashlib.sha256(res["transcript"].encode("utf-8")).hexdigest()[:16]
                                         if res.get("transcript") else None)
                res["bench_rc"] = r["rc"]
                res["gpu_busy_before"] = busy_before
                obs["bench"].append(res)
                save()
    save()
    return 0


if __name__ == "__main__":
    sys.exit(main())
