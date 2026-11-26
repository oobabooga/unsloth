#!/usr/bin/env python3
"""Probe: does the video download counter advance while a hosted pre-cast text encoder downloads?

Per state, in a fresh child process with an empty HF cache: plan the load with the checkout's own
``_estimate_download_bytes`` (te_sources = the real public Wan 2.2 TI2V 5B FP8 encoder), publish the plan on a
loading state the way ``_run_load`` does, start the checkout's real ``_fetch_te_prequant`` in a thread, and poll
``load_progress`` every 3 s for ~60 s next to the partial file's on-disk size (st_size and blob_bytes_present, the
Windows allocated-size path). Then cancel. Observes only.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = "unsloth/Wan2.2-TI2V-5B-FP8"
FILE = "Wan2.2-TI2V-5B-text_encoder-FP8.safetensors"
CHILD = r'''
import json, os, sys, threading, time, platform
sys.path.insert(0, sys.argv[1])
out = {"platform": platform.platform(), "samples": []}
try:
    import huggingface_hub
    out["hf"] = huggingface_hub.__version__
    from core.inference import video as V
    from core.inference.diffusion_te_prequant import TePrequantSource
    from hub.utils.hf_cache_state import blob_bytes_present
    out["file"] = V.__file__
    out["hub_cache"] = V.hub_cache_dir()
    be = V.VideoBackend()
    te = {"text_encoder": TePrequantSource(kind="repo", location=sys.argv[2], filename=sys.argv[3])}
    planned = {}
    expected = be._estimate_download_bytes("Wan-AI/Wan2.2-TI2V-5B-Diffusers", None, "", None, "pipeline",
                                           te_sources=te, planned_files=planned)
    out["expected"] = expected
    out["plan"] = {k: [v.repo, v.revision, [list(f) for f in v.files]] for k, v in planned.items()}
    st = V._VideoLoadingState(repo_id="Wan-AI/Wan2.2-TI2V-5B-Diffusers", base_repo="")
    st.incomplete_baseline = V.VideoBackend._plan_partials(planned)
    st.fetch_plan = planned
    st.expected_bytes = expected
    be._loading = st
    ev = threading.Event()
    err = {}
    def run():
        try:
            be._fetch_te_prequant(te, None, cancel_event=ev)
        except BaseException as exc:
            err["e"] = repr(exc)[:300]
    th = threading.Thread(target=run, daemon=True); th.start()
    blobs = os.path.join(V.hub_cache_dir(), "models--" + sys.argv[2].replace("/", "--"), "blobs")
    t0 = time.time()
    while time.time() - t0 < float(sys.argv[4]) and th.is_alive():
        time.sleep(3)
        disk_size = disk_present = 0; names = []
        if os.path.isdir(blobs):
            for n in os.listdir(blobs):
                p = os.path.join(blobs, n)
                if os.path.isfile(p) and not os.path.islink(p):
                    names.append(n)
                    disk_size += os.path.getsize(p)
                    disk_present += blob_bytes_present(__import__("pathlib").Path(p))
        prog = be.load_progress()
        out["samples"].append({"t": round(time.time() - t0, 1), "downloaded": prog.get("downloaded_bytes"),
                               "expected": prog.get("expected_bytes"), "phase": prog.get("phase"),
                               "disk_st_size": disk_size, "disk_present": disk_present, "blobs": names[:4]})
    ev.set()
    out["fetch_error"] = err.get("e")
except BaseException as exc:
    import traceback
    out["error"] = traceback.format_exc()[-3000:]
print("PROBE_JSON " + json.dumps(out, default=str), flush=True)
os._exit(0)
'''


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True, type = Path)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--seconds", type = float, default = 60.0)
    a = ap.parse_args()
    backend = str(a.checkout / "studio" / "backend")
    obs: dict = {"state": a.state}
    deps = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub", "httpx", "structlog",
                           "safetensors", "hf_xet"], capture_output = True, text = True, timeout = 900)
    obs["pip_rc"] = deps.returncode
    root = os.environ.get("RUNNER_TEMP") or tempfile.gettempdir()
    cache = tempfile.mkdtemp(prefix = f"dlcount_{a.state}_", dir = root)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("HF_", "HUGGING"))}
    env.update({"HF_HOME": cache, "HF_HUB_CACHE": os.path.join(cache, "hub"), "PYTHONIOENCODING": "utf-8",
                "UNSLOTH_STUDIO_HOME": os.path.join(cache, "studio_home")})
    try:
        p = subprocess.run([sys.executable, "-c", CHILD, backend, REPO, FILE, str(a.seconds)], env = env,
                           cwd = backend, capture_output = True, text = True, encoding = "utf-8", errors = "replace",
                           timeout = 900)
        line = next((ln for ln in p.stdout.splitlines() if ln.startswith("PROBE_JSON ")), None)
        obs.update(json.loads(line[len("PROBE_JSON "):]) if line else {"error": (p.stderr or p.stdout)[-3000:]})
        obs["stderr_tail"] = (p.stderr or "")[-1500:]
    except subprocess.TimeoutExpired as exc:
        obs["error"] = f"timeout: {exc}"
    a.out.write_text(json.dumps(obs, indent = 1, default = str), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
