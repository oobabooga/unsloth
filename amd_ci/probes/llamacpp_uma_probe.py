#!/usr/bin/env python3
"""Probe: what tokens does llama-server produce with, and without, an env var set?

Observes only. Whether a difference is a regression is the criteria module's
job. Derived from probes/llamacpp_mtp_probe.py; everything about launching the
server, finding the device in the log, and stopping by PID is unchanged. Three
things are new, and each exists because the variable under test is an
ENVIRONMENT VARIABLE rather than a model file.

  * --env-file. lib/differential.py hands the SAME probe arguments to every
    state, so `--env GGML_CUDA_ENABLE_UNIFIED_MEMORY=1` would land on base and
    head alike and the cell would compare nothing. The per-state environment
    therefore has to arrive through the one thing that does differ between
    states: the state directory. Each state dir carries an `env` file.

  * Real unsetting. ggml tests `getenv(...) != nullptr`
    (ggml/src/ggml-cuda/ggml-cuda.cu:139-166), so PRESENCE enables the path and
    `=0` enables it too. A control state must have the name genuinely absent,
    not empty, which `KEY=` cannot express. Hence the `-KEY` form. The probe
    then records whether the name was actually present in the final child
    environment, so the criteria gate asserts ground truth rather than intent.

  * Token IDs, and more than one of them. "Is this text garbled" is a judgement
    a probe must not make and a human cannot make reproducibly. "Did these two
    states emit different token IDs under greedy decoding, same flags, no prompt
    cache" is arithmetic. Repeats within one server process let a gate establish
    that the control is self-consistent BEFORE any cross-state claim is made -
    without that, run-to-run nondeterminism would read as the defect.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

UMA_VAR = "GGML_CUDA_ENABLE_UNIFIED_MEMORY"

# Same signatures as the mtp probe: a hit is recorded, never judged.
MARKERS = {
    "ggml_assert": r"GGML_ASSERT",
    "ggml_abort": r"ggml_abort",
    "preallocated_tensor": r"pre-allocated tensor",
    "cannot_run_op": r"cannot run the operation",
    "device_lost": r"ErrorDeviceLost|device lost on Vulkan|context is lost|CS has been cancelled",
    "memory_access_fault": r"Memory access fault",
    "hsa_error": r"HSA_STATUS_ERROR",
    "invalid_configuration": r"invalid configuration argument",
    "out_of_memory": r"out of memory|failed to allocate",
    # hipMallocManaged returning something the runtime then refuses to advise on.
    "hip_invalid_value": r"hipErrorInvalidValue|invalid argument",
}

LAYER_DEV_RE = re.compile(r"load_tensors: layer\s+\d+ assigned to device ([\w:.\-]+)")
BUFFER_DEV_RE = re.compile(r"load_tensors:\s+(\S+)\s+model buffer size")
GFX_RE = re.compile(r"gfx\d{3,4}[a-z]*", re.IGNORECASE)
# Three spellings across the release eras. Older builds print
# "version: 10000 (47a39665e)", newer ones "version: 0.3.0-dev (build 10631,
# commit 5d5cb4c3a)". Missing the older one made every pre-b10xxx cell fail the
# "same build" gate with build=None and go INCONCLUSIVE, which is what made the
# first bisect sweep uncertifiable. The bare-version alternative is LAST so it
# cannot shadow the "0.3.0-dev" form.
BUILD_RE = re.compile(
    r"build\s*[:=]?\s*(\d+)\s*\(([0-9a-f]+)\)"
    r"|version:\s*\S+\s*\(build\s*(\d+)"
    r"|version:\s*(\d+)\s*\("
)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def post_json(url: str, payload: dict, timeout: float) -> tuple[int | None, dict | None, str]:
    req = urllib.request.Request(
        url, data = json.dumps(payload).encode(), headers = {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout = timeout) as r:
            body = r.read().decode("utf-8", errors = "replace")
            try:
                return r.status, json.loads(body), ""
            except json.JSONDecodeError:
                return r.status, None, body[:2000]
    except urllib.error.HTTPError as e:
        return e.code, None, e.read().decode("utf-8", errors = "replace")[:2000]
    except Exception as e:  # noqa: BLE001
        return None, None, f"{type(e).__name__}: {e}"


def get_ok(url: str, timeout: float) -> bool:
    try:
        with urllib.request.urlopen(url, timeout = timeout) as r:
            return r.status == 200
    except Exception:  # noqa: BLE001
        return False


def read_logs(*paths: Path) -> str:
    out = []
    for p in paths:
        if p.is_file():
            out.append(p.read_text(errors = "replace"))
    return "\n".join(out)


def host_arch() -> dict:
    """gfx name from the system, independent of anything llama.cpp says."""
    info: dict = {}
    for tool, cmd in (("rocminfo", ["rocminfo"]),
                      ("amd_smi", ["amd-smi", "static", "--asic"]),
                      ("vulkaninfo", ["vulkaninfo", "--summary"])):
        exe = shutil.which(cmd[0])
        if not exe:
            continue
        try:
            p = subprocess.run(cmd, capture_output = True, text = True, timeout = 120)
        except Exception as e:  # noqa: BLE001
            info[f"{tool}_error"] = f"{type(e).__name__}: {e}"
            continue
        found = sorted({m.group(0).lower() for m in GFX_RE.finditer(p.stdout + p.stderr)})
        if found:
            info.setdefault("gfx", found[0])
            info[f"{tool}_gfx"] = found
    return info


def binary_build(server: Path, env: dict) -> dict:
    try:
        p = subprocess.run([str(server), "--version"], capture_output = True,
                           text = True, timeout = 120, env = env)
    except Exception as e:  # noqa: BLE001
        return {"build_error": f"{type(e).__name__}: {e}"}
    text = (p.stdout or "") + (p.stderr or "")
    m = BUILD_RE.search(text)
    build = next((g for g in (m.groups() if m else ()) if g and g.isdigit()), None)
    return {"build": build, "version_text": text.strip()[:500]}


def gpu_mem_snapshot() -> dict:
    """Driver-reported VRAM use, recorded verbatim rather than interpreted."""
    for name, cmd in (
        ("rocm-smi", ["rocm-smi", "--showmeminfo", "vram", "--csv"]),
        ("amd-smi", ["amd-smi", "metric", "--mem", "--csv"]),
        ("nvidia-smi", ["nvidia-smi", "--query-gpu=memory.used,memory.total",
                        "--format=csv,noheader"]),
    ):
        if not shutil.which(cmd[0]):
            continue
        try:
            p = subprocess.run(cmd, capture_output = True, text = True, timeout = 120)
        except Exception as e:  # noqa: BLE001
            return {"tool": name, "error": f"{type(e).__name__}: {e}"}
        return {"tool": name, "raw": ((p.stdout or "") + (p.stderr or ""))[:2000]}
    return {"tool": "none"}


def parse_env_spec(lines: list[str]) -> tuple[dict, list[str]]:
    """`KEY=VALUE` sets, `-KEY` unsets, `#` comments, blanks ignored.

    The unset form is not decoration. ggml enables unified memory on the mere
    PRESENCE of the name, so a control state expressed as `KEY=` would enable
    exactly what it is supposed to be controlling for.
    """
    setters: dict = {}
    unsets: list[str] = []
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("-"):
            unsets.append(line[1:].strip())
        elif "=" in line:
            k, v = line.split("=", 1)
            setters[k.strip()] = v
    return setters, unsets


def stop(proc: subprocess.Popen) -> None:
    """Stop by PID. Never `pkill -f <pattern>`: it matches the calling shell."""
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout = 30)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout = 30)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True, type = Path,
                    help = "directory holding this state's GGUF and its `env` file")
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--bin-dir", required = True, type = Path)
    ap.add_argument("--gguf", default = "model.gguf", help = "filename inside --checkout")
    ap.add_argument("--env-file", default = "env",
                    help = "relative to --checkout; how the variable under test differs "
                           "BETWEEN states, since differential.py passes identical args to all")
    ap.add_argument("--ngl", type = int, default = 999)
    ap.add_argument("--ctx-size", type = int, default = 8192)
    ap.add_argument("--n-predict", type = int, default = 30)
    ap.add_argument("--repeats", type = int, default = 3,
                    help = "completions per state; >1 is what lets a gate prove the control "
                           "is self-consistent before any cross-state claim")
    ap.add_argument("--prompt-text", default = "The capital of France is",
                    help = "the reporter's exact prompt")
    ap.add_argument("--extra-arg", action = "append", default = [])
    ap.add_argument("--env", action = "append", default = [], metavar = "KEY=VALUE",
                    help = "environment applied to EVERY state; per-state goes in --env-file")
    ap.add_argument("--load-timeout", type = int, default = 1800)
    ap.add_argument("--request-timeout", type = int, default = 900)
    args = ap.parse_args()

    obs: dict = {
        "state": args.state,
        "ctx_size": args.ctx_size,
        "n_predict": args.n_predict,
        "repeats_requested": args.repeats,
        "prompt_text": args.prompt_text,
        "extra_args": list(args.extra_arg),
    }
    obs.update(host_arch())

    def finish() -> int:
        args.out.parent.mkdir(parents = True, exist_ok = True)
        args.out.write_text(json.dumps(obs, indent = 2))
        return 0

    server = args.bin_dir / "llama-server"
    gguf = args.checkout / args.gguf
    obs["server_binary"] = str(server)
    obs["gguf"] = str(gguf)
    obs["gguf_name"] = args.gguf
    obs["gguf_size"] = gguf.stat().st_size if gguf.is_file() else None
    # Every shard, not just the one named on the command line. A sharded model is
    # loaded by pointing at shard 1, which for this repo is ~10 MB of metadata -
    # so comparing only that file would let two states differ by 87 GiB of
    # weights and still look identical to the gate.
    obs["model_files"] = {p.name: p.stat().st_size
                          for p in sorted(args.checkout.glob("*.gguf"))}
    if not server.is_file():
        obs["setup_error"] = f"no llama-server at {server}"
        return finish()
    if not gguf.is_file():
        obs["setup_error"] = f"no model at {gguf}"
        return finish()

    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = os.pathsep.join(
        [str(args.bin_dir), env.get("LD_LIBRARY_PATH", "")]).strip(os.pathsep)

    shared_set, shared_unset = parse_env_spec(args.env)
    env_path = args.checkout / args.env_file
    obs["env_file"] = str(env_path)
    obs["env_file_present"] = env_path.is_file()
    per_state_set: dict = {}
    per_state_unset: list[str] = []
    if env_path.is_file():
        per_state_set, per_state_unset = parse_env_spec(
            env_path.read_text(errors = "replace").splitlines())
        obs["env_file_text"] = env_path.read_text(errors = "replace")[:2000]

    applied_set = {**shared_set, **per_state_set}
    applied_unset = list(dict.fromkeys([*shared_unset, *per_state_unset]))
    for k in applied_unset:
        env.pop(k, None)
    env.update(applied_set)
    obs["env_overrides"] = applied_set
    obs["env_unset"] = applied_unset

    # Ground truth about the child environment, not a restatement of intent.
    # This is what the non-vacuity gate reads: it is the difference between
    # "we asked for unified memory" and "the process ran with it".
    obs["uma_env_present"] = UMA_VAR in env
    obs["uma_env_value"] = env.get(UMA_VAR)
    obs["launch_blocking"] = env.get("HIP_LAUNCH_BLOCKING")

    # --version and --list-devices before any override, so a variable that
    # changes allocator behaviour cannot be mistaken for an unusable binary.
    clean_env = {k: v for k, v in env.items() if k not in applied_set}
    obs.update(binary_build(server, clean_env))
    try:
        p = subprocess.run([str(server), "--list-devices"], capture_output = True,
                           text = True, timeout = 300, env = clean_env)
        obs["list_devices"] = ((p.stdout or "") + (p.stderr or "")).strip()[:2000]
    except Exception as e:  # noqa: BLE001
        obs["list_devices"] = f"{type(e).__name__}: {e}"
    obs["vram_idle"] = gpu_mem_snapshot()

    port = free_port()
    work = args.out.parent
    log_file = work / f"server_{args.state}.log"
    cap_file = work / f"server_{args.state}.stdout"

    cmd = [str(server), "-m", str(gguf), "-ngl", str(args.ngl),
           "--host", "127.0.0.1", "--port", str(port),
           "-c", str(args.ctx_size), "--no-webui",
           "-v", "--log-file", str(log_file)]
    cmd += args.extra_arg
    obs["cmd"] = cmd

    started = time.time()
    with open(cap_file, "w") as fh:
        proc = subprocess.Popen(cmd, stdout = fh, stderr = subprocess.STDOUT, env = env)

        ready = False
        while time.time() - started < args.load_timeout:
            if proc.poll() is not None:
                break
            if get_ok(f"http://127.0.0.1:{port}/health", timeout = 5):
                ready = True
                break
            time.sleep(3)
        obs["server_ready"] = ready
        obs["load_seconds"] = round(time.time() - started, 1)
        obs["vram_while_loaded"] = gpu_mem_snapshot() if ready else None

        completions: list[dict] = []
        if ready:
            for i in range(max(1, args.repeats)):
                t0 = time.time()
                status, body, err = post_json(
                    f"http://127.0.0.1:{port}/completion",
                    {"prompt": args.prompt_text,
                     "n_predict": args.n_predict,
                     "temperature": 0.0, "top_k": 0, "top_p": 1.0, "seed": 1234,
                     "cache_prompt": False, "return_tokens": True},
                    timeout = args.request_timeout)
                rec: dict = {
                    "i": i,
                    "seconds": round(time.time() - t0, 1),
                    "status": status,
                    "error": err or None,
                }
                if isinstance(body, dict):
                    content = body.get("content") or ""
                    rec["content"] = content[:600]
                    rec["content_len"] = len(content)
                    toks = body.get("tokens")
                    rec["tokens"] = toks if isinstance(toks, list) else None
                    timings = body.get("timings") or {}
                    rec["timings"] = timings
                    rec["tokens_generated"] = timings.get(
                        "predicted_n", body.get("tokens_predicted"))
                    # An implausible decode rate is its own signal: #26148
                    # reports impossible eval rates alongside the corruption.
                    rec["predicted_per_second"] = timings.get("predicted_per_second")
                completions.append(rec)
                # Recorded per request: a server that dies on repeat 2 is a
                # different fact from one that never worked.
                rec["alive_after"] = proc.poll() is None
                if proc.poll() is not None:
                    break

        obs["completions"] = completions
        obs["alive_after_request"] = proc.poll() is None
        stop(proc)

    rc = proc.returncode
    obs["rc"] = rc
    obs["signal"] = -rc if rc is not None and rc < 0 else None
    obs["sigabrt"] = obs["signal"] == 6

    ok = [c for c in completions if c.get("status") == 200 and c.get("tokens_generated")]
    obs["completions_ok"] = len(ok)
    obs["token_vectors"] = [c.get("tokens") for c in ok]
    obs["contents"] = [c.get("content") for c in ok]
    obs["tokens_generated"] = ok[0].get("tokens_generated") if ok else None
    # Self-consistency within the state. A control that disagrees with itself
    # cannot support a claim that the head differs from it.
    vectors = [json.dumps(v) for v in obs["token_vectors"] if v is not None]
    obs["repeats_identical"] = bool(vectors) and len(set(vectors)) == 1
    obs["distinct_vectors"] = len(set(vectors))

    text = read_logs(log_file, cap_file)
    obs["markers"] = sorted(k for k, pat in MARKERS.items()
                            if re.search(pat, text, re.IGNORECASE))
    obs["devices"] = sorted(set(LAYER_DEV_RE.findall(text)) | set(BUFFER_DEV_RE.findall(text)))
    obs["layers_by_device"] = {
        d: LAYER_DEV_RE.findall(text).count(d) for d in set(LAYER_DEV_RE.findall(text))}
    gfx_in_log = sorted({m.group(0).lower() for m in GFX_RE.finditer(text)})
    if gfx_in_log:
        obs.setdefault("gfx", gfx_in_log[0])
        obs["gfx_in_log"] = gfx_in_log
    obs["log_tail"] = text[-4000:]

    generated = bool(ok) and len(ok) == len(completions) and bool(completions)
    obs["succeeded"] = generated
    obs["crashed"] = not bool(ok)

    return finish()


if __name__ == "__main__":
    sys.exit(main())
