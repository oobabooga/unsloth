#!/usr/bin/env python3
"""Probe: does llama-server survive one completion with this model and this flag set?

Observes only. It never decides whether a crash is a regression; that is the
criteria module's job, because "worse than base" depends on what base did.

Two things about this failure class shape the whole design, both taken from the
upstream reports rather than guessed:

  * A clean exit code is not the signal. ggml-org/llama.cpp#27306 loses the
    Vulkan device mid-prefill ("device lost on Vulkan0", ErrorDeviceLost) and
    leaves the process ALIVE, with /health and /v1/models still answering 200.
    So readiness is polled on /health, but SUCCESS is only ever a real
    POST /completion that returned tokens.
  * "No error at all, the model just unloads", as the r/LocalLLaMA reporters
    describe it, is most likely lost child stderr: llama-server's router mode
    spawns a child instance, the child SIGABRTs, and its message goes nowhere
    unless a log file was asked for. Hence -lv 3 --log-file, scanned alongside
    the captured stdout.

The state's `--checkout` is a directory holding the GGUF for that state; the
binary and every flag are passed in and held constant across states.
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

# Every abort signature seen in #24177, #24492, #27306 and #27460, plus the
# generic ROCm/HSA faults. A hit is recorded, never judged.
MARKERS = {
    "ggml_assert": r"GGML_ASSERT",
    "ggml_abort": r"ggml_abort",
    "preallocated_tensor": r"pre-allocated tensor",
    "cannot_run_op": r"cannot run the operation",
    "argsort_smpb": r"shared_mem <= .*smpb|shared_mem <= ggml",
    "top_k_failed": r"TOP_K failed",
    "top_k_unsupported": r"does not have support for op TOP_K",
    "device_lost": r"ErrorDeviceLost|device lost on Vulkan|context is lost|CS has been cancelled",
    "memory_access_fault": r"Memory access fault",
    "hsa_error": r"HSA_STATUS_ERROR",
    "invalid_configuration": r"invalid configuration argument",
    "out_of_memory": r"out of memory|failed to allocate",
    # Only ever present when the diagnostic cell's validation layer really loaded,
    # so its absence distinguishes "clean" from "the layer never ran".
    "vk_validation": r"VUID-|Validation Error",
}

# Where the weights actually went. system_info is the wrong place to look: it
# names ROCm but never Vulkan, so a detector keyed on it calls a working Vulkan
# run "cpu-only". Two forms are accepted because current builds emit the
# per-layer line and older ones the buffer-size line -- and NEITHER appears at
# the default verbosity, which is why the server is run with -v.
LAYER_DEV_RE = re.compile(r"load_tensors: layer\s+\d+ assigned to device ([\w:.\-]+)")
BUFFER_DEV_RE = re.compile(r"load_tensors:\s+(\S+)\s+model buffer size")
GFX_RE = re.compile(r"gfx\d{3,4}[a-z]*", re.IGNORECASE)
BUILD_RE = re.compile(r"build\s*[:=]?\s*(\d+)\s*\(([0-9a-f]+)\)|version:\s*\S+\s*\(build\s*(\d+)")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def make_prompt(n_tokens: int) -> str:
    """Roughly n_tokens of prose.

    Four characters per token is the usual rule of thumb and is close enough:
    the point of the long-prompt cell is to cross a ubatch boundary far into the
    context, not to hit an exact count. The probe records the server's own
    prompt_n so the real figure is in the observation.
    """
    unit = ("The quick brown fox jumps over the lazy dog while the "
            "kernel schedules another compute submission. ")
    return (unit * (1 + (n_tokens * 4) // len(unit)))[: n_tokens * 4]


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
    """gfx name from the system, independent of anything llama.cpp says.

    lib/gate.py cannot answer this: it reads gcnArchName out of torch, and this
    job has no torch. So the arch assertion moves here, and a criteria gate
    turns a miss into INCONCLUSIVE rather than a silent wrong-hardware pass.
    """
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
    """Driver-reported VRAM use, recorded verbatim rather than interpreted.

    The log says which device a layer was ASSIGNED to; this says whether memory
    was actually taken, which is the difference between an offload and a silent
    fallback. The output is kept raw on purpose: amd-smi and rocm-smi field
    names differ by version, and a parser that guesses the wrong column would
    produce a confident wrong number. The criteria gate keys on the per-layer
    device count, which comes from the log and is unambiguous; this is
    corroboration a human reads.
    """
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


def stop(proc: subprocess.Popen) -> None:
    """Stop by PID.

    Never `pkill -f <pattern>`: the pattern matches the shell that runs it, so
    the caller kills itself. The toolkit lints for that (E002) because it has
    already cost a run.
    """
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
                    help = "directory holding this state's GGUF")
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--bin-dir", required = True, type = Path)
    ap.add_argument("--gguf", default = "Qwen3.8-27B-UD-Q4_K_XL.gguf",
                    help = "filename inside --checkout")
    ap.add_argument("--draft-model", default = "",
                    help = "explicit MTP sidecar path, shared by every state")
    ap.add_argument("--spec-type", default = "none")
    ap.add_argument("--ngl", type = int, default = 999,
                    help = "passed once; llama-server warns and ignores duplicates, so this "
                           "is an option rather than something to hand-roll into --extra-arg")
    ap.add_argument("--ctx-size", type = int, default = 8192)
    ap.add_argument("--prompt-tokens", type = int, default = 64)
    ap.add_argument("--n-predict", type = int, default = 16)
    ap.add_argument("--extra-arg", action = "append", default = [])
    ap.add_argument("--env", action = "append", default = [], metavar = "KEY=VALUE",
                    help = "extra environment for the server, e.g. the Vulkan validation "
                           "layer or RADV_DEBUG settings used by the diagnostic cell")
    ap.add_argument("--load-timeout", type = int, default = 900)
    ap.add_argument("--request-timeout", type = int, default = 900)
    args = ap.parse_args()

    obs: dict = {
        "state": args.state,
        "spec_type": args.spec_type,
        "ctx_size": args.ctx_size,
        "prompt_tokens_requested": args.prompt_tokens,
        "extra_args": list(args.extra_arg),
        "draft_model": args.draft_model or None,
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
    obs["gguf_size"] = gguf.stat().st_size if gguf.is_file() else None
    if not server.is_file():
        obs["setup_error"] = f"no llama-server at {server}"
        return finish()
    if not gguf.is_file():
        obs["setup_error"] = f"no model at {gguf}"
        return finish()

    env = dict(os.environ)
    # Prebuilt tarballs ship their .so files next to the binaries.
    env["LD_LIBRARY_PATH"] = os.pathsep.join(
        [str(args.bin_dir), env.get("LD_LIBRARY_PATH", "")]).strip(os.pathsep)
    extra_env = dict(kv.split("=", 1) for kv in args.env if "=" in kv)
    env.update(extra_env)
    obs["env_overrides"] = extra_env
    # --version before the overrides, so a validation layer that refuses to load
    # cannot be mistaken for the binary being unusable.
    clean_env = {**env, **{k: "" for k in extra_env}}
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

    # -v, not -lv 3: at the default verbosity current builds print no
    # `load_tensors:` lines at all, so a device detector silently sees nothing.
    cmd = [str(server), "-m", str(gguf), "-ngl", str(args.ngl),
           "--host", "127.0.0.1", "--port", str(port),
           "-c", str(args.ctx_size), "--no-webui",
           "-v", "--log-file", str(log_file),
           "--spec-type", args.spec_type]
    if args.draft_model:
        cmd += ["-md", args.draft_model]
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

        if ready:
            t0 = time.time()
            status, body, err = post_json(
                f"http://127.0.0.1:{port}/completion",
                {"prompt": make_prompt(args.prompt_tokens),
                 "n_predict": args.n_predict, "temperature": 0.0,
                 "cache_prompt": False},
                timeout = args.request_timeout)
            obs["completion_seconds"] = round(time.time() - t0, 1)
            obs["completion_status"] = status
            obs["completion_error"] = err or None
            if isinstance(body, dict):
                content = body.get("content") or ""
                obs["content_len"] = len(content)
                obs["content_head"] = content[:200]
                timings = body.get("timings") or {}
                obs["timings"] = timings
                obs["tokens_generated"] = timings.get("predicted_n", body.get("tokens_predicted"))
                obs["prompt_tokens_actual"] = timings.get("prompt_n", body.get("tokens_evaluated"))
                # llama-server only reports draft counters when speculative
                # decoding actually ran, which is the strongest available signal
                # that MTP engaged rather than being silently skipped.
                obs["draft_n"] = timings.get("draft_n")
                obs["draft_n_accepted"] = timings.get("draft_n_accepted")

            # The zombie case: the device was lost mid-decode but the process is
            # still up and still answering /health. Recorded before the process
            # is stopped, or the distinction is gone.
            obs["alive_after_request"] = proc.poll() is None

        stop(proc)

    rc = proc.returncode
    obs["rc"] = rc
    obs["signal"] = -rc if rc is not None and rc < 0 else None
    obs["sigabrt"] = obs["signal"] == 6

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
    obs["mtp_log_lines"] = [ln.strip() for ln in text.splitlines()
                            if re.search(r"mtp|nextn|draft", ln, re.IGNORECASE)][:40]
    obs["validation_lines"] = [ln.strip() for ln in text.splitlines()
                               if re.search(r"VUID-|Validation Error|VK_LAYER", ln)][:40]
    obs["log_tail"] = text[-4000:]

    obs["mtp_engaged_timings"] = bool(obs.get("draft_n"))
    # An ACTIVATION line, not a mention. A loose `mtp|draft` match read True on a
    # run with --spec-type none, which would have let a cell whose whole point is
    # MTP pass its non-vacuity gate without MTP ever running. The positive form
    # is llama.cpp's own:
    #   common_speculative_init_result: creating MTP draft context against ...
    obs["mtp_engaged_log"] = bool(re.search(
        r"common_speculative_init|creating MTP draft context|draft context against",
        text, re.IGNORECASE))
    obs["mtp_engaged"] = obs["mtp_engaged_timings"] or obs["mtp_engaged_log"]

    generated = bool(obs.get("tokens_generated")) and obs.get("completion_status") == 200
    obs["succeeded"] = generated
    obs["zombie"] = bool(obs.get("server_ready")) and not generated \
        and bool(obs.get("alive_after_request"))
    # A crash is "the model never produced a token", whatever the exit code did.
    obs["crashed"] = not generated
    obs["timed_out"] = obs.get("server_ready") is False and rc is None

    return finish()


if __name__ == "__main__":
    sys.exit(main())
