#!/usr/bin/env python3
"""Probe: what perplexity does this GGUF reach on this backend?

Observes only. Whether one state is worse than another is the criteria module's
job.

Why perplexity rather than "did it emit sensible text". On this hardware class a
greedy-completion canary has been observed running clean for forty minutes
against a backend that was numerically broken throughout
(ggml-org/llama.cpp#27506). tok/s and HTTP 200 are not evidence of correctness
here, and the token-vector probes in this toolkit only compare two states to
each other - they have no oracle for "is either of them right". Perplexity
against a fixed corpus does, which is what makes it the instrument for asking
whether a quantisation recipe is numerically sound on a given backend.

The per-state `env` file is the same mechanism probes/llamacpp_uma_probe.py uses,
because lib/differential.py hands identical arguments to every state and the
state directory is the only thing that differs. `-KEY` genuinely unsets.

Emits JSON via --out. Never to stdout: llama-perplexity's own output would
corrupt it.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

FINAL_RE = re.compile(r"Final estimate:\s*PPL\s*=\s*([0-9.eE+-]+)\s*\+/-\s*([0-9.eE+-]+)")
CHUNK_RE = re.compile(r"\[(\d+)\]([0-9.eE+-]+),")
LAYER_DEV_RE = re.compile(r"load_tensors: layer\s+\d+ assigned to device ([\w:.\-]+)")
BUFFER_DEV_RE = re.compile(r"load_tensors:\s+(\S+)\s+model buffer size")
GFX_RE = re.compile(r"gfx\d{3,4}[a-z]*", re.IGNORECASE)
BUILD_RE = re.compile(r"build\s*[:=]?\s*(\d+)\s*\(([0-9a-f]+)\)|version:\s*\S+\s*\(build\s*(\d+)")

MARKERS = {
    "ggml_assert": r"GGML_ASSERT",
    "ggml_abort": r"ggml_abort",
    "device_lost": r"ErrorDeviceLost|device lost on Vulkan|context is lost|CS has been cancelled",
    "memory_access_fault": r"Memory access fault",
    "hsa_error": r"HSA_STATUS_ERROR",
    "out_of_memory": r"out of memory|failed to allocate",
    "nan": r"\bnan\b|\b-nan\b",
    "inf": r"\binf\b",
}


def parse_env_spec(lines):
    """`KEY=VALUE` sets, `-KEY` unsets, `#` comments and blanks ignored."""
    setters, unsets = {}, []
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


def host_arch() -> dict:
    """gfx name from the system, independent of anything llama.cpp reports."""
    info: dict = {}
    for tool, cmd in (("rocminfo", ["rocminfo"]),
                      ("vulkaninfo", ["vulkaninfo", "--summary"])):
        if not shutil.which(cmd[0]):
            continue
        try:
            p = subprocess.run(cmd, capture_output = True, text = True, timeout = 120)
        except Exception as e:  # noqa: BLE001
            info[f"{tool}_error"] = f"{type(e).__name__}: {e}"
            continue
        found = sorted({m.group(0).lower() for m in GFX_RE.finditer(p.stdout + p.stderr)})
        if found:
            info.setdefault("gfx", found[0])
    return info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True, type = Path,
                    help = "directory holding this state's GGUF and its `env` file")
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--bin-dir", required = True, type = Path)
    ap.add_argument("--gguf", default = "model.gguf")
    ap.add_argument("--corpus", required = True, type = Path)
    ap.add_argument("--env-file", default = "env")
    ap.add_argument("--ngl", type = int, default = 999)
    ap.add_argument("--ctx-size", type = int, default = 4096)
    ap.add_argument("--chunks", type = int, default = 16)
    ap.add_argument("--extra-arg", action = "append", default = [])
    ap.add_argument("--timeout", type = int, default = 7200)
    args = ap.parse_args()

    obs: dict = {
        "state": args.state,
        "ctx_size": args.ctx_size,
        "chunks_requested": args.chunks,
        "extra_args": list(args.extra_arg),
    }
    obs.update(host_arch())

    def finish() -> int:
        args.out.parent.mkdir(parents = True, exist_ok = True)
        args.out.write_text(json.dumps(obs, indent = 2))
        return 0

    exe = args.bin_dir / "llama-perplexity"
    gguf = args.checkout / args.gguf
    obs["binary"] = str(exe)
    obs["gguf"] = str(gguf)
    obs["gguf_name"] = args.gguf
    # Every shard, not just the one named: a sharded model is loaded by pointing
    # at shard 1, so comparing only that file would let two states differ by all
    # the weights and still look identical to a gate.
    obs["model_files"] = {p.name: p.stat().st_size
                          for p in sorted(args.checkout.glob("*.gguf"))}
    obs["corpus"] = str(args.corpus)
    obs["corpus_bytes"] = args.corpus.stat().st_size if args.corpus.is_file() else None
    if not exe.is_file():
        obs["setup_error"] = f"no llama-perplexity at {exe}"
        return finish()
    if not gguf.is_file():
        obs["setup_error"] = f"no model at {gguf}"
        return finish()
    if not args.corpus.is_file():
        obs["setup_error"] = f"no corpus at {args.corpus}"
        return finish()

    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = os.pathsep.join(
        [str(args.bin_dir), env.get("LD_LIBRARY_PATH", "")]).strip(os.pathsep)
    env_path = args.checkout / args.env_file
    obs["env_file_present"] = env_path.is_file()
    setters, unsets = ({}, [])
    if env_path.is_file():
        setters, unsets = parse_env_spec(env_path.read_text(errors = "replace").splitlines())
        obs["env_file_text"] = env_path.read_text(errors = "replace")[:2000]
    for k in unsets:
        env.pop(k, None)
    env.update(setters)
    obs["env_overrides"] = setters
    obs["env_unset"] = unsets

    clean_env = {k: v for k, v in env.items() if k not in setters}
    try:
        p = subprocess.run([str(exe), "--version"], capture_output = True, text = True,
                           timeout = 120, env = clean_env)
        text = (p.stdout or "") + (p.stderr or "")
        m = BUILD_RE.search(text)
        obs["build"] = next((g for g in (m.groups() if m else ()) if g and g.isdigit()), None)
        obs["version_text"] = text.strip()[:400]
    except Exception as e:  # noqa: BLE001
        obs["build_error"] = f"{type(e).__name__}: {e}"

    cmd = [str(exe), "-m", str(gguf), "-f", str(args.corpus),
           "-c", str(args.ctx_size), "--chunks", str(args.chunks),
           "-ngl", str(args.ngl)]
    cmd += args.extra_arg
    obs["cmd"] = cmd

    log_path = args.out.parent / f"ppl_{args.state}.log"
    started = time.time()
    try:
        with open(log_path, "w") as fh:
            r = subprocess.run(cmd, stdout = fh, stderr = subprocess.STDOUT,
                               timeout = args.timeout, env = env)
        obs["rc"] = r.returncode
        obs["timed_out"] = False
    except subprocess.TimeoutExpired:
        obs["rc"] = None
        obs["timed_out"] = True
    obs["seconds"] = round(time.time() - started, 1)
    obs["signal"] = -obs["rc"] if isinstance(obs["rc"], int) and obs["rc"] < 0 else None

    log = log_path.read_text(errors = "replace") if log_path.is_file() else ""
    m = FINAL_RE.search(log)
    obs["ppl"] = float(m.group(1)) if m else None
    obs["ppl_stderr"] = float(m.group(2)) if m else None
    chunks = [(int(a), float(b)) for a, b in CHUNK_RE.findall(log)]
    obs["chunks_done"] = len(chunks)
    # The running series, not just the final number: a backend that is wrong from
    # chunk 1 and one that degrades are different findings.
    obs["chunk_series"] = chunks[:64]
    obs["markers"] = sorted(k for k, pat in MARKERS.items()
                            if re.search(pat, log, re.IGNORECASE))
    obs["devices"] = sorted(set(LAYER_DEV_RE.findall(log)) | set(BUFFER_DEV_RE.findall(log)))
    obs["layers_by_device"] = {
        d: LAYER_DEV_RE.findall(log).count(d) for d in set(LAYER_DEV_RE.findall(log))}
    gfx_in_log = sorted({m.group(0).lower() for m in GFX_RE.finditer(log)})
    if gfx_in_log:
        obs.setdefault("gfx", gfx_in_log[0])
    obs["log_tail"] = log[-4000:]
    return finish()


if __name__ == "__main__":
    sys.exit(main())
