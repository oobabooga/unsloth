#!/usr/bin/env python3
"""Probe: what does llama-server emit when the BINARY is the variable between states?

Derived from probes/llamacpp_uma_probe.py: same launch, same device parsing, same
stop-by-PID. Three things differ, each because the variable under test is a build
rather than an environment variable.

  * --bin-file. lib/differential.py hands identical arguments to every state, so
    the binary has to arrive through the one thing that differs: the state
    directory. Each state dir carries a `bin` file naming the llama.cpp directory
    to run. --bin-dir is the fallback when the file is absent.

  * Two request shapes. `nonce`: N concurrent chat requests, each a long filler
    carrying a unique code, asking for the code back, then sequential repeats.
    Whether the code comes back is arithmetic, not a judgement about prose. It is
    walcz-de's harness from ggml-org#25992 and the shape lemonade#3160 hit
    `////////` with. `raw`: exact /completion prompts from a JSON file, for
    reproducing a reporter's prompt byte for byte (ggml-org#27797).

  * Degeneracy counters, recorded and never judged here: fraction of `/`, longest
    run of one character, longest run of one token id, distinct-token ratio. The
    criteria module owns the thresholds.
"""

from __future__ import annotations

import argparse
import hashlib
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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

UMA_VAR = "GGML_CUDA_ENABLE_UNIFIED_MEMORY"

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
    "hip_invalid_value": r"hipErrorInvalidValue|invalid argument",
}

LAYER_DEV_RE = re.compile(r"load_tensors: layer\s+\d+ assigned to device ([\w:.\-]+)")
BUFFER_DEV_RE = re.compile(r"load_tensors:\s+(\S+)\s+model buffer size")
GFX_RE = re.compile(r"gfx\d{3,4}[a-z]*", re.IGNORECASE)
BUILD_RE = re.compile(r"build\s*[:=]?\s*(\d+)\s*\(([0-9a-f]+)\)|version:\s*\S+\s*\(build\s*(\d+)")

# Plain English so the tokenizer sees ordinary text; varied per request so two
# concurrent prompts never share a prefix the server could dedupe.
WORDS = ("river stone bridge morning lantern harvest window copper meadow signal "
         "ladder orchard thunder pencil harbour candle valley marble engine forest "
         "saddle compass cellar timber garden anchor summer feather mirror pillow "
         "quarry ribbon shelter kettle canyon blanket falcon hammer island jacket").split()


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
    return "\n".join(p.read_text(errors = "replace") for p in paths if p.is_file())


def host_arch() -> dict:
    info: dict = {}
    for tool, cmd in (("rocminfo", ["rocminfo"]),
                      ("amd_smi", ["amd-smi", "static", "--asic"]),
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
    for name, cmd in (("rocm-smi", ["rocm-smi", "--showmeminfo", "vram", "--csv"]),
                      ("amd-smi", ["amd-smi", "metric", "--mem", "--csv"])):
        if not shutil.which(cmd[0]):
            continue
        try:
            p = subprocess.run(cmd, capture_output = True, text = True, timeout = 120)
        except Exception as e:  # noqa: BLE001
            return {"tool": name, "error": f"{type(e).__name__}: {e}"}
        return {"tool": name, "raw": ((p.stdout or "") + (p.stderr or ""))[:2000]}
    return {"tool": "none"}


def parse_env_spec(lines: list[str]) -> tuple[dict, list[str]]:
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


def max_run(seq) -> int:
    best = cur = 0
    prev = object()
    for x in seq:
        cur = cur + 1 if x == prev else 1
        prev = x
        best = max(best, cur)
    return best


def degeneracy(text: str, tokens: list | None) -> dict:
    d = {
        "chars": len(text),
        "slash_count": text.count("/"),
        "slash_frac": round(text.count("/") / len(text), 3) if text else 0.0,
        "max_char_run": max_run(text),
    }
    if tokens:
        d["max_token_run"] = max_run(tokens)
        d["distinct_ratio"] = round(len(set(tokens)) / len(tokens), 3)
    return d


def filler(words: int, seed: int) -> str:
    out = []
    n = len(WORDS)
    for i in range(words):
        out.append(WORDS[(i * 7 + seed * 13 + (i // n)) % n])
        if i % 11 == 10:
            out[-1] += "."
    return " ".join(out)


def nonce_prompt(idx: int, words: int) -> tuple[str, str]:
    """Filler with the code planted in the first ubatch and again past the middle.

    The race overwrites prompt tokens of an earlier chunk while the device is
    still reading them, so the code sits where chunk one lands and once more where
    a later chunk does. The question is the last line, which a reader of the log
    can check by eye as well as by string.
    """
    code = f"{(7919 * (idx + 1) + 104729) % 90000000 + 10000000}"
    head = filler(words // 3, idx)
    mid = filler(words // 3, idx + 100)
    tail = filler(words - 2 * (words // 3), idx + 200)
    prompt = (f"Here is some text. The secret code is {code}. Remember it.\n\n{head}\n\n"
              f"{mid}\n\nThe secret code, again, is {code}.\n\n{tail}\n\n"
              f"Reply with only the secret code and nothing else.")
    return code, prompt


def chat_request(port: int, prompt: str, system: str, n_predict: int, thinking: bool,
                 timeout: float) -> tuple[dict, str]:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    status, body, err = post_json(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        {"messages": messages, "max_tokens": n_predict, "temperature": 0.0, "top_k": 1,
         "top_p": 1.0, "seed": 1234, "cache_prompt": False,
         "chat_template_kwargs": {"enable_thinking": bool(thinking)}},
        timeout = timeout)
    rec: dict = {"status": status, "error": err or None}
    if isinstance(body, dict):
        choice = (body.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        content = msg.get("content") or ""
        reasoning = msg.get("reasoning_content") or ""
        rec["content"] = content[:600]
        rec["reasoning"] = reasoning[:600]
        rec["finish_reason"] = choice.get("finish_reason")
        usage = body.get("usage") or {}
        rec["prompt_tokens"] = usage.get("prompt_tokens")
        rec["tokens_generated"] = usage.get("completion_tokens")
        timings = body.get("timings") or {}
        rec["predicted_per_second"] = timings.get("predicted_per_second")
        rec["prompt_per_second"] = timings.get("prompt_per_second")
        return rec, content + reasoning
    return rec, ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True, type = Path,
                    help = "state directory: the GGUF, an `env` file, and a `bin` file")
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--bin-dir", type = Path, default = None,
                    help = "fallback when the state dir has no `bin` file")
    ap.add_argument("--bin-file", default = "bin",
                    help = "relative to --checkout; names the llama.cpp dir for THIS state")
    ap.add_argument("--gguf", default = "model.gguf")
    ap.add_argument("--env-file", default = "env")
    ap.add_argument("--ngl", type = int, default = 999)
    ap.add_argument("--ctx-size", type = int, default = 8192)
    ap.add_argument("--n-predict", type = int, default = 48)
    ap.add_argument("--mode", choices = ["nonce", "raw"], default = "nonce")
    ap.add_argument("--concurrent", type = int, default = 0,
                    help = "nonce mode: requests fired at once before the sequential ones")
    ap.add_argument("--repeats", type = int, default = 3,
                    help = "nonce mode: sequential single requests; raw mode: runs per prompt")
    ap.add_argument("--filler-words", type = int, default = 1400)
    ap.add_argument("--thinking", choices = ["on", "off"], default = "off")
    ap.add_argument("--system", default = "You are a helpful assistant.")
    ap.add_argument("--raw-prompts-file", type = Path, default = None,
                    help = "raw mode: JSON list of {name, prompt}")
    ap.add_argument("--extra-arg", action = "append", default = [])
    ap.add_argument("--env", action = "append", default = [], metavar = "KEY=VALUE")
    ap.add_argument("--load-timeout", type = int, default = 1800)
    ap.add_argument("--request-timeout", type = int, default = 900)
    args = ap.parse_args()

    raw_prompts: list[dict] = []
    if args.mode == "raw":
        if not args.raw_prompts_file or not args.raw_prompts_file.is_file():
            raise SystemExit("raw mode needs --raw-prompts-file")
        raw_prompts = json.loads(args.raw_prompts_file.read_text(encoding = "utf-8"))

    spec = {
        "mode": args.mode, "n_predict": args.n_predict, "concurrent": args.concurrent,
        "repeats": args.repeats, "filler_words": args.filler_words, "thinking": args.thinking,
        "system": args.system, "ctx_size": args.ctx_size, "ngl": args.ngl,
        "extra_args": list(args.extra_arg),
        "raw_prompts_sha": hashlib.sha256(
            json.dumps(raw_prompts, sort_keys = True).encode()).hexdigest()[:16],
    }
    obs: dict = {"state": args.state, "request_spec": spec,
                 "repeats_requested": args.repeats, "n_predict": args.n_predict}
    obs.update(host_arch())

    def finish() -> int:
        args.out.parent.mkdir(parents = True, exist_ok = True)
        args.out.write_text(json.dumps(obs, indent = 2))
        return 0

    bin_path = args.checkout / args.bin_file
    obs["bin_file_present"] = bin_path.is_file()
    bin_dir = Path(bin_path.read_text().strip()) if bin_path.is_file() else args.bin_dir
    if bin_dir is None:
        obs["setup_error"] = "no `bin` file in the state dir and no --bin-dir"
        return finish()
    server = bin_dir / "llama-server"
    if not server.is_file() and (bin_dir / "llama-server.exe").is_file():
        server = bin_dir / "llama-server.exe"
    gguf = args.checkout / args.gguf
    obs["bin_dir"] = str(bin_dir)
    obs["server_binary"] = str(server)
    obs["gguf"] = str(gguf)
    obs["gguf_name"] = args.gguf
    obs["gguf_size"] = gguf.stat().st_size if gguf.is_file() else None
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
        [str(bin_dir), env.get("LD_LIBRARY_PATH", "")]).strip(os.pathsep)
    shared_set, shared_unset = parse_env_spec(args.env)
    env_path = args.checkout / args.env_file
    per_set: dict = {}
    per_unset: list[str] = []
    obs["env_file_present"] = env_path.is_file()
    if env_path.is_file():
        per_set, per_unset = parse_env_spec(env_path.read_text(errors = "replace").splitlines())
    applied_set = {**shared_set, **per_set}
    applied_unset = list(dict.fromkeys([*shared_unset, *per_unset]))
    for k in applied_unset:
        env.pop(k, None)
    env.update(applied_set)
    obs["env_overrides"] = applied_set
    obs["env_unset"] = applied_unset
    obs["uma_env_present"] = UMA_VAR in env
    obs["uma_env_value"] = env.get(UMA_VAR)

    obs.update(binary_build(server, env))
    try:
        p = subprocess.run([str(server), "--list-devices"], capture_output = True,
                           text = True, timeout = 300, env = env)
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
           "-c", str(args.ctx_size), "--no-webui", "-v", "--log-file", str(log_file)]
    cmd += args.extra_arg
    obs["cmd"] = cmd

    started = time.time()
    completions: list[dict] = []
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

        thinking = args.thinking == "on"

        def one_nonce(idx: int, phase: str) -> dict:
            code, prompt = nonce_prompt(idx, args.filler_words)
            t0 = time.time()
            rec, text = chat_request(port, prompt, args.system, args.n_predict, thinking,
                                     args.request_timeout)
            rec.update({"i": idx, "phase": phase, "code": code,
                        "seconds": round(time.time() - t0, 1),
                        "nonce_ok": code in text if text else False})
            rec.update(degeneracy(text, None))
            return rec

        def one_raw(name: str, prompt: str, i: int) -> dict:
            t0 = time.time()
            status, body, err = post_json(
                f"http://127.0.0.1:{port}/completion",
                {"prompt": prompt, "n_predict": args.n_predict, "temperature": 0.0,
                 "top_k": 1, "top_p": 1.0, "seed": 1234, "cache_prompt": False,
                 "return_tokens": True},
                timeout = args.request_timeout)
            rec: dict = {"i": i, "phase": "raw", "name": name, "status": status,
                         "error": err or None, "seconds": round(time.time() - t0, 1)}
            if isinstance(body, dict):
                content = body.get("content") or ""
                toks = body.get("tokens")
                toks = toks if isinstance(toks, list) else None
                rec["content"] = content[:600]
                rec["tokens"] = toks
                timings = body.get("timings") or {}
                rec["tokens_generated"] = timings.get("predicted_n", body.get("tokens_predicted"))
                rec["predicted_per_second"] = timings.get("predicted_per_second")
                rec.update(degeneracy(content, toks))
            return rec

        if ready and args.mode == "nonce":
            if args.concurrent > 0:
                with ThreadPoolExecutor(max_workers = args.concurrent) as ex:
                    futs = [ex.submit(one_nonce, i, "concurrent") for i in range(args.concurrent)]
                    completions += [f.result() for f in futs]
            for _ in range(max(0, args.repeats)):
                completions.append(one_nonce(1000, "sequential"))
                if proc.poll() is not None:
                    break
        elif ready and args.mode == "raw":
            for item in raw_prompts:
                for r in range(max(1, args.repeats)):
                    completions.append(one_raw(item["name"], item["prompt"], r))
                    if proc.poll() is not None:
                        break

        for c in completions:
            c["alive_after"] = proc.poll() is None
        obs["completions"] = completions
        obs["alive_after_request"] = proc.poll() is None
        stop(proc)

    rc = proc.returncode
    obs["rc"] = rc
    obs["signal"] = -rc if rc is not None and rc < 0 else None
    obs["sigabrt"] = obs["signal"] == 6

    ok = [c for c in completions if c.get("status") == 200 and c.get("tokens_generated")]
    obs["completions_ok"] = len(ok)
    obs["completions_total"] = len(completions)
    seq = [c for c in ok if c.get("phase") == "sequential"]
    seq_texts = {(c.get("content") or "") + (c.get("reasoning") or "") for c in seq}
    obs["sequential_identical"] = len(seq) > 1 and len(seq_texts) == 1
    obs["nonce_total"] = sum(1 for c in ok if "nonce_ok" in c)
    obs["nonce_ok_count"] = sum(1 for c in ok if c.get("nonce_ok"))
    obs["max_slash_frac"] = max((c.get("slash_frac", 0.0) for c in ok), default = 0.0)
    obs["max_char_run"] = max((c.get("max_char_run", 0) for c in ok), default = 0)
    obs["max_token_run"] = max((c.get("max_token_run", 0) for c in ok), default = 0)
    by_name: dict = {}
    for c in ok:
        if c.get("phase") == "raw":
            by_name.setdefault(c["name"], set()).add(json.dumps(c.get("tokens")))
    obs["raw_repeats_identical"] = {n: len(v) == 1 for n, v in by_name.items()}

    text = read_logs(log_file, cap_file)
    obs["markers"] = sorted(k for k, pat in MARKERS.items() if re.search(pat, text, re.IGNORECASE))
    obs["devices"] = sorted(set(LAYER_DEV_RE.findall(text)) | set(BUFFER_DEV_RE.findall(text)))
    obs["layers_by_device"] = {
        d: LAYER_DEV_RE.findall(text).count(d) for d in set(LAYER_DEV_RE.findall(text))}
    gfx_in_log = sorted({m.group(0).lower() for m in GFX_RE.finditer(text)})
    if gfx_in_log:
        obs.setdefault("gfx", gfx_in_log[0])
        obs["gfx_in_log"] = gfx_in_log
    obs["log_tail"] = text[-4000:]
    obs["succeeded"] = bool(ok) and len(ok) == len(completions)
    obs["crashed"] = not bool(ok)
    return finish()


if __name__ == "__main__":
    sys.exit(main())
