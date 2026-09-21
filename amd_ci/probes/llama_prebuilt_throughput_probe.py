#!/usr/bin/env python3
"""Probe: what does an unslothai/llama.cpp PREBUILT actually do on this gfx1151?

unsloth#7371 reports a throughput collapse between the `b10069` and `b10079`
prebuilts on Strix Halo ROCm (39 tok/s -> 11 tok/s), with the log line

    llama_sampler_backend_support: device 'ROCm0' does not have support for op
    TOP_K needed for sampler 'top-k'

appearing on the slow build. The states here are RELEASE TAGS, not git commits:
the two builds carry the identical Unsloth patch mix (`fb3d4ca` either side), so
nothing in this repository differs between them and a worktree differential
would compare two identical trees.

Observes only; `criteria/llama_prebuilt_no_slowdown.py` judges.

Every state measures TWO builds in the same process on the same host:

  * the state's own tag, from --tag-map
  * a fixed REFERENCE tag (--reference-tag), the build the report calls fast

The reference leg is the negative control. It is the same binary, the same
model and the same flags in every state, so if its number moves between states
the host was not quiet and no comparison drawn from this run is trustworthy.
A performance claim without that control is how a phantom regression gets
reported.

Two independent throughput readings per build, because they separate the two
candidate causes and the report cannot:

  * `llama-bench`  -- pure kernel throughput, NO sampler runs at all
  * `llama-completion` -- a real generation, so the sampler chain does run

If only the second one collapses, the cause is the sampler backend fallback and
not `get_rows`. The generation leg is measured twice more, at `--top-k 40`
(the default, the op named in the warning) and at `--top-k 0` (top-k disabled),
which is the same question asked a second way.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import tarfile
import time
import urllib.request
import zipfile
from pathlib import Path

PROMPT = (
    "Write a detailed technical explanation of how a modern operating system "
    "schedules threads across multiple CPU cores, covering run queues, load "
    "balancing and priority inheritance."
)

# `eval time = 1234.56 ms / 128 runs ( 9.64 ms per token, 103.72 tokens per second)`
#
# The lookbehind is load-bearing. `prompt eval time` CONTAINS ` eval time`, it is
# printed first, and `search` returns the first match: without it every decode
# reading here was silently the PREFILL rate. Caught by the smoke run, where the
# two differ by 5x.
_EVAL_RE = re.compile(
    r"(?<!prompt )eval time\s*=.*?\(\s*[\d.]+\s*ms per token,\s*([\d.]+)\s*tokens per second\)"
)
_PROMPT_EVAL_RE = re.compile(
    r"prompt eval time\s*=.*?\(\s*[\d.]+\s*ms per token,\s*([\d.]+)\s*tokens per second\)"
)
# `load_tensors: layer 12 assigned to device ROCm0, is_swa = 0`. This is a debug
# line, so the generation runs pass -v; without it llama.cpp says nothing about
# where the weights went and a CPU-only run is indistinguishable from a GPU one.
# `system_info` is no substitute: it names ROCm even when nothing ran on it.
_LAYER_DEV_RE = re.compile(r"load_tensors:\s+layer\s+\d+\s+assigned to device\s+([A-Za-z0-9_]+)")
_TOPK_WARN_RE = re.compile(
    r"llama_sampler_backend_support:.*does not have support for op (\w+)"
)


def _is_windows() -> bool:
    return os.name == "nt" or sys.platform.startswith("win")


def _exe(name: str) -> str:
    return f"{name}.exe" if _is_windows() else name


def default_asset(tag: str) -> str:
    """The gfx1151 ROCm bundle for this OS."""
    if _is_windows():
        return f"app-{tag}-windows-x64-rocm-gfx1151.zip"
    return f"app-{tag}-linux-x64-rocm-gfx1151.tar.gz"


def download(url: str, dest: Path, log: list) -> bool:
    if dest.is_file() and dest.stat().st_size > 0:
        log.append({"download": url, "cached": True, "bytes": dest.stat().st_size})
        return True
    dest.parent.mkdir(parents = True, exist_ok = True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    t0 = time.time()
    try:
        req = urllib.request.Request(url, headers = {"User-Agent": "amd-ci/1.0"})
        with urllib.request.urlopen(req, timeout = 300) as r, open(tmp, "wb") as fh:
            shutil.copyfileobj(r, fh, 1 << 20)
        tmp.replace(dest)
    except Exception as e:  # noqa: BLE001
        log.append({"download": url, "error": f"{type(e).__name__}: {e}"[:300]})
        return False
    log.append({"download": url, "cached": False,
                "bytes": dest.stat().st_size, "seconds": round(time.time() - t0, 1)})
    return True


def extract(archive: Path, dest: Path, log: list) -> bool:
    marker = dest / ".extracted"
    if marker.is_file():
        log.append({"extract": str(archive), "cached": True})
        return True
    dest.mkdir(parents = True, exist_ok = True)
    try:
        if archive.name.endswith(".zip"):
            with zipfile.ZipFile(archive) as z:
                z.extractall(dest)
        else:
            # encoding names how MEMBER NAMES are decoded, and tarfile defaults to
            # the locale, which is not utf-8 everywhere. Same rule as every
            # other text read here.
            with tarfile.open(archive, "r:gz", encoding = "utf-8") as t:
                t.extractall(dest)
    except Exception as e:  # noqa: BLE001
        log.append({"extract": str(archive), "error": f"{type(e).__name__}: {e}"[:300]})
        return False
    if not _is_windows():
        for p in dest.rglob("llama-*"):
            if p.is_file():
                try:
                    p.chmod(0o755)
                except OSError:
                    pass
    marker.write_text("ok", encoding = "utf-8")
    # The archive has served its purpose and a sweep holds a dozen of them; the
    # extraction marker is what makes a second pass free, not the tarball.
    try:
        archive.unlink()
    except OSError:
        pass
    log.append({"extract": str(archive), "cached": False})
    return True


def find_binary(root: Path, name: str) -> Path | None:
    """The bundles flatten to the archive root, but do not assume it."""
    direct = root / _exe(name)
    if direct.is_file():
        return direct
    for p in root.rglob(_exe(name)):
        if p.is_file():
            return p
    return None


def run_tool(binary: Path, args: list[str], timeout: int, log: list,
             tag: str, what: str) -> dict:
    """Run a bundled tool with the bundle's own libraries ahead of the system's."""
    env = dict(os.environ)
    lib_dir = str(binary.parent)
    if _is_windows():
        env["PATH"] = lib_dir + os.pathsep + env.get("PATH", "")
    else:
        env["LD_LIBRARY_PATH"] = lib_dir + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    t0 = time.time()
    try:
        p = subprocess.run([str(binary), *args], capture_output = True, text = True,
                           timeout = timeout, env = env, cwd = lib_dir,
                           encoding = "utf-8", errors = "replace",
                           stdin = subprocess.DEVNULL)
        rc, out, err = p.returncode, p.stdout, p.stderr
        timed_out = False
    except subprocess.TimeoutExpired as e:
        rc, timed_out = None, True
        out = (e.stdout or b"").decode("utf-8", "replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
        err = (e.stderr or b"").decode("utf-8", "replace") if isinstance(e.stderr, bytes) else (e.stderr or "")
    seconds = round(time.time() - t0, 2)
    log.append({"tag": tag, "what": what, "rc": rc, "timed_out": timed_out,
                "seconds": seconds, "argv": args,
                "stderr_tail": err[-4000:], "stdout_tail": out[-2000:]})
    return {"rc": rc, "timed_out": timed_out, "seconds": seconds,
            "stdout": out, "stderr": err}


def parse_bench(stdout: str) -> list[dict]:
    """`llama-bench -o json` writes a JSON array to stdout; ggml banners go to
    stderr, but take the bracketed region rather than trusting that."""
    start, end = stdout.find("["), stdout.rfind("]")
    if start < 0 or end <= start:
        return []
    try:
        rows = json.loads(stdout[start:end + 1])
    except Exception:  # noqa: BLE001
        return []
    keep = ("model_filename", "n_prompt", "n_gen", "avg_ts", "stddev_ts", "backends",
            "n_gpu_layers", "build_number", "build_commit", "gpu_info")
    return [{k: r.get(k) for k in keep if k in r} for r in rows if isinstance(r, dict)]


def layer_devices(text: str) -> dict:
    """How many layers landed on each device, counted the way #9792 counted them."""
    out: dict = {}
    for dev in _LAYER_DEV_RE.findall(text):
        out[dev] = out.get(dev, 0) + 1
    return out


def measure_build(tag: str, root: Path, model: Path, args, log: list) -> dict:
    """Every reading for one prebuilt. Observes; decides nothing."""
    rec: dict = {"tag": tag, "root": str(root)}
    bench_bin = find_binary(root, "llama-bench")
    comp_bin = find_binary(root, "llama-completion")
    rec["llama_bench_present"] = bench_bin is not None
    rec["llama_completion_present"] = comp_bin is not None
    if bench_bin is None or comp_bin is None:
        rec["error"] = "bundle is missing llama-bench or llama-completion"
        return rec

    devices = run_tool(bench_bin, ["--list-devices"], 600, log, tag, "list-devices")
    rec["devices_raw"] = (devices["stdout"] + devices["stderr"])[-3000:]

    # --- kernel throughput, no sampler anywhere in the path
    bench = run_tool(
        bench_bin,
        ["-m", str(model), "-ngl", "999", "-p", str(args.n_prompt), "-n", str(args.n_gen),
         "-r", str(args.repeats), "-o", "json"],
        args.timeout, log, tag, "llama-bench")
    rows = parse_bench(bench["stdout"])
    rec["bench"] = {"rc": bench["rc"], "timed_out": bench["timed_out"], "rows": rows}
    for r in rows:
        if r.get("n_gen"):
            rec["bench_tg_ts"] = r.get("avg_ts")
            rec["bench_tg_stddev"] = r.get("stddev_ts")
        elif r.get("n_prompt"):
            rec["bench_pp_ts"] = r.get("avg_ts")
    rec["bench_backends"] = next((r.get("backends") for r in rows if r.get("backends")), None)
    rec["bench_gpu_info"] = next((r.get("gpu_info") for r in rows if r.get("gpu_info")), None)
    rec["bench_build_number"] = next(
        (r.get("build_number") for r in rows if r.get("build_number")), None)

    # --- real generation, so the sampler chain runs. Two top-k settings: the op
    # named in the #7371 warning, and the same generation with it disabled.
    gens: dict = {}
    for label, topk in (("topk40", args.top_k), ("topk0", 0)):
        samples: list[float] = []
        pp_samples: list[float] = []
        ops: list[str] = []
        bufs: dict = {}
        last_rc = None
        for _ in range(args.gen_repeats):
            g = run_tool(
                comp_bin,
                ["-m", str(model), "-ngl", "999", "-c", str(args.ctx),
                 "-n", str(args.n_gen), "-p", PROMPT, "-st", "-no-cnv", "--no-warmup",
                 "--perf", "-s", "1234", "--temp", "0.7", "--top-k", str(topk),
                 "-t", str(args.threads), "-v"],
                args.timeout, log, tag, f"llama-completion-{label}")
            blob = g["stderr"] + g["stdout"]
            m = _EVAL_RE.search(blob)
            if m:
                samples.append(float(m.group(1)))
            mp = _PROMPT_EVAL_RE.search(blob)
            if mp:
                pp_samples.append(float(mp.group(1)))
            ops += _TOPK_WARN_RE.findall(blob)
            for dev, n in layer_devices(blob).items():
                bufs[dev] = max(bufs.get(dev, 0), n)
            last_rc = g["rc"]
        gens[label] = {
            "tg_ts_samples": samples,
            "tg_ts": round(statistics.median(samples), 3) if samples else None,
            "tg_ts_spread_pct": (
                round(100.0 * (max(samples) - min(samples)) / max(samples), 2)
                if len(samples) > 1 and max(samples) > 0 else None),
            "pp_ts": round(statistics.median(pp_samples), 3) if pp_samples else None,
            "unsupported_sampler_ops": sorted(set(ops)),
            "layer_devices": bufs,
            "rc": last_rc,
        }
    rec["generation"] = gens
    rec["layer_devices"] = gens.get("topk40", {}).get("layer_devices", {})
    rec["gen_tg_ts"] = gens.get("topk40", {}).get("tg_ts")
    rec["gen_tg_ts_topk0"] = gens.get("topk0", {}).get("tg_ts")
    rec["unsupported_sampler_ops"] = gens.get("topk40", {}).get("unsupported_sampler_ops", [])
    return rec


def parse_tag_map(raw: str) -> dict:
    out = {}
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        name, _, tag = part.partition("=")
        out[name.strip()] = tag.strip()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--repo", default = "unslothai/llama.cpp")
    ap.add_argument("--tag-map", required = True,
                    help = "state=release-tag pairs, e.g. base=b10079-mix-fb3d4ca,head=...")
    ap.add_argument("--reference-tag", required = True,
                    help = "measured in EVERY state as the negative control")
    ap.add_argument("--sweep-tags", default = "",
                    help = "extra tags measured ONLY in the base state, to locate which "
                           "build a regression entered on. They are observations beside "
                           "the differential, never states in it: adding a state would "
                           "put them in the verdict, and 'which build is slow' is a "
                           "different question from 'does the reported defect reproduce'")
    ap.add_argument("--model-repo", default = "unsloth/Qwen3-4B-Instruct-2507-GGUF")
    ap.add_argument("--model-file", default = "Qwen3-4B-Instruct-2507-Q4_K_M.gguf")
    ap.add_argument("--cache-dir", default = None)
    ap.add_argument("--repeats", type = int, default = 3)
    ap.add_argument("--gen-repeats", type = int, default = 3)
    ap.add_argument("--n-prompt", type = int, default = 512)
    ap.add_argument("--n-gen", type = int, default = 128)
    ap.add_argument("--ctx", type = int, default = 4096)
    ap.add_argument("--top-k", type = int, default = 40)
    ap.add_argument("--threads", type = int, default = 8)
    ap.add_argument("--timeout", type = int, default = 3600)
    args = ap.parse_args()

    log: list = []
    tag_map = parse_tag_map(args.tag_map)
    state_tag = tag_map.get(args.state)
    obs: dict = {
        "state": args.state,
        "checkout": args.checkout,
        "state_tag": state_tag,
        "reference_tag": args.reference_tag,
        "repo": args.repo,
        "platform": platform.platform(),
        "os": "windows" if _is_windows() else sys.platform,
        "model": f"{args.model_repo}/{args.model_file}",
        "settings": {"n_prompt": args.n_prompt, "n_gen": args.n_gen,
                     "repeats": args.repeats, "gen_repeats": args.gen_repeats,
                     "ctx": args.ctx, "top_k": args.top_k, "threads": args.threads},
    }
    if not state_tag:
        obs["error"] = f"no release tag mapped for state {args.state!r} in --tag-map"
        args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
        return 0

    cache = Path(args.cache_dir) if args.cache_dir else Path(
        os.environ.get("AMD_CI_WORK") or os.environ.get("RUNNER_TEMP") or ".") / "llamacpp-cache"
    # Absolute, always: every tool is run with `cwd` set to its own bundle so it
    # finds its libraries, and a relative binary or model path does not survive
    # that. Same trap states.py documents for `--root`.
    cache = cache.expanduser().resolve()
    cache.mkdir(parents = True, exist_ok = True)
    obs["cache_dir"] = str(cache)

    model = cache / args.model_file
    model_url = f"https://huggingface.co/{args.model_repo}/resolve/main/{args.model_file}"
    if not download(model_url, model, log):
        obs["error"] = "model download failed"
        obs["_log"] = log
        args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
        return 0
    obs["model_bytes"] = model.stat().st_size

    wanted: list[tuple[str, str]] = [("reference", args.reference_tag), ("state", state_tag)]
    # The sweep rides along with the base state only: the head state is measured
    # to answer the differential, and repeating a dozen builds there would double
    # the job for no extra information.
    if args.state == "base":
        wanted += [("sweep", t.strip()) for t in args.sweep_tags.split(",") if t.strip()]
    obs["sweep_tags"] = [t for role, t in wanted if role == "sweep"]

    builds: dict = {}
    # Reference FIRST so a run that dies partway still carries the control.
    for role, tag in wanted:
        if tag in builds:
            continue
        asset = default_asset(tag)
        url = f"https://github.com/{args.repo}/releases/download/{tag}/{asset}"
        archive = cache / asset
        root = cache / f"x-{tag}"
        # Already extracted in an earlier state's probe: the archive was deleted
        # then, so asking download() for it would fetch 400 MB to throw away.
        already = (root / ".extracted").is_file()
        if not already and not download(url, archive, log):
            builds[tag] = {"tag": tag, "error": f"could not download {asset}"}
            continue
        if not extract(archive, root, log):
            builds[tag] = {"tag": tag, "error": f"could not extract {asset}"}
            continue
        builds[tag] = measure_build(tag, root, model, args, log)
        builds[tag]["role"] = role

    obs["builds"] = builds
    obs["reference"] = builds.get(args.reference_tag, {})
    obs["measured"] = builds.get(state_tag, {})
    obs["_log"] = log
    args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
