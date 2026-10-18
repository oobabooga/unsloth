#!/usr/bin/env python3
"""Cloud VM driver: build the llama.cpp PR trees, fetch models, run bench.py for one plan.

  job_gpu.py --plan A|B|L4 [--arch 75] [--gpus 0,1]
Shipped next to bench.py and build.sh. Results: ./results_<plan>.json (brought back as an artifact).
"""

import argparse
import fnmatch
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.environ.get("RPC_WORK", "/tmp/rpcw")
BUILDS = {
    "base": "base/upstream-af50f2f24",
    "h187": "pull/187/head",
    "h191": "pull/191/head",
    "h199": "pull/199/head",
    "h203": "pull/203/head",
    "base_norpc": "base/upstream-af50f2f24",
}
REAL = {
    "qwen3-0.6b": ("unsloth/Qwen3-0.6B-GGUF", "*Q4_K_M*.gguf"),
    "qwen3.5-0.8b": ("unsloth/Qwen3.5-0.8B-GGUF", "*Q4_K_M*.gguf"),
    "gemma4-e2b": ("unsloth/gemma-4-E2B-it-GGUF", "*Q4_K_M*.gguf"),
    "lfm2-8b-a1b": ("unsloth/LFM2-8B-A1B-GGUF", "*Q4_K_M*.gguf"),
    "gemma4-12b": ("unsloth/gemma-4-12B-it-GGUF", "*Q4_K_M*.gguf"),
    "gpt-oss-20b": ("unsloth/gpt-oss-20b-GGUF", "*F16*.gguf"),
    "gemma4-26b-a4b": ("unsloth/gemma-4-26B-A4B-it-GGUF", "*UD-Q4_K_XL*.gguf"),
    "qwen3.6-35b-a3b": ("unsloth/Qwen3.6-35B-A3B-GGUF", "*UD-Q3_K_XL*.gguf"),
    "glm-4.7-flash-mla": ("unsloth/GLM-4.7-Flash-GGUF", "*Q4_K_M*.gguf"),
}
FAKES = ["qwen3-0.6b", "qwen3.5-0.8b", "qwen3.6-35b-a3b", "gpt-oss-20b", "lfm2-8b-a1b", "deepseek-v3-mla", "gemma4-tiny"]
FAKE_REPO = "danielhanchen/rpc-harness-fakes"


def sh(cmd, **kw):
    print("+", cmd if isinstance(cmd, str) else " ".join(cmd), flush = True)
    return subprocess.run(cmd, shell = isinstance(cmd, str), **kw)


def fetch(names, fakes):
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    out = {}
    for n in names:
        repo, pat = REAL[n]
        files = [f for f in api.list_repo_files(repo) if fnmatch.fnmatch(f, pat) and "mmproj" not in f.lower()]
        files.sort()
        if not files:
            print(f"no file for {n} {pat}", flush = True)
            continue
        t = time.time()
        paths = [hf_hub_download(repo, f) for f in files]  # split GGUFs: llama.cpp opens part 1
        out[n] = paths[0]
        print(f"fetched {n}: {files} in {time.time() - t:.0f}s", flush = True)
    for n in fakes:
        out["fake-" + n] = hf_hub_download(FAKE_REPO, f"{n}.gguf", repo_type = "dataset")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required = True)
    ap.add_argument("--arch", default = "75")
    ap.add_argument("--gpus", default = "0,1")
    ap.add_argument("--builds", default = "base,h187,h191,h199,h203")
    ap.add_argument("--backend", default = "cuda", choices = ["cuda", "hip"])
    a = ap.parse_args()
    t0 = time.time()
    os.makedirs(WORK, exist_ok = True)
    vendor = "amd" if a.backend == "hip" else "nvidia"
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        sh([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"])
    if vendor == "nvidia":
        sh("(command -v ccache || (apt-get -qq update && apt-get -qq install -y ccache) || pip install -q ccache) >/dev/null 2>&1; true")
        sh([sys.executable, "-m", "pip", "install", "-q", "nvidia-ml-py"])
        sh("nvidia-smi --query-gpu=index,name,memory.total,driver_version,compute_cap --format=csv")
    else:
        sh("rocm-smi --showproductname 2>/dev/null | head -20; ls -d /opt/rocm* 2>/dev/null; command -v hipcc; hipcc --version 2>/dev/null | head -3")
    sh("nproc; free -g | head -2; df -h /tmp | tail -1")
    builds = [b for b in a.builds.split(",") if b]
    r = sh(["bash", os.path.join(HERE, "build.sh"), WORK, a.backend, a.arch] + [f"{b}={BUILDS[b]}" for b in builds],
           capture_output = True, text = True)
    print(r.stdout[-6000:], r.stderr[-3000:], flush = True)
    bins = {}
    for line in r.stdout.splitlines():
        if line.startswith("BUILD_OK"):
            _, name, sha, secs, bindir = line.split()
            bins[name] = bindir
    print(f"builds ok: {sorted(bins)} after {time.time() - t0:.0f}s", flush = True)
    if "base" not in bins:
        json.dump({"fatal": "base build failed", "build_log": r.stdout[-8000:]}, open(f"results_{a.plan}.json", "w"))
        return 1

    if a.plan == "A":
        real = ["qwen3-0.6b", "qwen3.5-0.8b", "gemma4-e2b", "lfm2-8b-a1b", "gpt-oss-20b", "gemma4-26b-a4b", "qwen3.6-35b-a3b"]
        models = fetch(real, FAKES)
        fakes = ["fake-" + f for f in FAKES]
        args = ["--suite", "single,correct,teardown",
                "--single-models", "qwen3-0.6b,qwen3.5-0.8b,gemma4-e2b,lfm2-8b-a1b",
                "--correct-models", ",".join(fakes + ["qwen3-0.6b", "qwen3.5-0.8b", "gemma4-e2b", "lfm2-8b-a1b",
                                                      "gpt-oss-20b", "gemma4-26b-a4b", "qwen3.6-35b-a3b"]),
                "--no-single", "gemma4-26b-a4b,qwen3.6-35b-a3b", "--native",
                "--teardown-model", "qwen3-0.6b", "--rounds", "2", "--parallel", "8", "--ctx", "8192"]
    elif a.plan == "B":
        real = ["gpt-oss-20b", "gemma4-26b-a4b", "qwen3.6-35b-a3b", "gemma4-12b"]
        models = fetch(real, [])
        sb = [b for b in ("base", "h187", "h199", "h203") if b in bins]
        args = ["--suite", "groups,stress,load",
                "--groups-models", "gemma4-12b,qwen3.6-35b-a3b", "--groups-builds", "h187" if "h187" in bins else "",
                "--groups-topologies", "single,rpc2,native2", "--groups", "1,2", "--orders", "rpc_first,local_first",
                "--stress-models", "gemma4-26b-a4b,qwen3.6-35b-a3b,gpt-oss-20b", "--stress-builds", ",".join(sb),
                "--stress-topologies", "rpc2",
                "--load-models", "gemma4-26b-a4b,gpt-oss-20b",
                "--rounds", "2", "--parallel", "16", "--ctx", "16384"]
    elif a.plan == "PP":
        # Pipeline-groups scaling: per-group batch sweep, so the ideal N x X1(P/G) is measured
        # where X1 is not in the small-batch kernel regime.
        real = ["gemma4-12b", "lfm2-8b-a1b"]
        models = fetch(real, [])
        args = ["--suite", "groups", "--groups-models", ",".join(real), "--groups-builds", "h187",
                "--groups-topologies", "single,rpc2,native2", "--groups", "1,2", "--orders", "rpc_first",
                "--groups-parallel", "16,32,64", "--groups-patterns", "single,equal,saturate",
                "--groups-ctx-per-slot", "512", "--rounds", "1"]
    elif a.plan == "TRACE":
        # Where the idle time goes with 2 groups over RPC: #191's tracer on both processes, then
        # device_idle.py per case. 16 sequences per group (the efficient CUDA regime).
        real = ["gemma4-12b"]
        models = fetch(real, [])
        os.environ["BENCH_TRACE"] = "1"
        args = ["--suite", "groups", "--groups-models", "gemma4-12b", "--groups-builds", "h191",
                "--groups-topologies", "rpc2", "--groups", "1,2", "--orders", "rpc_first",
                "--groups-parallel", "32", "--groups-patterns", "saturate",
                "--groups-ctx-per-slot", "512", "--rounds", "1"]
    elif a.plan == "TS":
        # Stage balance with 2 groups: the output-head stage (the coordinator's GPU, last) is the
        # slow one at -ts 1,1, so give the first stage more layers.
        real = ["gemma4-12b", "lfm2-8b-a1b"]
        models = fetch(real, [])
        args = ["--suite", "groups", "--groups-models", ",".join(real), "--groups-builds", "h187",
                "--groups-topologies", "rpc2,native2", "--groups", "2", "--orders", "rpc_first",
                "--groups-parallel", "32", "--groups-patterns", "saturate",
                "--groups-ts", "1,1;1.2,1;1.4,1;1.7,1", "--groups-ctx-per-slot", "512", "--rounds", "1"]
    elif a.plan == "PP2":
        # Groups beyond the stage count hide the per-step latency (activation copy, sampling): 16
        # sequences per group, G = 1..4 over 2 stages; ideal = 2 x one device at 16 sequences.
        real = ["gemma4-12b", "lfm2-8b-a1b"]
        models = fetch(real, [])
        args = ["--suite", "groups", "--groups-models", ",".join(real), "--groups-builds", "h187",
                "--groups-topologies", "single,rpc2,native2", "--groups", "1,2,3,4", "--orders", "rpc_first",
                "--groups-batch", "16", "--groups-patterns", "saturate",
                "--groups-ctx-per-slot", "512", "--rounds", "1"]
    elif a.plan == "AMD":
        real = ["qwen3-0.6b", "qwen3.5-0.8b", "gemma4-e2b", "lfm2-8b-a1b", "gpt-oss-20b", "glm-4.7-flash-mla"]
        models = fetch(real, [])
        # One GPU: the RPC topology runs the rpc-server on the same device (the RPC code path, not overlap).
        args = ["--suite", "single,correct,teardown",
                "--single-models", "qwen3-0.6b,qwen3.5-0.8b,gemma4-e2b,lfm2-8b-a1b",
                "--correct-models", ",".join(real), "--teardown-model", "qwen3-0.6b",
                "--rounds", "2", "--parallel", "8", "--ctx", "8192"]
    else:  # single-GPU box (Colab L4 / G4): single-device no-regression + correctness
        real = ["qwen3-0.6b", "qwen3.5-0.8b", "gemma4-e2b", "lfm2-8b-a1b", "gpt-oss-20b"]
        models = fetch(real, FAKES)
        fakes = ["fake-" + f for f in FAKES]
        args = ["--suite", "single,correct",
                "--single-models", "qwen3-0.6b,qwen3.5-0.8b,gemma4-e2b,lfm2-8b-a1b,gpt-oss-20b",
                "--correct-models", ",".join(fakes + real), "--rounds", "2", "--parallel", "8", "--ctx", "8192"]
    cmd = [sys.executable, "-u", os.path.join(HERE, "bench.py"), "--out", f"results_{a.plan}.json",
           "--work", os.path.join(WORK, "run"), "--gpus", a.gpus, "--vendor", vendor,
           "--threads", str(max(2, (os.cpu_count() or 4) // 2))]
    cmd += [f"--bin={k}={v}" for k, v in bins.items()] + [f"--model={k}={v}" for k, v in models.items()] + args
    rc = sh(cmd).returncode
    print(f"bench rc={rc} total {time.time() - t0:.0f}s", flush = True)
    if a.plan == "TRACE":
        import glob
        for d in sorted(glob.glob(os.path.join(WORK, "run", "gr_*"))):
            traces = sorted(glob.glob(os.path.join(d, "trace_*.jsonl")))
            if len(traces) < 2:
                continue
            out = f"device_idle_{os.path.basename(d)}.txt"
            sh([sys.executable, os.path.join(HERE, "device_idle.py"), *traces, "--out", out])
            print(open(out).read()[:6000], flush = True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
