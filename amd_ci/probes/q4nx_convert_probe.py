#!/usr/bin/env python3
"""Probe: Studio's own convert_gguf_to_q4nx on a Studio-style Qwen3.5 fine-tune GGUF.

Builds once per job (cached beside the checkouts): random-weight Qwen3.5-0.8B with the real
config and tensor shapes, saved as Studio saves a merge (folder `model`, so llama.cpp names it
"Model" with no basename), converted with llama.cpp b12340 (Studio's pinned prebuilt) and
quantized to Q4_K_M. Then runs the checkout's convert_gguf_to_q4nx with the real pinned
converter and, when a q4nx comes out, the converter's own HF-vs-q4nx checker.
Observes only; criteria/q4nx_convert_fixed.py judges.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
import zipfile
from pathlib import Path

LLAMA_TAG = "b12340"
REPO = "unsloth/Qwen3.5-0.8B"
DEPS = ["torch", "safetensors", "einops", "numpy", "mpmath", "huggingface_hub", "transformers",
        "sentencepiece", "protobuf", "structlog", "pyjwt", "psutil", "fastapi", "pydantic", "pyyaml"]


def sh(cmd, **kw):
    return subprocess.run([str(c) for c in cmd], capture_output = True, text = True,
                          encoding = "utf-8", errors = "replace", **kw)


def deps(py: str, work: Path, log: list) -> None:
    marker = work / "deps.ok"
    if marker.exists() or os.environ.get("Q4NX_PROBE_SKIP_DEPS") == "1":
        return
    r = sh([py, "-m", "pip", "install", "-q", "torch", "--index-url", "https://download.pytorch.org/whl/cpu"])
    log.append(("pip torch", r.returncode, r.stderr[-400:]))
    r = sh([py, "-m", "pip", "install", "-q", *[d for d in DEPS if d != "torch"]])
    log.append(("pip deps", r.returncode, r.stderr[-400:]))
    if r.returncode == 0:
        marker.write_text("ok")


def llama(work: Path, log: list) -> tuple[Path, Path]:
    src = work / f"llama.cpp-{LLAMA_TAG}"
    if not (src / "convert_hf_to_gguf.py").is_file():
        r = sh(["git", "clone", "-q", "--depth", "1", "--branch", LLAMA_TAG,
                "https://github.com/ggml-org/llama.cpp", src])
        log.append(("clone llama.cpp", r.returncode, r.stderr[-400:]))
    win = os.name == "nt"
    asset = f"llama-{LLAMA_TAG}-bin-" + ("win-cpu-x64.zip" if win else "ubuntu-x64.tar.gz")
    bindir = work / "llama-bin"
    exe = next(bindir.rglob("llama-quantize.exe" if win else "llama-quantize"), None) if bindir.is_dir() else None
    if exe is None:
        bindir.mkdir(parents = True, exist_ok = True)
        archive = work / asset
        urllib.request.urlretrieve(f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_TAG}/{asset}", archive)
        if win:
            zipfile.ZipFile(archive).extractall(bindir)
        else:
            with tarfile.open(archive) as t:
                t.extractall(bindir)
        exe = next(bindir.rglob("llama-quantize.exe" if win else "llama-quantize"))
        if not win:
            exe.chmod(0o755)
    return src, exe


def build_gguf(py: str, work: Path, log: list) -> Path:
    out = work / "gguf" / "model.Q4_K_M.gguf"
    if out.is_file():
        return out
    src, quantize = llama(work, log)
    merged = work / "merge" / "_tmp_model_x" / "model"
    merged.mkdir(parents = True, exist_ok = True)
    script = (
        "import sys, shutil, torch\n"
        "from pathlib import Path\n"
        "from huggingface_hub import get_safetensors_metadata, snapshot_download\n"
        "from safetensors.torch import save_file\n"
        "repo, out = sys.argv[1], Path(sys.argv[2])\n"
        "cfg = Path(snapshot_download(repo, allow_patterns = ['*.json', '*.jinja', 'tokenizer*']))\n"
        "for p in cfg.iterdir():\n"
        "    if p.is_file() and p.name != 'model.safetensors.index.json': shutil.copy(p, out / p.name)\n"
        "g = torch.Generator().manual_seed(3407); t = {}\n"
        "for fm in get_safetensors_metadata(repo).files_metadata.values():\n"
        "    for n, m in fm.tensors.items():\n"
        "        x = torch.randn(m.shape, generator = g) * 0.02\n"
        "        t[n] = (1 + x if n.endswith('norm.weight') else x).to(torch.bfloat16)\n"
        "save_file(t, out / 'model.safetensors', metadata = {'format': 'pt'})\n"
        "(out / 'README.md').write_text('---\\nbase_model: ' + repo + '\\ntags:\\n- unsloth\\n---\\n')\n"
    )
    r = sh([py, "-c", script, REPO, merged])
    log.append(("random checkpoint", r.returncode, r.stderr[-600:]))
    out.parent.mkdir(parents = True, exist_ok = True)
    bf16 = out.with_name("model.BF16.gguf")
    env = {**os.environ, "PYTHONPATH": str(src / "gguf-py"), "PYTHONIOENCODING": "utf-8"}
    r = sh([py, src / "convert_hf_to_gguf.py", merged, "--outtype", "bf16", "--outfile", bf16], env = env)
    log.append(("convert_hf_to_gguf", r.returncode, r.stderr[-600:]))
    r = sh([quantize, bf16, out, "Q4_K_M"])
    log.append(("llama-quantize", r.returncode, (r.stderr or r.stdout)[-400:]))
    bf16.unlink(missing_ok = True)
    return out


CALL = r"""
import json, sys, time
from pathlib import Path
studio, root, gguf_path, out = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4])
config = json.loads(Path(sys.argv[5]).read_text(encoding = "utf-8"))
sys.path[:0] = [str(studio / "backend"), str(studio)]
import utils.paths.storage_roots as sr
sr.studio_root = lambda: root
from core.export import q4nx
takes_config = "config" in q4nx.convert_gguf_to_q4nx.__code__.co_varnames
t = time.time()
try:
    q4nx.convert_gguf_to_q4nx(gguf_path, out, config) if takes_config else q4nx.convert_gguf_to_q4nx(gguf_path, out)
    result = "OK"
except Exception as e:
    result = f"{type(e).__name__}: {e}"
print("Q4NX_RESULT " + json.dumps({"result": result, "seconds": round(time.time() - t, 1), "takes_config": takes_config}))
"""


def npu_presence() -> dict:
    if os.name == "nt":
        r = sh(["powershell", "-NoProfile", "-Command",
                "Get-PnpDevice -PresentOnly | Where-Object { $_.FriendlyName -match 'NPU|IPU' } | "
                "Select-Object -ExpandProperty FriendlyName"])
        flm = shutil.which("flm")
        return {"devices": [l.strip() for l in r.stdout.splitlines() if l.strip()], "flm": flm}
    accel = sorted(str(p) for p in Path("/dev/accel").glob("accel*")) if Path("/dev/accel").is_dir() else []
    return {"devices": accel, "amdxdna_loaded": Path("/sys/module/amdxdna").is_dir(), "flm": shutil.which("flm")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--python", default = sys.executable)
    a = ap.parse_args()
    checkout = Path(a.checkout).absolute()
    work = Path(os.environ.get("Q4NX_PROBE_WORK") or checkout.parent / "q4nx_work")
    work.mkdir(exist_ok = True)
    log: list = []
    obs: dict = {"state": a.state, "os": platform.platform(), "npu": npu_presence(), "steps": log}
    try:
        deps(a.python, work, log)
        gguf = build_gguf(a.python, work, log)
        meta = sh([a.python, "-c",
                   "import sys, json; sys.path.insert(0, sys.argv[2]); from gguf import GGUFReader; r = GGUFReader(sys.argv[1]); "
                   "f = lambda k: r.fields[k].contents() if k in r.fields else None; "
                   "print(json.dumps({'name': f('general.name'), 'basename': f('general.basename'), "
                   "'architecture': f('general.architecture'), 'block_count': f('qwen35.block_count'), "
                   "'nextn': f('qwen35.nextn_predict_layers'), 'blk0': sorted(t.name for t in r.tensors if t.name.startswith('blk.0.'))}))",
                   gguf, work / f"llama.cpp-{LLAMA_TAG}" / "gguf-py"])
        obs["gguf"] = json.loads(meta.stdout.strip().splitlines()[-1]) if meta.returncode == 0 else {"error": meta.stderr[-400:]}
        hf_ref = work / "merge" / "_tmp_model_x" / "model"
        config = hf_ref / "config.json"
        root = work / f"root_{a.state}"
        out = work / f"out_{a.state}"
        shutil.rmtree(out, ignore_errors = True)
        # The interpreter's own gguf when it has one (a Studio venv does), else llama.cpp's.
        own_gguf = sh([a.python, "-c", "import gguf"]).returncode == 0
        env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
        if not own_gguf:
            env["PYTHONPATH"] = str(work / f"llama.cpp-{LLAMA_TAG}" / "gguf-py")
        obs["own_gguf"] = own_gguf
        t = time.time()
        r = sh([a.python, "-c", CALL, checkout / "studio", root, gguf, out, config], env = env)
        line = next((l for l in reversed(r.stdout.splitlines()) if l.startswith("Q4NX_RESULT ")), None)
        obs["convert"] = json.loads(line[len("Q4NX_RESULT "):]) if line else {"result": "NO_RESULT", "stderr": r.stderr[-800:]}
        obs["convert"]["wall"] = round(time.time() - t, 1)
        q = out / "model.q4nx"
        if q.is_file():
            obs["q4nx_sha256"] = hashlib.sha256(q.read_bytes()).hexdigest()
            checker = next(root.rglob("check_qwen35_family.py"))
            c = sh([a.python, checker, "--hf", hf_ref, "--q4nx", q, "--layers", "all", "--mtp", "absent"],
                   cwd = checker.parent, env = env)
            obs["checker"] = {"rc": c.returncode, "tail": c.stdout.strip().splitlines()[-3:],
                              "n_pass": sum(l.startswith("PASS") for l in c.stdout.splitlines()),
                              "fails": [l for l in c.stdout.splitlines() if l.startswith("FAIL")][:10]}
    except Exception as e:  # observed, not judged
        obs["error"] = f"{type(e).__name__}: {e}"
    a.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
