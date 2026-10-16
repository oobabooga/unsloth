#!/usr/bin/env python3
"""Probe: Studio's on-demand SSM kernel install (causal-conv1d / mamba-ssm), per state, on real
hardware. Observes only; ondemand_criteria.py judges.

Per state, starting from a venv with neither kernel installed:
  setup  - the state's setup SSM step for real (9f5b13325: _ensure_ssm_kernels; later states:
           _remove_stale_ssm_kernels), timed, subprocess / HTTP calls logged.
  for each model, in order:
    infer  - the state's inference worker _ensure_ssm_kernels (what a chat load runs first)
    train  - the state's training worker SSM block (the code before `import transformers`)
    run    - a fresh process with the venv's (shared) Unsloth: load, 4 LoRA steps, 16-token greedy
             decode, peak memory, conv / scan kernels seen by the profiler.
  Kernels a state installed are uninstalled afterwards, so every state starts equal. A state
  named in --decision-only only records the per-model kernel decisions.

Each piece runs in its own subprocess so one failure cannot hide the rest. JSON goes to --out.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

MODELS = [
    ("plain", "trl-internal-testing/tiny-Qwen3ForCausalLM"),
    ("qwen35", "unsloth/Qwen3.5-0.8B"),
    ("mamba2", "yujiepan/mamba2-tiny-random"),
    ("nemotronh", "trl-internal-testing/tiny-NemotronHForCausalLM-3.5-lightning"),
    ("falconh1", "tiiuae/Falcon-H1-Tiny-R-90M"),
]

COMMON = r"""
import json, os, sys, time, subprocess, inspect, urllib.request, http.client
LOG = []
def _fmt(cmd):
    if isinstance(cmd, (list, tuple)):
        return " ".join(str(x) for x in cmd)[:400]
    return str(cmd)[:400]
_orig_run = subprocess.run
def _run(*a, **k):
    t = time.time()
    cmd = a[0] if a else k.get("args")
    rc = None
    try:
        r = _orig_run(*a, **k)
        rc = getattr(r, "returncode", None)
        return r
    finally:
        LOG.append({"run": _fmt(cmd), "s": round(time.time() - t, 2), "rc": rc})
subprocess.run = _run
_OrigPopen = subprocess.Popen
class _Popen(_OrigPopen):
    def __init__(self, args, *a, **k):
        LOG.append({"popen": _fmt(args)})
        super().__init__(args, *a, **k)
subprocess.Popen = _Popen
_orig_urlopen = urllib.request.urlopen
def _urlopen(req, *a, **k):
    LOG.append({"http": getattr(req, "full_url", str(req))[:300]})
    return _orig_urlopen(req, *a, **k)
urllib.request.urlopen = _urlopen
_orig_req = http.client.HTTPConnection.request
def _request(self, method, url, *a, **k):
    LOG.append({"http": f"{method} {self.host}{url}"[:300]})
    return _orig_req(self, method, url, *a, **k)
http.client.HTTPConnection.request = _request
class Q:
    def __init__(self):
        self.items = []
    def put(self, x, *a, **k):
        self.items.append(x)
def _dump(tag, obj):
    print(tag + " " + json.dumps(obj, default = str), flush = True)
"""

BLOCK = COMMON + r"""
kind, studio, model = sys.argv[1], sys.argv[2], sys.argv[3]
backend = os.path.join(studio, "backend")
sys.path[:0] = [backend, studio]
os.chdir(backend)
out = {"kind": kind, "model": model}
t0 = time.time()
try:
    if kind == "setup":
        import install_python_stack as ips
        out["has_ensure_ssm_kernels"] = hasattr(ips, "_ensure_ssm_kernels")
        out["has_remove_stale"] = hasattr(ips, "_remove_stale_ssm_kernels")
        if out["has_ensure_ssm_kernels"]:
            ips._ensure_ssm_kernels()
        if out["has_remove_stale"]:
            ips._remove_stale_ssm_kernels()
    elif kind == "decide":
        from utils import ssm_runtime as s
        d = {"name_wants_cc1d": s.model_wants_causal_conv1d(model), "name_is_ssm": s.model_is_ssm(model)}
        if hasattr(s, "resolve_ssm_kernels"):
            k = s.resolve_ssm_kernels(model, model, None)
            d.update(api = "resolve_ssm_kernels", cc1d = k.causal_conv1d, ssm = k.mamba_ssm,
                     required = getattr(k, "mamba_ssm_required", None))
        elif hasattr(s, "resolved_model_is_ssm"):
            ssm = s.resolved_model_is_ssm(model, model, None)
            d.update(api = "resolved_model_is_ssm", ssm = ssm,
                     cc1d = ssm or s.resolved_model_wants_causal_conv1d(model, model, None))
        else:
            d.update(api = "name", cc1d = s.resolved_model_wants_causal_conv1d(model, model, None),
                     ssm = s.model_is_ssm(model))
        out["decision"] = d
    elif kind == "infer":
        import core.inference.worker as iw
        from utils.ssm_runtime import ssm_probe_identifier
        q = Q()
        probe = ssm_probe_identifier(model, None)
        params = inspect.signature(iw._ensure_ssm_kernels).parameters
        if "hf_token" in params:
            ok = iw._ensure_ssm_kernels([(probe, model)], q, None)
        else:
            ok = iw._ensure_ssm_kernels([probe], q)
        out["ok"] = ok
        out["responses"] = q.items
    elif kind == "train":
        import core.training.worker as w
        from utils import ssm_runtime as s
        q = Q()
        if hasattr(s, "resolve_ssm_kernels"):
            k = s.resolve_ssm_kernels(model, model, None, adapter_base = None)
            is_ssm, wants = k.mamba_ssm, k.causal_conv1d
            api = "C"
        elif hasattr(s, "resolved_model_is_ssm"):
            is_ssm = s.resolved_model_is_ssm(model, model, None)
            wants = is_ssm or s.resolved_model_wants_causal_conv1d(model, model, None)
            api = "B"
        else:
            is_ssm = None
            wants = s.resolved_model_wants_causal_conv1d(model, model, None)
            api = "A"
        out.update(api = api, wants = wants, is_ssm = is_ssm)
        w._ensure_causal_conv1d_fast_path(q, model, required = wants)
        w._install_fast_path_hooks(q, model, install_causal_conv1d = wants)
        if "required" in inspect.signature(w._ensure_mamba_ssm).parameters:
            w._ensure_mamba_ssm(q, model, required = is_ssm)
        else:
            w._ensure_mamba_ssm(q, model)
        out["events"] = q.items
    out["error"] = None
except BaseException as e:
    import traceback
    out["error"] = f"{type(e).__name__}: {e}"
    out["tb"] = traceback.format_exc()[-2000:]
out["seconds"] = round(time.time() - t0, 2)
out["calls"] = LOG
_dump("BLOCK_JSON", out)
sys.stdout.flush(); sys.stderr.flush(); os._exit(0)
"""

STATE = r"""
import importlib.metadata, json, sys
out = {}
for d in ("causal-conv1d", "mamba-ssm"):
    try:
        out[d] = importlib.metadata.version(d)
    except importlib.metadata.PackageNotFoundError:
        out[d] = None
for m in ("causal_conv1d", "causal_conv1d_cuda", "mamba_ssm", "selective_scan_cuda"):
    try:
        __import__(m)
        out["import_" + m] = True
    except BaseException as e:
        out["import_" + m] = f"{type(e).__name__}: {str(e)[:200]}"
print("STATE_JSON " + json.dumps(out), flush = True)
import os; os._exit(0)
"""

DECODE = r"""
import json, os, sys, time
model_id = sys.argv[1]
out = {"model": model_id, "attempts": []}
import torch
dev = "cuda" if torch.cuda.is_available() else "cpu"
t0 = time.time()

def _gen(model, tok, how):
    t = getattr(tok, "tokenizer", tok)  # Qwen3.5 loads a multimodal processor
    ids = t("The capital of France is", return_tensors = "pt").input_ids.to(dev)
    t1 = time.time()
    with torch.no_grad():
        gen = model.generate(input_ids = ids, max_new_tokens = 16, do_sample = False)
    out["path"] = how
    out["decode_s"] = round(time.time() - t1, 2)
    out["decode_tokens"] = gen[0, ids.shape[1]:].tolist()
    out["text"] = t.decode(gen[0, ids.shape[1]:])

try:
    from unsloth import FastLanguageModel
    model, tok = FastLanguageModel.from_pretrained(model_id, max_seq_length = 512,
        dtype = torch.bfloat16, load_in_4bit = False)
    FastLanguageModel.for_inference(model)
    _gen(model, tok, "unsloth for_inference")
except BaseException as e:
    out["attempts"].append(f"unsloth for_inference: {type(e).__name__}: {str(e)[:300]}")
if "decode_tokens" not in out:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype = torch.bfloat16,
                                                     attn_implementation = "eager").to(dev).eval()
        _gen(model, tok, "transformers model.generate")
    except BaseException as e:
        out["attempts"].append(f"transformers generate: {type(e).__name__}: {str(e)[:300]}")
        out["decode_error"] = out["attempts"][-1]
out["total_s"] = round(time.time() - t0, 2)
print("DECODE_JSON " + json.dumps(out, default = str), flush = True)
sys.stderr.flush(); os._exit(0)
"""

TRAIN = r"""
import json, os, re, sys, time
model_id = sys.argv[1]
out = {"model": model_id}
t_start = time.time()
import torch
out["torch"] = torch.__version__
out["hip"] = getattr(torch.version, "hip", None)
out["cuda_available"] = torch.cuda.is_available()
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    out["device"] = torch.cuda.get_device_name(0)
    out["arch"] = getattr(p, "gcnArchName", None)
dev = "cuda" if torch.cuda.is_available() else "cpu"
model = tok = None
t_load = time.time()
try:
    import unsloth
    from unsloth import FastLanguageModel
    out["unsloth_file"] = unsloth.__file__
    model, tok = FastLanguageModel.from_pretrained(model_id, max_seq_length = 512,
        dtype = torch.bfloat16, load_in_4bit = False)
    model = FastLanguageModel.get_peft_model(model, r = 8, lora_alpha = 8,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj",
                          "down_proj", "in_proj", "out_proj"],
        use_gradient_checkpointing = "unsloth", random_state = 0)
    out["path"] = "unsloth"
except BaseException as e:
    out["unsloth_error"] = f"{type(e).__name__}: {str(e)[:500]}"
    model = None
if model is None:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, get_peft_model
        tok = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype = torch.bfloat16,
                                                     attn_implementation = "eager").to(dev)
        try:
            model = get_peft_model(model, LoraConfig(r = 8, lora_alpha = 8, target_modules = "all-linear"))
        except ValueError:
            # PEFT refuses out_proj / conv1d on Mamba models; in_proj is the LoRA-able projection.
            model = get_peft_model(model, LoraConfig(r = 8, lora_alpha = 8, target_modules = ["in_proj"]))
        out["path"] = "transformers"
    except BaseException as e:
        out["load_error"] = f"{type(e).__name__}: {str(e)[:500]}"
        print("TRAIN_JSON " + json.dumps(out, default = str), flush = True)
        os._exit(0)
out["model_type"] = getattr(model.config, "model_type", None)
vocab = int(getattr(model.config, "vocab_size", 1000) or 1000)
g = torch.Generator().manual_seed(0)
ids = torch.randint(10, max(11, min(vocab, 20000)), (2, 128), generator = g).to(dev)
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr = 1e-4)
model.train()
losses, kernels = [], set()
from torch.profiler import profile, ProfilerActivity
acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if dev == "cuda" else [])
if dev == "cuda":
    torch.cuda.reset_peak_memory_stats()
try:
    for step in range(4):
        if step == 0:
            with profile(activities = acts) as prof:
                loss = model(input_ids = ids, labels = ids).loss
                loss.backward()
                if dev == "cuda":
                    torch.cuda.synchronize()
            for e in prof.key_averages():
                if re.search(r"conv1d|causal_conv|selective|ssd|chunk_scan|chunk_state|mamba|depthwise|convolution", e.key, re.I):
                    kernels.add(e.key[:100])
        else:
            loss = model(input_ids = ids, labels = ids).loss
            loss.backward()
        opt.step(); opt.zero_grad(set_to_none = True)
        losses.append(float(loss.detach().float().cpu()))
        if step == 0:
            out["load_to_first_step_s"] = round(time.time() - t_load, 2)
    if dev == "cuda":
        torch.cuda.synchronize()
        out["peak_alloc_gb"] = torch.cuda.max_memory_allocated() / 1e9
except BaseException as e:
    out["train_error"] = f"{type(e).__name__}: {str(e)[:500]}"
out["losses"] = losses
out["kernels"] = sorted(kernels)
out["total_s"] = round(time.time() - t_start, 2)
print("TRAIN_JSON " + json.dumps(out, default = str), flush = True)
sys.stderr.flush(); os._exit(0)
"""


def _run(cmd, env, cwd, timeout, tag):
    t = time.time()
    try:
        p = subprocess.run(cmd, env = env, cwd = cwd, capture_output = True, text = True,
                           encoding = "utf-8", errors = "replace", timeout = timeout)
        rc, text = p.returncode, (p.stdout or "") + "\n" + (p.stderr or "")
    except subprocess.TimeoutExpired as e:
        rc = "timeout"
        text = str(e.stdout or "")[-2000:] + "\nTimeoutExpired"
    payload = None
    for line in text.splitlines():
        if line.startswith(tag + " "):
            try:
                payload = json.loads(line[len(tag) + 1:])
            except Exception:  # noqa: BLE001
                pass
    return {"rc": rc, "wall_s": round(time.time() - t, 2), "data": payload,
            "tail": text[-2500:] if payload is None or rc not in (0,) else text[-600:]}


def _uninstall(py: str, env: dict) -> str:
    uv = shutil.which("uv")
    cmd = ([uv, "pip", "uninstall", "--python", py, "causal-conv1d", "mamba-ssm"] if uv
           else [py, "-m", "pip", "uninstall", "-y", "causal-conv1d", "mamba-ssm"])
    try:
        p = subprocess.run(cmd, env = env, capture_output = True, text = True, timeout = 300)
        return f"rc={p.returncode} {(p.stdout + p.stderr)[-300:]}"
    except Exception as e:  # noqa: BLE001
        return f"ERR {e}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--python", default = sys.executable)
    ap.add_argument("--decision-only", default = "main")
    ap.add_argument("--models", default = ",".join(m for m, _ in MODELS))
    ap.add_argument("--budget-min", type = float, default = 150)
    ap.add_argument("--block-timeout", type = int, default = 5400)
    args = ap.parse_args()

    t_state = time.time()
    studio = str(Path(args.checkout) / "studio")
    work = Path(os.environ.get("V4_SCRATCH") or args.out.parent)
    work.mkdir(parents = True, exist_ok = True)
    py = args.python
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    # Each state builds from scratch: no built-wheel cache shared between states.
    env["UV_CACHE_DIR"] = str(work / f"uvcache_{args.state}")
    env["PIP_NO_CACHE_DIR"] = "1"
    env["UNSLOTH_COMPILE_LOCATION"] = str(work / f"compiled_{args.state}")
    env.setdefault("PYTHONHASHSEED", "0")
    obs: dict = {"state": args.state, "platform": sys.platform,
                 "hipcc": shutil.which("hipcc"), "uv": shutil.which("uv")}
    try:
        obs["commit"] = subprocess.run(["git", "-C", args.checkout, "rev-parse", "HEAD"],
                                       capture_output = True, text = True).stdout.strip()
    except OSError:
        obs["commit"] = None
    wanted = [(k, m) for k, m in MODELS if k in args.models.split(",")]

    obs["decide"] = {k: _run([py, "-c", BLOCK, "decide", studio, m], env, studio, 600, "BLOCK_JSON")
                     for k, m in wanted}
    if args.state in args.decision_only.split(","):
        obs["decision_only"] = True
        args.out.write_text(json.dumps(obs, indent = 2, default = str), encoding = "utf-8")
        return 0

    obs["reset_before"] = _uninstall(py, env)
    obs["kernels_before"] = _run([py, "-c", STATE], env, str(work), 300, "STATE_JSON")
    obs["setup"] = _run([py, "-c", BLOCK, "setup", studio, "-"], env, studio,
                        args.block_timeout, "BLOCK_JSON")
    obs["kernels_after_setup"] = _run([py, "-c", STATE], env, str(work), 300, "STATE_JSON")
    obs["models"] = {}
    for key, model in wanted:
        if time.time() - t_state > args.budget_min * 60:
            obs["models"][key] = {"skipped": f"state budget {args.budget_min} min spent"}
            continue
        m: dict = {"model": model}
        m["infer"] = _run([py, "-c", BLOCK, "infer", studio, model], env, studio,
                          args.block_timeout, "BLOCK_JSON")
        m["train_block"] = _run([py, "-c", BLOCK, "train", studio, model], env, studio,
                                args.block_timeout, "BLOCK_JSON")
        m["kernels"] = _run([py, "-c", STATE], env, str(work), 300, "STATE_JSON")
        runs = 2 if key == "qwen35" else 1
        m["run"] = [_run([py, "-c", TRAIN, model], env, str(work), 2400, "TRAIN_JSON")
                    for _ in range(runs)]
        m["decode"] = _run([py, "-c", DECODE, model], env, str(work), 1800, "DECODE_JSON")
        obs["models"][key] = m
        args.out.write_text(json.dumps(obs, indent = 2, default = str), encoding = "utf-8")
    obs["reset_after"] = _uninstall(py, env)
    obs["state_wall_s"] = round(time.time() - t_state, 1)
    args.out.write_text(json.dumps(obs, indent = 2, default = str), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
