"""Timing A/B of two torch builds on this AMD GPU, ABBA, fresh process per measurement.

Arm A = torch 2.11.0+rocm7.2 (what the generic route installs today), arm B = 2.14.1+rocm7.2
(what #14350 installs for new users). Each measurement: Unsloth Qwen3-0.6B 16-bit, prefill +
greedy decode timing after warmup, then 20 LoRA steps (median step time after 5 warmup). Observes only.
"""

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import time

UNSLOTH = "git+https://github.com/unslothai/unsloth@a9d3f1e159d67cff61b26f7944927ae0348d1d1e"
ZOO = "git+https://github.com/unslothai/unsloth-zoo@e782f78c6f47ec77d06bf764f5c9fd920cdd0861"
IDX = "https://download.pytorch.org/whl/rocm7.2"
ARMS = {
    "A_2.11.0": ["torch==2.11.0+rocm7.2", "torchvision==0.26.0+rocm7.2", "torchaudio==2.11.0+rocm7.2"],
    "B_2.14.1": ["torch==2.14.1+rocm7.2", "torchvision==0.29.1+rocm7.2", "torchaudio==2.11.0+rocm7.2"],
}
ORDER = ["A_2.11.0", "B_2.14.1", "B_2.14.1", "A_2.11.0"] * 2

BENCH = r'''
import json, os, sys, time, statistics
import unsloth
from unsloth import FastLanguageModel
import torch
from datasets import Dataset
from trl import SFTConfig, SFTTrainer
from transformers import TrainerCallback
out = {"torch": torch.__version__}
model, tok = FastLanguageModel.from_pretrained("unsloth/Qwen3-0.6B", max_seq_length = 1024,
                                               dtype = torch.bfloat16, load_in_4bit = False)
prompt = "Write a long story about a robot who learns to paint. " * 20
ids = tok([prompt], return_tensors = "pt").to("cuda")
FastLanguageModel.for_inference(model)
def gen(n):
    torch.cuda.synchronize(); t = time.perf_counter()
    with torch.no_grad():
        o = model.generate(**ids, max_new_tokens = n, min_new_tokens = n, do_sample = False)
    torch.cuda.synchronize(); return time.perf_counter() - t, o.shape[1] - ids["input_ids"].shape[1]
gen(16); gen(16)
pre = [gen(1)[0] for _ in range(5)]
dec = []
for _ in range(3):
    t, n = gen(128)
    dec.append((n - 1) / (t - statistics.median(pre)))
out["prompt_tokens"] = int(ids["input_ids"].shape[1])
out["ttft_ms"] = round(statistics.median(pre) * 1000, 2)
out["prefill_tok_s"] = round(ids["input_ids"].shape[1] / statistics.median(pre), 1)
out["decode_tok_s"] = round(statistics.median(dec), 2)
out["infer_peak_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
FastLanguageModel.for_training(model)
model = FastLanguageModel.get_peft_model(model, r = 16, lora_alpha = 16, lora_dropout = 0, bias = "none",
    random_state = 3407, use_gradient_checkpointing = "unsloth",
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
rows = [f"Question: what is {i} plus {i+1}? Answer: {2*i+1}. " * 8 for i in range(200)]
ts = []
class Rec(TrainerCallback):
    def on_step_begin(self, a, s, c, **k): torch.cuda.synchronize(); self.t = time.perf_counter()
    def on_step_end(self, a, s, c, **k): torch.cuda.synchronize(); ts.append(time.perf_counter() - self.t)
torch.cuda.reset_peak_memory_stats()
tr = SFTTrainer(model = model, processing_class = tok, train_dataset = Dataset.from_dict({"text": rows}),
    callbacks = [Rec()], args = SFTConfig(output_dir = sys.argv[2], dataset_text_field = "text",
    max_length = 512, per_device_train_batch_size = 2, gradient_accumulation_steps = 1, max_steps = 25,
    learning_rate = 2e-4, logging_steps = 1, seed = 3407, bf16 = True, optim = "adamw_torch",
    report_to = "none", save_strategy = "no", dataset_num_proc = 1))
tr.train()
out["train_step_ms_median"] = round(statistics.median(ts[5:]) * 1000, 2)
out["train_peak_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
out["losses"] = [round(h["loss"], 4) for h in tr.state.log_history if "loss" in h]
json.dump(out, open(sys.argv[1], "w", encoding = "utf-8"))
'''

ap = argparse.ArgumentParser()
ap.add_argument("--work", required = True)
ap.add_argument("--out", required = True)
a = ap.parse_args()
os.makedirs(a.out, exist_ok = True)
uv = shutil.which("uv")
if uv is None:
    boot = os.path.join(a.work, "boot")
    if subprocess.run([sys.executable, "-m", "venv", boot]).returncode == 0:
        subprocess.run([os.path.join(boot, "bin", "python"), "-m", "pip", "install", "-q", "uv"], check = True)
        uv = os.path.join(boot, "bin", "uv")
    else:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--user", "uv"], check = True)
        uv = shutil.which("uv") or os.path.expanduser("~/.local/bin/uv")
print("uv:", uv, flush = True)
bench = os.path.join(a.work, "bench.py")
open(bench, "w", encoding = "utf-8").write(BENCH)
ov = os.path.join(a.work, "ov.txt")
open(ov, "w", encoding = "utf-8").write("".join(f"{p} ; sys_platform == 'never'\n" for p in ("torch", "torchvision", "torchaudio", "triton")))
env = dict(os.environ, HF_HOME = os.path.join(a.work, "hf"), UNSLOTH_DISABLE_AUTO_UPDATES = "1")
res = {"order": ORDER, "runs": [], "install": {}}
pys = {}
for arm, specs in ARMS.items():
    v = os.path.join(a.work, "v_" + arm)
    log = os.path.join(a.out, f"install_{arm}.log")
    with open(log, "w", encoding = "utf-8") as lf:
        rc = subprocess.run([uv, "venv", "--no-config", "--python", "3.12", "--seed", v], stdout = lf, stderr = subprocess.STDOUT).returncode
        py = os.path.join(v, "bin", "python")
        rc = rc or subprocess.run([uv, "pip", "install", "--no-config", "--python", py, *specs, "--default-index", IDX], stdout = lf, stderr = subprocess.STDOUT).returncode
        rc = rc or subprocess.run([uv, "pip", "install", "--no-config", "--python", py, f"unsloth @ {UNSLOTH}", f"unsloth_zoo @ {ZOO}", "--override", ov, "--default-index", "https://pypi.org/simple"], stdout = lf, stderr = subprocess.STDOUT).returncode
    res["install"][arm] = rc
    pys[arm] = py
def gpu_procs():
    # Other tenants on this box's GPU void a timing run (a private concurrency group cannot evict
    # us, but cannot keep a second job off the same machine either).
    for cmd in (["amd-smi", "process", "--json"], ["rocm-smi", "--showpids", "--json"]):
        try:
            r = subprocess.run(cmd, capture_output = True, text = True, timeout = 60)
            if r.returncode == 0:
                return {"cmd": cmd[0], "out": r.stdout[-4000:]}
        except (OSError, subprocess.TimeoutExpired):
            continue
    return {"cmd": None, "out": ""}


for i, arm in enumerate(ORDER):
    o = os.path.join(a.out, f"run{i}_{arm}.json")
    procs_before = gpu_procs()
    log = os.path.join(a.out, f"run{i}_{arm}.log")
    t = time.time()
    with open(log, "w", encoding = "utf-8") as lf:
        rc = subprocess.run([pys[arm], "-u", bench, o, os.path.join(a.work, f"sft{i}")], stdout = lf, stderr = subprocess.STDOUT, env = env, timeout = 3600).returncode
    r = {"i": i, "arm": arm, "rc": rc, "s": round(time.time() - t, 1),
         "gpu_procs_before": procs_before, "gpu_procs_after": gpu_procs()}
    if os.path.exists(o):
        r.update(json.load(open(o, encoding = "utf-8")))
    else:
        r["tail"] = open(log, encoding = "utf-8", errors = "replace").read()[-2000:]
    res["runs"].append(r)
    json.dump(res, open(os.path.join(a.out, "perf.json"), "w", encoding = "utf-8"), indent = 1)
    print(json.dumps({k: r.get(k) for k in ("i", "arm", "rc", "ttft_ms", "prefill_tok_s", "decode_tok_s", "train_step_ms_median", "infer_peak_gb", "train_peak_gb")}), flush = True)
for arm in ARMS:
    rs = [r for r in res["runs"] if r["arm"] == arm and r.get("rc") == 0]
    for k in ("ttft_ms", "prefill_tok_s", "decode_tok_s", "train_step_ms_median", "infer_peak_gb", "train_peak_gb"):
        vals = [r[k] for r in rs if k in r]
        if vals:
            print(arm, k, "median", statistics.median(vals), "min", min(vals), "max", max(vals), "n", len(vals), flush = True)
