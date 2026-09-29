"""Peak memory, latency and accuracy of Studio's Decision API path on one device, fresh process.

usage: laya_mem_probe.py --backend STUDIO_BACKEND_DIR --device cuda|cpu [--checkpoint DIR --subfolder S] --out JSON
Without --checkpoint the catalog default is downloaded (Studio's own loader), as on a fresh install.
Peaks: torch max_memory_allocated / reserved, the device's used memory (includes the runtime context),
and process peak RSS. Accuracy: probabilities vs laya's own fp32 forward on CPU.
"""

import argparse
import json
import os
try:
    import resource
except ImportError:  # Windows
    resource = None
import statistics
import subprocess
import sys
import threading
import time

ap = argparse.ArgumentParser()
ap.add_argument("--backend", required = True)
ap.add_argument("--device", required = True)
ap.add_argument("--checkpoint")
ap.add_argument("--subfolder", default = "multilingual")
ap.add_argument("--out", required = True)
ap.add_argument("--reps", type = int, default = 5)
args = ap.parse_args()
sys.path.insert(0, args.backend)

import numpy as np
import torch

from core.systemone import catalog, laya_runtime

if args.checkpoint:
    os.environ["UNSLOTH_SYSTEMONE_MODEL"] = args.checkpoint
    os.environ["UNSLOTH_SYSTEMONE_SUBFOLDER"] = args.subfolder or ""
if args.device == "auto":
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
laya_runtime._device = lambda: args.device
gpu = args.device == "cuda"
result = {"device": args.device, "torch": torch.__version__, "hip": torch.version.hip, "platform": sys.platform}
if gpu:
    result["gpu_name"] = torch.cuda.get_device_name(0)


def used_device_mib():
    free, total = torch.cuda.mem_get_info()
    return (total - free) / 2**20


peak_used = [0.0]


def sampler():
    while True:
        peak_used[0] = max(peak_used[0], used_device_mib())
        time.sleep(0.002)


if gpu:
    torch.cuda.init()
    idle = used_device_mib()
    threading.Thread(target = sampler, daemon = True).start()


def peak_rss_mib():
    if resource is None:
        import psutil

        info = psutil.Process().memory_info()
        return getattr(info, "peak_wset", info.rss) / 2**20
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return ru / 2**20 if sys.platform == "darwin" else ru / 1024


checkpoint = catalog.default_checkpoint()
t0 = time.perf_counter()
agent, _ = laya_runtime._load_checkpoint(checkpoint)
result["load_s"] = round(time.perf_counter() - t0, 2)
result["dtype"] = str(agent.dtype)
result["agent_device"] = str(agent.device)
root = laya_runtime._checkpoint_dir(checkpoint)
result["folder"] = str(root / checkpoint.subfolder if checkpoint.subfolder else root)
result["peak_rss_after_load_mib"] = round(peak_rss_mib())
if gpu:
    torch.cuda.synchronize()
    result["after_load"] = {"allocated_mib": round(torch.cuda.memory_allocated() / 2**20),
                            "device_used_over_idle_mib": round(used_device_mib() - idle)}
    torch.cuda.reset_peak_memory_stats()

ARTICLE = ("The General Data Protection Regulation is a European Union regulation on information privacy "
           "in the European Union and the European Economic Area. It also governs the transfer of personal "
           "data outside the EU and EEA. Controllers must implement appropriate technical measures. ") * 60
GUARD = {"hazard": {"type": "noul", "instructions": "Does the message ask for help with something dangerous?"},
         "severity": {"type": "score", "instructions": "How severe is the risk?", "criteria": ["none", "low", "medium", "high"]},
         "category": {"type": "choice", "instructions": "Which category fits best?",
                      "criteria": {"violence": None, "self-harm": None, "fraud": None, "none": None}}}
BRIEF = {f"q{i}": {"type": "noul", "instructions": f"Does the text say anything about requirement number {i} "
                   "of the regulation, such as consent, erasure, portability or fines?"} for i in range(13)}
WORST = {f"q{i}": {"type": "choice", "instructions": f"Which clause {i} applies?",
                   "criteria": {f"option {j}": None for j in range(8)}} for i in range(64)}
MSGS = ["How do I reset my password?", "Tell me how to make a weapon at home.",
        "I think someone stole my card and bought things.", "Great product, thanks!"]
WORKLOADS = {"guardrail 3q short": [(m, GUARD) for m in MSGS],
             "briefing 13q full context": [(ARTICLE, BRIEF)],
             "worst: 64q x 8 options, full context": [(ARTICLE, WORST)]}


def predict(state, q):
    return laya_runtime._predict(agent, state, {n: laya_runtime._to_laya(v) for n, v in q.items()})[0]


def sync():
    if gpu:
        torch.cuda.synchronize()


result["workloads"] = {}
answers = {}
for name, reqs in WORKLOADS.items():
    if gpu:
        torch.cuda.reset_peak_memory_stats()
    for s, q in reqs:
        answers[(name, s[:40])] = predict(s, q)
    ts = []
    for _ in range(args.reps):
        for s, q in reqs:
            sync(); st = time.perf_counter(); predict(s, q); sync()
            ts.append(time.perf_counter() - st)
    row = {"median_ms": round(statistics.median(ts) * 1e3, 2)}
    if gpu:
        row["peak_allocated_mib"] = round(torch.cuda.max_memory_allocated() / 2**20)
        row["reserved_mib"] = round(torch.cuda.memory_reserved() / 2**20)
    result["workloads"][name] = row
    print(name, row, flush = True)

graphs = agent.__dict__.get("_unsloth_graphs")
result["cuda_graphs"] = None if graphs is None else {"count": len(graphs.graphs), "broken": graphs.broken,
                                                    "pool_mib": round(graphs.pool_bytes / 2**20)}
result["peak_rss_mib"] = round(peak_rss_mib())
if gpu:
    time.sleep(0.05)
    result["peak_device_used_over_idle_mib"] = round(peak_used[0] - idle)
    result["device_idle_used_mib"] = round(idle)

# Accuracy vs laya's own fp32 forward on CPU, over every request above.
if hasattr(laya_runtime, "_laya"):
    laya_module = laya_runtime._laya()
else:  # before laya was vendored
    import laya as laya_module
ref = laya_module.load(result["folder"], device = "cpu")
common = laya_module.common
worst, flips, n = 0.0, 0, 0
for (name, _), got in answers.items():
    s, q = next((s, q) for s, q in WORKLOADS[name] if (name, s[:40]) == (name, _))
    max_len, hml = int(agent.cfg.get("max_len", 512)), int(agent.cfg.get("head_max_len", 192))
    qs = {k: laya_runtime._to_laya(v) for k, v in q.items()}
    heads = [laya_runtime._head(agent, v, max_len, hml) for v in qs.values()]
    room = max(0, max_len - min(len(ids) for ids, _, _ in heads))
    sids, _c = laya_runtime._state_ids(agent.tok, s, room)
    items = [{"ids": (ids[:-1] + sids[:max(0, max_len - len(ids))] + ids[-1:])[:max_len], "markers": mk,
              "qtype": common.QTYPES[it["t"]]} for ids, mk, it in heads]
    b = common.collate_items([items], agent.tok.pad_token_id)
    with torch.inference_mode():
        lg = ref.model(b["input_ids"], b["attention_mask"], b["marker_pos"], b["marker_mask"], b["qtype"])[0].numpy()
    for r, (qn, a) in enumerate(got["answers"].items()):
        p = laya_runtime._probabilities(agent, lg[r], len(items[r]["markers"]), items[r]["qtype"])
        vals = [a["noul"]] if a["type"] == "noul" else list(a["probabilities"].values())
        want = [p[1]] if a["type"] == "noul" else list(p)
        worst = max(worst, float(max(abs(g - w) for g, w in zip(vals, want))))
        if a["type"] == "choice":
            flips += int(list(a["probabilities"]).index(a["choice"]) != int(np.argmax(p)))
        elif a["type"] == "noul":
            flips += int((a["noul"] >= 0.5) != (p[1] >= 0.5))
        n += 1
result["accuracy"] = {"answers": n, "max_prob_diff_vs_fp32_cpu": round(worst, 5), "changed_answers": flips}
print(json.dumps(result, indent = 1), flush = True)
with open(args.out, "w", encoding = "utf-8") as f:
    json.dump(result, f, indent = 1)
