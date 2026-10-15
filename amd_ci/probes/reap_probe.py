#!/usr/bin/env python3
"""Probe: does this checkout's REAP expert pruning run on the GPU, and keep the CPU's experts?

Observes only. Each mode runs in its own process (a kernel crash ends one mode, not the probe),
imports unsloth first like a user would, builds a tiny random Qwen3-MoE and OLMoE, and calls
`unsloth/models/reap.py` from THIS checkout on fixed synthetic token rows. Records kept experts,
expert shapes after pruning and whether a forward afterwards is finite.

Pairs with criteria/reap_on_gpu.py.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

CHILD = r'''
import importlib.util, json, os, sys
checkout, mode, out = sys.argv[1], sys.argv[2], sys.argv[3]
res = {"mode": mode}
try:
    import unsloth  # noqa: F401  (user path: Unsloth's MoE patches active)
    from unsloth_zoo.temporary_patches.moe_utils import select_moe_backend
    res["moe_backend"] = select_moe_backend()
except Exception as e:
    res["unsloth_import_error"] = f"{type(e).__name__}: {e}"[:300]
import torch, transformers
res["torch"] = torch.__version__; res["transformers"] = transformers.__version__
res["hip"] = getattr(torch.version, "hip", None)
if torch.cuda.is_available():
    res["arch"] = getattr(torch.cuda.get_device_properties(0), "gcnArchName", None)
path = os.path.join(checkout, "unsloth", "models", "reap.py")
if not os.path.isfile(path):
    res["error"] = "no unsloth/models/reap.py in this checkout"
    json.dump(res, open(out, "w")); sys.exit(0)
spec = importlib.util.spec_from_file_location("_reap_probe", path)
reap = importlib.util.module_from_spec(spec); spec.loader.exec_module(reap)
device = "cpu" if mode.startswith("cpu") else "cuda"
dtype = torch.bfloat16 if mode.endswith("bf16") else torch.float32
g = torch.Generator().manual_seed(0)
rows = [torch.randint(0, 96, (int(n),), generator = g).tolist() for n in torch.randint(8, 48, (24,), generator = g)]
from transformers import AutoModelForCausalLM, Qwen3MoeConfig, OlmoeConfig
cfgs = {
    "qwen3_moe": Qwen3MoeConfig(vocab_size = 96, hidden_size = 64, intermediate_size = 64,
        moe_intermediate_size = 32, num_hidden_layers = 3, num_attention_heads = 4,
        num_key_value_heads = 2, head_dim = 16, num_experts = 16, num_experts_per_tok = 4,
        norm_topk_prob = True, tie_word_embeddings = False),
    "olmoe": OlmoeConfig(vocab_size = 96, hidden_size = 64, intermediate_size = 32,
        num_hidden_layers = 3, num_attention_heads = 4, num_key_value_heads = 4,
        num_experts = 16, num_experts_per_tok = 4),
}
res["models"] = {}
import copy
for name, cfg in cfgs.items():
    cfg = copy.deepcopy(cfg)
    r = {}
    try:
        torch.manual_seed(1234)
        model = AutoModelForCausalLM.from_config(cfg, dtype = dtype).to(device).eval()
        reap.reap_prune(model, None, sparsity = 0.5, dataset = rows, device = device)
        r["kept"] = model._unsloth_reap_report
        r["saliency"] = model._unsloth_reap_saliency
        r["num_experts_after"] = model.config.num_experts
        r["expert_rows"] = sorted({tuple(p.shape)[0] for n, p in model.named_parameters() if ".experts." in n})
        with torch.no_grad():
            logits = model(torch.tensor([rows[0]], device = device)).logits
        r["logits_finite"] = bool(torch.isfinite(logits).all())
    except Exception as e:
        r["error"] = f"{type(e).__name__}: {e}"[:500]
    res["models"][name] = r
json.dump(res, open(out, "w"))
'''

MODES = ("cpu_fp32", "gpu_fp32", "cpu_bf16", "gpu_bf16")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True, type = Path)
    ap.add_argument("--out", required = True, type = Path)
    args = ap.parse_args()

    obs: dict = {"state": args.state, "modes": {}}
    for mode in MODES:
        out = args.out.with_name(f"{args.out.stem}_{mode}.json")
        try:
            p = subprocess.run([sys.executable, "-c", CHILD, str(args.checkout), mode, str(out)],
                               capture_output = True, text = True, timeout = 1200)
            rc, tail = p.returncode, (p.stdout + p.stderr)[-1500:]
        except subprocess.TimeoutExpired:
            rc, tail = "timeout", ""
        entry = {"rc": rc}
        if out.is_file():
            entry.update(json.loads(out.read_text(encoding = "utf-8")))
        else:
            entry["error"] = f"child wrote nothing (rc {rc}): {tail}"
        obs["modes"][mode] = entry
    args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
