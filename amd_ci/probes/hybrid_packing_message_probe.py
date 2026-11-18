#!/usr/bin/env python3
"""Probe (unslothai/unsloth#17751): SFTTrainer packing=True on a tiny random Qwen3.5 text model on ROCm,
once with UNSLOTH_EXPERIMENTAL_HYBRID_PACKING=1 and once unset. Records which packing message printed."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

_CHILD = r'''
import json, os, sys
sys.path.insert(0, sys.argv[1])
res = {}
try:
    import unsloth
    import torch
    res["unsloth_file"] = unsloth.__file__
    res["torch"] = torch.__version__
    res["hip"] = getattr(torch.version, "hip", None)
    res["gpu"] = torch.cuda.is_available()
    import transformers
    res["transformers"] = transformers.__version__
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer
    cfg = AutoConfig.for_model(
        "qwen3_5_text", vocab_size = 1024, hidden_size = 128, intermediate_size = 256,
        num_hidden_layers = 2, layer_types = ["linear_attention", "full_attention"],
        num_attention_heads = 4, num_key_value_heads = 2, head_dim = 32,
        linear_num_key_heads = 2, linear_num_value_heads = 4,
        linear_key_head_dim = 32, linear_value_head_dim = 32,
    )
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(cfg).to("cuda", torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(sys.argv[2])
    texts = [" ".join(["word%d" % ((i * 7 + j) % 50) for j in range(20 + 5 * (i % 6))]) for i in range(64)]
    ds = Dataset.from_dict({"text": texts})
    args = SFTConfig(output_dir = sys.argv[3], packing = True, max_length = 256, max_steps = 2,
                     per_device_train_batch_size = 2, learning_rate = 1e-4, bf16 = True,
                     report_to = "none", logging_steps = 1, save_strategy = "no")
    trainer = SFTTrainer(model = model, args = args, train_dataset = ds, processing_class = tok)
    from unsloth.utils.packing import HYBRID_PACKING_STATS
    res["stats"] = dict(HYBRID_PACKING_STATS)
    res["opt_in_attr"] = getattr(model, "_unsloth_hybrid_opt_in_device", "<absent>")
    print("PROBE_RESULT " + json.dumps(res, default = str), flush = True)
    out = trainer.train()
    res["train_loss"] = float(out.training_loss)
except Exception as e:
    import traceback
    res["error"] = f"{type(e).__name__}: {e}"
    res["tb"] = traceback.format_exc()[-3000:]
print("PROBE_RESULT " + json.dumps(res, default = str), flush = True)
'''


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--python", default = sys.executable)
    ap.add_argument("--tokenizer", default = "Qwen/Qwen3-0.6B")
    a = ap.parse_args()
    tmp = Path(os.environ.get("RUNNER_TEMP", "/tmp")) / f"pr17751_{a.state}"
    tmp.mkdir(parents = True, exist_ok = True)
    child = tmp / "child.py"
    child.write_text(_CHILD, encoding = "utf-8")
    result = {"state": a.state, "runs": {}}
    # Same fla on both states: the varlen gate needs it, and the issue's Studio install had it.
    has_fla = subprocess.run([a.python, "-c", "import fla"], capture_output = True).returncode == 0
    if not has_fla:
        p = subprocess.run([a.python, "-m", "pip", "install", "-q", "--no-deps", "flash-linear-attention", "fla-core"],
                           capture_output = True, text = True)
        result["fla_install"] = {"rc": p.returncode, "tail": (p.stdout + p.stderr)[-1500:]}
    result["fla_preinstalled"] = has_fla
    for mode in ("opt_in", "unset"):
        env = dict(os.environ, PYTHONFAULTHANDLER = "1", UNSLOTH_DISABLE_AUTO_UPDATES = "1", UNSLOTH_COMPILE_LOCATION = str(tmp / f"cc_{mode}"))
        env.pop("UNSLOTH_EXPERIMENTAL_HYBRID_PACKING", None)
        if mode == "opt_in":
            env["UNSLOTH_EXPERIMENTAL_HYBRID_PACKING"] = "1"
        p = subprocess.run([a.python, str(child), a.checkout, a.tokenizer, str(tmp / f"o_{mode}")],
                           env = env, capture_output = True, text = True, timeout = 1800)
        text = p.stdout + p.stderr
        (a.out.parent / f"child_{a.state}_{mode}.log").write_text(text, encoding = "utf-8")
        run = {"rc": p.returncode,
               "old_claim": "training is >2x faster" in text,
               "opt_in_note": "through UNSLOTH_EXPERIMENTAL_HYBRID_PACKING=1" in text,
               "packing_ignored": "packing=True ignored" in text,
               "log_tail": text[-4000:]}
        for line in text.splitlines():
            if line.startswith("PROBE_RESULT "):
                run.update(json.loads(line[len("PROBE_RESULT "):]))
        result["runs"][mode] = run
    a.out.write_text(json.dumps(result, indent = 1), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
