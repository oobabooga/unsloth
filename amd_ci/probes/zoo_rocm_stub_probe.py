#!/usr/bin/env python3
"""Probe for unsloth-zoo PR 1363: what the Windows ROCm torchao stub does to a real session.

Observes only. Each arm runs in a fresh interpreter with the state's zoo checkout first on
PYTHONPATH, so an earlier import in this process cannot hide a failure:

  zoo_import    import unsloth_zoo, record whether zoo installed its torchao stub and what
                torchao.__version__ / __file__ read as, then import transformers.modeling_utils.
  unsloth_run   import unsloth, load Llama-3.2-1B-Instruct in bf16, greedy-decode 16 tokens,
                then 5 LoRA steps, recording losses.

--torchao-mode is a label only ("present" / "absent"): the workflow installs or removes torchao.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ZOO_IMPORT = r'''
import json, sys, traceback
res = {}
try:
    import torch
    import unsloth_zoo
    res["unsloth_zoo_file"] = unsloth_zoo.__file__
    try:
        import torchao
        res["torchao_type"] = type(torchao).__name__
        res["torchao_is_zoo_stub"] = any(type(f).__name__ == "_ROCmTorchaoFinder" for f in sys.meta_path)
        res["torchao_version"] = repr(getattr(torchao, "__version__", "<miss>"))[:80]
        res["torchao_file"] = repr(getattr(torchao, "__file__", "<miss>"))[:80]
    except BaseException as e:
        res["torchao_import_error"] = f"{type(e).__name__}: {str(e)[:200]}"
    import transformers
    res["transformers"] = transformers.__version__
    import transformers.modeling_utils
    res["modeling_utils"] = "ok"
    from transformers.utils.import_utils import is_torchao_available
    res["is_torchao_available"] = repr(is_torchao_available())
    res["ok"] = True
except BaseException as e:
    res["ok"] = False
    res["error"] = f"{type(e).__name__}: {str(e)[:300]}"
    res["tb"] = traceback.format_exc()[-1500:]
print("PROBE_JSON " + json.dumps(res))
'''

UNSLOTH_RUN = r'''
import json, os, sys, time, traceback
os.environ.setdefault("UNSLOTH_ENABLE_LOGGING", "0")
res = {}
try:
    t0 = time.time()
    from unsloth import FastLanguageModel
    import torch
    res["stub_finder_present"] = any(type(f).__name__ == "_ROCmTorchaoFinder" for f in sys.meta_path)
    model, tok = FastLanguageModel.from_pretrained(
        "unsloth/Llama-3.2-1B-Instruct", max_seq_length = 512, load_in_4bit = False, dtype = torch.bfloat16)
    res["load_seconds"] = round(time.time() - t0, 1)
    FastLanguageModel.for_inference(model)
    ids = tok("The capital of France is", return_tensors = "pt").to(model.device)
    t1 = time.time()
    out = model.generate(**ids, max_new_tokens = 16, do_sample = False)
    res["generate_seconds"] = round(time.time() - t1, 2)
    res["text"] = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens = True)
    FastLanguageModel.for_training(model)
    model = FastLanguageModel.get_peft_model(model, r = 8, target_modules = ["q_proj", "v_proj"],
                                             lora_alpha = 16, lora_dropout = 0, bias = "none", random_state = 3407)
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer
    ds = Dataset.from_dict({"text": [f"Q{i}: {i}+{i}? A: {2*i}." + tok.eos_token for i in range(32)]})
    tr = SFTTrainer(model = model, processing_class = tok, train_dataset = ds, args = SFTConfig(
        max_length = 128, dataset_text_field = "text", per_device_train_batch_size = 1, max_steps = 5,
        learning_rate = 2e-4, logging_steps = 1, optim = "adamw_torch", seed = 3407, save_strategy = "no",
        report_to = "none", output_dir = os.path.join(os.environ.get("AMD_CI_WORK", "."), "sft_out")))
    tr.train()
    res["loss"] = [round(l["loss"], 4) for l in tr.state.log_history if "loss" in l]
    res["peak_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
    res["ok"] = True
except BaseException as e:
    res["ok"] = False
    res["error"] = f"{type(e).__name__}: {str(e)[:300]}"
    res["tb"] = traceback.format_exc()[-1500:]
print("PROBE_JSON " + json.dumps(res))
'''


def _arm(python: str, code: str, checkout: str, timeout: int, extra_env: dict | None = None) -> dict:
    env = dict(os.environ)
    env.update(extra_env or {})
    env["PYTHONPATH"] = checkout + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONIOENCODING"] = "utf-8"
    try:
        p = subprocess.run([python, "-c", code], capture_output = True, text = True, encoding = "utf-8",
                           errors = "replace", env = env, timeout = timeout)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"timeout after {timeout}s"}
    for line in reversed(p.stdout.splitlines()):
        if line.startswith("PROBE_JSON "):
            res = json.loads(line[len("PROBE_JSON "):])
            res["returncode"] = p.returncode
            return res
    return {"ok": False, "returncode": p.returncode,
            "error": f"no result line; exit {p.returncode}", "stderr_tail": p.stderr[-1500:]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--torchao-mode", default = "present")
    args = ap.parse_args()

    obs: dict = {"state": args.state, "torchao_mode": args.torchao_mode}
    try:
        import torch
        obs["torch"] = torch.__version__
        obs["hip"] = getattr(torch.version, "hip", None)
        obs["device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        obs["arch"] = getattr(torch.cuda.get_device_properties(0), "gcnArchName", None) if torch.cuda.is_available() else None
    except Exception as e:  # noqa: BLE001
        obs["torch_error"] = f"{type(e).__name__}: {e}"
    obs["arms"] = {
        # unsloth sets UNSLOTH_IS_PRESENT before it imports the zoo; importing the zoo alone needs it.
        "zoo_import": _arm(sys.executable, ZOO_IMPORT, args.checkout, 900, {"UNSLOTH_IS_PRESENT": "1"}),
        "unsloth_run": _arm(sys.executable, UNSLOTH_RUN, args.checkout, 3600),
    }
    args.out.parent.mkdir(parents = True, exist_ok = True)
    args.out.write_text(json.dumps(obs, indent = 2, default = str), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
