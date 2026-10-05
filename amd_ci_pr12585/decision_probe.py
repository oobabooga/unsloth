#!/usr/bin/env python3
"""Probe for PR 12585 on gfx1151: observes, never judges.

Each scenario runs in its own subprocess (Unsloth patches are process-wide) with the
state's checkout first on sys.path, and writes one JSON result. Scenarios:
  control      tiny Qwen3 LoRA SFT through FastLanguageModel (exists at base and head)
  clef_bf16    tiny Clef (tiny Qwen3.5 backbone + random head) LoRA, bf16
  clef_fp16    same, float16 request
  clef_4bit    same, load_in_4bit (bitsandbytes on ROCm)
  laya         convaiinnovations/laya multilingual LoRA, 20 steps on LocalLLaMA/typed-decisions
The decision scenarios record `absent` when the state has no FastDecisionModel.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

COMMON = r'''
import json, os, sys, time, traceback
CHECKOUT = os.environ["PROBE_CHECKOUT"]
sys.path[:0] = [CHECKOUT, os.path.join(CHECKOUT, "tests"), os.path.join(CHECKOUT, "tests", "_shared")]
RES = {}
def done():
    with open(os.environ["PROBE_RESULT"], "w", encoding = "utf-8") as fh:
        json.dump(RES, fh, indent = 1, default = str)
def levers():
    out = {}
    for mod in ("causal_conv1d", "fla", "mamba_ssm", "bitsandbytes", "triton"):
        try:
            m = __import__(mod)
            out[mod] = getattr(m, "__version__", "present")
        except Exception as e:
            out[mod] = f"absent: {type(e).__name__}: {str(e)[:200]}"
    import torch
    out["torch"] = torch.__version__
    out["hip"] = torch.version.hip
    out["device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    try:
        from torch.utils._triton import has_triton
        out["has_triton"] = has_triton()
    except Exception as e:
        out["has_triton"] = f"error {e}"
    return out
'''

CONTROL = COMMON + r'''
try:
    import unsloth
    from unsloth import FastLanguageModel
    import torch
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer
    RES["unsloth_file"] = unsloth.__file__
    model, tok = FastLanguageModel.from_pretrained("trl-internal-testing/tiny-Qwen3ForCausalLM",
                                                   max_seq_length = 128, load_in_4bit = False)
    model = FastLanguageModel.get_peft_model(model, r = 8, lora_alpha = 8, random_state = 3407)
    ds = Dataset.from_dict({"text": [f"The answer to question {i} is {i * 7 % 13}." for i in range(64)]})
    losses = []
    from transformers import TrainerCallback
    class CB(TrainerCallback):
        def on_log(self, a, s, c, logs = None, **k):
            if logs and "loss" in logs:
                losses.append([float(logs["loss"]), float(logs.get("grad_norm") or float("nan"))])
    tr = SFTTrainer(model = model, processing_class = tok, train_dataset = ds, callbacks = [CB()],
                    args = SFTConfig(output_dir = os.environ["PROBE_TMP"] + "/sft", per_device_train_batch_size = 2,
                                     gradient_accumulation_steps = 1, max_steps = 5, learning_rate = 2e-4,
                                     logging_steps = 1, report_to = "none", save_strategy = "no", seed = 3407,
                                     dataset_text_field = "text", max_length = 128))
    tr.train()
    RES["losses"] = losses
    RES["ok"] = bool(losses) and all(l == l and abs(l) < 1e4 for l, _ in losses)
except BaseException as e:
    RES["ok"] = False
    RES["error"] = f"{type(e).__name__}: {e}"[:2000]
    RES["traceback"] = traceback.format_exc()[-4000:]
RES["levers"] = levers()
done()
'''

CLEF = COMMON + r'''
MODE = os.environ["PROBE_MODE"]
try:
    import unsloth
    try:
        from unsloth import DecisionTrainer, FastDecisionModel
    except ImportError as e:
        RES["absent"] = f"no FastDecisionModel at this state ({e})"
        RES["levers"] = levers()
        done()
        sys.exit(0)
    import math, shutil, torch
    from pathlib import Path
    from safetensors.torch import save_file
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration, TrainerCallback
    import test_decision_model as T
    from unsloth.models import clef as clef_mod, decision
    from unsloth.models.clef import JointSchemaHead
    RES["unsloth_file"] = unsloth.__file__
    folder = Path(os.environ["PROBE_TMP"]) / "clef"
    _, ref_path = T._clef_reference()
    torch.manual_seed(0)
    Qwen3_5ForConditionalGeneration.from_pretrained(T.TINY_QWEN3_5, dtype = torch.bfloat16).save_pretrained(str(folder))
    AutoProcessor.from_pretrained(T.TINY_QWEN3_5).save_pretrained(str(folder))
    head = JointSchemaHead(**T.CLEF_HEAD)
    with torch.no_grad():
        for p in head.parameters():
            p.add_(torch.randn_like(p) * 0.1)
    save_file({k: v.to(torch.bfloat16).contiguous() for k, v in head.state_dict().items()},
              str(folder / "joint_head.safetensors"))
    (folder / "joint_head_config.json").write_text(json.dumps(T.CLEF_HEAD))
    shutil.copyfile(ref_path, folder / "joint_schema_model.py")
    dtype = torch.float16 if MODE == "fp16" else torch.bfloat16
    t0 = time.time()
    model, processor = FastDecisionModel.from_pretrained(str(folder), max_seq_length = 512, dtype = dtype,
                                                         load_in_4bit = MODE == "4bit")
    RES["load_s"] = round(time.time() - t0, 2)
    RES["forced_float32"] = decision._clef_forced_float32(model)
    RES["fast_backbone"] = getattr(model, "_unsloth_fast_backbone", None)
    dev = next(model.parameters()).device
    RES["clef_compile_supported"] = clef_mod._compile_supported(dev)
    if MODE == "4bit":
        n4 = sum(1 for m in model.modules() if type(m).__name__ == "Linear4bit")
        RES["linear4bit_modules"] = n4
    # Which Qwen3.5 linear-attention kernels the backbone bound
    kinds = {}
    try:
        from transformers.utils.import_utils import is_causal_conv1d_available, is_flash_linear_attention_available
        kinds["fla_available"] = is_flash_linear_attention_available()
        kinds["causal_conv1d_available"] = is_causal_conv1d_available()
    except Exception as e:
        kinds["availability_error"] = str(e)[:200]
    for m in model.modules():
        if type(m).__name__.endswith("GatedDeltaNet"):
            fwd = type(m).forward
            kinds["forward_file"] = os.path.basename(fwd.__code__.co_filename)
            for name in ("torch_chunk_gated_delta_rule", "causal_conv1d_fn"):
                fn = fwd.__globals__.get(name)
                kinds[name] = "global missing" if fn is None else getattr(fn, "__module__", "?")
                while fn is not None:
                    cells = dict(zip(fn.__code__.co_freevars, fn.__closure__ or ()))
                    if "implementation" in cells:
                        kinds[name] = getattr(cells["implementation"].cell_contents, "__module__", "?")
                        break
                    fn = getattr(fn, "__wrapped__", None)
            break
    RES["gated_delta_kernels"] = kinds
    model = FastDecisionModel.get_peft_model(model, r = 8, lora_alpha = 8)
    items, _ = FastDecisionModel.build_dataset(T._clef_rows(32), processor, model)
    class CB(TrainerCallback):
        def on_log(self, a, s, c, logs = None, **k):
            if logs and "loss" in logs:
                RES.setdefault("logs", []).append([float(logs["loss"]), float(logs.get("grad_norm") or float("nan"))])
    trainer = DecisionTrainer(model = model, tokenizer = processor, train_dataset = items,
                              args = T._args(Path(os.environ["PROBE_TMP"]), max_steps = 6, logging_steps = 1,
                                             fp16 = MODE == "fp16", bf16 = MODE != "fp16"),
                              callbacks = [CB()])
    RES["trainer_fp16"], RES["trainer_bf16"] = trainer.args.fp16, trainer.args.bf16
    t0 = time.time()
    trainer.train()
    RES["train_s"] = round(time.time() - t0, 2)
    RES["eval_loss"] = float(FastDecisionModel.evaluate(model, processor, items)["loss"])
    ls = RES.get("logs", [])
    RES["ok"] = bool(ls) and all(math.isfinite(l) and math.isfinite(g) and g > 0 for l, g in ls) and math.isfinite(RES["eval_loss"])
    RES["clef_compile_after"] = os.environ.get("UNSLOTH_CLEF_COMPILE", "1")
    from torch._dynamo.utils import counters
    RES["dynamo"] = {"unique_graphs": counters["stats"].get("unique_graphs"),
                     "graph_breaks": sum(counters["graph_break"].values())}
except BaseException as e:
    RES["ok"] = False
    RES["error"] = f"{type(e).__name__}: {e}"[:3000]
    import traceback
    RES["traceback"] = traceback.format_exc()[-6000:]
RES["levers"] = levers()
done()
'''

LAYA = COMMON + r'''
try:
    import unsloth
    try:
        from unsloth import DecisionTrainer, FastDecisionModel
    except ImportError as e:
        RES["absent"] = f"no FastDecisionModel at this state ({e})"
        RES["levers"] = levers()
        done()
        sys.exit(0)
    import math, torch
    from datasets import load_dataset
    from transformers import TrainerCallback, TrainingArguments
    RES["unsloth_file"] = unsloth.__file__
    t0 = time.time()
    model, tok = FastDecisionModel.from_pretrained("convaiinnovations/laya", subfolder = "multilingual")
    model = FastDecisionModel.get_peft_model(model, r = 64, lora_alpha = 64)
    RES["load_s"] = round(time.time() - t0, 2)
    ds = load_dataset("LocalLLaMA/typed-decisions",
                      data_files = {"train": "all/train-00000-of-00001.parquet", "test": "all/test-00000-of-00001.parquet"})
    items, report = FastDecisionModel.build_dataset([dict(r) for r in ds["train"].select(range(1200))], tok, model)
    test, _ = FastDecisionModel.build_dataset([dict(r) for r in ds["test"].select(range(300))], tok, model)
    RES["n_train_items"], RES["n_test_items"] = len(items), len(test)
    RES["acc_before"] = FastDecisionModel.evaluate(model, tok, test)
    times = []
    class CB(TrainerCallback):
        def on_step_begin(self, *a, **k):
            torch.cuda.synchronize(); self.t = time.perf_counter()
        def on_step_end(self, *a, **k):
            torch.cuda.synchronize(); times.append(time.perf_counter() - self.t)
        def on_log(self, a, s, c, logs = None, **k):
            if logs and "loss" in logs:
                RES.setdefault("logs", []).append([float(logs["loss"]), float(logs.get("grad_norm") or float("nan"))])
    torch.cuda.reset_peak_memory_stats()
    trainer = DecisionTrainer(model = model, processing_class = tok, train_dataset = items, callbacks = [CB()],
                              args = TrainingArguments(output_dir = os.environ["PROBE_TMP"] + "/laya",
                                                       per_device_train_batch_size = 8, gradient_accumulation_steps = 4,
                                                       learning_rate = 8e-4, max_steps = 20, bf16 = True,
                                                       logging_steps = 1, report_to = "none", save_strategy = "no",
                                                       seed = 3407, lr_scheduler_type = "cosine", disable_tqdm = True))
    trainer.train()
    RES["times"] = [round(t, 3) for t in times]
    st = sorted(times[2:]) or sorted(times)
    RES["s_per_step_median"] = st[len(st) // 2] if st else None
    RES["peak_gb"] = torch.cuda.max_memory_allocated() / 2**30
    RES["acc_after"] = FastDecisionModel.evaluate(model, tok, test)
    enc = model.encoder
    RES["encoder_reference_compile"] = getattr(getattr(enc, "config", None), "reference_compile", None)
    from torch._dynamo.utils import counters
    RES["dynamo"] = {"unique_graphs": counters["stats"].get("unique_graphs"),
                     "graph_breaks": sum(counters["graph_break"].values())}
    ls = RES.get("logs", [])
    RES["ok"] = len(ls) == 20 and all(math.isfinite(l) for l, _ in ls)
except BaseException as e:
    RES["ok"] = False
    RES["error"] = f"{type(e).__name__}: {e}"[:3000]
    import traceback
    RES["traceback"] = traceback.format_exc()[-6000:]
RES["levers"] = levers()
done()
'''

SCENARIOS = {
    "control": (CONTROL, {}),
    "clef_bf16": (CLEF, {"PROBE_MODE": "bf16"}),
    "clef_fp16": (CLEF, {"PROBE_MODE": "fp16"}),
    "clef_4bit": (CLEF, {"PROBE_MODE": "4bit"}),
    "laya": (LAYA, {}),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--python", default = sys.executable)
    ap.add_argument("--timeout", type = int, default = 2400)
    ap.add_argument("--scenarios", default = ",".join(SCENARIOS))
    args = ap.parse_args()

    checkout = str(Path(args.checkout).resolve())
    obs: dict = {"state": args.state, "checkout": checkout, "scenarios": {}}
    for name in args.scenarios.split(","):
        code, env_extra = SCENARIOS[name]
        with tempfile.TemporaryDirectory(dir = os.environ.get("RUNNER_TEMP") or None) as tmp:
            script = Path(tmp) / f"{name}.py"
            script.write_text(code, encoding = "utf-8")
            result = Path(tmp) / "result.json"
            env = dict(os.environ, PROBE_CHECKOUT = checkout, PROBE_RESULT = str(result), PROBE_TMP = tmp,
                       PYTHONPATH = checkout + os.pathsep + os.environ.get("PYTHONPATH", ""),
                       UNSLOTH_COMPILE_LOCATION = str(Path(tmp) / "compiled_cache"),
                       UNSLOTH_DISABLE_AUTO_UPDATES = "1", **env_extra)
            t0 = time.time()
            try:
                p = subprocess.run([args.python, str(script)], cwd = tmp, env = env, capture_output = True,
                                   text = True, timeout = args.timeout)
                rc, tail = p.returncode, ((p.stdout or "") + (p.stderr or ""))[-6000:]
            except subprocess.TimeoutExpired as exc:
                rc, tail = -1, f"timeout after {args.timeout}s"
            rec: dict = {"rc": rc, "wall_s": round(time.time() - t0, 1)}
            if result.is_file():
                rec.update(json.loads(result.read_text(encoding = "utf-8")))
            else:
                rec["no_result"] = True
            rec["log_tail"] = tail
        obs["scenarios"][name] = rec
        args.out.write_text(json.dumps(obs, indent = 2, default = str), encoding = "utf-8")
    args.out.write_text(json.dumps(obs, indent = 2, default = str), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
