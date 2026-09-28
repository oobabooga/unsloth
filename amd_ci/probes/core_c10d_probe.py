# Unsloth Core on a torch without a distributed backend: base vs head, one fresh process per step.
# usage: core_c10d_probe.py --python PY --root STATES --out DIR [--steps a,b] [--tag T]
import argparse, json, os, subprocess, sys, time

PRE = r'''
import os, sys, json, traceback
os.environ.setdefault("UNSLOTH_DISABLE_AUTO_UPDATES", "1")
import torch
MODEL = "trl-internal-testing/tiny-LlamaForCausalLM-3.2"
'''
STEPS = {
"import_tf": r'''
import unsloth
print("UNSLOTH_FILE", unsloth.__file__)
print("DIST_AVAILABLE", torch.distributed.is_available(), "HIP", torch.version.hip, "TORCH", torch.__version__)
from transformers import AutoModelForCausalLM, GemmaForCausalLM, LlamaForCausalLM, Gemma2ForCausalLM
import peft, trl, accelerate, transformers
print("VERSIONS", transformers.__version__, trl.__version__, peft.__version__, accelerate.__version__)
print("TORCHAO", repr(sys.modules.get("torchao", "absent"))[:80])
''',
"generate": r'''
from unsloth import FastLanguageModel
model, tok = FastLanguageModel.from_pretrained(MODEL, max_seq_length = 256, load_in_4bit = False)
FastLanguageModel.for_inference(model)
tok.padding_side = "left"
if tok.pad_token is None: tok.pad_token = tok.eos_token
enc = tok(["Hello there my good friend, how are", "Hi"], return_tensors = "pt", padding = True).to(model.device)
out = model.generate(**enc, max_new_tokens = 8, min_new_tokens = 8, do_sample = False)
assert out.shape[1] == enc["input_ids"].shape[1] + 8, out.shape
one = model.generate(**tok(["Hi"], return_tensors = "pt").to(model.device), max_new_tokens = 8, min_new_tokens = 8, do_sample = False)
print("GENERATED", out.shape, one.shape, model.device)
''',
"train": r'''
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from datasets import Dataset
import accelerate
print("ACCELERATE", accelerate.__version__)
model, tok = FastLanguageModel.from_pretrained(MODEL, max_seq_length = 128, load_in_4bit = False)
model = FastLanguageModel.get_peft_model(model, r = 8, lora_alpha = 8, target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"])
if tok.pad_token is None: tok.pad_token = tok.eos_token
ds = Dataset.from_dict({"text": [f"Item {i} is a small red box. " * 6 for i in range(32)]})
trainer = SFTTrainer(model = model, processing_class = tok, train_dataset = ds,
    args = SFTConfig(output_dir = os.environ["PROBE_OUT"], max_steps = 3, per_device_train_batch_size = 2,
                     gradient_accumulation_steps = 1, learning_rate = 1e-3, logging_steps = 1, report_to = "none",
                     dataset_text_field = "text", seed = 3407, save_strategy = "no"))
trainer.train()
losses = [h["loss"] for h in trainer.state.log_history if "loss" in h]
print("LOSSES", losses)
assert len(losses) == 3 and all(l == l for l in losses), losses
''',
}

def run(py, root, side, step, out, tag):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.path.join(root, side)
    env["UNSLOTH_COMPILE_LOCATION"] = os.path.join(out, f"compiled_{side}_{step}{tag}")
    env["PROBE_OUT"] = os.path.join(out, f"trainer_{side}{tag}")
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    t = time.time()
    try:
        p = subprocess.run([py, "-c", PRE + STEPS[step]], env = env, capture_output = True, text = True,
                           encoding = "utf-8", errors = "replace", timeout = 1500, cwd = out)
        rc, text = p.returncode, p.stdout + "\n" + p.stderr
    except subprocess.TimeoutExpired as e:
        rc, text = "timeout", str(e)
    with open(os.path.join(out, f"{side}_{step}{tag}.log"), "w", encoding = "utf-8") as f:
        f.write(text)
    lines = text.splitlines()
    keep = [l for l in lines if l.split(" ")[0] in ("UNSLOTH_FILE", "DIST_AVAILABLE", "VERSIONS", "TORCHAO", "GENERATED", "LOSSES", "ACCELERATE")]
    errs = [l for l in lines if "Error" in l and not l.startswith(" ")][-3:]
    return dict(side = side, step = step + tag, rc = rc, ok = rc == 0, seconds = round(time.time() - t, 1), facts = keep, errors = errs)

def main():
    a = argparse.ArgumentParser()
    a.add_argument("--python", required = True); a.add_argument("--root", required = True); a.add_argument("--out", required = True)
    a.add_argument("--steps", default = "import_tf,generate,train"); a.add_argument("--tag", default = "")
    a = a.parse_args()
    os.makedirs(a.out, exist_ok = True)
    rows = [run(a.python, a.root, side, step, a.out, a.tag) for step in a.steps.split(",") for side in ("base", "head")]
    path = os.path.join(a.out, "results.json")
    old = json.load(open(path, encoding = "utf-8")) if os.path.exists(path) else []
    json.dump(old + rows, open(path, "w", encoding = "utf-8"), indent = 1)
    for r in rows:
        print(json.dumps(r))
    return 0 if all(r["ok"] for r in rows if r["side"] == "head") else 1

if __name__ == "__main__":
    sys.exit(main())
