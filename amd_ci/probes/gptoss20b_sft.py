"""Short LoRA SFT on real GPT-OSS 20b through Unsloth + TRL (what users run): losses, step times, adapter update.

    python gptoss20b_sft.py --out OUT.json [--model unsloth/gpt-oss-20b-unsloth-bnb-4bit] [--no-4bit] [--steps 5]

Synthetic dataset (no download), fixed seed, batch 1, 512-token rows, adamw_torch (no 8-bit optimizer dependency).
step_s = per-step wall time from a TrainerCallback (step 1 includes compile and is reported separately).
"""
import argparse
import json
import os
import random
import time

ap = argparse.ArgumentParser()
ap.add_argument("--out", required = True)
ap.add_argument("--model", default = "unsloth/gpt-oss-20b-unsloth-bnb-4bit")
ap.add_argument("--no-4bit", action = "store_true")
ap.add_argument("--steps", type = int, default = 5)
ap.add_argument("--seq", type = int, default = 512)
a = ap.parse_args()
os.environ.setdefault("UNSLOTH_DISABLE_AUTO_UPDATES", "1")

from unsloth import FastLanguageModel  # noqa: E402
import torch  # noqa: E402
from datasets import Dataset  # noqa: E402
from transformers import TrainerCallback  # noqa: E402
from trl import SFTConfig, SFTTrainer  # noqa: E402

res = {"model": a.model, "load_in_4bit": not a.no_4bit, "steps": a.steps}
try:
    t0 = time.time()
    model, tok = FastLanguageModel.from_pretrained(a.model, max_seq_length = a.seq + 64, load_in_4bit = not a.no_4bit,
                                                   dtype = torch.bfloat16)
    res["load_s"] = round(time.time() - t0, 1)
    model = FastLanguageModel.get_peft_model(model, r = 8, lora_alpha = 16, random_state = 3407,
                                             target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"],
                                             use_gradient_checkpointing = "unsloth")
    rng = random.Random(0)
    words = ("the model learns attention sinks flex kernels decode prefill memory speed gradient adapter rank "
             "token batch window sliding cache static dynamic compile graph kernel tile block").split()
    rows = []
    for i in range(64):
        body = " ".join(rng.choice(words) for _ in range(a.seq))
        rows.append({"text": tok.apply_chat_template([{"role": "user", "content": f"Item {i}: summarise."},
                                                      {"role": "assistant", "content": body}], tokenize = False)})
    ds = Dataset.from_list(rows)
    lora_b0 = {n: p.detach().float().norm().item() for n, p in model.named_parameters() if "lora_B" in n}

    times = []

    class T(TrainerCallback):
        def on_step_begin(self, args, state, control, **kw):
            torch.cuda.synchronize()
            self.t = time.perf_counter()

        def on_step_end(self, args, state, control, **kw):
            torch.cuda.synchronize()
            times.append(time.perf_counter() - self.t)

    torch.cuda.reset_peak_memory_stats()
    trainer = SFTTrainer(model = model, processing_class = tok, train_dataset = ds, callbacks = [T()],
                         args = SFTConfig(max_length = a.seq, dataset_text_field = "text", per_device_train_batch_size = 1,
                                          gradient_accumulation_steps = 1, max_steps = a.steps, learning_rate = 2e-4,
                                          logging_steps = 1, optim = "adamw_torch", seed = 3407, bf16 = True,
                                          report_to = "none", output_dir = os.path.join(os.getcwd(), "sft_out"),
                                          save_strategy = "no", packing = False))
    trainer.train()
    losses = [h["loss"] for h in trainer.state.log_history if "loss" in h]
    lora_b1 = {n: p.detach().float().norm().item() for n, p in model.named_parameters() if "lora_B" in n}
    res.update({
        "losses": losses, "step_s": [round(x, 3) for x in times],
        "peak_alloc_gb": round(torch.cuda.max_memory_allocated() / 2**30, 3),
        "lora_b_before_max": max(lora_b0.values()) if lora_b0 else None,
        "lora_b_after_min": min(lora_b1.values()) if lora_b1 else None,
        "lora_b_count": len(lora_b1),
        "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
    })
    import unsloth_zoo
    res["zoo_file"] = unsloth_zoo.__file__
except Exception as e:  # noqa: BLE001
    import traceback
    res["error"] = f"{type(e).__name__}: {str(e)[:500]}"
    res["traceback"] = traceback.format_exc()[-3000:]
with open(a.out, "w") as f:
    json.dump(res, f, indent = 1)
print("SFT", {k: v for k, v in res.items() if k != "traceback"}, flush = True)
