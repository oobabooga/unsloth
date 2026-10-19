"""Unsloth LoRA SFT on unsloth/Qwen3-0.6B, fixed synthetic data and seed. Writes --out JSON."""

import argparse
import json
import os
import time
import traceback

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices = ["16bit", "4bit"], required = True)
ap.add_argument("--out", required = True)
ap.add_argument("--work", required = True)
args = ap.parse_args()
res = {"mode": args.mode, "passed": False, "steps": []}


def flush():
    with open(args.out, "w", encoding = "utf-8") as f:
        json.dump(res, f, indent = 1)


try:
    import unsloth  # noqa: F401 - must precede transformers / trl
    from unsloth import FastLanguageModel
    import torch
    from datasets import Dataset
    from transformers import TrainerCallback
    from trl import SFTConfig, SFTTrainer

    import transformers, trl, peft, unsloth_zoo

    res["versions"] = {
        "unsloth": unsloth.__version__, "unsloth_zoo": unsloth_zoo.__version__, "torch": torch.__version__,
        "transformers": transformers.__version__, "trl": trl.__version__, "peft": peft.__version__,
    }
    try:
        import bitsandbytes

        res["versions"]["bitsandbytes"] = bitsandbytes.__version__
    except Exception as e:  # noqa: BLE001
        res["versions"]["bitsandbytes"] = f"unavailable: {e}"[:200]
    steps = 10 if args.mode == "16bit" else 3
    model, tok = FastLanguageModel.from_pretrained(
        "unsloth/Qwen3-0.6B", max_seq_length = 512, dtype = torch.bfloat16,
        load_in_4bit = args.mode == "4bit", full_finetuning = False,
    )
    model = FastLanguageModel.get_peft_model(
        model, r = 16, lora_alpha = 16, lora_dropout = 0, bias = "none", random_state = 3407,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing = "unsloth",
    )
    rows = [
        f"Question: what is {a} plus {b}? Answer: {a} plus {b} is {a + b}."
        + (" The sum is even." if (a + b) % 2 == 0 else " The sum is odd.")
        for a, b in ((i * 7 % 97, i * 13 % 89) for i in range(64))
    ]
    ds = Dataset.from_dict({"text": rows})
    lora_b = [p for n, p in model.named_parameters() if "lora_B" in n][0]
    before = lora_b.detach().float().clone()

    class Rec(TrainerCallback):
        def on_step_begin(self, a, s, c, **k):
            torch.cuda.synchronize()
            self.t = time.time()

        def on_log(self, a, s, c, logs = None, **k):
            if logs and "loss" in logs:
                torch.cuda.synchronize()
                res["steps"].append({
                    "step": s.global_step, "loss": logs.get("loss"), "grad_norm": logs.get("grad_norm"),
                    "time_ms": round((time.time() - self.t) * 1000, 1),
                    "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 3),
                })
                flush()

    trainer = SFTTrainer(
        model = model, processing_class = tok, train_dataset = ds, callbacks = [Rec()],
        args = SFTConfig(
            output_dir = os.path.join(args.work, f"sft_{args.mode}"), dataset_text_field = "text",
            max_length = 256, per_device_train_batch_size = 2, gradient_accumulation_steps = 1,
            max_steps = steps, learning_rate = 2e-4, logging_steps = 1, seed = 3407, bf16 = True,
            optim = "adamw_torch", report_to = "none", save_strategy = "no", dataset_num_proc = 1,
        ),
    )
    trainer.train()
    changed = float((lora_b.detach().float() - before).abs().max())
    losses = [s["loss"] for s in res["steps"]]
    gns = [s["grad_norm"] for s in res["steps"]]
    import math

    res["checks"] = {
        "steps": len(losses) == steps,
        "finite_loss": all(l is not None and math.isfinite(l) for l in losses),
        "grad_norm_ok": all(g is not None and 0 < g < 1e3 for g in gns),
        "lora_changed": changed > 0,
    }
    res["lora_b_max_change"] = changed
    gen_in = tok(["Question: what is 3 plus 4? Answer:"], return_tensors = "pt").to("cuda")
    FastLanguageModel.for_inference(model)
    with torch.no_grad():
        g = model.generate(**gen_in, max_new_tokens = 12, do_sample = False)
    res["generate"] = tok.decode(g[0][gen_in["input_ids"].shape[1]:], skip_special_tokens = True)
    res["passed"] = all(res["checks"].values())
except BaseException as e:  # noqa: BLE001 - a crash is the observation
    res["error"] = f"{type(e).__name__}: {e}"[:1500]
    res["tb"] = traceback.format_exc()[-3000:]
flush()
raise SystemExit(0 if res["passed"] else 1)
