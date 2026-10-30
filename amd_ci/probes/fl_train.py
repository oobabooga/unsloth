#!/usr/bin/env python3
"""Tiny-model fp16 LoRA SFT (3 steps) + GRPO (2 steps) through Unsloth; fused-loss engagement counters, losses,
grad finiteness. --device-map balanced splits the model over every visible GPU. One JSON to --out."""
import argparse, json, os, sys, traceback

ap = argparse.ArgumentParser()
ap.add_argument("--out", required = True)
ap.add_argument("--model", default = "trl-internal-testing/tiny-Qwen3ForCausalLM")
ap.add_argument("--device-map", default = None)
ap.add_argument("--sft-steps", type = int, default = 3)
ap.add_argument("--grpo-steps", type = int, default = 2)
args = ap.parse_args()
os.environ.setdefault("UNSLOTH_DISABLE_AUTO_UPDATES", "1")

from unsloth import FastLanguageModel  # noqa: E402
import torch  # noqa: E402
import unsloth_zoo.fused_losses.fused_linear as fl  # noqa: E402
from datasets import Dataset  # noqa: E402
from trl import GRPOConfig, GRPOTrainer, SFTConfig, SFTTrainer  # noqa: E402
import trl, transformers, unsloth, unsloth_zoo  # noqa: E402

res = dict(device_map = args.device_map, model = args.model, torch = torch.__version__, hip = torch.version.hip,
           trl = trl.__version__, transformers = transformers.__version__, unsloth = unsloth.__version__,
           zoo = unsloth_zoo.__file__, n_gpu = torch.cuda.device_count())


def save():
    with open(args.out, "w", encoding = "utf-8") as f:
        json.dump(res, f, indent = 1)


try:
    kw = dict(device_map = args.device_map) if args.device_map else {}
    torch.manual_seed(0)
    model, tok = FastLanguageModel.from_pretrained(args.model, max_seq_length = 256, dtype = torch.float16,
                                                   load_in_4bit = False, **kw)
    model = FastLanguageModel.get_peft_model(model, r = 8, lora_alpha = 16, random_state = 0,
                                             target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                                                               "gate_proj", "up_proj", "down_proj"])
    res["param_devices"] = sorted({str(p.device) for p in model.parameters()})
    res["lm_head_device"] = str(model.get_output_embeddings().weight.device)
    res["embed_device"] = str(model.get_input_embeddings().weight.device)
    res["vocab"], res["hidden"] = model.config.vocab_size, model.config.hidden_size
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    rows = [{"text": f"Question {i}: what is {i} plus {i}? Answer: {2 * i}." + (tok.eos_token or "")} for i in range(32)]
    c0 = dict(fl.CALLS)
    sft = SFTTrainer(model = model, processing_class = tok, train_dataset = Dataset.from_list(rows),
                     args = SFTConfig(output_dir = "sft_out", max_steps = args.sft_steps, per_device_train_batch_size = 4,
                                      learning_rate = 2e-4, logging_steps = 1, fp16 = True, bf16 = False, seed = 0,
                                      report_to = "none", save_strategy = "no", dataset_text_field = "text",
                                      max_length = 128))
    out = sft.train()
    res["sft_losses"] = [h["loss"] for h in sft.state.log_history if "loss" in h]
    res["sft_calls"] = {k: fl.CALLS[k] - c0.get(k, 0) for k in fl.CALLS if fl.CALLS[k] != c0.get(k, 0)}
    res["sft_lora_grad_finite"] = all(torch.isfinite(p).all().item() for n, p in model.named_parameters() if "lora" in n)
    save()

    def reward(completions, **kw):
        return [float(len(c) % 7) + 0.01 * i for i, c in enumerate(completions)]

    prompts = [{"prompt": f"Count to {i}:"} for i in range(16)]
    c0 = dict(fl.CALLS)
    cfg = GRPOConfig(output_dir = "grpo_out", max_steps = args.grpo_steps, per_device_train_batch_size = 4,
                     num_generations = 4, max_completion_length = 16, learning_rate = 1e-4, logging_steps = 1,
                     fp16 = True, bf16 = False, seed = 0, report_to = "none", save_strategy = "no")
    grpo = GRPOTrainer(model = model, reward_funcs = [reward], args = cfg, train_dataset = Dataset.from_list(prompts),
                       processing_class = tok)
    grpo.train()
    res["grpo_losses"] = [h["loss"] for h in grpo.state.log_history if "loss" in h]
    res["grpo_calls"] = {k: fl.CALLS[k] - c0.get(k, 0) for k in fl.CALLS if fl.CALLS[k] != c0.get(k, 0)}
    res["ok"] = True
except Exception as e:
    res["ok"] = False
    res["error"] = f"{type(e).__name__}: {e}"[:600]
    res["tb"] = traceback.format_exc()[-3000:]
save()
print("FLTRAIN_DONE", res.get("ok"), flush = True)
