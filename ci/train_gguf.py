"""Notebook-style SFT run + GGUF export, the way the Unsloth notebooks do it.

python train_gguf.py OUTDIR [--steps N] [--model NAME] [--no-gguf]
"""
import argparse, os, sys, time, json, subprocess

ap = argparse.ArgumentParser()
ap.add_argument("outdir")
ap.add_argument("--steps", type = int, default = 30)
ap.add_argument("--model", default = "unsloth/Qwen3-0.6B-unsloth-bnb-4bit")
ap.add_argument("--no-gguf", action = "store_true")
ap.add_argument("--compile-check", action = "store_true")
args = ap.parse_args()

from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

t0 = time.time()
if args.compile_check:
    f = torch.compile(lambda x: torch.nn.functional.gelu(x) * 2 + x.sin())
    x = torch.randn(1024, 1024, device = "cuda")
    y = f(x)
    torch.cuda.synchronize()
    print("TORCH_COMPILE_OK", float(y.abs().mean()), flush = True)

model, tok = FastLanguageModel.from_pretrained(args.model, max_seq_length = 1024, load_in_4bit = True)
model = FastLanguageModel.get_peft_model(
    model, r = 16, lora_alpha = 16, lora_dropout = 0, bias = "none",
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing = "unsloth", random_state = 3407,
)
ds = load_dataset("yahma/alpaca-cleaned", split = "train[:400]")
EOS = tok.eos_token


def fmt(ex):
    return {"text": [f"### Instruction:\n{i}\n\n### Input:\n{n}\n\n### Response:\n{o}{EOS}"
                     for i, n, o in zip(ex["instruction"], ex["input"], ex["output"])]}


ds = ds.map(fmt, batched = True)
trainer = SFTTrainer(
    model = model, tokenizer = tok, train_dataset = ds,
    args = SFTConfig(
        dataset_text_field = "text", max_seq_length = 1024, per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4, warmup_steps = 5, max_steps = args.steps, learning_rate = 2e-4,
        logging_steps = 1, optim = "adamw_8bit", weight_decay = 0.01, lr_scheduler_type = "linear",
        seed = 3407, output_dir = os.path.join(args.outdir, "trainer"), report_to = "none",
        dataloader_num_workers = 2,
    ),
)
stats = trainer.train()
losses = [h["loss"] for h in trainer.state.log_history if "loss" in h]
print("LOSSES", json.dumps([round(l, 4) for l in losses]), flush = True)
print("TRAIN_OK first=%.4f last=%.4f runtime=%.1fs peak_mem=%.2fGB" % (
    losses[0], losses[-1], stats.metrics["train_runtime"], torch.cuda.max_memory_reserved() / 1e9), flush = True)

FastLanguageModel.for_inference(model)
msgs = tok("### Instruction:\nWhat is the capital of France?\n\n### Input:\n\n\n### Response:\n", return_tensors = "pt").to("cuda")
out = model.generate(**msgs, max_new_tokens = 32)
print("GEN", repr(tok.decode(out[0][msgs["input_ids"].shape[1]:], skip_special_tokens = True)), flush = True)

if not args.no_gguf:
    g0 = time.time()
    model.save_pretrained_gguf(os.path.join(args.outdir, "gguf"), tok, quantization_method = "q4_k_m")
    files = []
    for root, _, fs in os.walk(args.outdir):
        files += [os.path.join(root, f) for f in fs if f.endswith(".gguf")]
    print("GGUF_FILES", files, "export %.0fs" % (time.time() - g0), flush = True)
    assert files, "no gguf produced"
print("TOTAL %.0fs" % (time.time() - t0))
