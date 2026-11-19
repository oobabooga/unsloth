# Gemma 4 LoRA / QLoRA training step on a right-padded batch whose length is not a multiple of 8.
# argv: model_path  [--4bit] [--lengths 37,32] [--steps 2] [--trainer]
import os, sys, json, argparse
os.environ.setdefault("UNSLOTH_DISABLE_AUTO_UPDATES", "1")
p = argparse.ArgumentParser()
p.add_argument("model"); p.add_argument("--4bit", dest="q4", action="store_true")
p.add_argument("--lengths", default="37,30"); p.add_argument("--steps", type=int, default=2)
p.add_argument("--trainer", action="store_true"); p.add_argument("--gc", default="unsloth"); p.add_argument("--tag", default="")
a = p.parse_args()
import unsloth
from unsloth import FastModel
import unsloth_zoo, torch, transformers
print("ARM", a.tag, "unsloth", unsloth.__file__, unsloth.__version__, "zoo", unsloth_zoo.__file__, unsloth_zoo.__version__,
      "torch", torch.__version__, "transformers", transformers.__version__, torch.cuda.get_device_name(0),
      "BLOCK_COMPILE", os.environ.get("UNSLOTH_BLOCK_COMPILE"), flush=True)
model, tok = FastModel.from_pretrained(a.model, max_seq_length=256, load_in_4bit=a.q4, dtype=torch.bfloat16)
model = FastModel.get_peft_model(model, r=8, lora_alpha=8, target_modules="all-linear" if False else
    ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing={"unsloth": "unsloth", "true": True, "false": False}[a.gc], random_state=0)
lengths = [int(x) for x in a.lengths.split(",")]
T = max(lengths)
g = torch.Generator().manual_seed(0)
ids = torch.randint(10, 1000, (len(lengths), T), generator=g)
mask = torch.zeros(len(lengths), T, dtype=torch.long)
for i, L in enumerate(lengths):
    mask[i, :L] = 1
ids[mask == 0] = 0
labels = ids.clone(); labels[mask == 0] = -100
dev = "cuda"
compiled = sorted({type(m).__name__ for m in model.modules() if getattr(type(m).forward, "_unsloth_block_compiled", False)})
print("block compiled classes:", compiled, flush=True)
import torch._inductor.lowering as _low
print("sdpa_constraint bwd_bias_aligned:", getattr(_low.sdpa_constraint, "_unsloth_bwd_bias_aligned", False),
      "table:", getattr(_low._maybe_layout_constraints.get(torch.ops.aten._scaled_dot_product_efficient_attention_backward.default), "_unsloth_bwd_bias_aligned", False), flush=True)
status = "PASS"
if a.trainer:
    from datasets import Dataset
    from trl import SFTTrainer, SFTConfig
    rows = [{"input_ids": ids[i, :L].tolist(), "attention_mask": [1] * L, "labels": ids[i, :L].tolist()} for i, L in enumerate(lengths)] * 4
    from transformers import DataCollatorForSeq2Seq
    tr = SFTTrainer(model=model, train_dataset=Dataset.from_list(rows), processing_class=tok.tokenizer if hasattr(tok, "tokenizer") else tok,
        data_collator=DataCollatorForSeq2Seq(tok.tokenizer if hasattr(tok, "tokenizer") else tok, padding=True),
        args=SFTConfig(per_device_train_batch_size=len(lengths), max_steps=a.steps, learning_rate=1e-4, logging_steps=1,
                       output_dir="/tmp/g4a_out", report_to="none", bf16=True, dataset_kwargs={"skip_prepare_dataset": True},
                       remove_unused_columns=False, seed=0))
    try:
        tr.train()
    except Exception as e:
        status = f"FAIL {type(e).__name__}: {str(e).splitlines()[0][:300]}"
else:
    model.train()
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    for step in range(a.steps):
        try:
            out = model(input_ids=ids.to(dev), attention_mask=mask.to(dev), labels=labels.to(dev))
            out.loss.backward()
            gn = torch.sqrt(sum((p.grad.float() ** 2).sum() for p in model.parameters() if p.grad is not None))
            torch.cuda.synchronize()
            print(f"step {step} T={T} loss={out.loss.item():.6f} grad_norm={gn.item():.6f}", flush=True)
            opt.step(); opt.zero_grad(set_to_none=True)
        except Exception as e:
            status = f"FAIL {type(e).__name__}: {str(e).splitlines()[0][:300]}"
            import traceback; tb = traceback.format_exc().splitlines()
            print("\n".join(tb[-12:]))
            break
print("RESULT", a.tag, status, flush=True)
