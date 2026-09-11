"""Real Apple Silicon MLX: the CLI's raw-text path (text chunks) through the real MLX SFTTrainer.
Reads the input_ids the trainer itself produces and counts EOS, main vs PR, then trains the PR."""
import argparse, importlib.util, json, sys

import unsloth

assert unsloth._IS_MLX, "expected the MLX backend on Apple Silicon"
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


cli = load("unsloth_cli_script", "unsloth-cli.py")
DOC = "ci_pr10734/doc.txt"
words = "river stone mountain signal archive lantern harbor meadow circuit orbit thunder".split()
with open(DOC, "w", encoding = "utf-8") as f:
    f.write("\n\n".join(
        " ".join(f"{words[(p * 7 + s * 3 + k) % len(words)]}" for k in range(12)).capitalize()
        + f" (paragraph {p}, sentence {s})." for p in range(12) for s in range(6)
    ))

model, tok = FastLanguageModel.from_pretrained("unsloth/gemma-3-270m-it", max_seq_length = 128)
model = FastLanguageModel.get_peft_model(model, r = 8, lora_alpha = 16,
                                         target_modules = ["q_proj", "v_proj"], random_state = 3407)
eos_id = tok.eos_token_id
print("eos", tok.eos_token, eos_id, "tokenizer", type(tok).__name__)
out = {}
for v, path in [("main", "ci_pr10734/main_raw_text.py"), ("head", "unsloth/dataprep/raw_text.py")]:
    rt = load(f"rt_{v}", path)
    loader = cli._raw_text_loader_for_backend(rt.RawTextDataLoader, tok, True, 64, 16)
    ds = loader.load_from_file(DOC)
    chunks = ds["text"]
    ns = argparse.Namespace(
        per_device_train_batch_size = 2, gradient_accumulation_steps = 1, warmup_steps = 0,
        max_steps = 3, learning_rate = 2e-4, logging_steps = 1, optim = "adamw_8bit",
        weight_decay = 0.0, lr_scheduler_type = "linear", seed = 3407, output_dir = f"ci_pr10734/out_{v}",
        report_to = "none", max_seq_length = 128, packing = False, per_device_eval_batch_size = 4,
    )
    targs = cli._build_sft_config(SFTConfig, ns, True, False)
    trainer = SFTTrainer(model = model, processing_class = tok, train_dataset = ds, args = targs)
    view = trainer.train_dataset
    rows = [list(view[i]["input_ids"]) for i in range(len(view))]
    out[v] = dict(
        n_chunks = len(chunks), text_chunks_with_eos = sum(c.endswith(tok.eos_token) for c in chunks),
        append_eos = getattr(trainer.args, "append_eos", None), view = type(view).__name__,
        rows_with_eos_id = sum(eos_id in r for r in rows), last_row_has_eos = eos_id in rows[-1],
        row_lens = sorted({len(r) for r in rows}),
    )
    print(v, json.dumps(out[v]))
    if v == "head":
        res = trainer.train()
        out[v]["train_loss"] = float(res.training_loss) if hasattr(res, "training_loss") else str(res)
        print("head training:", out[v]["train_loss"])
assert out["main"]["rows_with_eos_id"] == out["main"]["n_chunks"], out["main"]
assert out["head"]["rows_with_eos_id"] == 1 and out["head"]["last_row_has_eos"], out["head"]
assert out["head"]["text_chunks_with_eos"] == 1
print("MLX RAW TEXT PASS", json.dumps(out))
