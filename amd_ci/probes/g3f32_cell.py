#!/usr/bin/env python3
"""One cell of the Gemma 3 float32 check (unsloth #15879 + unsloth-zoo #2227). Observes only.

A cell loads Gemma 3 once or several times in ONE process (--seq, e.g. float16,bfloat16), generates greedily in
every loaded model, then LoRA SFTs the LAST one for --steps steps. --import-flag 1 sets UNSLOTH_FORCE_FLOAT32=1
before `import unsloth`. Writes JSON to --out after every stage (never stdout).
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
import traceback
from pathlib import Path

MODELS = {"tiny": "hf-internal-testing/tiny-random-Gemma3ForCausalLM", "270m": "unsloth/gemma-3-270m-it"}
WORDS = ("the of and to in is was for on as with by he at from his an were are which this be "
         "or has had not but first one their its new after who they have two also been other more all "
         "she time year can city world river music house water light green small large north south").split()
CLASSES = ("Gemma3TextScaledWordEmbedding", "Gemma3MLP", "Gemma3RMSNorm", "Gemma3Attention")


def make_docs(n = 32, seed = 0):
    rng = random.Random(seed)
    return [" ".join(rng.choice(WORDS) for _ in range(rng.choice([8, 15, 25, 40]))) + "." for _ in range(n)]


def ident(f):
    f = getattr(f, "_torchdynamo_orig_callable", f)
    while hasattr(f, "__wrapped__"):
        f = f.__wrapped__
    c = getattr(f, "__code__", None)
    mod = getattr(f, "__module__", "?")
    return f"{mod}.{getattr(f, '__qualname__', '?')}@{c.co_firstlineno if c else '?'}"


def forwards():
    import transformers.models.gemma3.modeling_gemma3 as mg
    return {k: ident(getattr(mg, k).forward) for k in CLASSES if hasattr(mg, k)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--model", required = True, choices = tuple(MODELS))
    ap.add_argument("--seq", required = True)
    ap.add_argument("--quant", type = int, default = 16, choices = (4, 16))
    ap.add_argument("--import-flag", default = "unset")
    ap.add_argument("--loader", default = "fast_model", choices = ("fast_model", "fast_language_model"))
    ap.add_argument("--steps", type = int, default = 4)
    ap.add_argument("--work", required = True, type = Path)
    args = ap.parse_args()
    res: dict = {"stage": "start", "model": MODELS[args.model], "seq": args.seq, "quant": args.quant,
                 "import_flag": args.import_flag, "loader": args.loader, "loads": []}

    def save():
        args.out.write_text(json.dumps(res, indent = 2, default = str), encoding = "utf-8")

    try:
        if args.import_flag != "unset":
            os.environ["UNSLOTH_FORCE_FLOAT32"] = args.import_flag
        res["env_at_import"] = {k: os.environ.get(k) for k in ("UNSLOTH_FORCE_FLOAT32", "TORCHDYNAMO_DISABLE",
                                                               "UNSLOTH_COMPILE_DISABLE", "UNSLOTH_COMPILE_LOCATION")}
        res["stage"] = "import"
        save()
        import unsloth
        from unsloth import FastLanguageModel, FastModel
        import unsloth_zoo
        import torch
        import torch._dynamo
        import transformers
        res["unsloth_file"], res["zoo_file"] = unsloth.__file__, unsloth_zoo.__file__
        p = torch.cuda.get_device_properties(0)
        v = {"torch": torch.__version__, "hip": torch.version.hip, "cuda": torch.version.cuda,
             "transformers": transformers.__version__, "device": p.name, "arch": getattr(p, "gcnArchName", None),
             "dynamo_disable": bool(getattr(torch._dynamo.config, "disable", False))}
        for m in ("trl", "peft", "bitsandbytes", "triton", "unsloth", "unsloth_zoo"):
            try:
                v[m] = __import__(m).__version__
            except Exception as e:  # noqa: BLE001
                v[m] = f"ERR {type(e).__name__}"
        res["versions"] = v
        res["flag_after_import"] = os.environ.get("UNSLOTH_FORCE_FLOAT32")
        res["forwards_after_import"] = safe_call(forwards)
        Loader = FastModel if args.loader == "fast_model" else FastLanguageModel
        model = tok = None
        seq = args.seq.split(",")
        for i, dt in enumerate(seq):
            res["stage"] = f"load{i}_{dt}"
            save()
            model = None
            gc.collect()
            torch.cuda.empty_cache()
            model, tok = Loader.from_pretrained(MODELS[args.model], dtype = getattr(torch, dt), max_seq_length = 256,
                                                load_in_4bit = args.quant == 4)
            tt = getattr(tok, "tokenizer", tok)
            rec = {"dtype": dt, "flag_after_load": os.environ.get("UNSLOTH_FORCE_FLOAT32"),
                   "forwards": safe_call(forwards), "model_dtype": str(getattr(model, "dtype", None)),
                   "n_linear4bit": sum(1 for m in model.modules() if type(m).__name__ == "Linear4bit")}
            ids = tt("The river and the green house", return_tensors = "pt").input_ids.to("cuda")
            try:
                with torch.no_grad():
                    rec["embed_out_dtype"] = str(model.get_input_embeddings()(ids).dtype)
            except Exception as e:  # noqa: BLE001
                rec["embed_out_dtype"] = f"ERR {type(e).__name__}: {str(e)[:200]}"
            res["loads"].append(rec)
            res["stage"] = f"gen{i}_{dt}"
            save()
            try:
                with torch.no_grad():
                    out = model.generate(input_ids = ids, attention_mask = torch.ones_like(ids), max_new_tokens = 12,
                                         do_sample = False, use_cache = True)
                rec["gen_ids"] = out[0, ids.shape[1]:].tolist()
            except Exception as e:  # noqa: BLE001
                rec["gen_error"] = f"{type(e).__name__}: {str(e)[:600]}"
                rec["gen_traceback"] = traceback.format_exc()[-2500:]
            save()
        res["stage"] = "peft"
        save()
        from datasets import Dataset
        from trl import SFTConfig, SFTTrainer
        model = Loader.get_peft_model(model, r = 8, lora_alpha = 16, lora_dropout = 0.0, random_state = 0,
                                      target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                                                        "gate_proj", "up_proj", "down_proj"],
                                      use_gradient_checkpointing = "unsloth")
        tt = getattr(tok, "tokenizer", tok)
        if tt.pad_token is None:
            tt.pad_token = tt.eos_token
        last = seq[-1]
        ds = Dataset.from_dict({"text": [d + (tt.eos_token or "") for d in make_docs()]})
        cfg = SFTConfig(output_dir = str(args.work / "trainer"), per_device_train_batch_size = 2,
                        gradient_accumulation_steps = 1, max_steps = args.steps, learning_rate = 2e-4, logging_steps = 1,
                        fp16 = last == "float16", bf16 = last == "bfloat16", report_to = "none", seed = 0,
                        dataset_text_field = "text", max_length = 64, save_strategy = "no", dataset_num_proc = 1,
                        warmup_steps = 0, packing = False)
        res["stage"] = "train"
        save()
        trainer = SFTTrainer(model = model, train_dataset = ds, processing_class = tt, args = cfg)
        res["attn_impl"] = getattr(model.config, "_attn_implementation", None)
        trainer.train()
        logs = [h for h in trainer.state.log_history if "loss" in h]
        res["losses"] = [float(h["loss"]) for h in logs]
        res["grad_norms"] = [float(h["grad_norm"]) if h.get("grad_norm") is not None else None for h in logs]
        res["forwards_after_train"] = safe_call(forwards)
        loc = os.environ.get("UNSLOTH_COMPILE_LOCATION")
        res["compiled_cache_files"] = len(list(Path(loc).glob("*.py"))) if loc and Path(loc).is_dir() else 0
        try:
            res["dynamo_frames"] = {k: int(v) for k, v in torch._dynamo.utils.counters["stats"].items()}
        except Exception:  # noqa: BLE001
            pass
        res["stage"] = "done"
    except Exception as e:  # noqa: BLE001
        res["error"] = f"{type(e).__name__}: {e}"[:1500]
        res["traceback"] = traceback.format_exc()[-5000:]
    save()
    return 0


def safe_call(fn):
    try:
        return fn()
    except Exception as e:  # noqa: BLE001
        return f"ERR {type(e).__name__}: {str(e)[:200]}"


if __name__ == "__main__":
    sys.exit(main())
