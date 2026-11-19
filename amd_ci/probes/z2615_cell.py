#!/usr/bin/env python3
"""Cell for unsloth-zoo #2615. --mode build: tiny random Llama (seeded) + tokenizer into --dir.
--mode run: FastLanguageModel LoRA, use_gradient_checkpointing="unsloth"; one forward under torch.no_grad() in train
mode (records NO_GRAD_CHECKPOINT_STATS and the logits), then 2 SFTTrainer steps (records losses). JSON to --out."""
import argparse
import json
import os
import sys
import traceback

ap = argparse.ArgumentParser()
ap.add_argument("--mode", required = True)
ap.add_argument("--dir", required = True)
ap.add_argument("--out", required = True)
ap.add_argument("--work", default = None)
a = ap.parse_args()
res = {"mode": a.mode, "stage": "start"}


def dump():
    with open(a.out, "w", encoding = "utf-8") as f:
        json.dump(res, f, indent = 1, default = str)


try:
    if a.mode == "build":
        import torch
        from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM
        tok = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-LlamaForCausalLM")
        cfg = LlamaConfig(vocab_size = len(tok), hidden_size = 256, intermediate_size = 512, num_hidden_layers = 2,
                          num_attention_heads = 4, num_key_value_heads = 2, max_position_embeddings = 512,
                          bos_token_id = tok.bos_token_id, eos_token_id = tok.eos_token_id, pad_token_id = tok.pad_token_id,
                          tie_word_embeddings = False)
        torch.manual_seed(0)
        m = LlamaForCausalLM(cfg).to(torch.bfloat16)
        m.save_pretrained(a.dir)
        tok.save_pretrained(a.dir)
        res.update(stage = "done", params = sum(p.numel() for p in m.parameters()))
    else:
        import unsloth
        from unsloth import FastLanguageModel
        import torch
        import unsloth_zoo
        import unsloth_zoo.gradient_checkpointing as gcm
        res["files"] = {"unsloth": unsloth.__file__, "unsloth_zoo": unsloth_zoo.__file__, "gc": gcm.__file__}
        res["versions"] = {"torch": torch.__version__, "hip": torch.version.hip, "unsloth": unsloth.__version__,
                           "arch": str(getattr(torch.cuda.get_device_properties(0), "gcnArchName", ""))}
        res["has_stats"] = hasattr(gcm, "NO_GRAD_CHECKPOINT_STATS")
        dump()
        model, tok = FastLanguageModel.from_pretrained(a.dir, max_seq_length = 128, dtype = torch.bfloat16,
                                                       load_in_4bit = False)
        model = FastLanguageModel.get_peft_model(model, r = 8, lora_alpha = 16, lora_dropout = 0.0, bias = "none",
                                                 target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj",
                                                                   "up_proj", "down_proj"],
                                                 use_gradient_checkpointing = "unsloth", random_state = 0)
        res["stage"] = "loaded"
        dump()
        g = torch.Generator().manual_seed(0)
        ids = torch.randint(10, len(tok) - 1, (2, 48), generator = g).cuda()
        model.train()
        before = dict(getattr(gcm, "NO_GRAD_CHECKPOINT_STATS", {}) or {})
        # Count the checkpoint autograd Functions' forwards during the no-grad pass: base routes there, head must not.
        fn_calls = {}
        originals = []
        for cname in ("UnslothCheckpointFunction", "Unsloth_Gradient_Checkpointer", "Unsloth_Offloaded_Gradient_Checkpointer"):
            cls = getattr(gcm, cname, None)
            if cls is None or "forward" not in cls.__dict__:
                continue
            orig = cls.__dict__["forward"]
            fwd = orig.__func__ if isinstance(orig, staticmethod) else orig

            def make(f, n):
                def counted(*args, **kw):
                    fn_calls[n] = fn_calls.get(n, 0) + 1
                    return f(*args, **kw)
                return staticmethod(counted)
            setattr(cls, "forward", make(fwd, cname))
            originals.append((cls, orig))
        try:
            with torch.no_grad():
                # use_cache=False: Unsloth's fast forward only checkpoints layers when the cache is off (as in
                # GRPO / DPO reference log-prob passes).
                out = model(input_ids = ids, attention_mask = torch.ones_like(ids), use_cache = False)
        finally:
            for cls, orig in originals:
                setattr(cls, "forward", orig)
        torch.cuda.synchronize()
        lg = out.logits.float()
        res["nograd"] = {"finite": bool(torch.isfinite(lg).all()), "sum": float(lg.sum()), "abs_sum": float(lg.abs().sum()),
                         "first": [float(x) for x in lg[0, 0, :4]],
                         "stats_before": before, "stats_after": dict(getattr(gcm, "NO_GRAD_CHECKPOINT_STATS", {}) or {}),
                         "training_flag": bool(model.training), "checkpoint_function_forwards": fn_calls}
        res["stage"] = "nograd_done"
        dump()
        from datasets import Dataset
        from trl import SFTConfig, SFTTrainer
        words = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"]
        rows = [{"text": " ".join(words[(i + j) % 8] for j in range(30))} for i in range(8)]
        args = SFTConfig(output_dir = os.path.join(a.work or ".", "sft_out"), per_device_train_batch_size = 2,
                         gradient_accumulation_steps = 1, max_steps = 2, learning_rate = 1e-3, logging_steps = 1,
                         report_to = "none", seed = 0, bf16 = True, fp16 = False, dataset_text_field = "text",
                         max_length = 64, packing = False, save_strategy = "no", dataloader_num_workers = 0)
        tr = SFTTrainer(model = model, train_dataset = Dataset.from_list(rows), processing_class = tok, args = args)
        tr.train()
        res["losses"] = {str(h["step"]): float(h["loss"]) for h in tr.state.log_history if "loss" in h}
        res["grad_norms"] = {str(h["step"]): float(h["grad_norm"]) for h in tr.state.log_history if "grad_norm" in h}
        res["stats_end"] = dict(getattr(gcm, "NO_GRAD_CHECKPOINT_STATS", {}) or {})
        res["peak_gb"] = torch.cuda.max_memory_allocated() / 1e9
        res["stage"] = "done"
except Exception as e:  # noqa: BLE001
    res["error"] = f"{type(e).__name__}: {str(e)[:500]}"
    res["traceback"] = traceback.format_exc()[-3000:]
dump()
