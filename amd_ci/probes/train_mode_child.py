"""One SFT run on the fake hybrid model: mode padded | packed | packed_varlen. Writes JSON with per-step
time / tokens / loss, packing stats and a profiler kernel table."""
import json, math, os, sys, time

model_dir, out_path, mode = sys.argv[1], sys.argv[2], sys.argv[3]
res = {"mode": mode}


def dump():
    with open(out_path, "w") as f:
        json.dump(res, f, indent = 1, default = str)


try:
    import unsloth
    res["unsloth_file"] = unsloth.__file__
    import torch
    res["torch"], res["hip"] = torch.__version__, torch.version.hip
    import transformers, trl
    res["transformers"], res["trl"] = transformers.__version__, trl.__version__
    if mode == "packed_varlen":
        import unsloth.utils.hf_packed_attention as hpa
        orig = hpa._torch_varlen

        def forced(device):
            orig(device)
            return hpa._TORCH_VARLEN[0]

        hpa._torch_varlen = forced
    from unsloth import FastModel
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    model, tok = FastModel.from_pretrained(model_dir, max_seq_length = 2048, load_in_4bit = False,
                                           dtype = torch.bfloat16)
    model = FastModel.get_peft_model(model, r = 16, lora_alpha = 32, lora_dropout = 0, bias = "none",
                                     use_gradient_checkpointing = "unsloth", random_state = 3407,
                                     target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj",
                                                       "up_proj", "down_proj", "in_proj_qkv", "in_proj_z",
                                                       "out_proj"])
    tok = getattr(tok, "tokenizer", tok)
    import random
    rng = random.Random(3407)
    words = [f"w{i}" for i in range(4000)]
    texts = []
    for i in range(400):
        L = int(min(1900, max(20, math.exp(rng.gauss(5.4, 0.9)))))
        texts.append(" ".join(rng.choice(words) for _ in range(L)))
    ds = Dataset.from_dict({"text": texts})
    args = SFTConfig(output_dir = os.path.join(os.path.dirname(out_path), f"o_{mode}"), packing = True,
                     max_length = 2048, per_device_train_batch_size = 4, gradient_accumulation_steps = 1,
                     max_steps = 12, learning_rate = 2e-4, bf16 = True, logging_steps = 1, seed = 3407,
                     report_to = "none", save_strategy = "no", dataset_text_field = "text",
                     dataloader_num_workers = 0)
    trainer = SFTTrainer(model = model, args = args, train_dataset = ds, processing_class = tok)
    from unsloth.utils import packing as pk
    from unsloth.utils import hf_packed_attention as hpa2
    res["varlen_active_after_init"] = dict(pk.HYBRID_PACKING_STATS)
    steps = []
    prof_steps = {7, 8}
    prof_holder = {}
    orig_step = trainer.training_step

    def timed_step(model, inputs, *a, **k):
        idx = len(steps) + 1
        ids = inputs["input_ids"]
        am = inputs.get("attention_mask")
        tokens = int(am.sum()) if am is not None else int(ids.numel())
        shape = list(ids.shape)
        torch.cuda.synchronize()
        if idx in prof_steps:
            from torch.profiler import profile, ProfilerActivity
            p = profile(activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA])
            p.__enter__()
        t0 = time.perf_counter()
        loss = orig_step(model, inputs, *a, **k)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        if idx in prof_steps:
            p.__exit__(None, None, None)
            rows = []
            for e in p.key_averages():
                dtime = getattr(e, "self_device_time_total", None) or getattr(e, "self_cuda_time_total", 0)
                if dtime:
                    rows.append((e.key[:140], e.count, dtime))
            rows.sort(key = lambda r: -r[2])
            prof_holder[idx] = {"total_device_us": sum(r[2] for r in rows), "top": rows[:45]}
        steps.append({"step": idx, "s": dt, "tokens": tokens, "shape": shape, "loss": float(loss.detach()),
                      "profiled": idx in prof_steps})
        res["steps"] = steps
        dump()
        return loss

    trainer.training_step = timed_step
    res["hf_packed_stats_before"] = dict(hpa2.HF_PACKED_ATTENTION_STATS)
    trainer.train()
    res["hf_packed_stats"] = dict(hpa2.HF_PACKED_ATTENTION_STATS)
    res["hybrid_stats"] = dict(pk.HYBRID_PACKING_STATS)
    res["profile"] = prof_holder
    timed = [s for s in steps if s["step"] >= 3 and not s["profiled"]]
    res["summary"] = {
        "median_s_per_step": sorted(s["s"] for s in timed)[len(timed) // 2],
        "useful_tok_per_s": sum(s["tokens"] for s in timed) / sum(s["s"] for s in timed),
        "mean_tokens_per_step": sum(s["tokens"] for s in timed) / len(timed),
        "peak_gb": torch.cuda.max_memory_allocated() / 2**30,
    }
except Exception as e:
    import traceback
    res["error"] = f"{type(e).__name__}: {e}"
    res["tb"] = traceback.format_exc()[-4000:]
dump()
print("TRAIN_MODE_DONE", mode, flush = True)
