"""T7h.5g profile scenario (state's unsloth checkout first on PYTHONPATH; UNSLOTH_LORA_FUSED_PASSES picks the order).

argv: OUT_JSON MODEL
Same LoRA SFT step as T7h.5e (Llama-3.2-1B bf16, r16 all linear, batch 2 x 512, gradient checkpointing "unsloth"),
no counter wrapper. WARM warmup steps, then STEPS timed steps, then PROF steps under torch.profiler: per-kernel
device self time (top 60) and per-op (aten) self device time, so the flag-on vs flag-off delta can be attributed.
"""
import json
import os
import sys
import time
import traceback

OUT, MODEL = sys.argv[1], sys.argv[2]
WARM = int(os.environ.get("T7H5G_WARM", "4"))
STEPS = int(os.environ.get("T7H5G_STEPS", "10"))
PROF = int(os.environ.get("T7H5G_PROF", "2"))
rec = {"model": MODEL, "flag": os.environ.get("UNSLOTH_LORA_FUSED_PASSES", "<unset>")}


def save():
    with open(OUT, "w", encoding = "utf-8") as fh:
        json.dump(rec, fh, indent = 1, default = str)


try:
    import unsloth  # noqa: F401
    from unsloth import FastLanguageModel
    import torch

    rec["unsloth_file"] = unsloth.__file__
    rec["torch"] = torch.__version__
    import unsloth.kernels.utils as KU

    rec["has_fused_rule"] = hasattr(KU, "_lora_fused_ok")
    rec["LORA_FUSED_PASSES"] = getattr(KU, "LORA_FUSED_PASSES", None)
    torch.manual_seed(3407)
    model, tok = FastLanguageModel.from_pretrained(MODEL, max_seq_length = 1024, dtype = torch.bfloat16,
                                                   load_in_4bit = False)
    model = FastLanguageModel.get_peft_model(model, r = 16, lora_alpha = 16, lora_dropout = 0, bias = "none",
                                             target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj",
                                                               "up_proj", "down_proj"],
                                             use_gradient_checkpointing = "unsloth", random_state = 3407)
    text = ("The quick brown fox jumps over the lazy dog. " * 40)
    ids = tok([text, "Unsloth makes fine-tuning faster and uses less memory. " * 40], return_tensors = "pt",
              truncation = True, max_length = 512, padding = "max_length").input_ids.cuda()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr = 1e-4)
    model.train()

    def step():
        out = model(input_ids = ids, labels = ids)
        out.loss.backward()
        opt.step()
        opt.zero_grad(set_to_none = True)
        return float(out.loss)

    ms, losses = [], []
    for i in range(WARM + STEPS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        losses.append(step())
        torch.cuda.synchronize()
        ms.append((time.perf_counter() - t0) * 1000)
    rec["step_ms"] = ms
    rec["losses"] = losses
    rec["median_timed_ms"] = sorted(ms[WARM:])[len(ms[WARM:]) // 2]
    save()

    from torch.profiler import ProfilerActivity, profile

    with profile(activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(PROF):
            step()
        torch.cuda.synchronize()

    def dev_self(e):
        for a in ("self_device_time_total", "self_cuda_time_total"):
            v = getattr(e, a, None)
            if v is not None:
                return v
        return 0

    def dev_total(e):
        for a in ("device_time_total", "cuda_time_total"):
            v = getattr(e, a, None)
            if v is not None:
                return v
        return 0

    ka = prof.key_averages()
    rows = [{"name": e.key[:200], "count": e.count, "self_dev_us": dev_self(e), "dev_total_us": dev_total(e),
             "cpu_total_us": e.cpu_time_total} for e in ka]
    rows.sort(key = lambda r: -r["self_dev_us"])
    rec["prof_steps"] = PROF
    rec["kernels_top"] = [r for r in rows if r["self_dev_us"] > 0][:80]
    rec["aten_top"] = sorted([r for r in rows if r["name"].startswith("aten::")], key = lambda r: -r["dev_total_us"])[:60]
    rec["total_self_dev_us"] = sum(r["self_dev_us"] for r in rows)
    rec["done"] = True
except Exception as e:  # noqa: BLE001
    rec["error"] = f"{type(e).__name__}: {e} | {traceback.format_exc()[-2000:]}"
save()
