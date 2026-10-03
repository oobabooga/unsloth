#!/usr/bin/env python3
"""Windows ROCm: run Unsloth LoRA SFT with the fused-SDPA fix and log every SDPA call that reaches torch.

Each call is printed (flushed) before it runs, with whether it needs grad, has a mask, uses enable_gqa and
whether the fused kernels are still allowed at that point, so the last line before a crash names the call.
"""
import json, os, sys
os.environ.setdefault("UNSLOTH_DISABLE_AUTO_UPDATES", "1")
os.environ.setdefault("UNSLOTH_ENABLE_LOGGING", "1")
import unsloth
import unsloth.import_fixes as fixes
from unsloth import FastLanguageModel
import torch, torch.nn.functional as F

w = F.scaled_dot_product_attention
info = {"wrapped": hasattr(w, "__wrapped__"), "flag": getattr(w, fixes._SDPA_ROCM_WINDOWS_FLAG, None) if hasattr(fixes, "_SDPA_ROCM_WINDOWS_FLAG") else None}
try:
    info["probe"] = {str(k): sorted(v) for k, v in fixes._rocm_windows_fused_sdpa_failures().items()}
except Exception as e:
    info["probe"] = f"{type(e).__name__}: {e}"
print("TRIP_INFO " + json.dumps(info), flush = True)
n = [0]
if info["wrapped"]:
    idx = w.__code__.co_freevars.index("original")
    orig = w.__closure__[idx].cell_contents
    def trip(*a, **kw):
        q = a[0]
        n[0] += 1
        if n[0] <= 400:
            m = kw.get("attn_mask") if "attn_mask" in kw else (a[3] if len(a) > 3 else None)
            print("TRIP_CALL " + json.dumps({"n": n[0], "grad": torch.is_grad_enabled() and q.requires_grad,
                  "mask": None if m is None else [list(m.shape), str(m.dtype)], "gqa": bool(kw.get("enable_gqa")),
                  "q": list(q.shape), "dtype": str(q.dtype),
                  "fused": torch.backends.cuda.flash_sdp_enabled() or torch.backends.cuda.mem_efficient_sdp_enabled()}), flush = True)
        out = orig(*a, **kw)
        torch.cuda.synchronize()
        return out
    w.__closure__[idx].cell_contents = trip
from datasets import Dataset
from trl import SFTConfig, SFTTrainer
model, tok = FastLanguageModel.from_pretrained(sys.argv[1], max_seq_length = 256, load_in_4bit = False, dtype = torch.bfloat16)
model = FastLanguageModel.get_peft_model(model, r = 8, lora_alpha = 16, random_state = 3407, target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"])
if tok.pad_token is None: tok.pad_token = tok.eos_token
ds = Dataset.from_list([{"text": f"Example {i}: the quick brown fox jumps over the lazy dog."} for i in range(16)])
cfg = SFTConfig(output_dir = "out", per_device_train_batch_size = 4, max_steps = 2, learning_rate = 2e-4, logging_steps = 1, report_to = "none", seed = 3407, bf16 = True, dataset_text_field = "text")
SFTTrainer(model = model, processing_class = tok, train_dataset = ds, args = cfg).train()
print("TRIP_DONE ok", flush = True)
