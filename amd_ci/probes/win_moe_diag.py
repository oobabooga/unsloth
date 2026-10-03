#!/usr/bin/env python3
"""Diagnose the Windows ROCm hipErrorInvalidValue in a LoRA matmul on tiny Mixtral (observe only).

Runs with AMD_SERIALIZE_KERNEL=3 so the failing call is the reported one. Steps, each recorded:
plain transformers + PEFT forward (no Unsloth); Unsloth forward with the shapes, strides and dtypes
of every LoRA input captured; then the captured failing matmul replayed standalone in variants.
"""
import json, math, os, sys, traceback

os.environ.setdefault("AMD_SERIALIZE_KERNEL", "3")
os.environ.setdefault("HIP_LAUNCH_BLOCKING", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("UNSLOTH_DISABLE_AUTO_UPDATES", "1")
MODEL = "hf-internal-testing/tiny-random-MixtralForCausalLM"
out = {}


def err(e):
    return f"{type(e).__name__}: {str(e).splitlines()[0][:300]}"


def batch(tok, device):
    tok.pad_token = tok.pad_token or tok.eos_token
    b = tok(["the quick brown fox jumps over the lazy dog"] * 4, return_tensors = "pt", padding = True)
    return {k: v.to(device) for k, v in b.items()}


which = sys.argv[1] if len(sys.argv) > 1 else "all"
if which in ("all", "plain"):
    try:
        import torch, transformers, peft
        tok = transformers.AutoTokenizer.from_pretrained(MODEL)
        m = transformers.AutoModelForCausalLM.from_pretrained(MODEL, dtype = torch.bfloat16).to("cuda")
        m = peft.get_peft_model(m, peft.LoraConfig(r = 8, lora_alpha = 16,
                                target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]))
        b = batch(tok, "cuda")
        loss = m(**b, labels = b["input_ids"]).loss
        loss.backward(); torch.cuda.synchronize()
        out["plain_hf_peft"] = f"ok loss={loss.item():.4f}"
        out["plain_config"] = {k: getattr(m.config, k, None) for k in ("hidden_size", "num_attention_heads",
                               "num_key_value_heads", "head_dim", "num_local_experts", "intermediate_size")}
    except Exception as e:
        out["plain_hf_peft"] = err(e); out["plain_tb"] = traceback.format_exc()[-1500:]

if which in ("all", "unsloth"):
    captured = []
    try:
        import unsloth
        from unsloth import FastLanguageModel
        import torch
        m, tok = FastLanguageModel.from_pretrained(MODEL, max_seq_length = 256, load_in_4bit = False,
                                                   dtype = torch.bfloat16)
        m = FastLanguageModel.get_peft_model(m, r = 8, lora_alpha = 16, random_state = 3407,
                                             target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"])
        def hook(name):
            def f(mod, args):
                x = args[0]
                captured.append({"module": name, "shape": list(x.shape), "stride": list(x.stride()),
                                 "dtype": str(x.dtype), "contig": x.is_contiguous(),
                                 "storage_offset": x.storage_offset(),
                                 "lora_A": list(mod.lora_A["default"].weight.shape) if hasattr(mod, "lora_A") else None,
                                 "lora_A_dtype": str(mod.lora_A["default"].weight.dtype) if hasattr(mod, "lora_A") else None})
            return f
        for n, mod in m.named_modules():
            if n.endswith(("q_proj", "k_proj", "v_proj", "o_proj")) and hasattr(mod, "lora_A"):
                mod.register_forward_pre_hook(hook(n))
        b = batch(tok, "cuda")
        m.train()
        loss = m(**b, labels = b["input_ids"]).loss
        loss.backward(); torch.cuda.synchronize()
        out["unsloth"] = f"ok loss={loss.item():.4f}"
    except Exception as e:
        out["unsloth"] = err(e); out["unsloth_tb"] = traceback.format_exc()[-2500:]
    out["captured_last"] = captured[-6:]
    if captured:
        c = captured[-1]
        import torch
        res = {}
        makers = [
            ("same_strides", lambda: torch.empty_strided(c["shape"], c["stride"], dtype = torch.bfloat16, device = "cuda").normal_()),
            ("contiguous", lambda: torch.randn(c["shape"], dtype = torch.bfloat16, device = "cuda")),
            ("flat2d", lambda: torch.randn(math.prod(c["shape"][:-1]), c["shape"][-1], dtype = torch.bfloat16, device = "cuda")),
        ]
        for blas in ("default", "cublas", "cublaslt"):
            try:
                if blas != "default":
                    torch.backends.cuda.preferred_blas_library(blas)
            except Exception as e:
                res[f"blas_{blas}"] = err(e); continue
            for label, make in makers:
                for wdtype in ("bf16", "fp32"):
                    try:
                        x = make()
                        w = torch.randn(c["lora_A"], dtype = torch.bfloat16 if wdtype == "bf16" else torch.float32, device = "cuda")
                        y = x.to(w.dtype) @ w.t(); torch.cuda.synchronize()
                        res[f"{blas}_{label}_{wdtype}"] = "ok"
                    except Exception as e:
                        res[f"{blas}_{label}_{wdtype}"] = err(e)
        out["replay"] = res
print("WIN_MOE_DIAG " + json.dumps(out))
