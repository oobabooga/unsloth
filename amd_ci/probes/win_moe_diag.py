#!/usr/bin/env python3
"""Diagnose the Windows ROCm hipErrorInvalidValue in a LoRA matmul on tiny Mixtral (observe only).

Errors surface asynchronously, so the plain run synchronizes after every module to name the first failing one. Steps, each recorded:
plain transformers + PEFT forward (no Unsloth); Unsloth forward with the shapes, strides and dtypes
of every LoRA input captured; then the captured failing matmul replayed standalone in variants.
"""
import json, math, os, sys, traceback

os.environ.setdefault("AMD_SERIALIZE_KERNEL", "3")  # the HIP runtime reads 3; torch warns it is not 0/1
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
ATTN = os.environ.get("DIAG_ATTN") or None
if os.environ.get("DIAG_MODEL"):
    MODEL = os.environ["DIAG_MODEL"]
out["model"], out["attn"] = MODEL, ATTN

if which in ("all", "sdpa"):
    # Standalone SDPA, synchronized: per backend, head_dim and call shape. Fused kernels are AOTriton on ROCm.
    import torch
    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel
    out["aotriton_experimental_env"] = os.environ.get("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL")
    out["arch"] = torch.cuda.get_device_properties(0).gcnArchName
    for fn in ("is_flash_attention_available",):
        try: out[fn] = getattr(torch.backends.cuda, fn)()
        except Exception as e: out[fn] = err(e)
    def call(variant, hd):
        dev, dt = "cuda", torch.bfloat16
        q = torch.randn(4, 4, 12, hd, device = dev, dtype = dt, requires_grad = True)
        nkv = 2 if variant == "gqa" else 4
        k = torch.randn(4, nkv, 12, hd, device = dev, dtype = dt, requires_grad = True)
        v = torch.randn(4, nkv, 12, hd, device = dev, dtype = dt, requires_grad = True)
        kw = {"is_causal": variant in ("causal_fwd", "causal_bwd", "gqa")}
        if variant == "gqa": kw["enable_gqa"] = True
        if variant == "bool_mask": kw["attn_mask"] = torch.ones(4, 1, 12, 12, device = dev, dtype = torch.bool).tril()
        o = F.scaled_dot_product_attention(q, k, v, **kw)
        torch.cuda.synchronize()
        if variant in ("causal_bwd", "gqa"):
            o.float().sum().backward(); torch.cuda.synchronize()
    for name, be in (("flash", SDPBackend.FLASH_ATTENTION), ("efficient", SDPBackend.EFFICIENT_ATTENTION),
                     ("default", None)):
        for hd in (64, 128):
            for variant in ("causal_fwd", "causal_bwd", "noncausal_fwd", "bool_mask", "gqa"):
                key = f"sdpa_{name}_hd{hd}_{variant}"
                try:
                    if be is None:
                        call(variant, hd)
                    else:
                        with sdpa_kernel([be]):
                            call(variant, hd)
                    out[key] = "ok"
                except Exception as e:
                    out[key] = err(e)
                    try: torch.cuda.synchronize()
                    except Exception as e2: out[key] += f" | sync after: {err(e2)}"

if which == "trace_sdpa":
    # What the dense path that trains actually sends to SDPA.
    import torch, torch.nn.functional as F
    calls = []
    _orig = F.scaled_dot_product_attention
    def traced(q, k, v, *a, **kw):
        if len(calls) < 4:
            m_ = kw.get("attn_mask")
            calls.append({"q": list(q.shape), "k": list(k.shape), "dtype": str(q.dtype), "q_contig": q.is_contiguous(),
                          "mask": None if m_ is None else [list(m_.shape), str(m_.dtype)],
                          **{x: kw[x] for x in ("is_causal", "enable_gqa", "dropout_p", "scale") if x in kw}})
        return _orig(q, k, v, *a, **kw)
    F.scaled_dot_product_attention = traced
    import unsloth
    from unsloth import FastLanguageModel
    try:
        m, tok = FastLanguageModel.from_pretrained(MODEL, max_seq_length = 256, load_in_4bit = False, dtype = torch.bfloat16)
        m = FastLanguageModel.get_peft_model(m, r = 8, lora_alpha = 16, target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"])
        b = batch(tok, "cuda")
        loss = m(**b, labels = b["input_ids"]).loss
        loss.backward(); torch.cuda.synchronize()
        out["trace"] = f"ok loss={loss.item():.4f}"
    except Exception as e:
        out["trace"] = err(e)
    out["sdpa_calls"] = calls

if which in ("all", "plain"):
    try:
        import torch, transformers, peft
        tok = transformers.AutoTokenizer.from_pretrained(MODEL)
        m = transformers.AutoModelForCausalLM.from_pretrained(MODEL, dtype = torch.bfloat16,
                                                              **({"attn_implementation": ATTN} if ATTN else {})).to("cuda")
        m = peft.get_peft_model(m, peft.LoraConfig(r = 8, lora_alpha = 16,
                                target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]))
        last = []
        def sync_hook(name):
            def f(mod, args, output):
                out["plain_sync_module"] = name
                torch.cuda.synchronize(); last[:] = [name]
            return f
        for n, mod in m.named_modules():
            if n: mod.register_forward_hook(sync_hook(n))
        b = batch(tok, "cuda")
        try:
            loss = m(**b, labels = b["input_ids"]).loss
        finally:
            out["plain_last_ok_module"] = last[0] if last else None
        loss.backward(); torch.cuda.synchronize()
        out["plain_hf_peft"] = f"ok loss={loss.item():.4f}"
        out["plain_config"] = {k: getattr(m.config, k, None) for k in ("hidden_size", "num_attention_heads",
                               "num_key_value_heads", "head_dim", "num_local_experts", "intermediate_size")}
    except Exception as e:
        out["plain_hf_peft"] = err(e); out["plain_tb"] = traceback.format_exc()[-1500:]

if which in ("all", "grouped_mm"):
    # The same dummy call zoo's _check_torch_grouped_mm_supported makes, but synchronized and at a real shape.
    import torch
    for label, (m_, k_, n_, e_) in {"zoo_probe": (1, 8, 8, 1), "tiny_mixtral": (64, 32, 64, 8)}.items():
        for dt in (torch.float16, torch.bfloat16):
            key = f"grouped_mm_{label}_{str(dt)[6:]}"
            try:
                x = torch.randn((m_, k_), device = "cuda", dtype = dt)
                w = torch.randn((e_, k_, n_), device = "cuda", dtype = dt)
                offs = torch.linspace(m_ / e_, m_, e_, device = "cuda").round().to(torch.int32)
                y = torch._grouped_mm(x, w, offs = offs); torch.cuda.synchronize()
                ref = torch.cat([x[a:b] @ w[i] for i, (a, b) in enumerate(zip([0] + offs.tolist()[:-1], offs.tolist()))])
                out[key] = f"ok maxdiff={(y.float() - ref.float()).abs().max().item():.3g}"
            except Exception as e:
                out[key] = err(e)

if which in ("all", "unsloth", "unsloth_native"):
    if which == "unsloth_native":
        os.environ["UNSLOTH_MOE_BACKEND"] = "native_torch"
    captured = []
    try:
        import unsloth
        from unsloth import FastLanguageModel
        import torch
        m, tok = FastLanguageModel.from_pretrained(MODEL, max_seq_length = 256, load_in_4bit = False,
                                                   dtype = torch.bfloat16,
                                                   **({"attn_implementation": ATTN} if ATTN else {}))
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
    try:
        from unsloth_zoo.temporary_patches.moe_utils import select_moe_backend
        out["moe_backend"] = select_moe_backend()
    except Exception as e:
        out["moe_backend"] = err(e)
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
