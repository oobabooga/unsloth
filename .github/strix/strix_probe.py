"""One-shot probe for a unified-memory box (Strix Halo): does the guard behind
block_swap_layers actually fire there?

Answers three things, each independent of the next, so partial output is still useful:
  1. what torch reports for is_integrated on every device (no unsloth needed)
  2. what unsloth's is_integrated_unified_memory_gpu() says (needs the feat/block-swap branch)
  3. whether get_peft_model(block_swap_layers=8) refuses with the unified-memory reason
     (needs the branch and a ~1.2 GB bf16 model download; no bitsandbytes in the loop)
  4. --bnb: does the Windows bitsandbytes wheel run a 4-bit matmul on this arch at all

Run:  python strix_probe.py            # 1 and 2
      python strix_probe.py --e2e      # also 3
      python strix_probe.py --e2e --bnb
Prints one STRIX_RESULT json line at the end.
"""
import json
import sys
import traceback

res = {"ok": True}

# 1. torch's view, no unsloth.
try:
    import torch
    res["torch"] = torch.__version__
    res["hip"] = getattr(torch.version, "hip", None)
    res["cuda"] = getattr(torch.version, "cuda", None)
    devs = []
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        devs.append({
            "index": i, "name": p.name,
            "arch": getattr(p, "gcnArchName", None),
            "total_GiB": round(p.total_memory / 2**30, 1),
            "has_is_integrated_attr": hasattr(p, "is_integrated"),
            "is_integrated": getattr(p, "is_integrated", None),
        })
    res["devices"] = devs
except Exception as exc:
    res["ok"] = False
    res["torch_error"] = f"{type(exc).__name__}: {exc}"

# 2. unsloth's probe, the exact function install_block_swap calls.
try:
    from unsloth.models._uma_safetensors import is_integrated_unified_memory_gpu
    res["unsloth_says_unified"] = bool(is_integrated_unified_memory_gpu())
    from unsloth.models import _utils
    res["install_block_swap_present"] = hasattr(_utils, "install_block_swap")
    res["zoo_block_swap_present"] = getattr(_utils, "BlockSwap", None) is not None
except Exception as exc:
    res["unsloth_probe_error"] = f"{type(exc).__name__}: {exc}"

# 3. end to end: the refusal, or the install that should not have happened.
if "--e2e" in sys.argv:
    try:
        from unsloth import FastLanguageModel
        m, tok = FastLanguageModel.from_pretrained(
            "unsloth/Qwen3-0.6B", max_seq_length = 512,
            load_in_4bit = False, dtype = torch.bfloat16)
        try:
            pm = FastLanguageModel.get_peft_model(
                m, r = 8, target_modules = ["q_proj", "v_proj"],
                use_gradient_checkpointing = "unsloth", block_swap_layers = 8)
            res["e2e"] = "INSTALLED (guard did not fire)"
            # get_peft_model returns the PEFT wrapper; the swapper hangs off that, not off m.
            sw = getattr(pm, "_unsloth_block_swap", None)
            res["e2e_swapper_present"] = sw is not None
            if sw is not None:
                res["e2e_swapped_blocks"] = len(sw.blocks)
                res["e2e_resident"] = sw.resident_count()
                res["e2e_host_MiB"] = round(sw.host_bytes() / 2**20, 1)
        except ValueError as exc:
            res["e2e"] = "refused"
            res["e2e_message"] = str(exc)[:200]
            res["e2e_named_unified_memory"] = "unified-memory" in str(exc)
    except Exception as exc:
        res["e2e_error"] = f"{type(exc).__name__}: {exc}"
        res["e2e_tb"] = traceback.format_exc()[-600:]

# 4. bitsandbytes on this arch, independent of block swap. Extra data only.
if "--bnb" in sys.argv:
    try:
        import bitsandbytes as bnb
        res["bnb"] = bnb.__version__
        lin = bnb.nn.Linear4bit(256, 256, bias = False, compute_dtype = torch.bfloat16,
                                quant_type = "nf4").to("cuda")
        with torch.no_grad():
            y = lin(torch.randn(4, 256, device = "cuda", dtype = torch.bfloat16))
        res["bnb_4bit_matmul"] = "ok" if torch.isfinite(y).all().item() else "non-finite output"
    except Exception as exc:
        res["bnb_error"] = f"{type(exc).__name__}: {exc}"[:300]

print("STRIX_RESULT " + json.dumps(res))
