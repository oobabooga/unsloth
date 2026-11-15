#!/usr/bin/env python3
"""One tiny Unsloth training cell on gfx1151 (Linux or Windows). Observes only: JSON to --out, nothing parsed
from stdout. One cell per process (Unsloth patches leak across models); the JSON is rewritten at every stage, so a
native abort still leaves the last stage reached.

  python train5_cell.py --cell flm16 --out X.json [--steps 5]

Cells:
  env       versions, triton kernel, torch.compile, bitsandbytes NF4, `import unsloth`
  flm16     FastLanguageModel tiny Llama LoRA 16-bit, UnslothSFTTrainer, use_gradient_checkpointing="unsloth"
  flm4      the same, load_in_4bit (bitsandbytes NF4)
  vision    FastVisionModel tiny Qwen2.5-VL LoRA (vision + language layers), one image batch
  moe       FastLanguageModel tiny Qwen3-MoE LoRA, UNSLOTH_MOE_BACKEND=grouped_mm
  gemma4    FastModel tiny Gemma-4 LoRA (text batch); large-head attention patch decision
  qwen38    FastModel tiny Qwen4Exp (Qwen3.8-Flash-Next) LoRA, SFT with packing=True
  dsv4      FastLanguageModel tiny DeepSeek-V4 LoRA
  mllama    FastVisionModel tiny Mllama: patched vision encoder layer fwd + bwd on the GPU, then LoRA steps
  rl_decode generate -> optimizer step -> generate, decode CUDA graphs forced on (UNSLOTH_DECODE_CUDA_GRAPHS=1)
            vs eager after the step (graphs must see the updated weights)

Every training cell records: per-step losses, finite / decreasing, peak allocated memory (process-local, so a
co-tenant on the card does not move it), layer gradient-checkpointing calls, Dynamo counters, and profiler op
counts for one step (grouped_mm, checkpoint, flex / sdpa).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback

# flm16 / flm4 / rl_decode use a Llama built here (--model-dir, seeded, same bytes for every state): the trl tiny
# Llama 3.2 has head_dim 8 with hidden 8 / 4 heads, which FastLlama rebuilds as head_dim 2 (load MISMATCH).
LLAMA_TOKENIZER = "trl-internal-testing/tiny-MistralForCausalLM-0.1"
MODELS = {
    "flm16": "built:llama",
    "flm4": "built:llama",
    "rl_decode": "built:llama",
    "gcmem": "built:llama",
    "vision": "trl-internal-testing/tiny-Qwen2_5_VLForConditionalGeneration",
    "moe": "trl-internal-testing/tiny-Qwen3MoeForCausalLM",
    "gemma4": "trl-internal-testing/tiny-Gemma4ForConditionalGeneration",
    "qwen38": "hf-tiny-v2/tiny-random-Qwen4ExpForCausalLM",
    "dsv4": "tiny-random/deepseek-v4-bf16",
    "mllama": "hf-tiny-v2/tiny-random-MllamaForConditionalGeneration",
}
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
DOCS = ["the river runs north past the old mill and the green house by the water.",
        "music from the city carried over the small lake all through the long night."]
PROFILE_KEYS = ("grouped_mm", "checkpoint", "Checkpoint", "flex_attention", "scaled_dot_product", "mm")
COUNTER_MODULES = (
    "unsloth_zoo.temporary_patches.moe_grouped_modulelist", "unsloth_zoo.temporary_patches.moe_utils_fp8",
    "unsloth_zoo.temporary_patches.moe_grouped_fp16", "unsloth_zoo.gradient_checkpointing",
    "unsloth_zoo.temporary_patches.fla_vendor", "unsloth_zoo.temporary_patches.qwen4_exp",
    "unsloth_zoo.temporary_patches.deepseek_v41", "unsloth.models._decode_graphs",
    "unsloth.models._block_compile", "unsloth.utils.hf_packed_attention", "unsloth.kernels.large_head_attention",
)

ap = argparse.ArgumentParser()
ap.add_argument("--cell", required = True, choices = ["env", "build", *MODELS])
ap.add_argument("--out", required = True)
ap.add_argument("--steps", type = int, default = 5)
ap.add_argument("--lr", type = float, default = 5e-3)
ap.add_argument("--model-dir", default = None, help = "where the built tiny Llama lives (built once if absent)")
args = ap.parse_args()
res: dict = {"cell": args.cell, "stage": "start", "model": MODELS.get(args.cell)}


def built_llama():
    """Tiny Llama with consistent shapes (hidden 256, 4 heads x 64, 2 kv heads), seeded on CPU, saved bf16."""
    d = os.path.abspath(args.model_dir or os.path.join(os.path.dirname(os.path.abspath(args.out)), "tiny_llama"))
    if not os.path.isfile(os.path.join(d, "config.json")):
        import torch
        import transformers
        tok = transformers.AutoTokenizer.from_pretrained(LLAMA_TOKENIZER)
        torch.manual_seed(0)
        cfg = transformers.LlamaConfig(vocab_size = len(tok), hidden_size = 256, intermediate_size = 512,
                                       num_hidden_layers = 4, num_attention_heads = 4, num_key_value_heads = 2,
                                       head_dim = 64, max_position_embeddings = 2048, bos_token_id = tok.bos_token_id,
                                       eos_token_id = tok.eos_token_id, tie_word_embeddings = False)
        model = transformers.LlamaForCausalLM(cfg).to(torch.bfloat16)
        tmp = f"{d}.tmp{os.getpid()}"
        model.save_pretrained(tmp)
        tok.save_pretrained(tmp)
        try:
            os.rename(tmp, d)
        except OSError:
            pass  # another state built it first: same seed, same bytes
    res["model_dir"] = d
    return d


def dump():
    with open(args.out, "w", encoding = "utf-8") as fh:
        json.dump(res, fh, indent = 1, default = str)


def stage(name):
    res["stage"] = name
    dump()


def safe(fn):
    try:
        return fn()
    except Exception as e:  # noqa: BLE001
        return f"ERR {type(e).__name__}: {str(e)[:300]}"


def versions():
    import torch
    v = {"torch": torch.__version__, "hip": torch.version.hip, "cuda": torch.version.cuda,
         "python": sys.version.split()[0], "platform": sys.platform}
    for m in ("transformers", "trl", "peft", "accelerate", "bitsandbytes", "triton", "datasets"):
        try:
            v[m] = __import__(m).__version__
        except Exception as e:  # noqa: BLE001
            v[m] = f"ERR {type(e).__name__}: {str(e)[:160]}"
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        v.update(device = p.name, arch = getattr(p, "gcnArchName", None), total_gb = round(p.total_memory / 2**30, 1))
    for m in ("unsloth", "unsloth_zoo"):
        mod = sys.modules.get(m)
        if mod is not None:
            res[f"{m}_file"] = mod.__file__
            v[m] = getattr(mod, "__version__", None)
    res["versions"] = v


def module_counters():
    """Upper-case int / dict / bool module globals of the patch modules: engagement counters and states."""
    out = {}
    for name in COUNTER_MODULES:
        mod = sys.modules.get(name)
        if mod is None:
            continue
        rec = {}
        for k, val in vars(mod).items():
            if not k.isupper() or k.startswith("__"):
                continue
            if isinstance(val, bool) or isinstance(val, int):
                rec[k] = val
            elif isinstance(val, dict) and len(val) <= 40 and all(isinstance(x, (int, float, bool, str, type(None)))
                                                                   for x in val.values()):
                rec[k] = dict(val)
            elif isinstance(val, list) and len(val) <= 4 and all(isinstance(x, (int, bool, str, type(None))) for x in val):
                rec[k] = list(val)
        if rec:
            out[name.split(".")[-1]] = rec
    return out


def dynamo_counters():
    try:
        from torch._dynamo.utils import counters
        gb = counters.get("graph_break", {})
        return {"unique_graphs": dict(counters.get("stats", {})).get("unique_graphs", 0),
                "graph_breaks": int(sum(gb.values())),
                "graph_break_reasons": [str(k)[:160] for k, _ in sorted(gb.items(), key = lambda t: -t[1])[:5]],
                "recompiles": int(sum(counters.get("recompiles", {}).values())) if "recompiles" in counters else None}
    except Exception as e:  # noqa: BLE001
        return {"error": repr(e)[:200]}


class GCCounter:
    """Counts layer gradient-checkpointing calls: transformers' per-layer `_gradient_checkpointing_func` (the
    function Unsloth's smart / offloaded checkpointing is bound into)."""

    def __init__(self, model):
        self.calls = 0
        self.layers = 0
        self.install(model)

    def install(self, model):
        """Idempotent; re-run after anything that re-binds the function (Trainer.train's
        gradient_checkpointing_enable does)."""
        for m in model.modules():
            f = m.__dict__.get("_gradient_checkpointing_func")
            if f is None or getattr(f, "_amd_ci_counted", False):
                continue
            self.layers += 1

            def wrapped(*a, _f = f, **k):
                self.calls += 1
                return _f(*a, **k)
            wrapped._amd_ci_counted = True
            m._gradient_checkpointing_func = wrapped


def gc_callback(gc, model):
    from transformers import TrainerCallback

    class _ReinstallGC(TrainerCallback):
        def on_step_begin(self, *a, **k):
            gc.install(model)
    return _ReinstallGC()


def profile_counts(fn):
    """Run fn once under the CPU profiler; count op names by substring (works inside compiled graphs too)."""
    import torch
    from torch.profiler import ProfilerActivity, profile
    with profile(activities = [ProfilerActivity.CPU]) as prof:
        fn()
    counts = {k: 0 for k in PROFILE_KEYS}
    names = {}
    for e in prof.events():
        for k in PROFILE_KEYS:
            if k in e.name:
                counts[k] += 1
                if len(names) < 30:
                    names[e.name[:80]] = names.get(e.name[:80], 0) + 1
    torch.cuda.synchronize()
    return {"counts": counts, "names": names}


def finish_losses(losses):
    res["losses"] = losses
    res["finite"] = bool(losses) and all(math.isfinite(x) for x in losses)
    res["decreasing"] = bool(res["finite"] and len(losses) >= 2 and losses[-1] < losses[0])


def peak(tag = "peak_alloc_mb"):
    import torch
    torch.cuda.synchronize()
    res[tag] = round(torch.cuda.max_memory_allocated() / 2**20, 1)


def manual_loop(model, batch, steps, gc = None):
    """AdamW on one fixed batch: loss must fall. Step 1 is profiled."""
    import torch
    params = [p for p in model.parameters() if p.requires_grad]
    res["trainable_params"] = int(sum(p.numel() for p in params))
    opt = torch.optim.AdamW(params, lr = args.lr)
    model.train()
    losses = []

    def one():
        out = model(**batch)
        out.loss.backward()
        opt.step()
        opt.zero_grad(set_to_none = True)
        return float(out.loss.detach())
    for s in range(steps):
        if s == 1:
            box = {}
            res["profile_step1"] = profile_counts(lambda: box.__setitem__("l", one()))
            losses.append(box["l"])
        else:
            losses.append(one())
        if s == 0 and gc is not None:
            res["gc_calls_step0"] = gc.calls
        res["losses_so_far"] = losses
        dump()
    finish_losses(losses)


def text_batch(tok, device, seq = 64):
    import torch
    tt = getattr(tok, "tokenizer", tok)
    if tt.pad_token is None:
        tt.pad_token = tt.eos_token
    enc = tt(DOCS, return_tensors = "pt", padding = "max_length", truncation = True, max_length = seq)
    labels = enc["input_ids"].clone()
    labels[enc["attention_mask"] == 0] = -100
    return {"input_ids": enc["input_ids"].to(device), "attention_mask": enc["attention_mask"].to(device),
            "labels": labels.to(device)}


def lora_stats(model):
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    res["lora_tensors"] = len(names)
    res["lora_on_experts"] = sum(1 for n in names if "expert" in n)
    res["lora_on_vision"] = sum(1 for n in names if "visual" in n or "vision" in n)


def cell_env():
    import torch
    versions()
    res["cuda_available"] = torch.cuda.is_available()

    def triton_kernel():
        import triton
        import triton.language as tl

        @triton.jit
        def add(x_ptr, o_ptr, n, BLOCK: tl.constexpr):
            i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
            m = i < n
            tl.store(o_ptr + i, tl.load(x_ptr + i, mask = m) * 2, mask = m)
        x = torch.arange(1000, device = "cuda", dtype = torch.float32)
        o = torch.empty_like(x)
        add[(4,)](x, o, 1000, BLOCK = 256)
        torch.cuda.synchronize()
        return float((o - 2 * x).abs().max())
    res["triton_kernel"] = safe(triton_kernel)

    def compiled():
        f = torch.compile(lambda a: torch.nn.functional.gelu(a) * 2 + 1)
        a = torch.randn(256, device = "cuda")
        return float((f(a) - (torch.nn.functional.gelu(a) * 2 + 1)).abs().max())
    res["torch_compile"] = safe(compiled)

    def bnb():
        import bitsandbytes as b
        lin = b.nn.Linear4bit(64, 64, bias = False, compute_dtype = torch.bfloat16, quant_type = "nf4").to("cuda")
        y = lin(torch.randn(2, 64, device = "cuda", dtype = torch.bfloat16))
        torch.cuda.synchronize()
        return bool(torch.isfinite(y).all())
    res["bnb_nf4_finite"] = safe(bnb)
    t0 = time.time()
    import unsloth  # noqa: F401
    res["import_unsloth_s"] = round(time.time() - t0, 1)
    versions()
    try:
        from torch._inductor import config as ic
        res["inductor_use_block_ptr"] = ic.triton.use_block_ptr
    except Exception as e:  # noqa: BLE001
        res["inductor_use_block_ptr"] = repr(e)[:200]
    res["triton_has_block_ptr"] = safe(lambda: hasattr(__import__("triton.language", fromlist = ["x"]), "make_block_ptr"))
    try:
        from unsloth_zoo.temporary_patches.moe_utils import select_moe_backend
        res["moe_backend_default"] = select_moe_backend()
    except Exception as e:  # noqa: BLE001
        res["moe_backend_default"] = repr(e)[:200]
    res["counters"] = module_counters()


def cell_flm(four_bit):
    import torch
    from unsloth import FastLanguageModel
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer
    versions()
    stage("load")
    model, tok = FastLanguageModel.from_pretrained(built_llama(), max_seq_length = 256, dtype = torch.bfloat16,
                                                   load_in_4bit = four_bit)
    res["n_linear4bit"] = sum(1 for m in model.modules() if type(m).__name__ == "Linear4bit")
    model = FastLanguageModel.get_peft_model(model, r = 16, lora_alpha = 32, lora_dropout = 0.0, bias = "none",
                                             target_modules = TARGETS, use_gradient_checkpointing = "unsloth",
                                             random_state = 0)
    lora_stats(model)
    gc = GCCounter(model)
    res["gc_layers"] = gc.layers
    tt = getattr(tok, "tokenizer", tok)
    if tt.pad_token is None:
        tt.pad_token = tt.eos_token
    ds = Dataset.from_dict({"text": [d + (tt.eos_token or "") for d in DOCS] * 8})
    cfg = SFTConfig(output_dir = os.path.join(os.path.dirname(os.path.abspath(args.out)), f"trainer_{args.cell}"),
                    per_device_train_batch_size = 2, gradient_accumulation_steps = 1, max_steps = args.steps,
                    learning_rate = args.lr, logging_steps = 1, bf16 = True, report_to = "none", seed = 0,
                    dataset_text_field = "text", max_length = 64, save_strategy = "no", dataset_num_proc = 1,
                    warmup_steps = 0, lr_scheduler_type = "constant", packing = False)
    stage("train")
    torch.cuda.reset_peak_memory_stats()
    trainer = SFTTrainer(model = model, train_dataset = ds, processing_class = tt, args = cfg)
    trainer.add_callback(gc_callback(gc, model))
    res["trainer_class"] = type(trainer).__name__
    trainer.train()
    peak()
    res["gc_calls"] = gc.calls
    logs = [h for h in trainer.state.log_history if "loss" in h]
    res["grad_norms"] = [h.get("grad_norm") for h in logs]
    finish_losses([float(h["loss"]) for h in logs])
    # One extra profiled step through the same model (eager loop): op-level engagement.
    batch = text_batch(tok, model.get_input_embeddings().weight.device)
    model.train()
    res["profile_step"] = profile_counts(lambda: model(**batch).loss.backward())


def cell_vision():
    import torch
    from PIL import Image
    from unsloth import FastVisionModel
    versions()
    stage("load")
    model, proc = FastVisionModel.from_pretrained(MODELS["vision"], max_seq_length = 512, dtype = torch.bfloat16,
                                                  load_in_4bit = False)
    model = FastVisionModel.get_peft_model(model, finetune_vision_layers = True, finetune_language_layers = True,
                                           finetune_attention_modules = True, finetune_mlp_modules = True, r = 16,
                                           lora_alpha = 32, lora_dropout = 0.0, bias = "none", random_state = 0,
                                           use_gradient_checkpointing = "unsloth")
    lora_stats(model)
    gc = GCCounter(model)
    res["gc_layers"] = gc.layers
    FastVisionModel.for_training(model)
    img = Image.new("RGB", (56, 56))
    img.putdata([((x * 37) % 256, (y * 53) % 256, ((x + y) * 11) % 256) for y in range(56) for x in range(56)])
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe the picture."}]},
            {"role": "assistant", "content": [{"type": "text", "text": "A small square of coloured stripes."}]}]
    text = proc.apply_chat_template(msgs, tokenize = False)
    enc = proc(text = [text], images = [img], return_tensors = "pt")
    dev = model.get_input_embeddings().weight.device
    batch = {k: v.to(dev) for k, v in enc.items()}
    labels = batch["input_ids"].clone()
    labels[batch["attention_mask"] == 0] = -100
    batch["labels"] = labels
    res["n_tokens"] = int(labels.shape[1])
    stage("train")
    torch.cuda.reset_peak_memory_stats()
    manual_loop(model, batch, args.steps, gc)
    peak()
    res["gc_calls"] = gc.calls


def cell_causal(loader, path, targets = TARGETS, **load_kw):
    import torch
    versions()
    stage("load")
    model, tok = loader.from_pretrained(path, max_seq_length = 256, dtype = torch.bfloat16, load_in_4bit = False,
                                        **load_kw)
    res["model_class"] = type(model).__name__
    # FastModel: language layers only (explicit names would also hit multimodal towers, e.g. Gemma-4's
    # Gemma4ClippableLinear, which PEFT refuses).
    which = (dict(finetune_vision_layers = False, finetune_language_layers = True, finetune_attention_modules = True,
                  finetune_mlp_modules = True) if loader.__name__ == "FastModel" else dict(target_modules = targets))
    model = loader.get_peft_model(model, r = 16, lora_alpha = 32, lora_dropout = 0.0, bias = "none",
                                  use_gradient_checkpointing = "unsloth", random_state = 0, **which)
    lora_stats(model)
    gc = GCCounter(model)
    res["gc_layers"] = gc.layers
    batch = text_batch(tok, model.get_input_embeddings().weight.device)
    stage("train")
    torch.cuda.reset_peak_memory_stats()
    manual_loop(model, batch, args.steps, gc)
    peak()
    res["gc_calls"] = gc.calls
    return model, tok


def cell_moe():
    from unsloth import FastLanguageModel
    res["moe_backend_env"] = os.environ.get("UNSLOTH_MOE_BACKEND")
    cell_causal(FastLanguageModel, MODELS["moe"])
    from unsloth_zoo.temporary_patches.moe_utils import select_moe_backend
    res["moe_backend"] = select_moe_backend()


def cell_gemma4():
    from unsloth import FastModel
    model, _ = cell_causal(FastModel, MODELS["gemma4"])
    import unsloth.kernels.large_head_attention as lh
    res["large_head_patch_returns"] = safe(lambda: lh.patch_large_head_attention(model))
    res["large_head_patched"] = safe(lambda: lh._PATCHED[0])
    res["head_dims"] = sorted({int(getattr(m, "head_dim")) for m in model.modules()
                               if isinstance(getattr(m, "head_dim", None), int)})


def cell_qwen38():
    import torch
    from unsloth import FastModel
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer
    versions()
    stage("load")
    model, tok = FastModel.from_pretrained(MODELS["qwen38"], max_seq_length = 256, dtype = torch.bfloat16,
                                           load_in_4bit = False)
    res["model_class"] = type(model).__name__
    model = FastModel.get_peft_model(model, r = 16, lora_alpha = 32, lora_dropout = 0.0, bias = "none",
                                     finetune_vision_layers = False, finetune_language_layers = True,
                                     finetune_attention_modules = True, finetune_mlp_modules = True,
                                     use_gradient_checkpointing = "unsloth", random_state = 0)
    lora_stats(model)
    gc = GCCounter(model)
    tt = getattr(tok, "tokenizer", tok)
    if tt.pad_token is None:
        tt.pad_token = tt.eos_token
    ds = Dataset.from_dict({"text": [d + (tt.eos_token or "") for d in DOCS] * 8})
    cfg = SFTConfig(output_dir = os.path.join(os.path.dirname(os.path.abspath(args.out)), "trainer_qwen38"),
                    per_device_train_batch_size = 2, max_steps = args.steps, learning_rate = args.lr, logging_steps = 1,
                    bf16 = True, report_to = "none", seed = 0, dataset_text_field = "text", max_length = 128,
                    save_strategy = "no", dataset_num_proc = 1, warmup_steps = 0, lr_scheduler_type = "constant",
                    packing = True)
    stage("train")
    torch.cuda.reset_peak_memory_stats()
    trainer = SFTTrainer(model = model, train_dataset = ds, processing_class = tt, args = cfg)
    trainer.add_callback(gc_callback(gc, model))
    trainer.train()
    peak()
    res["gc_calls"] = gc.calls
    logs = [h for h in trainer.state.log_history if "loss" in h]
    finish_losses([float(h["loss"]) for h in logs])
    res["varlen_hits"] = {k: sum(1 for m in model.modules() if m.__dict__.get(k)) for k in (
        "_unsloth_varlen_conv_hit", "_unsloth_varlen_embedding_hit", "_unsloth_varlen_fused_hit",
        "_unsloth_varlen_kwargs_hit")}


def cell_dsv4():
    from unsloth import FastLanguageModel
    cell_causal(FastLanguageModel, MODELS["dsv4"])


def cell_mllama():
    import torch
    from unsloth import FastVisionModel
    versions()
    stage("load")
    model, proc = FastVisionModel.from_pretrained(MODELS["mllama"], max_seq_length = 256, dtype = torch.bfloat16,
                                                  load_in_4bit = False)
    res["model_class"] = type(model).__name__
    # zoo #2523: the patched vision encoder layer, called the way transformers 5.x calls it, on the GPU.
    stage("vision_layer")
    layer = None
    for name, m in model.named_modules():
        if type(m).__name__ == "MllamaVisionEncoderLayer":
            layer = m
            res["vision_layer"] = name
            break
    if layer is not None:
        hid = int(getattr(layer, "hidden_size", 0) or model.config.vision_config.hidden_size)
        x = torch.randn(1, 5, hid, device = model.device, dtype = torch.bfloat16, requires_grad = True)
        rec = {}
        for kw in ("hidden_states", "hidden_state"):
            try:
                y = layer(**{kw: x})
                y = y[0] if isinstance(y, (tuple, list)) else y
                y.float().pow(2).mean().backward()
                rec[kw] = {"ok": bool(torch.isfinite(y).all()), "grad_finite": bool(torch.isfinite(x.grad).all())}
                x.grad = None
            except Exception as e:  # noqa: BLE001
                rec[kw] = {"ok": False, "error": f"{type(e).__name__}: {str(e)[:300]}"}
        res["vision_layer_calls"] = rec
    model = FastVisionModel.get_peft_model(model, finetune_vision_layers = True, finetune_language_layers = True,
                                           r = 16, lora_alpha = 32, lora_dropout = 0.0, bias = "none", random_state = 0,
                                           use_gradient_checkpointing = "unsloth")
    lora_stats(model)
    gc = GCCounter(model)
    FastVisionModel.for_training(model)
    from PIL import Image
    img = Image.new("RGB", (64, 64), (200, 30, 30))
    tt = getattr(proc, "tokenizer", proc)
    text = "<|image|>Describe the picture. A red square."
    enc = proc(text = [text], images = [[img]], return_tensors = "pt")
    batch = {k: v.to(model.device) for k, v in enc.items()}
    labels = batch["input_ids"].clone()
    if tt.pad_token_id is not None:
        labels[labels == tt.pad_token_id] = -100
    batch["labels"] = labels
    stage("train")
    torch.cuda.reset_peak_memory_stats()
    manual_loop(model, batch, args.steps, gc)
    peak()
    res["gc_calls"] = gc.calls


def cell_rl_decode():
    import torch
    from unsloth import FastLanguageModel
    versions()
    stage("load")
    model, tok = FastLanguageModel.from_pretrained(built_llama(), max_seq_length = 256, dtype = torch.bfloat16,
                                                   load_in_4bit = False)
    model = FastLanguageModel.get_peft_model(model, r = 16, lora_alpha = 32, lora_dropout = 0.0, bias = "none",
                                             target_modules = TARGETS, use_gradient_checkpointing = "unsloth",
                                             random_state = 0)
    tt = getattr(tok, "tokenizer", tok)
    if tt.pad_token is None:
        tt.pad_token = tt.eos_token
    dev = model.get_input_embeddings().weight.device
    prompt = tt([DOCS[0][:30]], return_tensors = "pt").to(dev)

    def gen(graphs):
        os.environ["UNSLOTH_DECODE_CUDA_GRAPHS"] = "1" if graphs else "0"
        FastLanguageModel.for_inference(model)
        with torch.no_grad():
            out = model.generate(**prompt, max_new_tokens = 12, do_sample = False, use_cache = True,
                                 return_dict_in_generate = True, output_scores = True)
        scores = torch.stack([s[0].float() for s in out.scores])
        return out.sequences[0, prompt["input_ids"].shape[1]:].tolist(), scores
    stage("generate_before")
    before_ids, before_scores = gen(True)
    res["counters_after_first_generate"] = module_counters()
    stage("train_step")
    FastLanguageModel.for_training(model)
    batch = text_batch(tok, dev)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr = 5e-2)
    losses = []
    for _ in range(2):
        out = model(**batch)
        out.loss.backward()
        opt.step()
        opt.zero_grad(set_to_none = True)
        losses.append(float(out.loss.detach()))
    finish_losses(losses)
    stage("generate_after")
    g_ids, g_scores = gen(True)
    e_ids, e_scores = gen(False)
    res["counters_end"] = module_counters()
    res["ids_before"], res["ids_after_graphs"], res["ids_after_eager"] = before_ids, g_ids, e_ids
    res["weights_moved_output"] = bool(float((g_scores - before_scores).abs().max()) > 0)
    res["graphs_vs_eager_max_abs"] = float((g_scores - e_scores).abs().max())
    res["graphs_vs_eager_rel"] = float((g_scores - e_scores).norm() / e_scores.norm().clamp_min(1e-12))
    res["graphs_match_eager_after_step"] = bool(g_ids == e_ids and res["graphs_vs_eager_rel"] < 2e-2)
    peak()


def cell_gcmem():
    """Layer gradient checkpointing engaged? The same model, GC "unsloth" vs off, one fwd+bwd at 4 x 1024 tokens:
    peak allocated memory (process-local) must drop with GC on, and the layer checkpoint function must be called."""
    import gc as pygc
    import torch
    from unsloth import FastLanguageModel
    versions()
    out = {}
    for mode in ("unsloth", False):
        stage(f"gcmem_{mode}")
        model, tok = FastLanguageModel.from_pretrained(built_llama(), max_seq_length = 1024, dtype = torch.bfloat16,
                                                       load_in_4bit = False)
        model = FastLanguageModel.get_peft_model(model, r = 16, lora_alpha = 32, lora_dropout = 0.0, bias = "none",
                                                 target_modules = TARGETS, use_gradient_checkpointing = mode,
                                                 random_state = 0)
        counter = GCCounter(model)
        model.train()
        vocab = model.get_input_embeddings().weight.shape[0]
        ids = torch.randint(10, vocab, (4, 1024), generator = torch.Generator().manual_seed(0)).to(model.device)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base_alloc = torch.cuda.memory_allocated()
        loss = model(input_ids = ids, labels = ids).loss
        loss.backward()
        torch.cuda.synchronize()
        out[str(mode)] = {"peak_over_resident_mb": round((torch.cuda.max_memory_allocated() - base_alloc) / 2**20, 1),
                          "gc_calls": counter.calls, "gc_layers": counter.layers, "loss": float(loss.detach()),
                          "layer_flags": sum(1 for m in model.modules() if getattr(m, "gradient_checkpointing", False) is True)}
        del model, loss
        pygc.collect()
        torch.cuda.empty_cache()
    res["gcmem"] = out
    on, off = out["unsloth"]["peak_over_resident_mb"], out["False"]["peak_over_resident_mb"]
    res["gc_mem_ratio"] = round(on / off, 3) if off else None
    res["gc_engaged"] = bool(off and on / off < 0.85)
    res["gc_calls"] = out["unsloth"]["gc_calls"]
    finish_losses([out["unsloth"]["loss"]])
    res["finite"] = all(math.isfinite(v["loss"]) for v in out.values())
    res["decreasing"] = None


CELLS = {"gcmem": cell_gcmem, "env": cell_env, "flm16": lambda: cell_flm(False), "flm4": lambda: cell_flm(True), "vision": cell_vision,
         "moe": cell_moe, "gemma4": cell_gemma4, "qwen38": cell_qwen38, "dsv4": cell_dsv4, "mllama": cell_mllama,
         "rl_decode": cell_rl_decode}


def main():
    if args.cell == "moe":
        os.environ["UNSLOTH_MOE_BACKEND"] = "grouped_mm"
    t0 = time.time()
    if args.cell == "build":
        try:
            built_llama()  # before any `import unsloth`: plain transformers classes
            res["stage"], res["ok"] = "done", True
        except Exception as e:  # noqa: BLE001
            res["ok"], res["error"] = False, f"{type(e).__name__}: {e}"[:1500]
        dump()
        return 0
    try:
        if args.cell != "env":
            import unsloth  # noqa: F401  (before transformers / peft / trl)
        CELLS[args.cell]()
        res["dynamo"] = dynamo_counters()
        res["counters"] = res.get("counters") or module_counters()
        res["stage"] = "done"
        res["ok"] = True if args.cell in ("env",) else bool(res.get("finite"))
    except BaseException as e:  # noqa: BLE001
        res["ok"] = False
        res["error"] = f"{type(e).__name__}: {e}"[:1500]
        res["traceback"] = traceback.format_exc()[-4000:]
        try:
            res["dynamo"] = dynamo_counters()
        except Exception:  # noqa: BLE001
            pass
    res["seconds"] = round(time.time() - t0, 1)
    dump()
    return 0


if __name__ == "__main__":
    sys.exit(main())
