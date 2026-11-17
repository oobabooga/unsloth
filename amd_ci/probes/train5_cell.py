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
    "flm16_gcsave": "built:llama",
    "lpad": "built:llama",
    "fft16": "built:llama",
    "fft16_sroff": "built:llama",
    "flm4": "built:llama",
    "rl_decode": "built:llama",
    "gcmem": "built:llama",
    "ctnvfp4": "built:ct-nvfp4-qwen3moe",
    "ctmxfp8": "built:ct-mxfp8-qwen3moe",
    "mxfp4probe": None,
    "decode_diag": None,
    "gcnr_qwen3": "trl-internal-testing/tiny-Qwen3ForCausalLM",
    "gcdef_qwen3": "trl-internal-testing/tiny-Qwen3ForCausalLM",
    "gcnr_gemma3": "trl-internal-testing/tiny-Gemma3ForConditionalGeneration",
    "gcdef_gemma3": "trl-internal-testing/tiny-Gemma3ForConditionalGeneration",
    "lora_noac": "trl-internal-testing/tiny-Qwen3ForCausalLM",
    "st_minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "st_qwen3": "trl-internal-testing/tiny-Qwen3ForCausalLM",
    "hd512_llama": "built:llama-hd512",
    "hd512_gemma4": "built:gemma4-global-hd512",
    "hd512_gemma4_nc": "built:gemma4-global-hd512 (UNSLOTH_COMPILE_DISABLE=1)",
    "seqcls": "trl-internal-testing/tiny-Qwen3ForCausalLM",
    "fsdp2_io": "random tiny untied Qwen3 (vocab 1024, hidden 32, 3 layers), native FSDP2 world 1 on the GPU",
    "fsdp2_gen": "random tiny untied Qwen3, native FSDP2 world 1 on the GPU, generate all / block",
    "pauto": "built:llama, torchrun 2 ranks on one GPU, UNSLOTH_PARALLEL=auto",
    "tphost": "random tiny untied Qwen3 + PEFT LoRA, tensor parallel tp=1 on the GPU, UNSLOTH_TP_EMBED_ON_CPU 1 vs 0",
    "atree": "trl-internal-testing/tiny-Qwen3ForCausalLM, TRL prefix-trie packed row vs unpacked log-probs (fast backbone)",
    "qwen38_long": "hf-tiny-v2/tiny-random-Qwen4ExpForCausalLM (2052-token packed rows)",
    "dec_mistral": "trl-internal-testing/tiny-MistralForCausalLM-0.1",
    "sft_eos": "trl-internal-testing/tiny-Qwen3ForCausalLM",
    "aol_qwen25vl": "trl-internal-testing/tiny-Qwen2_5_VLForConditionalGeneration",
    "aol_gemma3_unsloth": "trl-internal-testing/tiny-Gemma3ForConditionalGeneration",
    "aol_gemma3_trl": "trl-internal-testing/tiny-Gemma3ForConditionalGeneration",
    "aol_qwen25vl_none": "trl-internal-testing/tiny-Qwen2_5_VLForConditionalGeneration (no collator passed)",
    "aol_gemma3_none": "trl-internal-testing/tiny-Gemma3ForConditionalGeneration (no collator passed)",
    "cp_ring_kernels": "(no model: unsloth.context_parallel._ring_sdpa_backends on the GPU)",
    "gkd_proc_qwen35": "trl-internal-testing/tiny-Qwen3_5ForConditionalGeneration-NoThink",
    "gkd_proc_gemma3": "trl-internal-testing/tiny-Gemma3ForConditionalGeneration",
    "mrope_pack": "trl-internal-testing/tiny-Qwen2_5_VLForConditionalGeneration",
    "orpo_txt": "trl-internal-testing/tiny-Qwen3ForCausalLM",
    "orpo_txt_psoff": "trl-internal-testing/tiny-Qwen3ForCausalLM",
    "orpo_txt_unwrap": "trl-internal-testing/tiny-Qwen3ForCausalLM",
    "vision_gemma3": "trl-internal-testing/tiny-Gemma3ForConditionalGeneration",
    "dpo_txt": "trl-internal-testing/tiny-Qwen3ForCausalLM",
    "pfree_llama": "built:llama (FastLanguageModel, SFT padding_free=True)",
    "pfree_qwen3": "trl-internal-testing/tiny-Qwen3ForCausalLM",
    "sftdef_steps": "built:llama (SFT, packing unset, max_steps)",
    "sftdef_epoch": "built:llama (SFT, packing unset, num_train_epochs=1)",
    "padmix_qwen3": "trl-internal-testing/tiny-Qwen3ForCausalLM",
    "padmix_gemma4": "trl-internal-testing/tiny-Gemma4ForConditionalGeneration",
    "padmix_gemma3": "trl-internal-testing/tiny-Gemma3ForConditionalGeneration",
    "padmix_gemma3_flex_ones": "trl-internal-testing/tiny-Gemma3ForConditionalGeneration (flex, U = all-ones mask)",
    "padmix_gemma4_ones": "trl-internal-testing/tiny-Gemma4ForConditionalGeneration (U = all-ones mask)",
    "padmix_gemma3_flex": "trl-internal-testing/tiny-Gemma3ForConditionalGeneration (attn_implementation=flex_attention)",
    "grpo_txt": "trl-internal-testing/tiny-Qwen3ForCausalLM",
    "grpo_zadv": "trl-internal-testing/tiny-Qwen3ForCausalLM",
    "grpo_zadv_zoff": "trl-internal-testing/tiny-Qwen3ForCausalLM",
    "grpo_pack": "trl-internal-testing/tiny-Qwen3ForCausalLM (GRPO, mixed-length prompts, packing verify state)",
    "warm": "built:llama, tiny compiled LoRA job twice on one TORCHINDUCTOR_CACHE_DIR",
    "fp32attn": "trl-internal-testing/tiny-Qwen3ForCausalLM in float32, no autocast, vs HF fp32 eager",
    "grpo_pg": "trl-internal-testing/tiny-Qwen3ForCausalLM (long shared prompt, 4 generations)",
    "plelens": "hf-tiny-v2/tiny-random-Qwen4ExpForCausalLM",
    "granite_gc": "built:granitemoehybrid (attention layers, from tests/test_granitemoehybrid_float16.py, widened)",
    "moeaux_nr": "trl-internal-testing/tiny-Qwen3MoeForCausalLM",
    "moeaux_gc_qwen3moe": "trl-internal-testing/tiny-Qwen3MoeForCausalLM",
    "moeaux_gc_gptoss": "trl-internal-testing/tiny-GptOssForCausalLM",
    "gkd": "trl-internal-testing/tiny-Qwen3ForCausalLM",
    "dec_whisper": "hf-internal-testing/tiny-random-WhisperForConditionalGeneration",
    "dec_gemma3n": "built:gemma3n-shared-kv (config from tests/test_generic_decode_graphs_shared_kv.py)",
    "bclens": "trl-internal-testing/tiny-Qwen3ForCausalLM",
    "bswap": "trl-internal-testing/tiny-Qwen3ForCausalLM",
    "tok_save": "built:llama",
    "grpo_vl": "trl-internal-testing/tiny-Qwen2_5_VLForConditionalGeneration",
    # The PR test's model: on TRL 0.24, Qwen2.5-VL DPO drops image_grid_thw (fails at any state).
    "dpo_vl": "trl-internal-testing/tiny-LlavaForConditionalGeneration",
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
ap.add_argument("--cell", required = True, choices = ["env", "build", "build_ct", "build_hd512", *MODELS])
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


def _jsonable(x):
    """Stringify non-JSON dict keys (counters keyed by tuples) so one odd counter cannot lose the cell's record."""
    if isinstance(x, dict):
        return {(k if isinstance(k, (str, int, float, bool)) or k is None else str(k)): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x


def dump():
    with open(args.out, "w", encoding = "utf-8") as fh:
        json.dump(_jsonable(res), fh, indent = 1, default = str)


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
    try:  # zoo #2571 / unsloth #18113: UNSLOTH_GC_SAVE_ATTENTION engagement (absent at a base without the module)
        import importlib
        sa = importlib.import_module("unsloth_zoo.saved_attention")
        res["saved_attention_stats"] = dict(getattr(sa, "SAVED_ATTENTION_STATS", {}) or {})
    except Exception as e:  # noqa: BLE001
        res["saved_attention_stats"] = f"absent ({type(e).__name__})"
    res["gc_save_attention_env"] = os.environ.get("UNSLOTH_GC_SAVE_ATTENTION")


def cell_vision():
    import torch
    from PIL import Image
    from unsloth import FastVisionModel
    versions()
    stage("load")
    model, proc = FastVisionModel.from_pretrained(MODELS[args.cell], max_seq_length = 512, dtype = torch.bfloat16,
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


def cell_causal(loader, path, targets = TARGETS, language_only = None, **load_kw):
    import torch
    versions()
    stage("load")
    model, tok = loader.from_pretrained(path, max_seq_length = 256, dtype = torch.bfloat16, load_in_4bit = False,
                                        **load_kw)
    res["model_class"] = type(model).__name__
    # FastModel: language layers only (explicit names would also hit multimodal towers, e.g. Gemma-4's
    # Gemma4ClippableLinear, which PEFT refuses).
    which = (dict(finetune_vision_layers = False, finetune_language_layers = True, finetune_attention_modules = True,
                  finetune_mlp_modules = True) if (language_only if language_only is not None
                                                   else loader.__name__ == "FastModel") else dict(target_modules = targets))
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

CT_E, CT_H, CT_I = 4, 64, 32


def ct_dir(fmt):
    return os.path.abspath(os.path.join(os.path.dirname(args.model_dir or args.out), f"ct_{fmt}_qwen3moe"))


def build_ct():
    """Tiny Qwen3-MoE with per-expert compressed-tensors NVFP4 / MXFP8 experts (the layout of #17848's test), quantized
    with unsloth_zoo.packed_experts from the PAIRED zoo head (the probe puts it on PYTHONPATH), so every state loads the
    same bytes. Needs compressed_tensors."""
    import torch
    import transformers
    from compressed_tensors.quantization import QuantizationConfig, preset_name_to_scheme
    from safetensors.torch import save_file
    import unsloth_zoo.packed_experts as pe
    res["builder_zoo"] = pe.__file__
    for fmt, (preset, ct_format) in {"nvfp4": ("NVFP4A16", "nvfp4-pack-quantized"),
                                     "mxfp8": ("MXFP8A16", "mxfp8-quantized")}.items():
        d = ct_dir(fmt)
        if os.path.isfile(os.path.join(d, "config.json")):
            continue
        tmp = f"{d}.tmp{os.getpid()}"
        os.makedirs(tmp, exist_ok = True)
        cfg = transformers.Qwen3MoeConfig(vocab_size = 32000, hidden_size = CT_H, intermediate_size = 128,
                                          moe_intermediate_size = CT_I, num_experts = CT_E, num_experts_per_tok = 2,
                                          num_hidden_layers = 2, num_attention_heads = 2, num_key_value_heads = 1,
                                          head_dim = 32, decoder_sparse_step = 1, mlp_only_layers = [],
                                          tie_word_embeddings = False)
        torch.manual_seed(0)
        model = transformers.Qwen3MoeForCausalLM(cfg).to(torch.bfloat16)
        state = {k: v.contiguous() for k, v in model.state_dict().items()
                 if ".mlp.experts.gate_up_proj" not in k and ".mlp.experts.down_proj" not in k}
        for layer in range(2):
            ex = model.model.layers[layer].mlp.experts
            for e in range(CT_E):
                for proj, w in (("gate_proj", ex.gate_up_proj.data[e, :CT_I]), ("up_proj", ex.gate_up_proj.data[e, CT_I:]),
                                ("down_proj", ex.down_proj.data[e])):
                    q = (pe.quantize_experts_nvfp4 if fmt == "nvfp4" else pe.quantize_experts_mxfp8)(w[None].float())
                    if fmt == "nvfp4":
                        packed = {"weight_packed": q.data[0], "weight_scale": q.packed_scales[0].view(torch.float8_e4m3fn),
                                  "weight_global_scale": q.packed_global[0].reshape(1).clone()}
                    else:
                        packed = {"weight": q.data[0].view(torch.float8_e4m3fn), "weight_scale": q.packed_scales[0]}
                    for suffix, t in packed.items():
                        state[f"model.layers.{layer}.mlp.experts.{e}.{proj}.{suffix}"] = t.contiguous()
        save_file(state, os.path.join(tmp, "model.safetensors"), metadata = {"format": "pt"})
        config = model.config.to_dict()
        scheme = preset_name_to_scheme(preset, [r"re:.*mlp\.experts\.\d+\.(gate|up|down)_proj$"])
        quant = QuantizationConfig(config_groups = {"group_0": scheme}, format = ct_format,
                                   quantization_status = "compressed", ignore = ["lm_head"])
        q = json.loads(json.dumps(quant.model_dump(exclude_none = False), default = str))
        q["quant_method"] = "compressed-tensors"
        config["quantization_config"] = q
        with open(os.path.join(tmp, "config.json"), "w", encoding = "utf-8") as fh:
            json.dump(config, fh)
        transformers.AutoTokenizer.from_pretrained(LLAMA_TOKENIZER).save_pretrained(tmp)
        try:
            os.rename(tmp, d)
        except OSError:
            pass
    res["ct_dirs"] = [ct_dir("nvfp4"), ct_dir("mxfp8")]


def cell_ct(fmt):
    """compressed-tensors NVFP4 / MXFP8 MoE through FastLanguageModel, LoRA on attention + experts, 5 steps.
    Engagement: packed expert parameters installed and the packed op called (zoo PACKED_EXPERT_STATS)."""
    import torch
    from unsloth import FastLanguageModel
    versions()
    d = ct_dir(fmt)
    if not os.path.isfile(os.path.join(d, "config.json")):
        raise RuntimeError(f"tiny {fmt} checkpoint missing ({d}): build_ct failed")
    stage("load")
    model, tok = FastLanguageModel.from_pretrained(d, max_seq_length = 256, dtype = torch.bfloat16, load_in_4bit = False)
    res["expert_param_types"] = sorted({type(p).__name__ for n, p in model.named_parameters() if ".experts." in n})
    res["expert_param_dtypes"] = sorted({str(p.dtype) for n, p in model.named_parameters() if ".experts." in n})
    model = FastLanguageModel.get_peft_model(model, r = 8, lora_alpha = 16, lora_dropout = 0.0, bias = "none",
                                             target_modules = TARGETS, use_gradient_checkpointing = "unsloth",
                                             random_state = 0)
    lora_stats(model)
    gc = GCCounter(model)
    try:
        import unsloth_zoo.packed_experts as pe
        before = dict(pe.PACKED_EXPERT_STATS)
    except Exception as e:  # noqa: BLE001
        pe, before = None, {}
        res["packed_experts_module"] = f"absent: {type(e).__name__}"
    batch = text_batch(tok, model.get_input_embeddings().weight.device)
    stage("train")
    torch.cuda.reset_peak_memory_stats()
    manual_loop(model, batch, args.steps, gc)
    peak()
    res["gc_calls"] = gc.calls
    if pe is not None:
        after = dict(pe.PACKED_EXPERT_STATS)
        res["packed_expert_stats_delta"] = {k: after.get(k, 0) - before.get(k, 0) for k in after
                                            if isinstance(after.get(k), (int, float))}


def cell_mxfp4probe():
    """zoo #2528: does the fused MXFP4 grouped GEMM still probe / engage here? Availability flags, then a direct
    fwd + bwd through mxfp4_expert_grouped_mm against dequantize + matmul."""
    import torch
    import triton
    versions()
    import unsloth_zoo.mxfp4_gemm as mg
    from unsloth_zoo.mxfp4_dequant import mxfp4_dequantize_torch
    res["triton_version"] = triton.__version__
    res["triton_too_old_gate"] = safe(lambda: mg._triton_too_old()) if hasattr(mg, "_triton_too_old") else "<absent>"
    res["grouped_available"] = safe(lambda: mg.mxfp4_grouped_mm_available())
    res["gemm_available"] = safe(lambda: mg.mxfp4_gemm_available())
    dev = torch.device("cuda", 0)
    gen = torch.Generator(device = dev).manual_seed(0)
    E, R, G = 4, 128, 4
    blocks = torch.randint(0, 256, (E, R, G, 16), dtype = torch.uint8, device = dev, generator = gen)
    scales = torch.randint(118, 130, (E, R, G), dtype = torch.uint8, device = dev, generator = gen)
    counts = torch.tensor([7, 0, 20, 5], dtype = torch.int32, device = dev)
    x = torch.randn(32, G * 32, dtype = torch.bfloat16, device = dev, generator = gen).requires_grad_()
    dense = mxfp4_dequantize_torch(blocks, scales, dtype = torch.bfloat16).float()
    rows = torch.repeat_interleave(torch.arange(E, device = dev), counts.long())
    y = mg.mxfp4_expert_grouped_mm(x, blocks, scales, counts, transpose_b = True)
    gy = torch.randn_like(y)
    (y.float() * gy.float()).sum().backward()
    xr = x.detach().float().requires_grad_()
    yr = torch.bmm(xr.unsqueeze(1), dense[rows].transpose(1, 2)).squeeze(1)
    (yr * gy.float()).sum().backward()
    res["fwd_rel"] = float((y.float() - yr).norm() / yr.norm())
    res["bwd_rel"] = float((x.grad.float() - xr.grad).norm() / xr.grad.norm())
    res["finite"] = bool(torch.isfinite(y).all() and torch.isfinite(x.grad).all())
    res["engaged"] = bool(res["grouped_available"] is True and res["fwd_rel"] < 2e-2 and res["bwd_rel"] < 2e-2)
    res["losses"], res["decreasing"] = [res["fwd_rel"]], None


def _vl_dataset(kind):
    from datasets import Dataset
    from PIL import Image, ImageDraw
    rows = []
    for i in range(4):
        img = Image.new("RGB", (32, 32), (255, 255, 255))
        ImageDraw.Draw(img).rectangle([8, 8, 24, 24], fill = [(220, 40, 40), (40, 80, 220)][i % 2])
        prompt = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": f"Color {i}?"}]}]
        if kind == "dpo":
            rows.append({"images": [img], "prompt": prompt,
                         "chosen": [{"role": "assistant", "content": [{"type": "text", "text": "It is red, a warm color."}]}],
                         "rejected": [{"role": "assistant", "content": [{"type": "text", "text": "Blue" * (1 + i)}]}]})
        else:
            rows.append({"image": img, "prompt": prompt})
    return Dataset.from_list(rows)


def _cfg(cls, **kw):
    import inspect
    names = set(inspect.signature(cls.__init__).parameters)
    res["config_dropped"] = sorted(k for k in kw if k not in names)
    return cls(**{k: v for k, v in kw.items() if k in names})


def _vl_model(repo):
    import torch
    from unsloth import FastVisionModel
    model, proc = FastVisionModel.from_pretrained(repo, max_seq_length = 512, dtype = torch.bfloat16,
                                                  load_in_4bit = False)
    model = FastVisionModel.get_peft_model(model, finetune_vision_layers = True, finetune_language_layers = True,
                                           r = 8, lora_alpha = 16, lora_dropout = 0.0, bias = "none", random_state = 0,
                                           use_gradient_checkpointing = "unsloth")
    FastVisionModel.for_training(model)
    return model, proc


def cell_grpo_vl():
    """zoo #2530: a vision GRPO step through Unsloth's GRPO trainer; counts fused_linear_grpo_loss forward-gradient
    calls (UNSLOTH_GRPO_FUSED_WITHOUT_ENTROPY=1 so a frozen head without an entropy bonus may take the route)."""
    import torch
    from trl import GRPOConfig, GRPOTrainer
    versions()
    stage("load")
    model, proc = _vl_model(MODELS[args.cell])
    import unsloth_zoo.fused_losses.fused_linear as fl
    before = dict(fl.CALLS)

    def reward(completions, **kw):
        return [float(len(str(c))) / 10.0 for c in completions]
    cfg = _cfg(GRPOConfig, output_dir = os.path.join(os.path.dirname(os.path.abspath(args.out)), "trainer_grpo_vl"),
               per_device_train_batch_size = 2, num_generations = 2, max_completion_length = 8, max_prompt_length = None,
               max_steps = 2, learning_rate = 5e-5, logging_steps = 1, report_to = "none", bf16 = True, seed = 0,
               save_strategy = "no", temperature = 1.0, use_vllm = False, remove_unused_columns = False)
    stage("train")
    torch.cuda.reset_peak_memory_stats()
    trainer = GRPOTrainer(model = model, reward_funcs = [reward], args = cfg, train_dataset = _vl_dataset("grpo"),
                          processing_class = proc)
    res["trainer_class"] = type(trainer).__name__
    trainer.train()
    peak()
    after = dict(fl.CALLS)
    res["fused_calls_delta"] = {k: after.get(k, 0) - before.get(k, 0) for k in after if after.get(k, 0) != before.get(k, 0)}
    res["grpo_fused_engaged"] = res["fused_calls_delta"].get("grpo_fwd_grad", 0) > 0
    logs = [h for h in trainer.state.log_history if "loss" in h]
    finish_losses([float(h["loss"]) for h in logs])
    res["decreasing"] = None  # 2 RL steps: finite and completed is the bar, not a falling loss
    res["rewards"] = [h.get("reward") for h in logs]


def cell_dpo_vl():
    """unsloth #17850: a vision DPO run; counts the chunked DPO head (trainer._unsloth_dpo_chunked_calls). The chunked
    head rewrites concatenated_forward, which TRL >= 0.29 no longer has: there it cannot engage."""
    import torch
    import trl
    from trl import DPOConfig, DPOTrainer
    versions()
    res["trl_has_concatenated_forward"] = hasattr(DPOTrainer, "concatenated_forward")
    stage("load")
    model, proc = _vl_model(MODELS[args.cell])
    cfg = _cfg(DPOConfig, output_dir = os.path.join(os.path.dirname(os.path.abspath(args.out)), "trainer_dpo_vl"),
               per_device_train_batch_size = 2, max_steps = args.steps, learning_rate = 5e-4, logging_steps = 1,
               report_to = "none", bf16 = True, seed = 0, save_strategy = "no", max_length = None,
               max_prompt_length = None, remove_unused_columns = False, dataset_num_proc = None,
               lr_scheduler_type = "constant", warmup_steps = 0)
    stage("train")
    torch.cuda.reset_peak_memory_stats()
    trainer = DPOTrainer(model = model, ref_model = None, args = cfg, train_dataset = _vl_dataset("dpo"),
                         processing_class = proc)
    res["trainer_class"] = type(trainer).__name__
    trainer.train()
    peak()
    res["dpo_chunked_calls"] = getattr(trainer, "_unsloth_dpo_chunked_calls", None)
    res["dpo_chunked_fallbacks"] = str(trainer.__dict__.get("_unsloth_dpo_chunked_fallbacks"))[:400]
    res["dpo_chunked_engaged"] = bool(res["dpo_chunked_calls"])
    logs = [h for h in trainer.state.log_history if "loss" in h]
    finish_losses([float(h["loss"]) for h in logs])


def cell_decode_diag():
    """zoo #2529 fp16 byte decoders on this GPU: Triton fp16 output vs the CPU fp32->fp16 reference, mismatches
    classified; torch's own GPU cast of the Triton fp32 output; a candidate exact fp16 store (subnormals by hand)."""
    import torch
    import triton
    import triton.language as tl
    versions()
    import unsloth_zoo.packed_experts as pe
    res["zoo"] = pe.__file__
    E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]

    def rand_nvfp4(E, N, K, parts, seed = 0):
        g = torch.Generator().manual_seed(seed)
        codes = torch.randint(0, 256, (E, N, K // 2), dtype = torch.uint8, generator = g)
        sb = torch.randint(0, 0x7E, (E, N, K // 16), dtype = torch.uint8, generator = g)
        sb |= (torch.rand((E, N, K // 16), generator = g) < 0.1).to(torch.uint8) << 7
        return codes, sb.view(torch.float8_e4m3fn), torch.rand((E, parts), generator = g) * 3000 + 1.0

    def ref_nvfp4_f32(codes, scales, glob, multiply):
        E, N, half = codes.shape
        lut = torch.tensor(E2M1 + [-v for v in E2M1])
        vals = torch.empty(E, N, half * 2)
        vals[..., 0::2] = lut[(codes & 15).long()]
        vals[..., 1::2] = lut[(codes >> 4).long()]
        s_ = scales.to(torch.float32).repeat_interleave(16, dim = -1)
        parts = glob.reshape(E, -1).shape[1]
        gg = glob.reshape(E, parts).repeat_interleave(N // parts, dim = 1)[:, :, None]
        return vals * (s_ * gg if multiply else s_ / gg)

    def classify(out16, ref16, ref32):
        o, r = out16.view(torch.int16).to(torch.int32), ref16.view(torch.int16).to(torch.int32)
        bad = o != r
        tiny = 2.0 ** -14
        sub = (ref16.float().abs() < tiny) & (ref16.float() != 0)
        rec = {"n": int(bad.numel()), "mismatch": int(bad.sum()), "ref_subnormal": int(sub.sum()),
               "flushed_subnormal": int((bad & sub & (out16.float() == 0)).sum()),
               "subnormal_other": int((bad & sub & (out16.float() != 0)).sum()),
               "ref_inf_out_finite": int((bad & torch.isinf(ref16) & torch.isfinite(out16)).sum()),
               "out_inf_ref_finite": int((bad & torch.isinf(out16) & torch.isfinite(ref16)).sum())}
        normal = bad & ~sub & torch.isfinite(ref16) & torch.isfinite(out16) & (ref16.float().abs() >= tiny)
        rec["signed_zero_only"] = int((bad & (ref16 == 0) & (out16 == 0)).sum())
        nz = ref16 != 0
        rec["equal_up_to_zero_sign"] = bool(torch.equal(out16 == 0, ~nz) and torch.equal(o[nz], r[nz]))
        rec["normal_1ulp"] = int((normal & ((o - r).abs() == 1)).sum())
        rec["normal_more"] = int((normal & ((o - r).abs() > 1)).sum())
        if rec["normal_more"]:
            i = torch.nonzero(normal & ((o - r).abs() > 1))[0].tolist()
            rec["example_more"] = {"ref32": float(ref32[tuple(i)]), "ref16": float(ref16[tuple(i)]), "out16": float(out16[tuple(i)])}
        if rec["flushed_subnormal"]:
            rec["max_abs_flushed"] = float(ref16.float()[bad & sub].abs().max())
        return rec

    @triton.jit
    def _cast_kernel(x_ptr, d_ptr, e_ptr, n, BLOCK: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = i < n
        x = tl.load(x_ptr + i, mask = m, other = 0.0)
        tl.store(d_ptr + i, x.to(tl.float16), mask = m)
        # Candidate: |x| < 2^-14 -> fp16 subnormal bits round-to-nearest-even(|x| * 2^24), exact in fp32.
        a = tl.abs(x)
        q = a * 16777216.0
        r = tl.floor(q)
        frac = q - r
        r = tl.where((frac > 0.5) | ((frac == 0.5) & ((r.to(tl.int32) & 1) == 1)), r + 1.0, r)
        sign = (x.to(tl.int32, bitcast = True) >> 16) & 0x8000
        subbits = (sign | r.to(tl.int32)).to(tl.int16)
        sub16 = subbits.to(tl.float16, bitcast = True)
        tl.store(e_ptr + i, tl.where(a < 6.103515625e-05, sub16, x.to(tl.float16)), mask = m)

    out = {}
    for parts in (1, 2):
        for multiply in (False, True):
            codes, scales, glob = rand_nvfp4(3, 48, 64, parts)
            if multiply:
                glob = 1.0 / glob
            ref32 = ref_nvfp4_f32(codes, scales, glob, multiply)
            ref16 = ref32.to(torch.float16)
            cg, sg, gg = codes.cuda(), scales.cuda(), glob.cuda()
            t16 = pe.nvfp4_experts_dequantize(cg, sg, gg, torch.float16, multiply = multiply).cpu()
            t32 = pe.nvfp4_experts_dequantize(cg, sg, gg, torch.float32, multiply = multiply)
            rec = {"triton_fp16": classify(t16, ref16, ref32),
                   "triton_fp32_bitwise": bool(torch.equal(t32.cpu().view(torch.int32), ref32.view(torch.int32))),
                   "torch_gpu_cast": classify(t32.to(torch.float16).cpu(), ref16, ref32)}
            flat = t32.reshape(-1).contiguous()
            d16, e16 = torch.empty(flat.shape, dtype = torch.float16, device = "cuda"), torch.empty(flat.shape, dtype = torch.float16, device = "cuda")
            _cast_kernel[(triton.cdiv(flat.numel(), 1024),)](flat, d16, e16, flat.numel(), BLOCK = 1024)
            rec["triton_plain_cast"] = classify(d16.cpu().view_as(ref16), ref16, ref32)
            rec["triton_exact_cast"] = classify(e16.cpu().view_as(ref16), ref16, ref32)
            out[f"nvfp4_parts{parts}_mul{int(multiply)}"] = rec
    g = torch.Generator().manual_seed(0)
    codes = torch.randint(0, 256, (3, 40, 96), dtype = torch.uint8, generator = g)
    codes[(codes & 0x7F) == 0x7F] = 0
    scales = torch.randint(110, 135, (3, 40, 3), dtype = torch.uint8, generator = g)
    q = codes.view(torch.float8_e4m3fn).to(torch.float32)
    ref32 = q * (scales.to(torch.float32) - 127.0).exp2().repeat_interleave(32, dim = -1)
    ref16 = ref32.to(torch.float16)
    t16 = pe.mxfp8_experts_dequantize(codes.cuda(), scales.cuda(), torch.float16).cpu()
    t32 = pe.mxfp8_experts_dequantize(codes.cuda(), scales.cuda(), torch.float32)
    out["mxfp8"] = {"triton_fp16": classify(t16, ref16, ref32),
                    "triton_fp32_bitwise": bool(torch.equal(t32.cpu().view(torch.int32), ref32.view(torch.int32))),
                    "torch_gpu_cast": classify(t32.to(torch.float16).cpu(), ref16, ref32)}
    # Mutation proof on this GPU's real fp16 output: the fp16 check (bitwise on non-zeros, zero where the reference is
    # zero) must reject a 1-ulp nudge, a sign flip and a nibble-swapped decode.
    def same(o16, w16):
        nz = w16 != 0
        return bool(torch.equal(o16 == 0, ~nz) and torch.equal(o16[nz].view(torch.int16), w16[nz].view(torch.int16)))
    codes, scales, glob = rand_nvfp4(3, 48, 64, 1)
    ref16 = ref_nvfp4_f32(codes, scales, glob, False).to(torch.float16)
    real = pe.nvfp4_experts_dequantize(codes.cuda(), scales.cuda(), glob.cuda(), torch.float16).cpu()
    nudged = real.clone().view(torch.int16)
    idx = torch.nonzero(ref16.reshape(-1) != 0)[7].item()
    nudged.view(-1)[idx] += 1
    swapped = pe.nvfp4_experts_dequantize(((codes << 4) | (codes >> 4)).cuda(), scales.cuda(), glob.cuda(), torch.float16).cpu()
    res["mutation"] = {"accepts_real_output": same(real, ref16), "rejects_1ulp": not same(nudged.view(torch.float16), ref16),
                       "rejects_sign_flip": not same(-real, ref16), "rejects_nibble_swap": not same(swapped, ref16)}
    res["mutation_ok"] = all(res["mutation"].values())
    res["diag"] = out
    res["finite"], res["losses"], res["decreasing"] = True, [], None


def _block_modes(model):
    modes = {}
    for m in model.modules():
        v = m.__dict__.get("_unsloth_block_mode")
        if v is not None:
            modes[str(v)] = modes.get(str(v), 0) + 1
    return modes


def cell_gc(nonreentrant):
    """unsloth #17857: LoRA with gradient checkpointing, non-reentrant (use_reentrant = False) or Unsloth's default;
    block compile modes per decoder layer and every "block compile failed" message, 5 steps."""
    import contextlib
    import io
    import torch
    from unsloth import FastModel
    versions()
    stage("load")
    model, tok = FastModel.from_pretrained(MODELS[args.cell], max_seq_length = 256, dtype = torch.bfloat16,
                                           load_in_4bit = False)
    res["model_class"] = type(model).__name__
    model = FastModel.get_peft_model(model, r = 16, lora_alpha = 32, lora_dropout = 0.0, bias = "none",
                                     finetune_vision_layers = False, finetune_language_layers = True,
                                     finetune_attention_modules = True, finetune_mlp_modules = True,
                                     use_gradient_checkpointing = True if nonreentrant else "unsloth", random_state = 0)
    if nonreentrant:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs = {"use_reentrant": False})
    res["gc_mode"] = "use_reentrant=False" if nonreentrant else "unsloth"
    lora_stats(model)
    gc = GCCounter(model)
    batch = text_batch(tok, model.get_input_embeddings().weight.device)
    stage("train")
    torch.cuda.reset_peak_memory_stats()
    buf = io.StringIO()

    class Tee(io.TextIOBase):
        def write(self, x):
            buf.write(x)
            return sys.__stdout__.write(x)
    with contextlib.redirect_stdout(Tee()):
        manual_loop(model, batch, args.steps, gc)
    peak()
    res["gc_calls"] = gc.calls
    res["block_modes"] = _block_modes(model)
    res["block_compile_failed_msgs"] = [ln[:300] for ln in buf.getvalue().splitlines() if "compile failed" in ln][:6]


def cell_lora_noac():
    """zoo #2535: FastModel 16-bit LoRA, custom loop WITHOUT autocast: LoRA module output dtype (must be the base
    dtype), every LoRA tensor gets a non-zero gradient, losses (near-equal base vs head)."""
    import torch
    from unsloth import FastModel
    versions()
    stage("load")
    model, tok = FastModel.from_pretrained(MODELS["lora_noac"], max_seq_length = 256, dtype = torch.bfloat16,
                                           load_in_4bit = False)
    model = FastModel.get_peft_model(model, r = 16, lora_alpha = 32, lora_dropout = 0.0, bias = "none",
                                     target_modules = TARGETS, use_gradient_checkpointing = "unsloth", random_state = 0)
    seen = {}

    def hook(mod, inp, out):
        if isinstance(out, torch.Tensor):
            seen[str(out.dtype)] = seen.get(str(out.dtype), 0) + 1
    hooks = [m.register_forward_hook(hook) for n, m in model.named_modules() if n.endswith(("q_proj", "down_proj"))
             and hasattr(m, "lora_A")]
    batch = text_batch(tok, model.get_input_embeddings().weight.device)
    stage("train")
    torch.cuda.reset_peak_memory_stats()
    manual_loop(model, batch, args.steps)
    peak()
    # After the steps (lora_B is no longer zero, so lora_A gets a gradient too): one more fwd + bwd, no autocast.
    model.train()
    stage("grads")
    assert not torch.is_autocast_enabled()
    out = model(**batch)
    out.loss.backward()
    lora = {n: p for n, p in model.named_parameters() if p.requires_grad and "lora_" in n}
    res["lora_tensors"] = len(lora)
    res["lora_param_dtypes"] = sorted({str(p.dtype) for p in lora.values()})
    res["lora_grads_nonzero"] = sum(1 for p in lora.values() if p.grad is not None and float(p.grad.float().abs().sum()) > 0)
    res["lora_B_grads_nonzero"] = sum(1 for n, p in lora.items() if "lora_B" in n and p.grad is not None
                                      and float(p.grad.float().abs().sum()) > 0)
    res["lora_out_dtypes"] = seen
    for h in hooks:
        h.remove()
    model.zero_grad(set_to_none = True)
    res["all_lora_grads_nonzero"] = res["lora_grads_nonzero"] == res["lora_tensors"] > 0


def cell_st(decoder):
    """unsloth #17859: FastSentenceTransformer LoRA, SentenceTransformerTrainer + MultipleNegativesRankingLoss, 5 steps;
    prefetching loader / lean LoRA / decoder packing engagement where the head has them."""
    import torch
    from datasets import Dataset
    from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments, losses
    from unsloth import FastSentenceTransformer
    versions()
    stage("load")
    model = FastSentenceTransformer.from_pretrained(MODELS[args.cell], max_seq_length = 64)
    targets = ["q_proj", "k_proj", "v_proj", "o_proj"] if decoder else ["query", "key", "value", "dense"]
    model = FastSentenceTransformer.get_peft_model(model, r = 8, lora_alpha = 16, target_modules = targets,
                                                   use_gradient_checkpointing = False, random_state = 0)
    tr = model[0]
    res["packing_installed"] = bool(getattr(tr, "_unsloth_packing_installed", False))
    res["lean_lora_layers"] = sum(1 for m in tr.auto_model.modules() if hasattr(m, "_unsloth_peft_forward"))
    words = "red green blue square circle cat dog river house music light water north south".split()
    anchors = [" ".join(words[(i + j) % len(words)] for j in range(2 + i % 5)) for i in range(32)]
    data = Dataset.from_dict({"anchor": anchors, "positive": [a.split()[0] + " " + a.split()[-1] for a in anchors]})
    cfg = SentenceTransformerTrainingArguments(output_dir = os.path.join(os.path.dirname(os.path.abspath(args.out)),
                                                                         f"trainer_{args.cell}"),
                                               per_device_train_batch_size = 8, max_steps = args.steps,
                                               learning_rate = 1e-3, logging_steps = 1, report_to = "none", seed = 0,
                                               bf16 = True, save_strategy = "no", lr_scheduler_type = "constant",
                                               warmup_steps = 0)
    stage("train")
    torch.cuda.reset_peak_memory_stats()
    trainer = SentenceTransformerTrainer(model = model, args = cfg, train_dataset = data,
                                         loss = losses.MultipleNegativesRankingLoss(model))
    try:
        from unsloth.models import _sentence_transformer_prefetch as pf
        res["prefetch_loader"] = isinstance(trainer.get_train_dataloader(), pf.PrefetchedDataLoader)
    except Exception as e:  # noqa: BLE001
        res["prefetch_loader"] = f"absent: {type(e).__name__}"
    trainer.train()
    peak()
    logs = [h for h in trainer.state.log_history if "loss" in h]
    finish_losses([float(h["loss"]) for h in logs])


def hd512_dir(kind):
    return os.path.abspath(os.path.join(os.path.dirname(args.model_dir or args.out), f"tiny_{kind}_hd512"))


def build_hd512():
    """Seeded random tiny models with head_dim 512 (no unsloth import): a Llama, and Gemma-4 whose global layers use
    global_head_dim 512 (the tiny repo's config, random weights). Same bytes for every state."""
    import shutil
    import torch
    import transformers
    from huggingface_hub import snapshot_download
    d = hd512_dir("llama")
    if not os.path.isfile(os.path.join(d, "config.json")):
        tok = transformers.AutoTokenizer.from_pretrained(LLAMA_TOKENIZER)
        torch.manual_seed(0)
        cfg = transformers.LlamaConfig(vocab_size = len(tok), hidden_size = 256, intermediate_size = 512, num_hidden_layers = 2,
                                       num_attention_heads = 2, num_key_value_heads = 1, head_dim = 512,
                                       max_position_embeddings = 2048, bos_token_id = tok.bos_token_id,
                                       eos_token_id = tok.eos_token_id, tie_word_embeddings = False)
        tmp = f"{d}.tmp{os.getpid()}"
        transformers.LlamaForCausalLM(cfg).to(torch.bfloat16).save_pretrained(tmp)
        tok.save_pretrained(tmp)
        os.rename(tmp, d)
    d = hd512_dir("gemma4")
    if not os.path.isfile(os.path.join(d, "config.json")):
        src = snapshot_download(MODELS["gemma4"])
        tmp = f"{d}.tmp{os.getpid()}"
        os.makedirs(tmp)
        for f in os.listdir(src):
            if not f.endswith((".safetensors", ".bin")) and os.path.isfile(os.path.join(src, f)):
                shutil.copy(os.path.join(src, f), os.path.join(tmp, f))
        cp = os.path.join(tmp, "config.json")
        raw = json.load(open(cp, encoding = "utf-8"))
        raw["text_config"]["global_head_dim"] = 512
        json.dump(raw, open(cp, "w", encoding = "utf-8"))
        cfg = transformers.AutoConfig.from_pretrained(tmp)
        cls = getattr(transformers, (raw.get("architectures") or ["Gemma4ForConditionalGeneration"])[0])
        torch.manual_seed(0)
        cls(cfg).to(torch.bfloat16).save_pretrained(tmp)
        os.rename(tmp, d)
    res["hd512_dirs"] = [hd512_dir("llama"), hd512_dir("gemma4")]


def cell_hd512(kind):
    """unsloth #17860 (large-head attention tiles): a head_dim 512 LoRA step. The kernel refuses ROCm
    (patch_large_head_attention -> False), so this must train identically at base and head."""
    from unsloth import FastLanguageModel, FastModel
    d = hd512_dir(kind)
    if not os.path.isfile(os.path.join(d, "config.json")):
        raise RuntimeError(f"hd512 {kind} model missing ({d}): build_hd512 failed")
    model, _ = cell_causal(FastModel if kind == "gemma4" else FastLanguageModel, d, language_only = kind == "gemma4")
    import unsloth.kernels.large_head_attention as lh
    res["large_head_patch_returns"] = safe(lambda: lh.patch_large_head_attention(model))
    res["head_dims"] = sorted({int(getattr(m, "head_dim")) for m in model.modules()
                               if isinstance(getattr(m, "head_dim", None), int)})


def cell_bclens(key = "bclens", lens = (37, 64, 101, 72), dtype = "bfloat16"):
    """unsloth #17864: tiny Qwen3 LoRA with block compile, one step at each of 4 sequence lengths (aligned and not
    to 8): per-step losses and the Dynamo recompile count, which the dropped SDPA alignment guard should lower."""
    import logging
    import torch
    from unsloth import FastLanguageModel
    versions()
    stage("load")
    rec = []

    class H(logging.Handler):
        def emit(self, record):
            msg = record.getMessage()
            if "ecompil" in msg:
                rec.append(msg[:400])
    try:
        torch._logging.set_logs(recompiles = True)
    except Exception as e:  # noqa: BLE001
        res["set_logs_error"] = repr(e)[:120]
    logging.getLogger("torch").addHandler(H())
    from unsloth import FastModel
    torch.manual_seed(0)
    model, tok = FastModel.from_pretrained(MODELS[key], max_seq_length = 256, dtype = getattr(torch, dtype),
                                           load_in_4bit = False)
    model = FastModel.get_peft_model(model, r = 16, lora_alpha = 32, lora_dropout = 0.0, bias = "none",
                                     finetune_vision_layers = False, finetune_language_layers = True,
                                     finetune_attention_modules = True, finetune_mlp_modules = True,
                                     use_gradient_checkpointing = "unsloth", random_state = 0)
    dev = model.get_input_embeddings().weight.device
    vocab = model.get_input_embeddings().weight.shape[0]
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr = args.lr)
    model.train()
    stage("train")
    torch.cuda.reset_peak_memory_stats()
    lens, losses, graphs = list(lens), [], []
    g = torch.Generator().manual_seed(0)
    for L in lens:
        ids = torch.randint(10, min(vocab, 1000), (2, L), generator = g).to(dev)
        out = model(input_ids = ids, attention_mask = torch.ones_like(ids), labels = ids)
        out.loss.backward()
        opt.step()
        opt.zero_grad(set_to_none = True)
        losses.append(float(out.loss.detach()))
        graphs.append(dynamo_counters().get("unique_graphs"))
        res["losses_so_far"] = losses
        dump()
    res["seq_lens"] = lens
    res["block_modes"] = _block_modes(model)  # installed at the first forward
    res["unique_graphs_per_step"] = graphs
    res["recompile_logs"] = len(rec)
    res["recompile_reasons"] = [r[-200:] for r in rec[:6]]
    res["finite"] = all(math.isfinite(x) for x in losses)
    res["losses"] = losses
    res["decreasing"] = None  # different batch per step: equality base vs head is the check
    peak()


def cell_bswap():
    """unsloth-zoo #2536: zoo BlockSwap on a LoRA model (FastModel, block compile, Unsloth gradient checkpointing,
    transformers 5.23) built as an 8-layer seeded tiny Qwen3. For offload 2 and 6: one fwd + bwd equal to offload 0
    (loss and every LoRA grad) and whether the slot pool grew (the 'needs per-layer gradient checkpointing' message);
    then 5 training steps with offload 2 installed."""
    import contextlib
    import io
    import torch
    import transformers
    from unsloth import FastModel
    versions()
    stage("build")
    d = os.path.join(os.path.dirname(os.path.abspath(args.out)), "tiny_qwen3_8l")
    if not os.path.isfile(os.path.join(d, "config.json")):
        cfg = transformers.AutoConfig.from_pretrained(MODELS["bswap"])
        cfg.num_hidden_layers = 8
        if getattr(cfg, "layer_types", None):
            cfg.layer_types = [cfg.layer_types[0]] * 8
        torch.manual_seed(0)
        m = transformers.AutoModelForCausalLM.from_config(cfg, torch_dtype = torch.bfloat16)
        m.save_pretrained(d)
        transformers.AutoTokenizer.from_pretrained(MODELS["bswap"]).save_pretrained(d)
        del m
    stage("load")
    model, tok = FastModel.from_pretrained(d, max_seq_length = 256, dtype = torch.bfloat16, load_in_4bit = False)
    model = FastModel.get_peft_model(model, r = 16, lora_alpha = 32, lora_dropout = 0.0, bias = "none",
                                     finetune_vision_layers = False, finetune_language_layers = True,
                                     finetune_attention_modules = True, finetune_mlp_modules = True,
                                     use_gradient_checkpointing = "unsloth", random_state = 0)
    try:
        from unsloth.models._utils import _check_block_swap
        _check_block_swap(model)
        res["unsloth_offload_refusal"] = "none"
    except Exception as e:  # noqa: BLE001
        res["unsloth_offload_refusal"] = f"{type(e).__name__}: {e}"[:160]
    from unsloth_zoo import block_swap as bsm
    layers = bsm.find_decoder_layers(model)
    res["n_layers"] = len(layers)
    batch = text_batch(tok, model.get_input_embeddings().weight.device)
    model.train()

    def fwd_bwd():
        model.zero_grad(set_to_none = True)
        out = model(**batch)
        out.loss.backward()
        torch.cuda.synchronize()
        return float(out.loss.detach()), {n: p.grad.detach().float().clone() for n, p in model.named_parameters()
                                          if p.requires_grad and p.grad is not None}
    stage("ref")
    l0, g0 = fwd_bwd()
    res["block_modes"] = _block_modes(model)
    fwd_names = sorted({getattr(getattr(type(m).__dict__.get("forward", None) if "forward" not in m.__dict__ else
                                        getattr(m.forward, "__func__", m.forward), "__code__", None), "co_name", "?")
                        for m in layers})
    res["layer_forward_names"] = fwd_names
    cif = getattr(bsm, "_checkpoints_in_forward", None)
    res["ckpt_in_forward"] = safe(lambda: sum(bool(cif(m)) for m in layers)) if cif else "n/a (base)"
    buf = io.StringIO()
    stage("swap")
    eq, grew = {}, {}
    with contextlib.redirect_stdout(buf):
        for n in (2, 6):
            sw = bsm.BlockSwap(layers, n, model = model)
            l1, g1 = fwd_bwd()
            grew[n] = bool(getattr(sw, "_grew", None))
            diffs = [float((g0[k] - g1[k]).abs().max()) for k in g0 if k in g1]
            eq[n] = {"loss": [l0, l1], "grads": f"{len(diffs)}/{len(g0)}", "maxdiff": max(diffs) if diffs else None,
                     "equal": bool(l0 == l1 and diffs and len(diffs) == len(g0) and max(diffs) == 0.0)}
            if n == 2:
                stage("train")
                torch.cuda.reset_peak_memory_stats()
                manual_loop(model, batch, args.steps)
                peak()
            sw.remove()
            if n == 2:
                l0, g0 = fwd_bwd()  # new reference after the training steps
    out = buf.getvalue()
    print(out)
    res["bswap_detail"] = eq
    res["bswap_equal"] = all(v["equal"] for v in eq.values())
    res["bswap_grad_maxdiff"] = max((v["maxdiff"] or 0.0) for v in eq.values())
    res["bswap_grew"] = grew
    res["slot_pool_message"] = "growing the slot pool" in out

def _decode_run(model, gen_kwargs):
    """Greedy generate through Unsloth's generate twice: default env and UNSLOTH_DECODE_CUDA_GRAPHS=1 forced.
    Tokens plus a scores digest per run; graph backend availability (False on HIP)."""
    import hashlib
    import types
    import torch
    from unsloth.models import vision
    model._old_generate = model.generate
    model.generate = types.MethodType(vision.unsloth_base_fast_generate, model)
    try:
        import unsloth.models._generic_decode_graphs as GG
        res["graph_backend_available"] = safe(lambda: GG._BACKEND.available())
    except Exception as e:  # noqa: BLE001
        res["graph_backend_available"] = f"absent: {type(e).__name__}"
    runs = {}
    for tag, env in (("default", None), ("forced", "1")):
        if env is None:
            os.environ.pop("UNSLOTH_DECODE_CUDA_GRAPHS", None)
        else:
            os.environ["UNSLOTH_DECODE_CUDA_GRAPHS"] = env
        with torch.no_grad():
            out = model.generate(**gen_kwargs, do_sample = False, output_scores = True, return_dict_in_generate = True)
        sc = torch.stack(out.scores).float().cpu()
        runs[tag] = {"tokens": out.sequences[0].tolist(),
                     "scores_sha": hashlib.sha256(sc.numpy().tobytes()).hexdigest()[:16],
                     "scores_finite": bool(torch.isfinite(sc[sc > -1e30]).all())}
    os.environ.pop("UNSLOTH_DECODE_CUDA_GRAPHS", None)
    res["decode_runs"] = runs
    res["tokens"] = runs["default"]["tokens"]
    res["decode_forced_eq_default"] = runs["forced"]["tokens"] == runs["default"]["tokens"]
    res["finite"] = all(r["scores_finite"] for r in runs.values())
    res["losses"], res["decreasing"] = [], None


def cell_dec_whisper():
    """unsloth #17871: tiny Whisper greedy generate (12 tokens) through Unsloth's generate; decode graphs are off on
    ROCm, so tokens must match base."""
    import torch
    from transformers import WhisperForConditionalGeneration
    versions()
    stage("load")
    torch.manual_seed(0)
    model = WhisperForConditionalGeneration.from_pretrained(MODELS["dec_whisper"], dtype = torch.float32).cuda().eval()
    g = torch.Generator().manual_seed(0)
    feats = torch.randn(1, model.config.num_mel_bins, 2 * model.config.max_source_positions, generator = g).cuda()
    stage("generate")
    _decode_run(model, {"input_features": feats, "max_new_tokens": 12})


def cell_dec_gemma3n():
    """unsloth #17871: Gemma 3n shared-KV tiny (the PR test's config, seeded) greedy generate 10 tokens on GPU."""
    import torch
    from transformers import Gemma3nForCausalLM, Gemma3nTextConfig
    versions()
    stage("load")
    config = Gemma3nTextConfig(vocab_size = 128, vocab_size_per_layer_input = 128, hidden_size = 32,
                               hidden_size_per_layer_input = 8, intermediate_size = 64, num_hidden_layers = 4,
                               num_attention_heads = 2, num_key_value_heads = 1, head_dim = 16, laurel_rank = 4,
                               altup_num_inputs = 2, num_kv_shared_layers = 2, sliding_window = 16,
                               layer_types = ["sliding_attention", "full_attention", "sliding_attention", "full_attention"],
                               activation_sparsity_pattern = [0.0] * 4, max_position_embeddings = 64)
    torch.manual_seed(0)
    model = Gemma3nForCausalLM._from_config(config, dtype = torch.float32).cuda().eval()
    ids = torch.randint(3, 100, (1, 5), generator = torch.Generator().manual_seed(1)).cuda()
    stage("generate")
    _decode_run(model, {"input_ids": ids, "attention_mask": torch.ones_like(ids), "max_new_tokens": 10,
                        "min_new_tokens": 10, "pad_token_id": 0})


MOE_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def _moe_aux_model(repo, gc, r = 8):
    import torch
    from unsloth import FastModel
    model, tok = FastModel.from_pretrained(repo, max_seq_length = 512, dtype = torch.bfloat16, load_in_4bit = False)
    model.config.get_text_config().router_aux_loss_coef = 0.9
    model = FastModel.get_peft_model(model, r = r, lora_alpha = 2 * r, target_modules = MOE_TARGETS, random_state = 0,
                                     use_gradient_checkpointing = gc)
    for m in model.modules():
        if hasattr(m, "router_aux_loss_coef"):
            m.router_aux_loss_coef = 0.9
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "lora_B" in n:
                p.copy_((torch.randn(p.shape, generator = g) * 0.02).to(p.dtype))
    return model, tok


def cell_moeaux_nr():
    """unsloth #17872: tiny Qwen3-MoE LoRA, gradient checkpointing use_reentrant=False, router aux loss on
    (output_router_logits, coef 0.9), 5 steps; block modes and every "compile failed" message."""
    import contextlib
    import io
    import torch
    versions()
    stage("load")
    model, tok = _moe_aux_model(MODELS["moeaux_nr"], True)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs = {"use_reentrant": False})
    gc = GCCounter(model)
    ids = torch.randint(10, 1000, (2, 48), generator = torch.Generator().manual_seed(2)).cuda()
    batch = {"input_ids": ids, "labels": ids, "output_router_logits": True}
    with torch.no_grad():
        model.eval()
        o = model(**batch)
        res["aux_loss_step0"] = None if getattr(o, "aux_loss", None) is None else float(o.aux_loss)
    stage("train")
    torch.cuda.reset_peak_memory_stats()
    buf = io.StringIO()

    class Tee(io.TextIOBase):
        def write(self, x):
            buf.write(x)
            return sys.__stdout__.write(x)
    with contextlib.redirect_stdout(Tee()):
        manual_loop(model, batch, args.steps, gc)
    peak()
    res["gc_calls"] = gc.calls
    res["block_modes"] = _block_modes(model)
    res["block_compile_failed_msgs"] = [ln[:300] for ln in buf.getvalue().splitlines() if "compile failed" in ln][:6]


def cell_moeaux_gc(key):
    """unsloth #17873: LoRA grad norm with the router aux loss, no GC vs Unsloth GC (the PR's GPU test, recorded):
    rel diff must be < 2e-2 and the layer checkpoint function must run under GC."""
    import torch
    versions()
    out = {}
    for tag, gc in (("plain", False), ("unsloth", "unsloth")):
        stage(f"load_{tag}")
        model, tok = _moe_aux_model(MODELS[key], gc)
        calls = [0]
        for m in model.modules():
            f = m.__dict__.get("_gradient_checkpointing_func")
            if f is not None and getattr(m, "gradient_checkpointing", False):
                def counted(*a, _f = f, **k):
                    calls[0] += 1
                    return _f(*a, **k)
                m._gradient_checkpointing_func = counted
        model.train()
        ids = torch.randint(10, 1000, (2, 48), generator = torch.Generator().manual_seed(2)).cuda()
        stage(f"fwd_{tag}")
        o = model(input_ids = ids, labels = ids, output_router_logits = True)
        aux = None if getattr(o, "aux_loss", None) is None else float(o.aux_loss)
        o.loss.backward()
        norm = float(torch.sqrt(sum(p.grad.float().pow(2).sum() for n, p in model.named_parameters()
                                    if "lora_" in n and p.grad is not None)))
        out[tag] = {"loss": float(o.loss.detach()), "aux": aux, "lora_grad_norm": norm, "gc_calls": calls[0]}
        del model
        torch.cuda.empty_cache()
    res["moeaux"] = out
    pn, cn = out["plain"]["lora_grad_norm"], out["unsloth"]["lora_grad_norm"]
    res["aux_gc_rel"] = abs(cn - pn) / max(pn, 1e-12)
    res["aux_gc_ok"] = bool(res["aux_gc_rel"] < 2e-2 and out["unsloth"]["gc_calls"] > 0 and out["plain"]["gc_calls"] == 0)
    res["finite"] = all(math.isfinite(v["loss"]) and math.isfinite(v["lora_grad_norm"]) for v in out.values())
    res["losses"], res["decreasing"] = [out["plain"]["loss"], out["unsloth"]["loss"]], None


def cell_gkd():
    """unsloth #17876: single-GPU GKD step (TRL GKDTrainer through Unsloth), tiny Qwen3 student (LoRA) and a frozen
    tiny Qwen3 teacher, lmbda 0 (no on-policy generation), 3 steps; teacher device / trainable params recorded."""
    import tempfile
    import torch
    from unsloth import FastLanguageModel
    versions()
    try:
        from trl import GKDConfig, GKDTrainer
    except ImportError:
        from trl.experimental.gkd import GKDConfig, GKDTrainer
    import transformers
    from datasets import Dataset
    stage("load")
    model, tok = FastLanguageModel.from_pretrained(MODELS["gkd"], max_seq_length = 256, dtype = torch.bfloat16,
                                                   load_in_4bit = False)
    model = FastLanguageModel.get_peft_model(model, r = 16, lora_alpha = 32, lora_dropout = 0.0, bias = "none",
                                             target_modules = TARGETS, use_gradient_checkpointing = "unsloth",
                                             random_state = 0)
    torch.manual_seed(0)
    teacher = transformers.AutoModelForCausalLM.from_pretrained(MODELS["gkd"], dtype = torch.bfloat16)
    g = torch.Generator().manual_seed(3)
    with torch.no_grad():  # a teacher that differs from the student, so the distillation loss is not ~0
        for prm in teacher.parameters():
            prm.add_((torch.randn(prm.shape, generator = g) * 0.05).to(prm.dtype))
    ds = Dataset.from_list([{"messages": [{"role": "user", "content": d}, {"role": "assistant", "content": d}]}
                            for d in DOCS * 4])
    cfg = GKDConfig(output_dir = tempfile.mkdtemp(), max_steps = 3, per_device_train_batch_size = 2,
                    gradient_accumulation_steps = 1, learning_rate = 1e-3, logging_steps = 1, report_to = "none",
                    bf16 = True, lmbda = 0.0, beta = 0.5, max_new_tokens = 8, max_length = 128, seed = 0,
                    save_strategy = "no")
    tr = GKDTrainer(model = model, teacher_model = teacher, args = cfg, train_dataset = ds, processing_class = tok)
    tm = getattr(tr, "teacher_model", None)
    res["trainer_class"] = type(tr).__name__
    res["teacher_device"] = str(next(tm.parameters()).device) if tm is not None else None
    res["teacher_trainable"] = int(sum(p.numel() for p in tm.parameters() if p.requires_grad)) if tm is not None else None
    stage("train")
    torch.cuda.reset_peak_memory_stats()
    tr.train()
    peak()
    finish_losses([float(h["loss"]) for h in tr.state.log_history if "loss" in h])
    res["decreasing"] = None  # 3 steps on changing batches


def cell_granite_gc():
    """unsloth-zoo #2538: Granite-4 (GraniteMoeHybrid) float16 patch vs transformers 5.23 layer GC. Tiny seeded
    attention-only granitehybrid, fp32, LoRA (peft), seq 512: GC off vs on (non-reentrant) -> layer checkpoint calls,
    activations kept after forward, loss / LoRA grad norm parity; then 5 steps with GC on."""
    import peft
    import torch
    import transformers.models.granitemoehybrid.modeling_granitemoehybrid as gm
    versions()
    try:
        from unsloth_zoo.temporary_patches import granitemoehybrid as gpatch
        res["float16_patch"] = safe(lambda: gpatch.patch_GraniteMoeHybridDecoderLayer_float16())
    except Exception as e:  # noqa: BLE001
        res["float16_patch"] = f"ERR {type(e).__name__}: {e}"[:160]
    fwd = gm.GraniteMoeHybridDecoderLayer.forward
    res["layer_forward_name"] = getattr(getattr(fwd, "__code__", None), "co_name", None)
    res["layer_forward_float16_mark"] = bool(getattr(fwd, "_unsloth_granite_float16", False))

    def build(gc):
        cfg = gm.GraniteMoeHybridConfig(vocab_size = 1000, hidden_size = 256, intermediate_size = 512,
                                        shared_intermediate_size = 512, num_hidden_layers = 4,
                                        layer_types = ["attention"] * 4, num_attention_heads = 4, num_key_value_heads = 2,
                                        num_local_experts = 0, num_experts_per_tok = 0, residual_multiplier = 0.25,
                                        embedding_multiplier = 12.0, logits_scaling = 4.0, mamba_n_heads = 16,
                                        mamba_d_head = 32, mamba_n_groups = 1)  # 16 x 32 = 2 x hidden
        torch.manual_seed(0)
        m = gm.GraniteMoeHybridForCausalLM(cfg).float().cuda()
        m = peft.get_peft_model(m, peft.LoraConfig(r = 8, lora_alpha = 16, target_modules = ["q_proj", "k_proj", "v_proj",
                                                                                            "o_proj"], init_lora_weights = False))
        if gc:
            m.gradient_checkpointing_enable(gradient_checkpointing_kwargs = {"use_reentrant": False})
            m.enable_input_require_grads()
        m.train()
        calls = [0]
        for layer in m.base_model.model.model.layers:
            f = layer.__dict__.get("_gradient_checkpointing_func")
            if f is not None:
                def counted(*a, _f = f, **k):
                    calls[0] += 1
                    return _f(*a, **k)
                layer._gradient_checkpointing_func = counted
        return m, calls
    ids = torch.randint(0, 1000, (2, 512), generator = torch.Generator().manual_seed(2)).cuda()
    out = {}
    for tag, gc in (("off", False), ("on", True)):
        stage(f"gc_{tag}")
        m, calls = build(gc)
        torch.cuda.synchronize()
        m0 = torch.cuda.memory_allocated()
        o = m(input_ids = ids, labels = ids, use_cache = False)
        torch.cuda.synchronize()
        kept = (torch.cuda.memory_allocated() - m0) / 2**20
        o.loss.backward()
        norm = float(torch.sqrt(sum(p.grad.float().pow(2).sum() for n, p in m.named_parameters()
                                    if "lora_" in n and p.grad is not None)))
        out[tag] = {"loss": float(o.loss.detach()), "kept_mib": round(kept, 2), "gc_calls": calls[0], "lora_grad_norm": norm}
        if gc:
            stage("train")
            torch.cuda.reset_peak_memory_stats()
            batch = {"input_ids": ids, "labels": ids, "use_cache": False}
            manual_loop(m, batch, args.steps)
            peak()
            res["gc_calls"] = calls[0]
        del m, o
        torch.cuda.empty_cache()
    res["granite"] = out
    res["granite_gc_calls_on"] = out["on"]["gc_calls"]
    res["granite_kept_mib"] = [out["off"]["kept_mib"], out["on"]["kept_mib"]]
    res["granite_loss_eq"] = out["off"]["loss"] == out["on"]["loss"]
    res["granite_grad_rel"] = abs(out["on"]["lora_grad_norm"] - out["off"]["lora_grad_norm"]) / max(out["off"]["lora_grad_norm"], 1e-12)


def cell_padmix(key, attn = None, u_mask = None):
    """unsloth #17880: block compile with a fixed P P U P U P micro-batch sequence (P = padded batch with an attention
    mask, U = no attention mask), same length; per step: loss, compiled-graph count, Dynamo guard failures."""
    import torch
    from unsloth import FastModel
    versions()
    stage("load")
    torch.manual_seed(0)
    extra = {"attn_implementation": attn} if attn else {}
    model, tok = FastModel.from_pretrained(MODELS[key], max_seq_length = 256, dtype = torch.bfloat16, load_in_4bit = False,
                                           **extra)
    model = FastModel.get_peft_model(model, r = 16, lora_alpha = 32, lora_dropout = 0.0, bias = "none",
                                     finetune_vision_layers = False, finetune_language_layers = True,
                                     finetune_attention_modules = True, finetune_mlp_modules = True,
                                     use_gradient_checkpointing = "unsloth", random_state = 0)
    cfgs = [model.config] + [getattr(model.config, k) for k in ("text_config",) if getattr(model.config, k, None) is not None]
    res["attn_implementation"] = sorted({str(getattr(c, "_attn_implementation", None)) for c in cfgs})
    mask_calls = [0]
    if attn == "flex_attention":
        import transformers.masking_utils as mu
        reg = getattr(mu, "ALL_MASK_ATTENTION_FUNCTIONS", None)
        live = reg["flex_attention"] if reg is not None and "flex_attention" in reg else getattr(mu, "flex_attention_mask")
        res["flex_mask_keeps_padding_marker"] = bool(getattr(live, "_unsloth_keeps_padding", False))

        def counted(*a, _f = live, **k):
            mask_calls[0] += 1
            return _f(*a, **k)
        counted._unsloth_keeps_padding = getattr(live, "_unsloth_keeps_padding", False)
        if reg is not None:
            reg.register("flex_attention", counted)
        mu.flex_attention_mask = counted
    dev = model.get_input_embeddings().weight.device
    vocab = model.get_input_embeddings().weight.shape[0]
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr = args.lr)
    model.train()
    try:
        from torch._dynamo.utils import guard_failures
    except Exception:  # noqa: BLE001
        guard_failures = {}
    stage("train")
    torch.cuda.reset_peak_memory_stats()
    g = torch.Generator().manual_seed(0)
    pattern, losses, graphs, gfails = "PPUPUP", [], [], []
    for kind in pattern:
        ids = torch.randint(10, min(vocab, 1000), (2, 48), generator = g).to(dev)
        labels = ids.clone()
        kw = {"input_ids": ids}
        if kind == "P":
            mask = torch.ones_like(ids)
            mask[1, -16:] = 0
            labels[1, -16:] = -100
            kw["attention_mask"] = mask
        elif u_mask == "ones":  # an unpadded batch as a collator feeds it: an all-ones attention mask
            kw["attention_mask"] = torch.ones_like(ids)
        out = model(**kw, labels = labels)
        out.loss.backward()
        opt.step()
        opt.zero_grad(set_to_none = True)
        losses.append(float(out.loss.detach()))
        graphs.append(dynamo_counters().get("unique_graphs"))
        gfails.append(int(sum(len(v) for v in guard_failures.values())))
        res.setdefault("flex_mask_calls_per_step", []).append(mask_calls[0])
        res["losses_so_far"] = losses
        dump()
    res["pattern"] = pattern + (" (U = all-ones mask)" if u_mask == "ones" else " (U = no mask)")
    res["unique_graphs_per_step"] = graphs
    res["guard_failures_per_step"] = gfails
    res["guard_failure_reasons"] = [str(r)[:200] for v in list(guard_failures.values())[:3] for r in v[:2]]
    res["block_modes"] = _block_modes(model)
    res["losses"], res["finite"] = losses, all(math.isfinite(x) for x in losses)
    res["decreasing"] = None
    peak()


def _find_attr(model, name):
    seen, stack = set(), [model]
    while stack:
        m = stack.pop()
        if id(m) in seen or m is None:
            continue
        seen.add(id(m))
        if name in getattr(m, "__dict__", {}):
            return m.__dict__[name]
        for a in ("module", "base_model", "model"):
            stack.append(getattr(m, a, None) if hasattr(m, a) else None)
    return "unset"


def cell_grpo_txt(zadv = False):
    """unsloth #17875 + zoo #2540: single-GPU text GRPO (tiny Qwen3 LoRA, 2 steps, use_vllm False): losses, rewards,
    and whether sequence packing stayed on (_unsloth_seq_packing_grad_ok)."""
    import torch
    from unsloth import FastLanguageModel
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer
    versions()
    stage("load")
    model, tok = FastLanguageModel.from_pretrained(MODELS["grpo_txt"], max_seq_length = 256, dtype = torch.bfloat16,
                                                   load_in_4bit = False)
    model = FastLanguageModel.get_peft_model(model, r = 16, lora_alpha = 32, lora_dropout = 0.0, bias = "none",
                                             target_modules = TARGETS, use_gradient_checkpointing = "unsloth",
                                             random_state = 0)
    ds = Dataset.from_list([{"prompt": [{"role": "user", "content": d}]} for d in DOCS * 4])

    def reward(completions, prompts = None, **kw):
        if zadv:  # zoo #2577: a constant reward for prompts of even length gives whole zero-advantage groups
            return [1.0 if len(str(p)) % 2 == 0 else float(len(str(c))) / 10.0 for p, c in zip(prompts, completions)]
        return [float(len(str(c))) / 10.0 for c in completions]
    if zadv:
        res["zadv_prompt_parity"] = [len(str([{"role": "user", "content": d}])) % 2 for d in DOCS]
    cfg = _cfg(GRPOConfig, output_dir = os.path.join(os.path.dirname(os.path.abspath(args.out)), "trainer_grpo_txt"),
               per_device_train_batch_size = 4 if zadv else 2, num_generations = 2, max_completion_length = 8, max_prompt_length = None,
               max_steps = 3 if zadv else 2, learning_rate = 5e-5, logging_steps = 1, report_to = "none", bf16 = True, seed = 0,
               save_strategy = "no", temperature = 1.0, use_vllm = False)
    trainer = GRPOTrainer(model = model, reward_funcs = [reward], args = cfg, train_dataset = ds, processing_class = tok)
    res["trainer_class"] = type(trainer).__name__
    stage("train")
    torch.cuda.reset_peak_memory_stats()
    trainer.train()
    peak()
    res["seq_packing_grad_ok"] = str(_find_attr(trainer.model, "_unsloth_seq_packing_grad_ok"))
    logs = [h for h in trainer.state.log_history if "loss" in h]
    finish_losses([float(h["loss"]) for h in logs])
    res["decreasing"] = None
    res["rewards"] = [h.get("reward") for h in logs]
    res["zadv_rows_skipped"] = getattr(trainer, "_unsloth_zero_advantage_rows_skipped", None)  # zoo #2577
    res["zadv_policy_ok"] = str(getattr(trainer, "_unsloth_zero_advantage_policy_ok", None))[:80]
    res["zadv_metric"] = [h.get("unsloth/zero_advantage_rows_skipped") for h in logs]
    res["zadv_env"] = os.environ.get("UNSLOTH_GRPO_SKIP_ZERO_ADVANTAGE")


def _pack_stats():
    out = {}
    try:
        import importlib
        ad = importlib.import_module("unsloth.utils.attention_dispatch")  # unsloth.utils re-exports a same-named function
        stats = getattr(ad, "PACKED_BLOCK_STATS", None)
        out["packed_block"] = dict(stats) if stats is not None else "absent"
    except Exception as e:  # noqa: BLE001
        out["packed_block"] = f"ERR {type(e).__name__}"
    try:
        import importlib
        hp = importlib.import_module("unsloth.utils.hf_packed_attention")
        out["hf_packed_attention"] = dict(getattr(hp, "HF_PACKED_ATTENTION_STATS", {}) or {}) or "absent"
    except Exception as e:  # noqa: BLE001
        out["hf_packed_attention"] = f"ERR {type(e).__name__}"
    return out


def cell_sft_pack(kind, mode):
    """unsloth #17883 / #17884: SFTTrainer on varied-length rows. mode pfree = padding_free=True (max_steps);
    steps / epoch = packing and padding_free left unset (max_steps vs num_train_epochs=1). Records the resolved
    packing / padding_free, block modes, packed-block compiled vs eager counts, "uncompiled" log lines."""
    import contextlib
    import io
    import torch
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer
    versions()
    stage("load")
    if kind == "llama":
        from unsloth import FastLanguageModel as L
        model, tok = L.from_pretrained(built_llama(), max_seq_length = 256, dtype = torch.bfloat16, load_in_4bit = False)
        model = L.get_peft_model(model, r = 16, lora_alpha = 32, lora_dropout = 0.0, bias = "none", target_modules = TARGETS,
                                 use_gradient_checkpointing = "unsloth", random_state = 0)
    else:
        from unsloth import FastModel as L
        model, tok = L.from_pretrained(MODELS["pfree_qwen3"], max_seq_length = 256, dtype = torch.bfloat16, load_in_4bit = False)
        model = L.get_peft_model(model, r = 16, lora_alpha = 32, lora_dropout = 0.0, bias = "none",
                                 finetune_vision_layers = False, finetune_language_layers = True,
                                 finetune_attention_modules = True, finetune_mlp_modules = True,
                                 use_gradient_checkpointing = "unsloth", random_state = 0)
    tt = getattr(tok, "tokenizer", tok)
    if tt.pad_token is None:
        tt.pad_token = tt.eos_token
    texts = [" ".join(DOCS[: (k % len(DOCS)) + 1]) * (1 + k % 3) + (tt.eos_token or "") for k in range(12)]
    ds = Dataset.from_dict({"text": texts})
    kw = dict(output_dir = os.path.join(os.path.dirname(os.path.abspath(args.out)), f"trainer_{args.cell}"),
              per_device_train_batch_size = 2, gradient_accumulation_steps = 1, learning_rate = args.lr, logging_steps = 1,
              bf16 = True, report_to = "none", seed = 0, dataset_text_field = "text", max_length = 128,
              save_strategy = "no", dataset_num_proc = 1, warmup_steps = 0, lr_scheduler_type = "constant")
    if mode == "pfree":
        kw.update(max_steps = args.steps, padding_free = True, packing = False)
    elif mode == "steps":
        kw.update(max_steps = args.steps)
    else:
        kw.update(num_train_epochs = 1)
    cfg = SFTConfig(**kw)
    before = _pack_stats()
    buf = io.StringIO()

    class Tee(io.TextIOBase):
        def write(self, x):
            buf.write(x)
            return sys.__stdout__.write(x)
    stage("train")
    torch.cuda.reset_peak_memory_stats()
    with contextlib.redirect_stdout(Tee()):
        trainer = SFTTrainer(model = model, train_dataset = ds, processing_class = tt, args = cfg)
        res["trainer_class"] = type(trainer).__name__
        res["resolved_packing"] = getattr(trainer.args, "packing", None)
        res["resolved_padding_free"] = getattr(trainer.args, "padding_free", None)
        res["train_rows"] = len(trainer.train_dataset) if trainer.train_dataset is not None else None
        trainer.train()
    peak()
    res["pack_stats_before"], res["pack_stats_after"] = before, _pack_stats()
    res["block_modes"] = _block_modes(model)
    out = buf.getvalue()
    res["uncompiled_msgs"] = [ln[:200] for ln in out.splitlines() if "uncompiled" in ln.lower()][:4]
    res["packing_msgs"] = [ln[:200] for ln in out.splitlines() if "pack" in ln.lower()][:4]
    logs = [h for h in trainer.state.log_history if "loss" in h]
    res["global_steps"] = trainer.state.global_step
    res["packed_rows_verdict"] = str(getattr(trainer.model, "_unsloth_packed_rows_verdict", "unset"))
    finish_losses([float(h["loss"]) for h in logs])
    res["decreasing"] = None


def cell_fft16():
    """unsloth #18192: bf16 full finetuning (tiny built llama, full_finetuning=True, adamw_torch, lr 1e-5, 4 steps):
    fraction of trainable bf16 weights changed after step 1, stochastic-rounding engagement, losses / peak."""
    import torch
    from unsloth import FastLanguageModel
    from datasets import Dataset
    from transformers import TrainerCallback
    from trl import SFTConfig, SFTTrainer
    versions()
    stage("load")
    model, tok = FastLanguageModel.from_pretrained(built_llama(), max_seq_length = 256, dtype = torch.bfloat16,
                                                   load_in_4bit = False, full_finetuning = True)
    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    res["trainable_dtypes"] = sorted({str(p.dtype) for _, p in params})
    res["trainable_numel"] = sum(p.numel() for _, p in params)
    snap = {n: p.detach().clone() for n, p in params}
    tt = getattr(tok, "tokenizer", tok)
    if tt.pad_token is None:
        tt.pad_token = tt.eos_token
    ds = Dataset.from_dict({"text": [d + (tt.eos_token or "") for d in DOCS] * 8})
    cfg = SFTConfig(output_dir = os.path.join(os.path.dirname(os.path.abspath(args.out)), f"trainer_{args.cell}"),
                    per_device_train_batch_size = 2, gradient_accumulation_steps = 1, max_steps = 4,
                    learning_rate = 1e-5, logging_steps = 1, bf16 = True, report_to = "none", seed = 0,
                    dataset_text_field = "text", max_length = 64, save_strategy = "no", dataset_num_proc = 1,
                    warmup_steps = 0, lr_scheduler_type = "constant", packing = False, optim = "adamw_torch")

    class _Step1(TrainerCallback):
        def on_step_end(self, a, state, control, **kw):
            if state.global_step == 1:
                changed = total = 0
                for n, p in params:
                    changed += int((p.detach() != snap[n]).sum())
                    total += p.numel()
                res["frac_changed_step1"] = changed / max(total, 1)
    stage("train")
    torch.cuda.reset_peak_memory_stats()
    trainer = SFTTrainer(model = model, train_dataset = ds, processing_class = tt, args = cfg)
    trainer.add_callback(_Step1())
    res["trainer_class"] = type(trainer).__name__
    trainer.train()
    peak()
    opt = trainer.optimizer
    inner = getattr(opt, "optimizer", opt)
    res["sr_weights"] = getattr(inner, "_unsloth_bf16_stochastic_rounding", None)
    res["sr_env"] = os.environ.get("UNSLOTH_BF16_STOCHASTIC_ROUNDING")
    res["optimizer_class"] = type(inner).__name__
    logs = [h for h in trainer.state.log_history if "loss" in h]
    finish_losses([float(h["loss"]) for h in logs])
    res["decreasing"] = None



def cell_lpad():
    """unsloth #18202: tiny built llama + LoRA, training-mode forward/backward of the same 4 rows (lengths 12-36)
    padded left vs right by DataCollatorForSeq2Seq: loss, LoRA grad norm, max grad difference. The first real token's
    label is -100 in both layouts (left padding would otherwise predict it from a pad), so the targets are identical."""
    import torch
    from unsloth import FastLanguageModel
    from transformers import DataCollatorForSeq2Seq
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
    rows = []
    for i, n in enumerate((12, 20, 28, 36)):
        ids = tt(DOCS[i % len(DOCS)] * 4, add_special_tokens = False)["input_ids"][:n]
        labels = list(ids)
        labels[0] = -100
        rows.append({"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels})
    lora = [p for n, p in model.named_parameters() if p.requires_grad]
    model.train()
    stage("train")
    torch.cuda.reset_peak_memory_stats()
    out = {}
    for side in ("right", "left", "right2"):
        tt.padding_side = "left" if side == "left" else "right"
        batch = DataCollatorForSeq2Seq(tt, padding = True, return_tensors = "pt")([dict(r) for r in rows])
        batch = {k: v.to(model.get_input_embeddings().weight.device) for k, v in batch.items()}
        model.zero_grad(set_to_none = True)
        loss = model(**batch).loss
        loss.backward()
        grads = torch.cat([p.grad.detach().float().flatten() for p in lora])
        out[side] = (float(loss), grads.clone())
        res[f"loss_{side}"] = float(loss)
        res[f"gnorm_{side}"] = float(grads.norm())
    peak()
    res["max_grad_diff_left_vs_right"] = float((out["left"][1] - out["right"][1]).abs().max())
    res["max_grad_diff_right_vs_right2"] = float((out["right2"][1] - out["right"][1]).abs().max())
    res["gnorm_rel_diff_left_vs_right"] = abs(res["gnorm_left"] - res["gnorm_right"]) / max(res["gnorm_right"], 1e-12)
    res["loss_abs_diff_left_vs_right"] = abs(res["loss_left"] - res["loss_right"])
    finish_losses([res["loss_right"], res["loss_left"]])
    res["decreasing"] = None


def _dpo_prefix_share_stats():
    """unsloth #18169: shared-prompt preference forward counters (absent at the base)."""
    try:
        import unsloth.models.dpo_prefix_share as ps
    except Exception as e:
        res["prefix_share"] = f"absent ({type(e).__name__})"
        return
    res["prefix_share_engaged"] = getattr(ps, "ENGAGED_COUNT", None)
    res["prefix_share_declined"] = getattr(ps, "DECLINED_COUNT", None)
    res["prefix_share_verify"] = str(getattr(ps, "LAST_VERIFY", None))[:200]
    try:
        from unsloth.utils.prefix_grouper_kernel import compile_check_failed, TORCH_SUPPORTS_PREFIX_GROUPER
        res["prefix_share_torch_ok"] = bool(TORCH_SUPPORTS_PREFIX_GROUPER)
        res["prefix_share_compile_failed"] = str(compile_check_failed())[:300]
    except Exception as e:
        res["prefix_share_kernel"] = f"{type(e).__name__}: {e}"[:300]


def cell_dpo_txt():
    """unsloth #17886: single-GPU text DPO through Unsloth's DPO trainer (tiny Qwen3 LoRA, PEFT reference = adapter
    disabled, 3 steps): losses / rewards, chosen-rejected margins."""
    import tempfile
    import torch
    from unsloth import FastLanguageModel
    from datasets import Dataset
    from trl import DPOConfig, DPOTrainer
    versions()
    stage("load")
    model, tok = FastLanguageModel.from_pretrained(MODELS["dpo_txt"], max_seq_length = 256, dtype = torch.bfloat16,
                                                   load_in_4bit = False)
    model = FastLanguageModel.get_peft_model(model, r = 16, lora_alpha = 32, lora_dropout = 0.0, bias = "none",
                                             target_modules = TARGETS, use_gradient_checkpointing = "unsloth",
                                             random_state = 0)
    tt = getattr(tok, "tokenizer", tok)
    if tt.pad_token is None:
        tt.pad_token = tt.eos_token
    rows = [{"prompt": d[:30], "chosen": d[30:], "rejected": d[30:][::-1]} for d in DOCS] * 4
    cfg = _cfg(DPOConfig, output_dir = tempfile.mkdtemp(), per_device_train_batch_size = 2, gradient_accumulation_steps = 1,
               max_steps = 3, learning_rate = 1e-3, logging_steps = 1, report_to = "none", bf16 = True, seed = 0,
               save_strategy = "no", max_length = 128, beta = 0.1)
    trainer = DPOTrainer(model = model, ref_model = None, args = cfg, train_dataset = Dataset.from_list(rows),
                         processing_class = tt)
    res["trainer_class"] = type(trainer).__name__
    res["ref_model_is_none"] = getattr(trainer, "ref_model", None) is None
    res["hs_fallback_wrapped"] = bool(getattr(type(trainer).__init__, "_unsloth_grpo_hidden_states_init_wrapped", False))
    stage("train")
    torch.cuda.reset_peak_memory_stats()
    trainer.train()
    peak()
    logs = [h for h in trainer.state.log_history if "loss" in h]
    finish_losses([float(h["loss"]) for h in logs])
    res["decreasing"] = None
    res["reward_margins"] = [h.get("rewards/margins") for h in logs]
    _dpo_prefix_share_stats()


def cell_orpo_txt(unwrap = False):
    """unsloth #17906: single-GPU ORPO through Unsloth's ORPO trainer (tiny Qwen3 LoRA, 3 steps); whether the
    hidden-states fallback wrapper is installed on the trainer class."""
    import tempfile
    import torch
    from unsloth import FastLanguageModel
    from datasets import Dataset
    try:
        from trl import ORPOConfig, ORPOTrainer
    except ImportError:
        from trl.experimental.orpo import ORPOConfig, ORPOTrainer
    versions()
    stage("load")
    model, tok = FastLanguageModel.from_pretrained(MODELS["orpo_txt"], max_seq_length = 256, dtype = torch.bfloat16,
                                                   load_in_4bit = False)
    model = FastLanguageModel.get_peft_model(model, r = 16, lora_alpha = 32, lora_dropout = 0.0, bias = "none",
                                             target_modules = TARGETS, use_gradient_checkpointing = "unsloth",
                                             random_state = 0)
    tt = getattr(tok, "tokenizer", tok)
    if tt.pad_token is None:
        tt.pad_token = tt.eos_token
    rows = [{"prompt": d[:30], "chosen": d[30:], "rejected": d[30:][::-1]} for d in DOCS] * 4
    cfg = _cfg(ORPOConfig, output_dir = tempfile.mkdtemp(), per_device_train_batch_size = 2, gradient_accumulation_steps = 1,
               max_steps = 3, learning_rate = 1e-3, logging_steps = 1, report_to = "none", bf16 = True, seed = 0,
               save_strategy = "no", max_length = 128, beta = 0.1)
    trainer = ORPOTrainer(model = model, args = cfg, train_dataset = Dataset.from_list(rows), processing_class = tt)
    res["trainer_class"] = type(trainer).__name__
    if unwrap:  # unsloth #18169 diagnostic: restore the methods the shared-prefix wrapper replaced
        undone = []
        for _n in ("_compute_loss", "compute_ref_log_probs", "concatenated_forward"):
            _f = type(trainer).__dict__.get(_n)
            if _f is not None and getattr(_f, "_unsloth_dpo_prefix_share", False):
                setattr(type(trainer), _n, _f.__wrapped__)
                undone.append(_n)
        res["prefix_share_unwrapped"] = undone
    res["prefix_share_env"] = os.environ.get("UNSLOTH_DPO_PREFIX_SHARE")
    res["hs_fallback_wrapped"] = bool(getattr(type(trainer).__init__, "_unsloth_grpo_hidden_states_init_wrapped", False))
    stage("train")
    torch.cuda.reset_peak_memory_stats()
    trainer.train()
    peak()
    logs = [h for h in trainer.state.log_history if "loss" in h]
    finish_losses([float(h["loss"]) for h in logs])
    res["decreasing"] = None
    _dpo_prefix_share_stats()


def cell_mrope_pack():
    """unsloth #17910: tiny Qwen2.5-VL, text-only LoRA, SFTTrainer padding_free=True (3 steps), then the packed-row
    leak check on the trained model (fp32): loss of document B packed after A vs after a different A' vs B alone,
    fed as the padding-free collator feeds it (2D position_ids + packed_seq_lengths). Records the cached packed-row
    verdict and the position_ids shape the model actually received."""
    import tempfile
    import torch
    import torch.nn.functional as F
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastVisionModel as FastModel  # VL class on every backend
    versions()
    stage("load")
    model, proc = FastModel.from_pretrained(MODELS["mrope_pack"], max_seq_length = 256, dtype = torch.float32,
                                            load_in_4bit = False)
    model = FastModel.get_peft_model(model, r = 8, lora_alpha = 16, lora_dropout = 0.0, bias = "none",
                                     finetune_vision_layers = False, finetune_language_layers = True,
                                     finetune_attention_modules = True, finetune_mlp_modules = True,
                                     use_gradient_checkpointing = "unsloth", random_state = 0)
    tok = getattr(proc, "tokenizer", proc)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    vocab = model.get_input_embeddings().weight.shape[0]
    g = torch.Generator().manual_seed(0)
    rows = [{"input_ids": torch.randint(10, min(vocab, 1000), (int(n),), generator = g).tolist()}
            for n in (12, 9, 20, 7, 15, 11, 18, 8)]
    cfg = _cfg(SFTConfig, output_dir = tempfile.mkdtemp(), per_device_train_batch_size = 2, max_steps = 3,
               learning_rate = 1e-3, logging_steps = 1, report_to = "none", seed = 0, save_strategy = "no",
               padding_free = True, packing = False, max_length = 128, dataset_num_proc = 1)
    trainer = SFTTrainer(model = model, args = cfg, train_dataset = Dataset.from_list(rows), processing_class = tok)
    res["trainer_class"] = type(trainer).__name__
    res["resolved_padding_free"] = getattr(trainer.args, "padding_free", None)
    res["packed_rows_verdict"] = str(getattr(trainer.model, "_unsloth_packed_rows_verdict", "unset"))
    res["text_prepended_hook"] = bool(getattr(trainer.model, "_unsloth_text_prepended_position_ids", False))
    seen = []

    def spy(module, args_, kwargs):
        pid = kwargs.get("position_ids")
        if pid is not None and len(seen) < 64:
            seen.append(list(pid.shape))
    handle = trainer.model.register_forward_pre_hook(spy, with_kwargs = True)
    stage("train")
    torch.cuda.reset_peak_memory_stats()
    trainer.train()
    peak()
    logs = [h for h in trainer.state.log_history if "loss" in h]
    finish_losses([float(h["loss"]) for h in logs])
    res["decreasing"] = None
    res["train_position_ids_shapes"] = sorted({str(x[:1] + ["B", "T"]) if len(x) == 3 else str(len(x)) + "D" for x in seen})
    stage("leak")
    m = trainer.model
    dev = m.get_input_embeddings().weight.device

    def packed(docs):
        ids = torch.cat(docs)[None].to(dev)
        pos = torch.cat([torch.arange(len(d)) for d in docs])[None].to(dev)
        lengths = torch.tensor([len(d) for d in docs], dtype = torch.int32, device = dev)
        logits = m(input_ids = ids, position_ids = pos, packed_seq_lengths = lengths, use_cache = False).logits[0]
        b = docs[-1].to(dev)
        rb = logits[-len(b):]
        return float(F.cross_entropy(rb[:-1].float(), b[1:], reduction = "sum"))

    def alone(d):
        d = d.to(dev)
        logits = m(input_ids = d[None], position_ids = torch.arange(len(d), device = dev)[None], use_cache = False).logits[0]
        return float(F.cross_entropy(logits[:-1].float(), d[1:], reduction = "sum"))
    g = torch.Generator().manual_seed(1)
    a, a2, b = (torch.randint(10, min(vocab, 1000), (n,), generator = g) for n in (12, 12, 9))
    seen.clear()
    m.eval()
    try:
        with torch.no_grad():
            pb, pab, ab = packed([a, b]), packed([a2, b]), alone(b)
        res["leak_losses_b_after_a_after_a2_alone"] = [pb, pab, ab]
        res["leak"] = max(abs(pb - pab), abs(pb - ab))
        res["leak_position_ids_shapes"] = [list(x) for x in seen[:3]]
    except Exception as e:  # noqa: BLE001
        res["leak"] = f"ERR {type(e).__name__}: {e}"[:200]
    handle.remove()


def cell_gkd_proc(key):
    """unsloth #17920: GKD with a multimodal checkpoint loaded through Unsloth (FastModel returns a processor), text-only
    LoRA student, frozen teacher of the same checkpoint, processing_class = the processor, lmbda 0, 3 steps."""
    import tempfile
    import torch
    import transformers
    from datasets import Dataset
    from unsloth import FastModel
    try:
        from trl.experimental.gkd import GKDConfig, GKDTrainer
    except ImportError:
        from trl import GKDConfig, GKDTrainer
    versions()
    stage("load")
    torch.manual_seed(0)
    model, proc = FastModel.from_pretrained(MODELS[key], max_seq_length = 256, dtype = torch.bfloat16, load_in_4bit = False)
    res["processing_class_type"] = type(proc).__name__
    model = FastModel.get_peft_model(model, r = 8, lora_alpha = 16, lora_dropout = 0.0, bias = "none",
                                     finetune_vision_layers = False, finetune_language_layers = True,
                                     finetune_attention_modules = True, finetune_mlp_modules = True,
                                     use_gradient_checkpointing = "unsloth", random_state = 0)
    teacher = transformers.AutoModelForCausalLM.from_pretrained(MODELS[key], dtype = torch.bfloat16).cuda().eval()
    msgs = [[{"role": "user", "content": d[:30]}, {"role": "assistant", "content": d[30:]}] for d in DOCS] * 4
    cfg = GKDConfig(output_dir = tempfile.mkdtemp(), per_device_train_batch_size = 2, max_steps = 3, max_length = 64,
                    lmbda = 0.0, beta = 0.5, learning_rate = 1e-3, logging_steps = 1, report_to = "none",
                    save_strategy = "no", bf16 = True, seed = 0)
    stage("trainer")
    trainer = GKDTrainer(model = model, teacher_model = teacher, args = cfg,
                         train_dataset = Dataset.from_list([{"messages": m} for m in msgs]), processing_class = proc)
    res["trainer_class"] = type(trainer).__name__
    res["trainer_processing_class"] = type(trainer.processing_class).__name__
    stage("train")
    torch.cuda.reset_peak_memory_stats()
    trainer.train()
    peak()
    finish_losses([float(h["loss"]) for h in trainer.state.log_history if "loss" in h])
    res["decreasing"] = None


def cell_cp_ring_kernels():
    """unsloth #17932: the SDPA kernels the context-parallel ring pins (per dtype at head: half = no efficient kernel,
    float32 = efficient allowed), and whether a causal fwd + bwd runs on this GPU under each pinned list and under
    each single kernel. A dtype that ran at base and is refused at head is a regression."""
    import inspect
    import torch
    import torch.nn.functional as F
    from torch.nn.attention import sdpa_kernel
    import unsloth.context_parallel as cp
    versions()
    f = cp._ring_sdpa_backends
    takes_flag = "efficient" in inspect.signature(f).parameters
    lists = {}
    for dt in ("float16", "bfloat16", "float32"):
        lists[dt] = (f(dt == "float32") if takes_flag else f()) or []
    res["ring_backends"] = {k: [b.name for b in v] for k, v in lists.items()}

    def run(backends, dtype):
        g = torch.Generator(device = "cuda").manual_seed(0)
        q, k, v = (torch.randn(1, 4, 128, 64, device = "cuda", dtype = dtype, generator = g).requires_grad_() for _ in range(3))
        try:
            with sdpa_kernel(backends):
                o = F.scaled_dot_product_attention(q, k, v, is_causal = True)
                o.float().sum().backward()
            torch.cuda.synchronize()
            return "ok" if torch.isfinite(o).all() and torch.isfinite(q.grad).all() else "nonfinite"
        except Exception as e:  # noqa: BLE001
            return f"refused: {type(e).__name__}: {str(e).splitlines()[0][:90]}"
    pinned, single = {}, {}
    for dt, backends in lists.items():
        dtype = getattr(torch, dt)
        pinned[dt] = run(backends, dtype) if backends else "no list"
        for b in sorted({*backends, *[getattr(torch.nn.attention.SDPBackend, n) for n in ("FLASH_ATTENTION", "EFFICIENT_ATTENTION")]}, key = lambda x: x.name):
            single[f"{dt}:{b.name}"] = run([b], dtype)
    res["ring_pinned_runs"] = pinned
    res["single_kernel_runs"] = single
    res["finite"], res["losses"], res["decreasing"] = True, [], None


def cell_aol(key, collator):
    """unsloth #17942 / zoo #2554: vision SFT with assistant_only_loss=True on a 2-turn image chat. collator = "unsloth"
    (UnslothVisionDataCollator, inline image) or "trl" (TRL's own collator, images column). Records the decoded kept
    labels, whether only assistant text is kept, image tokens among kept labels, then 3 training steps."""
    import tempfile
    import torch
    from PIL import Image
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastVisionModel
    versions()
    stage("load")
    model, proc = FastVisionModel.from_pretrained(MODELS[key], max_seq_length = 512, dtype = torch.bfloat16, load_in_4bit = False)
    model = FastVisionModel.get_peft_model(model, finetune_vision_layers = False, finetune_language_layers = True,
                                           finetune_attention_modules = True, finetune_mlp_modules = True, r = 8,
                                           lora_alpha = 16, lora_dropout = 0.0, bias = "none", random_state = 0,
                                           use_gradient_checkpointing = "unsloth")
    FastVisionModel.for_training(model)
    image = Image.new("RGB", (32, 32), (255, 0, 0))
    inline = collator == "unsloth"
    part = {"type": "image", "image": image} if inline else {"type": "image"}
    msgs = [{"role": "user", "content": [part, {"type": "text", "text": "What is in this picture?"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "A red square."}]},
            {"role": "user", "content": [{"type": "text", "text": "What color is it?"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "It is red."}]}]
    rows = [({"messages": msgs} if inline else {"messages": msgs, "images": [image]}) for _ in range(6)]
    kw = dict(output_dir = tempfile.mkdtemp(), per_device_train_batch_size = 2, max_steps = 3, learning_rate = 1e-3,
              logging_steps = 1, report_to = "none", save_strategy = "no", bf16 = True, seed = 0, max_length = None,
              assistant_only_loss = True, remove_unused_columns = False)
    if inline:
        kw.update(dataset_text_field = "", dataset_kwargs = {"skip_prepare_dataset": True})
    from datasets import Dataset
    ds = rows if inline else Dataset.from_list(rows)
    extra = {}
    if inline:
        from unsloth.trainer import UnslothVisionDataCollator
        extra["data_collator"] = UnslothVisionDataCollator(model, proc)
    stage("trainer")
    trainer = SFTTrainer(model = model, processing_class = proc, args = SFTConfig(**kw), train_dataset = ds, **extra)
    res["trainer_class"] = type(trainer).__name__
    res["collator_class"] = type(trainer.data_collator).__name__
    res["args_assistant_only_loss"] = getattr(trainer.args, "assistant_only_loss", None)
    batch = next(iter(trainer.get_train_dataloader()))
    ids, labels = batch["input_ids"][0].cpu(), batch["labels"][0].cpu()
    tok = getattr(proc, "tokenizer", proc)
    kept = tok.decode(ids[labels != -100])
    res["kept_labels_text"] = kept[:400]
    res["kept_tokens"] = int((labels != -100).sum())
    res["total_tokens"] = int(ids.numel())
    img_ids = {i for i in (getattr(proc, "image_token_id", None), getattr(getattr(model, "config", None), "image_token_id", None),
                           getattr(getattr(model, "config", None), "image_token_index", None)) if isinstance(i, int)}
    res["kept_image_tokens"] = int(sum(int((ids[labels != -100] == i).sum()) for i in img_ids))
    res["assistant_only"] = ("A red square." in kept and "It is red." in kept and "What color is it?" not in kept
                             and "What is in this picture?" not in kept)
    stage("train")
    torch.cuda.reset_peak_memory_stats()
    trainer.train()
    peak()
    finish_losses([float(h["loss"]) for h in trainer.state.log_history if "loss" in h])
    res["decreasing"] = None


def cell_sft_eos():
    """zoo #2558: SFT text rows (raw text / text + EOS_TOKEN / chat-template-formatted) through Unsloth's SFTTrainer
    (packing off): the tokenized tail of row 0 and how many EOS it ends with, then 2 training steps per mode."""
    import tempfile
    import torch
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastLanguageModel
    versions()
    stage("load")
    model, tok = FastLanguageModel.from_pretrained(MODELS["sft_eos"], max_seq_length = 256, dtype = torch.bfloat16,
                                                   load_in_4bit = False)
    model = FastLanguageModel.get_peft_model(model, r = 8, lora_alpha = 16, lora_dropout = 0.0, bias = "none",
                                             target_modules = TARGETS, use_gradient_checkpointing = "unsloth",
                                             random_state = 0)
    tt = getattr(tok, "tokenizer", tok)
    if tt.pad_token is None:
        tt.pad_token = tt.eos_token
    eos = tt.eos_token
    chat = [tt.apply_chat_template([{"role": "user", "content": d[:30]}, {"role": "assistant", "content": d[30:]}],
                                   tokenize = False) for d in DOCS]
    modes = {"raw": list(DOCS), "raw_plus_eos": [d + eos for d in DOCS], "chat_template": chat}
    out, all_losses = {}, []
    for name, texts in modes.items():
        stage(f"mode_{name}")
        cfg = SFTConfig(output_dir = tempfile.mkdtemp(), per_device_train_batch_size = 2, max_steps = 2, learning_rate = 1e-3,
                        logging_steps = 1, report_to = "none", save_strategy = "no", bf16 = True, seed = 0,
                        dataset_text_field = "text", max_length = 128, packing = False, dataset_num_proc = 1)
        tr = SFTTrainer(model = model, args = cfg, train_dataset = Dataset.from_dict({"text": texts * 4}), processing_class = tt)
        ids = list(tr.train_dataset[0]["input_ids"])
        n_eos = 0
        for t in reversed(ids):
            if t == tt.eos_token_id:
                n_eos += 1
            else:
                break
        tr.train()
        losses = [float(h["loss"]) for h in tr.state.log_history if "loss" in h]
        all_losses += losses
        out[name] = {"tail_ids": ids[-4:], "tail_text": tt.decode(ids[-4:]), "trailing_eos": n_eos, "len": len(ids),
                     "losses": losses}
    res["eos_modes"] = out
    res["eos_counts"] = {k: v["trailing_eos"] for k, v in out.items()}
    finish_losses(all_losses)
    res["decreasing"] = None


def cell_dec_mistral():
    """unsloth #17964: tiny Mistral through FastLanguageModel, for_inference, greedy generate 32 tokens at batch 1 and
    batch 2, default env and UNSLOTH_DECODE_CUDA_GRAPHS=1 forced; tokens must match base."""
    import torch
    from unsloth import FastLanguageModel
    versions()
    stage("load")
    torch.manual_seed(0)
    model, tok = FastLanguageModel.from_pretrained(MODELS["dec_mistral"], max_seq_length = 256, dtype = torch.bfloat16,
                                                   load_in_4bit = False)
    FastLanguageModel.for_inference(model)
    tt = getattr(tok, "tokenizer", tok)
    if tt.pad_token is None:
        tt.pad_token = tt.eos_token
    tt.padding_side = "left"
    try:
        import unsloth.models._generic_decode_graphs as GG
        res["graph_backend_available"] = safe(lambda: GG._BACKEND.available())
    except Exception as e:  # noqa: BLE001
        res["graph_backend_available"] = f"absent: {type(e).__name__}"
    runs = {}
    stage("generate")
    for tag, env in (("default", None), ("forced", "1")):
        if env is None:
            os.environ.pop("UNSLOTH_DECODE_CUDA_GRAPHS", None)
        else:
            os.environ["UNSLOTH_DECODE_CUDA_GRAPHS"] = env
        toks = []
        for bs in (1, 2):
            enc = tt(DOCS[:bs], return_tensors = "pt", padding = True).to(model.device)
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens = 32, min_new_tokens = 32, do_sample = False,
                                     pad_token_id = tt.pad_token_id)
            toks.append(out[:, enc["input_ids"].shape[1]:].tolist())
        runs[tag] = {"tokens": toks}
    os.environ.pop("UNSLOTH_DECODE_CUDA_GRAPHS", None)
    res["decode_runs"] = runs
    res["tokens"] = [r[:8] for r in runs["default"]["tokens"][0]]
    res["decode_forced_eq_default"] = runs["forced"]["tokens"] == runs["default"]["tokens"]
    res["finite"], res["losses"], res["decreasing"] = True, [], None


def cell_qwen38_long():
    """zoo #2565: tiny Qwen3.8 (qwen4_exp) LoRA on packed rows of 2052 tokens (two documents per row, position_ids
    reset per document), so the QSA indexer path runs; records QWEN4_EXP_INDEXER_STATS / QWEN4_EXP_SPARSE_STATS deltas
    (sparse kernel vs dense indexer), 3 steps."""
    import torch
    from unsloth import FastModel
    versions()
    stage("load")
    torch.manual_seed(0)
    model, tok = FastModel.from_pretrained(MODELS["qwen38"], max_seq_length = 4096, dtype = torch.bfloat16, load_in_4bit = False)
    model = FastModel.get_peft_model(model, r = 8, lora_alpha = 16, lora_dropout = 0.0, bias = "none",
                                     finetune_vision_layers = False, finetune_language_layers = True,
                                     finetune_attention_modules = True, finetune_mlp_modules = True,
                                     use_gradient_checkpointing = "unsloth", random_state = 0)
    import unsloth_zoo.temporary_patches.qwen4_exp as q
    stats = {k: dict(getattr(q, k)) for k in ("QWEN4_EXP_INDEXER_STATS", "QWEN4_EXP_SPARSE_STATS") if hasattr(q, k)}
    res["stats_before"] = stats
    dev = model.get_input_embeddings().weight.device
    vocab = model.get_input_embeddings().weight.shape[0]
    g = torch.Generator().manual_seed(0)
    lens = [1100, 952]
    ids = torch.randint(10, min(vocab, 1000), (1, sum(lens)), generator = g).to(dev)
    pos = torch.cat([torch.arange(n) for n in lens])[None].to(dev)
    batch = {"input_ids": ids, "position_ids": pos, "labels": ids.clone()}
    res["row_tokens"] = int(ids.shape[1])
    stage("train")
    torch.cuda.reset_peak_memory_stats()
    manual_loop(model, batch, 3)
    peak()
    after = {k: dict(getattr(q, k)) for k in ("QWEN4_EXP_INDEXER_STATS", "QWEN4_EXP_SPARSE_STATS") if hasattr(q, k)}
    res["stats_delta"] = {k: {kk: after[k].get(kk, 0) - stats.get(k, {}).get(kk, 0) for kk in after[k]} for k in after}
    res["indexer_vectorised"] = (res["stats_delta"].get("QWEN4_EXP_INDEXER_STATS") or {}).get("vectorised", "absent")
    res["decreasing"] = None


def cell_seqcls():
    """unsloth #17861: tiny Qwen3 sequence classifier LoRA, 5 steps, single process; the DDP find_unused default the
    head computes for it (absent at the base)."""
    import torch
    import transformers
    from unsloth import FastLanguageModel
    versions()
    stage("load")
    from unsloth import FastModel
    torch.manual_seed(0)  # the new `score` head is randomly initialised: same weights at every state
    model, tok = FastModel.from_pretrained(MODELS["seqcls"], max_seq_length = 256, dtype = torch.bfloat16,
                                           load_in_4bit = False, num_labels = 2,
                                           auto_model = transformers.AutoModelForSequenceClassification)
    res["loader"] = "FastModel(AutoModelForSequenceClassification)"
    res["model_class"] = type(model).__name__
    if "SequenceClassification" not in res["model_class"]:
        raise RuntimeError(f"expected a sequence classifier, loaded {res['model_class']}")
    tt = getattr(tok, "tokenizer", tok)
    if tt.pad_token is None:
        tt.pad_token = tt.eos_token
    model.config.pad_token_id = tt.pad_token_id
    model = FastLanguageModel.get_peft_model(model, r = 16, lora_alpha = 32, lora_dropout = 0.0, bias = "none",
                                             target_modules = TARGETS, use_gradient_checkpointing = "unsloth",
                                             random_state = 0)
    lora_stats(model)
    try:
        import tempfile
        import unsloth.models._ddp_defaults as ddd
        targs = transformers.TrainingArguments(output_dir = tempfile.mkdtemp(), gradient_checkpointing = True,
                                               ddp_find_unused_parameters = None, report_to = "none")
        orig = ddd._is_distributed
        ddd._is_distributed = lambda *a, **k: True  # decide as a world>1 run would; single process here
        try:
            ret = ddd.unsloth_default_ddp_find_unused_parameters(targs, model)
        finally:
            ddd._is_distributed = orig
        res["ddp_find_unused_default"] = f"{targs.ddp_find_unused_parameters}"
        res["ddp_default_ret"] = str(ret)[:80]
    except Exception as e:  # noqa: BLE001
        res["ddp_find_unused_default"] = f"ERR {type(e).__name__}: {e}"[:200]
    enc = tt(DOCS, return_tensors = "pt", padding = True)
    dev = model.get_input_embeddings().weight.device
    batch = {"input_ids": enc["input_ids"].to(dev), "attention_mask": enc["attention_mask"].to(dev),
             "labels": torch.tensor([0, 1], device = dev)}
    stage("train")
    torch.cuda.reset_peak_memory_stats()
    manual_loop(model, batch, args.steps)
    peak()


def cell_tok_save():
    """unsloth #17862: single-process LoRA save + reload (adapter and tokenizer), the path the DDP change sits on:
    reloaded model gives the same loss, tokenizer files round-trip."""
    import torch
    from unsloth import FastLanguageModel
    versions()
    stage("load")
    model, tok = FastLanguageModel.from_pretrained(built_llama(), max_seq_length = 256, dtype = torch.bfloat16,
                                                   load_in_4bit = False)
    model = FastLanguageModel.get_peft_model(model, r = 16, lora_alpha = 32, lora_dropout = 0.0, bias = "none",
                                             target_modules = TARGETS, use_gradient_checkpointing = "unsloth",
                                             random_state = 0)
    batch = text_batch(tok, model.get_input_embeddings().weight.device)
    stage("train")
    manual_loop(model, batch, args.steps)
    model.eval()
    with torch.no_grad():
        before = float(model(**batch).loss)
    d = os.path.join(os.path.dirname(os.path.abspath(args.out)), "tok_save_adapter")
    stage("save")
    model.save_pretrained(d)
    tok.save_pretrained(d)
    res["saved_files"] = sorted(os.listdir(d))
    del model
    stage("reload")
    model2, tok2 = FastLanguageModel.from_pretrained(d, max_seq_length = 256, dtype = torch.bfloat16, load_in_4bit = False)
    FastLanguageModel.for_training(model2)
    model2.eval()
    with torch.no_grad():
        after = float(model2(**batch).loss)
    res["loss_before_save"], res["loss_after_reload"] = before, after
    res["reload_loss_rel"] = abs(after - before) / max(abs(before), 1e-12)
    probe_text = DOCS[0]
    res["tokenizer_roundtrip"] = getattr(tok2, "tokenizer", tok2)(probe_text)["input_ids"] == \
        getattr(tok, "tokenizer", tok)(probe_text)["input_ids"]
    res["reload_ok"] = bool(res["reload_loss_rel"] < 1e-2 and res["tokenizer_roundtrip"])


def cell_fsdp2_io():
    """unsloth #18028: native FSDP2 on the GPU (one process, world 1) with a tiny untied Qwen3 whose embed / lm_head are
    frozen, UNSLOTH_FSDP2_REPLICATE_IO_TABLES=1. Records whether the CUDA-only branch engages on this device (embedding
    kept on CPU, FSDPModule._set_unshard_async_op(True) called), and 3 SGD steps of loss vs an unsharded reference."""
    import contextlib
    import datetime
    import io
    import socket
    import torch
    import torch.distributed as dist
    import transformers
    versions()
    stage("load")
    os.environ.update(UNSLOTH_FSDP2 = "1", UNSLOTH_FSDP2_REPLICATE_IO_TABLES = "1")
    with socket.socket() as sk:
        sk.bind(("127.0.0.1", 0))
        port = sk.getsockname()[1]
    os.environ.update(MASTER_ADDR = "127.0.0.1", MASTER_PORT = str(port), RANK = "0", WORLD_SIZE = "1", LOCAL_RANK = "0")
    backend = "cuda:nccl,cpu:gloo" if (dist.is_nccl_available() and os.name != "nt") else "gloo"
    res["backend"] = backend
    stage("pg_init")
    if os.name == "nt":  # TCPStore + gloo on this Windows host crashed in c10d (0xC0000005): a FileStore rendezvous
        import tempfile
        from pathlib import Path
        store_file = Path(tempfile.mkdtemp()) / "pg_store"
        res["init_method"] = "file"
        dist.init_process_group(backend, init_method = store_file.as_uri(), rank = 0, world_size = 1,
                                timeout = datetime.timedelta(seconds = 300))
    else:
        dist.init_process_group(backend, rank = 0, world_size = 1, timeout = datetime.timedelta(seconds = 300))
    stage("pg_ok")
    try:
        from unsloth.models import fsdp2_native as native
        from torch.distributed.fsdp import FSDPModule
        calls = []
        orig = getattr(FSDPModule, "_set_unshard_async_op", None)
        res["has_set_unshard_async_op"] = orig is not None
        if orig is not None:
            def spy(self, *a, **k):
                calls.append(a[0] if a else k)
                return orig(self, *a, **k)
            FSDPModule._set_unshard_async_op = spy

        def make():
            config = transformers.Qwen3Config(vocab_size = 1024, hidden_size = 32, intermediate_size = 64, num_hidden_layers = 3,
                                              num_attention_heads = 4, num_key_value_heads = 2, head_dim = 8, tie_word_embeddings = False)
            torch.manual_seed(0)
            m = transformers.Qwen3ForCausalLM(config).float()
            for name, prm in m.named_parameters():
                prm.requires_grad_(".mlp." in name)
            return m
        dev = torch.device("cuda", torch.cuda.current_device())
        ref, model = make().to(dev), make().to(dev)
        stage("shard")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            native.shard_model(model, native.FSDP2Config(param_dtype = torch.float32, device = "cuda", fit_layout = False))
        res["shard_msg"] = [l for l in buf.getvalue().splitlines() if "embedding" in l][:2]
        from torch.distributed.tensor import DTensor
        embed, head = model.get_input_embeddings(), model.get_output_embeddings()
        res["embed_device"] = str(embed.weight.device)
        res["head_device"] = str(head.weight.device)
        res["embed_dtensor"] = isinstance(embed.weight, DTensor)
        res["head_dtensor"] = isinstance(head.weight, DTensor)
        res["unshard_async_calls"] = calls
        res["embed_on_cpu"] = embed.weight.device.type == "cpu"
        stage("train")
        opt = torch.optim.SGD([q for q in model.parameters() if q.requires_grad], lr = 0.5)
        ropt = torch.optim.SGD([q for q in ref.parameters() if q.requires_grad], lr = 0.5)
        losses, rlosses = [], []
        torch.manual_seed(1)
        for _ in range(3):
            ids = torch.randint(0, 1024, (2, 16), device = dev)
            loss = model(input_ids = ids, labels = ids, use_cache = False).loss
            loss.backward(); opt.step(); opt.zero_grad(set_to_none = True)
            rloss = ref(input_ids = ids, labels = ids, use_cache = False).loss
            rloss.backward(); ropt.step(); ropt.zero_grad(set_to_none = True)
            losses.append(float(loss)); rlosses.append(float(rloss))
        res["losses"], res["ref_losses"] = losses, rlosses
        res["max_loss_diff_vs_unsharded"] = max(abs(a - b) for a, b in zip(losses, rlosses))
        res["finite"] = all(math.isfinite(x) for x in losses)
        res["peak_alloc_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
    finally:
        dist.destroy_process_group()


def _pg_world1():
    """One-process torch.distributed group for GPU FSDP2 cells (RCCL on Linux; gloo + FileStore on Windows)."""
    import datetime
    import socket
    import torch.distributed as dist
    with socket.socket() as sk:
        sk.bind(("127.0.0.1", 0))
        port = sk.getsockname()[1]
    os.environ.update(MASTER_ADDR = "127.0.0.1", MASTER_PORT = str(port), RANK = "0", WORLD_SIZE = "1", LOCAL_RANK = "0")
    backend = "cuda:nccl,cpu:gloo" if (dist.is_nccl_available() and os.name != "nt") else "gloo"
    res["backend"] = backend
    stage("pg_init")
    if os.name == "nt":
        import tempfile
        from pathlib import Path
        res["init_method"] = "file"
        dist.init_process_group(backend, init_method = (Path(tempfile.mkdtemp()) / "pg_store").as_uri(), rank = 0,
                                world_size = 1, timeout = datetime.timedelta(seconds = 300))
    else:
        dist.init_process_group(backend, rank = 0, world_size = 1, timeout = datetime.timedelta(seconds = 300))
    stage("pg_ok")


def cell_fsdp2_gen():
    """unsloth #18038: native FSDP2 (one process, GPU) on a tiny untied Qwen3, then greedy generate under
    UNSLOTH_FSDP2_GENERATE_GATHER=all and =block vs the unsharded model. Records what _generate_streams_blocks decided
    per mode (absent at the base), so the block-streaming path is shown to run or not on this device."""
    import torch
    import torch.distributed as dist
    import transformers
    versions()
    stage("load")
    os.environ["UNSLOTH_FSDP2"] = "1"
    _pg_world1()
    try:
        from unsloth.models import fsdp2_native as native
        decided = []
        orig = getattr(native, "_generate_streams_blocks", None)
        res["has_generate_streams_blocks"] = orig is not None
        if orig is not None:
            def spy(*a, **k):
                out = orig(*a, **k)
                decided.append(bool(out))
                return out
            native._generate_streams_blocks = spy

        def make():
            config = transformers.Qwen3Config(vocab_size = 1024, hidden_size = 32, intermediate_size = 64, num_hidden_layers = 3,
                                              num_attention_heads = 4, num_key_value_heads = 2, head_dim = 8, tie_word_embeddings = False)
            torch.manual_seed(0)
            m = transformers.Qwen3ForCausalLM(config).float()
            for name, prm in m.named_parameters():
                prm.requires_grad_(".mlp." in name)
            return m
        dev = torch.device("cuda", torch.cuda.current_device())
        ref, model = make().to(dev), make().to(dev)
        stage("shard")
        native.shard_model(model, native.FSDP2Config(param_dtype = torch.float32, device = "cuda", fit_layout = False))
        stage("generate")
        torch.manual_seed(1)
        ids = torch.randint(0, 1024, (2, 8), device = dev)
        kw = dict(max_new_tokens = 12, do_sample = False, pad_token_id = 0, eos_token_id = None)
        with torch.no_grad():
            want = ref.generate(input_ids = ids, **kw)[:, 8:].tolist()
        res["ref_tokens"] = want
        per = {}
        for mode in ("all", "block"):
            os.environ["UNSLOTH_FSDP2_GENERATE_GATHER"] = mode
            n0 = len(decided)
            try:
                got = model.generate(input_ids = ids, **kw)[:, 8:].tolist()
                per[mode] = {"tokens_equal": got == want, "streamed": decided[n0:]}
            except Exception as e:  # noqa: BLE001
                per[mode] = {"error": f"{type(e).__name__}: {e}"[:400]}
        os.environ.pop("UNSLOTH_FSDP2_GENERATE_GATHER", None)
        res["gen_modes"] = per
        stage("train")  # the model still trains after generate (params resharded)
        opt = torch.optim.SGD([q for q in model.parameters() if q.requires_grad], lr = 0.5)
        losses = []
        for _ in range(3):
            x = torch.randint(0, 1024, (2, 16), device = dev)
            loss = model(input_ids = x, labels = x, use_cache = False).loss
            loss.backward(); opt.step(); opt.zero_grad(set_to_none = True)
            losses.append(float(loss))
        res["losses"] = losses
        res["finite"] = all(math.isfinite(v) for v in losses) and all("error" not in v for v in per.values())
        res["peak_alloc_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
    finally:
        dist.destroy_process_group()


_PAUTO_WORKER = r"""
import json, math, os, sys
import torch
import torch.distributed as dist
out_path, model_dir = sys.argv[1], sys.argv[2]
rank = int(os.environ["RANK"])
os.environ["LOCAL_RANK"] = "0"            # two ranks share the one card
torch.cuda.set_device(0)
dist.init_process_group("gloo")          # RCCL refuses two ranks on one GPU; Unsloth / accelerate reuse this group
rec = {"rank": rank, "world": dist.get_world_size()}
try:
    import unsloth  # noqa: F401
    from unsloth import FastLanguageModel
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer
    model, tok = FastLanguageModel.from_pretrained(model_dir, max_seq_length = 256, dtype = torch.bfloat16, load_in_4bit = False)
    rec["env_after_load"] = {k: v for k, v in os.environ.items() if k.startswith(("UNSLOTH_PARALLEL", "UNSLOTH_FSDP2"))}
    model = FastLanguageModel.get_peft_model(model, r = 8, lora_alpha = 16, lora_dropout = 0.0, bias = "none",
                                             target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"], random_state = 0)
    tt = getattr(tok, "tokenizer", tok)
    if tt.pad_token is None:
        tt.pad_token = tt.eos_token
    docs = ["the river runs north past the old mill and the green house by the water.",
            "a small cat sat on the warm stone wall and watched the birds fly over the field."]
    ds = Dataset.from_dict({"text": [d + (tt.eos_token or "") for d in docs] * 16})
    cfg = SFTConfig(output_dir = os.path.join(os.path.dirname(out_path), f"pauto_trainer_{rank}"), per_device_train_batch_size = 2,
                    max_steps = 5, learning_rate = 2e-4, logging_steps = 1, bf16 = True, report_to = "none", seed = 0,
                    dataset_text_field = "text", max_length = 64, save_strategy = "no", dataset_num_proc = 1,
                    lr_scheduler_type = "constant", packing = False, ddp_find_unused_parameters = False)
    trainer = SFTTrainer(model = model, train_dataset = ds, processing_class = tt, args = cfg)
    trainer.train()
    rec["wrapped"] = type(trainer.model_wrapped).__name__
    rec["losses"] = [float(h["loss"]) for h in trainer.state.log_history if "loss" in h]
    rec["ok"] = bool(rec["losses"]) and all(math.isfinite(x) for x in rec["losses"])
except Exception as e:
    import traceback
    rec["error"] = f"{type(e).__name__}: {e}"[:1500]
    rec["traceback"] = traceback.format_exc()[-3000:]
with open(f"{out_path}.rank{rank}.json", "w", encoding = "utf-8") as fh:
    json.dump(rec, fh, default = str)
dist.destroy_process_group()
"""


def cell_pauto():
    """unsloth #18061: torchrun 2 ranks on the one GPU (gloo group, both ranks on device 0) with UNSLOTH_PARALLEL=auto,
    tiny Llama LoRA SFT 5 steps. Records auto's decision line (rank 0 stdout), the resolved env, the Trainer wrapper and
    both ranks' losses. Base has no `auto` value."""
    import subprocess
    versions()
    stage("load")
    model_dir = built_llama()
    work = os.path.join(os.path.dirname(os.path.abspath(args.out)), "pauto")
    os.makedirs(work, exist_ok = True)
    worker = os.path.join(work, "pauto_worker.py")
    with open(worker, "w", encoding = "utf-8") as fh:
        fh.write(_PAUTO_WORKER)
    rec_base = os.path.join(work, "rec")
    env = dict(os.environ, UNSLOTH_PARALLEL = "auto", OMP_NUM_THREADS = "2")
    for k in ("UNSLOTH_FSDP2", "UNSLOTH_PARALLEL_AUTO_RESOLVED"):
        env.pop(k, None)
    stage("train")
    try:
        p = subprocess.run([sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node", "2", worker,
                            rec_base, model_dir], env = env, capture_output = True, text = True, encoding = "utf-8",
                           errors = "replace", timeout = 1200)
        rc, out = p.returncode, (p.stdout or "") + "\n--- stderr ---\n" + (p.stderr or "")
    except subprocess.TimeoutExpired as e:
        rc, out = "timeout", str(e.stdout or "")[-4000:]
    with open(os.path.join(work, "torchrun.log"), "w", encoding = "utf-8") as fh:
        fh.write(out)
    res["torchrun_rc"] = rc
    res["decision_line"] = [ln.strip() for ln in out.splitlines() if "UNSLOTH_PARALLEL=auto picked" in ln][:2]
    ranks = []
    for r in (0, 1):
        f = f"{rec_base}.rank{r}.json"
        if os.path.exists(f):
            with open(f, encoding = "utf-8") as fh:
                ranks.append(json.load(fh))
    res["ranks"] = [{k: v for k, v in r.items() if k != "traceback"} for r in ranks]
    if ranks and ranks[0].get("losses"):
        finish_losses(ranks[0]["losses"])
    res["finite"] = rc == 0 and len(ranks) == 2 and all(r.get("ok") for r in ranks)
    if not res["finite"]:
        res["torchrun_tail"] = out[-2500:]
        res["rank_errors"] = [r.get("error") for r in ranks]
        res["rank_tracebacks"] = [r.get("traceback", "")[-1200:] for r in ranks]


def cell_tphost():
    """unsloth #18068: tensor parallel tp=1 on the GPU (one process) over a tiny untied Qwen3 + PEFT LoRA (frozen input
    embedding), apply_tensor_parallel with UNSLOTH_TP_EMBED_ON_CPU=1 vs =0: whether the host-embedding hooks engage on
    this device, and 2 SGD steps' losses / first-step grads on vs off (they should be equal)."""
    import contextlib
    import io
    import torch
    import torch.distributed as dist
    import transformers
    from peft import LoraConfig, get_peft_model
    versions()
    stage("load")
    _pg_world1()
    try:
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.tensor import DTensor
        from unsloth.models import tp_native as tpn
        res["has_host_input_embedding"] = hasattr(tpn, "_host_input_embedding")
        mesh = init_device_mesh("cuda", (1,), mesh_dim_names = ("tp",))
        dev = torch.device("cuda", torch.cuda.current_device())

        def make():
            config = transformers.Qwen3Config(vocab_size = 1024, hidden_size = 32, intermediate_size = 64, num_hidden_layers = 2,
                                              num_attention_heads = 4, num_key_value_heads = 2, head_dim = 8, tie_word_embeddings = False)
            config._attn_implementation = "eager"
            torch.manual_seed(0)
            m = transformers.Qwen3ForCausalLM(config).float()
            torch.manual_seed(1)
            lora = LoraConfig(r = 4, lora_alpha = 8, target_modules = "all-linear", init_lora_weights = False)
            return get_peft_model(m, lora).to(dev)

        arms = {}
        for flag in ("0", "1"):
            os.environ["UNSLOTH_TP_EMBED_ON_CPU"] = flag
            stage(f"apply_{flag}")
            model = make()
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                tpn.apply_tensor_parallel(model, mesh["tp"])
            embed = model.get_input_embeddings()
            rec = {"embed_device": str(embed.weight.device),
                   "offload_hooks": bool(getattr(embed, "_unsloth_offload_hooks_installed", False)),
                   "cpu_lookup": bool(getattr(embed, "_unsloth_fsdp2_cpu_lookup", False)),
                   "msg": [ln for ln in buf.getvalue().splitlines() if "input embedding" in ln][:1]}
            params = [p for p in model.parameters() if p.requires_grad]
            opt = torch.optim.SGD(params, lr = 0.5)
            torch.manual_seed(2)
            ids = torch.randint(0, 1024, (2, 16), device = dev)
            losses, grads = [], {}
            for step in range(2):
                out = model(input_ids = ids, labels = ids, use_cache = False)
                out.loss.backward()
                if step == 0:
                    for n, p in model.named_parameters():
                        if p.grad is not None:
                            g = p.grad.full_tensor() if isinstance(p.grad, DTensor) else p.grad
                            grads[n] = g.detach().float().cpu()
                opt.step()
                opt.zero_grad(set_to_none = True)
                losses.append(float(out.loss))
            rec["losses"] = losses
            arms[flag] = (rec, grads)
        os.environ.pop("UNSLOTH_TP_EMBED_ON_CPU", None)
        res["arms"] = {k: v[0] for k, v in arms.items()}
        g0, g1 = arms["0"][1], arms["1"][1]
        res["grad_names_equal"] = sorted(g0) == sorted(g1)
        res["max_grad_diff_on_vs_off"] = max((float((g0[n] - g1[n]).abs().max()) for n in g0 if n in g1), default = None)
        res["max_loss_diff_on_vs_off"] = max(abs(a - b) for a, b in zip(arms["0"][0]["losses"], arms["1"][0]["losses"]))
        res["engaged_on"] = arms["1"][0]["embed_device"].startswith("cpu")
        finish_losses(arms["1"][0]["losses"])
    finally:
        dist.destroy_process_group()


def cell_atree():
    """unsloth #18070 (PR author's scripts/gpu_async_tree_check.py): a TRL >= 1.17 prefix-trie packed row through a real
    Unsloth fast backbone (tiny Qwen3 LoRA, fp32) + TRL's fused head on the GPU; packed log-probs vs each sample alone.
    PASS = this state's arm within 2e-3 of unpacked and the TRL-only arm visibly off. Expected FAIL at the base."""
    import subprocess
    import unsloth as _u
    versions()
    stage("train")
    tree = os.path.dirname(os.path.dirname(os.path.abspath(_u.__file__)))
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gpu_async_tree_check.py")
    env = dict(os.environ, UNSLOTH_COMPILE_LOCATION = os.path.join(os.path.dirname(os.path.abspath(args.out)), "uc_atree"))
    try:
        p = subprocess.run([sys.executable, script, tree], env = env, capture_output = True, text = True, encoding = "utf-8",
                           errors = "replace", timeout = 1200)
        rc, out = p.returncode, (p.stdout or "") + "\n--- stderr ---\n" + (p.stderr or "")
    except subprocess.TimeoutExpired as e:
        rc, out = "timeout", str(e.stdout or "")
    res["atree_rc"] = rc
    keep = ("trl ", "sentinel", "fast backbone", "unique tokens", "HEAD arm", "BASE arm", "VERDICT")
    res["atree_lines"] = [ln.strip() for ln in out.splitlines() if ln.startswith(keep)]
    res["atree_verdict"] = next((ln.split()[-1] for ln in res["atree_lines"] if ln.startswith("VERDICT")), None)
    if rc not in (0, 1):
        res["atree_tail"] = out[-2500:]
    res["finite"] = res["atree_verdict"] == "PASS"


def cell_grpo_pg():
    """unsloth #18095: single-GPU text GRPO without vLLM (tiny Qwen3 LoRA, 2 steps) on a long shared prompt with 4
    generations, so PrefixGrouper's token ratio clears its threshold. Counts build_group_layout calls / layouts built and
    reads the PrefixGrouper flags left on the model. The vLLM path the PR changes cannot run here (no vLLM on gfx1151)."""
    import torch
    from unsloth import FastLanguageModel
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer
    import unsloth.utils.prefix_grouper as pg
    versions()
    stage("load")
    calls = {"n": 0, "layouts": 0}
    orig = pg.build_group_layout

    def spy(*a, **k):
        out = orig(*a, **k)
        calls["n"] += 1
        calls["layouts"] += int(out is not None)
        return out
    pg.build_group_layout = spy
    model, tok = FastLanguageModel.from_pretrained(MODELS["grpo_txt"], max_seq_length = 512, dtype = torch.bfloat16,
                                                   load_in_4bit = False)
    model = FastLanguageModel.get_peft_model(model, r = 16, lora_alpha = 32, lora_dropout = 0.0, bias = "none",
                                             target_modules = TARGETS, use_gradient_checkpointing = "unsloth",
                                             random_state = 0)
    long_prompt = " ".join(DOCS * 6)
    ds = Dataset.from_list([{"prompt": [{"role": "user", "content": long_prompt + f" question {i}?"}]} for i in range(8)])

    def reward(completions, **kw):
        return [float(len(str(c))) / 10.0 for c in completions]
    cfg = _cfg(GRPOConfig, output_dir = os.path.join(os.path.dirname(os.path.abspath(args.out)), "trainer_grpo_pg"),
               per_device_train_batch_size = 4, num_generations = 4, max_completion_length = 8, max_prompt_length = None,
               max_steps = 2, learning_rate = 5e-5, logging_steps = 1, report_to = "none", bf16 = True, seed = 0,
               save_strategy = "no", temperature = 1.0, use_vllm = False)
    trainer = GRPOTrainer(model = model, reward_funcs = [reward], args = cfg, train_dataset = ds, processing_class = tok)
    res["trainer_class"] = type(trainer).__name__
    stage("train")
    torch.cuda.reset_peak_memory_stats()
    trainer.train()
    peak()
    res["pg_build_calls"], res["pg_layouts"] = calls["n"], calls["layouts"]
    res["pg_flags"] = {k: str(_find_attr(trainer.model, k)) for k in (
        "_unsloth_prefix_grouper_nograd_disabled", "_unsloth_prefix_grouper_nograd_verified",
        "_unsloth_prefix_grouper_nograd_unsafe")}
    res["pg_enabled"] = safe(lambda: bool(pg.prefix_grouper_enabled(trainer.model)))
    logs = [h for h in trainer.state.log_history if "loss" in h]
    finish_losses([float(h["loss"]) for h in logs])
    res["decreasing"] = None
    res["rewards"] = [h.get("reward") for h in logs]


_FP32_REF = r"""
import json, sys
import torch, transformers
name, ids_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
ids = torch.tensor(json.load(open(ids_path)))[None].cuda()
m = transformers.AutoModelForCausalLM.from_pretrained(name, torch_dtype = torch.float32, attn_implementation = "eager").cuda().eval()
with torch.no_grad():
    lp = torch.log_softmax(m(input_ids = ids, use_cache = False).logits.float()[:, :-1], -1)
    tok = lp.gather(-1, ids[:, 1:, None])[0, :, 0]
json.dump(tok.tolist(), open(out_path, "w"))
"""


def cell_fp32attn():
    """unsloth #18099: tiny Qwen3 loaded with dtype=torch.float32, no autocast: per-token logprobs of one sequence vs
    plain transformers fp32 eager (separate process, no Unsloth patches), and which attention backend / SDPA operand
    dtype actually ran (spies on attention_dispatch._run_attention and F.scaled_dot_product_attention)."""
    import subprocess
    import torch
    from unsloth import FastLanguageModel
    import unsloth.utils.attention_dispatch as ad
    versions()
    stage("load")
    seen = {"run": [], "sdpa": []}
    orig_run = ad._run_attention

    def run_spy(*a, **k):
        cfg, q = k.get("config"), k.get("Q")
        seen["run"].append((str(getattr(cfg, "backend", None)), str(getattr(q, "dtype", None))))
        return orig_run(*a, **k)
    ad._run_attention = run_spy
    F = torch.nn.functional
    orig_sdpa = F.scaled_dot_product_attention

    def sdpa_spy(q, *a, **k):
        seen["sdpa"].append(str(q.dtype))
        return orig_sdpa(q, *a, **k)
    F.scaled_dot_product_attention = sdpa_spy
    res["has_flash"] = bool(getattr(ad, "HAS_FLASH_ATTENTION", False))
    res["has_xformers"] = bool(getattr(ad, "HAS_XFORMERS", False))
    res["has_keeps_fp32"] = hasattr(ad, "_keeps_fp32_attention")
    name = MODELS["grpo_txt"]
    model, tok = FastLanguageModel.from_pretrained(name, max_seq_length = 256, dtype = torch.float32, load_in_4bit = False)
    model.eval()
    res["param_dtype"] = str(next(model.parameters()).dtype)
    tt = getattr(tok, "tokenizer", tok)
    ids = tt(" ".join(DOCS), return_tensors = "pt").input_ids[:, :48].cuda()
    stage("forward")
    res["autocast_enabled"] = torch.is_autocast_enabled()
    with torch.no_grad():
        lp = torch.log_softmax(model(input_ids = ids, use_cache = False).logits.float()[:, :-1], -1)
        mine = lp.gather(-1, ids[:, 1:, None])[0, :, 0].cpu()
    work = os.path.join(os.path.dirname(os.path.abspath(args.out)), "fp32attn")
    os.makedirs(work, exist_ok = True)
    ids_p, out_p, scr = (os.path.join(work, x) for x in ("ids.json", "ref.json", "ref.py"))
    json.dump(ids[0].tolist(), open(ids_p, "w"))
    open(scr, "w").write(_FP32_REF)
    stage("reference")
    p = subprocess.run([sys.executable, scr, name, ids_p, out_p], capture_output = True, text = True, timeout = 900)
    if p.returncode != 0:
        res["ref_error"] = (p.stderr or "")[-1500:]
    else:
        ref = torch.tensor(json.load(open(out_p)))
        res["max_abs_logprob_diff_vs_hf_fp32_eager"] = float((mine - ref).abs().max())
        res["mean_abs_logprob_diff_vs_hf_fp32_eager"] = float((mine - ref).abs().mean())
    res["run_attention_calls"] = sorted({f"{b}/{d}" for b, d in seen["run"]})
    res["n_run_attention"] = len(seen["run"])
    res["sdpa_q_dtypes"] = sorted(set(seen["sdpa"]))
    res["n_sdpa"] = len(seen["sdpa"])
    res["losses"] = [float(-mine.mean())]
    res["finite"] = bool(torch.isfinite(mine).all()) and "ref_error" not in res


_WARM_WORKER = r"""
import json, os, sys
model_dir, out_path = sys.argv[1], sys.argv[2]
import unsloth  # noqa: F401
from unsloth import FastLanguageModel
import torch
from torch._inductor.runtime import triton_heuristics as th
calls = {"n": 0}
orig = th.CachingAutotuner.benchmark_all_configs
def spy(self, *a, **k):
    calls["n"] += 1
    return orig(self, *a, **k)
th.CachingAutotuner.benchmark_all_configs = spy
model, tok = FastLanguageModel.from_pretrained(model_dir, max_seq_length = 128, dtype = torch.bfloat16, load_in_4bit = False)
model = FastLanguageModel.get_peft_model(model, r = 8, lora_alpha = 16, lora_dropout = 0.0, bias = "none",
                                         target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                                         use_gradient_checkpointing = "unsloth", random_state = 0)
model.train()
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr = 1e-4)
torch.manual_seed(0)
ids = torch.randint(5, 200, (2, 64), device = "cuda")
losses = []
for _ in range(3):
    loss = model(input_ids = ids, labels = ids).loss
    loss.backward(); opt.step(); opt.zero_grad(set_to_none = True)
    losses.append(float(loss))
tag = None
try:
    tag = torch.compiler.config.cache_key_tag
except Exception:
    pass
json.dump({"benchmark_all_configs": calls["n"], "losses": losses, "cache_key_tag": tag,
           "env_tag": os.environ.get("TORCH_COMPILE_CACHE_KEY_TAG")}, open(out_path, "w"))
"""


def cell_warm():
    """unsloth #18111: the same tiny compiled LoRA job (built Llama, 3 steps) run twice in fresh processes sharing one
    TORCHINDUCTOR_CACHE_DIR (and UNSLOTH_COMPILE_LOCATION); counts CachingAutotuner.benchmark_all_configs calls per
    run and records torch.compiler.config.cache_key_tag / TORCH_COMPILE_CACHE_KEY_TAG as each run saw them."""
    import subprocess
    import tempfile
    versions()
    stage("load")
    model_dir = built_llama()
    work = os.path.join(os.path.dirname(os.path.abspath(args.out)), "warm")
    os.makedirs(work, exist_ok = True)
    worker = os.path.join(work, "warm_worker.py")
    open(worker, "w", encoding = "utf-8").write(_WARM_WORKER)
    cache = tempfile.mkdtemp(prefix = "amdci_inductor_")
    env = dict(os.environ, TORCHINDUCTOR_CACHE_DIR = cache, UNSLOTH_COMPILE_LOCATION = os.path.join(cache, "uc"))
    env.pop("TORCH_COMPILE_CACHE_KEY_TAG", None)
    runs = []
    stage("train")
    for i in (1, 2):
        out = os.path.join(work, f"run{i}.json")
        p = subprocess.run([sys.executable, worker, model_dir, out], env = env, capture_output = True, text = True,
                           encoding = "utf-8", errors = "replace", timeout = 1500)
        rec = json.load(open(out)) if (p.returncode == 0 and os.path.exists(out)) else {"rc": p.returncode, "tail": (p.stderr or "")[-1500:]}
        runs.append(rec)
    res["warm_runs"] = runs
    res["autotune_calls"] = [r.get("benchmark_all_configs") for r in runs]
    if all("losses" in r for r in runs):
        finish_losses(runs[1]["losses"])
        res["warm_loss_eq"] = runs[0]["losses"] == runs[1]["losses"]
    else:
        res["finite"] = False


def cell_grpo_pack():
    """unsloth #18120: single-GPU text GRPO without vLLM (tiny Qwen3 LoRA, 3 steps, prompts of mixed length so
    batches hold several rows); records the padding-free packing self-verify state left on the model:
    _unsloth_seq_packing_{grad,nograd}_ok / _strikes, _verify_verdict, and any _nograd_verified_* / _unsafe_* keys."""
    import torch
    from unsloth import FastLanguageModel
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer
    versions()
    stage("load")
    model, tok = FastLanguageModel.from_pretrained(MODELS["grpo_txt"], max_seq_length = 256, dtype = torch.bfloat16,
                                                   load_in_4bit = False)
    model = FastLanguageModel.get_peft_model(model, r = 16, lora_alpha = 32, lora_dropout = 0.0, bias = "none",
                                             target_modules = TARGETS, use_gradient_checkpointing = "unsloth",
                                             random_state = 0)
    prompts = [" ".join(DOCS[: 1 + (i % len(DOCS))]) for i in range(12)]
    ds = Dataset.from_list([{"prompt": [{"role": "user", "content": p}]} for p in prompts])

    def reward(completions, **kw):
        return [float(len(str(c))) / 10.0 for c in completions]
    cfg = _cfg(GRPOConfig, output_dir = os.path.join(os.path.dirname(os.path.abspath(args.out)), "trainer_grpo_pack"),
               per_device_train_batch_size = 4, num_generations = 2, max_completion_length = 8, max_prompt_length = None,
               max_steps = 3, learning_rate = 5e-5, logging_steps = 1, report_to = "none", bf16 = True, seed = 0,
               save_strategy = "no", temperature = 1.0, use_vllm = False)
    trainer = GRPOTrainer(model = model, reward_funcs = [reward], args = cfg, train_dataset = ds, processing_class = tok)
    res["trainer_class"] = type(trainer).__name__
    stage("train")
    torch.cuda.reset_peak_memory_stats()
    trainer.train()
    peak()
    flags = {}
    for m in [trainer.model] + list(trainer.model.modules())[:4]:
        for k, v in vars(m).items():
            if k.startswith("_unsloth_seq_packing_") and k not in flags:
                flags[k] = v if isinstance(v, (bool, int, float, str, type(None))) else repr(v)[:120]
    for k in ("_unsloth_seq_packing_grad_ok", "_unsloth_seq_packing_nograd_ok", "_unsloth_seq_packing_grad_strikes",
              "_unsloth_seq_packing_nograd_strikes", "_unsloth_seq_packing_verify_verdict"):
        flags.setdefault(k, str(_find_attr(trainer.model, k)))
    res["seq_packing_flags"] = flags
    logs = [h for h in trainer.state.log_history if "loss" in h]
    finish_losses([float(h["loss"]) for h in logs])
    res["decreasing"] = None
    res["rewards"] = [h.get("reward") for h in logs]


CELLS = {"hd512_llama": lambda: cell_hd512("llama"), "hd512_gemma4": lambda: cell_hd512("gemma4"),
         "hd512_gemma4_nc": lambda: cell_hd512("gemma4"),
         "bclens": cell_bclens, "bswap": cell_bswap,
         "dec_whisper": cell_dec_whisper, "dec_gemma3n": cell_dec_gemma3n,
         "moeaux_nr": cell_moeaux_nr, "moeaux_gc_qwen3moe": lambda: cell_moeaux_gc("moeaux_gc_qwen3moe"),
         "moeaux_gc_gptoss": lambda: cell_moeaux_gc("moeaux_gc_gptoss"), "gkd": cell_gkd,
         "granite_gc": cell_granite_gc,
         "padmix_qwen3": lambda: cell_padmix("padmix_qwen3"), "padmix_gemma4": lambda: cell_padmix("padmix_gemma4"), "padmix_gemma3": lambda: cell_padmix("padmix_gemma3"),
         "padmix_gemma3_flex": lambda: cell_padmix("padmix_gemma3", "flex_attention"),
         "padmix_gemma3_flex_ones": lambda: cell_padmix("padmix_gemma3", "flex_attention", "ones"),
         "padmix_gemma4_ones": lambda: cell_padmix("padmix_gemma4", None, "ones"),
         "grpo_txt": cell_grpo_txt,
         "dpo_txt": cell_dpo_txt, "qwen38_long": cell_qwen38_long, "dec_mistral": cell_dec_mistral, "sft_eos": cell_sft_eos, "aol_qwen25vl": lambda: cell_aol("aol_qwen25vl", "unsloth"),
         "aol_gemma3_unsloth": lambda: cell_aol("aol_gemma3_unsloth", "unsloth"),
         "aol_gemma3_trl": lambda: cell_aol("aol_gemma3_trl", "trl"),
         "aol_qwen25vl_none": lambda: cell_aol("aol_qwen25vl", "trl"),
         "aol_gemma3_none": lambda: cell_aol("aol_gemma3_unsloth", "trl"), "cp_ring_kernels": cell_cp_ring_kernels, "gkd_proc_qwen35": lambda: cell_gkd_proc("gkd_proc_qwen35"),
         "gkd_proc_gemma3": lambda: cell_gkd_proc("gkd_proc_gemma3"), "mrope_pack": cell_mrope_pack, "orpo_txt": cell_orpo_txt, "lpad": cell_lpad, "fft16": cell_fft16, "fft16_sroff": cell_fft16, "grpo_zadv": lambda: cell_grpo_txt(True), "grpo_zadv_zoff": lambda: cell_grpo_txt(True), "orpo_txt_psoff": cell_orpo_txt, "orpo_txt_unwrap": lambda: cell_orpo_txt(True),
         "vision_gemma3": cell_vision,
         "pfree_llama": lambda: cell_sft_pack("llama", "pfree"), "pfree_qwen3": lambda: cell_sft_pack("qwen3", "pfree"),
         "sftdef_steps": lambda: cell_sft_pack("llama", "steps"), "sftdef_epoch": lambda: cell_sft_pack("llama", "epoch"),
         "plelens": lambda: cell_bclens("plelens", (37, 91, 150, 203, 64, 177, 119), "float32"),
         "seqcls": cell_seqcls, "tok_save": cell_tok_save, "fsdp2_io": cell_fsdp2_io, "fsdp2_gen": cell_fsdp2_gen, "pauto": cell_pauto, "tphost": cell_tphost, "atree": cell_atree, "grpo_pg": cell_grpo_pg, "fp32attn": cell_fp32attn, "warm": cell_warm, "grpo_pack": cell_grpo_pack,
         "gcnr_qwen3": lambda: cell_gc(True), "gcdef_qwen3": lambda: cell_gc(False),
         "gcnr_gemma3": lambda: cell_gc(True), "gcdef_gemma3": lambda: cell_gc(False), "lora_noac": cell_lora_noac,
         "st_minilm": lambda: cell_st(False), "st_qwen3": lambda: cell_st(True),
         "decode_diag": cell_decode_diag, "ctnvfp4": lambda: cell_ct("nvfp4"), "ctmxfp8": lambda: cell_ct("mxfp8"), "mxfp4probe": cell_mxfp4probe,
         "grpo_vl": cell_grpo_vl, "dpo_vl": cell_dpo_vl,
         "gcmem": cell_gcmem, "env": cell_env, "flm16": lambda: cell_flm(False), "flm16_gcsave": lambda: cell_flm(False), "flm4": lambda: cell_flm(True), "vision": cell_vision,
         "moe": cell_moe, "gemma4": cell_gemma4, "qwen38": cell_qwen38, "dsv4": cell_dsv4, "mllama": cell_mllama,
         "rl_decode": cell_rl_decode}


def main():
    if args.cell == "moe":
        os.environ["UNSLOTH_MOE_BACKEND"] = "grouped_mm"
    if args.cell.endswith("_gcsave"):
        os.environ["UNSLOTH_GC_SAVE_ATTENTION"] = "1"  # before any import: zoo #2571 reads it at checkpoint time
    if args.cell.endswith("_nc"):
        os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"  # diagnostic twin: same cell, no Unsloth compile
    if args.cell.endswith("_zoff"):
        os.environ["UNSLOTH_GRPO_SKIP_ZERO_ADVANTAGE"] = "0"  # zoo #2577 kill switch
    if args.cell.endswith("_sroff"):
        os.environ["UNSLOTH_BF16_STOCHASTIC_ROUNDING"] = "0"  # unsloth #18192 kill switch
    if args.cell.endswith("_psoff"):
        os.environ["UNSLOTH_DPO_PREFIX_SHARE"] = "off"  # unsloth #18169 kill switch
    if args.cell == "grpo_vl":
        os.environ["UNSLOTH_GRPO_FUSED_WITHOUT_ENTROPY"] = "1"
    t0 = time.time()
    if args.cell in ("build", "build_ct", "build_hd512"):
        try:
            {"build": built_llama, "build_ct": build_ct, "build_hd512": build_hd512}[args.cell]()  # before any `import unsloth`: plain transformers classes
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
