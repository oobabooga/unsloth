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
    "z12_whisper": "unsloth/whisper-large-v3-turbo (mem_ab worker, bf16 autocast LoRA, 10 steps)",
    "z12_w2v": "facebook/wav2vec2-base-960h (mem_ab worker, bf16 autocast LoRA, 10 steps)",
    "cg5_on": "tiny random Qwen3-MoE (cg5_parity.py, UNSLOTH_GRPO_PREFIX_GROUPER_MOE=1)",
    "cg5_off": "tiny random Qwen3-MoE (cg5_parity.py, flag unset)",
    "cg1_fp32": "unsloth/Qwen3-0.6B (cg1_tree_bench run_bench fp32, --no-perf)",
    "cg1_bf16": "unsloth/Qwen3-0.6B (cg1_tree_bench run_bench bf16, --no-perf)",
    "cg2_qwen3_nbc_ml": "tiny-Qwen3 (as cg2_qwen3_nbc, manual AdamW loop on one batch, no Trainer)",
    "cg2_qwen3_nbc_ml_off": "tiny-Qwen3 (as cg2_qwen3_nbc_ml, UNSLOTH_GC_SKIP_MLP_DOWN=0)",
    "cg2_llama_nbc_ml": "built:llama (bf16, block compile off, manual loop, no Trainer)",
    "cg2_llama4_nbc_ml": "built:llama (NF4, block compile off, manual loop, no Trainer)",
    "cg2_qwen3": "trl-internal-testing/tiny-Qwen3ForCausalLM (LoRA r16 all 7, GC unsloth, block compile default)",
    "cg2_qwen3_nbc": "trl-internal-testing/tiny-Qwen3ForCausalLM (as cg2_qwen3, UNSLOTH_BLOCK_COMPILE=0)",
    "cg2_qwen3_nbc_off": "tiny-Qwen3 (as cg2_qwen3_nbc, UNSLOTH_GC_SKIP_MLP_DOWN=0)",
    "cg2_llama": "built:llama (LoRA r16 all 7 bf16, GC unsloth, block compile default)",
    "cg2_llama_nbc": "built:llama (as cg2_llama, UNSLOTH_BLOCK_COMPILE=0)",
    "cg2_llama4_nbc": "built:llama (bitsandbytes NF4, UNSLOTH_BLOCK_COMPILE=0)",
    "z28_ernie": "built:tiny ERNIE 4.5 MoE (bf16 full FT via prepare_model_for_training, train_layernorms=True)",
    "z28_ernie_nl": "built:tiny ERNIE 4.5 MoE (as z28_ernie, train_layernorms=False)",
    "z28_llama": "built:llama (bf16 full FT via prepare_model_for_training, control)",
    "fx32": "trl-internal-testing/tiny-Gemma3ForConditionalGeneration (FastModel float32, head_dim 256, 600 s cap)",
    "fx16": "trl-internal-testing/tiny-Gemma3ForConditionalGeneration (FastModel bfloat16, head_dim 256, 600 s cap)",
    "cg_forced8": "w20a 8-layer tiny Qwen3 (seed 0, B200 repro model; dpo_cg_repro.py 10 steps, UNSLOTH_BLOCK_CUDAGRAPHS=1)",
    "cg_auto8": "w20a 8-layer tiny Qwen3 (dpo_cg_repro.py 10 steps, block CUDA graphs auto)",
    "cg_forced": "trl-internal-testing/tiny-Qwen3ForCausalLM (w20a dpo_cg_repro.py, DPO ref adapter, UNSLOTH_BLOCK_CUDAGRAPHS=1)",
    "cg_auto": "trl-internal-testing/tiny-Qwen3ForCausalLM (w20a dpo_cg_repro.py, DPO ref adapter, block CUDA graphs auto)",
    "dora16": "built:llama (DoRA r16 all 7 projections, bf16, GA 2)",
    "dora16_off": "built:llama (as dora16, UNSLOTH_DORA_FACTORED_NORM=0)",
    "kd_r16": "built:llama DoRA r16 bf16 (unsloth #18629)",
    "kd_r64": "built:llama DoRA r64 bf16 (unsloth #18629: float32 rank terms above r 32 on GPU)",
    "kd_h16acc": "built:llama DoRA r16 float16 + allow_fp16_accumulation (unsloth #18629)",
    "lsh_qwen3": "tiny Qwen3 PEFT LoRA on 7 projections + lm_head, label smoothing 0.1 (unsloth #18633)",
    "dora4": "built:llama (DoRA r16 all 7 projections, bitsandbytes NF4, GA 2)",
    "dora16_lr1": "built:llama (as dora16, lr 1e-3, 8 steps)",
    "dora4_lr1": "built:llama (as dora4, lr 1e-3, 8 steps)",
    "dora16_lr1_g64": "built:llama (gold of dora16_lr1)",
    "dora4_lr1_g64": "built:llama (gold of dora4_lr1)",
    "dora16_g64": "built:llama (as dora16, kill switch on, row norm in fp64 from the factors: gold trajectory)",
    "dora4_g64": "built:llama (as dora4, kill switch on, row norm in fp64 from the factors: gold trajectory)",
    "ls_qwen3": "trl-internal-testing/tiny-Qwen3ForCausalLM (SFT label_smoothing_factor 0.1, GA 2)",
    "ls_qwen3_off": "trl-internal-testing/tiny-Qwen3ForCausalLM (as ls_qwen3, UNSLOTH_FUSED_LABEL_SMOOTHING=0)",
    "pk_qwen3": "trl-internal-testing/tiny-Qwen3ForCausalLM (SFT packing=True, GA 4, loss-only eval steps)",
    "pk_qwen3_ef1": "trl-internal-testing/tiny-Qwen3ForCausalLM (as pk_qwen3, UNSLOTH_EVAL_EMPTY_CACHE=1)",
    "pf_qwen3": "trl-internal-testing/tiny-Qwen3ForCausalLM (SFT padding_free=True, GA 4, loss-only eval steps)",
    "mgsa_llama": "built:llama (16384-token LoRA, GC unsloth: saved-attention auto routing + pinned GC offload state)",
    "mgsa_qwen3": "trl-internal-testing/tiny-Qwen3ForCausalLM (16384-token LoRA, GC unsloth)",
    "mgsa_llama_sa0": "built:llama (as mgsa_llama, UNSLOTH_GC_SAVE_ATTENTION=0)",
    "mgpin_llama": "built:llama (16384-token LoRA, GC unsloth offload, 8 MiB slots)",
    "mgpin_llama_px0": "built:llama (as mgpin_llama, UNSLOTH_GC_PINNED_EXACT=0)",
    "flm16": "built:llama",
    "studiotok": "built:llama (Studio online_tokenization + dataset_map_num_proc routing, 2-step LoRA SFT on the map rows)",
    "qvlmoe_aux": "hf-internal-testing/tiny-random-Qwen3VLMoeForConditionalGeneration (output_router_logits, aux loss, 2-step LoRA SFT)",
    "flm16_gcsave": "built:llama",
    "lpad": "built:llama",
    "lpadx": "built:llama",
    "lpadx_gemma2": "trl-internal-testing/tiny-Gemma2ForCausalLM (FastLanguageModel, left vs right pad, leak probe)",
    "lpadx_gemma2_nbc": "trl-internal-testing/tiny-Gemma2ForCausalLM (as lpadx_gemma2, UNSLOTH_BLOCK_COMPILE=0)",
    "fpskip_uns": "built:llama (4 layers, LoRA on top 2 via finetune_last_n_layers=2, GC 'unsloth')",
    "fpskip_true": "built:llama (4 layers, LoRA on top 2 via finetune_last_n_layers=2, GC True)",
    "dsv4g": "tiny-random/deepseek-v4-bf16 (step-0 logits / loss / every grad + saved-for-backward bytes, then 5 LoRA steps)",
    "rpack_bf16": "trl-internal-testing/tiny-Qwen2_5_VLForConditionalGeneration (packed vs per-doc parity bf16 + block-varlen route counters)",
    "rpack_fp32": "trl-internal-testing/tiny-Qwen2_5_VLForConditionalGeneration (packed vs per-doc parity fp32 + block-varlen route counters)",
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
    "g2nr_flex": "trl-internal-testing/tiny-Gemma2ForCausalLM",
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
    "dpo_txt_psforce": "trl-internal-testing/tiny-Qwen3ForCausalLM",
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
    "grpo_pg_short": "tiny-Qwen3 GRPO, 8 short prompts x 4 gens, default GA (PrefixGrouper gate must decline)",
    "grpo_pg_long": "tiny-Qwen3 GRPO, 8 ~1.2K-token prompts x 4 gens, default GA (PrefixGrouper must engage at head)",
    "grpo_pg_long_kl": "as grpo_pg_long with beta 0.04 (reference log-probs through the no-grad PrefixGrouper path)",
    "plelens": "hf-tiny-v2/tiny-random-Qwen4ExpForCausalLM",
    "granite_gc": "built:granitemoehybrid (attention layers, from tests/test_granitemoehybrid_float16.py, widened)",
    "moeaux_nr": "trl-internal-testing/tiny-Qwen3MoeForCausalLM",
    "moeaux_gc_qwen3moe": "trl-internal-testing/tiny-Qwen3MoeForCausalLM",
    "moeaux_gc_gptoss": "trl-internal-testing/tiny-GptOssForCausalLM",
    "moesft_qwen3moe": "trl-internal-testing/tiny-Qwen3MoeForCausalLM",
    "moesft_gptoss": "trl-internal-testing/tiny-GptOssForCausalLM",
    "moesft_qwen35moe": "trl-internal-testing/tiny-Qwen3_5MoeForConditionalGeneration-3.6",
    "gkd": "trl-internal-testing/tiny-Qwen3ForCausalLM",
    "dec_whisper": "hf-internal-testing/tiny-random-WhisperForConditionalGeneration",
    "dec_gemma3n": "built:gemma3n-shared-kv (config from tests/test_generic_decode_graphs_shared_kv.py)",
    "bclens": "trl-internal-testing/tiny-Qwen3ForCausalLM",
    "bswap": "trl-internal-testing/tiny-Qwen3ForCausalLM",
    "tok_save": "built:llama",
    "grpo_vl": "trl-internal-testing/tiny-Qwen2_5_VLForConditionalGeneration",
    "qat_int4": "trl-internal-testing/small-Qwen3ForCausalLM (QAT int4)",
    "qat_int8": "trl-internal-testing/small-Qwen3ForCausalLM (QAT int8)",
    "qat_int4_ri0": "small-Qwen3 QAT int4, UNSLOTH_QAT_ROUND_INPLACE=0",
    "qat_int8_ri0": "small-Qwen3 QAT int8, UNSLOTH_QAT_ROUND_INPLACE=0",
    "grpo_txt_ts1": "trl-internal-testing/tiny-Qwen3ForCausalLM (UNSLOTH_GRPO_TRUNCATED_SUPPORT=1, top_p 0.9, min_p 0.05)",
    "grpo_txt_ts1_kr0": "tiny-Qwen3 GRPO ts1 + UNSLOTH_GRPO_KEPT_SET_RECOMPUTE=0",
    "keptset_scorer": "synthetic bf16 V=151936 H=896 4x512 kept-set scorer",
    "keptset_scorer_kr0": "kept-set scorer, UNSLOTH_GRPO_KEPT_SET_RECOMPUTE=0",
    "vlmpack_parity": "trl-internal-testing/tiny-Qwen2_5_VLForConditionalGeneration (packed vs per-doc parity fp32)",
    "vlmpack_parity_bf16": "trl-internal-testing/tiny-Qwen2_5_VLForConditionalGeneration (packed vs per-doc parity bf16)",
    "vlmpack_parity_k0": "trl-internal-testing/tiny-Qwen2_5_VLForConditionalGeneration (parity, UNSLOTH_VLM_MROPE_PACKING=0)",
    "vlmpack_sft": "trl-internal-testing/tiny-Qwen2_5_VLForConditionalGeneration (vision SFT padding_free=True)",
    "vlmpack_sft_k0": "trl-internal-testing/tiny-Qwen2_5_VLForConditionalGeneration (vision SFT padding_free=True, UNSLOTH_VLM_MROPE_PACKING=0)",
    "vlmpack_pad": "trl-internal-testing/tiny-Qwen2_5_VLForConditionalGeneration (vision SFT padded default)",
    "grpo_vl_dd0": "trl-internal-testing/tiny-Qwen2_5_VLForConditionalGeneration (UNSLOTH_GRPO_IMAGE_DEDUP=0)",
    # The PR test's model: on TRL 0.24, Qwen2.5-VL DPO drops image_grid_thw (fails at any state).
    "dpo_vl": "trl-internal-testing/tiny-LlavaForConditionalGeneration",
    "vision": "trl-internal-testing/tiny-Qwen2_5_VLForConditionalGeneration",
    "moe": "trl-internal-testing/tiny-Qwen3MoeForCausalLM",
    "gemma4": "trl-internal-testing/tiny-Gemma4ForConditionalGeneration",
    "gemma4_gcrec": "trl-internal-testing/tiny-Gemma4ForConditionalGeneration (decoder MLP recompute count)",
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
    res["force_float32_env"] = os.environ.get("UNSLOTH_FORCE_FLOAT32")  # unsloth #18208 / zoo #2580: fp16-only path
    res["mixed_precision_env"] = os.environ.get("UNSLOTH_MIXED_PRECISION")
    import unsloth.kernels.large_head_attention as lh
    res["large_head_patch_returns"] = safe(lambda: lh.patch_large_head_attention(model))
    res["large_head_patched"] = safe(lambda: lh._PATCHED[0])
    res["head_dims"] = sorted({int(getattr(m, "head_dim")) for m in model.modules()
                               if isinstance(getattr(m, "head_dim", None), int)})


def lost_checkpointing_classes():
    """transformers >= 5.23 checkpoints inside a per-class `forward_with_checkpointing` wrapper (__init_subclass__), so
    every GradientCheckpointingLayer subclass's `forward` is that wrapper unless something replaced it afterwards.
    Lists the imported subclasses whose forward is no longer the wrapper (= checkpointing silently off)."""
    from transformers.modeling_layers import GradientCheckpointingLayer
    if "__init_subclass__" not in GradientCheckpointingLayer.__dict__:
        return {"checkpoints_in_forward": False, "lost": [], "n_classes": None}
    seen, stack = set(), list(GradientCheckpointingLayer.__subclasses__())
    while stack:
        c = stack.pop()
        if c in seen:
            continue
        seen.add(c)
        stack.extend(c.__subclasses__())
    lost = sorted(f"{c.__module__}.{c.__qualname__}:{getattr(getattr(c.__dict__.get('forward'), '__code__', None), 'co_name', '?')}"
                  for c in seen if "forward" in c.__dict__
                  and getattr(getattr(c.__dict__["forward"], "__code__", None), "co_name", "") != "forward_with_checkpointing")
    return {"checkpoints_in_forward": True, "lost": lost, "n_classes": len(seen)}


def cell_gemma4_gcrec():
    """zoo #2582: tiny Gemma-4 LoRA (bf16, use_gradient_checkpointing="unsloth"): are the decoder layers recomputed
    in the backward? Counts each decoder layer's `mlp` forward per step (forward hook: fires in the forward and again
    in a checkpoint recompute; layer __call__ hooks do not fire on a 5.23 recompute). Checkpointing works = forward L,
    total 2L per micro-batch. Control arm: the same model with every layer's gradient_checkpointing flag off must give
    total L (proves the counter sees the difference)."""
    import torch
    from unsloth import FastModel
    from transformers.modeling_layers import GradientCheckpointingLayer
    versions()
    stage("load")
    model, tok = FastModel.from_pretrained(MODELS["gemma4"], max_seq_length = 256, dtype = torch.bfloat16,
                                           load_in_4bit = False)
    res["model_class"] = type(model).__name__
    model = FastModel.get_peft_model(model, r = 16, lora_alpha = 32, lora_dropout = 0.0, bias = "none",
                                     use_gradient_checkpointing = "unsloth", random_state = 0,
                                     finetune_vision_layers = False, finetune_language_layers = True,
                                     finetune_attention_modules = True, finetune_mlp_modules = True)
    lora_stats(model)
    res["lost_checkpointing"] = safe(lost_checkpointing_classes)
    layers = [m for m in model.modules() if isinstance(m, GradientCheckpointingLayer)
              and "DecoderLayer" in type(m).__name__ and isinstance(getattr(m, "mlp", None), torch.nn.Module)]
    res["n_layers"] = L = len(layers)
    res["layer_classes"] = sorted({type(m).__name__ for m in layers})
    res["layer_forward_names"] = sorted({getattr(getattr(type(m).forward, "__code__", None), "co_name", "?") for m in layers})
    res["layer_gc_flags"] = sorted({str(m.__dict__.get("gradient_checkpointing")) for m in layers})
    cnt = [0]

    def hook(*a):
        cnt[0] += 1
    for m in layers:
        m.mlp.register_forward_hook(hook)
    gc = GCCounter(model)
    res["gc_layers"] = gc.layers
    batch = text_batch(tok, model.get_input_embeddings().weight.device)
    params = [p for p in model.parameters() if p.requires_grad]
    res["trainable_params"] = int(sum(p.numel() for p in params))
    opt = torch.optim.AdamW(params, lr = args.lr)
    model.train()

    def arm(steps, tag):
        fw, tot, gcs, losses = [], [], [], []
        torch.cuda.reset_peak_memory_stats()
        for _ in range(steps):
            gc.install(model)
            cnt[0], g0 = 0, gc.calls
            out = model(**batch)
            fw.append(cnt[0])
            out.loss.backward()
            tot.append(cnt[0])
            gcs.append(gc.calls - g0)
            opt.step()
            opt.zero_grad(set_to_none = True)
            losses.append(float(out.loss.detach()))
        torch.cuda.synchronize()
        res[f"{tag}_mlp_fwd_per_step"], res[f"{tag}_mlp_total_per_step"] = fw, tot
        res[f"{tag}_gc_calls_per_step"] = gcs
        res[f"{tag}_peak_alloc_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
        dump()
        return losses
    stage("train_gc_on")
    losses = arm(args.steps, "gcon")
    res["gc_calls"] = gc.calls
    res["peak_alloc_mb"] = res["gcon_peak_alloc_mb"]
    # recompute = backward-time MLP calls (total - forward); step 0 can carry a block-compile trial forward
    res["gcon_recompute_per_step"] = [t - f for f, t in zip(res["gcon_mlp_fwd_per_step"], res["gcon_mlp_total_per_step"])]
    res["recomputed"] = all(r == L for r in res["gcon_recompute_per_step"]) and L > 0
    res["recompute_summary"] = (f"L={L} fwd/step={res['gcon_mlp_fwd_per_step']} total/step={res['gcon_mlp_total_per_step']} "
                                f"recomputed/step={res['gcon_recompute_per_step']}")
    finish_losses(losses)
    stage("control_gc_off")
    for m in layers:
        m.gradient_checkpointing = False
    try:
        arm(2, "gcoff")
        res["control_ok"] = all(t == L for t in res["gcoff_mlp_total_per_step"])
    except Exception as e:  # noqa: BLE001
        res["control_error"] = repr(e)[:300]
    finally:
        for m in layers:
            m.gradient_checkpointing = True


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
    model, proc = _vl_model(MODELS["grpo_vl"])
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
    try:
        import unsloth_zoo.vision_dedup as _vd
        _dd_before = dict(_vd.DEDUP_STATS)
    except Exception as _e:  # noqa: BLE001  (absent at the base of zoo #2590)
        _vd, _dd_before = None, None
        res["vision_dedup_module"] = f"absent: {type(_e).__name__}"
    trainer.train()
    peak()
    if _vd is not None:
        res["vision_dedup_stats"] = {k: v - _dd_before.get(k, 0) if isinstance(v, (int, float)) else v
                                     for k, v in _vd.DEDUP_STATS.items()}
        res["vision_dedup_engaged"] = res["vision_dedup_stats"].get("deduped_calls", 0) > 0
        res["UNSLOTH_GRPO_IMAGE_DEDUP"] = os.environ.get("UNSLOTH_GRPO_IMAGE_DEDUP")
    after = dict(fl.CALLS)
    res["fused_calls_delta"] = {k: after.get(k, 0) - before.get(k, 0) for k in after if after.get(k, 0) != before.get(k, 0)}
    res["grpo_fused_engaged"] = res["fused_calls_delta"].get("grpo_fwd_grad", 0) > 0
    logs = [h for h in trainer.state.log_history if "loss" in h]
    finish_losses([float(h["loss"]) for h in logs])
    res["decreasing"] = None  # 2 RL steps: finite and completed is the bar, not a falling loss
    res["rewards"] = [h.get("reward") for h in logs]


def cell_qat(scheme):
    """unsloth #18339: tiny Qwen3 LoRA + QAT (qat_scheme int4 / int8-int4) through SFTTrainer, 6 steps, then an eval
    forward. Records per-step losses, fake-quantized linear count, the in-place rounding / weight memo stats when the
    state has them. Cells ending _ri0 set UNSLOTH_QAT_ROUND_INPLACE=0 (the PR kill switch)."""
    import random
    import torch
    from unsloth import FastLanguageModel
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer
    versions()
    try:
        import torchao
        res["torchao"] = torchao.__version__
    except Exception as e:  # noqa: BLE001
        res["torchao"] = f"absent: {e!r}"[:200]
    stage("load")
    model, tok = FastLanguageModel.from_pretrained("trl-internal-testing/small-Qwen3ForCausalLM", max_seq_length = 128,
                                                   load_in_4bit = False, dtype = torch.bfloat16)
    model = FastLanguageModel.get_peft_model(model, r = 8, lora_alpha = 16, lora_dropout = 0, bias = "none",
                                             target_modules = TARGETS, qat_scheme = scheme,
                                             use_gradient_checkpointing = "unsloth", random_state = 3407)
    res["fake_quantized_linears"] = sum(1 for _, m in model.named_modules()
                                        if getattr(m, "weight_fake_quantizer", None) is not None)

    def stats():
        out = {}
        for modname, fn in (("unsloth.models._qat_weight_memo", "qat_weight_memo_stats"),
                            ("unsloth.models._qat_round", "qat_round_stats")):
            try:
                mod = __import__(modname, fromlist = ["x"])
            except Exception as e:  # noqa: BLE001
                out[modname] = f"absent: {type(e).__name__}"
                continue
            if fn:
                out[modname] = getattr(mod, fn)()
            else:
                out[modname] = {k: (v if isinstance(v, (int, float, bool, str, type(None))) else repr(v)[:200])
                                for k, v in vars(mod).items() if k.isupper() or "stat" in k.lower()
                                if not callable(v) and not k.startswith("__")}
        return out
    res["qat_stats_after_peft"] = stats()
    rng = random.Random(0)
    words = [w for d in DOCS for w in d.rstrip(".").split()]
    texts = [" ".join(rng.choice(words) for _ in range(rng.choice([10, 18, 26]))) + "." for _ in range(64)]
    cfg = _cfg(SFTConfig, output_dir = os.path.join(os.path.dirname(os.path.abspath(args.out)), f"trainer_{args.cell}"),
               max_steps = 6, per_device_train_batch_size = 2, gradient_accumulation_steps = 2, learning_rate = 2e-4,
               logging_steps = 1, seed = 3407, report_to = "none", max_length = 64, dataset_text_field = "text",
               optim = "adamw_torch", lr_scheduler_type = "linear", warmup_steps = 0, bf16 = True,
               save_strategy = "no", dataloader_num_workers = 0)
    trainer = SFTTrainer(model = model, processing_class = tok, train_dataset = Dataset.from_dict({"text": texts}),
                         args = cfg)
    res["trainer_class"] = type(trainer).__name__
    stage("train")
    torch.cuda.reset_peak_memory_stats()
    trainer.train()
    peak()
    res["qat_stats_after_train"] = stats()
    model.eval()
    batch = tok(["the river runs north past the old mill"], return_tensors = "pt").to(model.device)
    with torch.no_grad():
        logits = model(**batch).logits
    res["eval_logits_finite"] = bool(torch.isfinite(logits).all())
    res["UNSLOTH_QAT_ROUND_INPLACE"] = os.environ.get("UNSLOTH_QAT_ROUND_INPLACE")
    finish_losses([float(h["loss"]) for h in trainer.state.log_history if "loss" in h])
    res["finite"] = bool(res["finite"] and res["eval_logits_finite"])


def cell_keptset_scorer():
    """zoo #2595: grpo_chunked_hidden_states_logprobs with support_* kwargs (the kept-set scorer) on the GPU, bf16,
    Qwen-sized vocab. Per case: logps summary, grad norms, memory retained after the forward and the forward+backward
    peak above baseline. Base = old route, head = recompute route, *_kr0 = head with UNSLOTH_GRPO_KEPT_SET_RECOMPUTE=0."""
    import torch
    import unsloth_zoo.rl_replacements as rr
    versions()
    res["UNSLOTH_GRPO_KEPT_SET_RECOMPUTE"] = os.environ.get("UNSLOTH_GRPO_KEPT_SET_RECOMPUTE")
    res["has_grpo_kept_set_logprobs"] = hasattr(rr, "grpo_kept_set_logprobs")
    calls = {"kept_set": 0}
    if hasattr(rr, "grpo_kept_set_logprobs"):
        _orig = rr.grpo_kept_set_logprobs

        def _counted(*a, **k):
            calls["kept_set"] += 1
            return _orig(*a, **k)
        rr.grpo_kept_set_logprobs = _counted
    V, H, B, T = 151936, 896, 4, 512
    g = torch.Generator().manual_seed(0)
    head0 = (torch.randn(V, H, generator = g) * 0.05).to("cuda", torch.bfloat16)
    hidden0 = torch.randn(B, T, H, generator = g).to("cuda", torch.bfloat16)
    index = torch.randint(0, V, (B, T), generator = g).to("cuda")
    w = torch.rand(B, T, generator = g).to("cuda")
    cases = {"top_p0.9": (1.0, 0, 0.9, 0.0), "min_p0.05_t0.8": (0.8, 0, 1.0, 0.05), "k50_p0.9_m0.02_t0.9": (0.9, 50, 0.9, 0.02)}
    out = {}
    stage("score")
    for name, (temp, k, tp, mp) in cases.items():
        hidden = hidden0.clone().requires_grad_(True)
        head = head0.clone().requires_grad_(True)
        torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        base_mem = torch.cuda.memory_allocated()
        before = calls["kept_set"]
        logp = rr.grpo_chunked_hidden_states_logprobs(hidden, head, index, 4, 0.0, 0.0, 0.0, temp,
                                                      support_top_k = k, support_top_p = tp, support_min_p = mp)
        torch.cuda.synchronize()
        retained = torch.cuda.memory_allocated() - base_mem
        (logp.float() * w).sum().backward()
        torch.cuda.synchronize()
        peak_b = torch.cuda.max_memory_allocated() - base_mem
        with torch.no_grad():
            nog = rr.grpo_chunked_hidden_states_logprobs(hidden0, head0, index, 4, 0.0, 0.0, 0.0, temp,
                                                         support_top_k = k, support_top_p = tp, support_min_p = mp)
        lf = logp.detach().float()
        out[name] = {"logp_sum": float(lf.sum()), "logp_mean": float(lf.mean()),
                     "logp_head": [round(float(x), 5) for x in lf.flatten()[:6]],
                     "finite": bool(torch.isfinite(lf).all()), "nograd_max_abs_diff": float((nog.float() - lf).abs().max()),
                     "grad_hidden_norm": float(hidden.grad.float().norm()), "grad_head_norm": float(head.grad.float().norm()),
                     "retained_mb": round(retained / 2**20, 1), "peak_fwd_bwd_mb": round(peak_b / 2**20, 1),
                     "kept_set_calls": calls["kept_set"] - before}
        del hidden, head, logp, nog
    res["scorer"] = out
    res["finite"] = all(v["finite"] for v in out.values())
    res["losses"] = []


def cell_vlmpack_parity():
    """unsloth #18388: probes/vlmpack_parity.py (packed vs per-document logits / CE / LoRA grads, leak check, negative
    control) on tiny Qwen2.5-VL, own process. Records its JSON (refused: True when padding_free is refused)."""
    import subprocess
    versions()
    dt = "bf16" if "_bf16" in args.cell else "fp32"
    outp = os.path.join(os.path.dirname(os.path.abspath(args.out)), f"parity_{args.cell}.json")
    stage("parity")
    p = subprocess.run([sys.executable, "-u", os.path.join(os.path.dirname(os.path.abspath(__file__)), "vlmpack_parity.py"),
                        "--model", "trl-internal-testing/tiny-Qwen2_5_VLForConditionalGeneration", "--dtype", dt, "--out", outp],
                       capture_output = True, text = True, encoding = "utf-8", errors = "replace", timeout = 1500,
                       cwd = os.path.dirname(os.path.abspath(args.out)))
    res["parity_rc"] = p.returncode
    res["parity_tail"] = (p.stdout or "")[-1500:] + "\n--- stderr ---\n" + (p.stderr or "")[-1500:]
    res["UNSLOTH_VLM_MROPE_PACKING"] = os.environ.get("UNSLOTH_VLM_MROPE_PACKING")
    r = None
    if os.path.isfile(outp):
        with open(outp, encoding = "utf-8") as fh:
            r = json.load(fh)
    res["parity"] = r
    res["refused"] = bool(r and r.get("refused"))
    if r is None or res["refused"]:
        raise RuntimeError(f"padding_free parity did not run (rc {p.returncode}, refused={res['refused']})")
    res["finite"] = True
    res["losses"] = []


def cell_vlmpack_sft():
    """unsloth #18388: tiny Qwen2.5-VL LoRA vision SFT through SFTTrainer + UnslothVisionDataCollator, 3 steps,
    batch 4 (image docs + a text-only doc), padding_free=True (vlmpack_sft*) or the padded default (vlmpack_pad)."""
    import tempfile
    import torch
    from PIL import Image
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastVisionModel
    from unsloth.trainer import UnslothVisionDataCollator
    versions()
    stage("load")
    model, proc = FastVisionModel.from_pretrained("trl-internal-testing/tiny-Qwen2_5_VLForConditionalGeneration",
                                                  max_seq_length = 1024, dtype = torch.bfloat16, load_in_4bit = False)
    model = FastVisionModel.get_peft_model(model, r = 8, lora_alpha = 16, lora_dropout = 0.0, bias = "none",
                                           finetune_vision_layers = True, finetune_language_layers = True,
                                           random_state = 0, use_gradient_checkpointing = "unsloth")
    FastVisionModel.for_training(model)

    def img(w, h, seed):
        g = torch.Generator().manual_seed(seed)
        return Image.fromarray((torch.rand(h, w, 3, generator = g) * 255).to(torch.uint8).numpy())

    def ex(w, h, seed, q, ans):
        return {"messages": [
            {"role": "user", "content": [{"type": "image", "image": img(w, h, seed)}, {"type": "text", "text": q}]},
            {"role": "assistant", "content": [{"type": "text", "text": ans}]}]}
    rows = []
    for i in range(3):
        rows += [ex(56, 84, 1 + i, "Describe.", "A small picture with noise " * 3),
                 ex(112, 56, 10 + i, "What is it?", "Noise."),
                 {"messages": [{"role": "user", "content": [{"type": "text", "text": "Text only question here"}]},
                               {"role": "assistant", "content": [{"type": "text", "text": "text answer " * 5}]}]},
                 ex(84, 140, 20 + i, "Caption this image please.", "It is a random image " * 6)]
    pf = args.cell.startswith("vlmpack_sft")
    res["padding_free"] = pf
    res["UNSLOTH_VLM_MROPE_PACKING"] = os.environ.get("UNSLOTH_VLM_MROPE_PACKING")
    stage("collator")
    collator = UnslothVisionDataCollator(model, proc, **({"padding_free": True} if pf else {}))
    res["collator_padding_free"] = bool(getattr(collator, "padding_free", False))
    cfg = _cfg(SFTConfig, output_dir = tempfile.mkdtemp(), per_device_train_batch_size = 4, max_steps = 3,
               learning_rate = 2e-4, logging_steps = 1, report_to = "none", seed = 0, save_strategy = "no", bf16 = True,
               remove_unused_columns = False, dataset_text_field = "", dataset_kwargs = {"skip_prepare_dataset": True},
               max_length = None, lr_scheduler_type = "constant", warmup_steps = 0, dataloader_num_workers = 0)
    trainer = SFTTrainer(model = model, processing_class = proc.tokenizer if hasattr(proc, "tokenizer") else proc,
                         data_collator = collator, train_dataset = Dataset.from_list(rows), args = cfg)
    res["trainer_class"] = type(trainer).__name__
    first = collator(rows[:4])
    res["batch_shapes"] = {k: list(v.shape) for k, v in first.items() if hasattr(v, "shape")}
    stage("train")
    torch.cuda.reset_peak_memory_stats()
    trainer.train()
    peak()
    finish_losses([float(h["loss"]) for h in trainer.state.log_history if "loss" in h])
    res["decreasing"] = None


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


def cell_g2nr_flex():
    """unsloth #18219: tiny Gemma-2 (softcap -> flex_attention) LoRA in float32, block compile off
    (UNSLOTH_BLOCK_COMPILE=0, set in main), non-reentrant gradient checkpointing (vision.vlm_gradient_checkpointing_reentrant
    -> False, the DDP route). LoRA grads with checkpointing vs the same model with layer checkpointing switched off
    (g2_grad_rel), flex create_fw_bw_graph calls total / while saved-tensor hooks were active, whether the PR wrapper
    is installed, layer function calls during the checkpointed forward; then AdamW steps with checkpointing."""
    import sys as _sys
    import torch
    import unsloth.models.vision as uv
    from unsloth import FastModel
    versions()
    uv.vlm_gradient_checkpointing_reentrant = lambda: False
    hop = _sys.modules.get("torch._higher_order_ops.flex_attention")
    if hop is None:
        import importlib
        hop = importlib.import_module("torch._higher_order_ops.flex_attention")
    inner = getattr(hop, "create_fw_bw_graph", None)
    res["fix_installed"] = bool(getattr(inner, "_unsloth_saved_hooks_free", False))
    eng = [0, 0]
    if inner is not None:
        def counting(*a, **k):
            eng[0] += 1
            eng[1] += torch._C._autograd._top_saved_tensors_default_hooks(False) is not None
            return inner(*a, **k)
        hop.create_fw_bw_graph = counting
    stage("load")
    model, tok = FastModel.from_pretrained(MODELS[args.cell], max_seq_length = 256, dtype = torch.float32,
                                           load_in_4bit = False)
    res["attn_implementation"] = getattr(model.config, "_attn_implementation", None)
    model = FastModel.get_peft_model(model, r = 8, lora_alpha = 16, lora_dropout = 0.0, bias = "none",
                                     target_modules = TARGETS, random_state = 0, use_gradient_checkpointing = "unsloth")
    res["nonre_flag"] = getattr(model, "_unsloth_vlm_gc_non_reentrant", None)
    lora_stats(model)
    gc = GCCounter(model)
    dev = model.get_input_embeddings().weight.device
    vocab = min(int(model.config.get_text_config().vocab_size), 32000)
    ids = torch.randint(3, vocab, (2, 64), generator = torch.Generator().manual_seed(0)).to(dev)
    layers = [m for m in model.modules() if "gradient_checkpointing" in m.__dict__ and "DecoderLayer" in type(m).__name__]
    res["n_layers"] = len(layers)
    fwd_calls = [0]
    for l in layers:
        l.register_forward_pre_hook(lambda *a: fwd_calls.__setitem__(0, fwd_calls[0] + 1))

    def grads(gc_on):
        for l in layers:
            l.gradient_checkpointing = gc_on
        model.train()
        model.zero_grad(set_to_none = True)
        fwd_calls[0], gc0, e0 = 0, gc.calls, list(eng)
        out = model(input_ids = ids, labels = ids)
        n_fwd = fwd_calls[0]
        out.loss.backward()
        g = {n: p.grad.detach().float().clone() for n, p in model.named_parameters() if "lora_" in n and p.grad is not None}
        model.zero_grad(set_to_none = True)
        return {"loss": float(out.loss.detach()), "layer_fwd_calls_in_forward": n_fwd,
                "layer_fwd_calls_total": fwd_calls[0], "gc_calls": gc.calls - gc0,
                "fw_bw_graph_calls": eng[0] - e0[0], "fw_bw_graph_calls_under_hooks": eng[1] - e0[1]}, g
    stage("ref_nogc")
    r_off, g_off = grads(False)
    stage("gc_nonreentrant")
    try:
        r_on, g_on = grads(True)
    finally:
        res["g2_ref"] = r_off
    res["g2_gc"] = r_on
    num = sum(float((g_on[n] - g_off[n]).pow(2).sum()) for n in g_off if n in g_on)
    den = sum(float(g_off[n].pow(2).sum()) for n in g_off)
    res["g2_grad_rel"] = (num ** 0.5) / max(den ** 0.5, 1e-30)
    res["g2_grad_tensors"] = [len(g_off), len(g_on)]
    res["g2_loss_eq"] = r_off["loss"] == r_on["loss"]
    res["fw_bw_graph_calls"], res["fw_bw_graph_calls_under_hooks"] = eng
    stage("train")
    for l in layers:
        l.gradient_checkpointing = True
    torch.cuda.reset_peak_memory_stats()
    manual_loop(model, {"input_ids": ids, "labels": ids}, args.steps, gc)
    peak()
    res["gc_calls"] = gc.calls
    res["block_modes"] = _block_modes(model)


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


MOESFT_COEF = 0.9
MOESFT_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]
MOESFT_TEXT = "the river runs north past the old mill and the green house by the water, music carried over the lake."
# Stock TRL arm: its own process (no `import unsloth`), plain transformers + PEFT, same rows / config / lr 0.
MOESFT_STOCK = r'''
import json, sys, traceback
out = {"unsloth_imported": None}
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    from datasets import Dataset
    import trl
    from trl import SFTConfig, SFTTrainer
    repo, path, steps, coef, text, outdir = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4]), sys.argv[5], sys.argv[6]
    out["trl"] = trl.__version__
    tok = AutoTokenizer.from_pretrained(repo)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    from transformers import AutoConfig
    qc = getattr(AutoConfig.from_pretrained(repo), "quantization_config", None)
    kw = {}
    if (qc.get("quant_method") if isinstance(qc, dict) else getattr(qc, "quant_method", None)) == "mxfp4":
        from transformers import Mxfp4Config
        kw["quantization_config"] = Mxfp4Config(dequantize = True)  # bf16 experts, as Unsloth trains them
    out["load_kw"] = sorted(kw)
    model = AutoModelForCausalLM.from_pretrained(repo, dtype = torch.bfloat16, **kw).cuda()
    model.config.get_text_config().router_aux_loss_coef = coef
    for m in model.modules():
        if hasattr(m, "router_aux_loss_coef"):
            m.router_aux_loss_coef = coef
    model = get_peft_model(model, LoraConfig(r = 8, lora_alpha = 16, lora_dropout = 0.0, target_modules = %r))
    flags = []
    lm = [m for m in model.modules() if type(m).__name__.endswith("ForCausalLM")][0]
    lm.register_forward_pre_hook(lambda mod, a, k: flags.append(k.get("output_router_logits")), with_kwargs = True)
    cfg = SFTConfig(output_dir = outdir, per_device_train_batch_size = 2, gradient_accumulation_steps = 1, max_steps = steps,
                    learning_rate = 0.0, logging_steps = 1, bf16 = True, report_to = "none", seed = 0,
                    dataset_text_field = "text", max_length = 64, save_strategy = "no", dataset_num_proc = 1,
                    warmup_steps = 0, lr_scheduler_type = "constant", packing = False)
    tr = SFTTrainer(model = model, train_dataset = Dataset.from_dict({"text": [text] * 16}), processing_class = tok, args = cfg)
    out["aux_loss_enabled"] = getattr(tr, "aux_loss_enabled", None)
    out["router_aux_loss_coef"] = getattr(tr, "router_aux_loss_coef", None)
    seen = []
    orig = tr.compute_loss
    def wrapped(model_, inputs, *a, **k):
        if not seen:
            seen.append([int(x) for x in inputs["input_ids"][0].tolist()])
        return orig(model_, inputs, *a, **k)
    tr.compute_loss = wrapped
    tr.train()
    logs = [h for h in tr.state.log_history if "loss" in h]
    out["losses"] = [float(h["loss"]) for h in logs]
    out["aux_logged"] = [h.get("aux_loss") for h in logs]
    out["grad_norms"] = [h.get("grad_norm") for h in logs]
    out["router_logits_kw"] = flags[:4]
    out["input_ids0"] = seen[0] if seen else None
    out["unsloth_imported"] = "unsloth" in sys.modules
    out["ok"] = True
except BaseException as e:
    out["ok"], out["error"], out["traceback"] = False, f"{type(e).__name__}: {e}"[:1500], traceback.format_exc()[-3000:]
with open(sys.argv[2], "w", encoding = "utf-8") as fh:
    json.dump(out, fh)
''' % (MOESFT_TARGETS,)


def cell_moesft(key):
    """unsloth #18230: SFT on a tiny MoE through Unsloth's SFTTrainer with router_aux_loss_coef 0.9 and lr 0 (the weights
    never move, so every logged loss is the step-1 loss). Oracle from the same captured batch after training: nll only
    (no router logits) and nll + coef * aux (output_router_logits=True). The step-1 loss must equal the stock TRL arm
    (separate process, no unsloth) and the nll + aux oracle; at a state that drops the aux loss it equals the nll."""
    import subprocess
    import torch
    from unsloth import FastModel
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer
    versions()
    stage("load")
    repo = MODELS[key]
    model, tok = FastModel.from_pretrained(repo, max_seq_length = 256, dtype = torch.bfloat16, load_in_4bit = False)
    model.config.get_text_config().router_aux_loss_coef = MOESFT_COEF
    for m in model.modules():
        if hasattr(m, "router_aux_loss_coef"):
            m.router_aux_loss_coef = MOESFT_COEF
    model = FastModel.get_peft_model(model, r = 8, lora_alpha = 16, lora_dropout = 0.0, target_modules = MOESFT_TARGETS,
                                     random_state = 0, use_gradient_checkpointing = "unsloth")
    tt = getattr(tok, "tokenizer", tok)
    if tt.pad_token is None:
        tt.pad_token = tt.eos_token
    flags = []
    lm = [m for m in model.modules() if type(m).__name__.endswith("ForCausalLM")][0]
    lm.register_forward_pre_hook(lambda mod, a, k: flags.append(k.get("output_router_logits")), with_kwargs = True)
    cfg = SFTConfig(output_dir = os.path.join(os.path.dirname(os.path.abspath(args.out)), f"trainer_{args.cell}"),
                    per_device_train_batch_size = 2, gradient_accumulation_steps = 1, max_steps = 2,
                    learning_rate = 0.0, logging_steps = 1, bf16 = True, report_to = "none", seed = 0,
                    dataset_text_field = "text", max_length = 64, save_strategy = "no", dataset_num_proc = 1,
                    warmup_steps = 0, lr_scheduler_type = "constant", packing = False)
    trainer = SFTTrainer(model = model, train_dataset = Dataset.from_dict({"text": [MOESFT_TEXT] * 16}),
                         processing_class = tt, args = cfg)
    res["trainer_class"] = type(trainer).__name__
    res["aux_loss_enabled"] = getattr(trainer, "aux_loss_enabled", None)
    res["trainer_router_aux_loss_coef"] = getattr(trainer, "router_aux_loss_coef", None)
    seen = []
    orig = trainer.compute_loss

    def wrapped(model_, inputs, *a, **k):
        if not seen:
            seen.append({n: (v.detach().clone() if torch.is_tensor(v) else v) for n, v in inputs.items()})
        return orig(model_, inputs, *a, **k)
    trainer.compute_loss = wrapped
    stage("train")
    torch.cuda.reset_peak_memory_stats()
    trainer.train()
    peak()
    logs = [h for h in trainer.state.log_history if "loss" in h]
    res["grad_norms"] = [h.get("grad_norm") for h in logs]
    finish_losses([float(h["loss"]) for h in logs])
    res["decreasing"] = None  # lr 0 by design
    res["router_logits_kw"] = flags[:4]
    res["router_logits_requested"] = bool(flags) and flags[0] is True
    stage("oracle")
    res["batch_keys"] = sorted(seen[0])
    b = {n: v for n, v in seen[0].items() if n in ("input_ids", "attention_mask", "labels", "position_ids")}
    res["input_ids0"] = [int(x) for x in b["input_ids"][0].tolist()]
    with torch.no_grad():
        o_nll = model(**b)
        o_aux = model(**b, output_router_logits = True)
    aux = getattr(o_aux, "aux_loss", None)
    res["oracle_nll"] = float(o_nll.loss)
    res["oracle_aux"] = None if aux is None else float(aux)
    res["oracle_nll_plus_aux"] = None if aux is None else float(o_nll.loss) + MOESFT_COEF * float(aux)
    res["oracle_model_loss_with_router_logits"] = float(o_aux.loss)
    stage("stock_trl")
    sdir = os.path.dirname(os.path.abspath(args.out))
    script, sout = os.path.join(sdir, f"{args.cell}_stock_trl.py"), os.path.join(sdir, f"{args.cell}_stock_trl.json")
    with open(script, "w", encoding = "utf-8") as fh:
        fh.write(MOESFT_STOCK)
    env = {k: v for k, v in os.environ.items()}
    p = subprocess.run([sys.executable, script, repo, sout, "2", str(MOESFT_COEF), MOESFT_TEXT,
                        os.path.join(sdir, f"trainer_{args.cell}_stock")], capture_output = True, text = True,
                       encoding = "utf-8", errors = "replace", env = env, timeout = 1800)
    try:
        with open(sout, encoding = "utf-8") as fh:
            stock = json.load(fh)
    except Exception as e:  # noqa: BLE001
        stock = {"ok": False, "error": f"no stock json ({type(e).__name__}); rc {p.returncode}: {p.stderr[-1500:]}"}
    res["stock_trl"] = {k: v for k, v in stock.items() if k != "input_ids0"}
    s_ids, u_ids = stock.get("input_ids0") or [], res["input_ids0"]
    # Unsloth may pad-free-pack the two identical rows into one: same tokens = the stock row repeated.
    res["stock_same_tokens"] = (bool(s_ids) and len(u_ids) % len(s_ids) == 0 and u_ids == s_ids * (len(u_ids) // len(s_ids))) \
        if s_ids else None
    l1 = res["losses"][0] if res.get("losses") else None
    s1 = (stock.get("losses") or [None])[0]
    rel = lambda a, c: None if a is None or c is None else abs(a - c) / max(1.0, abs(c))  # noqa: E731
    res["step1_loss"] = l1
    res["stock_step1_loss"] = s1
    res["rel_vs_stock"] = rel(l1, s1)
    res["rel_vs_nll_plus_aux"] = rel(l1, res["oracle_nll_plus_aux"])
    res["rel_vs_nll_only"] = rel(l1, res["oracle_nll"])
    res["aux_gap_rel"] = rel(res["oracle_nll_plus_aux"], res["oracle_nll"])
    tol = 1e-2
    res["moesft_matches_stock"] = bool(res["rel_vs_stock"] is not None and res["rel_vs_stock"] < tol)
    res["moesft_keeps_aux"] = bool(res["rel_vs_nll_plus_aux"] is not None and res["rel_vs_nll_plus_aux"] < tol
                                   and (res["aux_gap_rel"] or 0) > 5 * tol)
    res["moesft_drops_aux"] = bool(res["rel_vs_nll_only"] is not None and res["rel_vs_nll_only"] < tol
                                   and (res["aux_gap_rel"] or 0) > 5 * tol)


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
               save_strategy = "no", temperature = 1.0, use_vllm = False,
               **({"top_p": 0.9, "min_p": 0.05} if args.cell.startswith("grpo_txt_ts1") else {}))
    res["UNSLOTH_GRPO_TRUNCATED_SUPPORT"] = os.environ.get("UNSLOTH_GRPO_TRUNCATED_SUPPORT")
    trainer = GRPOTrainer(model = model, reward_funcs = [reward], args = cfg, train_dataset = ds, processing_class = tok)
    res["trainer_class"] = type(trainer).__name__
    res["sampling"] = {k: getattr(trainer.args, k, None) for k in ("top_p", "min_p", "top_k", "temperature")}
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


def cell_lpadx():
    """unsloth #18272: tiny built llama + LoRA in training mode, the same 4 rows padded right and left by
    DataCollatorForSeq2Seq with the first real label KEPT (as the collator does). Per side: loss, LoRA grad norm, and
    the real-token logits (logsumexp over the vocab + the first 8 logits of every real position, no-label forward) for
    a base-vs-head comparison. Leak probe: the left-padded shortest row with every token after its second real token
    replaced; logits at the pad positions and the first real position must not move (the last pad position predicts
    the first real token). Then one AdamW step on the left-padded batch and the right-padded loss after it."""
    import torch
    from unsloth import FastLanguageModel
    from transformers import DataCollatorForSeq2Seq
    versions()
    stage("load")
    model, tok = FastLanguageModel.from_pretrained((built_llama() if args.cell == "lpadx" else MODELS["lpadx_gemma2"].split(" ")[0]), max_seq_length = 256, dtype = torch.bfloat16,
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
        rows.append({"input_ids": ids, "attention_mask": [1] * len(ids), "labels": list(ids)})
    dev = model.get_input_embeddings().weight.device
    lora = [p for n, p in model.named_parameters() if p.requires_grad]

    def collate(side, rws):
        tt.padding_side = side
        b = DataCollatorForSeq2Seq(tt, padding = True, return_tensors = "pt")([dict(r) for r in rws])
        return {k: v.to(dev) for k, v in b.items()}

    def real_logits(b):
        with torch.no_grad():
            lg = model(input_ids = b["input_ids"], attention_mask = b["attention_mask"]).logits.float()
        m = b["attention_mask"].bool()
        r = lg[m]
        return lg, [round(x, 5) for x in r.logsumexp(-1).tolist()], [[round(x, 4) for x in v] for v in r[:, :8].tolist()]

    model.train()
    stage("train")
    torch.cuda.reset_peak_memory_stats()
    for side in ("right", "left"):
        b = collate(side, rows)
        model.zero_grad(set_to_none = True)
        loss = model(**b).loss
        loss.backward()
        grads = torch.cat([p.grad.detach().float().flatten() for p in lora])
        res[f"loss_{side}"] = float(loss)
        res[f"gnorm_{side}"] = float(grads.norm())
        res[f"pad_count_{side}"] = int((b["attention_mask"] == 0).sum())
        _, lse, l8 = real_logits(b)
        res[f"real_lse_{side}"] = lse
        res[f"real_logits8_{side}"] = l8
    stage("leak")
    b = collate("left", rows)
    lg0, _, _ = real_logits(b)
    row = 0  # shortest row: most left padding
    npad = int((b["attention_mask"][row] == 0).sum())
    b2 = {k: v.clone() for k, v in b.items()}
    vocab = model.get_input_embeddings().weight.shape[0]
    b2["input_ids"][row, npad + 2:] = (b2["input_ids"][row, npad + 2:] * 7 + 13) % vocab
    lg1, _, _ = real_logits(b2)
    res["leak_npad"] = npad
    res["leak_last_pad"] = float((lg0[row, npad - 1] - lg1[row, npad - 1]).abs().max())
    res["leak_upto_first_real"] = float((lg0[row, :npad + 1] - lg1[row, :npad + 1]).abs().max())
    res["leak"] = res["leak_upto_first_real"]
    stage("step")
    opt = torch.optim.AdamW(lora, lr = 1e-3)
    model.zero_grad(set_to_none = True)
    model(**collate("left", rows)).loss.backward()
    opt.step()
    model.zero_grad(set_to_none = True)
    with torch.no_grad():
        res["loss_right_after_left_step"] = float(model(**collate("right", rows)).loss)
    peak()
    finish_losses([res["loss_right"]])
    res["decreasing"] = None


def cell_moesft_vlm(key):
    """unsloth #18271: SFT on a tiny MoE VLM wrapper (Qwen3.5-MoE, text-only rows) through Unsloth's SFTTrainer at
    lr 0 with SFTConfig(router_aux_loss_coef 0.9) and the checkpoint's own text_config coefficient left as shipped.
    Oracles on the captured batch: nll (output_router_logits False) and aux; the step-1 loss must equal
    nll + 0.9 * aux (keeps the user coefficient), not nll + checkpoint_coef * aux. The text config must hold the
    checkpoint value again after training."""
    import dataclasses
    import torch
    from unsloth import FastModel
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer
    versions()
    stage("load")
    repo = MODELS[key]
    model, tok = FastModel.from_pretrained(repo, max_seq_length = 256, dtype = torch.bfloat16, load_in_4bit = False)
    tc = model.config.get_text_config()
    res["ckpt_coef"] = getattr(tc, "router_aux_loss_coef", None)
    res["ckpt_output_router_logits"] = getattr(tc, "output_router_logits", None)
    model = FastModel.get_peft_model(model, r = 8, lora_alpha = 16, lora_dropout = 0.0, target_modules = MOESFT_TARGETS,
                                     random_state = 0, use_gradient_checkpointing = "unsloth")
    tt = getattr(tok, "tokenizer", tok)
    if tt.pad_token is None:
        tt.pad_token = tt.eos_token
    kw = dict(output_dir = os.path.join(os.path.dirname(os.path.abspath(args.out)), f"trainer_{args.cell}"),
              per_device_train_batch_size = 2, gradient_accumulation_steps = 1, max_steps = 2,
              learning_rate = 0.0, logging_steps = 1, bf16 = True, report_to = "none", seed = 0,
              dataset_text_field = "text", max_length = 64, save_strategy = "no", dataset_num_proc = 1,
              warmup_steps = 0, lr_scheduler_type = "constant", packing = False)
    has_field = "router_aux_loss_coef" in {f.name for f in dataclasses.fields(SFTConfig)}
    res["sftconfig_has_router_aux_loss_coef"] = has_field
    if has_field:
        kw["router_aux_loss_coef"] = MOESFT_COEF
    cfg = SFTConfig(**kw)
    trainer = SFTTrainer(model = model, train_dataset = Dataset.from_dict({"text": [MOESFT_TEXT] * 16}),
                         processing_class = tt, args = cfg)
    res["trainer_class"] = type(trainer).__name__
    res["aux_loss_enabled"] = getattr(trainer, "aux_loss_enabled", None)
    res["trainer_router_aux_loss_coef"] = getattr(trainer, "router_aux_loss_coef", None)
    seen = []
    orig = trainer.compute_loss

    def wrapped(model_, inputs, *a, **k):
        if not seen:
            seen.append({n: (v.detach().clone() if torch.is_tensor(v) else v) for n, v in inputs.items()})
        return orig(model_, inputs, *a, **k)
    trainer.compute_loss = wrapped
    stage("train")
    torch.cuda.reset_peak_memory_stats()
    trainer.train()
    peak()
    logs = [h for h in trainer.state.log_history if "loss" in h]
    finish_losses([float(h["loss"]) for h in logs])
    res["decreasing"] = None  # lr 0 by design
    res["coef_after_train"] = getattr(model.config.get_text_config(), "router_aux_loss_coef", None)
    stage("oracle")
    res["batch_keys"] = sorted(seen[0])
    b = {n: v for n, v in seen[0].items() if n in ("input_ids", "attention_mask", "labels", "position_ids")}
    with torch.no_grad():
        o_nll = model(**b, output_router_logits = False)
        o_aux = model(**b, output_router_logits = True)
    aux = getattr(o_aux, "aux_loss", None)
    res["oracle_nll"] = float(o_nll.loss)
    res["oracle_aux"] = None if aux is None else float(aux)
    ck = res["ckpt_coef"] or 0.0
    res["oracle_nll_plus_aux"] = None if aux is None else float(o_nll.loss) + MOESFT_COEF * float(aux)
    res["oracle_nll_plus_ckpt_aux"] = None if aux is None else float(o_nll.loss) + ck * float(aux)
    rel = lambda a, c: None if a is None or c is None else abs(a - c) / max(1.0, abs(c))  # noqa: E731
    l1 = res["losses"][0] if res.get("losses") else None
    res["step1_loss"] = l1
    res["rel_vs_nll_plus_aux"] = rel(l1, res["oracle_nll_plus_aux"])
    res["rel_vs_nll_plus_ckpt_aux"] = rel(l1, res["oracle_nll_plus_ckpt_aux"])
    res["aux_gap_rel"] = rel(res["oracle_nll_plus_aux"], res["oracle_nll_plus_ckpt_aux"])
    tol = 1e-2
    gap_ok = (res["aux_gap_rel"] or 0) > 5 * tol
    res["moesft_keeps_aux"] = bool(res["rel_vs_nll_plus_aux"] is not None and res["rel_vs_nll_plus_aux"] < tol and gap_ok)
    res["moesft_ckpt_coef"] = bool(res["rel_vs_nll_plus_ckpt_aux"] is not None and res["rel_vs_nll_plus_ckpt_aux"] < tol
                                   and gap_ok)
    res["config_restored"] = res["coef_after_train"] == res["ckpt_coef"]


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



def cell_grpo_pg_len(long, beta = 0.0):
    """unsloth #18405 + zoo #2596: tiny Qwen3 LoRA GRPO without vLLM at Unsloth's default GA (TRL shuffles the
    generation batch across micro-batches), 8 distinct prompts x 4 generations, 2 steps. short = one-sentence prompts
    (the saved-token gate must decline: 0 layouts, same losses as base); long = ~1.2K-token prompts (head must build
    layouts; losses within bf16 noise of base). Counts build_group_layout calls / layouts and reads the verify flags."""
    import torch
    from unsloth import FastLanguageModel
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer
    import unsloth.utils.prefix_grouper as pg
    versions()
    stage("load")
    calls = {"n": 0, "layouts": 0, "saved": []}
    orig = pg.build_group_layout

    def spy(*a, **k):
        out = orig(*a, **k)
        calls["n"] += 1
        calls["layouts"] += int(out is not None)
        if out is not None:
            calls["saved"].append([safe(lambda: int(out.flat_ids.shape[1])), safe(lambda: round(float(out.tok_r), 3)), str(getattr(out, "signature", ""))[:60]])
        return out
    pg.build_group_layout = spy
    model, tok = FastLanguageModel.from_pretrained(MODELS["grpo_txt"], max_seq_length = 2048, dtype = torch.bfloat16,
                                                   load_in_4bit = False)
    model = FastLanguageModel.get_peft_model(model, r = 16, lora_alpha = 32, lora_dropout = 0.0, bias = "none",
                                             target_modules = TARGETS, use_gradient_checkpointing = "unsloth",
                                             random_state = 0)
    body = " ".join(DOCS * 38) if long else DOCS[0]
    prompts = [body + f" question {i}?" for i in range(8)]
    tt = getattr(tok, "tokenizer", tok)
    res["prompt_tokens"] = safe(lambda: [len(tt(tt.apply_chat_template([{"role": "user", "content": q}], tokenize = False,
                                                                     add_generation_prompt = True)).input_ids) for q in prompts[:2]])
    ds = Dataset.from_list([{"prompt": [{"role": "user", "content": q}]} for q in prompts])

    def reward(completions, **kw):
        return [float(len(str(c))) / 10.0 for c in completions]
    cfg = _cfg(GRPOConfig, output_dir = os.path.join(os.path.dirname(os.path.abspath(args.out)), "trainer_grpo_pg_len"),
               per_device_train_batch_size = 4, num_generations = 4, max_completion_length = 8, max_prompt_length = None,
               max_steps = 2, learning_rate = 5e-5, logging_steps = 1, report_to = "none", bf16 = True, seed = 0,
               save_strategy = "no", temperature = 1.0, use_vllm = False, beta = beta)
    trainer = GRPOTrainer(model = model, reward_funcs = [reward], args = cfg, train_dataset = ds, processing_class = tok)
    res["trainer_class"] = type(trainer).__name__
    res["beta"] = getattr(trainer.args, "beta", None)
    res["ga"] = getattr(trainer.args, "gradient_accumulation_steps", None)
    res["pg_min_saved"] = safe(lambda: pg.min_saved()) if hasattr(pg, "min_saved") else "absent"
    stage("train")
    torch.cuda.reset_peak_memory_stats()
    trainer.train()
    peak()
    res["pg_build_calls"], res["pg_layouts"] = calls["n"], calls["layouts"]
    res["pg_layout_T_sig"] = calls["saved"][:8]
    res["pg_flags"] = {k: str(_find_attr(trainer.model, k))[:160] for k in (
        "_unsloth_prefix_grouper_nograd_disabled", "_unsloth_prefix_grouper_nograd_verified",
        "_unsloth_prefix_grouper_nograd_unsafe", "_unsloth_prefix_grouper_grad_verified",
        "_unsloth_prefix_grouper_grad_unsafe")}
    res["pg_enabled"] = safe(lambda: bool(pg.prefix_grouper_enabled(trainer.model)))
    logs = [h for h in trainer.state.log_history if "loss" in h]
    finish_losses([float(h["loss"]) for h in logs])
    res["decreasing"] = None
    res["rewards"] = [h.get("reward") for h in logs]
    res["grad_norms"] = [h.get("grad_norm") for h in logs]

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


def _rows_sha(rows):
    import hashlib
    h = hashlib.sha256()
    for r in rows:
        h.update(json.dumps(list(map(int, r)), separators = (",", ":")).encode("utf-8"))
        h.update(b"|")
    return h.hexdigest()


def cell_studiotok():
    """unsloth #17701: Studio's tokenize routing and rows, then 2 LoRA SFT steps on those rows. Imports Studio's
    backend from this state's unsloth checkout. Routing = the trainer's expression
    dataset_map_num_proc(1 if veto else None, serial_as_none = False) with the real host-memory reading (no veto
    module at the base = no veto). Rows: eager Dataset.map at the routed num_proc (a forked pool on Linux), in process
    (num_proc None), and the online with_transform view; sha256 of input_ids per path."""
    import importlib
    import torch
    import unsloth
    from pathlib import Path as _P
    backend = _P(unsloth.__file__).resolve().parents[1] / "studio" / "backend"
    res["studio_backend"] = str(backend)
    sys.path.insert(0, str(backend))
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer
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
    stage("routing")
    ot = importlib.import_module("utils.datasets.online_tokenization")
    hwmod = importlib.import_module("utils.hardware")
    dmnp = hwmod.dataset_map_num_proc
    res["has_fork_veto"] = hasattr(ot, "host_memory_fork_veto")
    veto = None
    if res["has_fork_veto"]:
        snap = ot.host_memory_snapshot()
        res["host_memory_snapshot_gib"] = None if snap is None else [round(x / 2**30, 2) for x in snap]
        veto = ot.host_memory_fork_veto(0, snap)
        res["fork_veto_real"] = veto
        res["fork_veto_blockswap"] = ot.host_memory_fork_veto(28, snap)
        res["fork_veto_oversize"] = ot.host_memory_fork_veto(0, (40 << 30, 8 << 30))
        res["num_proc_if_vetoed"] = dmnp(1, serial_as_none = False)
    res["num_proc_unvetoed"] = dmnp(None, serial_as_none = False)
    routed = dmnp(1 if veto else None, serial_as_none = False)
    res["num_proc_routed"] = routed
    stage("tokenize")
    texts = [f"{d} row {i}." + (tt.eos_token or "") for i, d in enumerate(DOCS * 40)]
    ds = Dataset.from_dict({"text": texts})
    res["rows"] = len(ds)
    add_special = ot.resolve_add_special_tokens(tt, texts[0])
    res["add_special_tokens"] = add_special

    def _tok(b):
        return dict(tt(b["text"], truncation = True, max_length = 64, add_special_tokens = add_special))
    import multiprocessing as _mp
    n_children_before = len(_mp.active_children())
    eager = ds.map(_tok, batched = True, num_proc = routed, remove_columns = ["text"])
    res["map_children_seen_before"] = n_children_before
    inproc = ds.map(_tok, batched = True, num_proc = None, remove_columns = ["text"])
    view = ot.attach_online_tokenization(ds, tokenizer = tt, text_field = "text", max_length = 64,
                                         add_special_tokens = add_special)
    lazy = [view[i]["input_ids"] for i in range(len(ds))]
    res["rows_sha_eager_routed"] = _rows_sha(eager["input_ids"])
    res["rows_sha_in_process"] = _rows_sha(inproc["input_ids"])
    res["rows_sha_online_view"] = _rows_sha(lazy)
    res["rows_identical_all_paths"] = len({res["rows_sha_eager_routed"], res["rows_sha_in_process"],
                                           res["rows_sha_online_view"]}) == 1
    res["n_tokens"] = int(sum(len(r) for r in eager["input_ids"]))
    try:
        kw = dict(dataset = ds, processing_class = tt, model = model, text_field = "text", num_train_epochs = 1.0,
                  max_steps = 0, grad_accum = 1)
        if res["has_fork_veto"]:
            kw["host_memory"] = snap
        d = ot.decide_online_tokenization(**kw)
        res["online_decision"] = {"enabled": bool(d.enabled), "reason": str(d.reason)[:300], "workers": d.workers}
    except Exception as e:  # noqa: BLE001
        res["online_decision"] = f"ERR {type(e).__name__}: {str(e)[:300]}"
    stage("train")
    cfg = SFTConfig(output_dir = os.path.join(os.path.dirname(os.path.abspath(args.out)), f"trainer_{args.cell}"),
                    per_device_train_batch_size = 2, gradient_accumulation_steps = 1, max_steps = 2,
                    learning_rate = args.lr, logging_steps = 1, bf16 = True, report_to = "none", seed = 0,
                    max_length = 64, save_strategy = "no", dataset_num_proc = 1,
                    warmup_steps = 0, lr_scheduler_type = "constant", packing = False)
    torch.cuda.reset_peak_memory_stats()
    trainer = SFTTrainer(model = model, train_dataset = eager, processing_class = tt, args = cfg)
    res["trainer_class"] = type(trainer).__name__
    trainer.train()
    peak()
    logs = [h for h in trainer.state.log_history if "loss" in h]
    res["grad_norms"] = [h.get("grad_norm") for h in logs]
    finish_losses([float(h["loss"]) for h in logs])


def qvlmoe_local():
    """The tiny Qwen3-VL-MoE repo was saved by transformers 5 (rope_parameters, rope_scaling None); 4.57.x reads
    text_config.rope_scaling["mrope_section"] and dies on None. Below 5, a local copy whose text_config carries the
    4.57 keys (default rope, mrope_section summing to head_dim / 2); 5.x loads the repo as shipped."""
    import shutil
    import transformers
    from huggingface_hub import snapshot_download
    repo = MODELS["qvlmoe_aux"].split()[0]
    src = snapshot_download(repo)
    if int(transformers.__version__.split(".")[0]) >= 5:
        res["qvlmoe_config_patched"] = False
        return src
    dst = os.path.join(os.path.dirname(os.path.abspath(args.out)), "qvlmoe_tf4")
    shutil.copytree(src, dst, dirs_exist_ok = True)
    cp = os.path.join(dst, "config.json")
    with open(cp, encoding = "utf-8") as fh:
        cfg = json.load(fh)
    tc = cfg["text_config"]
    rp = tc.get("rope_parameters") or {}
    half = int(tc.get("head_dim") or tc["hidden_size"] // tc["num_attention_heads"]) // 2
    sec = [half - 2 * (half // 3), half // 3, half // 3]
    tc["rope_scaling"] = {"rope_type": rp.get("rope_type", "default"), "mrope_section": sec,
                          "mrope_interleaved": True}
    tc["rope_theta"] = tc.get("rope_theta") or rp.get("rope_theta", 5000000.0)
    with open(cp, "w", encoding = "utf-8") as fh:
        json.dump(cfg, fh, indent = 1)
    res["qvlmoe_config_patched"] = tc["rope_scaling"]
    return dst


def cell_qvlmoe_aux():
    """unsloth-zoo #2587: tiny Qwen3-VL-MoE loaded through Unsloth (compiled), LoRA; one forward with
    output_router_logits=True (aux_loss finite? router_logits attached?), then 2 SFT steps with the text config's
    output_router_logits on (TRL adds the aux loss), lr 0 so step losses equal nll + coef * aux on fixed weights."""
    import torch
    from unsloth import FastModel
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer
    versions()
    stage("load")
    model, tok = FastModel.from_pretrained(qvlmoe_local(), max_seq_length = 256, dtype = torch.bfloat16,
                                           load_in_4bit = False)
    tc = model.config.get_text_config()
    res["num_experts"] = getattr(tc, "num_experts", None)
    res["ckpt_coef"] = getattr(tc, "router_aux_loss_coef", None)
    model = FastModel.get_peft_model(model, r = 8, lora_alpha = 16, lora_dropout = 0.0, target_modules = MOESFT_TARGETS,
                                     random_state = 0, use_gradient_checkpointing = "unsloth",
                                     finetune_vision_layers = False)
    tt = getattr(tok, "tokenizer", tok)
    if tt.pad_token is None:
        tt.pad_token = tt.eos_token
    stage("forward_router_logits")
    dev = model.get_input_embeddings().weight.device
    enc = tt([MOESFT_TEXT, MOESFT_TEXT[:60]], return_tensors = "pt", padding = True)
    b = {"input_ids": enc["input_ids"].to(dev), "attention_mask": enc["attention_mask"].to(dev)}
    b["labels"] = b["input_ids"].clone()
    try:
        with torch.no_grad():
            o = model(**b, output_router_logits = True)
        aux = getattr(o, "aux_loss", None)
        rl = getattr(o, "router_logits", None)
        res["fwd_aux_loss"] = None if aux is None else float(aux)
        res["fwd_aux_finite"] = aux is not None and bool(torch.isfinite(aux).all())
        res["fwd_router_logits_n"] = None if rl is None else len(rl)
        res["fwd_router_logits_shape0"] = None if not rl else list(rl[0].shape)
        res["fwd_loss"] = float(o.loss)
        res["fwd_error"] = None
    except Exception as e:  # noqa: BLE001
        res["fwd_error"] = f"{type(e).__name__}: {str(e)[:400]}"
        res["fwd_aux_finite"] = False
    stage("train")
    tc = model.config.get_text_config()
    tc.output_router_logits = True
    model.config.output_router_logits = True
    cfg = SFTConfig(output_dir = os.path.join(os.path.dirname(os.path.abspath(args.out)), f"trainer_{args.cell}"),
                    per_device_train_batch_size = 2, gradient_accumulation_steps = 1, max_steps = 2,
                    learning_rate = 0.0, logging_steps = 1, bf16 = True, report_to = "none", seed = 0,
                    dataset_text_field = "text", max_length = 64, save_strategy = "no", dataset_num_proc = 1,
                    warmup_steps = 0, lr_scheduler_type = "constant", packing = False)
    trainer = SFTTrainer(model = model, train_dataset = Dataset.from_dict({"text": [MOESFT_TEXT] * 16}),
                         processing_class = tt, args = cfg)
    res["trainer_class"] = type(trainer).__name__
    res["aux_loss_enabled"] = getattr(trainer, "aux_loss_enabled", None)
    torch.cuda.reset_peak_memory_stats()
    trainer.train()
    peak()
    logs = [h for h in trainer.state.log_history if "loss" in h]
    finish_losses([float(h["loss"]) for h in logs])
    res["decreasing"] = None  # lr 0 by design


def cell_rpack():
    """unsloth #18435: probes/rpack_parity.py on tiny Qwen2.5-VL (padding_free packed vs per-document logits / CE / LoRA
    grads), with block_varlen_kwargs results (tuple = varlen in the compiled block, 'leave' = eager wrapper, None = dense
    mask) and eager-wrapper (_sdpa_packed_varlen_eager) calls counted during the packed forward + backward."""
    import subprocess
    versions()
    dt = "bf16" if args.cell.endswith("_bf16") else "fp32"
    outp = os.path.join(os.path.dirname(os.path.abspath(args.out)), f"parity_{args.cell}.json")
    stage("parity")
    p = subprocess.run([sys.executable, "-u", os.path.join(os.path.dirname(os.path.abspath(__file__)), "rpack_parity.py"),
                        "--model", "trl-internal-testing/tiny-Qwen2_5_VLForConditionalGeneration", "--dtype", dt, "--out", outp],
                       capture_output = True, text = True, encoding = "utf-8", errors = "replace", timeout = 1500,
                       cwd = os.path.dirname(os.path.abspath(args.out)))
    res["parity_rc"] = p.returncode
    res["parity_tail"] = (p.stdout or "")[-2500:] + "\n--- stderr ---\n" + (p.stderr or "")[-2500:]
    r = None
    if os.path.isfile(outp):
        with open(outp, encoding = "utf-8") as fh:
            r = json.load(fh)
    res["parity"] = r
    if r is None or r.get("refused"):
        raise RuntimeError(f"packed parity did not run (rc {p.returncode})")
    for k in ("route", "eager_wrapper_calls", "max_logit_maxabs", "grad_max_rel", "ce_packed", "ce_docs",
              "leak_others_maxabs", "varlen_stats_delta", "attn_ops", "unsloth_file", "block_modes"):
        res[k] = r.get(k)
    res["finite"] = True
    res["losses"] = []


def cell_fpskip(gc_mode):
    """unsloth #18442: tiny built Llama (4 layers) LoRA on the top 2 layers only, gradient checkpointing gc_mode.
    Step 0: loss + every LoRA grad (norm, sum, first values) on a fixed batch; frozen layer 0 / trainable layer 3
    forward calls per step; FROZEN_PREFIX_STATS; then 4 AdamW steps (losses)."""
    import torch
    from unsloth import FastLanguageModel
    versions()
    stage("load")
    model, tok = FastLanguageModel.from_pretrained(built_llama(), max_seq_length = 256, dtype = torch.bfloat16,
                                                   load_in_4bit = False)
    kw = dict(r = 16, lora_alpha = 32, lora_dropout = 0.0, bias = "none", target_modules = TARGETS,
              use_gradient_checkpointing = gc_mode, random_state = 0)
    try:
        model = FastLanguageModel.get_peft_model(model, finetune_last_n_layers = 2, **kw)
        res["layer_select"] = "finetune_last_n_layers=2"
    except TypeError:
        model = FastLanguageModel.get_peft_model(model, layers_to_transform = [2, 3], **kw)
        res["layer_select"] = "layers_to_transform=[2,3]"
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    res["lora_tensors"] = len(named)
    res["lora_layers"] = sorted({int(x) for n, _ in named for x in n.split(".") if x.isdigit()})
    layers = None
    for m in model.modules():
        if isinstance(getattr(m, "layers", None), torch.nn.ModuleList) and len(m.layers) == 4:
            layers = m.layers
            break
    calls = {0: 0, 3: 0}
    for i in calls:
        layers[i].register_forward_pre_hook(lambda mod, a, _i = i: calls.__setitem__(_i, calls[_i] + 1))
    dev = model.get_input_embeddings().weight.device
    batch = text_batch(tok, dev)
    try:
        import unsloth.models._frozen_prefix as fp
        st = fp.FROZEN_PREFIX_STATS
    except Exception:  # noqa: BLE001
        st = None
    s0 = dict(st) if st else None
    model.train()
    stage("train")
    torch.cuda.reset_peak_memory_stats()
    opt = torch.optim.AdamW([p for _, p in named], lr = args.lr)
    losses = []
    for step in range(4):
        before = dict(calls)
        out = model(**batch)
        out.loss.backward()
        if step == 0:
            res["step0_loss"] = float(out.loss.detach())
            g = {}
            for n, p in named:
                t = p.grad.detach().float() if p.grad is not None else None
                g[n.split("base_model.model.")[-1]] = None if t is None else \
                    [float(t.norm()), float(t.sum()), [round(float(x), 6) for x in t.flatten()[:4]]]
            res["step0_grads"] = g
            res["step0_grads_none"] = sum(1 for v in g.values() if v is None)
            res["layer_calls_step0"] = {str(k): calls[k] - before[k] for k in calls}
        opt.step()
        opt.zero_grad(set_to_none = True)
        losses.append(float(out.loss.detach()))
        res["losses_so_far"] = losses
        dump()
    res["frozen_prefix_stats_delta"] = {k: st[k] - s0[k] for k in st} if st else "absent"
    peak()
    finish_losses(losses)


def cell_dsv4g():
    """unsloth #18443: tiny DeepSeek-V4.1 LoRA (FastLanguageModel, GC 'unsloth'). Step 0 on a fixed batch: logits
    summary, loss, every trainable grad (norm, sum, first values) and bytes / count of tensors saved for backward
    (saved_tensors_hooks around the forward); then the standard 5-step AdamW loop."""
    import torch
    from unsloth import FastLanguageModel
    versions()
    stage("load")
    model, tok = FastLanguageModel.from_pretrained(MODELS["dsv4g"].split(" ")[0], max_seq_length = 256,
                                                   dtype = torch.bfloat16, load_in_4bit = False)
    model = FastLanguageModel.get_peft_model(model, r = 16, lora_alpha = 32, lora_dropout = 0.0, bias = "none",
                                             target_modules = TARGETS, use_gradient_checkpointing = "unsloth",
                                             random_state = 0)
    lora_stats(model)
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    batch = text_batch(tok, model.get_input_embeddings().weight.device)
    model.train()
    stage("step0")
    saved = {"n": 0, "bytes": 0, "fp32_bytes": 0}

    def pack(t):
        if isinstance(t, torch.Tensor):
            saved["n"] += 1
            b = t.numel() * t.element_size()
            saved["bytes"] += b
            if t.dtype == torch.float32:
                saved["fp32_bytes"] += b
        return t
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        out = model(**batch)
    lg = out.logits.detach().float() if getattr(out, "logits", None) is not None else None
    if lg is not None and lg.numel() > 0:
        res["step0_logits"] = [float(lg.sum()), float(lg.abs().sum()), [round(float(x), 6) for x in lg.flatten()[:6]]]
    res["step0_loss"] = float(out.loss.detach())
    out.loss.backward()
    g = {}
    for n, p in named:
        t = p.grad.detach().float() if p.grad is not None else None
        g[n.split("base_model.model.")[-1]] = None if t is None else \
            [float(t.norm()), float(t.sum()), [round(float(x), 6) for x in t.flatten()[:4]]]
    res["step0_grads"] = g
    res["step0_grads_none"] = sum(1 for v in g.values() if v is None)
    res["saved_for_backward"] = {k: (round(v / 2**20, 3) if "bytes" in k else v) for k, v in saved.items()}
    model.zero_grad(set_to_none = True)
    stage("train")
    torch.cuda.reset_peak_memory_stats()
    manual_loop(model, batch, args.steps)
    peak()


MG_SEQ = 16384  # >= every saved-attention auto threshold (sm75/sm80 8192, sm89 16384); 8 MiB GC offload slots at hidden 256


def _mg_state():
    """Memory-gate PR state: zoo #2619 / unsloth #18488 saved-attention auto routing and zoo #2620 exact pinned knob."""
    import torch
    out = {"hip": getattr(torch.version, "hip", None),
           "env_UNSLOTH_GC_SAVE_ATTENTION": os.environ.get("UNSLOTH_GC_SAVE_ATTENTION"),
           "env_UNSLOTH_GC_PINNED_EXACT": os.environ.get("UNSLOTH_GC_PINNED_EXACT")}
    try:
        out["capability"] = list(torch.cuda.get_device_capability())
    except Exception as e:  # noqa: BLE001
        out["capability"] = repr(e)[:120]
    try:
        from unsloth.utils import attention_dispatch as ad
        out["attention_dispatch_file"] = ad.__file__
        out["dispatch_SAVE_ATTENTION"] = getattr(ad, "_SAVE_ATTENTION", "absent")
        out["dispatch_MIN_SEQ"] = getattr(ad, "_SAVE_ATTENTION_MIN_SEQ", "absent")
        out["dispatch_has_auto_fn"] = hasattr(ad, "_auto_saves_attention")
    except Exception as e:  # noqa: BLE001
        out["attention_dispatch"] = repr(e)[:200]
    try:
        import unsloth_zoo.saved_attention as sa
        out["saved_attention_file"] = sa.__file__
        out["sa_setting"] = sa._save_attention_setting()
        out["sa_enabled"] = sa.save_attention_enabled()
        f = getattr(sa, "save_attention_auto_min_seq", None)
        out["auto_min_seq"] = f() if f is not None else "absent"
        out["auto_active"] = getattr(sa, "_AUTO_ACTIVE", ["absent"])[0]
        out["auto_min_seq_map"] = str(getattr(sa, "_AUTO_MIN_SEQ", "absent"))
        out["saved_attention_stats"] = dict(sa.SAVED_ATTENTION_STATS)
        out["budgets"] = len(getattr(sa, "_BUDGETS", {}) or {})
    except Exception as e:  # noqa: BLE001
        out["saved_attention"] = repr(e)[:200]
    try:
        import unsloth_zoo.gradient_checkpointing as g
        out["gc_file"] = g.__file__
        out["gc_PINNED_EXACT"] = getattr(g, "_GC_PINNED_EXACT", "absent")
        out["gc_has_exact_fn"] = hasattr(g, "_maybe_enable_exact_pinned")
        out["gc_EXACT_OUTGROWN"] = getattr(g, "_EXACT_OUTGROWN", "absent")
        out["gc_PINNED_BYTES"] = getattr(g, "GC_PINNED_BYTES", "absent")
        hs = getattr(g, "_host_allocator_settings", None)
        out["zoo_host_allocator_settings"] = hs() if hs is not None else "absent"
    except Exception as e:  # noqa: BLE001
        out["gradient_checkpointing"] = repr(e)[:200]
    getter = getattr(torch._C, "_accelerator_getAllocatorSettings", None)
    try:
        out["accelerator_allocator_settings"] = getter() if getter is not None else "absent"
    except Exception as e:  # noqa: BLE001
        out["accelerator_allocator_settings"] = repr(e)[:200]
    s = str(out.get("accelerator_allocator_settings")) + str(out.get("zoo_host_allocator_settings"))
    out["pinned_round_knob_present"] = "pinned_max_round_threshold_mb" in s
    try:
        hm = torch.cuda.host_memory_stats()
        out["host_memory_stats"] = {k: v for k, v in hm.items() if k in (
            "allocated_bytes.current", "allocated_bytes.peak", "reserved_bytes.current", "reserved_bytes.peak",
            "num_host_alloc", "num_host_free")}
    except Exception as e:  # noqa: BLE001
        out["host_memory_stats"] = repr(e)[:120]
    return out


def cell_mg(key):
    """zoo #2619 + unsloth #18488 / zoo #2620: one long-sequence LoRA run with GC "unsloth" (offloaded), 1 x 16384
    random tokens, AdamW 5 steps; per-step loss and LoRA grad norm, routed saved-attention calls (engagement),
    the PRs' state (auto threshold, pinned exact knob, allocator settings) and pinned GC bytes per step."""
    import torch
    from unsloth import FastLanguageModel
    versions()
    path = MODELS["gcdef_qwen3"] if "qwen3" in key else built_llama()
    stage("load")
    model, tok = FastLanguageModel.from_pretrained(path, max_seq_length = MG_SEQ, dtype = torch.bfloat16,
                                                   load_in_4bit = False)
    res["model_class"] = type(model).__name__
    model = FastLanguageModel.get_peft_model(model, r = 16, lora_alpha = 32, lora_dropout = 0.0, bias = "none",
                                             target_modules = TARGETS, use_gradient_checkpointing = "unsloth",
                                             random_state = 0)
    lora_stats(model)
    gc = GCCounter(model)
    res["gc_layers"] = gc.layers
    from unsloth.utils import attention_dispatch as ad
    routed = [0]
    if getattr(ad, "_saved_attention_call", None) is not None:
        orig = ad._saved_attention_call

        def counted(*a, **k):
            routed[0] += 1
            return orig(*a, **k)
        ad._saved_attention_call = counted
    res["mg_before"] = _mg_state()
    import unsloth_zoo.gradient_checkpointing as gmod
    device = model.get_input_embeddings().weight.device
    vocab = model.get_input_embeddings().weight.shape[0]
    ids = torch.randint(10, vocab, (1, MG_SEQ), generator = torch.Generator().manual_seed(0)).to(device)
    batch = {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": ids.clone()}
    params = [p for p in model.parameters() if p.requires_grad]
    res["trainable_params"] = int(sum(p.numel() for p in params))
    opt = torch.optim.AdamW(params, lr = args.lr)
    model.train()
    losses, gnorms, pinned = [], [], []
    stage("train")
    torch.cuda.reset_peak_memory_stats()
    for s in range(args.steps):
        out = model(**batch)
        out.loss.backward()
        gnorms.append(float(torch.linalg.vector_norm(torch.stack([p.grad.float().norm() for p in params
                                                                  if p.grad is not None]))))
        opt.step()
        opt.zero_grad(set_to_none = True)
        losses.append(float(out.loss.detach()))
        pinned.append(getattr(gmod, "GC_PINNED_BYTES", None))
        if s == 0:
            res["gc_calls_step0"] = gc.calls
        res["losses_so_far"] = losses
        dump()
    peak()
    res["grad_norms"] = gnorms
    res["gc_calls"] = gc.calls
    res["routed_saved_attention_calls"] = routed[0]
    res["gc_pinned_bytes_per_step"] = pinned
    res["mg"] = _mg_state()
    finish_losses(losses)


def cell_pk(mode):
    """unsloth #18494: tiny Qwen3 LoRA SFT, GA 4, eval_strategy steps (loss-only eval). mode pack = packing=True,
    pfree = padding_free=True. Engagement: per packed micro-batch moved by accelerate's send_to_device, whether the moved
    packed_seq_lengths carries _unsloth_lengths_tuple (head) and whether packing._attached_host_lengths returned it
    (the .tolist() skip); torch.cuda.empty_cache calls inside trainer.evaluate() (eval_accumulation_steps flush)."""
    import torch
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastLanguageModel
    versions()
    stage("load")
    model, tok = FastLanguageModel.from_pretrained(MODELS["gcdef_qwen3"], max_seq_length = 256, dtype = torch.bfloat16,
                                                   load_in_4bit = False)
    model = FastLanguageModel.get_peft_model(model, r = 16, lora_alpha = 32, lora_dropout = 0.0, bias = "none",
                                             target_modules = TARGETS, use_gradient_checkpointing = "unsloth",
                                             random_state = 0)
    tt = getattr(tok, "tokenizer", tok)
    if tt.pad_token is None:
        tt.pad_token = tt.eos_token
    texts = [" ".join(DOCS[: (k % len(DOCS)) + 1]) * (1 + k % 3) + (tt.eos_token or "") for k in range(160)]
    ds = Dataset.from_dict({"text": texts[:128]})
    eds = Dataset.from_dict({"text": texts[128:]})
    kw = dict(output_dir = os.path.join(os.path.dirname(os.path.abspath(args.out)), f"trainer_{args.cell}"),
              per_device_train_batch_size = 2, per_device_eval_batch_size = 2, gradient_accumulation_steps = 4,
              learning_rate = args.lr, logging_steps = 1, max_steps = args.steps, eval_strategy = "steps",
              eval_steps = max(1, args.steps // 2), bf16 = True, report_to = "none", seed = 0, dataset_text_field = "text",
              max_length = 128, save_strategy = "no", dataset_num_proc = 1, warmup_steps = 0, lr_scheduler_type = "constant")
    if mode == "pack":
        kw.update(packing = True)
    else:
        kw.update(padding_free = True, packing = False)
    cfg = SFTConfig(**kw)
    import accelerate.data_loader as adl
    from unsloth.utils import packing as pk
    eng = {"moves_with_lengths": 0, "moves_with_host_tuple": 0, "host_tuple_matches_values": 0,
           "attached_calls": 0, "attached_hits": 0, "stage_calls": 0, "eval_empty_cache_calls": 0,
           "train_empty_cache_calls": 0}
    res["pk_engagement"] = eng
    res["pk_has_carry_fn"] = hasattr(pk, "_install_host_lengths_carry")
    try:
        from unsloth.models import rl as rlm
        res["pk_has_eval_patch_fn"] = hasattr(rlm, "_patch_loss_only_eval_flush")
    except Exception as e:  # noqa: BLE001
        res["pk_has_eval_patch_fn"] = repr(e)[:120]
    phase = ["init"]
    orig_ec = torch.cuda.empty_cache

    def counted_ec(*a, **k):
        if phase[0] == "eval":
            eng["eval_empty_cache_calls"] += 1
        elif phase[0] == "train":
            eng["train_empty_cache_calls"] += 1
        return orig_ec(*a, **k)
    torch.cuda.empty_cache = counted_ec
    if hasattr(pk, "_attached_host_lengths"):
        orig_att = pk._attached_host_lengths

        def counted_att(seq_lengths):
            out = orig_att(seq_lengths)
            eng["attached_calls"] += 1
            eng["attached_hits"] += out is not None
            return out
        pk._attached_host_lengths = counted_att
    if hasattr(pk, "_stage_packed_metadata"):
        orig_stage = pk._stage_packed_metadata

        def counted_stage(*a, **k):
            eng["stage_calls"] += 1
            return orig_stage(*a, **k)
        pk._stage_packed_metadata = counted_stage
    stage("trainer")
    trainer = SFTTrainer(model = model, train_dataset = ds, eval_dataset = eds, processing_class = tt, args = cfg)
    res["trainer_class"] = type(trainer).__name__
    res["resolved_packing"] = getattr(trainer.args, "packing", None)
    res["resolved_padding_free"] = getattr(trainer.args, "padding_free", None)
    res["eval_accumulation_steps"] = getattr(trainer.args, "eval_accumulation_steps", None)
    res["send_to_device_wrapped"] = bool(getattr(adl.send_to_device, "_unsloth_host_lengths_carry", False))
    res["evaluation_loop_patched"] = bool(getattr(type(trainer).evaluation_loop, "_unsloth_loss_only_no_flush", False))
    inner = adl.send_to_device  # head: Unsloth's carry wrapper; base: accelerate's own

    def observed(tensor, *a, **k):
        moved = inner(tensor, *a, **k)
        try:
            if isinstance(moved, dict) and isinstance(moved.get("packed_seq_lengths"), torch.Tensor):
                eng["moves_with_lengths"] += 1
                t = getattr(moved["packed_seq_lengths"], "_unsloth_lengths_tuple", None)
                if t is not None:
                    eng["moves_with_host_tuple"] += 1
                    src = tensor.get("packed_seq_lengths") if isinstance(tensor, dict) else None
                    if isinstance(src, torch.Tensor) and tuple(int(x) for x in src.reshape(-1).tolist()) == t:
                        eng["host_tuple_matches_values"] += 1
        except Exception as e:  # noqa: BLE001
            eng["observe_error"] = repr(e)[:200]
        return moved
    adl.send_to_device = observed
    orig_eval = trainer.evaluate

    def wrapped_eval(*a, **k):
        prev = phase[0]
        phase[0] = "eval"
        try:
            return orig_eval(*a, **k)
        finally:
            phase[0] = prev
    trainer.evaluate = wrapped_eval
    stage("train")
    phase[0] = "train"
    torch.cuda.reset_peak_memory_stats()
    trainer.train()
    phase[0] = "done"
    peak()
    res["pk_engagement"] = eng
    hist = trainer.state.log_history
    res["global_steps"] = trainer.state.global_step
    res["grad_norms"] = [float(h["grad_norm"]) for h in hist if "loss" in h and "grad_norm" in h]
    res["eval_losses"] = [float(h["eval_loss"]) for h in hist if "eval_loss" in h]
    res["eval_runs"] = len(res["eval_losses"])
    finish_losses([float(h["loss"]) for h in hist if "loss" in h])
    res["decreasing"] = None


def cell_ls():
    """unsloth #18495 + zoo #2624: tiny Qwen3 LoRA SFT, label_smoothing_factor 0.1, GA 2. Records per-step loss / grad
    norm, engagement (unsloth fused_label_smoothing.STATS, zoo _apply_label_smoothing_override applications) and, on one
    fixed first train batch before training, this arm's trainer.compute_loss vs an fp32 LabelSmoother oracle."""
    import torch
    import torch.nn.functional as F
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastLanguageModel
    versions()
    stage("load")
    model, tok = FastLanguageModel.from_pretrained(MODELS["gcdef_qwen3"], max_seq_length = 256, dtype = torch.bfloat16,
                                                   load_in_4bit = False)
    model = FastLanguageModel.get_peft_model(model, r = 16, lora_alpha = 32, lora_dropout = 0.0, bias = "none",
                                             target_modules = TARGETS, use_gradient_checkpointing = "unsloth",
                                             random_state = 0)
    tt = getattr(tok, "tokenizer", tok)
    if tt.pad_token is None:
        tt.pad_token = tt.eos_token
    texts = [" ".join(DOCS[: (k % len(DOCS)) + 1]) * (1 + k % 3) + (tt.eos_token or "") for k in range(64)]
    ds = Dataset.from_dict({"text": texts})
    cfg = SFTConfig(output_dir = os.path.join(os.path.dirname(os.path.abspath(args.out)), f"trainer_{args.cell}"),
                    per_device_train_batch_size = 2, gradient_accumulation_steps = 2, learning_rate = args.lr,
                    logging_steps = 1, max_steps = min(args.steps, 4), label_smoothing_factor = 0.1, bf16 = True,
                    report_to = "none", seed = 0, dataset_text_field = "text", max_length = 128, save_strategy = "no",
                    dataset_num_proc = 1, warmup_steps = 0, lr_scheduler_type = "constant")
    zapp = [0]
    try:
        import unsloth_zoo.fused_losses.cross_entropy_loss as zce
        res["zoo_has_override"] = hasattr(zce, "unsloth_label_smoothing_override")
        if hasattr(zce, "_apply_label_smoothing_override"):
            orig_apply = zce._apply_label_smoothing_override

            def counted_apply(n_items, kwargs):
                out = orig_apply(n_items, kwargs)
                if out[1] is not kwargs:
                    zapp[0] += 1
                return out
            zce._apply_label_smoothing_override = counted_apply
    except Exception as e:  # noqa: BLE001
        res["zoo_has_override"] = repr(e)[:160]
    try:
        import unsloth.models.fused_label_smoothing as fls
        res["unsloth_fls_file"] = fls.__file__
    except Exception as e:  # noqa: BLE001
        fls = None
        res["unsloth_fls_file"] = f"absent ({type(e).__name__})"
    stage("trainer")
    trainer = SFTTrainer(model = model, train_dataset = ds, processing_class = tt, args = cfg)
    res["trainer_class"] = type(trainer).__name__
    res["label_smoother"] = type(getattr(trainer, "label_smoother", None)).__name__
    # Fixed first batch: this arm's compute_loss vs the fp32 oracle (LabelSmoother math on fp32 logits).
    stage("oracle")
    try:
        batch = trainer._prepare_inputs(next(iter(trainer.get_train_dataloader())))
        model.train()
        fwd = {k: v for k, v in batch.items() if k in ("input_ids", "attention_mask", "position_ids")}
        with torch.no_grad():
            logits = model(**fwd).logits.float()
            labels = batch["labels"]
            lg, lb = logits[:, :-1, :].reshape(-1, logits.shape[-1]), labels[:, 1:].reshape(-1)
            res["oracle_fp32_loss"] = float(F.cross_entropy(lg, lb, ignore_index = -100, label_smoothing = 0.1))
            res["oracle_fp32_loss_bf16logits"] = float(F.cross_entropy(lg.bfloat16().float(), lb, ignore_index = -100, label_smoothing = 0.1))
            res["oracle_active_tokens"] = int((lb != -100).sum())
        f0 = dict(fls.STATS) if fls is not None else None
        z0 = zapp[0]
        with torch.enable_grad():
            l = trainer.compute_loss(model, batch)
        res["fixed_batch_loss"] = float(l.detach().float())
        res["fixed_batch_engaged_unsloth"] = (fls.STATS.get("fused", 0) - f0.get("fused", 0)) if fls is not None else 0
        res["fixed_batch_engaged_zoo"] = zapp[0] - z0
        del l
        model.zero_grad(set_to_none = True)
        o = res["oracle_fp32_loss"]
        res["fixed_batch_abs_diff_vs_fp32"] = abs(res["fixed_batch_loss"] - o)
        res["fixed_batch_rel_diff_vs_fp32"] = abs(res["fixed_batch_loss"] - o) / max(1e-6, abs(o))
    except Exception as e:  # noqa: BLE001
        res["oracle_error"] = f"{type(e).__name__}: {e}"[:400]
        res["oracle_tb"] = traceback.format_exc()[-1500:]
    stats0 = dict(fls.STATS) if fls is not None else None
    z0 = zapp[0]
    stage("train")
    torch.cuda.reset_peak_memory_stats()
    trainer.train()
    peak()
    if fls is not None:
        res["fls_stats_train"] = {k: v - (stats0 or {}).get(k, 0) for k, v in fls.STATS.items()}
    else:
        res["fls_stats_train"] = "absent"
    res["ls_engaged_unsloth"] = res["fls_stats_train"].get("fused", 0) if isinstance(res["fls_stats_train"], dict) else 0
    res["ls_fallback_unsloth"] = res["fls_stats_train"].get("fallback", 0) if isinstance(res["fls_stats_train"], dict) else 0
    res["ls_engaged_zoo"] = zapp[0] - z0
    hist = trainer.state.log_history
    res["global_steps"] = trainer.state.global_step
    res["micro_batches"] = trainer.state.global_step * 2
    res["grad_norms"] = [float(h["grad_norm"]) for h in hist if "loss" in h and "grad_norm" in h]
    finish_losses([float(h["loss"]) for h in hist if "loss" in h])
    res["decreasing"] = None


def cell_lsh():
    """unsloth #18633: tiny Qwen3 with an UNMERGED PEFT LoRA adapter on the 7 projections AND lm_head (plain peft
    get_peft_model, as PeftModel.from_pretrained leaves it), SFT label_smoothing_factor 0.1, GA 2, 4 steps. Records the
    lm_head lora_A / lora_B grad norm per micro-batch (post-accumulate hooks), unsloth fused_label_smoothing.STATS."""
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastLanguageModel
    versions()
    stage("load")
    model, tok = FastLanguageModel.from_pretrained(MODELS["gcdef_qwen3"], max_seq_length = 256, dtype = torch.bfloat16,
                                                   load_in_4bit = False)
    res["tie_word_embeddings"] = bool(getattr(model.config, "tie_word_embeddings", None))
    torch.manual_seed(0)
    model = get_peft_model(model, LoraConfig(r = 16, lora_alpha = 32, lora_dropout = 0.0, bias = "none",
                                             target_modules = TARGETS + ["lm_head"], task_type = "CAUSAL_LM"))
    try:
        model = FastLanguageModel.get_peft_model(model, r = 16, lora_alpha = 32, lora_dropout = 0.0, bias = "none",
                                                 target_modules = TARGETS, use_gradient_checkpointing = "unsloth",
                                                 random_state = 0)
        res["lsh_route"] = "plain peft + FastLanguageModel.get_peft_model (existing adapters)"
    except Exception as e:  # noqa: BLE001
        res["lsh_route"] = f"plain peft only ({type(e).__name__}: {str(e)[:200]})"
    head = model.get_output_embeddings()
    res["lm_head_type"] = type(head).__name__
    hA = hB = None
    for n, p in model.named_parameters():
        if "lm_head" in n and "lora_A" in n:
            hA = p
        if "lm_head" in n and "lora_B" in n:
            hB = p
    res["lm_head_lora_found"] = [hA is not None, hB is not None]
    res["lm_head_lora_requires_grad"] = [bool(hA is not None and hA.requires_grad), bool(hB is not None and hB.requires_grad)]
    gA, gB = [], []
    if hA is not None:
        hA.register_post_accumulate_grad_hook(lambda p: gA.append(float(p.grad.float().norm()) if p.grad is not None else None))
        hB.register_post_accumulate_grad_hook(lambda p: gB.append(float(p.grad.float().norm()) if p.grad is not None else None))
    tt = getattr(tok, "tokenizer", tok)
    if tt.pad_token is None:
        tt.pad_token = tt.eos_token
    texts = [" ".join(DOCS[: (k % len(DOCS)) + 1]) * (1 + k % 3) + (tt.eos_token or "") for k in range(64)]
    ds = Dataset.from_dict({"text": texts})
    cfg = SFTConfig(output_dir = os.path.join(os.path.dirname(os.path.abspath(args.out)), f"trainer_{args.cell}"),
                    per_device_train_batch_size = 2, gradient_accumulation_steps = 2, learning_rate = args.lr,
                    logging_steps = 1, max_steps = min(args.steps, 4), label_smoothing_factor = 0.1, bf16 = True,
                    report_to = "none", seed = 0, dataset_text_field = "text", max_length = 128, save_strategy = "no",
                    dataset_num_proc = 1, warmup_steps = 0, lr_scheduler_type = "constant")
    try:
        import unsloth.models.fused_label_smoothing as fls
        res["unsloth_fls_file"] = fls.__file__
    except Exception as e:  # noqa: BLE001
        fls = None
        res["unsloth_fls_file"] = f"absent ({type(e).__name__})"
    stage("trainer")
    trainer = SFTTrainer(model = model, train_dataset = ds, processing_class = tt, args = cfg)
    res["trainer_class"] = type(trainer).__name__
    res["label_smoother"] = type(getattr(trainer, "label_smoother", None)).__name__
    stats0 = dict(fls.STATS) if fls is not None else None
    stage("train")
    torch.cuda.reset_peak_memory_stats()
    trainer.train()
    peak()
    if fls is not None:
        res["fls_stats_train"] = {k: v - (stats0 or {}).get(k, 0) for k, v in fls.STATS.items()}
    else:
        res["fls_stats_train"] = "absent"
    st = res["fls_stats_train"] if isinstance(res["fls_stats_train"], dict) else {}
    res["ls_engaged_unsloth"] = st.get("fused", 0)
    res["lm_head_lora_A_grad_norms"] = gA
    res["lm_head_lora_B_grad_norms"] = gB
    res["lm_head_lora_B_norm_after"] = float(hB.detach().float().norm()) if hB is not None else None
    hist = trainer.state.log_history
    res["global_steps"] = trainer.state.global_step
    res["grad_norms"] = [float(h["grad_norm"]) for h in hist if "loss" in h and "grad_norm" in h]
    finish_losses([float(h["loss"]) for h in hist if "loss" in h])
    res["decreasing"] = None


def cell_dora(four_bit, r = 16, half = False, fp16acc = False):
    """unsloth #18515: tiny Llama DoRA (use_dora=True, r 16, all 7 projections) bf16 or bitsandbytes NF4, 5 SFT steps,
    UnslothSFTTrainer. Engagement: unsloth.kernels.dora.STATS delta (factored / fallback) and a counting wrapper on
    peft DoraLinearLayer.forward (calls, whether the class forward carries _unsloth_factored_norm). After training, per
    DoRA module: row norm of W + s * B @ A in fp64 vs PEFT's stock get_weight_norm and (head) the factored norm."""
    import torch
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastLanguageModel
    from peft.tuners.lora import dora as peft_dora
    versions()
    cls = peft_dora.DoraLinearLayer
    res["dora_env"] = os.environ.get("UNSLOTH_DORA_FACTORED_NORM")
    res["dora_forward_patched"] = bool(getattr(cls.forward, "_unsloth_factored_norm", False))
    try:
        import unsloth.kernels.dora as kd
        res["dora_kernel_file"] = kd.__file__
    except Exception as e:  # noqa: BLE001
        kd = None
        res["dora_kernel_file"] = f"absent ({type(e).__name__})"
    calls = [0]
    inner = cls.forward
    if args.cell.endswith("_g64"):
        # Gold arm (identical code at both states): PEFT's forward with the row norm of W + s * B @ A computed in fp64
        # from the factors, then cast to the activation dtype. The best any bf16 path can do; base and head trajectories
        # are both judged by their distance to it.
        import torch.nn.functional as F

        def gold(self, x, *, lora_A, lora_B, scaling, base_layer, base_result = None, **kw):
            weight = peft_dora.dequantize_module_weight(base_layer).to(x.dtype)
            W = peft_dora.transpose(weight, self.fan_in_fan_out)
            with torch.no_grad():
                wn = torch.linalg.vector_norm(W.double() + scaling * (lora_B.weight.double() @ lora_A.weight.double()),
                                              dim = 1).to(x.dtype)
            mag = (self.weight / wn).view(1, -1)
            lora_result = lora_B(lora_A(x))
            if base_result is not None:
                bias = base_layer.bias
                if bias is not None:
                    base_result = base_result - bias
            else:
                base_result = F.linear(x, W)
            return (mag - 1) * base_result + mag * lora_result * scaling
        inner = gold
        res["dora_gold_fp64"] = True

    def counted(self, *a, **k):
        calls[0] += 1
        return inner(self, *a, **k)
    cls.forward = counted
    stage("load")
    wdt = torch.float16 if half else torch.bfloat16
    res["dora_rank"], res["dora_half"] = r, half
    if kd is not None and hasattr(kd, "_rank_dtype"):  # unsloth #18629: rank-term dtype actually chosen
        _rd = kd._rank_dtype
        seen = res.setdefault("rank_dtypes_seen", [])

        def _rd_spy(*a, **k):
            out = _rd(*a, **k)
            if str(out) not in seen:
                seen.append(str(out))
            return out
        kd._rank_dtype = _rd_spy
    if fp16acc:
        try:
            torch.backends.cuda.matmul.allow_fp16_accumulation = True
            res["fp16acc_set"] = bool(getattr(torch.backends.cuda.matmul, "allow_fp16_accumulation", None))
        except Exception as e:  # noqa: BLE001
            res["fp16acc_set"] = False
            res["fp16acc_error"] = f"{type(e).__name__}: {e}"[:300]
    model, tok = FastLanguageModel.from_pretrained(built_llama(), max_seq_length = 256, dtype = wdt,
                                                   load_in_4bit = four_bit)
    res["n_linear4bit"] = sum(1 for m in model.modules() if type(m).__name__ == "Linear4bit")
    model = FastLanguageModel.get_peft_model(model, r = r, lora_alpha = 2 * r, lora_dropout = 0.0, bias = "none",
                                             target_modules = TARGETS, use_gradient_checkpointing = "unsloth",
                                             random_state = 0, use_dora = True)
    lora_stats(model)
    res["dora_modules"] = sum(1 for m in model.modules() if type(m) is cls)
    tt = getattr(tok, "tokenizer", tok)
    if tt.pad_token is None:
        tt.pad_token = tt.eos_token
    ds = Dataset.from_dict({"text": [d + (tt.eos_token or "") for d in DOCS] * 8})
    cfg = SFTConfig(output_dir = os.path.join(os.path.dirname(os.path.abspath(args.out)), f"trainer_{args.cell}"),
                    per_device_train_batch_size = 2, gradient_accumulation_steps = 2,
                    max_steps = 8 if "_lr1" in args.cell else args.steps,
                    learning_rate = 1e-3 if "_lr1" in args.cell else args.lr, logging_steps = 1, bf16 = not half,
                    fp16 = half,
                    report_to = "none", seed = 0,
                    dataset_text_field = "text", max_length = 64, save_strategy = "no", dataset_num_proc = 1,
                    warmup_steps = 0, lr_scheduler_type = "constant", packing = False)
    res["dora_lr"], res["dora_steps"] = cfg.learning_rate, cfg.max_steps
    trainer = SFTTrainer(model = model, train_dataset = ds, processing_class = tt, args = cfg)
    res["trainer_class"] = type(trainer).__name__
    # #18515 c2eb72b: cross term path. ROCm has no aten::mm.dtype route (_MM_OUT_DTYPE False) -> chunked fp32 upcast.
    xt = {"cross_calls": 0, "norm_calls": 0, "nonfinite_norms": 0, "norm_dtypes": []}
    if kd is not None:
        res["dora_mm_out_dtype"] = getattr(kd, "_MM_OUT_DTYPE", "absent")
        if hasattr(kd, "_cross_term_fp32"):
            _ct = kd._cross_term_fp32

            def _ct_counted(*a, **k):
                xt["cross_calls"] += 1
                return _ct(*a, **k)
            kd._cross_term_fp32 = _ct_counted
        _fn = kd.factored_dora_weight_norm

        def _fn_counted(*a, **k):
            out = _fn(*a, **k)
            xt["norm_calls"] += 1
            if not bool(torch.isfinite(out).all()):
                xt["nonfinite_norms"] += 1
            if len(xt["norm_dtypes"]) < 1:
                xt["norm_dtypes"].append(str(out.dtype))
            return out
        kd.factored_dora_weight_norm = _fn_counted
    s0 = dict(kd.STATS) if kd is not None else None
    c0 = calls[0]
    stage("train")
    torch.cuda.reset_peak_memory_stats()
    trainer.train()
    peak()
    res["dora_forward_calls"] = calls[0] - c0
    res["dora_cross_term"] = dict(xt)
    if kd is not None:
        res["dora_stats_train"] = {k: v - (s0 or {}).get(k, 0) for k, v in kd.STATS.items()}
    else:
        res["dora_stats_train"] = "absent"
    st = res["dora_stats_train"] if isinstance(res["dora_stats_train"], dict) else {}
    res["dora_factored"] = st.get("factored", 0)
    res["dora_fallback"] = st.get("fallback", 0)
    logs = [h for h in trainer.state.log_history if "loss" in h]
    res["grad_norms"] = [h.get("grad_norm") for h in logs]
    finish_losses([float(h["loss"]) for h in logs])
    # Trained adapters (B != 0): row norm accuracy vs fp64 for every DoRA module.
    stage("norm_check")
    try:
        errs_peft, errs_fact, n = [], [], 0
        for m in model.modules():
            if not hasattr(m, "lora_magnitude_vector") or "default" not in getattr(m, "lora_magnitude_vector", {}):
                continue
            base_layer = m.get_base_layer()
            A, B, s = m.lora_A["default"].weight, m.lora_B["default"].weight, m.scaling["default"]
            dm = m.lora_magnitude_vector["default"]
            with torch.no_grad():
                W = peft_dora.transpose(peft_dora.dequantize_module_weight(base_layer), dm.fan_in_fan_out).to(wdt)
                ref = torch.linalg.vector_norm(W.double() + s * (B.double() @ A.double()), dim = 1)
                lw = (B.to(wdt) @ A.to(wdt))
                pn = dm.get_weight_norm(W, lw, s).double()
                errs_peft.append(float(((pn - ref).abs() / ref.clamp_min(1e-12)).max()))
                if kd is not None:
                    fn = kd.factored_dora_weight_norm(W, A, B, s, wdt).double()
                    errs_fact.append(float(((fn - ref).abs() / ref.clamp_min(1e-12)).max()))
                    fn32 = kd.factored_dora_weight_norm(W, A, B, s, torch.float32).double()
                    res.setdefault("_fact32", []).append(float(((fn32 - ref).abs() / ref.clamp_min(1e-12)).max()))
                res.setdefault("_bnorm", []).append(float(B.float().norm()))
            n += 1
        res["norm_modules"] = n
        res["norm_rel_err_peft_bf16"] = max(errs_peft) if errs_peft else None
        res["norm_rel_err_factored_bf16"] = max(errs_fact) if errs_fact else None
        res["norm_rel_err_factored_fp32"] = max(res.pop("_fact32")) if "_fact32" in res else None
        b = res.pop("_bnorm", [])
        res["lora_B_norm_min"] = min(b) if b else None
    except Exception as e:  # noqa: BLE001
        res["norm_error"] = f"{type(e).__name__}: {e}"[:400]
        res["norm_tb"] = traceback.format_exc()[-1500:]
    cls.forward = inner
    if fp16acc:
        res["fp16acc_after_train"] = bool(getattr(torch.backends.cuda.matmul, "allow_fp16_accumulation", None))


W20A_8L_CONFIG = {"architectures": ["Qwen3ForCausalLM"], "attention_bias": False, "attention_dropout": 0.0, "bos_token_id": 151643,
                  "eos_token_id": 151645, "head_dim": 64, "hidden_act": "silu", "hidden_size": 256, "initializer_range": 0.02,
                  "intermediate_size": 512, "max_position_embeddings": 40960, "max_window_layers": 8, "num_attention_heads": 4,
                  "num_hidden_layers": 8, "num_key_value_heads": 2, "rms_norm_eps": 1e-06, "rope_theta": 1000000,
                  "tie_word_embeddings": True, "use_cache": True, "use_sliding_window": False, "vocab_size": 151936}


def build_w20a_8l(d):
    """The W20A 8-layer tiny Qwen3 (B200 repro model): same config, seed-0 random bf16 weights, Qwen3-0.6B tokenizer."""
    import hashlib
    import torch
    from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM
    if not os.path.exists(os.path.join(d, "model.safetensors")):
        torch.manual_seed(0)
        m = Qwen3ForCausalLM(Qwen3Config(**W20A_8L_CONFIG)).to(torch.bfloat16)
        m.save_pretrained(d)
        AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B").save_pretrained(d)
    with open(os.path.join(d, "model.safetensors"), "rb") as fh:
        res["w20a_8l_sha256"] = hashlib.sha256(fh.read()).hexdigest()
    return d


def cell_cg(forced, eight = False):
    """unsloth #18496: scripts/w20a/dpo_cg_repro.py (tiny Qwen3 DPO, PEFT ref adapter, 8 steps) in a child process with
    this state's unsloth / zoo checkouts; UNSLOTH_BLOCK_CUDAGRAPHS=1 (forced) or unset (auto)."""
    import subprocess
    import unsloth
    import unsloth_zoo
    versions()
    src_u = os.path.dirname(os.path.dirname(os.path.abspath(unsloth.__file__)))
    src_z = os.path.dirname(os.path.dirname(os.path.abspath(unsloth_zoo.__file__)))
    here = os.path.dirname(os.path.abspath(args.out))
    result = os.path.join(here, f"cg_result_{args.cell}_{os.getpid()}.json")
    env = dict(os.environ, W20A_OUT_DIR = os.path.join(here, f"w20a_{args.cell}"), W20A_RESULT = result)
    env.pop("UNSLOTH_BLOCK_CUDAGRAPHS", None)
    if forced:
        env["UNSLOTH_BLOCK_CUDAGRAPHS"] = "1"
    steps = "8"
    if eight:
        env["REPO"] = build_w20a_8l(os.path.join(here, "w20a_tiny_qwen3_8l"))
        steps = "10"
    res["cg_repo"], res["cg_steps"] = env.get("REPO", "default tiny-Qwen3"), steps
    res["cg_env"] = env.get("UNSLOTH_BLOCK_CUDAGRAPHS")
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dpo_cg_repro.py")
    stage("child")
    p = subprocess.run([sys.executable, script, src_u, src_z, "dpo", steps], env = env, capture_output = True, text = True,
                       encoding = "utf-8", errors = "replace", timeout = 1200)
    res["cg_rc"] = p.returncode
    res["cg_stderr_tail"] = (p.stderr or "")[-2500:]
    r = {}
    if os.path.exists(result):
        with open(result, encoding = "utf-8") as fh:
            r = json.load(fh)
    res["cg_result"] = r
    res["cg_ok"] = r.get("ok")
    res["cg_error"] = r.get("error")
    res["cg_hooks"] = r.get("hooks")
    res["cg_auto"] = r.get("auto")
    res["cg_graphs_off"] = r.get("graphs_off")
    res["cg_graph_ids"] = r.get("graph_ids")
    res["grad_norms"] = r.get("grad_norms")
    finish_losses([float(x) for x in (r.get("losses") or [])] if r.get("ok") else [])
    res["decreasing"] = None


def cell_fx(dtype):
    """unsloth w20c-fp32-flex: tiny Gemma3ForConditionalGeneration (head_dim 256) FastModel load in float32 / bf16 in a child
    capped at 600 s (base may sit compiling flex in float32): attn_implementation, load / forward / train times, loss."""
    import subprocess
    versions()
    here = os.path.dirname(os.path.abspath(args.out))
    result = os.path.join(here, f"fx_result_{args.cell}_{os.getpid()}.json")
    env = dict(os.environ, W20C_DTYPE = dtype)
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fx_gemma3_child.py")
    stage("child")
    t0 = time.time()
    try:
        p = subprocess.run([sys.executable, script, result], env = env, capture_output = True, text = True,
                           encoding = "utf-8", errors = "replace", timeout = 600)
        res["fx_rc"], res["fx_timeout"] = p.returncode, False
        res["fx_stderr_tail"] = (p.stderr or "")[-2500:]
    except subprocess.TimeoutExpired as e:
        res["fx_rc"], res["fx_timeout"] = "timeout", True
        res["fx_stderr_tail"] = (e.stderr.decode("utf-8", "replace") if isinstance(e.stderr, bytes) else str(e.stderr or ""))[-2500:]
    res["fx_wall_s"] = round(time.time() - t0, 1)
    r = {}
    if os.path.exists(result):
        with open(result, encoding = "utf-8") as fh:
            r = json.load(fh)
    res["fx"] = r
    res["fx_attn"] = r.get("attn")
    res["fx_stage"] = r.get("stage")
    res["fx_load_s"] = r.get("load_s")
    res["fx_train_s"] = r.get("train_s")
    res["unsloth_file"] = r.get("unsloth_file", res.get("unsloth_file"))
    res["fx_loss"] = r.get("loss")
    finish_losses([float(r["loss"])] if r.get("stage") == "done" and r.get("loss") is not None else [])
    res["decreasing"] = None



def cell_cg2(kind):
    """unsloth #18607 + zoo #2630: LoRA r16 all 7 projections, GC 'unsloth', 5 SFT steps (bs 2, GA 1). Records the
    replay skip counter (unsloth.kernels.fast_lora.MLP_DOWN_SKIP_STATS, absent at base), per-LoRA-tensor grad norms of
    step 1 (post-accumulate hooks, pre-clip), trainer grad norms, block-compile modes and layer GC calls."""
    import torch
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastLanguageModel
    versions()
    four = kind == "llama4"
    path = MODELS["gcdef_qwen3"] if kind == "qwen3" else built_llama()
    res["skip_env"] = os.environ.get("UNSLOTH_GC_SKIP_MLP_DOWN")
    res["block_compile_env"] = os.environ.get("UNSLOTH_BLOCK_COMPILE")
    stage("load")
    # max_seq_length >= 512: below it apply_unsloth_gradient_checkpointing swaps "unsloth" for torch's own checkpoint
    # (round 1 used 256 and so never reached an Unsloth checkpointer). Batches stay 64 tokens.
    model, tok = FastLanguageModel.from_pretrained(path, max_seq_length = 1024, dtype = torch.bfloat16,
                                                   load_in_4bit = four)
    res["max_seq_length"] = 1024
    res["n_linear4bit"] = sum(1 for m in model.modules() if type(m).__name__ == "Linear4bit")
    model = FastLanguageModel.get_peft_model(model, r = 16, lora_alpha = 32, lora_dropout = 0.0, bias = "none",
                                             target_modules = TARGETS, use_gradient_checkpointing = "unsloth",
                                             random_state = 0)
    lora_stats(model)
    res["smart_gc_patched"] = getattr(torch.utils.checkpoint.CheckpointFunction, "__name__", "")
    gc = GCCounter(model)
    fl = sys.modules.get("unsloth.kernels.fast_lora")
    st = getattr(fl, "MLP_DOWN_SKIP_STATS", None)
    s0 = dict(st) if st is not None else None
    try:
        import unsloth_zoo.gradient_checkpointing as zgc
        res["zoo_has_replay_module"] = hasattr(zgc, "gradient_checkpoint_replay_module")
    except Exception as e:  # noqa: BLE001
        res["zoo_has_replay_module"] = f"ERR {type(e).__name__}"
    # Diagnostics (observe only): why the head does or does not arm the skip on each decoder-layer call.
    diag = {"arm_calls": 0, "arm_true": 0, "reasons": {}, "layer_forward": None, "gc_func": None}
    lm = sys.modules.get("unsloth.models.llama")
    try:
        lay = next(m for m in model.modules() if type(m).__name__.endswith("DecoderLayer"))
        diag["layer_forward"] = f"{type(lay).__name__}.forward={getattr(type(lay).forward, '__qualname__', '?')}"
        diag["gc_func"] = repr(lay.__dict__.get("_gradient_checkpointing_func"))[:200]
    except StopIteration:
        pass
    if lm is not None and hasattr(lm, "arm_mlp_output_unread"):
        _arm = lm.arm_mlp_output_unread

        def _arm_diag(layer, mlp, _arm = _arm):
            r = _arm(layer, mlp)
            diag["arm_calls"] += 1
            diag["arm_true"] += int(bool(r))
            if not r:
                try:
                    import unsloth_zoo.gradient_checkpointing as _z
                    tls = _z._GC_RECOMPUTE_TLS
                    fn = getattr(tls, "function", None)
                    rm = _z._replayed_module(fn) if fn is not None else None
                    key = (f"compiling={fl._is_compiling()} enabled={fl._SKIP_DOWN_ENABLED} "
                           f"active={getattr(tls, 'active', False)} fn={type(fn).__name__}:{str(getattr(fn, 'func', fn))[:70]} "
                           f"replayed={type(rm).__name__} is_layer={rm is layer} ghooks={len(torch.nn.modules.module._global_forward_hooks)} "
                           f"lhooks={len(layer._forward_hooks)} mhooks={len(mlp._forward_hooks)} grad={torch.is_grad_enabled()}")
                except Exception as e:  # noqa: BLE001
                    key = f"diag ERR {type(e).__name__}: {e}"[:200]
                diag["reasons"][key] = diag["reasons"].get(key, 0) + 1
                sk = f"stack_grad{int(torch.is_grad_enabled())}"
                if sk not in diag:
                    import threading
                    diag[sk] = [threading.current_thread().name] + [
                        f"{os.path.basename(fr.filename)}:{fr.lineno}:{fr.name}" for fr in traceback.extract_stack()[-28:-1]]
            return r
        lm.arm_mlp_output_unread = _arm_diag
    res["cg2_diag"] = diag
    g1 = {}
    hooks = []
    for n, p in model.named_parameters():
        if p.requires_grad:
            def _h(param, _n = n):
                if _n not in g1 and param.grad is not None:
                    g1[_n] = float(param.grad.detach().float().norm())
            hooks.append(p.register_post_accumulate_grad_hook(_h))
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
    ml = "_ml" in args.cell
    if ml:  # manual AdamW loop on one fixed batch: no Trainer (no gradient_checkpointing_enable re-bind)
        try:
            lay0 = next(m for m in model.modules() if type(m).__name__.endswith("DecoderLayer"))
            f0 = lay0.__dict__.get("_gradient_checkpointing_func")
            inner_f = (getattr(f0, "__kwdefaults__", None) or {}).get("_f", f0)
            res["gc_func_inner"] = repr(inner_f)[:300]
        except Exception as e:  # noqa: BLE001
            res["gc_func_inner"] = repr(e)[:200]
        manual_loop(model, text_batch(tok, model.get_input_embeddings().weight.device), args.steps, gc)
        trainer = None
    else:
        trainer = SFTTrainer(model = model, train_dataset = ds, processing_class = tt, args = cfg)
        trainer.add_callback(gc_callback(gc, model))
        res["trainer_class"] = type(trainer).__name__
        trainer.train()
        try:
            lay0 = next(m for m in model.modules() if type(m).__name__.endswith("DecoderLayer"))
            f0 = lay0.__dict__.get("_gradient_checkpointing_func")
            inner_f = (getattr(f0, "__kwdefaults__", None) or {}).get("_f", f0)
            res["gc_func_inner"] = repr(inner_f)[:300]
        except Exception as e:  # noqa: BLE001
            res["gc_func_inner"] = repr(e)[:200]
    peak()
    for h in hooks:
        h.remove()
    res["gc_calls"] = gc.calls
    res["block_modes"] = _block_modes(model)
    names = sorted(g1)
    res["step1_grad_norms"] = [g1[n] for n in names]
    res["step1_grad_names_n"] = len(names)
    res["step1_grad_names_sha"] = __import__("hashlib").sha256("\n".join(names).encode()).hexdigest()[:16]
    st = getattr(sys.modules.get("unsloth.kernels.fast_lora"), "MLP_DOWN_SKIP_STATS", None)
    res["mlp_down_skip_stats"] = ({k: v - (s0 or {}).get(k, 0) for k, v in st.items()} if st is not None else "absent")
    res["mlp_down_skipped"] = (res["mlp_down_skip_stats"].get("skipped", 0)
                               if isinstance(res["mlp_down_skip_stats"], dict) else 0)
    if ml:
        res["grad_norms"] = [math.sqrt(sum(v * v for v in res["step1_grad_norms"]))] if names else []
        return
    logs = [h for h in trainer.state.log_history if "loss" in h]
    res["grad_norms"] = [h.get("grad_norm") for h in logs]
    finish_losses([float(h["loss"]) for h in logs])


ERNIE_KW = dict(vocab_size = 103424, hidden_size = 64, intermediate_size = 128, moe_intermediate_size = 64,
                num_hidden_layers = 2, num_attention_heads = 4, num_key_value_heads = 2, moe_k = 2,
                moe_num_experts = 4, moe_num_shared_experts = 2, moe_layer_start_index = 1, moe_layer_end_index = 27,
                moe_layer_interval = 1, moe_norm_min = 1e-12, hidden_act = "silu", max_position_embeddings = 2048,
                rms_norm_eps = 1e-5, rope_theta = 500000.0, use_bias = False, bos_token_id = 1, eos_token_id = 2,
                pad_token_id = 0, tie_word_embeddings = True, output_router_logits = False)


def built_ernie():
    """Tiny ERNIE 4.5 MoE (the temp/tiny_cache tiny_make shape: hidden 64, 2 layers, 4 experts top-2, 2 shared),
    seeded random fp32 weights built on CPU with this state's transformers, saved fp32 (so a bf16 load keeps only the
    model-declared _keep_in_fp32_modules_strict params in fp32)."""
    import torch
    import transformers
    d = os.path.join(os.path.dirname(os.path.abspath(args.out)), f"tiny_ernie_tf{transformers.__version__}")
    if not os.path.isfile(os.path.join(d, "config.json")):
        cfg = transformers.AutoConfig.for_model("ernie4_5_moe", **ERNIE_KW)
        torch.manual_seed(0)
        model = transformers.AutoModelForCausalLM.from_config(cfg).float()
        tmp = f"{d}.tmp{os.getpid()}"
        model.save_pretrained(tmp)
        try:
            os.rename(tmp, d)
        except OSError:
            pass
    res["model_dir"] = d
    return d


def cell_z28(kind):
    """zoo #2628: bf16 full fine-tuning through unsloth_zoo prepare_model_for_training (full_finetuning=True,
    float32_mixed_precision=False, train_embedding / lm_head True; train_layernorms True, False for *_nl). Plain
    transformers model loaded dtype=bf16 on the GPU, 5 AdamW steps (lr 1e-3) on fixed random ids under bf16 autocast.
    Records the dtype map before / after prepare, which loaded-fp32 params stayed fp32 (and bitwise), losses."""
    import hashlib
    import torch
    import transformers
    from unsloth_zoo.training_utils import prepare_model_for_training
    versions()
    for k in ("UNSLOTH_MIXED_PRECISION", "UNSLOTH_DISABLE_FLOAT32_UPCAST", "UNSLOTH_UPCAST_LAYERNORM"):
        os.environ.pop(k, None)
    stage("build")
    d = built_llama() if kind == "llama" else built_ernie()
    stage("load")
    try:
        model = transformers.AutoModelForCausalLM.from_pretrained(d, dtype = torch.bfloat16)
    except TypeError:
        model = transformers.AutoModelForCausalLM.from_pretrained(d, torch_dtype = torch.bfloat16)
    model = model.to("cuda")
    res["keep_fp32_strict"] = str(getattr(model, "_keep_in_fp32_modules_strict", None))
    res["keep_fp32"] = str(getattr(model, "_keep_in_fp32_modules", None))
    loaded = {n: (str(p.dtype), p.detach().float().clone()) for n, p in model.named_parameters()}
    res["fp32_loaded"] = sorted(n for n, (dt, _) in loaded.items() if dt == "torch.float32")
    stage("prepare")
    prepare_model_for_training(model, use_gradient_checkpointing = False, use_reentrant = False, full_finetuning = True,
                               train_layernorms = kind != "ernie_nl", train_embedding = True, train_lm_head = True,
                               float32_mixed_precision = False)
    params = dict(model.named_parameters())
    after = {n: str(p.dtype) for n, p in params.items()}
    res["dtype_map"] = after
    res["dtype_map_sha"] = hashlib.sha256(json.dumps(after, sort_keys = True).encode()).hexdigest()[:16]
    res["fp32_after"] = sorted(n for n, dt in after.items() if dt == "torch.float32")
    res["router_dtypes"] = {n: dt for n, dt in after.items() if n.endswith("mlp.gate.weight") or "e_score_correction" in n}
    res["fp32_loaded_kept"] = sorted(n for n in res["fp32_loaded"] if after.get(n) == "torch.float32")
    res["fp32_kept_bitwise"] = all(torch.equal(params[n].detach().float(), loaded[n][1]) for n in res["fp32_loaded_kept"])
    res["trainable_dtypes"] = sorted({str(p.dtype) for p in params.values() if p.requires_grad})
    g = torch.Generator().manual_seed(1234)
    ids = torch.randint(3, 1000, (4, 32), generator = g).to("cuda")
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr = 1e-3)
    model.train()
    stage("train")
    torch.cuda.reset_peak_memory_stats()
    losses = []
    for _ in range(args.steps):
        with torch.autocast("cuda", dtype = torch.bfloat16):
            out = model(input_ids = ids, labels = ids, use_cache = False)
        out.loss.backward()
        opt.step()
        opt.zero_grad(set_to_none = True)
        losses.append(float(out.loss.detach()))
        res["losses_so_far"] = losses
        dump()
    peak()
    res["router_dtypes_after_train"] = {n: str(p.dtype) for n, p in model.named_parameters()
                                        if n.endswith("mlp.gate.weight") or "e_score_correction" in n}
    finish_losses(losses)



def cell_cg1(dtype_name):
    """unsloth #18614: scripts/cg1_tree_bench.py run_bench (in-process, harness torch, no install), unsloth/Qwen3-0.6B,
    correctness only: per-token logprobs + LoRA grads of a prefix-trie row, tree vs unroll vs each sample alone."""
    import importlib
    versions()
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    B = importlib.import_module("cg1_tree_bench")
    outj = os.path.join(os.path.dirname(os.path.abspath(args.out)), f"{args.cell}_bench.json")
    os.environ["CG1_OUT"] = outj
    try:
        pgk = importlib.import_module("unsloth.utils.prefix_grouper_kernel")
        res["torch_supports_prefix_grouper"] = getattr(pgk, "TORCH_SUPPORTS_PREFIX_GROUPER", "absent")
    except Exception as e:  # noqa: BLE001
        pgk = None
        res["torch_supports_prefix_grouper"] = f"ERR {type(e).__name__}: {e}"[:200]
    stage("bench")
    try:
        B.run_bench(["unsloth/Qwen3-0.6B"], 1, dtype_name, perf = False)
    finally:
        rl = sys.modules.get("unsloth.models.rl")
        res["tree_counts_total"] = dict(getattr(rl, "_ASYNC_GRPO_TREE_COUNTS", {}) or {}) if rl else None
        if pgk is not None and hasattr(pgk, "compile_check_failed"):
            res["compile_check_failed"] = safe(lambda: pgk.compile_check_failed())
        else:
            res["compile_check_failed"] = "absent"
        if os.path.isfile(outj):
            res["bench"] = json.load(open(outj, encoding = "utf-8"))
    m = (res.get("bench") or {}).get("models", {}).get("unsloth/Qwen3-0.6B") or {}
    c = m.get("correctness") or {}
    res["cg1"] = {mode: {k: v for k, v in (c.get(mode) or {}).items()} for mode in ("unroll", "tree")}
    res["cg1_tree_vs_unroll"] = m.get("tree_vs_unroll")
    res["finite"] = bool(c)



def cell_cg5(flag_on):
    """unsloth #18616 + zoo #2631: cg5_parity.py (tiny random Qwen3-MoE, FastModel LoRA, fp32 + bf16) in a child
    process (it imports unsloth itself); UNSLOTH_GRPO_PREFIX_GROUPER_MOE=1 for cg5_on, unset for cg5_off."""
    import subprocess
    versions()
    here = os.path.dirname(os.path.abspath(__file__))
    outj = os.path.join(os.path.dirname(os.path.abspath(args.out)), f"{args.cell}_parity.json")
    env = dict(os.environ, CG5_OUT = outj)
    env.pop("UNSLOTH_GRPO_PREFIX_GROUPER_MOE", None)
    if flag_on:
        env["UNSLOTH_GRPO_PREFIX_GROUPER_MOE"] = "1"
    stage("parity")
    p = subprocess.run([sys.executable, os.path.join(here, "cg5_parity.py")], env = env, capture_output = True,
                       text = True, encoding = "utf-8", errors = "replace", timeout = 1300)
    res["cg5_rc"] = p.returncode
    res["cg5_tail"] = (p.stdout[-1500:] + "\n" + p.stderr[-2500:])
    if os.path.isfile(outj):
        res["cg5"] = json.load(open(outj, encoding = "utf-8"))
    res["finite"] = os.path.isfile(outj)
    if not res["finite"]:
        raise RuntimeError(f"cg5_parity rc {p.returncode}: {p.stderr[-600:]}")



def cell_z12(model_key):
    """zoo #2612: scripts/z2612/mem_ab.py worker (bf16 autocast, 10 LoRA steps, conv I/O dtype hooks, eval with
    autocast on / off) in a child process importing this arm's unsloth + unsloth_zoo."""
    import subprocess
    import tempfile
    versions()
    here = os.path.dirname(os.path.abspath(__file__))
    env = dict(os.environ, Z2612_ARM = os.path.basename(os.path.dirname(os.path.abspath(args.out))),
               UNSLOTH_COMPILE_LOCATION = tempfile.mkdtemp(prefix = f"ucc_{args.cell}_"))
    stage("worker")
    p = subprocess.run([sys.executable, "-u", os.path.join(here, "z2612_mem_ab.py"), "--worker", "--model", model_key,
                        "--dtype", "bf16"], env = env, capture_output = True, text = True, encoding = "utf-8",
                       errors = "replace", timeout = 1400)
    res["z12_rc"] = p.returncode
    r = None
    for line in p.stdout.splitlines():
        if line.startswith("RESULT "):
            r = json.loads(line[7:])
    res["z12"] = r
    res["z12_tail"] = (p.stdout[-1500:] + "\n" + p.stderr[-2500:])
    if not r or r.get("error"):
        raise RuntimeError(f"mem_ab worker: {(r or {}).get('error') or p.stderr[-600:]}")
    finish_losses([float(x) for x in r.get("losses") or []])
    res["peak_alloc_mb"] = r.get("peak_warm_mib")


CELLS = {"kd_r16": lambda: cell_dora(False), "kd_r64": lambda: cell_dora(False, r = 64), "kd_h16acc": lambda: cell_dora(False, half = True, fp16acc = True), "lsh_qwen3": cell_lsh, "z12_whisper": lambda: cell_z12("whisper"), "z12_w2v": lambda: cell_z12("wav2vec2"), "cg5_on": lambda: cell_cg5(True), "cg5_off": lambda: cell_cg5(False), "cg1_fp32": lambda: cell_cg1("fp32"), "cg1_bf16": lambda: cell_cg1("bf16"), "cg2_qwen3_nbc_ml": lambda: cell_cg2("qwen3"), "cg2_qwen3_nbc_ml_off": lambda: cell_cg2("qwen3"), "cg2_llama_nbc_ml": lambda: cell_cg2("llama"), "cg2_llama4_nbc_ml": lambda: cell_cg2("llama4"), "cg2_qwen3": lambda: cell_cg2("qwen3"), "cg2_qwen3_nbc": lambda: cell_cg2("qwen3"), "cg2_qwen3_nbc_off": lambda: cell_cg2("qwen3"), "cg2_llama": lambda: cell_cg2("llama"), "cg2_llama_nbc": lambda: cell_cg2("llama"), "cg2_llama4_nbc": lambda: cell_cg2("llama4"), "z28_ernie": lambda: cell_z28("ernie"), "z28_ernie_nl": lambda: cell_z28("ernie_nl"), "z28_llama": lambda: cell_z28("llama"), "dora16": lambda: cell_dora(False), "dora16_off": lambda: cell_dora(False), "dora4": lambda: cell_dora(True), "dora16_g64": lambda: cell_dora(False), "dora16_lr1": lambda: cell_dora(False), "dora4_lr1": lambda: cell_dora(True), "dora16_lr1_g64": lambda: cell_dora(False), "dora4_lr1_g64": lambda: cell_dora(True), "dora4_g64": lambda: cell_dora(True), "fx32": lambda: cell_fx("float32"), "fx16": lambda: cell_fx("bfloat16"), "cg_forced8": lambda: cell_cg(True, True), "cg_auto8": lambda: cell_cg(False, True), "cg_forced": lambda: cell_cg(True), "cg_auto": lambda: cell_cg(False), "ls_qwen3": cell_ls, "ls_qwen3_off": cell_ls, "pk_qwen3": lambda: cell_pk("pack"), "pk_qwen3_ef1": lambda: cell_pk("pack"), "pf_qwen3": lambda: cell_pk("pfree"), "mgsa_llama": lambda: cell_mg("mgsa_llama"), "mgsa_qwen3": lambda: cell_mg("mgsa_qwen3"), "mgsa_llama_sa0": lambda: cell_mg("mgsa_llama"), "mgpin_llama": lambda: cell_mg("mgpin_llama"), "mgpin_llama_px0": lambda: cell_mg("mgpin_llama"),
         "hd512_llama": lambda: cell_hd512("llama"), "hd512_gemma4": lambda: cell_hd512("gemma4"),
         "hd512_gemma4_nc": lambda: cell_hd512("gemma4"),
         "bclens": cell_bclens, "bswap": cell_bswap,
         "dec_whisper": cell_dec_whisper, "dec_gemma3n": cell_dec_gemma3n,
         "moeaux_nr": cell_moeaux_nr, "moeaux_gc_qwen3moe": lambda: cell_moeaux_gc("moeaux_gc_qwen3moe"),
         "moeaux_gc_gptoss": lambda: cell_moeaux_gc("moeaux_gc_gptoss"), "gkd": cell_gkd,
         "moesft_qwen3moe": lambda: cell_moesft("moesft_qwen3moe"), "moesft_gptoss": lambda: cell_moesft("moesft_gptoss"),
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
         "gkd_proc_gemma3": lambda: cell_gkd_proc("gkd_proc_gemma3"), "mrope_pack": cell_mrope_pack, "orpo_txt": cell_orpo_txt, "lpad": cell_lpad, "lpadx": cell_lpadx, "lpadx_gemma2": cell_lpadx, "lpadx_gemma2_nbc": cell_lpadx, "rpack_bf16": cell_rpack, "dsv4g": cell_dsv4g, "fpskip_uns": lambda: cell_fpskip("unsloth"), "fpskip_true": lambda: cell_fpskip(True), "rpack_fp32": cell_rpack, "moesft_qwen35moe": lambda: cell_moesft_vlm("moesft_qwen35moe"), "dpo_txt_psforce": cell_dpo_txt, "fft16": cell_fft16, "fft16_sroff": cell_fft16, "grpo_zadv": lambda: cell_grpo_txt(True), "grpo_zadv_zoff": lambda: cell_grpo_txt(True), "orpo_txt_psoff": cell_orpo_txt, "orpo_txt_unwrap": lambda: cell_orpo_txt(True),
         "vision_gemma3": cell_vision,
         "pfree_llama": lambda: cell_sft_pack("llama", "pfree"), "pfree_qwen3": lambda: cell_sft_pack("qwen3", "pfree"),
         "sftdef_steps": lambda: cell_sft_pack("llama", "steps"), "sftdef_epoch": lambda: cell_sft_pack("llama", "epoch"),
         "plelens": lambda: cell_bclens("plelens", (37, 91, 150, 203, 64, 177, 119), "float32"),
         "seqcls": cell_seqcls, "tok_save": cell_tok_save, "fsdp2_io": cell_fsdp2_io, "fsdp2_gen": cell_fsdp2_gen, "pauto": cell_pauto, "tphost": cell_tphost, "atree": cell_atree, "grpo_pg": cell_grpo_pg, "grpo_pg_short": lambda: cell_grpo_pg_len(False), "grpo_pg_long": lambda: cell_grpo_pg_len(True), "grpo_pg_long_kl": lambda: cell_grpo_pg_len(True, 0.04), "fp32attn": cell_fp32attn, "warm": cell_warm, "grpo_pack": cell_grpo_pack,
         "g2nr_flex": cell_g2nr_flex, "gcnr_qwen3": lambda: cell_gc(True), "gcdef_qwen3": lambda: cell_gc(False),
         "gcnr_gemma3": lambda: cell_gc(True), "gcdef_gemma3": lambda: cell_gc(False), "lora_noac": cell_lora_noac,
         "st_minilm": lambda: cell_st(False), "st_qwen3": lambda: cell_st(True),
         "decode_diag": cell_decode_diag, "ctnvfp4": lambda: cell_ct("nvfp4"), "ctmxfp8": lambda: cell_ct("mxfp8"), "mxfp4probe": cell_mxfp4probe,
         "grpo_vl": cell_grpo_vl, "grpo_vl_dd0": cell_grpo_vl, "vlmpack_parity": cell_vlmpack_parity, "vlmpack_parity_bf16": cell_vlmpack_parity, "vlmpack_parity_k0": cell_vlmpack_parity, "vlmpack_sft": cell_vlmpack_sft, "vlmpack_sft_k0": cell_vlmpack_sft, "vlmpack_pad": cell_vlmpack_sft, "grpo_txt_ts1": cell_grpo_txt, "grpo_txt_ts1_kr0": cell_grpo_txt, "keptset_scorer": cell_keptset_scorer, "keptset_scorer_kr0": cell_keptset_scorer, "qat_int4": lambda: cell_qat("int4"), "qat_int8": lambda: cell_qat("int8"), "qat_int4_ri0": lambda: cell_qat("int4"), "qat_int8_ri0": lambda: cell_qat("int8"), "dpo_vl": cell_dpo_vl,
         "gcmem": cell_gcmem, "env": cell_env, "flm16": lambda: cell_flm(False), "flm16_gcsave": lambda: cell_flm(False), "flm4": lambda: cell_flm(True), "vision": cell_vision,
         "moe": cell_moe, "gemma4": cell_gemma4, "gemma4_gcrec": cell_gemma4_gcrec, "qwen38": cell_qwen38, "dsv4": cell_dsv4, "mllama": cell_mllama,
         "rl_decode": cell_rl_decode, "studiotok": cell_studiotok, "qvlmoe_aux": cell_qvlmoe_aux}


def main():
    if args.cell == "moe":
        os.environ["UNSLOTH_MOE_BACKEND"] = "grouped_mm"
    if args.cell.startswith("cg2_") and "_nbc" in args.cell:
        os.environ["UNSLOTH_BLOCK_COMPILE"] = "0"  # torch >= 2.11 compiles the block: the eager replay skip path needs it off
    if args.cell in ("cg2_qwen3_nbc_off", "cg2_qwen3_nbc_ml_off"):
        os.environ["UNSLOTH_GC_SKIP_MLP_DOWN"] = "0"  # unsloth #18607 kill switch (read at import)
    if args.cell == "dora16_off" or args.cell.endswith("_g64"):
        os.environ["UNSLOTH_DORA_FACTORED_NORM"] = "0"  # unsloth #18515 kill switch (read at import)
    if args.cell == "ls_qwen3_off":
        os.environ["UNSLOTH_FUSED_LABEL_SMOOTHING"] = "0"  # unsloth #18495 kill switch
    if args.cell.endswith("_ef1"):
        os.environ["UNSLOTH_EVAL_EMPTY_CACHE"] = "1"  # unsloth #18494 eval-flush opt-out
    if args.cell.endswith("_sa0"):
        os.environ["UNSLOTH_GC_SAVE_ATTENTION"] = "0"  # zoo #2619 kill switch
    if args.cell.endswith("_px0"):
        os.environ["UNSLOTH_GC_PINNED_EXACT"] = "0"  # zoo #2620 kill switch
    if args.cell.endswith("_gcsave"):
        os.environ["UNSLOTH_GC_SAVE_ATTENTION"] = "1"  # before any import: zoo #2571 reads it at checkpoint time
    if args.cell.endswith("_nbc"):
        os.environ["UNSLOTH_BLOCK_COMPILE"] = "0"  # Gemma2 block compile crashes in Inductor on gfx1151 (both arms)
    if args.cell == "g2nr_flex":
        os.environ["UNSLOTH_BLOCK_COMPILE"] = "0"  # unsloth #18219: eager flex (block compile off)
    if args.cell.endswith("_nc"):
        os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"  # diagnostic twin: same cell, no Unsloth compile
    if args.cell.endswith("_zoff"):
        os.environ["UNSLOTH_GRPO_SKIP_ZERO_ADVANTAGE"] = "0"  # zoo #2577 kill switch
    if args.cell.endswith("_sroff"):
        os.environ["UNSLOTH_BF16_STOCHASTIC_ROUNDING"] = "0"  # unsloth #18192 kill switch
    if args.cell.endswith("_psforce"):  # unsloth #18273: force prefix sharing on every pair batch (verify on)
        os.environ.update(UNSLOTH_DPO_PREFIX_SHARE = "1", UNSLOTH_DPO_PREFIX_SHARE_TOKR = "1.0",
                          UNSLOTH_DPO_PREFIX_SHARE_MIN_SAVED = "0", UNSLOTH_DPO_PREFIX_SHARE_VERIFY = "1")
    if args.cell.startswith("lpadx") or args.cell == "dsv4g":
        os.environ["UNSLOTH_RETURN_LOGITS"] = "1"
    if args.cell.endswith("_psoff"):
        os.environ["UNSLOTH_DPO_PREFIX_SHARE"] = "off"  # unsloth #18169 kill switch
    if args.cell in ("grpo_vl", "grpo_vl_dd0"):
        os.environ["UNSLOTH_GRPO_FUSED_WITHOUT_ENTROPY"] = "1"
    if args.cell.startswith("grpo_txt_ts1"):
        os.environ["UNSLOTH_GRPO_TRUNCATED_SUPPORT"] = "1"
    if args.cell.endswith("_kr0"):
        os.environ["UNSLOTH_GRPO_KEPT_SET_RECOMPUTE"] = "0"  # zoo #2595 kill switch
    if args.cell.endswith("_k0") and args.cell.startswith("vlmpack"):
        os.environ["UNSLOTH_VLM_MROPE_PACKING"] = "0"  # unsloth #18388 kill switch
    if args.cell.endswith("_ri0"):
        os.environ["UNSLOTH_QAT_ROUND_INPLACE"] = "0"  # unsloth #18339 kill switch
    if args.cell.endswith("_dd0"):
        os.environ["UNSLOTH_GRPO_IMAGE_DEDUP"] = "0"  # zoo #2590 kill switch
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
        try:
            CELLS[args.cell]()
        finally:
            try:  # W19C A10: engagement of unsloth's Inductor Triton-decomposition fallback (absent at base)
                from unsloth import import_fixes as _uif
                res["triton_eager_vals_fallbacks"] = dict(getattr(_uif, "STATS_TRITON_EAGER_VALS", {"absent": True}))
            except Exception as _e:  # noqa: BLE001
                res["triton_eager_vals_fallbacks"] = repr(_e)[:200]
            try:  # zoo #2617 engagement: Inductor pointwise launch-config autotune flag after unsloth's patch_torch_compile
                import torch._inductor.config as _ic
                res["inductor_flags"] = {"triton.autotune_pointwise": _ic.triton.autotune_pointwise,
                                         "max_autotune_pointwise": _ic.max_autotune_pointwise,
                                         "max_autotune": _ic.max_autotune}
            except Exception as _e:  # noqa: BLE001
                res["inductor_flags"] = repr(_e)[:200]
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
