#!/usr/bin/env python3
"""Criteria: tiny Unsloth training on gfx1151 (Linux / Windows), base vs head of one PR (or an unsloth + zoo pair).

Regression mode. Head is WORSE than base when any of:
  * a training cell finished with finite losses at the base and not at the head;
  * its losses were decreasing at the base and are not at the head;
  * both finished and a per-step loss differs beyond max(RTOL relative, 3 x the base A/A spread);
  * peak allocated memory (process-local) grew past 1.25 x base + 32 MiB;
  * layer gradient checkpointing ran at the base (calls > 0) and not at the head;
  * rl_decode: graphs matched eager after the optimizer step at the base and not at the head;
  * a test id fails / errors at the head that did not at the base (head-only files included).
Gates (non-vacuity): the environment built; every cell of each state imported unsloth and unsloth_zoo from that
state's checkouts and ran on a gfx1151 HIP device; the base flm16 cell trained to a verdict; selected tests ran.
"""
from __future__ import annotations

import os

TITLE = "Tiny Unsloth training (LoRA / QLoRA / vision / MoE grouped_mm + PR cells) on gfx1151, base vs head"
MODE = "regression"
NEEDS = ["rocm", "gpu", "nvidia", "multi_gpu", "windows"]
RTOL = 2e-2
MEM_RATIO, MEM_SLACK = 1.25, 32.0
TRAIN = ("z12_whisper", "z12_w2v", "cg2_qwen3_nbc_ml", "cg2_qwen3_nbc_ml_off", "cg2_llama_nbc_ml", "cg2_llama4_nbc_ml", 'cg2_qwen3', 'cg2_qwen3_nbc', 'cg2_qwen3_nbc_off', 'cg2_llama', 'cg2_llama_nbc', 'cg2_llama4_nbc', 'z28_ernie', 'z28_ernie_nl', 'z28_llama', "dora16", "dora16_off", "dora4", "dora16_g64", "dora4_g64", "dora16_lr1", "dora4_lr1", "dora16_lr1_g64", "dora4_lr1_g64", "fx32", "fx16", "cg_auto8", "cg_forced8", "cg_auto", "cg_forced", "ls_qwen3", "ls_qwen3_off", "pk_qwen3", "pk_qwen3_ef1", "pf_qwen3", "mgsa_llama", "mgsa_qwen3", "mgsa_llama_sa0", "mgpin_llama", "mgpin_llama_px0", "flm16", "flm4", "gcmem", "vision", "moe", "gemma4", "gemma4_gcrec", "qwen38", "dsv4", "mllama", "rl_decode", "ctnvfp4",
         "ctmxfp8", "mxfp4probe", "grpo_vl", "dpo_vl", "gcnr_qwen3", "gcdef_qwen3", "gcnr_gemma3", "gcdef_gemma3",
         "lora_noac", "st_minilm", "st_qwen3", "hd512_llama", "hd512_gemma4", "hd512_gemma4_nc", "bclens", "bswap", "dec_whisper", "dec_gemma3n", "dec_mistral", "moeaux_nr", "moeaux_gc_qwen3moe", "moeaux_gc_gptoss", "moesft_qwen3moe", "moesft_gptoss", "gkd", "granite_gc", "plelens", "padmix_qwen3", "padmix_gemma4", "padmix_gemma3", "padmix_gemma3_flex", "padmix_gemma3_flex_ones", "padmix_gemma4_ones", "grpo_txt", "grpo_zadv", "grpo_zadv_zoff", "pfree_llama", "pfree_qwen3", "sftdef_steps", "sftdef_epoch", "dpo_txt", "orpo_txt", "orpo_txt_psoff", "orpo_txt_unwrap", "qwen38_long", "sft_eos", "aol_qwen25vl", "aol_gemma3_unsloth", "aol_gemma3_trl", "aol_qwen25vl_none", "aol_gemma3_none", "cp_ring_kernels", "gkd_proc_qwen35", "gkd_proc_gemma3", "vision_gemma3", "mrope_pack", "seqcls", "tok_save", "fsdp2_io", "fsdp2_gen", "pauto", "tphost", "atree", "grpo_pg", "fp32attn", "warm", "flm16_gcsave", "lpad", "lpadx", "moesft_qwen35moe", "dpo_txt_psforce", "fft16", "fft16_sroff", "grpo_pack", "g2nr_flex", "studiotok", "qvlmoe_aux", "grpo_vl_dd0", "qat_int4", "qat_int8", "qat_int4_ri0", "qat_int8_ri0", "grpo_txt_ts1", "grpo_txt_ts1_kr0", "keptset_scorer", "keptset_scorer_kr0", "vlmpack_parity", "vlmpack_parity_bf16", "vlmpack_parity_k0", "vlmpack_sft", "vlmpack_sft_k0", "vlmpack_pad", "grpo_pg_short", "grpo_pg_long", "grpo_pg_long_kl", "lpadx_gemma2", "lpadx_gemma2_nbc", "fpskip_uns", "fpskip_true", "dsv4g", "rpack_bf16", "rpack_fp32")
# Cells whose numbers the PR changes by design (the CT loader fixes the expert values; RL sampling / the fused GRPO
# route; the probe's numbers are errors, not losses): no per-step loss / memory equality, the rest still applies.
NO_LOSS_EQ = ("z28_ernie", "z28_ernie_nl", "fx32", "cg_forced8", "cg_forced", "rpack_bf16", "rpack_fp32", "vlmpack_parity", "vlmpack_parity_bf16", "vlmpack_parity_k0", "vlmpack_sft", "vlmpack_sft_k0", "keptset_scorer", "keptset_scorer_kr0", "grpo_vl_dd0", "moesft_qwen35moe", "moesft_qwen3moe", "moesft_gptoss", "ctnvfp4", "ctmxfp8", "mxfp4probe", "grpo_vl", "sftdef_epoch", "aol_qwen25vl", "aol_gemma3_unsloth", "aol_gemma3_trl", "aol_qwen25vl_none", "aol_gemma3_none", "sft_eos", "g2nr_flex")
ENGAGED = {"dpo_txt_psforce": "prefix_share_engaged", "mxfp4probe": "engaged", "grpo_vl": "grpo_fused_engaged", "grpo_vl_dd0": "grpo_fused_engaged", "dpo_vl": "dpo_chunked_engaged"}

# Why a PR's own new path does or does not engage on ROCm; rendered with the table.
NOTES = {
    "u18515": "DoRA factored weight row norm (unsloth #18515): dora16 (bf16) / dora4 (bitsandbytes NF4) train a tiny Llama "
              "with use_dora=True, r 16 on all 7 projections, 5 SFT steps, GA 2, GC 'unsloth'. Engagement at head: "
              "unsloth.kernels.dora.STATS['factored'] == DoraLinearLayer.forward calls > 0, fallback 0; base has no module "
              "(0 factored, stock forward calls > 0). Losses within the loss tolerance (bf16-level differences allowed: the "
              "norm is computed differently), grad norms within 5e-2 rel. dora16_off = UNSLOTH_DORA_FACTORED_NORM=0 kill "
              "switch: 0 factored and losses / grad norms equal base. After training, every DoRA module's row norm is "
              "compared with an fp64 W + s B A reference (PEFT stock at both arms, factored at head). The PR's new test "
              "file is copied from head into base.",
    "u18498": "float32 + text head_dim > 128 skips flex (unsloth #18498, w20c-fp32-flex): fx32 / fx16 load tiny "
              "Gemma3ForConditionalGeneration (text head_dim 256) with FastModel in float32 / bf16 in a child capped at 600 s, "
              "then 2 no-grad forwards and one train forward + backward. Head fx32 must finish without flex_attention "
              "(Gemma 3 falls back to eager); base fx32 may sit compiling flex (timeout) or may already avoid flex on ROCm. "
              "fx16 must pick the same attn_implementation and loss as base. The PR's new test is copied from head into base.",
    "u18496": "Block CUDA graphs + adapter switch (unsloth #18496): cg_forced runs scripts/w20a/dpo_cg_repro.py (tiny Qwen3 "
              "DPO, PEFT ref adapter, 8 steps) with UNSLOTH_BLOCK_CUDAGRAPHS=1 in a child process on this state's checkouts; "
              "head must train, base may crash (or ROCm may refuse graphs). cg_auto runs it with the variable unset: auto "
              "keeps graphs off on ROCm, so losses / grad norms / auto state must equal base exactly. The PR's new CPU test "
              "is copied from head into base.",
    "u18495": "label_smoothing_factor through the fused CE (unsloth #18495 + zoo #2624, both at head; both merge bases at "
              "base): ls_qwen3 trains tiny Qwen3 LoRA with label_smoothing_factor 0.1, GA 2. Engagement at head: "
              "fused_label_smoothing.STATS['fused'] == micro-batches, zoo override applied > 0, no fallback; 0 at base. "
              "Before training, one fixed batch: this arm's compute_loss vs an fp32 oracle (F.cross_entropy(label_smoothing) "
              "on fp32 logits). Losses may differ at bf16 level (stock LabelSmoother log_softmax runs on bf16 logits). "
              "ls_qwen3_off = UNSLOTH_FUSED_LABEL_SMOOTHING=0 kill switch, must equal base exactly. New test files of both "
              "PRs are copied from head into base.",
    "u18494": "Packed SFT host lengths + loss-only eval flush (unsloth #18494): pk_qwen3 (packing=True) and pf_qwen3 "
              "(padding_free=True) train tiny Qwen3 LoRA, GA 4, with a loss-only eval every half run. Engagement at head: every "
              "packed micro-batch moved by accelerate's send_to_device carries _unsloth_lengths_tuple equal to the CPU values and "
              "packing._attached_host_lengths returns it (no .tolist()); torch.cuda.empty_cache inside evaluate() is 0 at head, "
              ">0 at base. pk_qwen3_ef1 = UNSLOTH_EVAL_EMPTY_CACHE=1 opt-out (flushes again at head). Losses / grad norms / eval "
              "losses must equal base. The PR's new test file is copied from head into the base arm.",
    "zsa2619": "Saved-attention auto (zoo #2619 + unsloth #18488): save_attention_auto_min_seq() returns None on HIP, so the "
               "automatic mode never routes on gfx1151. mgsa_llama / mgsa_qwen3 train 1 x 16384 tokens (>= every threshold "
               "in the map) with GC 'unsloth'; routed_saved_attention_calls and SAVED_ATTENTION_STATS are the engagement "
               "counters; mgsa_llama_sa0 is the =0 kill switch. Losses / grad norms / graphs must equal base.",
    "zpin2620": "Exact-size pinned GC offload (zoo #2620): gated to CUDA (not HIP) torch >= 2.14, so on gfx1151 _GC_PINNED_EXACT "
                "must stay False and the allocator settings must not gain pinned_max_round_threshold_mb. mgpin_llama trains "
                "1 x 16384 tokens at hidden 256 (8 MiB offload slots, above the 4 MiB exact threshold) with GC 'unsloth'. "
                "The PR's new test file is copied from head into the base arm.",
    "u18271": "MoE VLM wrapper router aux coefficient on TRL 1.18: moesft_qwen35moe trains tiny Qwen3.5-MoE (VLM wrapper, "
              "text-only rows) through Unsloth's SFTTrainer at lr 0 with SFTConfig(router_aux_loss_coef 0.9) and the "
              "checkpoint coefficient left as shipped; step-1 loss vs nll + 0.9 aux (keeps_aux) and vs nll + checkpoint_coef aux. "
              "Base expected to train on the checkpoint coefficient, head on 0.9, config restored after. moesft_qwen3moe "
              "(text MoE, untouched path) is no-regression. The PR's CPU tests are copied from head into the base arm.",
    "u18433": "NF4 dequant cache: the PR's three new test files are copied from head into the base arm (dense-model host "
              "read, autocast on the compiled route, in-place scale invalidation); flm4 (QLoRA NF4) and flm16 (16-bit LoRA, "
              "the dense case the hook no longer installs on) train 5 steps.",
    "u18434": "QAT weight memo guards + re-round after reload: the PR's new tests are copied from head into base; qat_int4 / "
              "qat_int8 train tiny Qwen3 with QAT (torchao) at both states.",
    "u18435": "ROCm packed attention leaves the compiled block when varlen is refused: rpack_bf16 runs tiny Qwen2.5-VL "
              "padding_free packed vs per-document parity in bf16 and counts block_varlen_kwargs results (tuple = varlen in "
              "the compiled block, leave = eager wrapper) and eager-wrapper calls during the packed forward + backward. "
              "rpack_fp32 is the control (fp32 rows keep the route). The PR's new CPU test is copied from head into base.",
    "u18436": "VLM packing guards: the PR's tests (torch<2.6 refusal simulated via transformers' flag; spawn-worker pickling, "
              "which is the native default on Windows) are copied from head into base; vlmpack_pad / vlmpack_sft / "
              "vlmpack_parity train or check the packed path at both states (transformers 5.23, where the gate passes).",
    "u18437": "Gemma2 left-pad training masks: lpadx_gemma2 = tiny Gemma2 LoRA via FastLanguageModel, same rows right vs left "
              "padded; leak = max logit change at the pad positions + first real position when later tokens change (base > 0 "
              "expected on the Gemma2 sliding / global branch, head 0); real-token logits must match base. lpadx (Llama, "
              "shared branch, already fixed by #18272) must stay at leak 0. The PR's tests are copied from head into base.",
    "z2607": "Vision dedup: no rank-local self-check under sharded towers, fp32 duplicate grad sums, drop-path random. The "
             "zoo test file is copied from head into the base arm; grpo_vl / grpo_vl_dd0 train tiny Qwen2.5-VL GRPO with "
             "dedup on / off. ZeRO-3 / multi-rank FSDP2 hang not reachable on one GPU.",
    "u18442": "Frozen-prefix skip under gradient checkpointing: fpskip_uns / fpskip_true = tiny built Llama (4 layers), LoRA on "
              "the top 2 (finetune_last_n_layers=2), GC 'unsloth' / True; step-0 loss and every LoRA grad norm must match base "
              "(rel 1e-2, loss 1e-4), layer_calls_step0 counts frozen layer 0 vs trainable layer 3 forwards (head expected "
              "fewer layer-0 calls), frozen_prefix_stats_delta is the PR's engagement counter. Test copied from head into base.",
    "u18443": "DeepSeek-V4.1 hc_pre keeps only the bf16 stream for backward: dsv4g = tiny DeepSeek-V4.1 LoRA, step-0 logits / "
              "loss / every LoRA grad vs base (loss 1e-4, grad norm rel 1e-2, bitwise equality reported separately) and the "
              "bytes autograd saves during the forward (head expected lower, fp32 share lower); then 5 AdamW steps. The PR's "
              "test is copied from head into base.",
    "u18272": "Left-padded training, pad queries attend only themselves: lpadx = tiny Llama LoRA, same rows padded right and "
              "left with the first real label kept; right-padded loss / grad norm and real-token logits (lse + 8 logits per "
              "real position, both sides) must equal base; leak = max logit change at the pad positions + first real "
              "position when later tokens change (base > 0 when the SDPA mask unmasks the fully masked pad row, head 0). "
              "The PR's tests are copied from head into the base arm.",
    "u18273": "DPO prefix-share verify on a split-GPU model: gfx1151 has one GPU, so the split itself is not reachable (Kaggle "
              "2x T4 covers it). The PR's new test runs here with hidden states on the CPU (one GPU, no device spoof); "
              "dpo_txt_psforce forces prefix sharing (UNSLOTH_DPO_PREFIX_SHARE=1, "
              "TOKR 1.0, MIN_SAVED 0) so the changed verify runs on one device: engaged count and losses must match base.",
    "u17785": "Distributed FSDP2 / TP / PP for Qwen3.8 / DeepSeek-V4.1: multi-GPU paths unreachable on a one-GPU box. "
              "Linux runs with --spoof-devices (device_count 2, wiring only); the PR's gloo tests are CPU. Single-GPU "
              "training is real.",
    "m17817": "Gemma-4 large-head attention kernel: patch_large_head_attention returns False on ROCm "
              "(torch.version.hip gate), so the new kernel never engages here; the gemma4 cell proves Gemma-4 "
              "training is unchanged.",
    "u17826": "DCP cross-version optimizer checkpoint message: CPU-side logic, covered by its test; training cells are "
              "the no-regression check.",
    "p17827_2525": "Qwen3.8 packing / GDN (unsloth) + QSA / fla fp32 (zoo): qwen38 cell trains with packing=True "
                   "(varlen hit counters), fla runs through Triton on ROCm.",
    "u17828": "Decode CUDA graphs are default-off on ROCm (torch.version.hip gate); rl_decode forces them "
              "(UNSLOTH_DECODE_CUDA_GRAPHS=1) and checks graphs see the post-step weights.",
    "p17829_2526": "Block compile on transformers 5.23 + grouped MoE compile: Dynamo graph breaks / recompiles "
                   "recorded per cell; block CUDA graphs are off on ROCm.",
    "u18230": "SFT MoE router aux loss on TRL >= 1.18 (here 1.18.0, where the change engages): moesft_qwen3moe / moesft_gptoss "
              "train a tiny MoE through Unsloth's SFTTrainer at lr 0 with router_aux_loss_coef 0.9 and compare the step-1 loss "
              "with stock TRL (own process, no unsloth) and with an nll / nll + coef * aux oracle on the same batch. Base is "
              "expected to drop the aux loss (equals nll only), head to keep it (equals stock TRL); dense cells must be "
              "unchanged. The PR's CPU test is copied from head into the base arm.",
    "u18219": "Eager flex_attention under non-reentrant checkpointing: g2nr_flex trains tiny Gemma-2 (softcap -> flex) "
              "in fp32 with block compile off and non-reentrant GC; g2_grad_rel = LoRA grads with GC vs the same model with "
              "layer GC off (base defect: large or CheckpointError; head: ~0). fw_bw_graph_calls_under_hooks > 0 proves the "
              "eager flex trace ran inside the checkpoint hooks. The PR's GPU test is copied from head into the base arm.",
    "z2582": "transformers >= 5.23 per-class forward_with_checkpointing re-applied after zoo patches replace a "
             "GradientCheckpointingLayer forward (patch_function, Gemma-4 forced-float32 boundary). gfx1151 trains bf16, so "
             "the Gemma-4 forced-float32 boundary is not installed here; gemma4_gcrec counts decoder MLP forwards per step "
             "(2L = recomputed) with a GC-off control arm (L), and lists GradientCheckpointingLayer classes whose forward is no "
             "longer the 5.23 wrapper after unsloth's bf16 patches. The PR's CPU test is copied from head into base.",
    "z2581": "gpt-oss MXFP4 float32 experts output / residual when training in float16: engages only for fp16 training on "
             "MXFP4 gpt-oss, which no cell here runs (gfx1151 cells train bf16; moeaux_gc_gptoss is a bf16 tiny gpt-oss "
             "without MXFP4), so every cell is no-regression evidence only. The PR's CPU test is copied from head into base.",
    "u17701": "Studio fork veto (host-RAM / Block Swap): studiotok imports Studio's backend from each state's checkout, "
              "records the trainer's num_proc routing with the real host-memory reading (tiny model, so no veto is "
              "expected and the routed num_proc must equal base), the veto on synthetic oversize / Block Swap inputs "
              "(head only; absent at base), sha256 of the tokenized rows by the routed map, in process and via the online "
              "view, then 2 LoRA SFT steps on the routed rows. Rows, routing and losses must match base. On Windows "
              "dataset_map_num_proc returns None (no fork) at both states. The PR's CPU tests are copied from head into base.",
    "z2587_t457": "Qwen3-VL-MoE router logits on transformers 4.57.6 (the defect version): qvlmoe_aux loads tiny Qwen3-VL-MoE "
                  "through Unsloth (compiled), forward with output_router_logits=True (aux_loss finite, router_logits "
                  "attached), then 2 SFT steps at lr 0 with output_router_logits on. Base expected to fail the forward / "
                  "train; head to train. The PR's CPU test is copied from head into base.",
    "z2587_t5": "Qwen3-VL-MoE router logits on transformers 5.x: the new wrapper is a no-op there; qvlmoe_aux and the "
                "standard cells are no-regression evidence. The PR's CPU test is copied from head into base.",
    "u18299": "AsyncGRPO fast_inference decoding during the adapter save: CPU fake-engine tests (vLLM is not installed on "
              "the runners; skips are listed per test). The head's test_async_grpo_fast_inference.py replaces the base copy. "
              "Standard cells are no-regression evidence only (no async GRPO / vLLM decode on gfx1151 here).",
    "u18294": "Pipeline-parallel Gemma-4 per-layer inputs in the source table dtype under forced float32: the PP path needs "
              "2 ranks, covered by the gloo CPU test (2 processes) and the new CPU test (copied from head into base). gfx1151 "
              "cells train bf16 on one GPU, so the gemma4 and standard cells are no-regression evidence only.",
    "u17830": "FastLanguageModel layer GC on transformers 5.23: gc_calls counts the layer checkpoint function "
              "(0 = checkpointing dropped).",
    "z2522": "triton.use_block_ptr gated on Triton having block pointers: env cell records the inductor flag and "
             "whether this Triton has make_block_ptr.",
    "z2523": "Mllama vision layer accepts hidden_states=: mllama cell calls the patched layer both ways on the GPU.",
    "p17848_2529": "compressed-tensors NVFP4 / MXFP8 MoE experts packed for LoRA: ctnvfp4 / ctmxfp8 load a tiny CT "
                   "checkpoint through FastLanguageModel and train LoRA on attention + experts; engagement = packed "
                   "expert parameter types and zoo PACKED_EXPERT_STATS. Base losses differ by design (the fix changes "
                   "the loaded expert values).",
    "z2528": "MXFP4 fused grouped GEMM gated on Triton >= 3.1: mxfp4probe records the gate, availability, and a direct "
             "fwd + bwd against dequantize + matmul on gfx1151.",
    "p17850_2530": "Vision GRPO fused loss (zoo) + chunked VLM DPO head (unsloth): grpo_vl counts fused_linear_grpo_loss "
                   "calls, dpo_vl counts trainer._unsloth_dpo_chunked_calls (the chunked head needs TRL < 0.29, see the "
                   "_trl024 corner). Mllama block CUDA graphs: block CUDA graphs are off on ROCm, so that part does not "
                   "engage here.",
    "p17850_2530_trl024": "The TRL 0.24 / transformers 4.57.6 corner of #17850 + zoo #2530, where DPOTrainer still has "
                          "concatenated_forward and the chunked vision DPO head can engage.",
    "u17857": "Block compile with non-reentrant checkpointing: gcnr_* run LoRA with use_reentrant = False, gcdef_* the "
              "Unsloth default; block modes and every 'block compile failed' message per state.",
    "z2535": "Vanilla LoRA stays in the base dtype without autocast: lora_noac (FastModel bf16, no autocast) records the "
             "LoRA output dtype and that every LoRA tensor gets a non-zero gradient; flm16 / vision are the Trainer paths.",
    "u17858": "Tensor logits_to_keep: its test, plus the plain LoRA SFT cells.",
    "u17859": "FastSentenceTransformer: st_minilm (encoder: lean LoRA, prefetching loader) and st_qwen3 (decoder "
              "embedder; packing needs flash-attn, absent on ROCm, so it stays off) train 5 steps.",
    "u17864": "Block compile drops Inductor's SDPA 8-alignment guard: bclens trains one LoRA step at lengths 37/64/101/72 "
              "with block compile; losses must match base and recompile_logs must not rise.",
    "z2536": "Block swap inside the checkpoint (transformers 5.23): unsloth refuses offload_layers on gfx1151 (unified memory, "
             "see unsloth_offload_refusal), so bswap installs zoo BlockSwap directly: offload 2 vs 0 loss / grads, slot-pool growth.",
    "p17827_2525b": "Re-run at the new heads: #2525 skips the seq_idx GDN conv test on ROCm; qwen38 + GEN cells.",
    "p17871_2537": "Decode CUDA graphs for Whisper / CSM / shared-KV Gemma 3n: off on ROCm (_Cuda.available() needs "
                   "torch.version.hip None); dec_whisper / dec_gemma3n greedy-generate via unsloth_base_fast_generate with the "
                   "default env and UNSLOTH_DECODE_CUDA_GRAPHS=1, tokens must equal base.",
    "u17860": "head_dim 512 attention tiles: the kernel refuses ROCm (lh_patch=False), so hd512_llama / hd512_gemma4 "
              "(Gemma-4-style global_head_dim 512) LoRA steps must train identically at base and head.",
    "u17861": "DDP find_unused default for a LoRA sequence classifier: single-process Qwen3 seq-cls LoRA steps plus the "
              "default the head computes; the new unit test.",
    "u17862": "Tokenizer save under world > 1: its gloo test, plus a single-process LoRA save / reload (loss and tokenizer "
              "round trip).",
    "z2524": "DeepSeek-V4.1 FP4 experts through the fused MXFP4 grouped GEMM: needs an MXFP4 checkpoint and a "
             "native-FP4 GPU; gfx1151 has neither, so the route does not engage here (dsv4 cell trains bf16).",
}


def _norm(p):
    return os.path.normcase(os.path.normpath(str(p or "")))


def _cells(o):
    return (o or {}).get("cells") or {}


def _arch(o):
    for c in _cells(o).values():
        a = (c.get("versions") or {}).get("arch")
        if a:
            return str(a)
    return ""


def _trained(c):
    return bool(c) and c.get("stage") == "done" and c.get("finite") is True


def gates(obs):
    out = []
    for name in ("base", "head"):
        o = obs.get(name) or {}
        env = o.get("env") or {}
        out.append((f"{name} environment built", not env.get("error") and bool((env.get("env") or {}).get("hip")),
                    str(env.get("error") or env.get("env"))[:300]))
        paths = o.get("paths") or {}
        bad = []
        for cname, c in _cells(o).items():
            for mod, key in (("unsloth", "unsloth_file"), ("zoo", "unsloth_zoo_file")):
                f = c.get(key)
                if f and paths.get(mod) and not _norm(f).startswith(_norm(paths[mod])):
                    bad.append(f"{cname}:{mod}={f}")
        files = sorted({str(c.get("unsloth_file")) for c in _cells(o).values() if c.get("unsloth_file")})
        out.append((f"{name} cells imported this state's unsloth + unsloth_zoo", bool(files) and not bad,
                    ("; ".join(bad) or ", ".join(files))[:300] or "no cell imported unsloth"))
        out.append((f"{name} ran on gfx1151 (HIP)", "gfx1151" in _arch(o), _arch(o) or "no device recorded"))
        for key in ("pytest", "pytest_companion"):
            p = o.get(key)
            if p is not None and name == "head":
                n = p.get("n_passed", 0) + p.get("n_failed", 0) + p.get("n_skipped", 0)
                out.append((f"head {key} collected tests", n > 0 or not p.get("selected"),
                            f"selected={p.get('selected')} passed={p.get('n_passed')} failed={p.get('n_failed')} "
                            f"skipped={p.get('n_skipped')} crashed={p.get('crashed_files')}"[:300]))
    hf = (obs.get("head") or {}).get("head_files")
    if hf is not None:
        out.append(("head file overrides applied (local fixes)", bool(hf) and not any(r.get("error") for r in hf),
                    "; ".join(f"{r['repo']}:{r['path']} {r.get('error') or 'ok'}" for r in hf)[:300]))
    # Non-vacuity: the base actually trained something, so "no cell got worse" compared real runs. A cell that
    # fails at BOTH states is a finding reported in the table and the verdict text, not a gate.
    # unsloth #18494: the base arm must show the defect (else VOID): no host tuple on the moved lengths, eval flushes > 0.
    for cname in ("pk_qwen3", "pf_qwen3"):
        c = _cells(obs.get("base")).get(cname)
        if c is not None:
            e = c.get("pk_engagement") or {}
            out.append((f"base {cname} shows the defect (eval flushes > 0, no host tuple, packed moves seen)",
                        _trained(c) and (e.get("eval_empty_cache_calls") or 0) > 0 and (e.get("moves_with_lengths") or 0) > 0
                        and not e.get("moves_with_host_tuple") and not e.get("attached_hits"), str(e)[:300]))
    for cname in ("dora16", "dora4", "dora16_lr1", "dora4_lr1"):
        c = _cells(obs.get("base")).get(cname)
        if c is not None and _trained(c):
            out.append((f"base {cname} shows the defect (stock DoRA forward ran, nothing factored)",
                        (c.get("dora_forward_calls") or 0) > 0 and not c.get("dora_factored")
                        and not c.get("dora_forward_patched"),
                        f"calls={c.get('dora_forward_calls')} factored={c.get('dora_factored')} "
                        f"patched={c.get('dora_forward_patched')} file={c.get('dora_kernel_file')}"[:300]))
    c = _cells(obs.get("base")).get("ls_qwen3")
    if c is not None:
        out.append(("base ls_qwen3 shows the defect (stock LabelSmoother, nothing fused)",
                    _trained(c) and not c.get("ls_engaged_unsloth") and not c.get("ls_engaged_zoo"),
                    f"unsloth={c.get('ls_engaged_unsloth')} zoo={c.get('ls_engaged_zoo')} fls={c.get('unsloth_fls_file')}"[:300]))
    trained = sorted(n for n, c in _cells(obs.get("base")).items() if n in TRAIN and _trained(c))
    out.append(("base trained at least one cell (finite losses)", bool(trained), ", ".join(trained) or "none"))
    return out


def _aa_spread(base, cell):
    a, b = (_cells(base).get(cell) or {}).get("losses"), (_cells(base).get(cell + "@aa") or {}).get("losses")
    if not a or not b or len(a) != len(b):
        return None
    return max(abs(x - y) / max(1.0, abs(x)) for x, y in zip(a, b))


def _loss_tol(base, cell):
    spreads = [s for s in (_aa_spread(base, c) for c in ("flm16", "moe", cell)) if s is not None]
    return max([RTOL] + [3 * s for s in spreads])


def _pyt(o):
    p = (o or {}).get("pytest") or {}
    q = (o or {}).get("pytest_companion") or {}
    fails = set(p.get("failed") or []) | set(p.get("errors") or []) | set(q.get("failed") or []) | set(q.get("errors") or [])
    passed = set(p.get("passed") or []) | set(q.get("passed") or [])
    crashed = set(p.get("crashed_files") or []) | {"companion:" + x for x in (q.get("crashed_files") or [])}
    return passed, fails, crashed


def head_is_worse(base, head):
    worse, fixed, both = [], [], []
    bc, hc = _cells(base), _cells(head)
    for cell in TRAIN:
        b, h = bc.get(cell), hc.get(cell)
        if b is None and h is None:
            continue
        if _trained(b) and not _trained(h):
            worse.append(f"{cell}: trained at base, not at head ({(h or {}).get('error') or (h or {}).get('stage')})"[:300])
            continue
        if not _trained(b) and _trained(h):
            fixed.append(cell)
            continue
        if not _trained(b) and not _trained(h):
            both.append(f"{cell} ({str((h or {}).get('error') or (h or {}).get('stage'))[:120]})")
            continue
        if b.get("decreasing") and h.get("decreasing") is False:
            worse.append(f"{cell}: loss decreasing at base, not at head {h.get('losses')}")
        lb, lh = b.get("losses") or [], h.get("losses") or []
        tol = _loss_tol(base, cell)
        if cell in NO_LOSS_EQ:
            lb = lh = []
        if len(lb) == len(lh) and lb:
            d = max(abs(x - y) / max(1.0, abs(x)) for x, y in zip(lb, lh))
            if d > tol:
                worse.append(f"{cell}: per-step loss differs by {d:.3g} > tol {tol:.3g}")
        elif lb or lh:
            worse.append(f"{cell}: {len(lb)} base vs {len(lh)} head loss steps")
        pb, ph = b.get("peak_alloc_mb"), h.get("peak_alloc_mb")
        if cell in ENGAGED and b.get(ENGAGED[cell]) and not h.get(ENGAGED[cell]):
            worse.append(f"{cell}: engaged at base, not at head")
        if cell not in NO_LOSS_EQ and isinstance(pb, (int, float)) and isinstance(ph, (int, float)) and ph > pb * MEM_RATIO + MEM_SLACK:
            worse.append(f"{cell}: peak allocated {ph} MiB vs base {pb} MiB")
        if (b.get("gc_calls") or 0) > 0 and not (h.get("gc_calls") or 0) > 0:
            worse.append(f"{cell}: layer gradient checkpointing ran at base ({b.get('gc_calls')} calls), 0 at head")
        if cell == "fx16":
            if b.get("fx_attn") != h.get("fx_attn"):
                worse.append(f"fx16: bf16 attn_implementation base {b.get('fx_attn')} vs head {h.get('fx_attn')}")
            if b.get("fx_loss") != h.get("fx_loss"):
                worse.append(f"fx16: bf16 loss base {b.get('fx_loss')} vs head {h.get('fx_loss')}")
        if cell in ("cg_auto", "cg_auto8"):
            for k in ("losses", "grad_norms", "cg_auto", "cg_graphs_off", "cg_graph_ids"):
                if b.get(k) != h.get(k):
                    worse.append(f"cg_auto: {k} base {b.get(k)} vs head {h.get(k)}"[:300])
        if cell in ("dora16", "dora4", "dora16_off", "dora16_g64", "dora4_g64", "dora16_lr1", "dora4_lr1", "dora16_lr1_g64", "dora4_lr1_g64"):
            vb, vh = b.get("grad_norms") or [], h.get("grad_norms") or []
            gold = hc.get(cell + "_g64") if not cell.endswith(("_g64", "_off")) else None

            def _dev(x, g, k):
                xs, gs = (x or {}).get(k) or [], (g or {}).get(k) or []
                if not xs or len(xs) != len(gs):
                    return None
                return max(abs(p - q) / max(1e-6, abs(q)) for p, q in zip(xs, gs))
            if len(vb) != len(vh) or not vb:
                worse.append(f"{cell}: grad_norms count base {len(vb)} vs head {len(vh)}")
            elif _trained(gold):
                # Step-local judge: step 2 is the first step whose forward sees B != 0 (the only place the norm path
                # differs from one step's worth of state); it must sit within bf16 eps (3.9e-3) of the fp64-norm gold
                # arm. Later steps compound bf16 rounding through the optimizer (trajectory, recorded, not gated
                # beyond the generic loss tolerance).
                bg = b.get("grad_norms") or []
                for k in ("grad_norms", "losses"):
                    db, dh = _dev(b, gold, k), _dev(h, gold, k)
                    h[f"_dev_gold_{k}"], h[f"_dev_gold_{k}_base"] = dh, db
                    xs, gs, bs = h.get(k) or [], gold.get(k) or [], b.get(k) or []
                    if len(xs) < 2 or len(gs) < 2 or len(bs) < 2:
                        worse.append(f"{cell}: fewer than 2 {k} steps for the step-2 check")
                        continue
                    r2h, r2b = abs(xs[1] - gs[1]) / max(1e-6, abs(gs[1])), abs(bs[1] - gs[1]) / max(1e-6, abs(gs[1]))
                    h[f"_step2_rel_gold_{k}"], h[f"_step2_rel_gold_{k}_base"] = r2h, r2b
                    if xs[0] != bs[0]:
                        worse.append(f"{cell}: step-1 {k} differs base {bs[0]} vs head {xs[0]} (B == 0 there: must be bitwise)")
                    if r2h > 3.9e-3:
                        worse.append(f"{cell}: step-2 {k} {r2h:.3g} rel from the fp64-norm gold at head (base {r2b:.3g})")
            elif max(abs(x - y) / max(1e-6, abs(x)) for x, y in zip(vb, vh)) > 5e-2:
                worse.append(f"{cell}: grad norms differ base {vb} vs head {vh}"[:300])
            calls, fac = h.get("dora_forward_calls") or 0, h.get("dora_factored") or 0
            if cell.endswith("_g64"):
                if b.get("losses") != h.get("losses") or vb != vh or fac:
                    worse.append(f"{cell}: gold arm (identical code) not bitwise equal base vs head (factored {fac})")
            elif cell == "dora16_off":
                aa = _aa_spread(base, "dora16")
                lb, lh = b.get("losses") or [], h.get("losses") or []
                exact = aa == 0.0
                if fac or h.get("dora_forward_patched") or not calls:
                    worse.append(f"dora16_off: kill switch did not restore the stock forward (factored {fac}, patched "
                                 f"{h.get('dora_forward_patched')}, calls {calls})")
                if exact and (lb != lh or vb != vh):
                    worse.append(f"dora16_off: kill switch not bitwise equal to base (A/A bitwise) losses {lb} vs {lh}"[:300])
            else:
                if "_lr1" in cell:  # stable cells: base vs head within bf16 eps on every step (fixed before c2eb72b results)
                    for k in ("losses", "grad_norms"):
                        xs, bs = h.get(k) or [], b.get(k) or []
                        if not xs or len(xs) != len(bs):
                            worse.append(f"{cell}: {k} step count base {len(bs)} vs head {len(xs)}")
                            continue
                        d = max(abs(x - y) / max(1e-6, abs(y)) for x, y in zip(xs, bs))
                        h[f"_lr1_rel_{k}"] = d
                        if d > 3.9e-3:
                            worse.append(f"{cell}: stable-lr {k} base vs head {d:.3g} > bf16 eps 3.9e-3")
                xt = h.get("dora_cross_term")
                if isinstance(xt, dict):  # c2eb72b+: cross term counted; on HIP it must be the chunked fp32 path
                    if h.get("dora_mm_out_dtype") is not False:
                        worse.append(f"{cell}: aten::mm.dtype route selected on HIP ({h.get('dora_mm_out_dtype')}): chunked path not used")
                    if not (xt.get("cross_calls") or 0) >= fac or not fac or xt.get("nonfinite_norms"):
                        worse.append(f"{cell}: chunked cross term not engaged on every factored norm or non-finite {xt} factored {fac}"[:300])
                if not (calls > 0 and fac == calls and not h.get("dora_fallback")):
                    worse.append(f"{cell}: factored norm not used on every DoRA forward at head (calls {calls}, factored "
                                 f"{fac}, fallback {h.get('dora_fallback')})")
                ef, ep = h.get("norm_rel_err_factored_bf16"), b.get("norm_rel_err_peft_bf16")
                if not isinstance(ef, (int, float)):
                    worse.append(f"{cell}: no factored-norm accuracy check at head ({h.get('norm_error')})"[:300])
                elif ef > max(1e-2, 2 * (ep if isinstance(ep, (int, float)) else 0)):
                    worse.append(f"{cell}: factored norm rel err {ef:.3g} vs fp64 (PEFT stock {ep})")
        if cell in ("ls_qwen3", "ls_qwen3_off"):
            vb, vh = b.get("grad_norms") or [], h.get("grad_norms") or []
            if len(vb) != len(vh) or not vb:
                worse.append(f"{cell}: grad_norms count base {len(vb)} vs head {len(vh)}")
            elif max(abs(x - y) / max(1e-6, abs(x)) for x, y in zip(vb, vh)) > 5e-2:
                worse.append(f"{cell}: grad norms differ base {vb} vs head {vh}"[:300])
            if cell == "ls_qwen3":
                mb = h.get("micro_batches") or 0
                if not (mb > 0 and h.get("ls_engaged_unsloth") == mb and (h.get("ls_engaged_zoo") or 0) >= mb
                        and not h.get("ls_fallback_unsloth")):
                    worse.append(f"ls_qwen3: fused label smoothing not engaged on every micro-batch at head "
                                 f"(unsloth {h.get('ls_engaged_unsloth')}, zoo {h.get('ls_engaged_zoo')}, fallback "
                                 f"{h.get('ls_fallback_unsloth')}, micro-batches {mb})")
                rb, rh = b.get("fixed_batch_rel_diff_vs_fp32"), h.get("fixed_batch_rel_diff_vs_fp32")
                if not isinstance(rh, (int, float)):
                    worse.append(f"ls_qwen3: no fixed-batch fp32 comparison at head ({h.get('oracle_error')})"[:300])
                elif rh > max(4e-3, 2 * (rb if isinstance(rb, (int, float)) else 0)):
                    worse.append(f"ls_qwen3: fixed-batch loss {rh:.3g} rel from fp32 at head vs {rb} at base")
            else:
                lb, lh = b.get("losses") or [], h.get("losses") or []
                if lb != lh or vb != vh or h.get("ls_engaged_unsloth") or h.get("ls_engaged_zoo"):
                    worse.append(f"ls_qwen3_off: kill switch not identical to base (losses {lb} vs {lh}, engaged "
                                 f"{h.get('ls_engaged_unsloth')}/{h.get('ls_engaged_zoo')})"[:300])
        if cell in ("pk_qwen3", "pf_qwen3", "pk_qwen3_ef1"):
            for k in ("grad_norms", "eval_losses"):
                vb, vh = b.get(k) or [], h.get(k) or []
                if len(vb) != len(vh) or not vb:
                    worse.append(f"{cell}: {k} count base {len(vb)} vs head {len(vh)}")
                elif max(abs(x - y) / max(1.0, abs(x)) for x, y in zip(vb, vh)) > tol:
                    worse.append(f"{cell}: {k} differ base {vb} vs head {vh}"[:300])
            e = h.get("pk_engagement") or {}
            n = e.get("moves_with_lengths") or 0
            if not (n > 0 and e.get("moves_with_host_tuple") == n and e.get("host_tuple_matches_values") == n
                    and (e.get("attached_hits") or 0) > 0):
                worse.append(f"{cell}: host lengths not carried on every packed micro-batch at head {e}"[:300])
            if cell == "pk_qwen3_ef1":
                if not (e.get("eval_empty_cache_calls") or 0) > 0:
                    worse.append(f"{cell}: UNSLOTH_EVAL_EMPTY_CACHE=1 did not restore the eval flush at head {e}"[:300])
            elif (e.get("eval_empty_cache_calls") or 0) != 0:
                worse.append(f"{cell}: loss-only eval still flushed {e.get('eval_empty_cache_calls')} times at head")
        if cell == "gemma4_gcrec" and b.get("recomputed") and not h.get("recomputed"):
            worse.append(f"gemma4_gcrec: decoder layers recomputed at base, not at head ({h.get('recompute_summary')})")
        if cell == "gemma4_gcrec" and len(((h.get("lost_checkpointing") or {}).get("lost") or [])) > len(((b.get("lost_checkpointing") or {}).get("lost") or [])):
            worse.append(f"gemma4_gcrec: more layer classes lost checkpointing at head {(h.get('lost_checkpointing') or {}).get('lost')}"[:300])
        if cell.startswith("moeaux_gc") and b.get("aux_gc_ok") and not h.get("aux_gc_ok"):
            worse.append(f"{cell}: aux-loss grad norm GC vs no GC within 2e-2 at base, not at head ({h.get('aux_gc_rel')})")
        if cell.startswith("moesft") and b.get("moesft_matches_stock") and not h.get("moesft_matches_stock"):
            worse.append(f"{cell}: step-1 SFT loss matched stock TRL at base, not at head (rel {h.get('rel_vs_stock')})")
        if cell.startswith("moesft") and b.get("moesft_keeps_aux") and not h.get("moesft_keeps_aux"):
            worse.append(f"{cell}: router aux loss kept at base, not at head ({h.get('step1_loss')} vs oracle {h.get('oracle_nll_plus_aux')})")
        if cell.startswith("lpadx"):
            def _mx(k):
                a, c = b.get(k), h.get(k)
                if not a or not c or len(a) != len(c):
                    return None
                fa = [x for r in a for x in (r if isinstance(r, list) else [r])]
                fc = [x for r in c for x in (r if isinstance(r, list) else [r])]
                return max(abs(x - y) for x, y in zip(fa, fc)) if len(fa) == len(fc) else None
            for k in ("real_lse_right", "real_logits8_right", "real_lse_left", "real_logits8_left"):
                d = _mx(k)
                h[f"_d_{k}"] = d
                if d is None or d > 5e-2:
                    worse.append(f"{cell}: {k} base vs head max abs diff {d}")
            for k in ("loss_right", "gnorm_right"):
                if isinstance(b.get(k), (int, float)) and isinstance(h.get(k), (int, float)):
                    d = abs(b[k] - h[k]) / max(1e-6, abs(b[k]))
                    h[f"_d_{k}"] = d
                    if d > 2e-3:
                        worse.append(f"{cell}: {k} differs base {b[k]} head {h[k]} (rel {d:.3g})")
            if isinstance(h.get("leak"), (int, float)) and h["leak"] > 1e-3:
                worse.append(f"{cell}: pad / first-real logits still move with later tokens at head (leak {h['leak']})")
        if cell.startswith("fpskip") or cell == "dsv4g":
            lb0, lh0 = b.get("step0_loss"), h.get("step0_loss")
            if isinstance(lb0, float) and isinstance(lh0, float) and abs(lb0 - lh0) > 1e-4 * max(1.0, abs(lb0)):
                worse.append(f"{cell}: step-0 loss {lh0} at head vs {lb0} at base")
            gb, gh = b.get("step0_grads") or {}, h.get("step0_grads") or {}
            if set(gb) != set(gh):
                worse.append(f"{cell}: LoRA tensor set differs ({len(gb)} base vs {len(gh)} head)")
            mx = 0.0
            for k in gb:
                vb, vh = gb.get(k), gh.get(k)
                if (vb is None) != (vh is None):
                    worse.append(f"{cell}: {k} grad None at one state only")
                    continue
                if vb and vh:
                    mx = max(mx, abs(vb[0] - vh[0]) / max(1e-6, abs(vb[0])))
            h["_grad_norm_max_rel"] = mx
            if mx > 1e-2:
                worse.append(f"{cell}: LoRA grad norm differs base vs head (max rel {mx:.3g})")
        if cell.startswith("rpack"):
            for k, floor in (("max_logit_maxabs", 5e-2), ("grad_max_rel", 2e-2), ("leak_others_maxabs", 1e-6)):
                vb, vh = b.get(k), h.get(k)
                if isinstance(vh, (int, float)) and vh > max(floor, 2 * (vb if isinstance(vb, (int, float)) else 0)):
                    worse.append(f"{cell}: packed vs per-document {k} {vh} at head vs {vb} at base")
        if cell == "moesft_qwen35moe" and h.get("config_restored") is False:
            worse.append(f"moesft_qwen35moe: text_config coef not restored after training ({h.get('coef_after_train')})")
        if cell == "gkd" and (h.get("teacher_trainable") or 0) > (b.get("teacher_trainable") or 0):
            worse.append(f"gkd: teacher trainable params {b.get('teacher_trainable')} at base, {h.get('teacher_trainable')} at head")
        if cell == "gkd" and b.get("teacher_device") and h.get("teacher_device") != b.get("teacher_device"):
            worse.append(f"gkd: teacher on {h.get('teacher_device')} at head, {b.get('teacher_device')} at base")
        if cell.startswith("dec_"):
            for tag in ("default", "forced"):
                rb, rh = (b.get("decode_runs") or {}).get(tag), (h.get("decode_runs") or {}).get(tag)
                if rb and rh and rb.get("tokens") != rh.get("tokens"):
                    worse.append(f"{cell}: {tag} generate tokens differ base {rb.get('tokens')} vs head {rh.get('tokens')}"[:300])
        if cell == "bswap" and b.get("bswap_equal") and not h.get("bswap_equal"):
            worse.append(f"bswap: offload 2 equal to offload 0 at base, not at head (grad maxdiff "
                         f"{h.get('bswap_grad_maxdiff')}, losses {h.get('bswap_loss_ref_vs_swap')})")
        if cell == "bswap" and not any((b.get("bswap_grew") or {}).values()) and any((h.get("bswap_grew") or {}).values()):
            worse.append("bswap: slot pool grows at head, not at base")
        if cell.startswith("padmix"):
            gb_, gh_ = b.get("unique_graphs_per_step") or [], h.get("unique_graphs_per_step") or []
            if gb_ and gh_ and gh_[-1] > gb_[-1]:
                worse.append(f"{cell}: more compiled graphs at head {gh_} than base {gb_}")
        if cell in ("grpo_pg_long", "grpo_pg_long_kl") and not (h.get("pg_layouts") or 0) > 0:
            worse.append(f"grpo_pg_long: PrefixGrouper built no layout at head ({h.get('pg_layouts')}/{h.get('pg_build_calls')})")
        if cell == "grpo_pg_short" and (h.get("pg_layouts") or 0) > 0:
            worse.append(f"grpo_pg_short: saved-token gate did not decline at head ({h.get('pg_layouts')} layouts)")
        if cell == "grpo_txt" and b.get("seq_packing_grad_ok") == "True" and h.get("seq_packing_grad_ok") != "True":
            worse.append(f"grpo_txt: sequence packing on at base, {h.get('seq_packing_grad_ok')} at head")
        if cell == "sftdef_steps" and h.get("resolved_packing") and not b.get("resolved_packing"):
            worse.append("sftdef_steps: a max_steps run turned packing on at head")
        if cell == "fp32attn" and isinstance(h.get("max_abs_logprob_diff_vs_hf_fp32_eager"), (int, float)) \
                and h["max_abs_logprob_diff_vs_hf_fp32_eager"] > max(1e-3, 2 * (b.get("max_abs_logprob_diff_vs_hf_fp32_eager") or 0)):
            worse.append(f"fp32attn: head logprobs drift {h['max_abs_logprob_diff_vs_hf_fp32_eager']} from HF fp32 eager")
        if cell == "tphost" and h.get("engaged_on") and isinstance(h.get("max_loss_diff_on_vs_off"), (int, float)) \
                and (h["max_loss_diff_on_vs_off"] > 1e-5 or (h.get("max_grad_diff_on_vs_off") or 0) > 1e-5):
            worse.append(f"tphost: host embedding changes loss / grads at head ({h['max_loss_diff_on_vs_off']}, {h.get('max_grad_diff_on_vs_off')})")
        if cell == "fsdp2_gen" and isinstance(h.get("gen_modes"), dict):
            for mode, r in h["gen_modes"].items():
                if r.get("tokens_equal") is False or r.get("error"):
                    b_r = (b.get("gen_modes") or {}).get(mode) or {}
                    if b_r.get("tokens_equal") is True:
                        worse.append(f"fsdp2_gen: generate {mode} at head {r}")
        if cell == "fsdp2_io" and isinstance(h.get("max_loss_diff_vs_unsharded"), (int, float)) \
                and h["max_loss_diff_vs_unsharded"] > max(1e-4, 2 * (b.get("max_loss_diff_vs_unsharded") or 0)):
            worse.append(f"fsdp2_io: FSDP2 loss differs from the unsharded model by {h['max_loss_diff_vs_unsharded']} at head")
        if cell == "mrope_pack" and isinstance(b.get("leak"), (int, float)) and isinstance(h.get("leak"), (int, float)) \
                and h["leak"] > max(1e-3, b["leak"]):
            worse.append(f"mrope_pack: cross-document leak {h['leak']} at head vs {b['leak']} at base")
        if cell == "cp_ring_kernels":
            for dt, st in (b.get("ring_pinned_runs") or {}).items():
                if st == "ok" and (h.get("ring_pinned_runs") or {}).get(dt) != "ok":
                    worse.append(f"cp_ring_kernels: {dt} ran at base under the ring kernel list, refused at head")
        if cell.startswith("aol_") and b.get("assistant_only") and not h.get("assistant_only"):
            worse.append(f"{cell}: assistant-only labels at base, not at head")
        if cell == "plelens":
            gb_, gh_ = b.get("unique_graphs_per_step") or [], h.get("unique_graphs_per_step") or []
            if gb_ and gh_ and gh_[-1] > gb_[-1]:
                worse.append(f"plelens: more compiled graphs at head {gh_} than base {gb_}")
        if cell == "granite_gc" and (b.get("granite_grad_rel") or 0) < 2e-2 and (h.get("granite_grad_rel") or 0) >= 2e-2:
            worse.append(f"granite_gc: GC on vs off LoRA grad norm differs at head ({h.get('granite_grad_rel')})")
        if cell == "g2nr_flex" and isinstance(h.get("g2_grad_rel"), (int, float)) and h["g2_grad_rel"] >= 1e-3 \
                and not (isinstance(b.get("g2_grad_rel"), (int, float)) and b["g2_grad_rel"] >= 1e-3):
            worse.append(f"g2nr_flex: non-reentrant GC LoRA grads differ from no-GC at head (rel {h.get('g2_grad_rel')})")
        if cell == "bclens" and (h.get("recompile_logs") or 0) > (b.get("recompile_logs") or 0):
            worse.append(f"bclens: more recompiles at head ({h.get('recompile_logs')}) than base ({b.get('recompile_logs')})")
        if cell == "studiotok":
            for k in ("rows_sha_eager_routed", "rows_sha_in_process", "rows_sha_online_view", "num_proc_unvetoed",
                      "num_proc_routed", "add_special_tokens", "n_tokens"):
                if b.get(k) != h.get(k):
                    worse.append(f"studiotok: {k} base {b.get(k)!r} vs head {h.get(k)!r}"[:300])
            if not h.get("rows_identical_all_paths"):
                worse.append("studiotok: routed / in-process / online rows differ at head")
            ob, oh = b.get("online_decision"), h.get("online_decision")
            if isinstance(ob, dict) and isinstance(oh, dict) and ob.get("enabled") != oh.get("enabled"):
                worse.append(f"studiotok: online tokenization decision base {ob} vs head {oh}"[:300])
        if cell == "qvlmoe_aux" and b.get("fwd_aux_finite") and not h.get("fwd_aux_finite"):
            worse.append(f"qvlmoe_aux: finite aux_loss at base, not at head ({h.get('fwd_error')})")
        if cell == "tok_save" and b.get("reload_ok") and not h.get("reload_ok"):
            worse.append(f"tok_save: save / reload round trip broke at head ({h.get('reload_loss_rel')}, "
                         f"tokenizer {h.get('tokenizer_roundtrip')})")
        if cell == "lora_noac" and b.get("all_lora_grads_nonzero") and not h.get("all_lora_grads_nonzero"):
            worse.append(f"lora_noac: a LoRA tensor got no gradient at head ({h.get('lora_grads_nonzero')}/{h.get('lora_tensors')})")
        if cell.startswith("gc") and len(h.get("block_compile_failed_msgs") or []) > len(b.get("block_compile_failed_msgs") or []):
            worse.append(f"{cell}: more block compile failures at head: {h.get('block_compile_failed_msgs')}")
        if cell == "gcmem" and b.get("gc_engaged") and not h.get("gc_engaged"):
            worse.append(f"gcmem: layer GC saved memory at base (ratio {b.get('gc_mem_ratio')}), not at head "
                         f"(ratio {h.get('gc_mem_ratio')})")
        if cell == "rl_decode" and b.get("graphs_match_eager_after_step") and not h.get("graphs_match_eager_after_step"):
            worse.append(f"rl_decode: decode graphs diverge from eager after the step at head (rel {h.get('graphs_vs_eager_rel')})")
    hx_ = hc.get("fx32")
    if hx_ is not None and (not _trained(hx_) or hx_.get("fx_attn") == "flex_attention"):
        worse.append(f"fx32: head float32 load did not finish off flex (attn {hx_.get('fx_attn')}, stage {hx_.get('fx_stage')}, rc {hx_.get('fx_rc')})")
    for _cgc in ("cg_forced", "cg_forced8"):
      hf_ = hc.get(_cgc)
      if hf_ is not None and not _trained(hf_):
        worse.append(f"{_cgc}: head did not train with UNSLOTH_BLOCK_CUDAGRAPHS=1 ({hf_.get('cg_error') or hf_.get('error') or hf_.get('stage')})"[:300])
    pb_, fb, cb = _pyt(base)
    ph_, fh, ch = _pyt(head)
    new_fail = sorted(fh - fb)
    if new_fail:
        worse.append("tests failing at head only: " + ", ".join(f"`{t.split('::')[-1]}`" for t in new_fail[:12]))
    new_crash = sorted(ch - cb)
    if new_crash:
        worse.append("test files crashing at head only: " + ", ".join(new_crash))
    detail = (f"newly training at head: {', '.join(fixed) or 'none'}; failing at both (pre-existing): "
              f"{'; '.join(both) or 'none'}; tests failing at both: "
              f"{', '.join(t.split('::')[-1] for t in sorted(fh & fb)[:10]) or 'none'}; "
              f"tests passing at head only: {len(ph_ - pb_ - fb)}")
    if worse:
        return True, "; ".join(worse) + ". " + detail
    return False, "no training cell or test that passed at the base fails at the head; " + detail


def _fmt(c):
    if not c:
        return "-"
    if c.get("cell") == "env" or "env" == c.get("_name"):
        return "ok" if c.get("stage") == "done" else f"FAIL {str(c.get('error'))[:80]}"
    if not _trained(c):
        return f"FAIL@{c.get('stage')} {str(c.get('error') or c.get('rc'))[:110]}"
    losses = c.get("losses") or []
    s = f"{losses[0]:.4f}->{losses[-1]:.4f}" if losses else "?"
    s += "" if c.get("decreasing") is None else (" dec" if c.get("decreasing") else " NOT-dec")
    s += f" pk={c.get('peak_alloc_mb')}M gc={c.get('gc_calls')}"
    dy = c.get("dynamo") or {}
    if dy.get("graph_breaks") is not None:
        s += f" gb={dy.get('graph_breaks')} g={dy.get('unique_graphs')}"
    prof = ((c.get("profile_step1") or c.get("profile_step") or {}).get("counts") or {})
    if prof.get("grouped_mm"):
        s += f" grouped_mm={prof['grouped_mm']}"
    if c.get("moe_backend"):
        s += f" backend={c['moe_backend']}"
    if c.get("lora_on_experts"):
        s += f" lora_experts={c['lora_on_experts']}"
    if "large_head_patch_returns" in c:
        s += f" lh_patch={c['large_head_patch_returns']}"
    if c.get("varlen_hits"):
        s += f" varlen={c['varlen_hits']}"
    for k in ("z12", "cg5_rc", "cg1", "cg1_tree_vs_unroll", "tree_counts_total", "compile_check_failed", "torch_supports_prefix_grouper", "smart_gc_patched", "gc_func_inner", "cg2_diag", "mlp_down_skipped", "skip_env", "zoo_has_replay_module", "router_dtypes", "router_dtypes_after_train", "fp32_kept_bitwise", "dtype_map_sha", "dora_mm_out_dtype", "dora_cross_term", "_lr1_rel_losses", "_lr1_rel_grad_norms", "_step2_rel_gold_losses", "_step2_rel_gold_losses_base", "_step2_rel_gold_grad_norms", "_step2_rel_gold_grad_norms_base", "_dev_gold_grad_norms", "_dev_gold_grad_norms_base", "_dev_gold_losses", "_dev_gold_losses_base", "dora_gold_fp64", "dora_forward_calls", "dora_factored", "dora_fallback", "dora_forward_patched", "dora_env", "dora_modules", "norm_modules", "norm_rel_err_peft_bf16", "norm_rel_err_factored_bf16", "norm_rel_err_factored_fp32", "lora_B_norm_min", "norm_error", "fx_attn", "fx_stage", "fx_rc", "fx_timeout", "fx_wall_s", "fx_load_s", "fx_train_s", "fx_loss", "cg_repo", "cg_steps", "w20a_8l_sha256", "cg_rc", "cg_ok", "cg_error", "cg_hooks", "cg_auto", "cg_graphs_off", "cg_graph_ids", "ls_engaged_unsloth", "ls_engaged_zoo", "ls_fallback_unsloth", "micro_batches", "fls_stats_train", "label_smoother", "oracle_fp32_loss", "oracle_fp32_loss_bf16logits", "fixed_batch_loss", "fixed_batch_rel_diff_vs_fp32", "fixed_batch_engaged_unsloth", "fixed_batch_engaged_zoo", "oracle_error", "pk_engagement", "eval_losses", "eval_accumulation_steps", "send_to_device_wrapped", "evaluation_loop_patched", "pk_has_carry_fn", "saved_for_backward", "step0_logits", "step0_loss", "_grad_norm_max_rel", "step0_grads_none", "lora_layers", "layer_select", "layer_calls_step0", "frozen_prefix_stats_delta", "route", "eager_wrapper_calls", "max_logit_maxabs", "grad_max_rel", "leak_others_maxabs", "ce_packed", "ce_docs", "loss_right", "loss_left", "gnorm_right", "leak_last_pad", "_d_real_lse_right", "_d_real_logits8_right", "_d_real_lse_left", "_d_real_logits8_left", "_d_loss_right", "_d_gnorm_right", "loss_right_after_left_step", "ckpt_coef", "trainer_router_aux_loss_coef", "oracle_nll_plus_ckpt_aux", "rel_vs_nll_plus_ckpt_aux", "moesft_ckpt_coef", "coef_after_train", "config_restored", "prefix_share_engaged", "prefix_share_declined", "prefix_share_verify", "recompute_summary", "recomputed", "gcon_mlp_total_per_step", "gcon_gc_calls_per_step", "control_ok", "gcoff_mlp_total_per_step", "gcoff_peak_alloc_mb", "lost_checkpointing", "layer_forward_names", "seq_packing_flags", "saved_attention_stats", "gc_save_attention_env", "autotune_calls", "warm_loss_eq", "run_attention_calls", "sdpa_q_dtypes", "max_abs_logprob_diff_vs_hf_fp32_eager", "has_flash", "has_xformers", "has_keeps_fp32", "param_dtype", "pg_build_calls", "pg_layouts", "pg_flags", "pg_enabled", "ga", "prompt_tokens", "pg_min_saved", "grad_norms", "beta", "atree_verdict", "atree_lines", "atree_rc", "engaged_on", "arms", "max_loss_diff_on_vs_off", "max_grad_diff_on_vs_off", "has_host_input_embedding", "decision_line", "torchrun_rc", "ranks", "rank_errors", "gen_modes", "has_generate_streams_blocks", "init_method", "backend", "has_set_unshard_async_op", "embed_on_cpu", "embed_device", "head_device", "embed_dtensor", "head_dtensor", "unshard_async_calls", "max_loss_diff_vs_unsharded", "shard_msg", "packed_rows_verdict", "text_prepended_hook", "leak", "leak_position_ids_shapes", "train_position_ids_shapes", "row_tokens", "indexer_vectorised", "stats_delta", "eos_counts", "assistant_only", "kept_tokens", "kept_image_tokens", "kept_labels_text", "collator_class", "ring_backends", "ring_pinned_runs", "processing_class_type", "trainer_processing_class", "hs_fallback_wrapped", "reward_margins", "ref_model_is_none", "resolved_packing", "resolved_padding_free", "global_steps", "pack_stats_after", "uncompiled_msgs", "attn_implementation", "flex_mask_keeps_padding_marker", "flex_mask_calls_per_step", "pattern", "guard_failures_per_step", "seq_packing_grad_ok", "rewards", "granite_gc_calls_on", "granite_kept_mib", "granite_loss_eq", "granite_grad_rel", "layer_forward_name", "seq_lens", "aux_gc_ok", "aux_gc_rel", "aux_loss_step0", "aux_loss_enabled", "router_logits_requested", "step1_loss", "stock_step1_loss", "oracle_nll", "oracle_nll_plus_aux", "rel_vs_stock", "rel_vs_nll_plus_aux", "rel_vs_nll_only", "stock_same_tokens", "moesft_matches_stock", "moesft_keeps_aux", "moesft_drops_aux", "teacher_device", "teacher_trainable", "trainer_class", "tokens", "decode_forced_eq_default", "graph_backend_available", "bswap_equal", "bswap_grad_maxdiff", "n_layers", "layer_forward_names", "block_modes_",  "bswap_grads_compared", "bswap_grew", "slot_pool_message", "ckpt_in_forward", "unsloth_offload_refusal", "swapped_indices", "recompile_logs", "unique_graphs_per_step", "reload_ok", "reload_loss_rel", "ddp_find_unused_default", "loader", "block_modes", "block_compile_failed_msgs", "lora_out_dtypes", "lora_grads_nonzero", "lora_tensors",
              "g2_grad_rel", "g2_ref", "g2_gc", "fix_installed", "fw_bw_graph_calls", "fw_bw_graph_calls_under_hooks", "nonre_flag",
              "packing_installed", "lean_lora_layers", "prefetch_loader", "engaged", "grpo_fused_engaged", "dpo_chunked_engaged", "grouped_available", "fwd_rel", "bwd_rel",
              "triton_version", "dpo_chunked_calls", "trl_has_concatenated_forward", "expert_param_types",
              "packed_expert_stats_delta", "fused_calls_delta"):
        if k in c:
            s += f" {k}={c[k]}"
    if "gc_mem_ratio" in c:
        s += f" gc_engaged={c.get('gc_engaged')} mem_ratio(on/off)={c.get('gc_mem_ratio')}"
    if "graphs_match_eager_after_step" in c:
        s += f" graphs==eager:{c['graphs_match_eager_after_step']} rel={c.get('graphs_vs_eager_rel'):.2g}"
    if c.get("vision_layer_calls"):
        s += " vlayer=" + ",".join(f"{k}:{v.get('ok')}" for k, v in c["vision_layer_calls"].items())
    return s


def table(obs):
    b, h = obs.get("base") or {}, obs.get("head") or {}
    unit = h.get("unit") or b.get("unit") or "?"
    rows = [f"Unit `{unit}` on `{h.get('platform') or b.get('platform')}`. Companion: base "
            f"`{str((b.get('companion') or {}).get('sha'))[:10]}`, head `{str((h.get('companion') or {}).get('sha'))[:10]}`.",
            "", "| cell | base | head |", "|---|---|---|"]
    names = [n for n in list(_cells(b)) + [n for n in _cells(h) if n not in _cells(b)]]
    for n in names:
        rows.append(f"| {n} | {_fmt(_cells(b).get(n))} | {_fmt(_cells(h).get(n))} |")
    env = (_cells(h).get("env") or {})
    if env:
        rows += ["", f"env (head): triton_kernel={env.get('triton_kernel')} torch_compile={env.get('torch_compile')} "
                 f"bnb_nf4_finite={env.get('bnb_nf4_finite')} inductor_use_block_ptr={env.get('inductor_use_block_ptr')} "
                 f"triton_has_block_ptr={env.get('triton_has_block_ptr')} moe_backend_default={env.get('moe_backend_default')}"]
    benv = (_cells(b).get("env") or {})
    if benv and "inductor_use_block_ptr" in benv:
        rows.append(f"env (base): inductor_use_block_ptr={benv.get('inductor_use_block_ptr')}")
    rows.append("")
    rows.append("Versions (head): " + str((h.get("env") or {}).get("env")))
    spreads = {c: _aa_spread(b, c) for c in ("flm16", "moe")}
    rows.append(f"A/A spread at base (max rel per-step loss diff): {spreads}; loss tolerance floor {RTOL}")
    for key in ("pytest", "pytest_companion"):
        for name, o in (("base", b), ("head", h)):
            p = o.get(key)
            if p is not None:
                rows.append(f"{key} {name}: passed={p.get('n_passed')} failed={p.get('n_failed')} "
                            f"skipped={p.get('n_skipped')} absent={p.get('absent_at_this_state')} "
                            f"crashed={p.get('crashed_files')}")
    if unit in NOTES:
        rows += ["", f"ROCm applicability: {NOTES[unit]}"]
    return "\n".join(rows)
