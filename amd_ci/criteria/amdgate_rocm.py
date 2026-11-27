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
TRAIN = ("flm16", "flm16_a8", "flm16_a8on", "ls_qwen3", "ls_qwen3_off", "mgsa_llama", "mgsa_qwen3", "mgsa_llama_sa0", "pk_qwen3", "flm4", "gcmem", "vision", "moe", "gemma4", "gemma4_gcrec", "qwen38", "dsv4", "mllama", "rl_decode", "ctnvfp4",
         "ctmxfp8", "mxfp4probe", "grpo_vl", "dpo_vl", "gcnr_qwen3", "gcdef_qwen3", "gcnr_gemma3", "gcdef_gemma3",
         "lora_noac", "st_minilm", "st_qwen3", "hd512_llama", "hd512_gemma4", "hd512_gemma4_nc", "bclens", "bswap", "dec_whisper", "dec_gemma3n", "dec_mistral", "moeaux_nr", "moeaux_gc_qwen3moe", "moeaux_gc_gptoss", "moesft_qwen3moe", "moesft_gptoss", "gkd", "granite_gc", "plelens", "padmix_qwen3", "padmix_gemma4", "padmix_gemma3", "padmix_gemma3_flex", "padmix_gemma3_flex_ones", "padmix_gemma4_ones", "grpo_txt", "grpo_padeos", "grpo_zadv", "grpo_zadv_zoff", "pfree_llama", "pfree_qwen3", "sftdef_steps", "sftdef_epoch", "dpo_txt", "orpo_txt", "orpo_txt_psoff", "orpo_txt_unwrap", "qwen38_long", "sft_eos", "aol_qwen25vl", "aol_gemma3_unsloth", "aol_gemma3_trl", "aol_qwen25vl_none", "aol_gemma3_none", "cp_ring_kernels", "gkd_proc_qwen35", "gkd_proc_gemma3", "vision_gemma3", "mrope_pack", "seqcls", "tok_save", "fsdp2_io", "fsdp2_gen", "pauto", "tphost", "atree", "grpo_pg", "fp32attn", "warm", "flm16_gcsave", "lpad", "lpadx", "moesft_qwen35moe", "dpo_txt_psforce", "fft16", "fft16_sroff", "grpo_pack", "g2nr_flex", "studiotok", "qvlmoe_aux", "grpo_vl_dd0", "qat_int4", "qat_int8", "qat_int4_ri0", "qat_int8_ri0", "grpo_txt_ts1", "grpo_txt_ts1_kr0", "keptset_scorer", "keptset_scorer_kr0", "vlmpack_parity", "vlmpack_parity_bf16", "vlmpack_parity_k0", "vlmpack_sft", "vlmpack_sft_k0", "vlmpack_pad", "grpo_pg_short", "grpo_pg_long", "grpo_pg_long_kl", "lpadx_gemma2", "lpadx_gemma2_nbc", "fpskip_uns", "fpskip_true", "dsv4g", "rpack_bf16", "rpack_fp32", "f32bf_llama", "f32na_llama", "rlaux_mixtral", "lmhlora_llama", "rlgen", "rlgen_fold", "rlgen_fp16", "gen_llama", "gen_llama_f32", "gen_mistral", "gen_gemma3", "plain_llama", "plain_qwen2", "plain_after", "grpo_padeos_ref", "grpo_padeos_refctl", "gen_llama_mlpbias", "sa8k_q06", "sa8k_q06_r", "sa2k_q06", "cmp_qwen3", "vis_frozen", "vis_frozen_ks", "vis_unfrozen", "aot_q06", "cpp_qwen3", "oom_ce", "qgal", "cthreads", "gcfit", "gcfit_gk0", "flm16_gk0", "loraround", "cputhr", "moe_auto", "rocmgate", "moeimp")
# Cells whose numbers the PR changes by design (the CT loader fixes the expert values; RL sampling / the fused GRPO
# route; the probe's numbers are errors, not losses): no per-step loss / memory equality, the rest still applies.
NO_LOSS_EQ = ("grpo_padeos", "grpo_padeos_ref", "rpack_bf16", "rpack_fp32", "vlmpack_parity", "vlmpack_parity_bf16", "vlmpack_parity_k0", "vlmpack_sft", "vlmpack_sft_k0", "keptset_scorer", "keptset_scorer_kr0", "grpo_vl_dd0", "moesft_qwen35moe", "moesft_qwen3moe", "moesft_gptoss", "ctnvfp4", "ctmxfp8", "mxfp4probe", "grpo_vl", "sftdef_epoch", "aol_qwen25vl", "aol_gemma3_unsloth", "aol_gemma3_trl", "aol_qwen25vl_none", "aol_gemma3_none", "sft_eos", "g2nr_flex", "cputhr", "rocmgate")
ENGAGED = {"flm16_a8on": "flat_engaged", "dpo_txt_psforce": "prefix_share_engaged", "mxfp4probe": "engaged", "grpo_vl": "grpo_fused_engaged", "grpo_vl_dd0": "grpo_fused_engaged", "dpo_vl": "dpo_chunked_engaged"}

# Why a PR's own new path does or does not engage on ROCm; rendered with the table.
NOTES = {
    'g19': "Gate 19 (BEFORE = gate 18 AFTER: unsloth 41e9a14091 + zoo 3dd11fcfc2; AFTER = unsloth a8fb477bf8 + zoo 31fab5c7bd): post-merge sweep, gate 16 to 18 cells kept. Zoo #2849 makes the grouped_mm training probe accept gfx1151 (moe / moe_auto: unsloth_triton at BEFORE, grouped_mm at AFTER, losses within tolerance), and zoo #2864 runs the eager loop for CPU tensors under unsloth_triton (the 31 gate 18 routing replay IDs).",
    'g18': "Gate 18 (BEFORE = gate 17 AFTER: unsloth c479c38bb6 + zoo 07901303dc; AFTER = unsloth 41e9a14091 + zoo 3dd11fcfc2): gate 16 / 17 cells kept. unsloth #19766 plans checkpoint_skip_layers='auto' by default on one GPU: where the head planner keeps layers un-checkpointed (gcskip record), fewer layer checkpoint calls and a higher peak are by design and only the losses are judged; _gk0 cells set UNSLOTH_GC_KEEP_LAYERS=0 (opt-out) and must keep checkpointing. loraround (#19764) measures the LoRA share kept by matmul_lora in the default order (bitwise base vs head) and inside vllm_lora_rounding() (head: within 0.05 of the vLLM-style rollout, order restored after). cputhr (zoo #2839) fakes 32 compute cores at 32 torch threads: head must cap at 16; the real fit must agree when the host has <= 16 cores. moe forces grouped_mm, moe_auto leaves the backend to the zoo (zoo #2840 training probe); losses within tolerance, backends reported. rocmgate (zoo #2841): on a ROCm torch the head must keep the GRPO / score-centering head in its own dtype (no fp32 copy).",
    'g17': "Gate 17 (BEFORE = gate 16 AFTER: unsloth 260c206c57 + zoo dba93c5446; AFTER = unsloth c479c38bb6 + zoo 07901303dc): gate 16 cells unchanged; cpp_qwen3 trains tiny Qwen3 LoRA with block compile, UNSLOTH_BLOCK_CPP_WRAPPER unset and UNSLOTH_BLOCK_CPP_WRAPPER_AFTER=0, so only unsloth #19623's gate decides: 0 C++ wrapper regions on ROCm, losses bitwise. oom_ce injects one OOM into zoo's compiled fused-CE chunk: at BEFORE the probe latches compile off (and resets Dynamo), at AFTER (zoo #2808) the next call runs compiled. qgal runs Q-GaLore 8-bit AdamW (unsloth #19700) on a 2-block MLP in 4 configs: final weights bitwise. cthreads reads compile_threads in a child pinned to 6 CPUs (zoo #2812: host count at BEFORE, 6 at AFTER) and compiles one fwd + bwd there.",
    'g16': "Gate 16 (BEFORE = gate 15 AFTER: unsloth 0d76f9495a + zoo c7761938a6; AFTER = unsloth 260c206c57 + zoo dba93c5446): vis_frozen trains tiny Qwen2.5-VL LoRA on the language layers only; unsloth #19568 drops the frozen vision tower's backward (tower_bwd > 0 at BEFORE, 0 at AFTER), vis_frozen_ks sets the kill switch UNSLOTH_FROZEN_TOWER_INPUT_GRAD=1 (backward back), vis_unfrozen keeps the tower trainable; losses and LoRA grad norms must be bitwise. aot_q06 runs Qwen3-0.6B LoRA compile in a cold then a warm process on one fresh Inductor cache dir; zoo #2802 lets the graph holding the fused LoRA Function use the AOTAutograd cache (bypass at BEFORE, hit at warm AFTER); losses bitwise cold vs warm and base vs head. grpo_txt (HF generate, no vLLM) must be unchanged by #19560 / zoo #2794.",
    'g15': "Gate 15 (BEFORE = gate 14 AFTER: unsloth 31c9556c83 + zoo 641c14e7b3; AFTER = unsloth 0d76f9495a + zoo c7761938a6): sa8k_q06 / sa8k_q06_r train Qwen3-0.6B LoRA at 8192 tokens with UNSLOTH_GC_SAVE_ATTENTION unset; zoo #2778 + unsloth #19508 turn automatic saved attention on for gfx1151 from 8192, so engagement 0/N at BEFORE and N/N at AFTER is expected, losses must be bitwise, and a peak-memory rise of up to 1.5 GB is by design when AFTER engaged. sa2k_q06 (2048 tokens, below the threshold) must be unchanged. cmp_qwen3: single-GPU compile on a non-EP model (unsloth #19556 / #19548): is_big_gpu(0) and losses unchanged. New / edited tests of both repos are copied from AFTER into BEFORE.",
    'g14': "Gate 14 (BEFORE = gate 13 AFTER, AFTER = unsloth 31c9556c83 + zoo 641c14e7b3): grpo_padeos_ref trains tiny Qwen3 GRPO with pad = eos = <|im_end|> (chat prompts whose turn ends share the pad id) at beta 0.04, num_iterations 2, so old / ref logps exist; unsloth #19516 + zoo #2782 lay rows out by the real-token mask there (losses may differ by design, finite + trained judged; logps_prompt_from_mask expected only at AFTER). grpo_padeos_refctl is the same run with the distinct pad <|endoftext|> (unchanged path, loss equality judged). gen_llama_mlpbias: Llama with random nonzero MLP biases through Unsloth generate vs plain transformers (unsloth #19510 prefill bias; tokens expected to differ at BEFORE). New / edited tests of both repos are copied from AFTER into BEFORE.",
    'u19334': "GRPO pad == eos (unsloth #19334 + zoo #2748 as one unit): grpo_padeos trains tiny Qwen3 GRPO with pad_token set to eos_token (prompts left-padded with the eos id, 16-token completions); the PRs change which rows count as EOS-terminated, so losses may differ by design (finite + trained judged, no equality). grpo_txt / grpo_pack (pad != eos) must match base. The PR tests of both repos run at both arms (new files copied from head into base).",
    'u19350': "GRPO distributed guards (unsloth #19350 + zoo #2747 as one unit): the guards engage only under world > 1 / ZeRO-3 / vLLM sleep, which one GPU cannot reach; grpo_txt and flm16 are no-regression cells, the zoo gloo test runs the collective forwards on CPU ranks.",
    'u19318': "Default packing for epoch runs: sftdef_epoch (num_train_epochs=1, packing unset) is the new path (head expected packing resolved on, base off; losses differ by design); sftdef_steps (max_steps) must be unchanged.",
    'u19277': "Expert parallel: EP needs 2+ ranks, unreachable on one GPU; flm16 / flm4 / moe are single-GPU no-regression cells; the EP gloo tests (CPU ranks) run on Linux.",
    'u19175': "Exact LoRA fold in RL rollouts: rlgen / rlgen_fold / rlgen_fp16 run tiny Qwen3 GRPO (2 steps, 160-token rollouts) in bf16 default (fold refused 'bf16'), bf16 with UNSLOTH_LORA_FOLD_GENERATE=1, and fp16 (fold on by default). Decode CUDA graphs are off on ROCm, so the eager 128-token threshold applies at both arms; rollout ids, losses and greedy decode logits must be bitwise base vs head. The PR's test runs at both arms.",
    'u19176': "Skinny linear on L4 + ROCm fallbacks / config isolation: on gfx1151 the ROCm skinny decode kernel (#18874) must engage with the same call count in rollouts and greedy decode (rlgen / rlgen_fold), and decode logits must be bitwise base vs head. The PR's test runs at both arms.",
    'u18610': "DoRA factored norm + Triton cross kernel: the kernel is gated on torch.version.cuda (None on ROCm), so on gfx1151 dora_triton_loaded must be False and the cross term keeps the torch path (with the PR's bit-masked hi / lo split of a float32 adapter). dora16 / dora4 / dora16_lr1 train DoRA at both arms; dora16_lr1_g64 is the fp64-row-norm gold trajectory (identical code at both arms). The PR's test runs at both arms.",
    'z2714': "MoE dequant chunk decision replayed under torch's non-reentrant checkpoint: autograd saved-tensor-hook bookkeeping (device-neutral); the PR's test (fake headroom on cuda:0) runs at both arms; flm16 / moe are the no-regression cells.",
    'z2717': "Chunked CE for a live LoRA lm_head: lmhlora_llama trains LoRA on the 7 projections + lm_head; head_chunked_calls counts zoo's chunked head path (absent at base). The PR's test runs at both arms.",
    'u19079': "Fused LoRA for float32 bases under bf16 autocast: f32bf_llama loads the built tiny Llama in float32 and trains LoRA with SFT bf16 = True; fp32_fused_bf16 counts fast_lora._autocast_master_dtype decisions that returned bfloat16 (absent at base, so base runs PEFT's forward). f32na_llama is the control: float32, no autocast, must stay on PEFT (count 0) and train as the base. HIP tensors report is_cuda, so the autocast device type is cuda on ROCm. The PR's tests run at both arms (new / edited files copied from head into base).",
    'u19086': "Llama / Mistral forward guards (Gemma2 reentrant replay kwargs, logits_to_keep with labels, shift-only fused CE, biased-head decode): device-neutral Python in the forward; flm16 / flm4 (fused CE, GC), dec_mistral (decode) and lpadx_gemma2 (Gemma2 FastLanguageModel) are the no-regression cells; the PR's tests are copied from head into base.",
    'u19109': "FSDP2 native fp16 GradScaler skip with a fused optimizer: world-1 FSDP2 only on this box; the PR's gloo test (CPU process group) and mixed-device GPU test run at both arms; fsdp2_io trains native FSDP2 world 1 on the GPU.",
    'u19111': 'GRPO autotune units / split count / PrefixGrouper token caps (unsloth #19111 + zoo #2711 as one unit): the zoo autotune now reads torch.cuda.mem_get_info + memory_reserved + per-process fraction, which HIP implements; grpo_txt / grpo_pg_long (PrefixGrouper must still build layouts) / grpo_pack train at both arms; the PR tests of both repos run at both arms (copied from head into base).',
    'u19112': "Saved attention fullgraph routing on every build (unsloth #19112 + zoo #2712): the ROCm branch is meant to be unchanged; mgsa_* (16k-token LoRA, GC unsloth, saved attention auto) and flm16 must match base, and the PR's tests of both repos run at both arms.",
    'u19069': "Gemma 3n / 4 reentrant wrapper accepts gradient_checkpointing_kwargs positionally: gemma4 / gemma4_gcrec train tiny Gemma-4 (wrapper installed, keyword call path); the positional call (TRL unwrap_model_for_generation) is covered by the PR's test, copied from head into base.",
    'u19058': "MoE router aux loss under the RL config's reentrant pin: rlaux_mixtral is the GPU port of the PR's CPU test (random tiny Mixtral, Unsloth SFTTrainer, gradient_checkpointing True, aux coef 0.5, fp32, lr 0, one step); router-gate gradient vs a float64 no-GC oracle with and without the aux loss. Base expected closer to the nll-only oracle (aux gradient dropped), head to match the full oracle.",
    'u18973': "Tied embeddings stay tied after add_new_tokens: device-neutral get_peft_model ordering; the PR's test (imports unsloth, so it needs a GPU host) runs at both arms; flm16 is the no-regression cell.",
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
NOTES["g14t"] = NOTES["g14"]
NOTES["g18t"] = NOTES["g18"]
NOTES["g19t"] = NOTES["g19"]
NOTES["g20"] = ("Gate 20 (BEFORE = gate 19 AFTER: unsloth a8fb477bf8 + zoo 31fab5c7bd; AFTER = unsloth 3f4a7e5fc5 + zoo c2b1a436e9): "
                "zoo #2882 moves the grouped_mm probe from import to model load (moeimp: probe globals None right after import at AFTER) and "
                "zoo #2879 makes auto pick unsloth_triton on gfx1151 (moe_auto backend grouped_mm at BEFORE, unsloth_triton at AFTER, by design); "
                "explicit UNSLOTH_MOE_BACKEND=grouped_mm (moe) keeps grouped_mm. The SR DDP file runs 4 times (3 byte-identical copies).")
NOTES["g20t"] = NOTES["g20"]
NOTES["g20z"] = NOTES["g20"]
NOTES["sr19"] = "Gate 19 SR DDP triage: tests/test_bf16_stochastic_rounding_ddp_gloo.py (AFTER-19 version) plus 3 byte-identical copies at both arms."
NOTES["sr19m"] = NOTES["sr19"]


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
    hcell = _cells(obs.get("head")).get("flm16_a8on")
    if hcell is not None:  # unsloth #18977: the new fused 8-bit AdamW step must provably run at head
        out.append(("head flm16_a8on ran the fused 8-bit AdamW step", bool(hcell.get("flat_engaged")),
                    f"flat_engaged={hcell.get('flat_engaged')} optimizer={hcell.get('optimizer_class')} {hcell.get('flat_module') or ''}"[:300]))
    hcell = _cells(obs.get("head")).get("f32bf_llama")
    if hcell is not None:  # unsloth #19079: float32 bases under bf16 autocast must take the fused route at head
        out.append(("head f32bf_llama fused LoRA on float32 bases under bf16 autocast",
                    (hcell.get("fp32_fused_bf16") or 0) > 0,
                    f"fp32_fused_bf16={hcell.get('fp32_fused_bf16')} calls={hcell.get('fp32_fused_calls')} "
                    f"hook={hcell.get('fp32_fused_hook')} flagged={hcell.get('bf16_autocast_only_modules')}"[:300]))
    hcell = _cells(obs.get("head")).get("rlaux_mixtral")
    if hcell is not None:  # unsloth #19058: the cell measured the router gradient against the oracle
        out.append(("head rlaux_mixtral compared the router gradient with the fp64 oracle",
                    hcell.get("router_grad_rel_max") is not None and bool(hcell.get("aux_loss_enabled")),
                    f"aux_loss_enabled={hcell.get('aux_loss_enabled')} rel={hcell.get('router_grad_rel_max')} "
                    f"rel_nll_only={hcell.get('router_grad_rel_vs_nll_only_max')} error={str(hcell.get('error'))[:120]}"[:300]))
    hcell = _cells(obs.get("head")).get("lmhlora_llama")
    if hcell is not None:  # zoo #2717: the live LoRA head must be scored chunk by chunk at head
        out.append(("head lmhlora_llama scored the LoRA lm_head chunk by chunk",
                    isinstance(hcell.get("head_chunked_calls"), int) and hcell["head_chunked_calls"] > 0
                    and (hcell.get("lm_head_lora_tensors") or 0) > 0,
                    f"head_chunked_calls={hcell.get('head_chunked_calls')} lm_head_lora_tensors={hcell.get('lm_head_lora_tensors')} "
                    f"error={str(hcell.get('error'))[:120]}"[:300]))
    # Non-vacuity: the base actually trained something, so "no cell got worse" compared real runs. A cell that
    # fails at BOTH states is a finding reported in the table and the verdict text, not a gate.
    trained = sorted(n for n, c in _cells(obs.get("base")).items() if n in TRAIN and _trained(c))
    out.append(("base trained at least one cell (finite losses)", bool(trained), ", ".join(trained) or "none"))
    return out


def _gc_kept(c):
    g = (c or {}).get("gcskip") or {}
    return max([p.get("kept") or 0 for p in g.get("plans") or []] + [x for x in g.get("applies") or [] if isinstance(x, int)] + [0])


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
        if cell.startswith("sa") and lb and lb != lh:
            worse.append(f"{cell}: losses not bitwise base vs head ({lb} vs {lh})"[:300])
        sa_ok = (cell.startswith("sa8k") and isinstance(pb, (int, float)) and isinstance(ph, (int, float))
                 and (h.get("sa_engaged_steps") or 0) > (b.get("sa_engaged_steps") or 0) and ph - pb <= 1536)
        if (cell.startswith("vis_") or cell == "aot_q06") and lb and (lb != lh or b.get("grad_norms") != h.get("grad_norms")):
            worse.append(f"{cell}: losses / grad norms not bitwise base vs head ({lb} vs {lh})"[:300])
        if cell == "aot_q06" and b.get("cold_eq_warm") and not h.get("cold_eq_warm"):
            worse.append(f"aot_q06: cold and warm processes bitwise at base, not at head ({h.get('runs', {}).get('warm', {}).get('losses')})"[:300])
        if cell == "vis_frozen" and _trained(hc.get("vis_frozen_ks")) and (hc.get("vis_frozen_ks") or {}).get("losses") != h.get("losses"):
            worse.append("vis_frozen: head losses differ from the head kill-switch run (UNSLOTH_FROZEN_TOWER_INPUT_GRAD=1)")
        if cell in ("cpp_qwen3", "qgal", "cthreads") and lb and (lb != lh or b.get("grad_norms") != h.get("grad_norms")):
            worse.append(f"{cell}: losses / grad norms (qgal: final weight sha) not bitwise base vs head ({lb} vs {lh}; {b.get('grad_norms')} vs {h.get('grad_norms')})"[:300])
        if cell == "cpp_qwen3" and (not isinstance(h.get("cpp_stats"), dict) or h.get("cpp_regions") != 0):
            worse.append(f"cpp_qwen3: C++ wrapper regions at head {h.get('cpp_stats')} (must be 0 on ROCm; reason {h.get('cpp_reason')})"[:300])
        if cell == "oom_ce" and (h.get("latched_after_oom") is not False or not h.get("next_call_compiled") or (h.get("resets") or 0) > 0):
            worse.append(f"oom_ce: head latched / reset after one injected OOM (latched {h.get('latched_after_oom')}, next compiled {h.get('next_call_compiled')}, resets {h.get('resets')})")
        if cell == "qgal":
            shares = {k: v.get("saved_shares_storage") for k, v in (h.get("configs") or {}).items() if k.endswith("wq0")}
            if not shares or not all(x and all(x) for x in shares.values()):
                worse.append(f"qgal: head saved weight not held by reference (unsloth #19700) {shares}"[:300])
        if cell == "cthreads":
            pin = (h.get("runs") or {}).get("pinned6") or {}
            if not pin.get("_skipped") and pin.get("compile_threads") != max(4, pin.get("affinity") or 0):
                worse.append(f"cthreads: pinned child compile_threads {pin.get('compile_threads')} at head for affinity {pin.get('affinity')} (zoo #2812)")
        if cell == "cmp_qwen3" and b.get("is_big_gpu_0") != h.get("is_big_gpu_0"):
            worse.append(f"cmp_qwen3: is_big_gpu(0) {b.get('is_big_gpu_0')} at base, {h.get('is_big_gpu_0')} at head")
        gk_b, gk_h = _gc_kept(b), _gc_kept(h)
        gk_design = gk_h > gk_b  # unsloth #19766: the default auto planner keeps layers un-checkpointed at head
        if gk_design:
            h["_gcskip_by_design"] = f"head kept {gk_h} layer(s) un-checkpointed (base {gk_b}); gc calls / peak not judged"
        if cell.endswith("_gk0") and (gk_h > 0 or not (h.get("gc_calls") or 0) > 0):
            worse.append(f"{cell}: UNSLOTH_GC_KEEP_LAYERS=0 at head still kept {gk_h} layer(s) or ran no checkpointing ({h.get('gc_calls')} calls)")
        if cell == "loraround":
            if b.get("sha_default") and b.get("sha_default") != h.get("sha_default"):
                worse.append(f"loraround: default-order output not bitwise base vs head ({b.get('sha_default')} vs {h.get('sha_default')})")
            if not h.get("has_vllm_lora_rounding") or h.get("lora_fused_passes_after") != h.get("lora_fused_passes_initial") or not isinstance(h.get("vllm_order_gap_vs_rollout"), (int, float)) or h["vllm_order_gap_vs_rollout"] > 0.05:
                worse.append(f"loraround: head vllm_lora_rounding kept {h.get('kept_vllm_order')} vs rollout {h.get('kept_rollout')}, order after {h.get('lora_fused_passes_after')} (initial {h.get('lora_fused_passes_initial')})")
        if str((base or {}).get("unit")).startswith("z2882") and cell in ("moe", "moe_auto", "moeimp") and b.get("moe_backend") != h.get("moe_backend"):
            worse.append(f"{cell}: MoE backend {b.get('moe_backend')} at base, {h.get('moe_backend')} at head")
        if cell == "cputhr":
            if h.get("fake32_threads") != 16:
                worse.append(f"cputhr: head fit with 32 faked cores gives {h.get('fit_fake32')} (zoo #2839 cap 16)")
            if isinstance(h.get("cores"), int) and h["cores"] <= 16 and b.get("fit_real") != h.get("fit_real"):
                worse.append(f"cputhr: real fit differs base {b.get('fit_real')} vs head {h.get('fit_real')} on a {h.get('cores')}-core host")
        if cell == "rocmgate":
            if h.get("is_rocm_torch") is True and (h.get("grpo_fp32_lm_head_default") is not False or h.get("score_centering_mode") not in ("plain",) or h.get("truncated_support_auto") is not False):
                worse.append(f"rocmgate: ROCm head still takes a CUDA-only default (fp32 head {h.get('grpo_fp32_lm_head_default')}, score centering {h.get('score_centering_mode')}, truncated {h.get('truncated_support_auto')})")
            mb, mh = b.get("logprob_maxabs_vs_fp64"), h.get("logprob_maxabs_vs_fp64")
            if isinstance(mb, (int, float)) and isinstance(mh, (int, float)) and mh > max(5e-2, 2 * mb):
                worse.append(f"rocmgate: GRPO logprobs vs fp64 max abs {mh} at head vs {mb} at base")
        if not gk_design and not sa_ok and cell not in NO_LOSS_EQ and isinstance(pb, (int, float)) and isinstance(ph, (int, float)) and ph > pb * MEM_RATIO + MEM_SLACK:
            worse.append(f"{cell}: peak allocated {ph} MiB vs base {pb} MiB")
        if not gk_design and (b.get("gc_calls") or 0) > 0 and not (h.get("gc_calls") or 0) > 0:
            worse.append(f"{cell}: layer gradient checkpointing ran at base ({b.get('gc_calls')} calls), 0 at head")
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
        if cell.startswith("rlgen"):
            if b.get("rollout_shas") and b.get("rollout_shas") != h.get("rollout_shas"):
                worse.append(f"{cell}: rollout token ids differ base vs head ({b.get('rollout_shas')} vs {h.get('rollout_shas')})"[:300])
            if b.get("greedy_logits_sha") and b.get("greedy_logits_sha") != h.get("greedy_logits_sha"):
                worse.append(f"{cell}: greedy decode logits not bitwise base vs head")
            for k in ("skinny_calls_train", "skinny_calls_greedy"):
                if isinstance(b.get(k), int) and b[k] > 0 and h.get(k) != b[k]:
                    worse.append(f"{cell}: {k} {b[k]} at base, {h.get(k)} at head")
            fb_, fh_ = b.get("fold_stats") or {}, h.get("fold_stats") or {}
            if isinstance(fb_, dict) and isinstance(fh_, dict) and (fb_.get("engaged") or 0) > 0 and not (fh_.get("engaged") or 0) > 0:
                worse.append(f"{cell}: LoRA fold engaged at base, not at head ({fh_.get('refused')})")
        if cell == "f32na_llama" and (h.get("fp32_fused_bf16") or 0) > 0:
            worse.append(f"f32na_llama: float32 training without autocast took the bf16 fused route at head ({h.get('fp32_fused_bf16')})")
        if cell == "rlaux_mixtral" and b.get("rlaux_ok") and not h.get("rlaux_ok"):
            worse.append(f"rlaux_mixtral: router gradient matched the fp64 oracle at base, not at head ({h.get('router_grad_rel_max')})")
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
        if cell.startswith("gen_"):  # unsloth #19420 / zoo #2757: tokens may change by design (the base drops the cache on 5.24)
            if b.get("cache_used") and not h.get("cache_used"):
                worse.append(f"{cell}: generate used the KV cache at base, not at head (input lens {h.get('input_lens')})"[:300])
            if b.get("tokens_eq_hf") and h.get("tokens_eq_hf") is False:
                worse.append(f"{cell}: tokens matched plain transformers at base, not at head ({h.get('first_diff_vs_hf')})")
        if cell.startswith("plain_"):  # unsloth #19464 / zoo #2768: plain transformers models after `import unsloth`
            if cell == "plain_after" and b.get("unsloth_fused") and not h.get("unsloth_fused"):
                worse.append(f"plain_after: the Unsloth model's logits were empty (fused lm_head + CE) at base, not at head {h.get('unsloth_logits_numels')}")
            for m, pb_ in (b.get("plain") or {}).items():
                ph_ = (h.get("plain") or {}).get(m) or {}
                if pb_.get("labels_logits_present") and not ph_.get("labels_logits_present"):
                    worse.append(f"{cell}/{m}: plain model kept its logits with labels at base, not at head ({ph_.get('labels_logits_type')})")
                for k in ("max_logit_diff", "loss_diff"):
                    xb, xh = pb_.get(k), ph_.get(k)
                    if isinstance(xb, (int, float)) and (not isinstance(xh, (int, float)) or xh > xb + 1e-6):
                        worse.append(f"{cell}/{m}: {k} vs a fresh process {xb} at base, {xh} at head")
                if pb_.get("gen_eq_ref") and not ph_.get("gen_eq_ref"):
                    worse.append(f"{cell}/{m}: greedy tokens equalled a fresh process at base, not at head")
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
    for k in ("unsloth_fused", "unsloth_logits_numels", "plain", "cache_used", "input_lens", "cache_keys_seen", "tokens_eq_hf", "first_diff_vs_hf", "saved_for_backward", "step0_logits", "step0_loss", "_grad_norm_max_rel", "step0_grads_none", "lora_layers", "layer_select", "layer_calls_step0", "frozen_prefix_stats_delta", "route", "eager_wrapper_calls", "max_logit_maxabs", "grad_max_rel", "leak_others_maxabs", "ce_packed", "ce_docs", "loss_right", "loss_left", "gnorm_right", "leak_last_pad", "_d_real_lse_right", "_d_real_logits8_right", "_d_real_lse_left", "_d_real_logits8_left", "_d_loss_right", "_d_gnorm_right", "loss_right_after_left_step", "ckpt_coef", "trainer_router_aux_loss_coef", "oracle_nll_plus_ckpt_aux", "rel_vs_nll_plus_ckpt_aux", "moesft_ckpt_coef", "coef_after_train", "config_restored", "prefix_share_engaged", "prefix_share_declined", "prefix_share_verify", "recompute_summary", "recomputed", "gcon_mlp_total_per_step", "gcon_gc_calls_per_step", "control_ok", "gcoff_mlp_total_per_step", "gcoff_peak_alloc_mb", "lost_checkpointing", "layer_forward_names", "seq_packing_flags", "saved_attention_stats", "gc_save_attention_env", "autotune_calls", "warm_loss_eq", "run_attention_calls", "sdpa_q_dtypes", "max_abs_logprob_diff_vs_hf_fp32_eager", "has_flash", "has_xformers", "has_keeps_fp32", "param_dtype", "pg_build_calls", "pg_layouts", "pg_flags", "pg_enabled", "ga", "prompt_tokens", "pg_min_saved", "grad_norms", "beta", "atree_verdict", "atree_lines", "atree_rc", "engaged_on", "arms", "max_loss_diff_on_vs_off", "max_grad_diff_on_vs_off", "has_host_input_embedding", "decision_line", "torchrun_rc", "ranks", "rank_errors", "gen_modes", "has_generate_streams_blocks", "init_method", "backend", "has_set_unshard_async_op", "embed_on_cpu", "embed_device", "head_device", "embed_dtensor", "head_dtensor", "unshard_async_calls", "max_loss_diff_vs_unsharded", "shard_msg", "packed_rows_verdict", "text_prepended_hook", "leak", "leak_position_ids_shapes", "train_position_ids_shapes", "row_tokens", "indexer_vectorised", "stats_delta", "eos_counts", "assistant_only", "kept_tokens", "kept_image_tokens", "kept_labels_text", "collator_class", "ring_backends", "ring_pinned_runs", "processing_class_type", "trainer_processing_class", "hs_fallback_wrapped", "reward_margins", "ref_model_is_none", "resolved_packing", "resolved_padding_free", "global_steps", "pack_stats_after", "uncompiled_msgs", "attn_implementation", "flex_mask_keeps_padding_marker", "flex_mask_calls_per_step", "pattern", "guard_failures_per_step", "seq_packing_grad_ok", "rewards", "granite_gc_calls_on", "granite_kept_mib", "granite_loss_eq", "granite_grad_rel", "layer_forward_name", "seq_lens", "aux_gc_ok", "aux_gc_rel", "aux_loss_step0", "aux_loss_enabled", "router_logits_requested", "step1_loss", "stock_step1_loss", "oracle_nll", "oracle_nll_plus_aux", "rel_vs_stock", "rel_vs_nll_plus_aux", "rel_vs_nll_only", "stock_same_tokens", "moesft_matches_stock", "moesft_keeps_aux", "moesft_drops_aux", "teacher_device", "teacher_trainable", "trainer_class", "tokens", "decode_forced_eq_default", "graph_backend_available", "bswap_equal", "bswap_grad_maxdiff", "n_layers", "layer_forward_names", "block_modes_",  "bswap_grads_compared", "bswap_grew", "slot_pool_message", "ckpt_in_forward", "unsloth_offload_refusal", "swapped_indices", "recompile_logs", "unique_graphs_per_step", "reload_ok", "reload_loss_rel", "ddp_find_unused_default", "loader", "block_modes", "block_compile_failed_msgs", "lora_out_dtypes", "lora_grads_nonzero", "lora_tensors", "fp32_fused_hook", "fp32_fused_calls", "fp32_fused_bf16", "bf16_autocast_only_modules", "fused_qkv_layers", "param_dtype_after", "pinned_in_config", "use_reentrant_config", "use_reentrant_after_init", "aux_loss_enabled", "n_gates", "gates_with_grad", "router_grad_rel_max", "router_grad_rel_vs_nll_only_max", "rlaux_keeps_aux", "rlaux_ok", "train_device", "dora_triton_loaded", "dora_triton_kernel_ok", "dora_cross_term", "dora_mm_out_dtype", "head_chunked_calls", "lm_head_lora_tensors", "lm_head_lora_B_moved", "module_head_calls", "lm_head_type", "fold_stats", "skinny_triton_loaded", "skinny_calls_train", "skinny_calls_greedy", "rollout_shas", "greedy_logits_sha", "gc_kept_max", "kept_default", "kept_vllm_order", "kept_rollout", "vllm_order_eq_rollout", "lora_fused_passes_after", "thread_cap", "cores", "fit_real", "fit_fake32", "moe_backend", "moe_probe_state", "grouped_mm_trains_on", "torch_version_hip", "is_rocm_torch", "grpo_fp32_lm_head_default", "truncated_support_auto", "score_centering_mode", "logprob_dtype", "logprob_maxabs_vs_fp64", "probe_at_import", "probe_after_load", "import_unsloth_s", "load_train_s",
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
