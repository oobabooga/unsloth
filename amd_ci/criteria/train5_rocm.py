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
TRAIN = ("flm16", "flm4", "gcmem", "vision", "moe", "gemma4", "gemma4_gcrec", "qwen38", "dsv4", "mllama", "rl_decode", "ctnvfp4",
         "ctmxfp8", "mxfp4probe", "grpo_vl", "dpo_vl", "gcnr_qwen3", "gcdef_qwen3", "gcnr_gemma3", "gcdef_gemma3",
         "lora_noac", "st_minilm", "st_qwen3", "hd512_llama", "hd512_gemma4", "hd512_gemma4_nc", "bclens", "bswap", "dec_whisper", "dec_gemma3n", "dec_mistral", "moeaux_nr", "moeaux_gc_qwen3moe", "moeaux_gc_gptoss", "gkd", "granite_gc", "plelens", "padmix_qwen3", "padmix_gemma4", "padmix_gemma3", "padmix_gemma3_flex", "padmix_gemma3_flex_ones", "padmix_gemma4_ones", "grpo_txt", "grpo_zadv", "grpo_zadv_zoff", "pfree_llama", "pfree_qwen3", "sftdef_steps", "sftdef_epoch", "dpo_txt", "orpo_txt", "orpo_txt_psoff", "orpo_txt_unwrap", "qwen38_long", "sft_eos", "aol_qwen25vl", "aol_gemma3_unsloth", "aol_gemma3_trl", "aol_qwen25vl_none", "aol_gemma3_none", "cp_ring_kernels", "gkd_proc_qwen35", "gkd_proc_gemma3", "vision_gemma3", "mrope_pack", "seqcls", "tok_save", "fsdp2_io", "fsdp2_gen", "pauto", "tphost", "atree", "grpo_pg", "fp32attn", "warm", "flm16_gcsave", "lpad", "fft16", "fft16_sroff", "grpo_pack", "g2nr_flex")
# Cells whose numbers the PR changes by design (the CT loader fixes the expert values; RL sampling / the fused GRPO
# route; the probe's numbers are errors, not losses): no per-step loss / memory equality, the rest still applies.
NO_LOSS_EQ = ("ctnvfp4", "ctmxfp8", "mxfp4probe", "grpo_vl", "sftdef_epoch", "aol_qwen25vl", "aol_gemma3_unsloth", "aol_gemma3_trl", "aol_qwen25vl_none", "aol_gemma3_none", "sft_eos", "g2nr_flex")
ENGAGED = {"mxfp4probe": "engaged", "grpo_vl": "grpo_fused_engaged", "dpo_vl": "dpo_chunked_engaged"}

# Why a PR's own new path does or does not engage on ROCm; rendered with the table.
NOTES = {
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
        if cell == "gemma4_gcrec" and b.get("recomputed") and not h.get("recomputed"):
            worse.append(f"gemma4_gcrec: decoder layers recomputed at base, not at head ({h.get('recompute_summary')})")
        if cell == "gemma4_gcrec" and len(((h.get("lost_checkpointing") or {}).get("lost") or [])) > len(((b.get("lost_checkpointing") or {}).get("lost") or [])):
            worse.append(f"gemma4_gcrec: more layer classes lost checkpointing at head {(h.get('lost_checkpointing') or {}).get('lost')}"[:300])
        if cell.startswith("moeaux_gc") and b.get("aux_gc_ok") and not h.get("aux_gc_ok"):
            worse.append(f"{cell}: aux-loss grad norm GC vs no GC within 2e-2 at base, not at head ({h.get('aux_gc_rel')})")
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
    for k in ("recompute_summary", "recomputed", "gcon_mlp_total_per_step", "gcon_gc_calls_per_step", "control_ok", "gcoff_mlp_total_per_step", "gcoff_peak_alloc_mb", "lost_checkpointing", "layer_forward_names", "seq_packing_flags", "saved_attention_stats", "gc_save_attention_env", "autotune_calls", "warm_loss_eq", "run_attention_calls", "sdpa_q_dtypes", "max_abs_logprob_diff_vs_hf_fp32_eager", "has_flash", "has_xformers", "has_keeps_fp32", "param_dtype", "pg_build_calls", "pg_layouts", "pg_flags", "pg_enabled", "atree_verdict", "atree_lines", "atree_rc", "engaged_on", "arms", "max_loss_diff_on_vs_off", "max_grad_diff_on_vs_off", "has_host_input_embedding", "decision_line", "torchrun_rc", "ranks", "rank_errors", "gen_modes", "has_generate_streams_blocks", "init_method", "backend", "has_set_unshard_async_op", "embed_on_cpu", "embed_device", "head_device", "embed_dtensor", "head_dtensor", "unshard_async_calls", "max_loss_diff_vs_unsharded", "shard_msg", "packed_rows_verdict", "text_prepended_hook", "leak", "leak_position_ids_shapes", "train_position_ids_shapes", "row_tokens", "indexer_vectorised", "stats_delta", "eos_counts", "assistant_only", "kept_tokens", "kept_image_tokens", "kept_labels_text", "collator_class", "ring_backends", "ring_pinned_runs", "processing_class_type", "trainer_processing_class", "hs_fallback_wrapped", "reward_margins", "ref_model_is_none", "resolved_packing", "resolved_padding_free", "global_steps", "pack_stats_after", "uncompiled_msgs", "attn_implementation", "flex_mask_keeps_padding_marker", "flex_mask_calls_per_step", "pattern", "guard_failures_per_step", "seq_packing_grad_ok", "rewards", "granite_gc_calls_on", "granite_kept_mib", "granite_loss_eq", "granite_grad_rel", "layer_forward_name", "seq_lens", "aux_gc_ok", "aux_gc_rel", "aux_loss_step0", "teacher_device", "teacher_trainable", "trainer_class", "tokens", "decode_forced_eq_default", "graph_backend_available", "bswap_equal", "bswap_grad_maxdiff", "n_layers", "layer_forward_names", "block_modes_",  "bswap_grads_compared", "bswap_grew", "slot_pool_message", "ckpt_in_forward", "unsloth_offload_refusal", "swapped_indices", "recompile_logs", "unique_graphs_per_step", "reload_ok", "reload_loss_rel", "ddp_find_unused_default", "loader", "block_modes", "block_compile_failed_msgs", "lora_out_dtypes", "lora_grads_nonzero", "lora_tensors",
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
