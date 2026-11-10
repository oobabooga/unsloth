#!/usr/bin/env python3
"""One cell of the 2026-10-28 unsloth main check (25fd237aad -> 987731ec56) on one state. Observes only.

Modes (JSON to --out, never stdout):
  decisions  routing / gating decisions the day's PRs touch, read directly:
             block CUDA graphs auto (#15768), torch varlen + varlen-over-flex switch (#15514),
             FP8 rowwise routing + outputs (#15584 / #15791), CE chunk config (#15744),
             MoE heuristic dW tile (#15505), SDPA decode pin (#15525)
  sft        tiny Mistral FastModel LoRA SFT (--quant 16|4), padded, 8 steps; block-compile auto state,
             forward pre-hooks, losses
  packed     tiny Mistral head_dim 256 FastModel LoRA SFT with packing=True; varlen-over-flex switch calls
  infer      tiny Mistral FastModel for_inference greedy generate, token ids + first-step logits digest

The tiny model is built from a fixed seed on CPU once per --model-dir (same bytes for base and head).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import traceback
from pathlib import Path

TOKENIZER = "trl-internal-testing/tiny-MistralForCausalLM-0.1"
WORDS = ("the of and to in is was for on as with by he at from his an were are which this be "
         "or has had not but first one their its new after who they have two also been other more all "
         "she time year can city world river music house water light green small large north south").split()


def build_model(model_dir: Path, head_dim: int) -> None:
    if (model_dir / "config.json").is_file():
        return
    import torch
    import transformers
    tmp = model_dir.with_name(model_dir.name + ".tmp")
    tok = transformers.AutoTokenizer.from_pretrained(TOKENIZER)
    torch.manual_seed(0)
    heads = 4 if head_dim <= 64 else 2
    cfg = transformers.MistralConfig(
        vocab_size = len(tok), hidden_size = 256, intermediate_size = 512, num_hidden_layers = 2,
        num_attention_heads = heads, num_key_value_heads = max(1, heads // 2), head_dim = head_dim,
        sliding_window = None, max_position_embeddings = 4096, bos_token_id = tok.bos_token_id,
        eos_token_id = tok.eos_token_id, tie_word_embeddings = False,
    )
    model = transformers.MistralForCausalLM(cfg).to(torch.bfloat16)
    model.save_pretrained(tmp)
    tok.save_pretrained(tmp)
    tmp.rename(model_dir)


def make_docs(n = 64, seed = 0):
    rng = random.Random(seed)
    return [" ".join(rng.choice(WORDS) for _ in range(rng.choice([8, 15, 25, 40, 60]))) + "." for _ in range(n)]


def digest(t) -> str:
    import torch
    return hashlib.sha256(t.detach().to(torch.float32).cpu().contiguous().numpy().tobytes()).hexdigest()[:16]


def versions(res):
    import torch
    import transformers
    import unsloth
    res["unsloth_file"] = unsloth.__file__
    v = {"torch": torch.__version__, "hip": torch.version.hip, "cuda": torch.version.cuda,
         "transformers": transformers.__version__, "unsloth": getattr(unsloth, "__version__", None)}
    for m in ("trl", "peft", "unsloth_zoo", "bitsandbytes", "triton"):
        try:
            v[m] = __import__(m).__version__
        except Exception as e:  # noqa: BLE001
            v[m] = f"ERR {type(e).__name__}: {str(e)[:200]}"
    p = torch.cuda.get_device_properties(0)
    v["device"], v["arch"] = p.name, getattr(p, "gcnArchName", None)
    v["capability"] = list(torch.cuda.get_device_capability(0))
    res["versions"] = v


def safe(fn):
    try:
        return fn()
    except Exception as e:  # noqa: BLE001
        return f"ERR {type(e).__name__}: {str(e)[:300]}"


def mode_decisions(res, args):
    import torch
    dev = torch.device("cuda", 0)
    import unsloth  # noqa: F401
    import unsloth.models._block_compile as bc
    d = {}
    d["COMPILE_OPTIONS_triton_cudagraphs"] = bc._COMPILE_OPTIONS.get("triton.cudagraphs")
    d["CUDAGRAPH_MODE"] = getattr(bc, "_CUDAGRAPH_MODE", "<absent>")
    d["has_auto"] = hasattr(bc, "_auto_platform_reason")
    if hasattr(bc, "_auto_platform_reason"):
        d["auto_platform_reason"] = safe(bc._auto_platform_reason)
        hs = torch.zeros(2, 8, 16, device = dev, dtype = torch.bfloat16)
        layer = torch.nn.Linear(16, 16).to(dev)
        for p in layer.parameters():
            p.requires_grad_(False)
        d["auto_static_reason_cuda_tensor"] = safe(lambda: bc._auto_static_reason(layer, hs))
    res["block_graphs"] = d

    # varlen (#15514)
    import unsloth.utils.hf_packed_attention as hpa
    v = {"torch_varlen": safe(lambda: repr(hpa._torch_varlen(dev)))}
    try:
        from torch.nn.attention.varlen import varlen_attn  # noqa: F401
        v["torch_has_varlen_attn"] = True
    except Exception as e:  # noqa: BLE001
        v["torch_has_varlen_attn"] = f"no: {type(e).__name__}"
    import unsloth.models._utils as mu
    v["has_switch"] = hasattr(mu, "use_sdpa_for_packed_head_dim_flex")
    if v["has_switch"]:
        # Reach the varlen gate on purpose: a head_dim-256 config on flex with the flex preference forced on.
        import transformers
        hpa.enable_hf_packed_attention()
        cfg = transformers.MistralConfig(vocab_size = 128, hidden_size = 256, intermediate_size = 512,
                                         num_hidden_layers = 1, num_attention_heads = 2, num_key_value_heads = 1,
                                         head_dim = 256)
        cfg._attn_implementation = "flex_attention"
        model = transformers.MistralForCausalLM(cfg).to(torch.bfloat16)
        calls = {"takes_model": 0, "torch_varlen": 0}
        orig_pref, orig_takes, orig_tv = mu._prefers_flex_for_head_dim, mu._packed_varlen_takes_model, hpa._torch_varlen
        mu._prefers_flex_for_head_dim = lambda c: True

        def takes(*a, **k):
            calls["takes_model"] += 1
            return orig_takes(*a, **k)

        def tv(device):
            calls["torch_varlen"] += 1
            return orig_tv(device)
        mu._packed_varlen_takes_model, hpa._torch_varlen = takes, tv
        try:
            v["forced_switch_returned"] = safe(lambda: mu.use_sdpa_for_packed_head_dim_flex(model, 512, 2048))
        finally:
            mu._prefers_flex_for_head_dim, mu._packed_varlen_takes_model, hpa._torch_varlen = orig_pref, orig_takes, orig_tv
        v["forced_switch_attn_after"] = getattr(model.config, "_attn_implementation", None)
        v["forced_switch_calls"] = calls
        v["orig_sdpa_installed"] = hpa._ORIG_SDPA[0] is not None
    res["varlen"] = v

    # FP8 rowwise (#15584 / #15791)
    import unsloth.kernels.fp8 as f8
    f = {"backend": safe(lambda: f8._fp8_rowwise_backend(dev)),
         "has_direct_route": hasattr(f8, "_fp8_rowwise_direct_route"),
         "dequant_max_rows_large": safe(lambda: f8._fp8_dequant_max_rows(dev, 4096 * 14336))
         if hasattr(f8, "_fp8_dequant_max_rows") else "<absent>"}
    spies = {}
    for name in ("_fp8_rowwise_direct", "_fbgemm_fp8_linear_compiled"):
        if hasattr(f8, name):
            orig = getattr(f8, name)

            def mk(orig, name):
                def w(*a, **k):
                    spies[name] = spies.get(name, 0) + 1
                    return orig(*a, **k)
                return w
            setattr(f8, name, mk(orig, name))
    g = torch.Generator(device = "cpu").manual_seed(0)
    N, K = 256, 512
    w16 = (torch.randn(N, K, generator = g) * 0.05)
    scale = (w16.abs().amax(dim = 1, keepdim = True) / 448.0).clamp(min = 1e-8)
    wq = (w16 / scale).to(torch.float8_e4m3fn).to(dev)
    scale = scale.to(torch.float32).to(dev)
    ref_w = wq.to(torch.float32) * scale
    cases = {}
    for rows in (1, 8, 16, 17, 32, 64, 65, 128, 512):
        X = (torch.randn(rows, K, generator = g) * 0.5).to(torch.bfloat16).to(dev)
        rec = {}
        if f["has_direct_route"]:
            rec["direct_route"] = safe(lambda: f8._fp8_rowwise_direct_route(X, wq, scale))
        before = dict(spies)
        try:
            with torch.no_grad():
                y = f8.fp8_linear(X, wq, scale)
            torch.cuda.synchronize()
            ref = X.to(torch.float32) @ ref_w.t()
            rec.update(ok = True, digest = digest(y), dtype = str(y.dtype),
                       max_abs_err_vs_fp32 = float((y.float() - ref).abs().max()))
        except Exception as e:  # noqa: BLE001
            rec.update(ok = False, error = f"{type(e).__name__}: {str(e)[:300]}")
        rec["spies"] = {k: spies.get(k, 0) - before.get(k, 0) for k in spies}
        cases[str(rows)] = rec
    f["cases"] = cases
    res["fp8"] = f

    # CE chunk config (#15744)
    import unsloth.kernels.cross_entropy_loss as ce
    cap = tuple(torch.cuda.get_device_capability(0))
    res["ce"] = {"capability": list(cap), "chunk_config_for_capability": repr(ce._CHUNKED_FORWARD_CONFIGS.get(cap))}

    # MoE heuristic dW tile (#15505): gpt-oss K = 2880
    from unsloth.kernels.moe import autotune_cache as ac
    m = {}
    for n, k in ((5760, 2880), (2880, 2880), (1024, 576)):
        m[f"{n}x{k}"] = safe(lambda: [repr(c) for c in ac._get_heuristic_configs(n, k, torch.bfloat16, torch.bfloat16, dev)])
    m["wide_dw_device_ok"] = safe(lambda: ac._wide_dw_device_ok(dev)) if hasattr(ac, "_wide_dw_device_ok") else "<absent>"
    res["moe"] = m

    # SDPA decode pin (#15525)
    from unsloth.utils import attention_dispatch as ad
    q = torch.randn(1, 8, 1, 64, generator = g).to(torch.bfloat16).to(dev)
    kk = torch.randn(1, 8, 32, 64, generator = g).to(torch.bfloat16).to(dev)
    vv = torch.randn(1, 8, 32, 64, generator = g).to(torch.bfloat16).to(dev)
    s = {}
    for name, ctx in (("default", None), ("math_only", "math")):
        try:
            if ctx == "math":
                from torch.nn.attention import SDPBackend, sdpa_kernel
                with sdpa_kernel([SDPBackend.MATH]):
                    y = ad.decode_scaled_dot_product_attention(q, kk, vv)
            else:
                y = ad.decode_scaled_dot_product_attention(q, kk, vv)
            ref = torch.nn.functional.scaled_dot_product_attention(q.float(), kk.float(), vv.float())
            s[name] = {"ok": True, "digest": digest(y), "max_abs_err_vs_fp32": float((y.float() - ref).abs().max())}
        except Exception as e:  # noqa: BLE001
            s[name] = {"ok": False, "error": f"{type(e).__name__}: {str(e)[:300]}"}
    s["cudnn_free_index"] = safe(lambda: ad._decode_cudnn_free_index()) if hasattr(ad, "_decode_cudnn_free_index") else "<absent>"
    res["sdpa_decode"] = s


def _hooks(model):
    n_pre = sum(len(m._forward_pre_hooks) for m in model.modules())
    watched = sum(1 for m in model.modules() if m.__dict__.get("_unsloth_watch_hidden_states", False))
    return {"forward_pre_hooks": n_pre, "watch_hidden_states_flags": watched}


def _bc_state():
    import unsloth.models._block_compile as bc
    st = getattr(bc, "_CUDAGRAPH_STATE", {}) or {}
    auto = st.get("auto") or {}
    return {"off": st.get("off"), "auto_phase": auto.get("phase"), "auto_why": auto.get("why"),
            "auto_decision": auto.get("decision"), "auto_passes": auto.get("passes"),
            "triton_cudagraphs": bc._COMPILE_OPTIONS.get("triton.cudagraphs"),
            "mode": getattr(bc, "_CUDAGRAPH_MODE", "<absent>")}


def mode_train(res, args, packed: bool):
    import torch
    import unsloth
    from unsloth import FastModel
    import unsloth.models._block_compile as bc
    import unsloth.models._utils as mu
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer
    versions(res)
    wrapped = []
    orig_patch = bc.patch_block_compile

    def spy_patch(model):
        r = orig_patch(model)
        wrapped.append([getattr(c, "__name__", str(c)) for c in (r or [])])
        return r
    bc.patch_block_compile = spy_patch
    switch_calls = []
    if hasattr(mu, "use_sdpa_for_packed_head_dim_flex"):
        orig_sw = mu.use_sdpa_for_packed_head_dim_flex

        def spy_sw(model, *a, **k):
            before = getattr(getattr(model, "config", None), "_attn_implementation", None)
            r = orig_sw(model, *a, **k)
            switch_calls.append({"returned": r, "attn_before": before,
                                 "attn_after": getattr(getattr(model, "config", None), "_attn_implementation", None)})
            return r
        mu.use_sdpa_for_packed_head_dim_flex = spy_sw
    res["stage"] = "load"
    model, tok = FastModel.from_pretrained(str(args.model_dir), max_seq_length = 512, dtype = torch.bfloat16,
                                           load_in_4bit = args.quant == 4, load_in_16bit = args.quant == 16)
    res["model_class"] = type(model).__name__
    res["attn_impl_loaded"] = getattr(model.config, "_attn_implementation", None)
    res["n_linear4bit"] = sum(1 for m in model.modules() if type(m).__name__ == "Linear4bit")
    res["hooks_after_load"] = _hooks(model)
    model = FastModel.get_peft_model(model, r = 8, lora_alpha = 16, lora_dropout = 0.0, random_state = 0,
                                     target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                                                       "gate_proj", "up_proj", "down_proj"],
                                     use_gradient_checkpointing = "unsloth")
    res["hooks_after_peft"] = _hooks(model)
    tt = getattr(tok, "tokenizer", tok)
    if tt.pad_token is None:
        tt.pad_token = tt.eos_token
    ds = Dataset.from_dict({"text": [d + (tt.eos_token or "") for d in make_docs()]})
    cfg = SFTConfig(output_dir = str(args.out.parent / (args.out.stem + "_trainer")),
                    per_device_train_batch_size = 4, gradient_accumulation_steps = 1, max_steps = 8,
                    learning_rate = 1e-3, logging_steps = 1, bf16 = True, report_to = "none", seed = 0,
                    dataset_text_field = "text", max_length = 256 if packed else 128, save_strategy = "no",
                    dataset_num_proc = 1, warmup_steps = 0, packing = packed)
    res["stage"] = "train"
    trainer = SFTTrainer(model = model, train_dataset = ds, processing_class = tt, args = cfg)
    res["attn_impl_after_trainer_init"] = getattr(model.config, "_attn_implementation", None)
    trainer.train()
    res["block_compile_wrapped"] = wrapped
    res["block_graphs"] = _bc_state()
    res["hooks_after_train"] = _hooks(model)
    res["varlen_switch_calls"] = switch_calls
    res["attn_impl_after_train"] = getattr(model.config, "_attn_implementation", None)
    logs = [h for h in trainer.state.log_history if "loss" in h]
    res["losses"] = [float(h["loss"]) for h in logs]
    res["grad_norms"] = [float(h["grad_norm"]) if h.get("grad_norm") is not None else None for h in logs]
    if packed:
        try:
            import unsloth.utils.hf_packed_attention as hpa
            res["packed_stats"] = dict(getattr(hpa, "HF_PACKED_ATTENTION_STATS", {}) or {})
        except Exception as e:  # noqa: BLE001
            res["packed_stats"] = repr(e)


def mode_infer(res, args):
    import torch
    from unsloth import FastModel
    versions(res)
    model, tok = FastModel.from_pretrained(str(args.model_dir), max_seq_length = 512, dtype = torch.bfloat16,
                                           load_in_4bit = False, load_in_16bit = True)
    FastModel.for_inference(model)
    tt = getattr(tok, "tokenizer", tok)
    ids = tt("the river music house water", return_tensors = "pt").input_ids.to("cuda")
    with torch.no_grad():
        logits = model(input_ids = ids).logits
        out = model.generate(input_ids = ids, max_new_tokens = 16, do_sample = False, use_cache = True)
    res["prompt_len"] = int(ids.shape[1])
    res["logits_digest"] = digest(logits[0, -1])
    res["logits_last_head"] = [round(float(x), 4) for x in logits[0, -1, :8]]
    res["generated_ids"] = out[0, ids.shape[1]:].tolist()
    res["n_new"] = len(res["generated_ids"])


def mode_moe_control(res, args):
    """Control for tests/test_multi_lora_experts.py: the same tiny Qwen3.5 MoE (fp32, GPU) WITHOUT multi-LoRA.
    Steps run in order and each is recorded; a HIP illegal access aborts the process, so the JSON is
    rewritten after every step (the last recorded step tells where it died)."""
    import torch
    import unsloth  # noqa: F401
    import transformers
    import peft
    import copy
    versions(res)
    res["steps"] = {}

    def step(name, fn):
        res["current_step"] = name
        args.out.write_text(json.dumps(res, indent = 2, default = str), encoding = "utf-8")
        try:
            fn()
            torch.cuda.synchronize()
            res["steps"][name] = "ok"
        except Exception as e:  # noqa: BLE001
            res["steps"][name] = f"ERR {type(e).__name__}: {str(e)[:400]}"
        args.out.write_text(json.dumps(res, indent = 2, default = str), encoding = "utf-8")

    from huggingface_hub import snapshot_download
    cls = getattr(transformers, "Qwen3_5MoeForConditionalGeneration", None)
    if cls is None:
        res["steps"]["load"] = "no Qwen3_5Moe class"
        return
    path = snapshot_download("trl-internal-testing/tiny-Qwen3_5MoeForConditionalGeneration-3.6")
    holder = {}
    step("load", lambda: holder.__setitem__("m", cls.from_pretrained(path, torch_dtype = torch.float32).eval().to("cuda")))
    ids = torch.randint(0, 100, (3, 6), generator = torch.Generator().manual_seed(1)).to("cuda")
    step("base_forward", lambda: holder["m"](input_ids = ids, use_cache = False).logits)
    cfg = peft.LoraConfig(r = 4, lora_alpha = 8, target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
                          target_parameters = ["mlp.experts.down_proj", "mlp.experts.gate_up_proj"], lora_dropout = 0.0)
    step("peft_wrap", lambda: holder.__setitem__("p", peft.get_peft_model(copy.deepcopy(holder["m"]), cfg)))
    step("peft_forward", lambda: holder["p"](input_ids = ids, use_cache = False).logits)
    step("peft_forward_backward", lambda: holder["p"](input_ids = ids, use_cache = False).logits.pow(2).mean().backward())
    res["current_step"] = None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required = True, choices = ("decisions", "sft", "packed", "infer", "moe_control"))
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--model-dir", type = Path)
    ap.add_argument("--head-dim", type = int, default = 64)
    ap.add_argument("--quant", type = int, default = 16, choices = (4, 16))
    args = ap.parse_args()
    res: dict = {"mode": args.mode, "quant": args.quant, "stage": "start"}
    try:
        if args.model_dir is not None:
            build_model(args.model_dir, args.head_dim)
        import unsloth  # noqa: F401  (before transformers' model code is used)
        if args.mode == "decisions":
            versions(res)
            mode_decisions(res, args)
        elif args.mode == "moe_control":
            mode_moe_control(res, args)
        elif args.mode == "infer":
            mode_infer(res, args)
        else:
            mode_train(res, args, packed = args.mode == "packed")
        res["stage"] = "done"
        res["ok"] = True
    except Exception as e:  # noqa: BLE001
        res["error"] = f"{type(e).__name__}: {e}"[:1500]
        res["traceback"] = traceback.format_exc()[-4000:]
    args.out.write_text(json.dumps(res, indent = 2, default = str), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
