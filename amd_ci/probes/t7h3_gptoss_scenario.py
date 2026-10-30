"""T7h.3 scenario (run by t7h3_gptoss_probe.py with the state's zoo checkout first on PYTHONPATH). Observes only.

1. Triton facts: version, file, active backend target, one tiny kernel launch on the GPU, torch.compile (inductor) on the GPU.
2. Random-weight GPT-OSS (bf16 where supported), zoo patches applied: prefill peak memory and greedy tokens with the
   eager kill switch (UNSLOTH_GPT_OSS_EAGER_INFERENCE=1) vs flex (=0), at 2k and 4k, plus how many forwards took the flex
   inference route (zoo's _gpt_oss_flex_inference returning True), counted per arm.
3. trl-internal-testing/tiny-GptOssForCausalLM through FastLanguageModel with load_in_4bit=True: greedy generate with each
   arm (tokens compared), then LoRA + 3 SFT steps (losses, finite grads).
Writes JSON to argv[1] after every section so a crash keeps what ran.
"""
import json
import os
import sys
import time
import traceback

OUT = sys.argv[1]
rec = {"sections": {}}


def save():
    with open(OUT, "w", encoding = "utf-8") as fh:
        json.dump(rec, fh, indent = 1, default = str)


def err(e):
    return (f"{type(e).__name__}: {e}")[:800] + " | " + traceback.format_exc()[-1200:]


os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")
os.environ.setdefault("UNSLOTH_MODEL_NAME", "unsloth/gpt-oss-test")
try:
    import unsloth  # noqa: F401  (before torch / transformers, as users import it)

    rec["unsloth_file"] = unsloth.__file__
except Exception as e:  # noqa: BLE001
    rec["unsloth_import_error"] = (f"{type(e).__name__}: {e}")[:800] + " | " + traceback.format_exc()[-1500:]
import torch  # noqa: E402

rec["torch"] = {"version": torch.__version__, "hip": getattr(torch.version, "hip", None),
                "is_available": torch.cuda.is_available()}
if torch.cuda.is_available():
    rec["torch"]["device_name"] = torch.cuda.get_device_name(0)
    rec["torch"]["arch"] = getattr(torch.cuda.get_device_properties(0), "gcnArchName", None)
save()

# 1. Triton facts.
s = rec["sections"]["triton"] = {}
try:
    import triton
    import triton.language as tl

    s["version"] = triton.__version__
    s["file"] = triton.__file__
    try:
        from importlib.metadata import distributions

        s["dists"] = sorted(f"{d.metadata['Name']}=={d.version}" for d in distributions()
                            if "triton" in (d.metadata["Name"] or "").lower())
    except Exception as e:  # noqa: BLE001
        s["dists_error"] = err(e)
    s["target"] = str(triton.runtime.driver.active.get_current_target())

    @triton.jit
    def _add(x, y, o, n, BLOCK: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = i < n
        tl.store(o + i, tl.load(x + i, mask = m) + tl.load(y + i, mask = m), mask = m)

    a = torch.randn(1000, device = "cuda"); b = torch.randn(1000, device = "cuda"); o = torch.empty_like(a)
    _add[(4,)](a, b, o, 1000, BLOCK = 256)
    torch.cuda.synchronize()
    s["kernel_ok"] = bool(torch.equal(o, a + b))
except Exception as e:  # noqa: BLE001
    s["error"] = err(e)
try:
    f = torch.compile(lambda x: torch.nn.functional.gelu(x) * 2 + 1)
    x = torch.randn(64, 64, device = "cuda")
    t0 = time.time(); y = f(x); torch.cuda.synchronize()
    s["compile_gpu_ok"] = bool(torch.allclose(y, torch.nn.functional.gelu(x) * 2 + 1, atol = 1e-5))
    s["compile_gpu_s"] = round(time.time() - t0, 1)
except Exception as e:  # noqa: BLE001
    s["compile_gpu_error"] = err(e)
save()

# 2. Random-weight GPT-OSS: eager kill switch vs flex.
s = rec["sections"]["flex_random"] = {}
try:
    from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig
    import transformers.models.gpt_oss.modeling_gpt_oss as M
    from unsloth_zoo.temporary_patches import gpt_oss as G
    import unsloth_zoo

    rec["unsloth_zoo_file"] = unsloth_zoo.__file__
    COUNT = {"flex": 0, "eager": 0}
    _orig = G._gpt_oss_flex_inference

    def _count(*a, **k):
        r = _orig(*a, **k)
        COUNT["flex" if r else "eager"] += 1
        return r

    G._gpt_oss_flex_inference = _count
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation = False) else torch.float16
    s["dtype"] = str(dtype)
    torch.manual_seed(0)
    L = 4
    c = GptOssConfig(hidden_size = 512, intermediate_size = 512, num_hidden_layers = L, num_attention_heads = 32,
                     num_key_value_heads = 8, head_dim = 64, num_local_experts = 4, num_experts_per_tok = 2,
                     vocab_size = 1024, max_position_embeddings = 8192, sliding_window = 128,
                     layer_types = ["sliding_attention", "full_attention"] * (L // 2))
    c._attn_implementation = "eager"
    model = M.GptOssForCausalLM(c).to("cuda", dtype).eval()
    G.patch_GptOssAttention(); G.patch_GptOssModel()
    ids = torch.randint(1, 1024, (1, 4096), device = "cuda")
    for arm, flag in (("eager", "1"), ("flex", "0"), ("eager", "1"), ("flex", "0")):
        os.environ["UNSLOTH_GPT_OSS_EAGER_INFERENCE"] = flag
        COUNT["flex"] = COUNT["eager"] = 0
        r = {}
        with torch.no_grad():
            for T in (2048, 4096):
                try:
                    torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
                    base = torch.cuda.memory_allocated()
                    model(input_ids = ids[:, :T], use_cache = False, logits_to_keep = 1)
                    torch.cuda.synchronize()
                    r[f"prefill_{T}_peak_mib"] = round((torch.cuda.max_memory_allocated() - base) / 2**20, 1)
                except Exception as e:  # noqa: BLE001
                    r[f"prefill_{T}_error"] = err(e)
            try:
                g = model.generate(ids[:, :512], attention_mask = torch.ones_like(ids[:, :512]), max_new_tokens = 32,
                                   do_sample = False, pad_token_id = 0, output_scores = True,
                                   return_dict_in_generate = True)
                r["tokens"] = g.sequences[0, 512:].tolist()
                top2 = [torch.topk(sc[0].float(), 2).values for sc in g.scores]
                r["min_margin"] = round(float(min(float(t[0] - t[1]) for t in top2)), 5)
            except Exception as e:  # noqa: BLE001
                r["generate_error"] = err(e)
        r["route_counts"] = dict(COUNT)
        s.setdefault("rounds", []).append({"arm": arm, **r})
        save()
except Exception as e:  # noqa: BLE001
    s["error"] = err(e)
save()

# 3. Tiny GPT-OSS through FastLanguageModel, load_in_4bit=True: generate both arms + 3 SFT steps.
s = rec["sections"]["tiny_4bit"] = {}
try:
    os.environ["UNSLOTH_GPT_OSS_EAGER_INFERENCE"] = "0"
    from unsloth import FastLanguageModel
    try:
        import bitsandbytes as bnb

        s["bitsandbytes"] = bnb.__version__
    except Exception as e:  # noqa: BLE001
        s["bitsandbytes_error"] = err(e)
    from unsloth_zoo.temporary_patches import gpt_oss as G2

    COUNT2 = {"flex": 0, "eager": 0}
    _o2 = G2._gpt_oss_flex_inference

    def _c2(*a, **k):
        r = _o2(*a, **k)
        COUNT2["flex" if r else "eager"] += 1
        return r

    G2._gpt_oss_flex_inference = _c2
    model, tok = FastLanguageModel.from_pretrained("trl-internal-testing/tiny-GptOssForCausalLM", max_seq_length = 512,
                                                   load_in_4bit = True)
    n4 = sum(1 for m in model.modules() if "4bit" in type(m).__name__.lower())
    s["linear4bit_modules"] = n4
    FastLanguageModel.for_inference(model)
    ids = torch.randint(5, 200, (1, 64), device = "cuda")
    for arm, flag in (("eager", "1"), ("flex", "0")):
        os.environ["UNSLOTH_GPT_OSS_EAGER_INFERENCE"] = flag
        COUNT2["flex"] = COUNT2["eager"] = 0
        try:
            with torch.no_grad():
                g = model.generate(input_ids = ids, attention_mask = torch.ones_like(ids), max_new_tokens = 16,
                                   do_sample = False, output_scores = True, return_dict_in_generate = True)
            top2 = [torch.topk(sc[0].float(), 2).values for sc in g.scores]
            s[f"gen_{arm}"] = {"tokens": g.sequences[0, 64:].tolist(), "route_counts": dict(COUNT2),
                               "min_margin": round(float(min(float(t[0] - t[1]) for t in top2)), 5)}
        except Exception as e:  # noqa: BLE001
            s[f"gen_{arm}"] = {"error": err(e), "route_counts": dict(COUNT2)}
        save()
    os.environ["UNSLOTH_GPT_OSS_EAGER_INFERENCE"] = "0"
    try:
        FastLanguageModel.for_training(model)
        model = FastLanguageModel.get_peft_model(model, r = 8, lora_alpha = 8,
                                                 target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"],
                                                 use_gradient_checkpointing = "unsloth")
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr = 1e-3)
        x = torch.randint(5, 200, (2, 128), device = "cuda")
        losses, gnorms = [], []
        for _ in range(3):
            out = model(input_ids = x, labels = x)
            out.loss.backward()
            gn = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1e9)
            opt.step(); opt.zero_grad(set_to_none = True)
            losses.append(round(float(out.loss), 5)); gnorms.append(round(float(gn), 5))
        s["sft"] = {"losses": losses, "grad_norms": gnorms}
    except Exception as e:  # noqa: BLE001
        s["sft"] = {"error": err(e)}
except Exception as e:  # noqa: BLE001
    s["error"] = err(e)
save()
rec["done"] = True
save()
