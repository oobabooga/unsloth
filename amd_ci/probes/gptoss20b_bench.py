"""GPT-OSS inference benchmark through Unsloth's generate path (what users run): prefill and decode, speed and memory.
amd_ci copy of scripts/gptoss_static/bench_infer.py with --cases (CTXxBATCH list) and --gen-out (greedy token check).

    PYTHONPATH=<unsloth wt>:<zoo wt> python scripts/gptoss_b/bench_infer.py --out OUT.json \
        [--model unsloth/gpt-oss-20b-unsloth-bnb-4bit] [--contexts 1024,4096,16384] [--batches 1,8] \
        [--new-tokens 256] [--repeats 3] [--no-4bit] [--dtype bf16|fp16]

Per (context, batch):
  prefill_s  = median wall time of generate(max_new_tokens=1) (prefill + one decode step), warm
  decode_ms  = (median generate(N) - prefill_s) / (N - 1), with min_new_tokens = N (no early EOS)
  ttft_cold_s = first ever generate(max_new_tokens=1) at that shape, compile included
  peak_alloc_gb / peak_reserved_gb = during the generate(N) run, reset before it
Batch 8 uses mixed left padding (lengths ctx, 7/8 ctx, ..., 1/2 ctx). Useful prompt tokens/s counts real tokens.
Attention path: calls of every closure function named *flex* / *eager* / *attention* reachable from the
patched GptOssAttention.forward (counted per phase), plus the CUDA kernel names seen in one profiled prefill and
one profiled decode step (flex kernels contain "flex" in their Triton template name).
"""
import argparse
import json
import os
import statistics
import time

ap = argparse.ArgumentParser()
ap.add_argument("--out", required = True)
ap.add_argument("--model", default = "unsloth/gpt-oss-20b-unsloth-bnb-4bit")
ap.add_argument("--contexts", default = "1024,4096,16384")
ap.add_argument("--batches", default = "1,8")
ap.add_argument("--new-tokens", type = int, default = 256)
ap.add_argument("--repeats", type = int, default = 3)
ap.add_argument("--no-4bit", action = "store_true")
ap.add_argument("--dtype", default = "bf16", choices = ["bf16", "fp16"])
ap.add_argument("--no-profile", action = "store_true")
ap.add_argument("--cases", default = "", help = "comma list of CTXxBATCH, overrides --contexts/--batches; 'none' = no bench")
ap.add_argument("--gen-out", default = "", help = "write greedy tokens of a padded 2-prompt batch here")
ap.add_argument("--gen-tokens", type = int, default = 96)
ap.add_argument("--static-cache-len", type = int, default = 0, help = "generate with a fresh StaticCache of this size")
a = ap.parse_args()
os.environ.setdefault("UNSLOTH_DISABLE_AUTO_UPDATES", "1")

from unsloth import FastLanguageModel  # noqa: E402
import torch  # noqa: E402

if a.cases:
    cases = [] if a.cases == "none" else [tuple(int(v) for v in c.split("x")) for c in a.cases.split(",")]
else:
    cases = [(int(c), int(b)) for c in a.contexts.split(",") for b in a.batches.split(",")]
contexts = sorted({c for c, _ in cases}) or [1024]
dtype = torch.bfloat16 if a.dtype == "bf16" else torch.float16
t0 = time.time()
model, tok = FastLanguageModel.from_pretrained(
    a.model, max_seq_length = max(contexts) + a.new_tokens + 8, load_in_4bit = not a.no_4bit, dtype = dtype,
)
FastLanguageModel.for_inference(model)
load_s = time.time() - t0
dev = "cuda"
print("LOADED", {"load_s": round(load_s, 1), "free_mib": torch.cuda.mem_get_info()[0] >> 20}, flush = True)
base = model.get_base_model() if hasattr(model, "get_base_model") else model

# ---- attention path counters (closure functions of the patched attention forward) ----
COUNTS = {}


def _wrap_closures(fn, seen):
    for cell in getattr(fn, "__closure__", None) or ():
        try:
            v = cell.cell_contents
        except ValueError:
            continue
        if not callable(v) or id(v) in seen or not hasattr(v, "__code__"):
            continue
        seen.add(id(v))
        name = v.__name__
        if any(s in name.lower() for s in ("flex", "eager", "attention", "sink")):
            def counted(*args, __f = v, __n = name, **kw):
                COUNTS[__n] = COUNTS.get(__n, 0) + 1
                return __f(*args, **kw)
            counted.__name__ = name
            _wrap_closures(v, seen)
            cell.cell_contents = counted
        else:
            _wrap_closures(v, seen)


attn_cls = type(base.model.layers[0].self_attn)
_wrap_closures(attn_cls.forward, set())
attn_module = getattr(attn_cls.forward, "__module__", "?")

vocab = base.config.vocab_size
g = torch.Generator().manual_seed(0)


def make_batch(ctx, bs):
    lengths = [ctx] if bs == 1 else [max(ctx // 2, int(ctx * (1 - i / (2 * (bs - 1))))) for i in range(bs)]
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    ids = torch.full((bs, ctx), pad, dtype = torch.long)
    am = torch.zeros((bs, ctx), dtype = torch.long)
    for i, L in enumerate(lengths):
        ids[i, ctx - L:] = torch.randint(1000, min(vocab, 150000), (L,), generator = g)
        am[i, ctx - L:] = 1
    return ids.to(dev), am.to(dev), sum(lengths)


_SEEN_CACHES = set()
def _log_static_cache():
    from transformers.generation.utils import GenerationMixin
    orig = GenerationMixin._prepare_static_cache
    def wrapped(self, cache_implementation, batch_size, max_cache_len, *args, **kwargs):
        key = (cache_implementation, batch_size, max_cache_len)
        if key not in _SEEN_CACHES:
            _SEEN_CACHES.add(key)
            print("CACHE", {"impl": cache_implementation, "batch": batch_size, "max_cache_len": max_cache_len}, flush = True)
        return orig(self, cache_implementation, batch_size, max_cache_len, *args, **kwargs)
    GenerationMixin._prepare_static_cache = wrapped
_log_static_cache()


def gen(ids, am, n):
    torch.cuda.synchronize()
    t = time.perf_counter()
    with torch.no_grad():
        extra = {}
        if a.static_cache_len:
            # Unsloth's generate rewrites cache_implementation itself, so size its static cache instead of
            # passing a cache object (transformers raises on both).
            extra["max_cache_len"] = a.static_cache_len
        out = model.generate(input_ids = ids, attention_mask = am, max_new_tokens = n, min_new_tokens = n,
                             do_sample = False, pad_token_id = tok.pad_token_id or tok.eos_token_id, **extra)
    torch.cuda.synchronize()
    assert out.shape[1] == ids.shape[1] + n, out.shape
    return time.perf_counter() - t


def kernels(ids, am, n):
    from torch.profiler import ProfilerActivity, profile
    with profile(activities = [ProfilerActivity.CUDA]) as prof:
        gen(ids, am, n)
    names = {e.key for e in prof.key_averages()}
    return sorted(k for k in names if any(s in k.lower() for s in ("flex", "attention", "softmax", "sdpa")))[:20]


gen_res = None
if a.gen_out:
    msgs = [[{"role": "user", "content": "Explain in detail why the sky is blue, step by step."}],
            [{"role": "user", "content": "Write a short Python function that returns the n-th Fibonacci number, then explain how it works."}]]
    texts = [tok.apply_chat_template(m, tokenize = False, add_generation_prompt = True) for m in msgs]
    tok.padding_side = "left"
    enc = tok(texts, return_tensors = "pt", padding = True).to(dev)
    COUNTS.clear()
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens = a.gen_tokens, min_new_tokens = a.gen_tokens, do_sample = False,
                             pad_token_id = tok.pad_token_id or tok.eos_token_id, output_scores = True,
                             return_dict_in_generate = True)
    new = out.sequences[:, enc["input_ids"].shape[1]:]
    # top-2 logit margin per step, to tell near-tie divergence from a real one
    margins = []
    for sc in out.scores:
        t2 = torch.topk(sc.float(), 2, dim = -1).values
        margins.append((t2[:, 0] - t2[:, 1]).tolist())
    gen_res = {"tokens": new.tolist(), "margins": margins, "pad_lens": (enc["attention_mask"] == 0).sum(1).tolist(),
               "calls": dict(COUNTS), "text": [tok.decode(r, skip_special_tokens = True)[:300] for r in new]}
    with open(a.gen_out, "w") as f:
        json.dump(gen_res, f)
    print("GEN", {"calls": gen_res["calls"], "text0": gen_res["text"][0][:120]}, flush = True)

rows = []
for ctx, bs in cases:
    if True:
        row = {"ctx": ctx, "batch": bs}
        try:
            ids, am, useful = make_batch(ctx, bs)
            COUNTS.clear()
            row["ttft_cold_s"] = round(gen(ids, am, 1), 4)
            row["calls_prefill_cold"] = dict(COUNTS)
            gen(ids, am, 4)  # warm the decode shapes
            pre = []
            for _ in range(a.repeats):
                pre.append(gen(ids, am, 1))
            row["prefill_s"] = round(statistics.median(pre), 4)
            row["prefill_s_all"] = [round(x, 4) for x in pre]
            row["prefill_useful_tok_s"] = round(useful / row["prefill_s"], 1)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            COUNTS.clear()
            dec = []
            for _ in range(max(1, a.repeats - 1)):
                dec.append(gen(ids, am, a.new_tokens))
            row["calls_decode_run"] = dict(COUNTS)
            row["gen_N_s"] = round(statistics.median(dec), 4)
            row["decode_ms_per_token"] = round((row["gen_N_s"] - row["prefill_s"]) / (a.new_tokens - 1) * 1e3, 3)
            row["decode_tok_s"] = round(bs * 1e3 / row["decode_ms_per_token"], 1)
            row["peak_alloc_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
            row["peak_reserved_gb"] = round(torch.cuda.max_memory_reserved() / 2**30, 3)
            if not a.no_profile and (ctx, bs) == cases[0]:
                row["kernels_prefill"] = kernels(ids, am, 1)
        except torch.cuda.OutOfMemoryError as e:
            row["oom"] = str(e)[:200]
        except Exception as e:  # noqa: BLE001
            row["error"] = f"{type(e).__name__}: {str(e)[:300]}"
        torch.cuda.empty_cache()
        print("ROW", json.dumps(row), flush = True)
        rows.append(row)

res = {
    "model": a.model, "load_in_4bit": not a.no_4bit, "dtype": a.dtype, "new_tokens": a.new_tokens,
    "load_s": round(load_s, 1), "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
    "attn_forward_module": attn_module, "unsloth_compile_disable": os.environ.get("UNSLOTH_COMPILE_DISABLE"),
    "free_mib_start": None, "rows": rows, "gen_calls": (gen_res or {}).get("calls"),
}
try:
    import unsloth_zoo
    res["zoo_file"] = unsloth_zoo.__file__
except Exception:  # noqa: BLE001
    pass
# Round 7f decode levers (#2233 fast launch, #2236 rope_decode, #2246 bnb direct): gates and call counts, None if absent.
eng = {}
try:
    import unsloth_zoo.triton_fast_launch as _tfl
    eng["fast_launch_enabled"], eng["fast_launch_stats"] = bool(_tfl._ENABLED), dict(_tfl.FAST_LAUNCH_STATS)
except Exception as e:  # noqa: BLE001
    eng["fast_launch"] = f"absent: {type(e).__name__}"
try:
    import unsloth_zoo.gpt_oss_decode_fused as _gdf
    eng["fused_decode_enabled"] = bool(_gdf._ENABLED)
    eng["rope_enabled"] = bool(getattr(_gdf, "_ROPE_ENABLED", False)) if hasattr(_gdf, "_ROPE_ENABLED") else None
    eng["rope_certified"] = len(getattr(_gdf, "_ROPE_CHECKED", {}) or {}) if hasattr(_gdf, "_ROPE_CHECKED") else None
except Exception as e:  # noqa: BLE001
    eng["fused_decode"] = f"absent: {type(e).__name__}"
try:
    import unsloth_zoo.temporary_patches.bitsandbytes as _zb
    eng["bnb_direct_on"] = bool(_zb._BNB_DIRECT["on"]) if hasattr(_zb, "_BNB_DIRECT") else None
    eng["bnb_direct_calls"] = dict(_zb.BNB_4BIT_DIRECT_CALLS) if hasattr(_zb, "BNB_4BIT_DIRECT_CALLS") else None
except Exception as e:  # noqa: BLE001
    eng["bnb_direct"] = f"absent: {type(e).__name__}"
res["engagement"] = eng
with open(a.out, "w") as f:
    json.dump(res, f, indent = 1)
print("DONE", a.out)
