"""Sanity (not a benchmark): eager vs flex GPT-OSS inference on a small random model (bf16 where supported), 3 alternating rounds, plus training memory scaling. Run by gptoss_flex_probe.py."""
import os, sys, time
os.environ.setdefault("UNSLOTH_MODEL_NAME", "unsloth/gpt-oss-test")
import torch
from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig
import transformers.models.gpt_oss.modeling_gpt_oss as M
dtype = torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation = False) else torch.float32
torch.manual_seed(0)
L = 4
c = GptOssConfig(hidden_size=512, intermediate_size=512, num_hidden_layers=L, num_attention_heads=64,
                 num_key_value_heads=8, head_dim=64, num_local_experts=4, num_experts_per_tok=2, vocab_size=1024,
                 max_position_embeddings=8192, sliding_window=128, layer_types=["sliding_attention", "full_attention"] * (L // 2))
c._attn_implementation = "eager"
model = M.GptOssForCausalLM(c).to("cuda", dtype).eval()
from unsloth_zoo.temporary_patches import gpt_oss as G
G.patch_GptOssAttention(); G.patch_GptOssModel()
print("dtype", dtype, torch.cuda.get_device_name(), flush=True)
def peak_of(fn):
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); base = torch.cuda.memory_allocated()
    t = time.perf_counter(); fn(); torch.cuda.synchronize()
    return time.perf_counter() - t, (torch.cuda.max_memory_allocated() - base) / 2**20
B = 4
ids = torch.randint(1, 1024, (B, 4096), device="cuda"); pad = torch.ones_like(ids)
pad[1, :100] = 0; pad[3, :700] = 0
for mode in ("eager", "flex") * 3:
    os.environ["UNSLOTH_GPT_OSS_EAGER_INFERENCE"] = "1" if mode == "eager" else "0"
    r = {}
    with torch.no_grad():
        model.generate(ids[:, :256], attention_mask=pad[:, :256], max_new_tokens=8, do_sample=False, pad_token_id=0)
        for T in (1024, 4096):
            f = lambda: model(input_ids=ids[:, :T], attention_mask=pad[:, :T], use_cache=False, logits_to_keep=1)
            f()
            s, p = peak_of(f)
            r[f"prefill_{T}_s"] = round(s, 4); r[f"prefill_{T}_peak_mib"] = round(p, 1)
        for ctx in (512, 2048):
            times = []
            for n in (16, 144):
                s, _ = peak_of(lambda: model.generate(ids[:, :ctx], attention_mask=pad[:, :ctx], max_new_tokens=n,
                                                      min_new_tokens=n, do_sample=False, pad_token_id=0))
                times.append(s)
            r[f"decode_ms_per_token_ctx{ctx}"] = round((times[1] - times[0]) / 128 * 1e3, 2)
    print(mode, r, flush=True)
# Training path (flex_attention_with_sink): peak should grow ~linearly with T, not with T^2.
model.train()
r = {}
for T in (1024, 4096):
    x = ids[:1, :T]
    def step():
        model(input_ids=x, labels=x).loss.backward(); model.zero_grad(set_to_none=True)
    step()
    s, p = peak_of(step)
    r[f"train_{T}_peak_mib"] = round(p, 1)
print("train", r, flush=True)
