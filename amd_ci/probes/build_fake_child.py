"""Fake Qwen3.5-4B text model: real 4B layer dims, 4 layers (3 gated-delta + 1 full attention), random weights."""
import json, os, sys
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

out = sys.argv[1]
tok = AutoTokenizer.from_pretrained("unsloth/Qwen3.5-4B")
cfg = AutoConfig.for_model(
    "qwen3_5_text", vocab_size = 248320, hidden_size = 2560, intermediate_size = 9216,
    num_hidden_layers = 4, layer_types = ["linear_attention"] * 3 + ["full_attention"],
    full_attention_interval = 4, num_attention_heads = 16, num_key_value_heads = 4, head_dim = 256,
    attn_output_gate = True, linear_conv_kernel_dim = 4, linear_key_head_dim = 128, linear_num_key_heads = 16,
    linear_num_value_heads = 32, linear_value_head_dim = 128, rms_norm_eps = 1e-6, tie_word_embeddings = True,
    max_position_embeddings = 262144, eos_token_id = tok.eos_token_id,
    rope_parameters = {"rope_type": "default", "rope_theta": 10000000, "partial_rotary_factor": 0.25,
                       "mrope_section": [11, 11, 10], "mrope_interleaved": True},
    torch_dtype = "bfloat16",
)
torch.manual_seed(3407)
model = AutoModelForCausalLM.from_config(cfg, torch_dtype = torch.bfloat16)
model.save_pretrained(out, safe_serialization = True)
tok.save_pretrained(out)
print("BUILT", type(model).__name__, sum(p.numel() for p in model.parameters()), flush = True)
