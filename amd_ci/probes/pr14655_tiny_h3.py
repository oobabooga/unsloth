"""Build a tiny real diffusers MiniMaxH3Transformer3DModel the way from_pretrained(torch_dtype=bf16) leaves it."""
import torch


def build():
    from diffusers import MiniMaxH3Transformer3DModel as M
    m = M(num_attention_heads=2, attention_head_dim=16, hidden_size=32, num_layers=1, num_refiner_layers=1,
          ffn_dim=64, in_channels=4, audio_in_channels=4, text_dim=16, freq_dim=16, time_embed_hidden_dim=32,
          time_embed_dim=16, rope_freq_dim=4).to(torch.bfloat16)
    keep = tuple(M._keep_in_fp32_modules or ())
    for name, mod in m.named_modules():
        if name and any(k == name.split(".")[0] or k in name.split(".") for k in keep):
            mod.to(torch.float32)
    return m


if __name__ == "__main__":
    m = build()
    print("first param", next(m.named_parameters())[0], next(m.parameters()).dtype)
    q = next(x for x in m.modules() if getattr(x, "to_q", None) is not None or getattr(x, "to_qkv", None) is not None)
    print("attn", type(q).__name__, getattr(q, "to_q", getattr(q, "to_qkv", None)).weight.dtype)
