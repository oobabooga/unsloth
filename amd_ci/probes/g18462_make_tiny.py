# Tiny random Gemma 4 (26B-A4B layout) keeping the real attention head dims (256 sliding, 512 global).
import json, os, shutil, sys, torch
from transformers import AutoConfig, AutoModelForImageTextToText
src, out = sys.argv[1], sys.argv[2]
moe = "--dense" not in sys.argv
window = int(os.environ.get("G4A_WINDOW", "1024"))
cfg = json.load(open(os.path.join(src, "config.json")))
t = cfg["text_config"]
t.update(hidden_size=256, num_attention_heads=4, num_key_value_heads=2, num_global_key_value_heads=1,
         num_hidden_layers=2, layer_types=["sliding_attention", "full_attention"], intermediate_size=512,
         moe_intermediate_size=128, num_experts=4, top_k_experts=2, enable_moe_block=moe, sliding_window=window)
v = cfg["vision_config"]
v.update(hidden_size=64, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2, head_dim=32,
         global_head_dim=32, intermediate_size=128)
os.makedirs(out, exist_ok=True)
json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=2)
for f in os.listdir(src):
    if f != "config.json" and os.path.isfile(os.path.join(src, f)):
        shutil.copy(os.path.join(src, f), out)
config = AutoConfig.from_pretrained(out)
torch.manual_seed(0)
model = AutoModelForImageTextToText.from_config(config, dtype=torch.bfloat16)
model.save_pretrained(out)
print(type(model).__name__, sum(p.numel() for p in model.parameters()) / 1e6, "M params")
