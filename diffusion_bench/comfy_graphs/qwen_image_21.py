"""Qwen-Image-2.1 text to image: Comfy-Org's image_qwen_image_2_1_t2i template (the graph qwen21_comfy_legb.py ran).

UNETLoader -> KSampler(euler / simple, CFG 1, 25 steps) <- TextEncodeQwenImage21(CLIPLoader type qwen_image);
EmptyLatentImage; VAEDecode(VAELoader). Files as Comfy-Org/Qwen-Image-2.1 ships them:
  dit  qwen_image_2.1_bf16.safetensors | qwen_image_2.1_int8_convrot.safetensors
  te   qwen3vl_8b_bf16.safetensors | qwen3vl_8b_int8_convrot.safetensors | qwen3vl_8b_w4a8.safetensors
  vae  qwen_image_2.1_vae_bf16.safetensors
"""

from . import SAVE, GraphParams, pick, save_node, wrap_model

TEMPLATE = "image_qwen_image_2_1_t2i.json"
DEFAULTS = {"sampler": "euler", "scheduler": "simple", "cfg": 1.0, "steps": 25}
FILES = {"dit": "diffusion_models", "te": "text_encoders", "vae": "vae"}


def build(p: GraphParams) -> dict:
    import sys

    mod = sys.modules[__name__]
    g = {
        "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": p.files["dit"], "weight_dtype": p.weight_dtype}},
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": p.files["te"], "type": "qwen_image",
                                                        "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": p.files["vae"]}},
        "cond": {"class_type": "TextEncodeQwenImage21",
                 "inputs": {"clip": ["clip", 0], "prompt": p.prompt, "negative_prompt": p.negative or "",
                            "resolution": int(p.extra.get("te_resolution") or max(p.width, p.height))}},
        "latent": {"class_type": "EmptyLatentImage", "inputs": {"width": p.width, "height": p.height, "batch_size": 1}},
    }
    model = wrap_model(g, ["unet", 0], p)
    g["sampler"] = {"class_type": "KSampler",
                    "inputs": {"model": model, "seed": int(p.seed), "steps": int(p.steps), "cfg": float(pick(p, mod, "cfg")),
                               "sampler_name": pick(p, mod, "sampler"), "scheduler": pick(p, mod, "scheduler"),
                               "positive": ["cond", 0], "negative": ["cond", 1], "latent_image": ["latent", 0],
                               "denoise": 1.0}}
    g["decode"] = {"class_type": "VAEDecode", "inputs": {"samples": ["sampler", 0], "vae": ["vae", 0]}}
    g[SAVE] = save_node(["decode", 0], p.prefix)
    return g
