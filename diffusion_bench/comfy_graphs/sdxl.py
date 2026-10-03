"""SDXL base 1.0 text to image: Comfy-Org's image_sdxl_simple template. CheckpointLoaderSimple
(sd_xl_base_1.0.safetensors) -> CLIPTextEncode pos / neg -> KSampler(25 steps, CFG 7, dpmpp_2m / karras, denoise 1)
on EmptyLatentImage 1024 -> VAEDecode. The template's negative prompt is the default when the cell gives none.
Files: checkpoint sd_xl_base_1.0.safetensors (stabilityai/stable-diffusion-xl-base-1.0, public).
"""

from . import SAVE, GraphParams, pick, save_node, wrap_model

TEMPLATE = "image_sdxl_simple.json"
DEFAULTS = {"sampler": "dpmpp_2m", "scheduler": "karras", "cfg": 7.0, "steps": 25}
FILES = {"checkpoint": "checkpoints"}
DEFAULT_NEGATIVE = ("color, colored, lowres, blurry, out of focus, deformed, bad anatomy, extra limbs, mutated, "
                    "watermark, text, logo, signature\n")


def build(p: GraphParams) -> dict:
    import sys

    mod = sys.modules[__name__]
    g: dict = {"ckpt": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": p.files["checkpoint"]}}}
    g["pos"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["ckpt", 1], "text": p.prompt}}
    g["neg"] = {"class_type": "CLIPTextEncode",
                "inputs": {"clip": ["ckpt", 1], "text": DEFAULT_NEGATIVE if p.negative is None else p.negative}}
    g["latent"] = {"class_type": "EmptyLatentImage", "inputs": {"width": p.width, "height": p.height, "batch_size": 1}}
    model = wrap_model(g, ["ckpt", 0], p)
    g["sampler"] = {"class_type": "KSampler",
                    "inputs": {"model": model, "seed": int(p.seed), "steps": int(p.steps), "cfg": float(pick(p, mod, "cfg")),
                               "sampler_name": pick(p, mod, "sampler"), "scheduler": pick(p, mod, "scheduler"),
                               "positive": ["pos", 0], "negative": ["neg", 0], "latent_image": ["latent", 0],
                               "denoise": 1.0}}
    g["decode"] = {"class_type": "VAEDecode", "inputs": {"samples": ["sampler", 0], "vae": ["ckpt", 2]}}
    g[SAVE] = save_node(["decode", 0], p.prefix)
    return g
