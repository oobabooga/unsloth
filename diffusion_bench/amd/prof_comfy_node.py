"""Baseline custom node file (copied into its custom_nodes/ by the job): wraps the denoiser call
(comfy.model_base.BaseModel.apply_model), text encoding (comfy.sd.CLIP.encode_from_tokens) and VAE decode
(comfy.sd.VAE.decode) with prof_hook when PROF_OUT is set. Registers no nodes."""

import os
import sys

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

if os.environ.get("PROF_OUT") and os.environ.get("PROF_HOOK_DIR"):
    sys.path.insert(0, os.environ["PROF_HOOK_DIR"])
    import prof_hook
    import comfy.model_base as mb
    import comfy.sd as sd

    for owner, attr, label in ((mb.BaseModel, "apply_model", "base_denoiser"), (sd.CLIP, "encode_from_tokens", "base_te"),
                               (sd.VAE, "decode", "base_vae")):
        fn = getattr(owner, attr)
        if not getattr(fn, "__prof_hook__", False):
            setattr(owner, attr, prof_hook.wrap(fn, label))
            print(f"prof_hook: wrapped {owner.__name__}.{attr} as {label}", flush = True)
