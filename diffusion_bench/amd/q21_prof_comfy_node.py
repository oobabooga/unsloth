"""ComfyUI custom node file (copied into ComfyUI/custom_nodes/ by the profile job): wraps the Qwen-Image-2.1
diffusion model's _forward with q21_prof_hook.wrap when Q21_PROF_OUT is set. Registers no nodes."""

import os
import sys

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

if os.environ.get("Q21_PROF_OUT") and os.environ.get("Q21_PROF_HOOK_DIR"):
    sys.path.insert(0, os.environ["Q21_PROF_HOOK_DIR"])
    import q21_prof_hook
    import comfy.ldm.qwen_image21.model as m

    if not getattr(m.QwenImage21Transformer2DModel._forward, "__q21_prof__", False):
        m.QwenImage21Transformer2DModel._forward = q21_prof_hook.wrap(m.QwenImage21Transformer2DModel._forward, "comfy")
        print("q21_prof: wrapped comfy QwenImage21Transformer2DModel._forward", flush = True)
