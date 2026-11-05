#!/usr/bin/env python3
"""Generate specs/amd_nov5.json: Unsloth Studio (main) vs ComfyUI v0.44.0 on gfx1151 (Strix Halo, Linux ROCm), bf16.

Every ComfyUI file is built on the runner from the OFFICIAL model repos by amd/nov5_prep.py (klein, f3, zimg, q21)
and convert_comfy.py (Wan2.2-TI2V-5B); nothing comes from a Comfy-Org / comfyanonymous Hugging Face repo. Studio
reads the same official snapshots (DBENCH_MODEL_<ALIAS> set by the prep step), so both stacks run the same weights.

Arms per family (tag = <fam>_<arm>):
  s_def     Studio main defaults, its own gfx1151 torch (repo.amd.com 2.11 + ROCm 7.13); no flash_attn importable
            (flash-attention is installed with pip --target into a separate dir that only the flash arms put on
            sys.path, via DBENCH_EXTRA_SYSPATH in run_cell.py)
  s_flash   s_def + DAO-AILab flash-attention (Triton AMD backend) importable: attention auto (what shipping it gives)
  s_flashx  s_flash with attention_backend="flash" (explicit)
  s_pt72n   Studio main in a venv on download.pytorch.org torch 2.14.1+rocm7.2 (ComfyUI README torch)
  c_rock    ComfyUI v0.44.0 on Studio's torch (same torch as s_def), default flags
  c_def     ComfyUI v0.44.0, README ROCm install (download.pytorch.org rocm7.2 -> 2.14.1), default flags
  c_nockhip c_rock + COMFY_KITCHEN_DISABLE_HIP=1 (comfy-kitchen fused HIP kernels off: isolates their share)
  s_pt72n_flash  s_pt72n + the flash-attention dir (both Studio levers stacked)
  *_2_repeat     A/A copies of s_def, s_flash and c_rock (noise floor)
"""

from __future__ import annotations

import json
from pathlib import Path

W = "$AMD_CI_WORK"
CM = f"{W}/cm"
COMFY_DIR = f"{W}/ComfyUI"
PY = {"def": f"{W}/comfy_venv_default/bin/python", "rock": f"{W}/comfy_venv_rock/bin/python"}
PT72N_PY = f"{W}/studio_pt72n/bin/python"
FA_SITE = f"{W}/fa_site"

MODELS = {
    "flux2_klein_4b": {"repo": "black-forest-labs/FLUX.2-klein-4B", "download": {"ignore": ["*.png", "*.jpg"]}},
    "flux3_klein": {"repo": "unsloth/flux-3-image-klein",
                    "download": {"ignore": ["*.png", "*.jpg", "variants/*"]}},
    "zimage_turbo": {"repo": "Tongyi-MAI/Z-Image-Turbo", "download": {"ignore": ["*.png", "*.jpg", "*.webp", "assets/*"]}},
    "qwen_image_21": {"repo": "Qwen/Qwen-Image-2.1", "download": {"ignore": ["*.png", "*.jpg"]}},
    "wan22_5b": {"repo": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
                 "download": {"ignore": ["*.png", "*.jpg", "*.gif", "*.mp4", "assets/*", "examples/*"]}},
}

FAMILIES = {
    "klein": {"studio_model": "@flux2_klein_4b", "studio_family": "flux.2-klein", "comfy_family": "flux2-klein",
              "width": 1024, "height": 1024, "steps": 4, "short_steps": 2, "n": 2,
              "template": "image_flux2_klein_text_to_image (distilled): 4 steps, CFG 1, euler",
              "files": {"dit": f"{CM}/flux-2-klein-4b.safetensors", "te": f"{CM}/qwen3_4b_klein_merged.safetensors",
                        "vae": f"{CM}/flux2_klein_vae_diffusers.safetensors"}},
    "f3": {"studio_model": "@flux3_klein", "studio_family": "flux.3-klein", "comfy_family": "flux3-klein",
           "width": 1024, "height": 1024, "steps": 16, "short_steps": 4, "n": 2, "guidance": 1.0,
           "template": "image_flux3_klein_distilled_t2i: 16 steps, CFG 1, er_sde / simple",
           "files": {"dit": f"{CM}/flux-3-image-klein.safetensors", "te": f"{CM}/qwen3vl_4b_flux3_te_bf16.safetensors",
                     "vae": f"{CM}/flux3_ae.safetensors"}},
    "zimg": {"studio_model": "@zimage_turbo", "studio_family": "z-image", "comfy_family": "z-image",
             "width": 1024, "height": 1024, "steps": 8, "short_steps": 3, "n": 2, "guidance": 0.0, "comfy_cfg": 1.0,
             "template": "image_z_image_turbo: 8 steps, CFG 1, res_multistep / simple, shift 3",
             "files": {"dit": f"{CM}/z_image_turbo_official_dit_bf16.safetensors",
                       "te": f"{CM}/qwen3_4b_zimage_official_merged.safetensors", "vae": f"{CM}/ae_zimage.safetensors"}},
    "q21": {"studio_model": "@qwen_image_21", "studio_family": "qwen-image-2.1", "comfy_family": "qwen-image-2.1",
            "width": 1024, "height": 1024, "steps": 25, "short_steps": 5, "n": 1, "guidance": 1.0,
            "template": "image_qwen_image_2_1_t2i: 25 steps, CFG 1, euler / simple",
            "files": {"dit": f"{CM}/qwen_image_2.1_official_bf16.safetensors",
                      "te": f"{CM}/qwen3vl_8b_q21_official_bf16.safetensors",
                      "vae": f"{CM}/qwen_image_2.1_vae_official_bf16.safetensors"}},
    "wan": {"studio_model": "@wan22_5b", "studio_family": "wan2.2-ti2v-5b", "comfy_family": "wan2.2-5b",
            "kind": "video", "prompts": "video_3", "width": 1280, "height": 704, "frames": 21, "fps": 24,
            "steps": 20, "n": 1, "guidance": 5.0,
            "template": "video_wan2_2_5B_ti2v: 1280x704, 20 steps, CFG 5, uni_pc / simple, shift 8 (21 frames)",
            "files": {"dit": f"{CM}/wan2.2_ti2v_5B_bf16_official.safetensors",
                      "te": f"{CM}/umt5_xxl_bf16_official.safetensors", "vae": f"{CM}/wan2.2_vae_official.safetensors"}},
}


def common(f: dict) -> dict:
    c = {k: f[k] for k in ("width", "height", "steps", "short_steps", "n") if k in f}
    for k in ("kind", "prompts", "frames", "fps", "guidance", "negative_prompt"):
        if k in f:
            c[k] = f[k]
    # images: one short render pair gives s/step = (long - short) / step difference; video: the backend's step timing
    c.update(short_n = 0 if f.get("kind") == "video" else 1, warmup = 1)
    c["note"] = f["template"]
    return c


def cells() -> list:
    out = []
    for fam, f in FAMILIES.items():
        base = common(f)
        sopts = {"family_override": f["studio_family"], "model_kind": "pipeline"}
        # The server sets this at import (studio/backend/main.py); the in-process backend imports neither.
        senv = {"DBENCH_ATTN_PROFILE": "1", "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL": "1"}
        fa_env = {**senv, "DBENCH_EXTRA_SYSPATH": FA_SITE, "FLASH_ATTENTION_TRITON_AMD_ENABLE": "TRUE"}

        def s(tag, ref, opts = None, env = None, **kw):
            out.append({"tag": f"{fam}_{tag}", "backend": "studio", "model": f["studio_model"], "ref": f"{fam}_{ref}",
                        **base, "options": {**sopts, **(opts or {})}, "env": dict(env or senv), **kw})

        def c(tag, venv, env = None):
            opts = {"family": f["comfy_family"], "files": f["files"]}
            if "comfy_cfg" in f:
                opts["cfg"] = f["comfy_cfg"]
            out.append({"tag": f"{fam}_{tag}", "backend": "comfyui", "venv": None, "ref": f"{fam}_c_rock", **base,
                        "options": opts, "env": {"DIFFUSION_BENCH_COMFY_PYTHON": PY[venv],
                                                 "DIFFUSION_BENCH_COMFY_DIR": COMFY_DIR, **(env or {})}})

        # Interleaved: s, c, s, c ... so slow drift (thermals, page cache) does not land on one stack.
        s("s_def", "s_def")
        c("c_rock", "rock")
        s("s_flash", "s_def", env = fa_env)
        c("c_def", "def")
        s("s_pt72n", "s_def", venv = PT72N_PY)
        c("c_nockhip", "rock", env = {"COMFY_KITCHEN_DISABLE_HIP": "1"})
        s("s_flashx", "s_def", opts = {"attention_backend": "flash"}, env = fa_env)
        s("s_def_2_repeat", "s_def")
        c("c_rock_2_repeat", "rock")
        # follow-ups (nov5 m4): the two Studio levers stacked, and an A/A copy of the flash arm
        s("s_pt72n_flash", "s_def", env = fa_env, venv = PT72N_PY)
        s("s_flash_2_repeat", "s_flash", env = fa_env)
        # native int8 on ROCm (torchao is refused there): weight-only, and W8A8 (int8 activations through _int_mm),
        # whose compiled fused linear main refuses on ROCm (studio/rocm-fused-w8a8 lifts it on RDNA3 / RDNA4)
        s("s_int8", "s_def", opts = {"transformer_quant": "int8"})
        s("s_int8act", "s_def", opts = {"transformer_quant": "int8"}, env = {**senv, "UNSLOTH_NATIVE_INT8_ACT": "1"})
    return out


def main() -> None:
    spec = {
        "name": "amd_nov5",
        "description": __doc__,
        "models": MODELS,
        "defaults": {"backend": "studio", "kind": "image", "prompts": "image_24", "n": 2, "warmup": 1, "short_n": 0,
                     "options": {"local_files_only": True}},
        "gate": {"max_util": 15, "min_free_mib": 0, "timeout_s": 600},
        "cells": cells(),
    }
    out = Path(__file__).with_name("amd_nov5.json")
    out.write_text(json.dumps(spec, indent = 1) + "\n", encoding = "utf-8")
    print(f"wrote {out} ({len(spec['cells'])} cells)")


if __name__ == "__main__":
    main()
