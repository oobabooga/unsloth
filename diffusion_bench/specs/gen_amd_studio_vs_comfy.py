#!/usr/bin/env python3
"""Generate specs/amd_studio_vs_comfy.json: Unsloth Studio vs ComfyUI on gfx1151, per family, same prompt / seed /
steps / size at ComfyUI's template operating point (outputs/comfy_defaults_audit.md; templates read from
comfyui_workflow_templates_json at the pinned ComfyUI commit).

Arms per family (tag = <fam>_<arm>):
  s_def     Studio, its defaults (attention auto -> diffusers native -> torch SDPA; AOTriton on ROCm)
  s_flash   Studio with attention_backend="flash" (DAO-AILab flash-attention, Triton AMD backend). On main the
            ROCm gate drops the request (falls back to native); on the patched head it engages.
  c_def     ComfyUI, README ROCm install (download.pytorch.org/whl/rocm7.2 torch), default flags
  c_rock    ComfyUI, AMD gfx1151 torch (repo.amd.com/rocm/whl/gfx1151, the build Studio installs), default flags
  c_flash   c_rock + --use-flash-attention (same DAO-AILab Triton AMD build) + --enable-dynamic-vram
  c_kitchen c_rock + --use-ck-attention (Comfy Kitchen INT8 attention, quantized) + --enable-dynamic-vram

Every ComfyUI file is public (token=False); each family's Studio repo is public too (FLUX.1-schnell via the
unsloth mirror, since black-forest-labs/FLUX.1-schnell is gated).
"""

from __future__ import annotations

import json
from pathlib import Path

W = "$AMD_CI_WORK"
COMFY_DIR = f"{W}/ComfyUI"
PY = {"def": f"{W}/comfy_venv_default/bin/python", "rock": f"{W}/comfy_venv_rock/bin/python"}
SDXL_NEG = ("color, colored, lowres, blurry, out of focus, deformed, bad anatomy, extra limbs, mutated, watermark, "
            "text, logo, signature\n")

MODELS = {
    "zimage_turbo": {"repo": "Tongyi-MAI/Z-Image-Turbo", "download": {"ignore": ["*.png", "*.jpg", "*.webp", "assets/*"]}},
    "comfy_zimage": {"repo": "Comfy-Org/z_image_turbo", "download": {"allow": [
        "split_files/diffusion_models/z_image_turbo_bf16.safetensors",
        "split_files/text_encoders/qwen_3_4b.safetensors", "split_files/vae/ae.safetensors"]}},
    "flux2_klein_4b": {"repo": "black-forest-labs/FLUX.2-klein-4B", "download": {"ignore": ["*.png", "*.jpg"]}},
    "comfy_klein": {"repo": "Comfy-Org/vae-text-encorder-for-flux-klein-4b",  # canonical id of Comfy-Org/flux2-klein-4B (renamed; the runner mirror 404s on the redirect)
                     "download": {"allow": [
        "split_files/diffusion_models/flux-2-klein-4b.safetensors",
        "split_files/text_encoders/qwen_3_4b.safetensors", "split_files/vae/flux2-vae.safetensors"]}},
    # One download serves both sides: diffusers folders for Studio, the single-file checkpoint for ComfyUI.
    "sdxl_base": {"repo": "stabilityai/stable-diffusion-xl-base-1.0", "download": {"ignore": [
        "*.bin", "*.onnx", "*.onnx_data", "*.msgpack", "*.ckpt", "*openvino*", "sd_xl_base_1.0_0.9vae.safetensors",
        "*.png", "*.jpg"]}},
    "flux1_schnell": {"repo": "unsloth/FLUX.1-schnell", "download": {"ignore": ["*.png", "*.jpg"]}},
    "comfy_flux_te": {"repo": "comfyanonymous/flux_text_encoders",
                      "download": {"allow": ["clip_l.safetensors", "t5xxl_fp16.safetensors"]}},
    "qwen_image_21": {"repo": "Qwen/Qwen-Image-2.1", "revision": "790c92633540aa0cb11d9abf19eb46d861714758",
                      "download": {"ignore": ["*.png", "*.jpg"]}},
    "comfy_q21": {"repo": "Comfy-Org/Qwen-Image-2.1", "download": {"allow": [
        "diffusion_models/qwen_image_2.1_bf16.safetensors", "text_encoders/qwen3vl_8b_bf16.safetensors",
        "vae/qwen_image_2.1_vae_bf16.safetensors"]}},
    "wan22_5b": {"repo": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
                 "download": {"ignore": ["*.png", "*.jpg", "*.gif", "*.mp4", "assets/*", "examples/*"]}},
    "comfy_wan": {"repo": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged", "download": {"allow": [
        "split_files/diffusion_models/wan2.2_ti2v_5B_fp16.safetensors",
        "split_files/text_encoders/umt5_xxl_fp16.safetensors", "split_files/vae/wan2.2_vae.safetensors"]}},
    "h3_gguf": {"repo": "unsloth/MiniMax-H3-GGUF", "download": {"allow": [
        "minimax_h3_fl2va_pruned-Q8_0.gguf", "qwen3vl_32b_minimax_h3-Q4_K_M.gguf", "vae/*"]}},
    "comfy_h3": {"repo": "Comfy-Org/MiniMax-H3", "download": {"allow": [
        "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors",
        "text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
        "vae/minimax_h3_video_vae_int8_convrot.safetensors", "vae/minimax_h3_audio_vae_fp32.safetensors"]}},
}


def m(alias: str, rel: str) -> str:
    return f"$DBENCH_MODEL_{alias.upper()}/{rel}"


# family key -> operating point (ComfyUI template) and the two sides' inputs
FAMILIES = {
    "zimg": {"studio_model": "@zimage_turbo", "studio_family": "z-image", "comfy_family": "z-image",
             "fetch": "@comfy_zimage", "width": 1024, "height": 1024, "steps": 8, "short_steps": 3, "n": 2,
             "template": "image_z_image_turbo: 8 steps, CFG 1, res_multistep / simple, AuraFlow shift 3",
             "files": {"dit": m("comfy_zimage", "split_files/diffusion_models/z_image_turbo_bf16.safetensors"),
                       "te": m("comfy_zimage", "split_files/text_encoders/qwen_3_4b.safetensors"),
                       "vae": m("comfy_zimage", "split_files/vae/ae.safetensors")}},
    "klein": {"studio_model": "@flux2_klein_4b", "studio_family": "flux.2-klein", "comfy_family": "flux.2-klein",
              "fetch": "@comfy_klein", "width": 1024, "height": 1024, "steps": 4, "short_steps": 2, "n": 2,
              "template": "image_flux2_klein_text_to_image (distilled): 4 steps, CFG 1, euler, Flux2Scheduler",
              "files": {"dit": m("comfy_klein", "split_files/diffusion_models/flux-2-klein-4b.safetensors"),
                        "te": m("comfy_klein", "split_files/text_encoders/qwen_3_4b.safetensors"),
                        "vae": m("comfy_klein", "split_files/vae/flux2-vae.safetensors")}},
    "sdxl": {"studio_model": "@sdxl_base", "studio_family": "sdxl", "comfy_family": "sdxl", "fetch": None,
             "width": 1024, "height": 1024, "steps": 25, "short_steps": 5, "n": 2, "guidance": 7.0,
             "negative_prompt": SDXL_NEG,
             "template": "image_sdxl_simple: 25 steps, CFG 7, dpmpp_2m / karras (Studio samples Euler: known LIST item)",
             "files": {"checkpoint": m("sdxl_base", "sd_xl_base_1.0.safetensors")}},
    "flux1": {"studio_model": "@flux1_schnell", "studio_family": "flux.1", "comfy_family": "flux.1",
              "fetch": "@comfy_flux_te", "width": 1024, "height": 1024, "steps": 4, "short_steps": 2, "n": 2,
              "template": "flux_schnell: 4 steps, CFG 1, euler / simple (template ships an fp8 checkpoint; bf16 split "
                          "files here so both sides run the same weights)",
              "files": {"dit": m("flux1_schnell", "flux1-schnell.safetensors"),
                        "te": [m("comfy_flux_te", "clip_l.safetensors"), m("comfy_flux_te", "t5xxl_fp16.safetensors")],
                        "vae": m("flux1_schnell", "ae.safetensors")}},
    "q21": {"studio_model": "@qwen_image_21", "studio_family": "qwen-image-2.1", "comfy_family": "qwen-image-2.1",
            "fetch": "@comfy_q21", "width": 1024, "height": 1024, "steps": 25, "short_steps": 5, "n": 1,
            "guidance": 1.0, "template": "image_qwen_image_2_1_t2i: 25 steps, CFG 1, euler / simple",
            "files": {"dit": m("comfy_q21", "diffusion_models/qwen_image_2.1_bf16.safetensors"),
                      "te": m("comfy_q21", "text_encoders/qwen3vl_8b_bf16.safetensors"),
                      "vae": m("comfy_q21", "vae/qwen_image_2.1_vae_bf16.safetensors")}},
    "wan": {"studio_model": "@wan22_5b", "studio_family": "wan2.2-ti2v-5b", "comfy_family": "wan2.2-5b",
            "fetch": "@comfy_wan", "kind": "video", "prompts": "video_3", "width": 1280, "height": 704, "frames": 21,
            "fps": 24, "steps": 20, "short_steps": 4, "n": 1, "guidance": 5.0,
            "template": "video_wan2_2_5B_ti2v: 1280x704, 20 steps, CFG 5, uni_pc / simple, shift 8 (121 frames in "
                        "the template; 21 here: a short clip)",
            "files": {"dit": m("comfy_wan", "split_files/diffusion_models/wan2.2_ti2v_5B_fp16.safetensors"),
                      "te": m("comfy_wan", "split_files/text_encoders/umt5_xxl_fp16.safetensors"),
                      "vae": m("comfy_wan", "split_files/vae/wan2.2_vae.safetensors")}},
    "h3": {"studio_model": "@h3_gguf", "studio_family": "minimax-h3", "comfy_family": "minimax-h3",
           "fetch": "@comfy_h3", "kind": "video", "prompts": "video_3", "width": 832, "height": 480, "frames": 39,
           "fps": 24, "steps": 20, "short_steps": 4, "n": 1,
           "template": "video_minimax_h3_t2v: 20 steps res_multistep, no CFG; 0.4 MP 39 frames here",
           "studio_options": {"model_kind": "gguf", "gguf_filename": "minimax_h3_fl2va_pruned-Q8_0.gguf",
                              "base_repo": "Comfy-Org/MiniMax-H3", "local_files_only": False},
           "files": {"dit": m("comfy_h3", "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors"),
                     "te": m("comfy_h3", "text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors"),
                     "vae": m("comfy_h3", "vae/minimax_h3_video_vae_int8_convrot.safetensors"),
                     "audio_vae": m("comfy_h3", "vae/minimax_h3_audio_vae_fp32.safetensors")}},
}


def common(f: dict) -> dict:
    c = {k: f[k] for k in ("width", "height", "steps", "short_steps", "n") if k in f}
    for k in ("kind", "prompts", "frames", "fps", "guidance", "negative_prompt"):
        if k in f:
            c[k] = f[k]
    c["note"] = f["template"]
    return c


def cells() -> list:
    out = []
    for fam, f in FAMILIES.items():
        sopts = {"family_override": f["studio_family"], "model_kind": "pipeline", **(f.get("studio_options") or {})}
        base = common(f)
        # Studio's server sets TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 at the top of studio/backend/main.py (and
        # unsloth/__init__.py does too); the in-process backend imports neither, so set it here as the product does.
        # Without it torch's SDPA on gfx1151 refuses flash / efficient and runs math (run 37130594778, job 1).
        senv = {"DBENCH_ATTN_PROFILE": "1", "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL": "1"}
        out.append({"tag": f"{fam}_s_def", "backend": "studio", "model": f["studio_model"], "ref": f"{fam}_s_def",
                    **base, "options": sopts, "env": dict(senv)})
        out.append({"tag": f"{fam}_s_flash", "backend": "studio", "model": f["studio_model"], "ref": f"{fam}_s_def",
                    **base, "options": {**sopts, "attention_backend": "flash"},
                    "env": {**senv, "FLASH_ATTENTION_TRITON_AMD_ENABLE": "TRUE"}})
        arms = {"c_def": ("def", None, {}), "c_rock": ("rock", None, {}),
                "c_flash": ("rock", "--use-flash-attention --enable-dynamic-vram",
                            {"FLASH_ATTENTION_TRITON_AMD_ENABLE": "TRUE"}),
                "c_kitchen": ("rock", "--use-ck-attention --enable-dynamic-vram", {})}
        for arm, (venv, flags, env) in arms.items():
            opts = {"family": f["comfy_family"], "files": f["files"]}
            if flags:
                opts["comfy_args"] = flags
            cell = {"tag": f"{fam}_{arm}", "backend": "comfyui", "venv": None, "ref": f"{fam}_c_rock", **base,
                    "options": opts,
                    "env": {"DIFFUSION_BENCH_COMFY_PYTHON": PY[venv], "DIFFUSION_BENCH_COMFY_DIR": COMFY_DIR, **env}}
            if f.get("fetch"):
                cell["model"] = f["fetch"]  # never read by the comfyui backend; makes the probe fetch the files
            elif fam == "sdxl":
                cell["model"] = "@sdxl_base"
            out.append(cell)
    return out


def main() -> None:
    spec = {
        "name": "amd_studio_vs_comfy",
        "description": __doc__,
        "models": MODELS,
        "defaults": {"backend": "studio", "kind": "image", "prompts": "image_24", "n": 2, "warmup": 1, "short_n": 1,
                     "options": {"local_files_only": True}},
        "gate": {"max_util": 15, "min_free_mib": 0, "timeout_s": 600},
        "cells": cells(),
    }
    out = Path(__file__).with_name("amd_studio_vs_comfy.json")
    out.write_text(json.dumps(spec, indent = 1) + "\n", encoding = "utf-8")
    print(f"wrote {out} ({len(spec['cells'])} cells)")


if __name__ == "__main__":
    main()
