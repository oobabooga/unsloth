#!/usr/bin/env python3
"""Generate specs/prio8_img.json: Studio (main) vs the baseline v0.44.0 on gfx1151, image priority families x formats.

Families (template settings): q21 Qwen-Image-2.1 (25 steps, CFG 1), q21t Qwen-Image-2.1-Turbo (8, the template's
sigma list), f3 FLUX.3 klein distilled (16, CFG 1), f3b FLUX.3 klein-base (50, CFG 3.5, empty negative). 1024^2.
Baseline files are built on the runner by amd/q5/build_q5.py from the official repos (FLUX.3 via the ungated
sha256-checked unsloth mirrors), GGUF from the public unsloth/*-GGUF repos (the same file Studio loads).

Arms (tag = <fam>_<arm>), each stack's LPIPS reference = its own bf16 arm:
  s_bf16 / c_bf16     bf16 DiT + bf16 text encoder (Studio: transformer_quant / text_encoder_quant none)
  c_bf16d             the baseline on its README ROCm torch (download.pytorch.org rocm7.2)
  s_int8 / c_int8     INT8: Studio W8A8 (transformer + text encoder int8); baseline int8 ConvRot DiT + int8 TE
  s_q4 / c_q4         GGUF Q4_K_M DiT (no int8 substitution on Studio), bf16 text encoder on both
  s_def               Studio defaults (Auto: what a user gets)
  s_fp8 / c_fp8       FP8 file (P2)       s_nvfp4 / c_nvfp4  NVFP4 (P2)       s_q8 / c_q8  GGUF Q8_0 (P2)
  *_r                 A/A repeat
"""

from __future__ import annotations

import json
from pathlib import Path

W = "$AMD_CI_WORK"
CM = f"{W}/cm"
PY = {"rock": f"{W}/comfy_venv_rock/bin/python", "def": f"{W}/comfy_venv_default/bin/python"}

FAM = {
    "q21": dict(model = "Qwen/Qwen-Image-2.1", sfam = "qwen-image-2.1", cfam = "qwen-image-2.1", kind = "pipeline",
                steps = 25, short_steps = 5, guidance = 1.0, gguf = ("unsloth/Qwen-Image-2.1-GGUF", "qwen-image-2.1-{q}.gguf"),
                te = "q21", prefix_cache = True, fp8f = ("unsloth/Qwen-Image-2.1-FP8", "Qwen-Image-2.1-FP8.safetensors")),
    "q21t": dict(model = "Qwen/Qwen-Image-2.1-Turbo", sfam = "qwen-image-2.1-turbo", cfam = "qwen-image-2.1-turbo",
                 kind = "pipeline", steps = 8, short_steps = 0, guidance = 1.0,
                 gguf = ("unsloth/Qwen-Image-2.1-Turbo-GGUF", "qwen-image-2.1-turbo-{q}.gguf"), te = "q21",
                 fp8f = ("unsloth/Qwen-Image-2.1-Turbo-FP8", "Qwen-Image-2.1-Turbo-FP8.safetensors")),
    "f3": dict(model = "unsloth/flux-3-image-klein", sfam = "flux.3-klein", cfam = "flux3-klein", kind = None,
               steps = 16, short_steps = 4, guidance = 1.0,
               gguf = ("unsloth/flux-3-image-klein-GGUF", "flux-3-image-klein-{q}.gguf"), te = "f3",
               fp8f = ("unsloth/flux-3-image-klein-FP8", "flux-3-image-klein-FP8.safetensors"),
               nvfp4f = ("unsloth/flux-3-image-klein", "variants/nvfp4/model.safetensors")),
    "f3b": dict(model = "unsloth/flux-3-image-klein-base", sfam = "flux.3-klein", cfam = "flux3-klein", kind = None,
                steps = 50, short_steps = 10, guidance = 3.5, negative = "",
                gguf = ("unsloth/flux-3-image-klein-base-GGUF", "flux-3-image-klein-base-{q}.gguf"), te = "f3",
                fp8f = ("unsloth/flux-3-image-klein-base-FP8", "flux-3-image-klein-base-FP8.safetensors"),
                nvfp4f = ("unsloth/flux-3-image-klein-base", "variants/nvfp4/model.safetensors")),
}


def cells() -> list:
    out = []
    for fam, f in FAM.items():
        base = {"kind": "image", "prompts": "image_24", "width": 1024, "height": 1024, "steps": f["steps"],
                "short_steps": f["short_steps"] or f["steps"], "short_n": 1 if f["short_steps"] else 0, "n": 3,
                "warmup": 1, "guidance": f["guidance"]}
        if "negative" in f:
            base["negative_prompt"] = f["negative"]
        so = {"local_files_only": False, "load_timeout_s": 3600}
        if f["kind"]:
            so["model_kind"] = f["kind"]
        senv = {"TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL": "1"}

        def S(tag, ref = "s_bf16", model = None, env = None, **o):
            out.append({"tag": f"{fam}_{tag}", "backend": "studio", "model": model or f["model"], "ref": f"{fam}_{ref}",
                        **base, "options": {**so, **o}, "env": {**senv, **(env or {})}})

        te16, te8 = f"{CM}/{f['te']}_te_bf16.safetensors", f"{CM}/{f['te']}_te_int8_convrot.safetensors"
        vae = f"{CM}/{f['te']}_vae.safetensors"

        def Cc(tag, dit, te, venv = "rock", **o):
            opts = {"family": f["cfam"], "files": {"dit": f"{CM}/{fam}_{dit}", "te": te, "vae": vae},
                    "cache_bust": "sampler", **o}
            if f.get("prefix_cache"):  # Qwen-Image-2.1's template carries QwenImage21Cache (the turbo graph always has it)
                opts["graph_extra"] = {"prefix_cache": "auto"}
            out.append({"tag": f"{fam}_{tag}", "backend": "comfyui", "ref": f"{fam}_c_bf16", **base, "options": opts,
                        "env": {"DIFFUSION_BENCH_COMFY_PYTHON": PY[venv], "DIFFUSION_BENCH_COMFY_DIR": f"{W}/ComfyUI"}})

        dense = dict(transformer_quant = "none", text_encoder_quant = "none")
        S("s_bf16", **dense)
        Cc("c_bf16", "dit_bf16.safetensors", te16)
        S("s_int8", transformer_quant = "int8", text_encoder_quant = "int8")
        Cc("c_int8", "dit_int8_convrot.safetensors", te8)
        # INT8 DiT with the bf16 text encoder on both (Studio's int8 encoder is a torchao path, refused on ROCm)
        S("s_int8d", transformer_quant = "int8", text_encoder_quant = "none")
        Cc("c_int8d", "dit_int8_convrot.safetensors", te16)
        # explicit int8 on ROCm is weight-only unless UNSLOTH_NATIVE_INT8_ACT=1 (AUTO picks W8A8 for Qwen-Image families)
        # the same with the denoiser compiled (explicit speed_mode default): explicit int8 on ROCm resolves speed off
        S("s_int8adc", env = {"UNSLOTH_NATIVE_INT8_ACT": "1"}, transformer_quant = "int8", text_encoder_quant = "none",
          speed_mode = "default")
        S("s_int8dc", transformer_quant = "int8", text_encoder_quant = "none", speed_mode = "default")
        S("s_int8ad", env = {"UNSLOTH_NATIVE_INT8_ACT": "1"}, transformer_quant = "int8", text_encoder_quant = "none")
        S("s_q4", model = f["gguf"][0], gguf_filename = f["gguf"][1].format(q = "Q4_K_M"), model_kind = "gguf", **dense)
        Cc("c_q4", "dit_Q4_K_M.gguf", te16)
        S("s_def")
        Cc("c_bf16d", "dit_bf16.safetensors", te16, venv = "def")
        S("s_bf16_r", **dense)
        # MIOpen Find instead of Studio's immediate mode (VAE conv solver choice)
        S("s_bf16mf", env = {"MIOPEN_FIND_MODE": "NORMAL"}, **dense)
        S("s_bf16mff", env = {"MIOPEN_FIND_MODE": "FAST"}, **dense)
        # hipBLASLt TunableOp on the bf16 arm (the first render pays the tuning; its csv per family)
        S("s_bf16t", env = {"PYTORCH_TUNABLEOP_ENABLED": "1", "PYTORCH_TUNABLEOP_TUNING": "1",
                            "PYTORCH_TUNABLEOP_FILENAME": f"{W}/tunableop_{fam}_%d.csv"}, **dense)
        Cc("c_bf16_r", "dit_bf16.safetensors", te16)
        # P2
        S("s_fp8", transformer_quant = "fp8", text_encoder_quant = "none")
        Cc("c_fp8", "dit_fp8_scaled.safetensors", te16)
        S("s_nvfp4", env = {"UNSLOTH_NVFP4_DIFFUSION": "1"}, transformer_quant = "nvfp4", text_encoder_quant = "none")
        Cc("c_nvfp4", "dit_nvfp4.safetensors", te16)
        # single files (Studio's single-file route, the published file as is), bf16 encoder
        S("s_fp8f", model = f["fp8f"][0], gguf_filename = f["fp8f"][1], model_kind = "single_file",
          base_repo = f["model"], transformer_quant = "none", text_encoder_quant = "none")
        if "nvfp4f" in f:
            S("s_nvfp4f", model = f["nvfp4f"][0], gguf_filename = f["nvfp4f"][1], model_kind = "single_file",
              base_repo = f["model"], transformer_quant = "none", text_encoder_quant = "none")
        S("s_q8", model = f["gguf"][0], gguf_filename = f["gguf"][1].format(q = "Q8_0"), model_kind = "gguf", **dense)
        Cc("c_q8", "dit_Q8_0.gguf", te16)
        if fam.startswith("f3"):  # UD-Q4_K_XL is published for FLUX.3 only
            S("s_ud", model = f["gguf"][0], gguf_filename = f["gguf"][1].format(q = "UD-Q4_K_XL"), model_kind = "gguf", **dense)
            Cc("c_ud", "dit_UD-Q4_K_XL.gguf", te16)
        # A/A repeats of the explicit int8 DiT on both stacks, and Studio's INT8 attention on top of it
        S("s_int8d_r", transformer_quant = "int8", text_encoder_quant = "none")
        Cc("c_int8d_r", "dit_int8_convrot.safetensors", te16)
        S("s_int8d_i8a", env = {"UNSLOTH_DIFFUSION_INT8_ATTENTION_KERNEL": "1"}, transformer_quant = "int8",
          text_encoder_quant = "none")
        S("s_int8_r", transformer_quant = "int8", text_encoder_quant = "int8")
        Cc("c_int8_r", "dit_int8_convrot.safetensors", te8)
    return out


def main() -> None:
    spec = {"name": "prio8_img", "description": __doc__, "models": {},
            "defaults": {"backend": "studio", "kind": "image", "prompts": "image_24", "n": 3, "warmup": 1, "short_n": 0,
                         "options": {"local_files_only": False}},
            "gate": {"max_util": 15, "min_free_mib": 0, "timeout_s": 600}, "cells": cells()}
    out = Path(__file__).with_name("prio8_img.json")
    out.write_text(json.dumps(spec, indent = 1) + "\n", encoding = "utf-8")
    print(f"wrote {out} ({len(spec['cells'])} cells)")


if __name__ == "__main__":
    main()
