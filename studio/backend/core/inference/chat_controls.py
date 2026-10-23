# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Which chat request controls each local backend honors.

Load and status responses publish the loaded model's row as ``chat_controls``, the chat route
refuses or ignores a control its row lacks, and the chat UI offers only the controls in the row.
Reasoning, tools, images, audio and video input are per-model status fields instead.
"""

# The backends that lack each control. continue_thought resumes a reasoning-only partial inside
# the thought; context_overflow (fitting the prompt to the window) needs an exact prompt count;
# compact_context (/compact) needs the forced checkpoint only the GGUF and MLX fits start.
_LACKING = {
    "seed": {"npu"},
    "frequency_penalty": {"npu"},
    "logit_bias": {"npu"},
    "response_format": {"transformers", "vllm", "sglang", "npu"},
    "continue_final_message": {"vllm", "sglang"},
    "continue_thought": {"transformers", "vllm", "sglang", "npu"},
    "context_overflow": {"transformers", "sglang", "npu"},
    "compact_context": {"transformers", "vllm", "sglang", "npu"},
}

LABELS = {
    "gguf": "GGUF",
    "mlx": "MLX",
    "transformers": "transformers",
    "vllm": "vLLM",
    "sglang": "SGLang",
    "npu": "NPU",
}


def in_process_backend(model_info: dict) -> str:
    """The row for a model the inference orchestrator serves."""
    if model_info.get("is_mlx"):
        return "mlx"
    engine = model_info.get("engine")
    return engine if engine in ("vllm", "sglang") else "transformers"


def chat_controls(backend: str, *, resumes_thoughts: bool = True) -> dict:
    """Each control mapped to whether ``backend`` honors it. An older llama-server cannot reopen
    a thought (``resumes_thoughts`` False)."""
    if backend not in LABELS:
        raise ValueError(f"No chat controls are declared for {backend!r}.")
    controls = {name: backend not in lacking for name, lacking in _LACKING.items()}
    controls["continue_thought"] = controls["continue_thought"] and resumes_thoughts
    return controls
