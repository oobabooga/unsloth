# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""One field list per direction for a model's saved launch settings, kept in step.

Run settings sends ApiModelOverride (frontend model-overrides.ts), the route accepts
ModelOverridePayload, the store keeps what normalize_model_override returns, and
saved_load_settings turns that row into LoadRequest fields. A field missing from any of them is
saved and then never applied, which is how load_mode and three others went missing once.
"""

import asyncio
import re
from pathlib import Path

import pytest

import utils.openai_auto_switch_settings as settings
from models.inference import LoadRequest
from routes.settings import ModelOverridePayload

TS_OVERRIDES = (
    Path(__file__).resolve().parents[2]
    / "frontend"
    / "src"
    / "features"
    / "model-picker"
    / "api"
    / "model-overrides.ts"
)

# Write modes, not settings.
CONTROL_FIELDS = {
    "model_id",
    "remove",
    "fill_absent_fields",
    "mirrors_server_tuning",
    "mirrors_reasoning_budget",
    "mirrors_spec_draft_model",
}

VLLM = {"engine": "vllm"}
# One valid non-default value per setting, and what it needs beside it to take effect.
SAMPLES = {
    "engine": ("vllm", {}),
    "engine_precision": ("fp8", VLLM),
    "engine_parallelism": ("pipeline", VLLM),
    "engine_kv_cache_dtype": ("fp8", VLLM),
    "engine_seed": (7, VLLM),
    "engine_gpu_memory_utilization": (0.5, VLLM),
    "load_in_4bit": (False, {}),
    "llama_extra_args": (["--no-warmup"], {}),
    "max_seq_length": (8192, {}),
    "custom_context_length": (4096, {}),
    "kv_cache_dtype": ("q8_0", {}),
    "mlx_kv_quant": ("4", {}),
    "mlx_kv_bits": (8, {}),
    "mlx_int8_prefill": (True, {}),
    "mlx_ple_offload": (0.5, {}),
    "mlx_expert_offload": ("off", {}),
    "speculative_type": ("mtp", {}),
    "spec_draft_n_max": (3, {"speculative_type": "mtp"}),
    "spec_draft_model": ("org/drafter", {"speculative_type": "mtp"}),
    "spec_draft_cache_type": ("q8_0", {"speculative_type": "dspark"}),
    "n_parallel": (4, {}),
    "reasoning_budget": (256, {}),
    "reasoning_budget_message": ("Done thinking.", {}),
    "n_batch": (1024, {}),
    "n_ubatch": (256, {}),
    "load_mode": ("mmap", {}),
    "ctx_checkpoints": (4, {}),
    "cache_ram": (0, {}),
    "tensor_parallel": (True, {}),
    "disable_vision": (True, {}),
    "chat_template_override": ("{{ messages }}", {}),
    "gpu_memory_mode": ("manual", {}),
    "gpu_layers": (10, {}),
    "n_cpu_moe": (2, {}),
    "gpu_ids": ([0], {}),
    "gpu_index_kind": ("vulkan", {"gpu_ids": [0]}),
    "tensor_split": ([1.0, 2.0], {"gpu_ids": [0, 1]}),
}


def _ts_fields() -> set[str]:
    source = TS_OVERRIDES.read_text(encoding = "utf-8")
    body = re.search(r"export interface ApiModelOverride \{(.*?)\n\}", source, re.S)
    assert body, "ApiModelOverride not found in model-overrides.ts"
    return set(re.findall(r"^\s*(\w+)\?:", body.group(1), re.M))


def _route_fields() -> set[str]:
    return set(ModelOverridePayload.model_fields) - CONTROL_FIELDS


def _load_fields(payload: dict, *, is_gguf: bool, monkeypatch) -> dict:
    """What a load applies for a row saved from ``payload``, through the shared reader."""
    row = settings.normalize_model_override(payload)
    monkeypatch.setattr(settings, "resolve_override_for_load", lambda *a, **k: ("m", row))
    monkeypatch.setattr(settings, "standing_load_choices", lambda **k: {})

    async def physical_only(ids, kind):
        return kind == "physical"

    return asyncio.run(
        settings.saved_load_settings(
            "m", None, None, is_gguf = is_gguf, gpu_ids_resolve = physical_only
        )
    )[2]


def test_the_ui_sends_exactly_the_fields_the_route_accepts():
    assert _ts_fields() == _route_fields()


def test_every_field_has_a_sample():
    assert set(SAMPLES) == _route_fields()


@pytest.mark.parametrize("field", sorted(SAMPLES))
def test_every_saved_field_reaches_a_load(field, monkeypatch):
    value, needs = SAMPLES[field]
    reached = [
        is_gguf
        for is_gguf in (True, False)
        if _load_fields({**needs, field: value}, is_gguf = is_gguf, monkeypatch = monkeypatch)
        != _load_fields(needs, is_gguf = is_gguf, monkeypatch = monkeypatch)
    ]
    assert reached, f"{field} is saved but no load applies it"


@pytest.mark.parametrize("is_gguf", [True, False])
def test_the_load_fields_are_valid_load_request_fields(is_gguf, monkeypatch):
    payload = {}
    for field, (value, needs) in SAMPLES.items():
        payload.update(needs)
        payload.setdefault(field, value)
    # The width folds into mlx_kv_quant, and a Vulkan-numbered pin is dropped on this host.
    payload.pop("mlx_kv_bits")
    payload.pop("gpu_index_kind")
    kwargs = _load_fields(payload, is_gguf = is_gguf, monkeypatch = monkeypatch)
    assert set(kwargs) <= set(LoadRequest.model_fields)
    LoadRequest(model_path = "m", **kwargs)
