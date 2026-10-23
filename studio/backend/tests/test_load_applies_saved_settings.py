# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""POST /load fills the fields a caller leaves out from the launch settings saved in Studio."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest  # noqa: E402

import utils.openai_auto_switch_settings as settings  # noqa: E402
from models.inference import LoadRequest  # noqa: E402
from test_openai_auto_switch import _mock_override_store, _put  # noqa: E402

DEFAULTS = {"engine": "auto", "engine_precision": "auto", "engine_parallelism": "tensor"}


def _inference():
    from routes import inference
    return inference


@pytest.fixture
def gguf_target(monkeypatch):
    import core.inference.local_model_resolver as resolver
    monkeypatch.setattr(resolver, "local_target_is_gguf", lambda path, alias = None: True)


def _apply(request):
    return asyncio.run(_inference()._apply_saved_load_settings(request))


def test_omitted_fields_take_the_saved_row(monkeypatch, gguf_target):
    _mock_override_store(monkeypatch)
    _put(
        "unsloth/gemma-4-E2B-it-GGUF:Q4_K_M",
        max_seq_length = 32768,
        kv_cache_dtype = "q8_0",
        n_parallel = 2,
        **DEFAULTS,
    )
    request, applied = _apply(
        LoadRequest(model_path = "unsloth/gemma-4-E2B-it-GGUF", gguf_variant = "Q4_K_M")
    )
    assert request.max_seq_length == 32768
    assert request.cache_type_kv == "q8_0"
    assert request.n_parallel == 2
    assert {"max_seq_length", "cache_type_kv", "n_parallel"} <= set(applied)


def test_a_field_the_caller_sets_wins(monkeypatch, gguf_target):
    _mock_override_store(monkeypatch)
    _put("org/m-GGUF", max_seq_length = 32768, kv_cache_dtype = "q8_0", **DEFAULTS)
    request, applied = _apply(
        LoadRequest(model_path = "org/m-GGUF", max_seq_length = 4096, cache_type_kv = "f16")
    )
    assert request.max_seq_length == 4096
    assert request.cache_type_kv == "f16"
    assert applied == []


def test_an_explicit_zero_context_is_the_callers_choice(monkeypatch, gguf_target):
    """`unsloth start --context-length 0` is a typed reset and is sent; it is not filled."""
    _mock_override_store(monkeypatch)
    _put("org/m-GGUF", max_seq_length = 32768, **DEFAULTS)
    request, applied = _apply(LoadRequest(model_path = "org/m-GGUF", max_seq_length = 0))
    assert request.max_seq_length == 0
    assert "max_seq_length" not in applied


def test_the_ui_opts_out(monkeypatch, gguf_target):
    _mock_override_store(monkeypatch)
    _put("org/m-GGUF", max_seq_length = 32768, kv_cache_dtype = "q8_0", **DEFAULTS)
    original = LoadRequest(model_path = "org/m-GGUF", use_saved_settings = False)
    request, applied = _apply(original)
    assert request is original
    assert applied == []
    assert request.max_seq_length == 0


def test_no_saved_row_changes_nothing(monkeypatch, gguf_target):
    _mock_override_store(monkeypatch)
    original = LoadRequest(model_path = "org/unsaved-GGUF")
    request, applied = _apply(original)
    assert request is original and applied == []


def test_the_merged_request_is_validated_and_keeps_private_markers(monkeypatch, gguf_target):
    _mock_override_store(monkeypatch)
    _put("org/m-GGUF", max_seq_length = 16384, **DEFAULTS)
    original = LoadRequest(model_path = "org/m-GGUF", hf_token = "hf_x")
    original._override_alias_id = "org/m-GGUF"
    original._advertised_id = "advertised"
    request, applied = _apply(original)
    assert request.max_seq_length == 16384 and request.hf_token == "hf_x"
    assert request._override_alias_id == "org/m-GGUF"
    assert request._advertised_id == "advertised"
    # The filled field counts as set, so the load path treats it like a requested value.
    assert "max_seq_length" in request.model_fields_set


def test_a_broken_settings_store_never_fails_the_load(monkeypatch, gguf_target):
    def _boom(*a, **k):
        raise RuntimeError("settings db locked")

    monkeypatch.setattr(settings, "resolve_override_for_load", _boom)
    original = LoadRequest(model_path = "org/m-GGUF")
    request, applied = _apply(original)
    assert request is original and applied == []


def test_a_stale_gpu_pin_is_dropped_by_the_shared_reader(monkeypatch):
    monkeypatch.setattr(
        settings,
        "resolve_override_for_load",
        lambda *a, **k: ("org/m-GGUF", {"gpu_ids": [3], "tensor_split": [1.0], "n_parallel": 2}),
    )

    async def _never(ids, kind):
        return False

    key, row, kwargs = asyncio.run(
        settings.saved_load_settings("org/m-GGUF", None, None, is_gguf = True, gpu_ids_resolve = _never)
    )
    assert key == "org/m-GGUF"
    assert "gpu_ids" not in kwargs and "tensor_split" not in kwargs
    assert kwargs["n_parallel"] == 2


def test_auto_switch_and_load_read_the_same_settings(monkeypatch, gguf_target):
    _mock_override_store(monkeypatch)
    _put("org/m-GGUF:Q4_K_M", max_seq_length = 65536, kv_cache_dtype = "q4_0", **DEFAULTS)
    _, _, auto_switch_kwargs = asyncio.run(
        settings.saved_load_settings("org/m-GGUF", "org/m-GGUF", "Q4_K_M", is_gguf = True)
    )
    request, _ = _apply(LoadRequest(model_path = "org/m-GGUF", gguf_variant = "Q4_K_M"))
    for name, value in auto_switch_kwargs.items():
        assert getattr(request, name) == value


def _standing(monkeypatch, **chat_settings):
    """The Chat UI's standing choices, as /api/chat/settings stores them."""
    import storage.studio_db as studio_db
    monkeypatch.setattr(studio_db, "list_chat_settings", lambda: dict(chat_settings))


def test_standing_choices_fill_what_the_saved_row_leaves_unset(monkeypatch, gguf_target):
    _mock_override_store(monkeypatch)
    _standing(monkeypatch, speculativeType = "off", gpuMemoryMode = "manual")
    _put("org/m-GGUF", custom_context_length = 32768, kv_cache_dtype = "q8_0", **DEFAULTS)
    request, applied = _apply(LoadRequest(model_path = "org/m-GGUF"))
    assert request.speculative_type == "off"
    assert request.gpu_memory_mode == "manual"
    # Manual with Auto layers sends the context pin, as the UI's resolveFitMaxSeqLength does.
    assert request.max_seq_length == 32768
    assert {"speculative_type", "gpu_memory_mode", "cache_type_kv"} <= set(applied)


def test_standing_choices_apply_to_a_model_with_no_saved_row(monkeypatch, gguf_target):
    _mock_override_store(monkeypatch)
    _standing(monkeypatch, speculativeType = "ngram", gpuMemoryMode = "manual")
    request, applied = _apply(LoadRequest(model_path = "org/unsaved-GGUF"))
    assert request.speculative_type == "ngram"
    assert request.gpu_memory_mode == "manual"
    # No pin under Manual with Auto layers: --fit sizes the context.
    assert request.max_seq_length == 0
    assert set(applied) == {"speculative_type", "gpu_memory_mode", "max_seq_length"}


def test_the_saved_row_and_the_request_outrank_the_standing_choices(monkeypatch, gguf_target):
    _mock_override_store(monkeypatch)
    _standing(monkeypatch, speculativeType = "off", gpuMemoryMode = "manual")
    _put("org/m-GGUF", speculative_type = "mtp", **DEFAULTS)
    request, _ = _apply(LoadRequest(model_path = "org/m-GGUF", gpu_memory_mode = "auto"))
    assert request.speculative_type == "mtp"
    assert request.gpu_memory_mode == "auto"


def test_standing_auto_leaves_the_loader_default(monkeypatch, gguf_target):
    _mock_override_store(monkeypatch)
    _standing(monkeypatch, speculativeType = "auto", gpuMemoryMode = "auto")
    original = LoadRequest(model_path = "org/unsaved-GGUF")
    request, applied = _apply(original)
    assert request is original and applied == []


def test_a_non_gguf_load_takes_only_a_standing_off(monkeypatch):
    import core.inference.local_model_resolver as resolver

    monkeypatch.setattr(resolver, "local_target_is_gguf", lambda path, alias = None: False)
    _mock_override_store(monkeypatch)
    _standing(monkeypatch, speculativeType = "ngram", gpuMemoryMode = "manual")
    original = LoadRequest(model_path = "org/model")
    request, applied = _apply(original)
    # ngram reads as auto off a GGUF, and GPU memory mode is GGUF-only.
    assert request is original and applied == []
    _standing(monkeypatch, speculativeType = "off")
    request, applied = _apply(LoadRequest(model_path = "org/model"))
    assert request.speculative_type == "off" and applied == ["speculative_type"]


def test_the_auto_switch_row_carries_the_standing_choices(monkeypatch):
    """The speech budget estimates the context from this row, so it sizes what the load runs."""
    _mock_override_store(monkeypatch)
    _standing(monkeypatch, gpuMemoryMode = "manual")
    _put("org/m-GGUF", max_seq_length = 65536, **DEFAULTS)
    _, row, kwargs = asyncio.run(
        settings.saved_load_settings("org/m-GGUF", None, None, is_gguf = True)
    )
    assert row["gpu_memory_mode"] == "manual"
    assert settings.resolve_fit_max_seq_length(row, is_gguf = True) == kwargs["max_seq_length"] == 0


def test_the_speech_estimate_reads_the_row_as_the_load_does(monkeypatch):
    _mock_override_store(monkeypatch)
    _put("org/tts-GGUF", max_seq_length = 65536, **DEFAULTS)
    inference = _inference()
    monkeypatch.setattr(inference, "_target_native_context_length", lambda *a, **k: 4096)
    _standing(monkeypatch)
    assert inference._target_effective_context_length("org/tts-GGUF", True) == 65536
    # Manual with Auto layers hands sizing to --fit, so the saved length is not the window.
    _standing(monkeypatch, gpuMemoryMode = "manual")
    assert inference._target_effective_context_length("org/tts-GGUF", True) == 4096
