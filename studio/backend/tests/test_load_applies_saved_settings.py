# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""POST /load fills the fields a caller leaves out from the model's saved launch settings."""

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
