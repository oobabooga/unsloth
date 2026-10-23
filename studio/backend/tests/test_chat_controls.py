# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""One row per backend says which chat controls it honors; status and admission both read it."""

import ast
import asyncio
from pathlib import Path

import httpx
import pytest
from fastapi import HTTPException

from core.inference.chat_controls import chat_controls, in_process_backend
from models.inference import ChatCompletionRequest
from routes import inference as route

from .llama_backend_double import FakeLlamaCppBackend

_TOOL = {"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}


def _payload(**fields):
    return ChatCompletionRequest(model = "m", messages = [{"role": "user", "content": "hi"}], **fields)


def _honored(backend, **kwargs):
    return {name for name, honored in chat_controls(backend, **kwargs).items() if honored}


def test_each_backend_declares_what_it_honors():
    everything = {
        "seed",
        "frequency_penalty",
        "logit_bias",
        "response_format",
        "continue_final_message",
        "continue_thought",
        "context_overflow",
        "compact_context",
    }
    assert set(chat_controls("gguf")) == everything
    assert _honored("gguf") == _honored("mlx") == everything
    assert _honored("transformers") == {
        "seed",
        "frequency_penalty",
        "logit_bias",
        "continue_final_message",
    }
    assert _honored("vllm") == {"seed", "frequency_penalty", "logit_bias", "context_overflow"}
    assert _honored("sglang") == {"seed", "frequency_penalty", "logit_bias"}
    assert _honored("npu") == {"continue_final_message"}
    # An older llama-server cannot reopen a thought.
    assert _honored("gguf", resumes_thoughts = False) == everything - {"continue_thought"}
    # A backend missing from the table is an error, not one that honors everything.
    with pytest.raises(ValueError):
        chat_controls("tensorrt")


@pytest.mark.parametrize(
    "model_info, backend",
    [
        ({"is_mlx": True}, "mlx"),
        ({"engine": "vllm"}, "vllm"),
        ({"engine": "sglang"}, "sglang"),
        ({"engine": "auto"}, "transformers"),
        ({}, "transformers"),
    ],
)
def test_an_in_process_model_reads_its_row_from_its_own_info(model_info, backend):
    assert in_process_backend(model_info) == backend


@pytest.mark.parametrize(
    "backend, source", [("transformers", "inference.py"), ("mlx", "mlx_inference.py")]
)
def test_the_worker_forwards_exactly_what_an_in_process_row_honors(backend, source):
    # worker.py forwards these only when generate_chat_response declares them (_backend_declares).
    # Parsed, not imported: core.inference.inference needs unsloth.
    path = Path(__file__).resolve().parent.parent / "core" / "inference" / source
    tree = ast.parse(path.read_text(encoding = "utf-8"))
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "generate_chat_response"
    )
    declared = {arg.arg for arg in method.args.args + method.args.kwonlyargs}
    for name in ("seed", "frequency_penalty", "logit_bias", "response_format"):
        assert (name in declared) is chat_controls(backend)[name], name


def test_status_and_load_responses_publish_the_row_admission_reads():
    assert route._unsloth_serving_fields({"engine": "vllm"})["chat_controls"] == chat_controls(
        "vllm"
    )
    # A backend holding no model declares nothing.
    assert route._unsloth_serving_fields({})["chat_controls"] is None
    gguf = FakeLlamaCppBackend()
    assert route._llama_runtime_fields(gguf)["chat_controls"] == chat_controls("gguf")
    gguf._resumes_thoughts = False
    assert route._llama_runtime_fields(gguf)["chat_controls"]["continue_thought"] is False


@pytest.mark.parametrize(
    "backend, fields, param",
    [
        ("npu", {"seed": 7}, "seed"),
        ("npu", {"frequency_penalty": 0.5}, "frequency_penalty"),
        ("npu", {"logit_bias": {"42": -100}}, "logit_bias"),
        ("npu", {"response_format": {"type": "json_object"}}, "response_format"),
        ("transformers", {"response_format": {"type": "json_object"}}, "response_format"),
        ("vllm", {"response_format": {"type": "json_object"}}, "response_format"),
        ("vllm", {"continue_final_message": True}, "continue_final_message"),
        ("sglang", {"continue_final_message": True}, "continue_final_message"),
    ],
)
def test_a_control_the_row_lacks_is_refused_by_name(backend, fields, param):
    with pytest.raises(HTTPException) as caught:
        route._refuse_unsupported_chat_controls(_payload(**fields), backend)
    assert caught.value.status_code == 400
    error = caught.value.detail["error"]
    assert (error["param"], error["code"]) == (param, "unsupported_parameter")


@pytest.mark.parametrize("backend", ["gguf", "mlx", "transformers", "vllm", "sglang", "npu"])
def test_defaults_and_a_set_aside_format_ask_for_nothing(backend):
    # What every client sends when it sets none of them, and a spelled-out text format.
    route._refuse_unsupported_chat_controls(
        _payload(
            seed = None,
            frequency_penalty = 0.0,
            logit_bias = {},
            response_format = {"type": "text"},
            continue_final_message = False,
        ),
        backend,
    )
    # Callable client tools set a structured format aside on every backend.
    route._refuse_unsupported_chat_controls(
        _payload(response_format = {"type": "json_object"}, tools = [_TOOL]), backend
    )


@pytest.mark.parametrize("backend", ["gguf", "mlx", "transformers", "vllm", "sglang"])
def test_every_local_engine_but_the_npu_takes_the_sampling_controls(backend):
    route._refuse_unsupported_chat_controls(
        _payload(seed = 3, frequency_penalty = 0.4, logit_bias = {"7": 2.0}), backend
    )


@pytest.mark.parametrize(
    "backend",
    [name for name in route._CHAT_BACKEND_LABELS if not chat_controls(name)["response_format"]],
)
def test_deep_research_retries_every_missing_grammar_refusal_without_the_format(backend):
    from core.research_runs import _response_format_unsupported

    with pytest.raises(HTTPException) as caught:
        route._refuse_unsupported_chat_controls(
            _payload(response_format = {"type": "json_object"}), backend
        )
    response = httpx.Response(400, json = caught.value.detail)
    assert asyncio.run(_response_format_unsupported(response))
