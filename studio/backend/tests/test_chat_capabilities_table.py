# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""One table of which OpenAI chat controls each engine honors, and one way to refuse the rest.

The transformers worker dropped seed, frequency_penalty and logit_bias without a word while the
NPU refused them by name and the managed engines refused a bundle with a sentence naming none.
Now transformers honors all three, every refusal names its parameter, and status publishes the row.
"""

import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.inference import chat_capabilities as caps  # noqa: E402
from core.inference.presence_penalty import (  # noqa: E402
    apply_frequency_penalty,
    apply_logit_bias,
    logit_bias_pairs,
    make_sampling_processors,
    seeded_generation,
)


def _payload(**fields):
    base = dict(
        seed = None,
        frequency_penalty = 0.0,
        presence_penalty = 0.0,
        logit_bias = None,
        continue_final_message = False,
        use_adapter = None,
        enable_thinking = None,
        preserve_thinking = None,
        reasoning_effort = None,
        response_format = None,
    )
    base.update(fields)
    return SimpleNamespace(**base)


def _constrains(payload):
    return bool(payload.response_format) and payload.response_format.get("type") != "text"


class _Refused(Exception):
    pass


def _raise(param, message):
    raise _Refused(param, message)


# ── the table ─────────────────────────────────────────────────────────────


def test_defaults_ask_for_nothing():
    """A client that always sends seed None, zero penalties and {} is never refused."""
    assert caps.requested_controls(_payload(logit_bias = {}), constrains_decoding = _constrains) == []


@pytest.mark.parametrize(
    "fields,param",
    [
        ({"seed": 7}, "seed"),
        ({"frequency_penalty": 0.5}, "frequency_penalty"),
        ({"logit_bias": {"42": -100}}, "logit_bias"),
        ({"response_format": {"type": "json_object"}}, "response_format"),
    ],
)
def test_the_npu_row_refuses_by_name(fields, param):
    with pytest.raises(_Refused) as exc:
        caps.refuse_unsupported(
            _payload(**fields),
            caps.capabilities_for("npu"),
            _raise,
            constrains_decoding = _constrains,
        )
    assert exc.value.args[0] == param
    assert "NPU" in exc.value.args[1]


def test_response_format_is_reported_before_seed_as_the_npu_path_always_did():
    with pytest.raises(_Refused) as exc:
        caps.refuse_unsupported(
            _payload(seed = 1, response_format = {"type": "json_object"}),
            caps.capabilities_for("npu"),
            _raise,
            constrains_decoding = _constrains,
        )
    assert exc.value.args[0] == "response_format"


@pytest.mark.parametrize("engine", ["gguf", "mlx", "transformers"])
def test_local_engines_honor_every_sampling_control(engine):
    row = caps.capabilities_for(engine)
    for control in ("seed", "frequency_penalty", "presence_penalty", "logit_bias"):
        assert row.honors(control), (engine, control)


def test_only_transformers_lacks_a_grammar_engine():
    assert not caps.capabilities_for("transformers").honors("response_format")
    assert caps.capabilities_for("mlx").honors("response_format")
    assert caps.capabilities_for("gguf").honors("response_format")


@pytest.mark.parametrize(
    "fields,param",
    [
        ({"use_adapter": "my-lora"}, "use_adapter"),
        ({"response_format": {"type": "json_schema", "json_schema": {}}}, "response_format"),
        ({"continue_final_message": True}, "continue_final_message"),
    ],
)
def test_managed_engines_name_the_control_instead_of_a_bundle(fields, param):
    row = caps.capabilities_for("vllm")
    hit = caps.first_unsupported(_payload(**fields), row, constrains_decoding = _constrains)
    assert hit is not None and hit[0] == param
    assert "vLLM" in hit[1]


def test_reasoning_is_refused_only_without_a_thinking_switch():
    asked = _payload(reasoning_effort = "high")
    assert (
        caps.first_unsupported(asked, caps.capabilities_for("sglang", reasoning_controls = True))
        is None
    )
    with pytest.raises(_Refused) as exc:
        caps.refuse_unsupported(
            asked, caps.capabilities_for("sglang", reasoning_controls = False), _raise
        )
    assert exc.value.args[0] == "reasoning_effort"


def test_describe_lists_every_control_once():
    row = caps.capabilities_for("npu").describe()
    assert set(row["supported"]) | set(row["unsupported"]) == set(caps.CONTROLS)
    assert not set(row["supported"]) & set(row["unsupported"])


def test_the_managed_bundle_check_still_answers_like_before():
    from routes.inference import _managed_engine_unsupported_controls
    assert not _managed_engine_unsupported_controls(_payload(seed = 3, frequency_penalty = 0.4))
    assert _managed_engine_unsupported_controls(_payload(continue_final_message = True))


# ── transformers now honors the three it dropped ─────────────────────────────


def test_the_transformers_backend_declares_the_three_controls():
    """The worker forwards a control only when the backend's signature declares it."""
    from core.inference.inference import InferenceBackend

    params = inspect.signature(InferenceBackend.generate_chat_response).parameters
    for name in ("seed", "frequency_penalty", "logit_bias"):
        assert name in params


def test_frequency_penalty_charges_every_repeat_of_completion_tokens_only():
    input_ids = torch.tensor([[5, 5, 9, 2, 2, 2]])  # prompt [5, 5], completion [9, 2, 2, 2]
    scores = torch.zeros(1, 10)
    apply_frequency_penalty(input_ids, scores, 0.5, prompt_len = 2)
    assert scores[0, 2].item() == pytest.approx(-1.5)
    assert scores[0, 9].item() == pytest.approx(-0.5)
    assert scores[0, 5].item() == 0.0


def test_logit_bias_adds_and_ignores_ids_past_the_vocab():
    scores = torch.zeros(1, 4)
    apply_logit_bias(scores, logit_bias_pairs({"1": 5.0, 3: -100, "999": 7}))
    assert scores[0].tolist() == [0.0, 5.0, 0.0, -100.0]


def test_no_controls_means_no_processor():
    assert make_sampling_processors(0.0, 3, 0.0, None) is None
    assert make_sampling_processors(0.0, 3, 0.0, {}) is None
    assert make_sampling_processors(0.0, 3, 0.3, None) is not None


def test_a_seeded_draw_repeats_and_leaves_the_rng_as_it_found_it():
    torch.manual_seed(1234)
    before = torch.random.get_rng_state().clone()
    with seeded_generation(42):
        first = torch.rand(4)
    assert torch.equal(torch.random.get_rng_state(), before)
    with seeded_generation(42):
        second = torch.rand(4)
    assert torch.equal(first, second)
    with seeded_generation(-1):  # llama-server's "random" spelling, forwarded unchanged
        torch.rand(1)


def test_no_seed_touches_nothing():
    torch.manual_seed(99)
    state = torch.random.get_rng_state().clone()
    with seeded_generation(None):
        pass
    assert torch.equal(torch.random.get_rng_state(), state)
