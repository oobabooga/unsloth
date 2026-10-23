# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The transformers backend honors seed, frequency_penalty and logit_bias instead of dropping them."""

import ast
from pathlib import Path

import pytest
import torch

from core.inference.presence_penalty import (
    _make_presence_penalty_processor,
    apply_frequency_penalty,
    apply_logit_bias,
    seeded_generation,
)


def test_generate_chat_response_declares_the_three_controls():
    # The worker forwards a control only when this signature declares it (_backend_declares in worker.py).
    # Parsed, not imported: core.inference.inference needs unsloth.
    source = Path(__file__).resolve().parent.parent / "core" / "inference" / "inference.py"
    tree = ast.parse(source.read_text(encoding = "utf-8"))
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "generate_chat_response"
    )
    names = {arg.arg for arg in method.args.args + method.args.kwonlyargs}
    assert {"seed", "frequency_penalty", "logit_bias"} <= names


def test_frequency_penalty_charges_every_repeat_of_completion_tokens_only():
    input_ids = torch.tensor([[5, 5, 9, 2, 2, 2]])  # prompt [5, 5], completion [9, 2, 2, 2]
    scores = torch.zeros(1, 10)
    apply_frequency_penalty(input_ids, scores, 0.5, prompt_len = 2)
    assert scores[0, 2].item() == pytest.approx(-1.5)
    assert scores[0, 9].item() == pytest.approx(-0.5)
    assert scores[0, 5].item() == 0.0


def test_logit_bias_adds_and_ignores_ids_past_the_vocab():
    scores = torch.zeros(1, 4)
    apply_logit_bias(scores, {"1": 5.0, 3: -100, "999": 7})
    assert scores[0].tolist() == [0.0, 5.0, 0.0, -100.0]


def test_a_processor_only_when_a_control_is_set():
    assert _make_presence_penalty_processor(0.0, 3, 0.0, None) is None
    assert _make_presence_penalty_processor(0.0, 3, 0.0, {}) is None
    assert _make_presence_penalty_processor(0.0, 3, 0.3, None) is not None
    assert _make_presence_penalty_processor(0.0, 3, 0.0, {"1": 1}) is not None


def test_a_seeded_draw_repeats_and_leaves_the_rng_as_it_found_it():
    torch.manual_seed(1234)
    before = torch.random.get_rng_state().clone()
    with seeded_generation(42):
        first = torch.rand(4)
    assert torch.equal(torch.random.get_rng_state(), before)
    with seeded_generation(42):
        second = torch.rand(4)
    assert torch.equal(first, second)
    with seeded_generation(-1):
        torch.rand(1)


def test_no_seed_touches_nothing():
    torch.manual_seed(99)
    state = torch.random.get_rng_state().clone()
    with seeded_generation(None):
        pass
    assert torch.equal(torch.random.get_rng_state(), state)
