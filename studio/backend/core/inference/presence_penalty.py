# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Sampling-control logits helpers (presence and frequency penalty, logit bias, seed) for the
safetensors inference paths.

Kept in a dependency-light leaf module (torch + transformers only, no unsloth /
peft) so the pure logic can be imported and unit-tested without pulling in the
full inference backend. ``core.inference.inference`` re-exports these for the
runtime generate paths.
"""

import contextlib

import torch


def apply_presence_penalty(input_ids, scores, penalty: float, prompt_len: int):
    """OpenAI/llama.cpp presence penalty: subtract ``penalty`` once per distinct
    completion token (positions >= prompt_len; prompt excluded, multiplicity
    ignored, negatives raise). In place; zero is a no-op."""
    if not penalty:
        return scores
    vocab_size = scores.shape[-1]
    for b in range(input_ids.shape[0]):
        generated = input_ids[b, prompt_len:]
        if generated.numel() == 0:
            continue
        seen = torch.unique(generated)
        # Bound generated ids to the valid range [0, vocab_size). Real completion tokens are always in range, so this
        # is a zero-regression safety net that drops any stray out-of-range or negative id before indexing (mirrors
        # the MLX path's bound). Filtering both ends avoids indexing scores with a negative id, which would silently
        # wrap to the wrong row.
        seen = seen[(seen >= 0) & (seen < vocab_size)]
        if seen.numel():
            scores[b, seen] = scores[b, seen] - penalty
    return scores


def apply_frequency_penalty(input_ids, scores, penalty: float, prompt_len: int):
    """OpenAI/llama.cpp frequency penalty: subtract ``penalty`` once per occurrence of each
    completion token. In place; zero is a no-op."""
    if not penalty:
        return scores
    vocab_size = scores.shape[-1]
    for b in range(input_ids.shape[0]):
        generated = input_ids[b, prompt_len:]
        generated = generated[(generated >= 0) & (generated < vocab_size)]
        if generated.numel():
            counts = torch.bincount(generated, minlength = vocab_size).to(torch.float32)
            scores[b] = scores[b] - (penalty * counts).to(scores.dtype)
    return scores


def apply_logit_bias(scores, logit_bias):
    """Add each OpenAI ``logit_bias`` entry to its token's logit. Ids past the logit width are
    ignored, as on the MLX path. In place."""
    vocab_size = scores.shape[-1]
    for token, bias in (logit_bias or {}).items():
        token = int(token)
        if 0 <= token < vocab_size:
            scores[:, token] = scores[:, token] + float(bias)
    return scores


def _make_presence_penalty_processor(
    penalty: float,
    prompt_len: int,
    frequency_penalty: float = 0.0,
    logit_bias = None,
):
    """``LogitsProcessorList`` for the presence and frequency penalties and logit bias; ``None``
    when all are off (generate call stays byte-identical)."""
    if not penalty and not frequency_penalty and not logit_bias:
        return None
    from transformers import LogitsProcessor, LogitsProcessorList

    class _PresencePenaltyLogitsProcessor(LogitsProcessor):
        @torch.no_grad()
        def __call__(self, input_ids, scores):
            scores = apply_presence_penalty(input_ids, scores, penalty, prompt_len)
            scores = apply_frequency_penalty(input_ids, scores, frequency_penalty, prompt_len)
            return apply_logit_bias(scores, logit_bias)

    return LogitsProcessorList([_PresencePenaltyLogitsProcessor()])


@contextlib.contextmanager
def seeded_generation(seed):
    """Seed one generate call and restore the CPU and every GPU RNG afterwards, so later
    unseeded requests stay random. ``None`` is a no-op."""
    if seed is None:
        yield
        return
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices = devices):
        # llama-server accepts -1, and the field is shared, so map any int into torch's range.
        torch.manual_seed(int(seed) % (2**64))
        yield
