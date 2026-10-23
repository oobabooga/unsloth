# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Presence-penalty logits helpers for the safetensors/MLX inference paths.

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


def _make_presence_penalty_processor(penalty: float, prompt_len: int):
    """``LogitsProcessorList`` for ``apply_presence_penalty``; ``None`` at zero penalty (generate call stays byte-identical)."""
    if not penalty:
        return None
    from transformers import LogitsProcessor, LogitsProcessorList

    class _PresencePenaltyLogitsProcessor(LogitsProcessor):
        @torch.no_grad()
        def __call__(self, input_ids, scores):
            return apply_presence_penalty(input_ids, scores, penalty, prompt_len)

    return LogitsProcessorList([_PresencePenaltyLogitsProcessor()])


def apply_frequency_penalty(input_ids, scores, penalty: float, prompt_len: int):
    """OpenAI/llama.cpp frequency penalty: subtract ``penalty`` once per OCCURRENCE of each
    completion token (positions >= prompt_len), so a token repeated N times is charged N x
    penalty. Counted in float32 and scaled once, as the MLX path does. In place; zero is a no-op."""
    if not penalty:
        return scores
    vocab_size = scores.shape[-1]
    for b in range(input_ids.shape[0]):
        generated = input_ids[b, prompt_len:]
        generated = generated[(generated >= 0) & (generated < vocab_size)]
        if generated.numel() == 0:
            continue
        counts = torch.bincount(generated, minlength = vocab_size).to(torch.float32)
        scores[b] = scores[b] - (penalty * counts).to(scores.dtype)
    return scores


def logit_bias_pairs(logit_bias) -> list:
    """``[(token_id, bias)]`` from an OpenAI ``logit_bias`` map (string or int keys)."""
    if not logit_bias:
        return []
    return [(int(token), float(bias)) for token, bias in logit_bias.items()]


def apply_logit_bias(scores, pairs: list):
    """Add each bias to its token's logit. An id outside the model's logit width is ignored,
    as on the MLX path, rather than raising mid-generation. In place."""
    vocab_size = scores.shape[-1]
    for token, bias in pairs:
        if 0 <= token < vocab_size:
            scores[:, token] = scores[:, token] + bias
    return scores


def make_sampling_processors(
    presence_penalty: float,
    prompt_len: int,
    frequency_penalty: float = 0.0,
    logit_bias = None,
):
    """One ``LogitsProcessorList`` for the OpenAI sampling knobs the transformers path honors
    (presence and frequency penalty, logit bias), or ``None`` when all are off, so the generate
    call stays byte-identical to the default. The transformers worker used to drop
    ``frequency_penalty`` and ``logit_bias`` silently while the GGUF and MLX paths applied them."""
    pairs = logit_bias_pairs(logit_bias)
    if not presence_penalty and not frequency_penalty and not pairs:
        return None
    from transformers import LogitsProcessor, LogitsProcessorList

    class _OpenAISamplingLogitsProcessor(LogitsProcessor):
        @torch.no_grad()
        def __call__(self, input_ids, scores):
            scores = apply_presence_penalty(input_ids, scores, presence_penalty, prompt_len)
            scores = apply_frequency_penalty(input_ids, scores, frequency_penalty, prompt_len)
            return apply_logit_bias(scores, pairs)

    return LogitsProcessorList([_OpenAISamplingLogitsProcessor()])


@contextlib.contextmanager
def seeded_generation(seed):
    """Run a generate call with ``seed`` and leave the process RNG as it was.

    ``torch.random.fork_rng`` saves and restores the CPU and the model's CUDA RNG state, so a
    seeded request is reproducible and an unseeded request after it is not silently made
    deterministic (the MLX path uses a request-scoped key for the same reason). Any Python int is
    accepted: it is reduced into torch's unsigned 64-bit seed domain, since llama-server forwards
    ``-1`` unchanged and the field is shared. ``None`` is a no-op."""
    if seed is None:
        yield
        return
    # Every visible GPU: a model split across cards samples on the last one, and torch.manual_seed
    # seeds them all, so all of them have to be put back afterwards.
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices = devices):
        torch.manual_seed(int(seed) % (2**64))
        yield
