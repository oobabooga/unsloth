# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Which OpenAI chat request controls each local chat engine honors, in one table.

Each engine used to decide this in its own place and its own way: the NPU path refused seed,
frequency_penalty and logit_bias by name, the transformers worker dropped the same three without a
word (it filtered them by function signature), and the managed vLLM / SGLang path refused a bundle
of controls with one sentence that named none of them. A client could not tell which of its
parameters had been ignored, and the frontend gated controls on ``isGguf`` / ``isMlx`` checks of
its own.

Rule: a control an engine cannot honor is refused by name, never dropped. ``refuse_unsupported``
is the one admission check; ``describe`` is what /api/inference/status publishes, so a client can
grey out a control before it sends it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Optional

# The request controls this table speaks for. Others (n, logprobs, audio and video input, tools)
# keep their own checks, which depend on the loaded model rather than on the engine.
CONTROLS = (
    "response_format",
    "seed",
    "frequency_penalty",
    "presence_penalty",
    "logit_bias",
    "continue_final_message",
    "use_adapter",
    "reasoning_controls",
)


@dataclass(frozen = True)
class ChatCapabilities:
    """One engine's answer: every control in ``CONTROLS`` is honored unless listed here."""

    engine: str
    label: str
    unsupported: Mapping[str, str] = field(default_factory = dict)

    def honors(self, control: str) -> bool:
        return control not in self.unsupported

    def describe(self) -> dict:
        return {
            "engine": self.engine,
            "supported": [c for c in CONTROLS if self.honors(c)],
            "unsupported": dict(self.unsupported),
        }


_MANAGED_ENGINE_GAPS = {
    "use_adapter": "{label} serves the base model only here; LoRA adapter switching needs a GGUF or "
    "transformers load.",
    "response_format": "{label} structured output is not wired through Unsloth yet; load a GGUF or "
    "MLX model to use response_format.",
    "continue_final_message": "{label} cannot resume a partial assistant message here; send the turn "
    "without continue_final_message.",
}
_REASONING_GAP = (
    "{label} has no reasoning controls for this model: it publishes no thinking switch, so "
    "enable_thinking, preserve_thinking and reasoning_effort cannot be honored."
)


def capabilities_for(engine: str, *, reasoning_controls: bool = True) -> ChatCapabilities:
    """The table row for a loaded engine: ``gguf``, ``mlx``, ``transformers``, ``vllm``,
    ``sglang`` or ``npu``. ``reasoning_controls`` is False when the loaded model publishes no
    thinking switch, which only the managed engines cannot work around."""
    engine = (engine or "").strip().lower()
    if engine in ("vllm", "sglang"):
        label = "vLLM" if engine == "vllm" else "SGLang"
        gaps = {key: text.format(label = label) for key, text in _MANAGED_ENGINE_GAPS.items()}
        if not reasoning_controls:
            gaps["reasoning_controls"] = _REASONING_GAP.format(label = label)
        return ChatCapabilities(engine, label, gaps)
    if engine == "npu":
        label = "FastFlowLM on the NPU"
        return ChatCapabilities(
            engine,
            label,
            {
                "seed": f"{label} does not take a sampling seed.",
                "frequency_penalty": f"{label} does not take a frequency penalty.",
                "logit_bias": f"{label} does not take a logit bias.",
                "response_format": f"{label} has no guided decoding for response_format.",
            },
        )
    if engine == "transformers":
        return ChatCapabilities(
            engine,
            "transformers",
            {
                "response_format": "response_format needs a grammar engine, and the transformers "
                "backend has none; load an MLX or GGUF model to use it.",
            },
        )
    if engine == "mlx":
        return ChatCapabilities(engine, "MLX")
    return ChatCapabilities("gguf", "llama.cpp")


def requested_controls(
    payload, *, constrains_decoding: Optional[Callable[[object], bool]] = None
) -> list[str]:
    """The controls ``payload`` actually asks for, in ``CONTROLS`` order. A default value
    (``seed`` None, a zero penalty, an empty bias map, a plain-text response_format) asks for
    nothing, so a client that always sends defaults is never refused."""
    asked = []
    if constrains_decoding is not None and constrains_decoding(payload):
        asked.append("response_format")
    if getattr(payload, "seed", None) is not None:
        asked.append("seed")
    if getattr(payload, "frequency_penalty", 0.0):
        asked.append("frequency_penalty")
    if getattr(payload, "presence_penalty", 0.0):
        asked.append("presence_penalty")
    if getattr(payload, "logit_bias", None):
        asked.append("logit_bias")
    if getattr(payload, "continue_final_message", False):
        asked.append("continue_final_message")
    if getattr(payload, "use_adapter", None) is not None:
        asked.append("use_adapter")
    if (
        getattr(payload, "enable_thinking", None)
        or getattr(payload, "preserve_thinking", None)
        or getattr(payload, "reasoning_effort", None) not in (None, "none")
    ):
        asked.append("reasoning_controls")
    return asked


def first_unsupported(
    payload,
    capabilities: ChatCapabilities,
    *,
    constrains_decoding: Optional[Callable[[object], bool]] = None,
) -> Optional[tuple[str, str]]:
    """``(control, reason)`` for the first control ``payload`` asks for that the engine cannot
    honor, or None."""
    for control in requested_controls(payload, constrains_decoding = constrains_decoding):
        if not capabilities.honors(control):
            return control, capabilities.unsupported[control]
    return None


def refuse_unsupported(
    payload,
    capabilities: ChatCapabilities,
    raise_unsupported: Callable[[str, str], None],
    *,
    constrains_decoding: Optional[Callable[[object], bool]] = None,
    controls: Optional[tuple[str, ...]] = None,
) -> None:
    """Refuse the request, naming the parameter, when it asks for a control the engine cannot
    honor. ``raise_unsupported(param, message)`` is the caller's OpenAI-shaped 400; the parameter
    reported for reasoning is the one the client actually set. ``controls`` limits the check to a
    subset when the caller still settles the rest itself."""
    for control in requested_controls(payload, constrains_decoding = constrains_decoding):
        if controls is not None and control not in controls:
            continue
        if capabilities.honors(control):
            continue
        param = control
        if control == "reasoning_controls":
            param = next(
                name
                for name in ("enable_thinking", "preserve_thinking", "reasoning_effort")
                if getattr(payload, name, None) not in (None, False, "none")
            )
        raise_unsupported(param, capabilities.unsupported[control])
