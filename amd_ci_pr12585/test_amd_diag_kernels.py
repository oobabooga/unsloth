# Proposed replacement for test_clef_backbone_runs_the_compiled_gated_delta_and_conv_kernels:
# reads the kernel off the layer on transformers that bind it there (5.5.0, Studio's pin), and
# off the hub-kernel closure on later ones.
import pytest
import torch
from real_accelerator import has_real_cuda
from test_decision_model import clef_checkpoint  # noqa: F401

from unsloth import FastDecisionModel


def _chosen_module(layer, forward, global_name, attr):
    if attr in vars(layer):
        fn = vars(layer)[attr]
    else:
        fn = forward.__globals__[global_name]
    top = fn
    while fn is not None:
        cells = dict(zip(fn.__code__.co_freevars, fn.__closure__ or ()))
        if "implementation" in cells:
            return cells["implementation"].cell_contents.__module__
        fn = getattr(fn, "__wrapped__", None)
    return getattr(top, "__module__", None)


@pytest.mark.skipif(not has_real_cuda(), reason = "the fast kernels need a CUDA device")
def test_clef_backbone_runs_the_compiled_gated_delta_and_conv_kernels_any_transformers(clef_checkpoint):  # noqa: F811
    from transformers.utils.import_utils import (
        is_causal_conv1d_available,
        is_flash_linear_attention_available,
    )

    model, _ = FastDecisionModel.from_pretrained(str(clef_checkpoint), max_seq_length = 512)
    if not is_flash_linear_attention_available():
        pytest.skip("flash-linear-attention is disabled on this GPU / Triton")
    layers = [m for m in model.encoder.modules() if type(m).__name__.endswith("GatedDeltaNet")]
    assert layers
    for layer in layers:
        forward = type(layer).forward
        assert forward.__code__.co_filename.endswith("unsloth_compiled_module_qwen3_5.py")
        delta = _chosen_module(layer, forward, "torch_chunk_gated_delta_rule", "chunk_gated_delta_rule")
        conv = _chosen_module(layer, forward, "causal_conv1d_fn", "causal_conv1d_fn")
        print("KERNELS", delta, conv)
        assert delta is not None and delta.startswith("fla."), delta
        if is_causal_conv1d_available():
            assert conv is not None and conv.startswith("causal_conv1d"), conv
