"""ComfyUI API-format graphs per model family, for the ``comfyui`` backend.

Each family module mirrors the official Comfy-Org workflow template for that model (the node list and widget
values of comfyui_workflow_templates_json/templates/<name>.json at ComfyUI b5cc8830), rebuilt in API format with the
cell's prompt, seed, size and steps. A module defines:

    TEMPLATE  the template file it follows
    DEFAULTS  sampler / scheduler / shift / cfg the template uses (a cell's options override them)
    FILES     {role: comfy folder} the graph needs, e.g. {"dit": "diffusion_models", "te": "text_encoders"}
    build(p) -> graph dict; the node that saves the output is always id "save"

``p`` is a GraphParams. Shared wrappers (TorchCompileModel, EasyCache, the cache-busting nonce) are applied here,
so every family gets them the same way.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any, Optional

FAMILIES = {
    "qwen-image-2.1": "qwen_image_21",
    "z-image": "z_image",
    "flux.1": "flux1",
    "sdxl-turbo": "sdxl_turbo",
    "wan2.2-5b": "wan22_5b",
}
ALIASES = {
    "qwen-image-21": "qwen-image-2.1", "qwen_image_2.1": "qwen-image-2.1", "qwen21": "qwen-image-2.1",
    "z-image-turbo": "z-image", "zimage": "z-image", "z_image": "z-image",
    "flux.1-schnell": "flux.1", "flux.1-dev": "flux.1", "flux-schnell": "flux.1", "flux1": "flux.1", "flux": "flux.1",
    "sdxl_turbo": "sdxl-turbo", "sdxlturbo": "sdxl-turbo",
    "wan2.2-ti2v-5b": "wan2.2-5b", "wan22-5b": "wan2.2-5b", "wan2.2": "wan2.2-5b",
}
SAVE = "save"


@dataclass
class GraphParams:
    prompt: str
    seed: int
    steps: int
    width: int = 1024
    height: int = 1024
    negative: Optional[str] = None
    cfg: Optional[float] = None
    sampler: Optional[str] = None
    scheduler: Optional[str] = None
    shift: Optional[float] = None
    guidance: Optional[float] = None  # FLUX-dev distilled guidance (FluxGuidance)
    frames: Optional[int] = None
    files: dict = field(default_factory = dict)  # role -> basename (as ComfyUI lists it)
    weight_dtype: str = "default"
    compile: Any = None  # None / False, True, or a TorchCompileModel backend name
    easycache: Optional[float] = None
    easycache_start: float = 0.15
    easycache_end: float = 0.95
    prefix: str = "bench"
    nonce: Optional[int] = None  # an ignored input that changes the cache signature (see apply_nonce)
    nonce_nodes: str = "all"  # all | sampler | none
    extra: dict = field(default_factory = dict)


def canonical(family: str) -> str:
    fam = (family or "").strip().lower()
    fam = ALIASES.get(fam, fam)
    if fam not in FAMILIES:
        raise KeyError(f"no ComfyUI graph for family {family!r}; known: {sorted(FAMILIES)} (aliases {sorted(ALIASES)})")
    return fam


def module(family: str):
    return importlib.import_module(f"{__name__}.{FAMILIES[canonical(family)]}")


def pick(p: GraphParams, mod, key: str):
    val = getattr(p, key)
    return mod.DEFAULTS.get(key) if val is None else val


def wrap_model(g: dict, model_link: list, p: GraphParams) -> list:
    """TorchCompileModel then EasyCache on the MODEL link, in the order the EasyCache template chains them."""
    if p.compile:
        backend = p.compile if isinstance(p.compile, str) else "inductor"
        g["compile"] = {"class_type": "TorchCompileModel", "inputs": {"model": model_link, "backend": backend}}
        model_link = ["compile", 0]
    if p.easycache is not None:
        g["easycache"] = {"class_type": "EasyCache",
                          "inputs": {"model": model_link, "reuse_threshold": float(p.easycache),
                                     "start_percent": p.easycache_start, "end_percent": p.easycache_end,
                                     "verbose": False}}
        model_link = ["easycache", 0]
    return model_link


_LOADERS = ("UNETLoader", "CLIPLoader", "DualCLIPLoader", "VAELoader", "CheckpointLoaderSimple", "DiffusersLoader",
            "TorchCompileModel", "EasyCache", "ModelSamplingAuraFlow", "ModelSamplingSD3")
_SAMPLERS = ("KSampler", "SamplerCustom", "SamplerCustomAdvanced")


def apply_nonce(g: dict, p: GraphParams) -> dict:
    """ComfyUI caches every node output keyed on its inputs, so re-running a graph it has seen returns the cached
    image in ~0.1 s without sampling. run_cell renders the same prompt and seed twice (the timed pass and the
    long/short pairs), so without this the second one is a cache hit. An input the node does not declare is part
    of the cache signature (comfy_execution/caching.py hashes every raw input) but is dropped before the node runs
    (execution.get_input_data keeps declared inputs only), so a per-render ``_bench_nonce`` forces re-execution
    without changing any result. ``all`` re-runs text encoding, sampling and decoding (the models stay loaded):
    the same work Studio does per render. ``sampler`` re-runs from the sampler on, ``none`` leaves ComfyUI's cache
    alone."""
    if p.nonce is None or p.nonce_nodes == "none":
        return g
    for node in g.values():
        ct = node["class_type"]
        if ct in _LOADERS:
            continue
        if p.nonce_nodes == "sampler" and ct not in _SAMPLERS:
            continue
        node["inputs"]["_bench_nonce"] = int(p.nonce)
    return g


def save_node(images_link: list, prefix: str) -> dict:
    return {"class_type": "SaveImage", "inputs": {"images": images_link, "filename_prefix": prefix}}


def build(family: str, p: GraphParams) -> dict:
    mod = module(family)
    g = mod.build(p)
    if SAVE not in g:
        raise RuntimeError(f"comfy_graphs.{mod.__name__} built no '{SAVE}' node")
    return apply_nonce(g, p)
