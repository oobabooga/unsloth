#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Build a tiny random-weight copy of any transformers model, locally, when no tiny repo exists.

    python3 tiny_make.py Qwen/Qwen3-0.6B                # real config shrunk + the real tokenizer
    python3 tiny_make.py DeepseekV4ForCausalLM          # an architecture: its default config shrunk
    python3 tiny_make.py qwen3_moe --hidden 128         # a model_type; sizes are overridable

    from tiny_make import make
    path = make("Qwen/Qwen3-30B-A3B")   # a local directory; from_pretrained(path) loads it

The tiny test repos on the Hub are random weights too, so the same thing can be made on demand:
read the config, shrink every size in it (layers, hidden, heads, head dims, MLP, experts, MLA
ranks, linear-attention heads, the same for vision / audio sub-configs), build the model from
that config with a fixed seed, run one forward pass, and save it with the source's tokenizer /
processor. Vocabulary, token ids, rope settings, image and patch sizes are kept, so the real
tokenizer and processor still fit.

Kept on purpose: at least one layer of every distinct ``layer_types`` entry (so sliding / full /
linear attention all appear), a shared expert and a dense first layer when the source has them.
Every size is a multiple of 16, so grouped-mm, Triton and 4-bit kernels take their real paths.

Built models are cached under ``$WORKSPACE/temp/tiny_cache`` (``TINY_CACHE`` overrides), keyed on
the source, the sizes, the seed and the transformers version, and built under a file lock so
parallel sessions share one copy. Needs torch + transformers; ``tiny_models.py`` stays torch-free
and calls this only when no tiny repo matches.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
_CODE_VERSION = 5  # bump when the shrinking rules change, so cached builds are redone


@dataclasses.dataclass(frozen=True)
class Sizes:
    hidden: int = 64
    layers: int = 2
    heads: int = 4
    head_dim: int = 16
    intermediate: int = 128
    moe_intermediate: int = 64
    experts: int = 4
    experts_per_tok: int = 2
    lora_rank: int = 32


# ── Shrinking a config dict (pure; no torch) ──────────────────────────────────

_LAYERS = ("num_hidden_layers", "num_layers", "n_layer", "n_layers", "depth", "encoder_layers", "decoder_layers")


def _is_layers(k: str) -> bool:  # also prefixed forms such as conf_num_hidden_layers (Gemma 3n audio)
    return k in _LAYERS or k.endswith("_num_hidden_layers")
_HIDDEN = re.compile(r"^(hidden_size|d_model|n_embd|embed_dim|emb_dim|hidden_dim|dim|width|attention_hidden_size)$")
_HEADS = re.compile(r"^(num_attention_heads|n_head|n_heads|num_heads|encoder_attention_heads|"
                    r"decoder_attention_heads|attention_heads|index_n_heads|linear_num_heads|[a-z]+_num_attention_heads)$")
_KV_HEADS = re.compile(r"(key_value_heads|kv_heads|multi_query_group_num)$")
_HEAD_DIM = re.compile(r"(head_dim|d_head)$")
_INTER = re.compile(r"^(intermediate_size|ffn_dim|n_inner|intermediate_size_mlp|mlp_dim|"
                    r"encoder_ffn_dim|decoder_ffn_dim|dense_intermediate_size|ffn_hidden_size|shared_intermediate_size)$")
_MOE_INTER = re.compile(r"^(moe_intermediate_size|shared_expert_intermediate_size|expert_intermediate_size|"
                        r"moe_ffn_hidden_size|expert_ffn_hidden_size)$")
_EXPERTS = re.compile(r"^(num_experts|num_local_experts|n_routed_experts|moe_num_experts|num_routed_experts|"
                      r"zero_expert_num)$")
_TOPK = re.compile(r"^(num_experts_per_tok|num_experts_per_token|moe_topk|moe_k)$")
_ONE = re.compile(r"^(n_shared_experts|num_shared_experts|n_group|num_expert_group|topk_group|"
                  r"first_k_dense_replace)$")
_RANK = re.compile(r"lora_rank$")
_INDEX_LIST = re.compile(r"(layers|_idx|_idxs|indexes|indices)$")
_ROTARY = re.compile(r"^(rotary_dim|rotary_emb_dim|rope_dim)$")
# Widths no rule names: scaled with the model's hidden size, so ratios between them survive.
_WIDTH = re.compile(r"(hidden|_dim$|_size$|channels|width)")
_NOT_WIDTH = re.compile(r"(vocab|position|window|token|image_size|patch|crop|max_|seq|length|chunk|scale|topk|"
                        r"bins|codebook|num_embeddings|sampling|batch|buckets|stride|kernel|group_size|block_size)")


def _first(d: dict, keys) -> str | None:
    return next((k for k in keys if isinstance(d.get(k), int) and not isinstance(d.get(k), bool)), None)


def _keep_layers(d: dict, n0: int, want: int) -> list[int]:
    """Original layer indices to keep: the first layer of every distinct layer type, topped up
    with the earliest remaining layers to ``want``."""
    keep: list[int] = []
    for k, types in d.items():  # layer_types, mlp_layer_types, layers_block_type, ...
        if "type" in k and isinstance(types, list) and len(types) == n0 and all(isinstance(t, str) for t in types):
            seen = set()
            for i, t in enumerate(types):
                if t not in seen:
                    seen.add(t)
                    if i not in keep:
                        keep.append(i)
    # A repeating pattern shorter than the stack (block_types = [recurrent, recurrent, attention]):
    # keep enough leading layers to include every entry of it.
    for k, v in d.items():
        if k.endswith("types") and isinstance(v, list) and 0 < len(v) < n0 and all(isinstance(t, str) for t in v):
            want = max(want, max(v.index(t) for t in set(v)) + 1)
    for i in range(n0):
        if len(keep) >= min(want, n0) and keep:
            break
        if i not in keep:
            keep.append(i)
    return sorted(keep)


def _scale_sections(sections: list[int], total: int) -> list[int]:
    """Rescale rope sections (mrope) to add up to ``total`` (largest remainders get the rest)."""
    s = sum(sections)
    if not s or total <= 0:
        return [0] * len(sections)
    exact = [x * total / s for x in sections]
    out = [int(x) for x in exact]
    for i in sorted(range(len(out)), key=lambda i: out[i] - exact[i])[:total - sum(out)]:
        out[i] += 1
    return out


def _is_config(v: Any, key: str = "") -> bool:
    if not isinstance(v, dict) or not v:
        return False
    if "model_type" in v:
        return True
    if all(str(i).isdigit() for i in v):  # per-layer overrides keyed by layer index
        return False
    # *_config, or a plain dict that carries model sizes (remote-code configs such as Cohere ASR keep
    # their encoder / decoder / head as NeMo-style dicts: encoder.d_model, transf_decoder.config_dict)
    return key.endswith("_config") or _has_sizes(v)


def _has_sizes(d: dict, depth: int = 0) -> bool:
    for k, v in d.items():
        if isinstance(v, int) and not isinstance(v, bool) and (_is_layers(k) or _HIDDEN.match(k) or _HEADS.match(k)):
            return True
        if depth < 2 and isinstance(v, dict) and v and not all(str(i).isdigit() for i in v) \
                and _has_sizes(v, depth + 1):
            return True
    return False


def _shrink_node(d: dict, s: Sizes, text_hidden0: int | None, text_hidden: int | None,
                 hidden0: int | None = None) -> list[str]:
    """Shrink one config level in place; returns the changed keys."""
    changed: list[str] = []

    def put(k, v):
        if d.get(k) != v:
            d[k] = v
            changed.append(k)

    heads0 = d.get(_first(d, [k for k in d if _HEADS.match(k)]) or "")
    orig = dict(d)
    layer_keys = [k for k in _LAYERS if k in d] + [k for k in d if _is_layers(k) and k not in _LAYERS]
    lk = next((k for k in layer_keys if isinstance(d.get(k), int) and not isinstance(d.get(k), bool)
               and d[k] > s.layers), None)
    if lk:
        n0 = d[lk]
        keep = _keep_layers(d, n0, s.layers)
        for k, v in list(d.items()):
            if k == "architectures" or not isinstance(v, list) or not v:
                continue
            if len(v) == n0:  # one entry per layer
                put(k, [v[i] for i in keep])
            elif _INDEX_LIST.search(k) and all(isinstance(i, int) and not isinstance(i, bool) for i in v):
                mapped = [min(len(keep) - 1, i * len(keep) // n0) for i in v]
                # Mllama concatenates the hidden states of intermediate_layers_indices, so their
                # count sets a projection width: keep it (repeats are fine). Others dedupe.
                put(k, mapped if "intermediate" in k else sorted(set(mapped)))
        for k, v in list(d.items()):  # per-layer overrides keyed by layer index
            if isinstance(v, dict) and v and all(str(i).isdigit() and int(i) < n0 for i in v):
                put(k, {str(keep.index(int(i))) if isinstance(i, str) else keep.index(i): x
                        for i, x in v.items() if int(i) in keep})
        for k in layer_keys:
            if isinstance(d.get(k), int) and not isinstance(d.get(k), bool) and d[k] > len(keep):
                put(k, len(keep))
        if isinstance(d.get("max_window_layers"), int):
            put("max_window_layers", min(d["max_window_layers"], len(keep)))
        if d.get("num_kv_shared_layers"):
            put("num_kv_shared_layers", 0)
        for k in [k for k in d if k.endswith("_period") and isinstance(d[k], int) and d[k] > len(keep)]:
            put(k, len(keep))  # every Nth layer is attention / MoE: make the last kept layer one
            off = k.replace("period", "offset")
            if isinstance(d.get(off), int):
                put(off, min(d[off], len(keep) - 1))
    for k, v in list(d.items()):
        if not isinstance(v, int) or isinstance(v, bool) or k.endswith("token_id") or _is_layers(k):
            continue
        if _HIDDEN.match(k):
            new = s.hidden
        elif _KV_HEADS.search(k):
            new = v if v == 1 else (s.heads if heads0 and v == heads0 else max(1, s.heads // 2))
        elif k == "linear_num_key_heads":
            new = max(1, s.heads // 2)
        elif k == "linear_num_value_heads":
            ratio = max(1, v // max(1, d.get("linear_num_key_heads") or v))
            new = max(1, s.heads // 2) * ratio
            if new >= v:
                continue
        elif _HEADS.match(k):
            new = s.heads
        elif _HEAD_DIM.search(k):
            new = s.head_dim
        elif _MOE_INTER.match(k):
            new = s.moe_intermediate
        elif _INTER.match(k):
            new = s.intermediate
        elif _EXPERTS.match(k):
            new = s.experts
        elif _TOPK.match(k):
            new = min(v, s.experts_per_tok)
        elif _ONE.match(k):
            new = min(v, 1)
        elif _RANK.search(k):
            new = s.lora_rank
        elif _ROTARY.match(k):
            new = s.head_dim
        elif text_hidden0 and v == text_hidden0 and text_hidden:  # a projection into the text model
            new = text_hidden
        elif v >= 256 and _WIDTH.search(k) and not _NOT_WIDTH.search(k) and (hidden0 or text_hidden0):
            ref = hidden0 or text_hidden0
            new = max(16, round(v * s.hidden / ref / 16) * 16)
        else:
            continue
        if new < v:
            put(k, new)
    for k, v in list(d.items()):  # per-stage lists (Florence2 / Swin style: embed_dim, num_heads, depths)
        if not (isinstance(v, list) and v and all(isinstance(x, int) and not isinstance(x, bool) for x in v)):
            continue
        if _HIDDEN.match(k) and v[0] > s.hidden:
            put(k, [max(16, round(x * s.hidden / v[0] / 16) * 16) for x in v])
        elif _HEADS.match(k):
            put(k, [min(x, s.heads) for x in v])
        elif k == "depths":
            put(k, [min(x, 1) for x in v])
    # Derived sizes some configs store alongside their parts.
    rope0, hd0 = orig.get("qk_rope_head_dim"), orig.get("head_dim")
    if isinstance(rope0, int) and isinstance(hd0, int) and rope0 != hd0 and rope0 and d.get("qk_rope_head_dim"):
        # head_dim * partial_rotary_factor == qk_rope_head_dim (Mistral4 / DeepSeek-V4 MLA)
        put("head_dim", max(16, hd0 * d["qk_rope_head_dim"] // rope0))
    ratio = d.get("pixel_shuffle_ratio")
    if isinstance(ratio, float) and 0 < ratio < 1 and isinstance(d.get("hidden_size"), int) \
            and isinstance(d.get("intermediate_size"), int):
        # Llama 4's vision adapter takes the pixel-shuffled width, hidden / ratio**2
        put("intermediate_size", round(d["hidden_size"] / ratio ** 2))
    for p in ("", "mamba_"):  # Mamba2: heads * head_dim == expand * hidden
        hk, dk = (f"{p}n_heads" if p else "num_heads"), (f"{p}d_head" if p else "head_dim")
        ek, hid = f"{p}expand", d.get("hidden_size")
        if all(isinstance(d.get(x), int) for x in (hk, dk, ek)) and isinstance(hid, int) and d[dk]:
            put(hk, max(1, d[ek] * hid // d[dk]))
            gk = f"{p}n_groups"
            if isinstance(d.get(gk), int) and d[hk] % d[gk]:
                put(gk, 1)
    if isinstance(d.get("qk_nope_head_dim"), int) and isinstance(d.get("qk_rope_head_dim"), int) \
            and "qk_head_dim" in d:
        put("qk_head_dim", d["qk_nope_head_dim"] + d["qk_rope_head_dim"])
    if "query_pre_attn_scalar" in d and isinstance(d.get("head_dim"), int):
        put("query_pre_attn_scalar", d["head_dim"])
    for k in [k for k in d if k.endswith("_groups") and isinstance(d[k], int) and d[k] > 1]:
        heads = d.get(_first(d, [h for h in d if _HEADS.match(h)]) or "")
        if heads and heads % d[k]:
            put(k, max(1, min(d[k], heads // 2)))
    return changed


def _fix_rope(node: Any, head_dim: int | None, factor: float = 1.0) -> None:
    """mrope sections add up to half the rotary dim; rescale them to the new head_dim."""
    if isinstance(node, dict):
        factor = node.get("partial_rotary_factor") or factor
        if isinstance(node.get("mrope_section"), list) and head_dim:
            node["mrope_section"] = _scale_sections(node["mrope_section"], int(head_dim * factor) // 2)
        for v in node.values():
            if isinstance(v, dict) and "model_type" not in v:
                _fix_rope(v, head_dim, factor)


def shrink_config(d: dict, sizes: Sizes = Sizes()) -> dict:
    """A shrunk copy of a config dict (``PretrainedConfig.to_dict()`` form), recursing into
    sub-configs. Pure; the unit tests run it without torch."""
    d = json.loads(json.dumps(d))
    text = d["text_config"] if _is_config(d.get("text_config"), "text_config") else d
    text_hidden0 = text.get(_first(text, [k for k in text if _HIDDEN.match(k)]) or "")

    def walk(node: dict, is_text: bool) -> None:
        if isinstance(node.get("architecture"), str):  # a timm tower: a fixed named model, no sizes to change
            return
        hidden0 = node.get(_first(node, [k for k in node if _HIDDEN.match(k)]) or "")
        _shrink_node(node, sizes, None if is_text else text_hidden0,
                     None if is_text else text.get("hidden_size") or sizes.hidden, hidden0)
        _fix_rope(node, node.get("head_dim") if isinstance(node.get("head_dim"), int) else sizes.head_dim)
        for k, v in node.items():
            if _is_config(v, k):
                walk(v, v is text)

    if text is not d:  # text first, so the other sub-configs can project into its new width
        walk(text, True)
    for k, v in list(d.items()):
        if _is_config(v, k) and v is not text:
            walk(v, False)
    _shrink_node(d, sizes, text_hidden0 if text is not d else None,
                 text.get("hidden_size") or sizes.hidden, text_hidden0)
    _fix_rope(d, text.get("head_dim") if isinstance(text.get("head_dim"), int) else sizes.head_dim)
    for k in ("_name_or_path", "transformers_version", "_commit_hash", "quantization_config"):
        d.pop(k, None)
    _clamp_feature_layers(d)
    _drop_sub_defaults(d)
    return d


def _clamp_feature_layers(d: dict) -> None:
    """vision_feature_layer (-2, or a list such as Granite Vision's [-24, -20, -12, -1]) indexes the
    vision tower's hidden states (layers + 1 entries): keep every index inside the shrunk tower."""
    vision = d.get("vision_config")
    n = next((vision[k] for k in _LAYERS if isinstance(vision, dict) and isinstance(vision.get(k), int)), None)
    if n is None:
        return
    lo, hi = -(n + 1), n

    def clamp(i):
        return max(lo, min(hi, i)) if isinstance(i, int) and not isinstance(i, bool) else i

    for k, v in d.items():
        if "feature_layer" in k:
            d[k] = [clamp(i) for i in v] if isinstance(v, list) else clamp(v)


def _drop_sub_defaults(d: dict) -> None:
    """Sub-configs carry the base-class bookkeeping keys, which a few config classes reject."""
    for v in d.values():
        if isinstance(v, dict):
            for k in ("_name_or_path", "transformers_version", "output_attentions", "output_hidden_states",
                      "return_dict"):
                v.pop(k, None)
            _drop_sub_defaults(v)


# ── Resolving a key to a config + model class ─────────────────────────────────

_TASK_MAPS = ("MODEL_FOR_CAUSAL_LM_MAPPING_NAMES", "MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES",
              "MODEL_FOR_MULTIMODAL_LM_MAPPING_NAMES", "MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES",
              "MODEL_MAPPING_NAMES")


def _arch_model_type(arch: str) -> str | None:
    from transformers.models.auto import modeling_auto as ma

    for name in dir(ma):
        if name.endswith("_MAPPING_NAMES") and isinstance(getattr(ma, name), dict):
            for mt, cls in getattr(ma, name).items():
                if cls == arch or (isinstance(cls, (tuple, list)) and arch in cls):
                    return mt
    return None


def _default_arch(model_type: str) -> str | None:
    from transformers.models.auto import modeling_auto as ma

    for name in _TASK_MAPS:
        cls = getattr(ma, name, {}).get(model_type)
        if cls:
            return cls if isinstance(cls, str) else cls[0]
    return None


# Remote-code models (config.json `auto_map`): their classes come from the repo's own .py files,
# loaded only with trust_remote_code (--trust-remote-code / TINY_TRUST_REMOTE_CODE=1). Only config,
# code and tokenizer files are fetched, never weights.
_DYNAMIC: dict[str, Any] = {}       # architecture -> model class loaded from a repo's code
_AUTO_ORDER = ("AutoModelForCausalLM", "AutoModelForSpeechSeq2Seq", "AutoModelForSeq2SeqLM",
               "AutoModelForImageTextToText", "AutoModelForVision2Seq", "AutoModelForCTC", "AutoModel")


class RemoteCodeError(RuntimeError):
    """The source needs its own modeling code, and it was not allowed or does not exist."""


def trust_remote_code(flag: bool | None = None) -> bool:
    if flag is not None:
        return bool(flag)
    return os.environ.get("TINY_TRUST_REMOTE_CODE", "") not in ("", "0")


def _raw_config(key: str) -> dict:
    if Path(key).is_dir():
        return json.loads((Path(key) / "config.json").read_text())
    from huggingface_hub import hf_hub_download

    return json.loads(Path(hf_hub_download(key, "config.json")).read_text())


def _weights_gb(repo: str) -> float | None:
    with contextlib.suppress(Exception):
        from huggingface_hub import HfApi

        info = HfApi().model_info(repo, files_metadata=True)
        return sum((f.size or 0) for f in info.siblings or [] if _WEIGHTS.search(f.rfilename)) / 1e9
    return None


def _closest_model_type(raw: dict) -> str | None:
    import difflib

    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    name = ((raw.get("architectures") or [""])[0] or "").lower()
    name = re.sub(r"(for)?(causallm|conditionalgeneration|model|lmhead)$", "", name)
    same = [mt for mt in CONFIG_MAPPING.keys() if name and mt.replace("_", "").startswith(name)]
    if same:  # LongcatCausalLM -> longcat_flash: the same family name beats a near spelling
        return min(same, key=len)
    near = difflib.get_close_matches(name, list(CONFIG_MAPPING.keys()), n=1, cutoff=0.5)
    return near[0] if near else None


def _known_rope(raw: dict) -> dict:
    """A rope_scaling / rope_parameters type this transformers lacks ("deepseek_yarn") as its
    in-library base ("yarn"), else default rope: the --model-type approximation must load."""
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    raw = dict(raw)
    for key in ("rope_scaling", "rope_parameters"):
        rs = raw.get(key)
        if not isinstance(rs, dict):
            continue
        kind = rs.get("rope_type") or rs.get("type")
        if kind and kind not in ROPE_INIT_FUNCTIONS:
            base = next((b for b in ROPE_INIT_FUNCTIONS if b != "default" and kind.endswith(b)), None)
            raw[key] = dict(rs, rope_type=base, type=base) if base else None
    return raw


def _remote_source(key: str, raw: dict, trust: bool, model_type: str | None):
    """(config, architecture) for a repo transformers cannot load by itself, or None when it can."""
    from transformers import AutoConfig
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    if model_type:  # caller's choice of an in-library model_type for a config that names none
        cfg = CONFIG_MAPPING[model_type].from_dict({k: v for k, v in _known_rope(raw).items() if k != "model_type"})
        arch = _default_arch(model_type)
        cfg.architectures = [arch]
        return cfg, arch
    auto_map = raw.get("auto_map") or {}
    if not auto_map:
        if raw.get("model_type") in CONFIG_MAPPING:
            return None
        arch = (raw.get("architectures") or ["?"])[0]
        if hasattr(_transformers(), arch):
            return None
        gb = _weights_gb(key) if "/" in key else None
        near = _closest_model_type(raw)
        raise RemoteCodeError(
            f"{key}: config.json has no model_type transformers {_tf_version()} knows "
            f"({raw.get('model_type')!r}) and no auto_map, and transformers has no {arch}: there is no "
            f"modeling code to build a tiny copy from"
            + (f" (closest model_type here: {near}; --model-type {near} builds that architecture with "
               f"this config's sizes, an approximation)" if near else "")
            + (f". Its weights are {gb:.0f} GB: do not download them to test support" if gb else ""))
    if not trust and raw.get("model_type") in CONFIG_MAPPING:
        return None         # code shipped for a model transformers also has: the library class serves
    if not trust:
        raise RemoteCodeError(f"{key} ships its own modeling code (auto_map: {', '.join(sorted(auto_map))}); "
                              f"pass --trust-remote-code (or TINY_TRUST_REMOTE_CODE=1) to build it from that code")
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    cfg = AutoConfig.from_pretrained(key, trust_remote_code=True)
    arch = (cfg.architectures or [None])[0]
    ref = next((v for v in auto_map.values() if isinstance(v, str) and arch and v.endswith("." + arch)), None)
    ref = ref or next((auto_map[k] for k in _AUTO_ORDER if isinstance(auto_map.get(k), str)), None)
    if ref is None:
        raise RemoteCodeError(f"{key}: auto_map names no model class ({', '.join(sorted(auto_map))})")
    try:
        cls = get_class_from_dynamic_module(ref, key)
    except ImportError as e:  # the repo's code imports packages this environment lacks
        raise RemoteCodeError(f"{key}: its modeling code cannot import here: {str(e).splitlines()[0]}") from e
    arch = cls.__name__
    _DYNAMIC[arch] = cls
    cfg.architectures = [arch]
    return cfg, arch


def _sources(key: Any, trust: bool | None = None, model_type: str | None = None) -> list[tuple]:
    """Configs to try for ``key``, best first: [(config, architecture, tokenizer repo or None)].

    An architecture or model_type gets two: transformers' default config (the real model's
    shape), then the config of its tiny test repo (fills fields some defaults leave unset);
    both take the tokenizer from that repo."""
    from transformers import AutoConfig, PretrainedConfig
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    if isinstance(key, PretrainedConfig):
        found = [(key, None)]
    elif isinstance(key, dict):
        found = [(CONFIG_MAPPING[key["model_type"]].from_dict(dict(key)), None)]
    elif "/" in key or Path(key).is_dir():
        remote = _remote_source(key, _raw_config(key), trust_remote_code(trust), model_type)
        if remote:
            return [(remote[0], remote[1], key)]
        found = [(AutoConfig.from_pretrained(key), key)]
    else:
        tiny = _curated(key)
        mt = _arch_model_type(key) if key[:1].isupper() else key
        found = []
        if mt in CONFIG_MAPPING:
            found.append((CONFIG_MAPPING[mt](), tiny))
        if tiny:
            with contextlib.suppress(Exception):
                found.append((AutoConfig.from_pretrained(tiny), tiny))
        if not found:
            raise LookupError(f"{key} is not a model class or model_type in transformers {_tf_version()}")
        if key[:1].isupper():
            for cfg, _ in found:
                cfg.architectures = [key]
    out = []
    for cfg, repo in found:
        arch = (cfg.architectures or [None])[0]
        if arch is None or not hasattr(_transformers(), arch):  # none named, or renamed since
            arch = _default_arch(cfg.model_type) or arch
        if not arch:
            raise LookupError(f"no model class for model_type {cfg.model_type}")
        want = getattr(_model_class(arch), "config_class", None)
        if want is not None and getattr(want, "model_type", None) and want.model_type != cfg.model_type:
            # e.g. Qwen3_5ForCausalLM under a composite qwen3_5 config: the text part is its config
            text = cfg.get_text_config() if hasattr(cfg, "get_text_config") else None
            cfg = text if isinstance(text, want) else want()
            cfg.architectures = [arch]
        out.append((cfg, arch, repo))
    return out


def _transformers():
    import transformers

    return transformers


def _model_class(arch: str):
    import transformers

    cls = _DYNAMIC.get(arch) or getattr(transformers, arch, None)
    if cls is None:
        raise LookupError(f"transformers {transformers.__version__} has no class {arch}")
    return cls


def _curated(key: str) -> str | None:
    """The best tiny repo for an architecture / model_type (weights or not), via tiny_models.py."""
    if not isinstance(key, str) or "/" in key:
        return None
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    try:
        import tiny_models
    except ImportError:
        return None
    return tiny_models.best(key, need_weights=False)


def _tf_version() -> str:
    import transformers

    return transformers.__version__


# ── Build, check, save ────────────────────────────────────────────────────────


def _cache_root() -> Path:
    if os.environ.get("TINY_CACHE"):
        return Path(os.environ["TINY_CACHE"])
    return Path(os.environ.get("WORKSPACE") or Path.cwd()) / "temp" / "tiny_cache"


@contextlib.contextmanager
def _lock(path: Path):
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def _embedding_sizes(model) -> tuple[int, int]:
    """(input embedding, untied output head) parameter counts; vocab-sized, so they dominate."""
    sizes = []
    for getter in ("get_input_embeddings", "get_output_embeddings"):
        try:
            mod = getattr(model, getter)()
        except Exception:  # noqa: BLE001 - some classes leave these unimplemented
            mod = None
        sizes.append(mod.weight if mod is not None and hasattr(mod, "weight") else None)
    inp, out = sizes
    n_in = inp.numel() if inp is not None else 0
    n_out = out.numel() if out is not None and (inp is None or out.data_ptr() != inp.data_ptr()) else 0
    return n_in, n_out


def _from_config(cls, config):
    """float32 model from a config; transformers 4.x names the argument torch_dtype."""
    import torch

    try:
        return cls._from_config(config, dtype=torch.float32)
    except TypeError:
        return cls._from_config(config, torch_dtype=torch.float32)


def _forward_check(model, config) -> str:
    """One forward pass on a few token ids; returns "ok" or "skipped: <why>"."""
    import inspect

    import torch

    params = inspect.signature(model.forward).parameters
    if "input_ids" not in params:
        return "skipped: forward takes no input_ids"
    text = config.get_text_config() if hasattr(config, "get_text_config") else config
    vocab = getattr(text, "vocab_size", None) or 100
    ids = torch.randint(0, min(vocab, 100), (1, 8))
    kwargs = {"input_ids": ids}
    if getattr(config, "is_encoder_decoder", False) and "decoder_input_ids" in params:
        kwargs["decoder_input_ids"] = ids
    if "input_features" in params:  # speech encoder-decoder (Cohere ASR): a short random log-mel clip
        n_mels = _find_int(config.to_dict(), ("feat_in", "num_mel_bins", "features", "n_mels", "feature_size")) or 80
        kwargs["input_features"] = torch.randn(1, n_mels, 160)
        if "length" in params:
            kwargs["length"] = torch.tensor([160])
    try:
        with torch.no_grad():
            out = model(**kwargs)
    except (TypeError, ValueError) as e:  # needs pixel_values / other inputs text alone cannot give
        msg = str(e)
        if "required positional argument" in msg or "pixel_values" in msg:
            return "skipped: " + msg.splitlines()[0][:120]
        raise
    logits = getattr(out, "logits", None)
    if logits is None:
        logits = getattr(out, "last_hidden_state", None)
    if logits is not None and not torch.isfinite(logits.float()).all():
        raise RuntimeError("forward produced non-finite outputs")
    return "ok"


def _find_int(d: Any, keys) -> int | None:
    """The first int under any of ``keys``, searching nested dicts breadth-first."""
    level = [d]
    while level:
        for node in level:
            for k in keys:
                if isinstance(node.get(k), int) and not isinstance(node.get(k), bool):
                    return node[k]
        level = [v for node in level for v in node.values() if isinstance(v, dict)]
    return None


def timm_towers(d: dict) -> list[str]:
    """Sub-configs that name a fixed timm model (kept full size: Gemma 3n's MobileNetV5)."""
    return [f"{k} ({v['architecture']})" for k, v in d.items()
            if isinstance(v, dict) and isinstance(v.get("architecture"), str)]


def _image_check(model, repo: str | None) -> str:
    """For a model with an image processor: one forward pass on a chat-templated image + text
    prompt built by the source's own processor. "n/a" when there is nothing to test."""
    import inspect

    import torch

    if repo is None or "pixel_values" not in inspect.signature(model.forward).parameters:
        return "n/a"
    try:
        from PIL import Image
        from transformers import AutoProcessor

        proc = AutoProcessor.from_pretrained(repo)
    except Exception:  # noqa: BLE001 - no processor to drive it with
        return "n/a"
    if getattr(proc, "image_processor", None) is None or not getattr(proc, "chat_template", None):
        return "n/a"
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "What is this?"}]}]
    text = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    image = Image.new("RGB", (64, 64), (200, 30, 30))
    inputs = proc(text=[text], images=[image], return_tensors="pt")
    dtype = next(model.parameters()).dtype
    inputs = {k: v.to(dtype) if torch.is_floating_point(v) else v for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs)
    logits = getattr(out, "logits", None)
    if logits is not None and not torch.isfinite(logits.float()).all():
        raise RuntimeError("image forward produced non-finite outputs")
    return "ok"


_WEIGHTS = re.compile(r"(\.safetensors|\.gguf|\.bin|\.pth|\.ckpt|\.h5|\.msgpack|\.onnx)$|^(original|onnx|gguf)/|"
                      r"(^|/)(consolidated|pytorch_model|model)[-.]")


def _copy_extras(repo: str | None, out: Path, limit: int = 20 * 2**20) -> list[str]:
    """Small files the source ships next to its weights that loading needs (Qwen2.5-Omni's
    spk_dict.pt, custom chat templates, ...): every non-weight file under ``limit`` not already written."""
    if repo is None or Path(repo).is_dir():
        return []
    copied = []
    try:
        from huggingface_hub import HfApi, hf_hub_download

        info = HfApi().model_info(repo, files_metadata=True)
    except Exception:  # noqa: BLE001 - optional extras
        return []
    for f in info.siblings or []:
        name, size = f.rfilename, f.size or 0
        if (size > limit or _WEIGHTS.search(name) or (out / name).exists() or name.startswith(".")
                or name.lower().startswith("readme")):
            continue
        with contextlib.suppress(Exception):
            dst = out / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(hf_hub_download(repo, name), dst)
            copied.append(name)
    return copied


def _save_processor(repo: str | None, out: Path, trust: bool = False) -> str | None:
    """Save the source's processor (or tokenizer) and generation config next to the weights."""
    if repo is None:
        return None
    saved = None
    kw = {"trust_remote_code": True} if trust else {}
    for loader in ("AutoProcessor", "AutoTokenizer"):
        try:
            import transformers

            getattr(transformers, loader).from_pretrained(repo, **kw).save_pretrained(out)
            saved = loader
            break
        except Exception:
            continue
    with contextlib.suppress(Exception):
        from transformers import GenerationConfig

        GenerationConfig.from_pretrained(repo).save_pretrained(out)
    return saved


def _tokenizer_len(repo: str | None, trust: bool = False) -> int | None:
    if repo is None:
        return None
    with contextlib.suppress(Exception):
        from transformers import AutoTokenizer

        return len(AutoTokenizer.from_pretrained(repo, **({"trust_remote_code": True} if trust else {})))
    return None


def _plan(cfg, arch: str, tok_repo: str | None, sizes: Sizes, dtype: str, trust: bool = False) -> dict:
    """The shrunk config for one source, vocabulary widened to fit its tokenizer."""
    small = shrink_config(cfg.to_dict(), sizes)
    small["architectures"] = [arch]
    small["dtype"] = dtype
    n_tok = _tokenizer_len(tok_repo, trust)
    holder = small["text_config"] if isinstance(small.get("text_config"), dict) else small
    if n_tok and isinstance(holder.get("vocab_size"), int) and holder["vocab_size"] < n_tok:
        holder["vocab_size"] = n_tok
    return small


def build(key: Any, *, sizes: Sizes = Sizes(), seed: int = 0, check: bool = True,
          tokenizer: str | None = None, dtype: str = "bfloat16", trust: bool | None = None,
          model_type: str | None = None):
    """(model, info) for the first source of ``key`` that builds and passes the forward check.
    The model is float32 on CPU; nothing is saved."""
    import torch

    errors = []
    trust = trust_remote_code(trust)
    for cfg, arch, repo in _sources(key, trust, model_type):
        tok_repo = tokenizer or repo
        try:
            small = _plan(cfg, arch, tok_repo, sizes, dtype, trust)
            config = type(cfg).from_dict(small)
            torch.manual_seed(seed)
            model = _from_config(_model_class(arch), config).eval()
            status = _forward_check(model, config) if check else "skipped"
            image = _image_check(model, tok_repo) if check else "skipped"
        except Exception as e:  # noqa: BLE001 - try the next source, report all at the end
            first = next((ln.strip() for ln in str(e).splitlines() if ln.strip()), "")
            errors.append(f"{'config of ' + repo if repo and cfg.name_or_path else 'default config'}: "
                          f"{type(e).__name__}: {first[:200]}")
            continue
        return model, {"config": small, "architecture": arch, "source": repo or arch, "remote_code": arch in _DYNAMIC,
                       "tokenizer_from": tok_repo, "forward": status, "image_forward": image,
                       "kept_full_size": timm_towers(small)}
    raise RuntimeError(f"could not build a tiny {key}:\n  " + "\n  ".join(errors))


def make(key: Any, *, sizes: Sizes = Sizes(), seed: int = 0, dtype: str = "bfloat16",
         tokenizer: str | None = None, out_dir: str | os.PathLike | None = None,
         check: bool = True, force: bool = False, trust: bool | None = None,
         model_type: str | None = None) -> str:
    """Build (or reuse from the cache) a tiny random copy of ``key`` and return its directory.

    ``key``: a model id or local path, an architecture (``Qwen3MoeForCausalLM``), a model_type,
    a config dict or a ``PretrainedConfig``. ``tokenizer`` names the repo to take the
    tokenizer / processor from; a model id is its own default."""
    import torch
    import transformers

    trust = trust_remote_code(trust)
    ident = json.dumps({"key": key if isinstance(key, str) else _sources(key)[0][0].to_dict(),
                        "trust_remote_code": trust, "model_type": model_type,
                        "sizes": dataclasses.asdict(sizes), "seed": seed, "dtype": dtype, "check": check,
                        "tokenizer": tokenizer, "transformers": transformers.__version__,
                        "torch": torch.__version__.split("+")[0], "code": _CODE_VERSION},
                       sort_keys=True, default=str)
    digest = hashlib.sha256(ident.encode()).hexdigest()[:12]
    name = key if isinstance(key, str) else "config"
    # a local source is named by its directory: a slug of the full path makes remote-code module
    # names longer than the filesystem allows (transformers names the module after the directory)
    short = Path(name).name if isinstance(key, str) and Path(key).is_dir() else name
    slug = re.sub(r"[^A-Za-z0-9._-]+", "--", short).strip("-")
    out = Path(out_dir) if out_dir else _cache_root() / f"{slug}-{digest}"
    marker = out / "tiny_make.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with _lock(out.parent / f".{out.name}.lock"):
        if marker.exists() and not force:
            return str(out)
        t0 = time.time()
        model, meta = build(key, sizes=sizes, seed=seed, check=check, tokenizer=tokenizer, dtype=dtype,
                            trust=trust, model_type=model_type)
        n = sum(p.numel() for p in model.parameters())
        n_emb, n_out = _embedding_sizes(model)
        model = model.to(getattr(torch, dtype))
        tmp = Path(tempfile.mkdtemp(prefix=f".{out.name}.", dir=out.parent))
        try:
            try:
                model.save_pretrained(tmp)
            except RuntimeError as e:
                # remote code that ties weights by assignment without declaring it (Cohere ASR ties its
                # classifier to the decoder embedding): safetensors refuses shared tensors, torch.save keeps them
                if "shared tensors" not in str(e):
                    raise
                model.save_pretrained(tmp, safe_serialization=False)
            processor = _save_processor(meta["tokenizer_from"], tmp, trust)
            extras = _copy_extras(meta["tokenizer_from"], tmp)
            if meta["remote_code"] and Path(str(meta["source"])).is_dir():
                # a local remote-code source: its .py files make the copy loadable (a Hub source's
                # come through _copy_extras)
                for py in sorted(Path(meta["source"]).glob("*.py")):
                    if not (tmp / py.name).exists():
                        shutil.copyfile(py, tmp / py.name)
                        extras.append(py.name)
            (tmp / "tiny_make.json").write_text(json.dumps({
                "key": name, "source": meta["source"], "architecture": meta["architecture"],
                "model_type": meta["config"].get("model_type"), "remote_code": meta["remote_code"],
                "sizes": dataclasses.asdict(sizes), "seed": seed,
                "dtype": dtype, "tokenizer_from": meta["tokenizer_from"], "processor": processor,
                "copied_from_source": extras, "forward": meta["forward"], "image_forward": meta["image_forward"],
                "kept_full_size": meta["kept_full_size"], "parameters": n, "embedding_parameters": n_emb,
                "body_parameters": n - n_emb - n_out,
                "transformers": transformers.__version__, "torch": torch.__version__,
                "build_seconds": round(time.time() - t0, 1),
            }, indent=1) + "\n")
            if out.exists():
                shutil.rmtree(out)
            os.replace(tmp, out)
        finally:
            if tmp.exists():
                shutil.rmtree(tmp, ignore_errors=True)
    return str(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("key", help="model id / path, architecture or model_type")
    for f in dataclasses.fields(Sizes):
        ap.add_argument(f"--{f.name.replace('_', '-')}", type=int, default=f.default)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--tokenizer", help="repo to copy the tokenizer / processor from")
    ap.add_argument("--out", help="output directory (default: the cache)")
    ap.add_argument("--no-check", action="store_true", help="skip the forward-pass check")
    ap.add_argument("--force", action="store_true", help="rebuild even if cached")
    ap.add_argument("--trust-remote-code", action="store_true", default=None,
                    help="build a model that ships its own code (config.json auto_map) from that code; "
                         "only config, code and tokenizer files are fetched (also TINY_TRUST_REMOTE_CODE=1)")
    ap.add_argument("--model-type", help="for a config that names no model_type transformers knows: build it as "
                                         "this in-library model_type (an approximation)")
    a = ap.parse_args(argv)
    sizes = Sizes(**{f.name: getattr(a, f.name) for f in dataclasses.fields(Sizes)})
    try:
        path = make(a.key, sizes=sizes, seed=a.seed, dtype=a.dtype, tokenizer=a.tokenizer, out_dir=a.out,
                    check=not a.no_check, force=a.force, trust=a.trust_remote_code, model_type=a.model_type)
    except RemoteCodeError as e:
        print(f"tiny_make: {e}", file=sys.stderr)
        return 2
    meta = json.loads((Path(path) / "tiny_make.json").read_text())
    print(path)
    print(f"{meta['architecture']}: {meta['parameters'] / 1e6:.2f}M parameters "
          f"({meta['embedding_parameters'] / 1e6:.2f}M embedding), forward {meta['forward']}, "
          f"processor {meta['processor']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
