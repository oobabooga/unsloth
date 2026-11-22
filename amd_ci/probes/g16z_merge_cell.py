"""AMD CI cell for unsloth-zoo #2655 (copy of scripts/g16_2655/merge_bytes.py): merged-save bytes per state.

Run once per arm in a fresh process with PYTHONPATH=<arm zoo tree>:
    python merge_bytes.py --tree <zoo tree> --out result.json [--cells a,b] [--workdir DIR]
Records per cell and per save (merged_16bit public entry point, GGUF staging merged_32bit): per-file sha256,
per-tensor sha256 / dtype / shape, raw safetensors header (key order + offsets), index.json, config.json (minus
transformers_version), generation_config.json, streamed-path engagement (head only), and that the live model is
unchanged afterwards.
"""
import argparse
import hashlib
import json
import os
import shutil
import struct
import sys
import time

os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("HIP_VISIBLE_DEVICES", "")

import torch


def _seed(s = 0):
    import random
    random.seed(s)
    torch.manual_seed(s)
    try:
        import numpy
        numpy.random.seed(s)
    except Exception:
        pass


def _llama(dtype = torch.float32, tie = False, vocab = 256, hidden = 64, layers = 2, inter = 128):
    import transformers
    cfg = transformers.LlamaConfig(hidden_size = hidden, intermediate_size = inter, num_hidden_layers = layers,
                                   num_attention_heads = 4, num_key_value_heads = 2, vocab_size = vocab,
                                   tie_word_embeddings = tie)
    _seed(0)
    return transformers.LlamaForCausalLM(cfg).to(dtype)


def _qwen3(dtype = torch.float32):
    import transformers
    cfg = transformers.Qwen3Config(hidden_size = 64, intermediate_size = 128, num_hidden_layers = 2,
                                   num_attention_heads = 4, num_key_value_heads = 2, vocab_size = 256, head_dim = 16,
                                   tie_word_embeddings = False)
    _seed(0)
    return transformers.Qwen3ForCausalLM(cfg).to(dtype)


def _qwen3moe():
    import transformers
    cfg = transformers.Qwen3MoeConfig(hidden_size = 64, intermediate_size = 128, moe_intermediate_size = 32,
                                      num_hidden_layers = 2, num_attention_heads = 4, num_key_value_heads = 2,
                                      vocab_size = 256, num_experts = 4, num_experts_per_tok = 2, head_dim = 16)
    _seed(0)
    return transformers.Qwen3MoeForCausalLM(cfg)


def _vlm():
    import transformers
    from transformers import LlavaConfig, LlavaForConditionalGeneration, CLIPVisionConfig, LlamaConfig
    text = LlamaConfig(hidden_size = 64, intermediate_size = 128, num_hidden_layers = 2, num_attention_heads = 4,
                       num_key_value_heads = 2, vocab_size = 300)
    vis = CLIPVisionConfig(hidden_size = 32, intermediate_size = 64, num_hidden_layers = 1, num_attention_heads = 2,
                           image_size = 28, patch_size = 14, projection_dim = 32)
    cfg = LlavaConfig(vision_config = vis, text_config = text, image_token_index = 299)
    _seed(0)
    return LlavaForConditionalGeneration(cfg)


def _lora(model, **kw):
    from peft import LoraConfig, get_peft_model
    _seed(1)
    kw.setdefault("target_modules", ["q_proj", "v_proj", "up_proj", "down_proj"])
    kw.setdefault("init_lora_weights", False)
    return get_peft_model(model, LoraConfig(r = 4, lora_alpha = 8, **kw))


def _multi():
    from peft import LoraConfig
    m = _lora(_llama(), target_modules = ["q_proj", "v_proj"])
    _seed(2)
    m.add_adapter("other", LoraConfig(r = 8, lora_alpha = 16, target_modules = ["q_proj", "k_proj", "o_proj"],
                                      init_lora_weights = False))
    return m  # active adapter stays "default"


def _multi_active_other():
    m = _multi()
    m.set_adapter("other")
    return m


def _frozen_bf16():
    from unsloth_zoo.cpu.ops import store_frozen_linears_in_bf16
    model = _llama(torch.bfloat16)
    store_frozen_linears_in_bf16(model)
    return _lora(model)


def _mixed():
    m = _llama(torch.bfloat16)
    m.model.norm.float()  # a keep-in-fp32 style module: bf16 + fp32 tensors in one file
    return _lora(m)


# name -> (builder, expected streamed engagement or None = depends on transformers, kwargs for save)
CELLS = {
    "llama_fp32":        (lambda: _lora(_llama()), True, {}),
    "llama_bf16":        (lambda: _lora(_llama(torch.bfloat16)), True, {}),
    "qwen3_fp32":        (lambda: _lora(_qwen3()), True, {}),
    "qwen3moe":          (lambda: _lora(_qwen3moe(), target_modules = ["q_proj", "v_proj"]), "tf4_only", {}),
    "tied_lora_qv":      (lambda: _lora(_llama(tie = True), target_modules = ["q_proj", "v_proj"]), True, {}),
    "tied_lora_lm_head": (lambda: _lora(_llama(tie = True), target_modules = ["q_proj", "lm_head"]), False, {}),
    "tied_lora_embed":   (lambda: _lora(_llama(tie = True), target_modules = ["q_proj", "embed_tokens"]), False, {}),
    "lora_embed":        (lambda: _lora(_llama(), target_modules = ["q_proj", "embed_tokens"]), False, {}),
    "lora_lm_head":      (lambda: _lora(_llama(), target_modules = ["q_proj", "lm_head"]), True, {}),
    "dora":              (lambda: _lora(_llama(), use_dora = True), False, {}),
    "rslora":            (lambda: _lora(_llama(), use_rslora = True), True, {}),
    "multi_inactive":    (_multi, True, {}),
    "multi_other_active": (_multi_active_other, True, {}),
    "frozen_bf16":       (_frozen_bf16, True, {}),
    "sharded_1MB":       (lambda: _lora(_llama(vocab = 4096, hidden = 128)), True, {"max_shard_size": "1MB"}),
    "sharded_int":       (lambda: _lora(_llama(vocab = 4096, hidden = 128)), True, {"max_shard_size": 300000}),
    "full_ft_fp32":      (lambda: _llama(), True, {}),
    "full_ft_bf16":      (lambda: _llama(torch.bfloat16), True, {}),
    "vlm_llava":         (lambda: _lora(_vlm(), target_modules = ["q_proj", "v_proj"]), "tf5_only", {}),
    "mixed_dtype":       (_mixed, True, {}),
    "safe_false":        (lambda: _lora(_llama()), False, {"safe_serialization": False}),
}


def _sha(b):
    return hashlib.sha256(b).hexdigest()


def _read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        raw = f.read(n)
    header = json.loads(raw, object_pairs_hook = lambda pairs: [list(p) for p in pairs])
    return {"raw_sha": _sha(raw), "raw_len": n, "key_order": [k for k, _ in header]}


def _record_dir(d):
    out = {"files": {}, "tensors": {}, "headers": {}, "index": None, "config": None, "generation_config": None}
    for name in sorted(os.listdir(d)):
        p = os.path.join(d, name)
        if not os.path.isfile(p):
            continue
        with open(p, "rb") as f:
            out["files"][name] = _sha(f.read())
        if name.endswith(".safetensors"):
            out["headers"][name] = _read_header(p)
            from safetensors import safe_open
            with safe_open(p, "pt") as f:
                for k in f.keys():
                    t = f.get_tensor(k)
                    out["tensors"][k] = {"file": name, "dtype": str(t.dtype), "shape": list(t.shape),
                                         "sha": _sha(t.contiguous().view(-1).view(torch.uint8).numpy().tobytes())}
        elif name.endswith(".bin"):
            sd = torch.load(p, map_location = "cpu", weights_only = True)
            for k, t in sd.items():
                out["tensors"][k] = {"file": name, "dtype": str(t.dtype), "shape": list(t.shape),
                                     "sha": _sha(t.contiguous().view(-1).view(torch.uint8).numpy().tobytes())}
        if name == "model.safetensors.index.json":
            out["index"] = json.load(open(p, encoding = "utf-8"))
        if name == "config.json":
            c = json.load(open(p, encoding = "utf-8"))
            c.pop("transformers_version", None)
            out["config"] = c
        if name == "generation_config.json":
            c = json.load(open(p, encoding = "utf-8"))
            c.pop("transformers_version", None)
            out["generation_config"] = c
    return out


def _state_hash(model):
    h = hashlib.sha256()
    for k, v in model.state_dict().items():
        h.update(k.encode())
        h.update(str(v.dtype).encode())
        h.update(v.detach().contiguous().view(-1).view(torch.uint8).numpy().tobytes())
    merged = []
    for m in model.modules():
        if getattr(m, "merged_adapters", None):
            merged.append(type(m).__name__)
    return h.hexdigest() + (":merged" if merged else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required = True)
    ap.add_argument("--out", required = True)
    ap.add_argument("--cells", default = "")
    ap.add_argument("--workdir", default = None)
    ap.add_argument("--keep", action = "store_true")
    a = ap.parse_args()
    import unsloth_zoo
    zoo_file = os.path.realpath(unsloth_zoo.__file__)
    tree = os.path.realpath(a.tree)
    own = os.path.normcase(zoo_file).startswith(os.path.normcase(tree) + os.sep)
    if not own:  # recorded and judged by the criteria gate; nothing is measured from a foreign tree
        with open(a.out, "w", encoding = "utf-8") as f:
            json.dump({"zoo_file": zoo_file, "tree": tree, "own_tree": False}, f)
        return
    import transformers
    import peft
    from unsloth_zoo.cpu import loader
    is_head = hasattr(loader, "_save_merged")
    engaged_log = []
    if is_head:
        orig = loader._stream_merged_save

        def traced(*args, **kw):
            r = orig(*args, **kw)
            engaged_log.append(bool(r))
            return r
        loader._stream_merged_save = traced

    def gguf_stage(model, d):
        if is_head:
            loader._save_merged(model, d, "merged_32bit")  # exactly what head save_pretrained_gguf calls
        else:
            staged, _ = loader._merged_copy(model, "merged_32bit")  # exactly what base save_pretrained_gguf does
            staged.save_pretrained(d)

    work = a.workdir or os.path.join(os.path.dirname(os.path.abspath(a.out)), "work_" + os.path.basename(a.out))
    result = {"zoo_file": zoo_file, "tree": tree, "own_tree": True, "arm": "head" if is_head else "base", "transformers": transformers.__version__,
              "peft": peft.__version__, "torch": torch.__version__, "cells": {}}
    names = [c for c in a.cells.split(",") if c] or list(CELLS)
    for name in names:
        build, expect, kw = CELLS[name]
        cell = {"expect_engaged": expect}
        try:
            model = build()
        except Exception as e:
            cell["build_error"] = repr(e)[:300]
            result["cells"][name] = cell
            continue
        before = _state_hash(model)
        for save in ("merged_16bit", "gguf_stage"):
            d = os.path.join(work, name, save)
            shutil.rmtree(d, ignore_errors = True)
            engaged_log.clear()
            t = time.time()
            try:
                if save == "merged_16bit":
                    loader.save_pretrained_merged(model, d, tokenizer = None, save_method = "merged_16bit", **kw)
                else:
                    if kw.get("safe_serialization") is False or kw.get("max_shard_size"):
                        continue  # GGUF staging takes no kwargs
                    gguf_stage(model, d)
                rec = _record_dir(d)
                rec["seconds"] = round(time.time() - t, 3)
                rec["engaged"] = engaged_log[-1] if engaged_log else None
                rec["live_unchanged"] = _state_hash(model) == before
            except Exception as e:
                import traceback
                rec = {"error": repr(e)[:500], "tb": traceback.format_exc()[-1500:]}
            cell[save] = rec
        result["cells"][name] = cell
        del model
        if not a.keep:
            shutil.rmtree(os.path.join(work, name), ignore_errors = True)
    try:
        import resource
        result["peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    except ImportError:  # Windows
        result["peak_rss_mb"] = None
    with open(a.out, "w", encoding = "utf-8") as f:
        json.dump(result, f, indent = 1)
    print("wrote", a.out, "cells", len(result["cells"]), "zoo", zoo_file)


if __name__ == "__main__":
    main()
