#!/usr/bin/env python3
"""One arm of the PR 13384 probe: tiny packed FastModel LoRA SFT on a 2-layer Gemma-4 text model.

Layers: one sliding (head_dim 256, window 32), one global (global_head_dim 512). The global layer is
the shape PR 13384 routes to aten._efficient_attention_* over cu_seqlens (gated off on ROCm).

PYTHONPATH puts the state's unsloth checkout first. Observes only, JSON to --out:
  * which unsloth / unsloth_zoo was imported, versions, device / arch
  * HF_PACKED_ATTENTION_STATS (fast / fallback / mask_checks / efficient) after training
  * per-step loss and grad_norm from the trainer log

The model is built once per --model-dir from a fixed seed on CPU (identical bytes for every state
and arm). Tokenizer from trl-internal-testing/tiny-MistralForCausalLM-0.1, as pr13326_run.
If the installed transformers has no Gemma 4, records stage=unsupported with the reason.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import traceback
from pathlib import Path

TOKENIZER = "trl-internal-testing/tiny-MistralForCausalLM-0.1"
WINDOW = 32
WORDS = ("the of and to in is was for on as with by he at from his an were are which this be "
         "or has had not but first one their its new after who they have two also been other more all "
         "she time year can city world river music house water light green small large north south "
         "during between under over").split()


def build_model(model_dir: Path) -> None:
    if (model_dir / "config.json").is_file():
        return
    import torch
    import transformers
    tmp = model_dir.with_name(model_dir.name + ".tmp")
    tok = transformers.AutoTokenizer.from_pretrained(TOKENIZER)
    torch.manual_seed(0)
    cfg = transformers.Gemma4TextConfig(
        vocab_size = len(tok), vocab_size_per_layer_input = len(tok), hidden_size = 256,
        intermediate_size = 512, num_hidden_layers = 2, num_attention_heads = 4, num_key_value_heads = 2,
        head_dim = 256, global_head_dim = 512, sliding_window = WINDOW,
        layer_types = ["sliding_attention", "full_attention"], hidden_size_per_layer_input = 0,
        max_position_embeddings = 4096, bos_token_id = tok.bos_token_id, eos_token_id = tok.eos_token_id,
        pad_token_id = tok.eos_token_id, tie_word_embeddings = True,
    )
    cfg.architectures = ["Gemma4ForCausalLM"]
    model = transformers.Gemma4ForCausalLM(cfg).to(torch.bfloat16)
    model.save_pretrained(tmp)
    tok.save_pretrained(tmp)
    tmp.rename(model_dir)


def make_docs(n = 96, seed = 0):
    rng = random.Random(seed)
    return [" ".join(rng.choice(WORDS) for _ in range(rng.choice([8, 15, 25, 40, 60, 90, 130]))) + "."
            for _ in range(n)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--model-dir", required = True, type = Path)
    ap.add_argument("--steps", type = int, default = 8)
    args = ap.parse_args()
    res: dict = {"stage": "probe-gemma4",
                 "env": {k: os.environ.get(k, "<unset>") for k in (
                     "UNSLOTH_HF_PACKED_EFFICIENT", "UNSLOTH_HF_PACKED_VARLEN", "UNSLOTH_COMPILE_DISABLE",
                     "UNSLOTH_FLASH_ATTENTION_IMPL")}}

    def dump():
        args.out.write_text(json.dumps(res, indent = 2, default = str), encoding = "utf-8")

    try:
        import transformers
        res["transformers"] = transformers.__version__
        if not (hasattr(transformers, "Gemma4TextConfig") and hasattr(transformers, "Gemma4ForCausalLM")):
            res["stage"] = "unsupported"
            res["unsupported"] = f"transformers {transformers.__version__} has no Gemma4TextConfig / Gemma4ForCausalLM"
            dump()
            return 0
        res["stage"] = "build"
        dump()
        build_model(args.model_dir)
        res["stage"] = "import"
        dump()
        import unsloth
        from unsloth import FastModel
        import torch
        import trl
        from datasets import Dataset
        from trl import SFTConfig, SFTTrainer
        res["unsloth_file"] = unsloth.__file__
        res["versions"] = {"torch": torch.__version__, "hip": torch.version.hip, "cuda": torch.version.cuda,
                           "transformers": transformers.__version__, "trl": trl.__version__,
                           "unsloth": getattr(unsloth, "__version__", None)}
        try:
            import unsloth_zoo
            res["versions"]["unsloth_zoo"] = unsloth_zoo.__version__
            res["unsloth_zoo_file"] = unsloth_zoo.__file__
        except Exception as e:  # noqa: BLE001
            res["versions"]["unsloth_zoo"] = repr(e)
        p = torch.cuda.get_device_properties(0)
        res["versions"]["device"] = p.name
        res["versions"]["arch"] = getattr(p, "gcnArchName", None)
        import unsloth.utils.hf_packed_attention as hpa

        res["stage"] = "load"
        dump()
        model, tok = FastModel.from_pretrained(str(args.model_dir), max_seq_length = 512, dtype = torch.bfloat16,
                                               load_in_4bit = False, load_in_16bit = True)
        res["model_class"] = type(model).__name__
        tc = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
        def _get(c, k):
            try:  # transformers 5.1x raises on per-layer attributes (head_dim) read globally
                return getattr(c, k, None)
            except Exception as e:  # noqa: BLE001
                return f"<{type(e).__name__}>"

        res["config"] = {k: _get(tc, k) for k in ("model_type", "layer_types", "head_dim", "global_head_dim",
                                                   "per_layer_config", "sliding_window")}
        res["config"]["attn_impl"] = _get(model.config, "_attn_implementation")
        # Head dims the attention modules actually run (the fact the efficient route keys on).
        res["attn_head_dims"] = sorted({int(m.head_dim) for m in model.modules()
                                        if type(m).__name__.endswith("Attention") and hasattr(m, "head_dim")})
        model = FastModel.get_peft_model(model, r = 8, lora_alpha = 16, lora_dropout = 0.0, random_state = 0,
                                         target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                                                           "gate_proj", "up_proj", "down_proj"],
                                         use_gradient_checkpointing = "unsloth")
        tt = getattr(tok, "tokenizer", tok)
        if tt.pad_token is None:
            tt.pad_token = tt.eos_token
        ds = Dataset.from_dict({"text": [d + (tt.eos_token or "") for d in make_docs()]})
        cfg = SFTConfig(output_dir = str(args.out.parent / (args.out.stem + "_trainer")),
                        per_device_train_batch_size = 4, gradient_accumulation_steps = 1, max_steps = args.steps,
                        learning_rate = 1e-3, logging_steps = 1, bf16 = True, report_to = "none", seed = 0,
                        dataset_text_field = "text", max_length = 512, save_strategy = "no",
                        dataset_num_proc = 1, warmup_steps = 0, packing = True)
        res["stage"] = "train"
        dump()
        trainer = SFTTrainer(model = model, train_dataset = ds, processing_class = tt, args = cfg)
        res["collator"] = type(trainer.data_collator).__name__
        res["args_packing"] = getattr(trainer.args, "packing", None)
        res["args_padding_free"] = getattr(trainer.args, "padding_free", None)
        for k in hpa.HF_PACKED_ATTENTION_STATS:
            hpa.HF_PACKED_ATTENTION_STATS[k] = 0
        trainer.train()
        sdpa = transformers.modeling_utils.ALL_ATTENTION_FUNCTIONS.get("sdpa")
        res["sdpa_entry"] = f"{getattr(sdpa, '__module__', '?')}.{getattr(sdpa, '__name__', '?')}"
        res["sdpa_wrapped"] = bool(getattr(sdpa, getattr(hpa, "_FLAG", "<none>"), False))
        res["stats"] = dict(hpa.HF_PACKED_ATTENTION_STATS)
        logs = [h for h in trainer.state.log_history if "loss" in h]
        res["losses"] = [float(h["loss"]) for h in logs]
        res["grad_norms"] = [float(h["grad_norm"]) if h.get("grad_norm") is not None else None for h in logs]
        res["stage"] = "done"
        res["ok"] = True
    except Exception as e:  # noqa: BLE001
        res["error"] = f"{type(e).__name__}: {e}"[:1500]
        res["traceback"] = traceback.format_exc()[-4000:]
    dump()
    return 0


if __name__ == "__main__":
    sys.exit(main())
