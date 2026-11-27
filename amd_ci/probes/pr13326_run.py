#!/usr/bin/env python3
"""One arm of the PR 13326 probe: tiny packed FastModel LoRA SFT on a sliding-window Mistral.

PYTHONPATH puts the state's unsloth checkout first. Observes only, JSON to --out:
  * which unsloth was imported, versions, device / arch
  * HF_PACKED_ATTENTION_STATS (fast / fallback / mask_checks) after training
  * the varlen kernels this build could offer: torch varlen (hf wrapper's _torch_varlen),
    select_attention_backend(use_varlen=True), HAS_FLASH_ATTENTION / HAS_XFORMERS
  * flash_impl state (head only: FLASH_IMPL_STATE, requested env)
  * per-step loss and grad_norm from the trainer log

The tiny model is built once per --model-dir from a fixed seed on CPU (identical bytes for
every state), tokenizer from trl-internal-testing/tiny-MistralForCausalLM-0.1.
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
WORDS = ("the of and to in is was for on as with by he at from his an were are which this be "
         "or has had not but first one their its new after who they have two also been other more all "
         "she time year can city world river music house water light green small large north south "
         "during between under over").split()


def build_model(model_dir: Path, window: int | None) -> None:
    if (model_dir / "config.json").is_file():
        return
    import torch
    import transformers
    tmp = model_dir.with_name(model_dir.name + ".tmp")
    tok = transformers.AutoTokenizer.from_pretrained(TOKENIZER)
    torch.manual_seed(0)
    cfg = transformers.MistralConfig(
        vocab_size = len(tok), hidden_size = 256, intermediate_size = 512, num_hidden_layers = 2,
        num_attention_heads = 4, num_key_value_heads = 2, head_dim = 64, sliding_window = window,
        max_position_embeddings = 4096, bos_token_id = tok.bos_token_id, eos_token_id = tok.eos_token_id,
        tie_word_embeddings = False,
    )
    model = transformers.MistralForCausalLM(cfg).to(torch.bfloat16)
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
    ap.add_argument("--window", type = int, default = 0, help = "0 = no sliding window")
    ap.add_argument("--steps", type = int, default = 8)
    args = ap.parse_args()
    window = args.window or None
    res: dict = {"stage": "build", "window": window, "env_varlen": os.environ.get("UNSLOTH_HF_PACKED_VARLEN", "<unset>"),
                 "env_flash_impl": os.environ.get("UNSLOTH_FLASH_ATTENTION_IMPL", "<unset>")}

    def dump():
        args.out.write_text(json.dumps(res, indent = 2, default = str), encoding = "utf-8")

    try:
        build_model(args.model_dir, window)
        res["stage"] = "import"
        dump()
        import unsloth
        from unsloth import FastModel
        import torch
        import transformers
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
        except Exception as e:  # noqa: BLE001
            res["versions"]["unsloth_zoo"] = repr(e)
        p = torch.cuda.get_device_properties(0)
        res["versions"]["device"] = p.name
        res["versions"]["arch"] = getattr(p, "gcnArchName", None)

        import unsloth.utils.hf_packed_attention as hpa
        import unsloth.utils.attention_dispatch as ad
        dev = torch.device("cuda", 0)
        res["kernels"] = {
            "hpa_torch_varlen": repr(hpa._torch_varlen(dev)),
            "dispatch_TORCH_VARLEN_ATTN": repr(getattr(ad, "_TORCH_VARLEN_ATTN", None)),
            "select_attention_backend_varlen": ad.select_attention_backend(use_varlen = True),
            "HAS_FLASH_ATTENTION": getattr(ad, "HAS_FLASH_ATTENTION", None),
            "HAS_XFORMERS": getattr(ad, "HAS_XFORMERS", None),
        }
        try:
            from unsloth.utils import flash_impl as fi
            res["flash_impl_module"] = True
            res["flash_impl_allows_hd64_window"] = fi.flash_impl_allows(64, window, dev)
        except ImportError:
            fi = None
            res["flash_impl_module"] = False

        res["stage"] = "load"
        dump()
        model, tok = FastModel.from_pretrained(str(args.model_dir), max_seq_length = 512, dtype = torch.bfloat16,
                                               load_in_4bit = False, load_in_16bit = True)
        res["model_class"] = type(model).__name__
        res["config_sliding_window"] = getattr(model.config, "sliding_window", None)
        res["attn_impl"] = getattr(model.config, "_attn_implementation", None)
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
        res["sdpa_wrapped"] = bool(getattr(sdpa, hpa._FLAG, False))
        res["stats"] = dict(hpa.HF_PACKED_ATTENTION_STATS)
        logs = [h for h in trainer.state.log_history if "loss" in h]
        res["losses"] = [float(h["loss"]) for h in logs]
        res["grad_norms"] = [float(h["grad_norm"]) if h.get("grad_norm") is not None else None for h in logs]
        if fi is not None:
            st = fi.FLASH_IMPL_STATE
            res["flash_impl_state"] = {"active": st.get("active"), "reason": st.get("reason"),
                                       "verdicts": {str(k): str(v) for k, v in (st.get("verdicts") or {}).items()}}
        res["stage"] = "done"
        res["ok"] = True
    except Exception as e:  # noqa: BLE001
        res["error"] = f"{type(e).__name__}: {e}"[:1500]
        res["traceback"] = traceback.format_exc()[-4000:]
    dump()
    return 0


if __name__ == "__main__":
    sys.exit(main())
