#!/usr/bin/env python3
"""One arm of the PR 13383 probe: tiny packed FastModel LoRA SFT on a Qwen3.5 gated-delta hybrid.

PYTHONPATH puts the state's unsloth checkout first. Observes only, JSON to --out:
  * which unsloth was imported, versions, device / arch
  * UNSLOTH_EXPERIMENTAL_HYBRID_PACKING as seen by this arm
  * whether the trainer ended up packing: args.packing / args.padding_free, the collator's
    padding_free, the varlen forward wrapper on the model, packed batches seen by the forward
  * unsloth.utils.packing.HYBRID_PACKING_STATS (head only; absent at the base)
  * every `unsloth.hybrid_packing` warning (rejection reasons) and Unsloth's "packing=True ignored"
  * per-step loss and grad_norm

The tiny model is built once per --model-dir from a fixed seed on CPU (identical bytes for every
state and arm): Qwen3_5TextConfig, random weights, tokenizer from
trl-internal-testing/tiny-MistralForCausalLM-0.1. If the installed transformers has no Qwen3.5
the arm records `skip` with the reason and exits 0 (criteria turn that into INCONCLUSIVE).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import traceback
from pathlib import Path

TOKENIZER = "trl-internal-testing/tiny-MistralForCausalLM-0.1"
ENV = "UNSLOTH_EXPERIMENTAL_HYBRID_PACKING"
WORDS = ("the of and to in is was for on as with by he at from his an were are which this be "
         "or has had not but first one their its new after who they have two also been other more all "
         "she time year can city world river music house water light green small large north south "
         "during between under over").split()


class _Capture(logging.Handler):
    def __init__(self):
        super().__init__(level = logging.DEBUG)
        self.records = []

    def emit(self, record):
        try:
            self.records.append(f"{record.name}: {record.getMessage()}"[:600])
        except Exception:  # noqa: BLE001
            pass


def build_model(model_dir: Path) -> None:
    if (model_dir / "config.json").is_file():
        return
    import torch
    import transformers
    tmp = model_dir.with_name(model_dir.name + f".tmp{os.getpid()}")
    tok = transformers.AutoTokenizer.from_pretrained(TOKENIZER)
    torch.manual_seed(0)
    cfg = transformers.Qwen3_5TextConfig(
        vocab_size = len(tok), hidden_size = 128, intermediate_size = 256, num_hidden_layers = 4,
        num_attention_heads = 4, num_key_value_heads = 2, head_dim = 32,
        linear_num_key_heads = 2, linear_num_value_heads = 4, linear_key_head_dim = 32,
        linear_value_head_dim = 32, linear_conv_kernel_dim = 4,
        layer_types = ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        max_position_embeddings = 4096, bos_token_id = tok.bos_token_id, eos_token_id = tok.eos_token_id,
        pad_token_id = tok.eos_token_id, tie_word_embeddings = False,
    )
    cfg.architectures = ["Qwen3_5ForCausalLM"]
    model = transformers.Qwen3_5ForCausalLM(cfg).to(torch.bfloat16)
    model.save_pretrained(tmp)
    tok.save_pretrained(tmp)
    try:
        tmp.rename(model_dir)
    except OSError:
        if not (model_dir / "config.json").is_file():
            raise


def make_docs(n = 96, seed = 0):
    rng = random.Random(seed)
    return [" ".join(rng.choice(WORDS) for _ in range(rng.choice([8, 15, 25, 40, 60, 90, 130]))) + "."
            for _ in range(n)]


def _varlen_wrapped(model) -> dict:
    fwd = 0
    mixers = 0
    for m in model.modules():
        if getattr(m, "_unsloth_varlen_forward_wrapped", False):
            fwd += 1
        if getattr(m, "_unsloth_varlen_wrapped", False):
            mixers += 1
    return {"forward_wrapped_modules": fwd, "mixers_wrapped": mixers}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--model-dir", required = True, type = Path)
    ap.add_argument("--steps", type = int, default = 8)
    args = ap.parse_args()
    res: dict = {"stage": "build", "env_hybrid_packing": os.environ.get(ENV, "<unset>")}

    def dump():
        args.out.write_text(json.dumps(res, indent = 2, default = str), encoding = "utf-8")

    cap = _Capture()
    try:
        import transformers as _t
        if not (hasattr(_t, "Qwen3_5TextConfig") and hasattr(_t, "Qwen3_5ForCausalLM")):
            res["skip"] = f"transformers {_t.__version__} has no Qwen3_5TextConfig / Qwen3_5ForCausalLM"
            res["stage"] = "skip"
            dump()
            return 0
        build_model(args.model_dir)
        res["stage"] = "import"
        dump()
        import unsloth
        from unsloth import FastModel
        import torch
        import transformers
        import trl
        from datasets import Dataset
        from trl import SFTConfig, SFTTrainer
        for name in ("unsloth.hybrid_packing", "unsloth.trainer", "unsloth", "unsloth_zoo"):
            lg = logging.getLogger(name)
            lg.addHandler(cap)
            if lg.level == logging.NOTSET or lg.level > logging.WARNING:
                lg.setLevel(logging.WARNING)
        try:
            import unsloth.trainer as _ut
            lg = getattr(_ut, "logger", None)
            if isinstance(lg, logging.Logger) and cap not in lg.handlers:
                lg.addHandler(cap)
        except Exception:  # noqa: BLE001
            pass
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
        import importlib.util as _iu
        res["kernel_pkgs"] = {k: _iu.find_spec(k) is not None for k in ("fla", "causal_conv1d")}
        p = torch.cuda.get_device_properties(0)
        res["versions"]["device"] = p.name
        res["versions"]["arch"] = getattr(p, "gcnArchName", None)

        import unsloth.utils.packing as pk
        stats = getattr(pk, "HYBRID_PACKING_STATS", None)
        res["has_hybrid_stats"] = stats is not None
        res["has_hybrid_mode"] = hasattr(pk, "_hybrid_packing_mode")
        if hasattr(pk, "_hybrid_packing_mode"):
            res["hybrid_mode"] = pk._hybrid_packing_mode()

        res["stage"] = "load"
        dump()
        model, tok = FastModel.from_pretrained(str(args.model_dir), max_seq_length = 512, dtype = torch.bfloat16,
                                               load_in_4bit = False, load_in_16bit = True)
        res["model_class"] = type(model).__name__
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
        if stats is not None:
            for k in stats:
                stats[k] = None if k == "last_probe_rel_err" else 0
        trainer = SFTTrainer(model = model, train_dataset = ds, processing_class = tt, args = cfg)
        coll = trainer.data_collator
        res["collator"] = type(coll).__name__
        res["collator_padding_free"] = getattr(coll, "padding_free", None)
        res["args_packing"] = getattr(trainer.args, "packing", None)
        res["args_padding_free"] = getattr(trainer.args, "padding_free", None)
        res["varlen_wrapped_after_init"] = _varlen_wrapped(trainer.model)

        # Count forwards that received packed metadata (position_ids restarting / packed_seq_lengths).
        packed_batches = {"n": 0, "packed": 0}
        orig_cs = trainer.compute_loss

        def compute_loss(model_, inputs, *a, **k):
            packed_batches["n"] += 1
            pos = inputs.get("position_ids") if hasattr(inputs, "get") else None
            if inputs.get("packed_seq_lengths") is not None or (
                    pos is not None and pos.shape[0] == 1 and int((pos[0] == 0).sum()) > 1):
                packed_batches["packed"] += 1
            return orig_cs(model_, inputs, *a, **k)

        trainer.compute_loss = compute_loss
        trainer.train()
        res["batches"] = packed_batches
        res["stats"] = dict(stats) if stats is not None else None
        logs = [h for h in trainer.state.log_history if "loss" in h]
        res["losses"] = [float(h["loss"]) for h in logs]
        res["grad_norms"] = [float(h["grad_norm"]) if h.get("grad_norm") is not None else None for h in logs]
        res["stage"] = "done"
        res["ok"] = True
    except Exception as e:  # noqa: BLE001
        res["error"] = f"{type(e).__name__}: {e}"[:1500]
        res["traceback"] = traceback.format_exc()[-4000:]
    res["log_records"] = cap.records[-50:]
    res["hybrid_rejections"] = [r for r in cap.records if "hybrid" in r.lower()]
    res["packing_ignored"] = [r for r in cap.records if "packing=True ignored" in r]
    dump()
    return 0


if __name__ == "__main__":
    sys.exit(main())
