#!/usr/bin/env python3
"""Training cell for unsloth #18313. Observes only; JSON to --out.

--build DIR: write a random-weight model (seeded) + tokenizer to DIR and exit (no unsloth import), so every arm
of both states trains the very same weights.
  qwen35: Qwen3.5-4B text layer sizes, 4 layers (3 gated delta + 1 full attention, head dim 256, 16 q / 4 kv heads)
  gemma3: Gemma-3-1B text layer sizes, 4 layers (3 sliding window 512 + 1 full, head dim 256, 4 q / 1 kv heads),
          the transformers modeling path with windows (qwen3 is kept but Unsloth runs its own attention there)
Otherwise: FastLanguageModel LoRA r=16 (all 7 projections), gradient checkpointing "unsloth", SFTTrainer
batch 4, max_length 2048, --steps steps, seed 3407, pre-tokenized random documents of 64..1800 tokens,
packing on (--packed 1, padding-free BFD) or off (padded: padding_free=False explicitly, so Unsloth's auto
padding-free does not pack it). Per step: loss, grad norm, wall seconds (synchronised
at step begin / end), non-padding tokens fed to the model. HF_PACKED_ATTENTION_STATS before / after.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import random
import sys
import time
import traceback
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--arch", choices = ("qwen35", "qwen3", "gemma3"), required = True)
ap.add_argument("--model-dir", required = True, type = Path)
ap.add_argument("--build", action = "store_true")
ap.add_argument("--small", action = "store_true", help = "shrunk widths for a harness smoke")
ap.add_argument("--packed", type = int, default = 0)
ap.add_argument("--steps", type = int, default = 12)
ap.add_argument("--seed", type = int, default = 3407)
ap.add_argument("--bsz", type = int, default = 4)
ap.add_argument("--max-length", type = int, default = 2048)
ap.add_argument("--work", type = Path, default = None)
ap.add_argument("--out", required = True, type = Path)
a = ap.parse_args()
res: dict = {"arch": a.arch, "packed": a.packed, "stage": "start"}


def dump():
    a.out.write_text(json.dumps(res, indent = 1, default = str), encoding = "utf-8")


TOKENIZER = {"qwen35": "Qwen/Qwen3.5-4B", "qwen3": "Qwen/Qwen3-4B", "gemma3": "unsloth/gemma-3-1b-it"}


def build():
    import torch
    import transformers
    from transformers import AutoTokenizer

    torch.manual_seed(a.seed)
    if a.arch == "qwen35":
        from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
        cfg = Qwen3_5TextConfig(
            hidden_size = 2560, intermediate_size = 9216, num_hidden_layers = 4, num_attention_heads = 16,
            num_key_value_heads = 4, head_dim = 256, vocab_size = 248320, tie_word_embeddings = True,
            layer_types = ["linear_attention"] * 3 + ["full_attention"], full_attention_interval = 4,
            linear_conv_kernel_dim = 4, linear_key_head_dim = 128, linear_num_key_heads = 16,
            linear_num_value_heads = 32, linear_value_head_dim = 128, attn_output_gate = True,
            rms_norm_eps = 1e-6, max_position_embeddings = 32768, eos_token_id = 248044,
            rope_parameters = {"rope_type": "default", "rope_theta": 10000000, "partial_rotary_factor": 0.25,
                               "mrope_interleaved": True, "mrope_section": [11, 11, 10]},
        )
        if a.small:
            cfg.hidden_size, cfg.intermediate_size, cfg.vocab_size = 512, 1024, 248320
            cfg.linear_num_key_heads, cfg.linear_num_value_heads = 4, 8
        model = Qwen3_5ForCausalLM(cfg)
    elif a.arch == "gemma3":
        from transformers import Gemma3ForCausalLM, Gemma3TextConfig
        cfg = Gemma3TextConfig(
            hidden_size = 1152, intermediate_size = 6912, num_hidden_layers = 4, num_attention_heads = 4,
            num_key_value_heads = 1, head_dim = 256, vocab_size = 262144, sliding_window = 512,
            layer_types = ["sliding_attention"] * 3 + ["full_attention"], query_pre_attn_scalar = 256,
            max_position_embeddings = 32768, rms_norm_eps = 1e-6, tie_word_embeddings = True,
        )
        if a.small:
            cfg.hidden_size, cfg.intermediate_size = 512, 1024
        model = Gemma3ForCausalLM(cfg)
    else:
        from transformers import Qwen3Config, Qwen3ForCausalLM
        cfg = Qwen3Config(
            hidden_size = 2560, intermediate_size = 9728, num_hidden_layers = 4, num_attention_heads = 32,
            num_key_value_heads = 8, head_dim = 128, vocab_size = 151936, tie_word_embeddings = True,
            rms_norm_eps = 1e-6, max_position_embeddings = 32768, rope_theta = 1000000, eos_token_id = 151645,
        )
        if a.small:
            cfg.hidden_size, cfg.intermediate_size = 512, 1024
        model = Qwen3ForCausalLM(cfg)
    model = model.to(torch.bfloat16)
    model.save_pretrained(a.model_dir, safe_serialization = True)
    tok = AutoTokenizer.from_pretrained(TOKENIZER[a.arch])
    tok.save_pretrained(a.model_dir)
    res.update({"stage": "built", "transformers": transformers.__version__,
                "params": sum(p.numel() for p in model.parameters())})


def train():
    import unsloth  # noqa: F401  (before transformers / trl)
    from unsloth import FastLanguageModel
    import torch
    import transformers
    import trl
    import peft
    from datasets import Dataset
    from transformers import TrainerCallback
    from trl import SFTConfig, SFTTrainer
    from unsloth.utils import hf_packed_attention as hpa

    res["unsloth_file"] = unsloth.__file__
    res["hpa_file"] = hpa.__file__
    v = {"torch": torch.__version__, "hip": torch.version.hip, "transformers": transformers.__version__,
         "trl": trl.__version__, "peft": peft.__version__, "platform": sys.platform}
    try:
        import unsloth_zoo
        v["unsloth_zoo"] = unsloth_zoo.__version__
        props = torch.cuda.get_device_properties(0)
        v["arch"] = getattr(props, "gcnArchName", None) or f"sm{props.major}{props.minor}"
    except Exception as e:  # noqa: BLE001
        v["arch"] = f"ERR {e}"
    res["versions"] = v
    res["stage"] = "imported"
    dump()
    model, tok = FastLanguageModel.from_pretrained(str(a.model_dir), max_seq_length = a.max_length,
                                                   load_in_4bit = False, dtype = torch.bfloat16)
    model = FastLanguageModel.get_peft_model(
        model, r = 16, lora_alpha = 16, lora_dropout = 0, bias = "none",
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing = "unsloth", random_state = a.seed)
    res["attn_impl"] = getattr(model.config, "_attn_implementation", None)
    res["stage"] = "loaded"
    dump()
    rng = random.Random(a.seed)
    vocab = 100000
    rows = []
    for _ in range(a.bsz * (a.steps + 4) * 3):
        n = rng.randint(64, 1800)
        ids = [rng.randrange(1000, vocab) for _ in range(n)]
        rows.append({"input_ids": ids})
    ds = Dataset.from_list(rows)
    cfg = SFTConfig(output_dir = str(a.work / "out"), max_steps = a.steps, per_device_train_batch_size = a.bsz,
                    gradient_accumulation_steps = 1, learning_rate = 2e-4, logging_steps = 1, seed = a.seed,
                    data_seed = a.seed, report_to = "none", max_length = a.max_length, optim = "adamw_torch",
                    lr_scheduler_type = "linear", warmup_steps = 1, weight_decay = 0.0, bf16 = True, fp16 = False,
                    save_strategy = "no", packing = bool(a.packed), dataloader_num_workers = 0,
                    **({} if a.packed else {"padding_free": False}))
    trainer = SFTTrainer(model = model, processing_class = tok, train_dataset = ds, args = cfg)
    res["collator"] = type(trainer.data_collator).__name__
    res["padding_free"] = getattr(trainer.args, "padding_free", None)
    tokens, times = [], []

    def pre_hook(module, args, kwargs):
        am = kwargs.get("attention_mask")
        ids = kwargs.get("input_ids")
        if ids is None and args:
            ids = args[0]
        if module.training and ids is not None:
            labels = kwargs.get("labels")
            if isinstance(am, torch.Tensor) and am.dim() == 2:
                tokens.append([int(am.sum()), int(ids.numel()), list(ids.shape)])
            else:
                tokens.append([int(ids.numel()), int(ids.numel()), list(ids.shape)])
        return None

    hook = trainer.model.register_forward_pre_hook(pre_hook, with_kwargs = True)

    class Timer(TrainerCallback):
        t0 = None

        def on_step_begin(self, args, state, control, **kw):
            torch.cuda.synchronize()
            self.t0 = time.perf_counter()

        def on_step_end(self, args, state, control, **kw):
            torch.cuda.synchronize()
            times.append(time.perf_counter() - self.t0)

    trainer.add_callback(Timer())
    stats0 = dict(hpa.HF_PACKED_ATTENTION_STATS)
    res["stage"] = "training"
    dump()
    torch.cuda.reset_peak_memory_stats()
    log = io.StringIO()
    t = time.perf_counter()
    with contextlib.redirect_stdout(log):
        trainer.train()
    res["train_s"] = time.perf_counter() - t
    hook.remove()
    stats1 = dict(hpa.HF_PACKED_ATTENTION_STATS)
    res["attn_stats_delta"] = {k: stats1[k] - stats0.get(k, 0) for k in stats1}
    try:
        from unsloth.utils.packing import HYBRID_PACKING_STATS
        res["hybrid_stats"] = dict(HYBRID_PACKING_STATS)
    except Exception as e:  # noqa: BLE001
        res["hybrid_stats"] = repr(e)
    res["varlen_route_cache"] = {str(k): v for k, v in hpa._TORCH_VARLEN_DEVICES.items()}
    res["window_cache"] = {str(k): v for k, v in getattr(hpa, "_HIP_WINDOW_OK", {}).items()}
    res["max_memory_allocated"] = torch.cuda.max_memory_allocated()
    hist = trainer.state.log_history
    res["losses"] = {int(h["step"]): float(h["loss"]) for h in hist if "loss" in h}
    res["grad_norms"] = {int(h["step"]): float(h["grad_norm"]) for h in hist if "grad_norm" in h}
    res["step_s"] = times
    res["tokens"] = tokens
    res["train_stdout_tail"] = log.getvalue()[-1500:]
    res["stage"] = "done"


try:
    if a.build:
        build()
    else:
        train()
except BaseException as e:  # noqa: BLE001
    res["error"] = f"{type(e).__name__}: {str(e)[:1500]}"
    res["traceback"] = traceback.format_exc()[-4000:]
dump()
