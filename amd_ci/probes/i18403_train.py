#!/usr/bin/env python3
"""Model build, route and training cells for unsloth #18403. Observes only; JSON to --out.

--build: seeded random-weight Qwen3 (hidden 1024, 2 layers, 8 q / 2 kv heads, head_dim 128, intermediate 2048,
  real Qwen3 tokenizer and vocab) saved to --model-dir. No unsloth import.
--mode route: FastLanguageModel fp16 (+ LoRA r=16, B=0 so outputs equal the base model), one packed row
  (segments 1300 + 748, padding-free form: position_ids reset, packed_seq_lengths) forward + backward in train
  mode with every torch SDPA call logged (backend torch picks for those exact args under the active policy,
  enable_gqa, mask, Q contiguity, shapes); then the same forward under sdpa_kernel([MATH]) as the reference.
  Records rocm_sdpa_verdict(0) when the tree has it.
--mode train: the reporter's settings on that model: fp16, LoRA r=16 / alpha 32 / dropout 0.1 on all 7
  projections, standard gradient checkpointing (True), batch 2, grad accum 1, packing False (Unsloth's padding-free
  default left on), ChatML rows of ~1200-2000 tokens, train_on_responses_only, --steps steps,
  adamw_8bit when --bnb 1 (and load_in_4bit), else adamw_torch / 16-bit LoRA.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import random
import sys
import traceback
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--model-dir", required = True, type = Path)
ap.add_argument("--build", action = "store_true")
ap.add_argument("--mode", choices = ("route", "train"), default = "train")
ap.add_argument("--bnb", type = int, default = 0)
ap.add_argument("--steps", type = int, default = 6)
ap.add_argument("--seed", type = int, default = 3407)
ap.add_argument("--work", type = Path, default = None)
ap.add_argument("--out", required = True, type = Path)
a = ap.parse_args()
res: dict = {"mode": "build" if a.build else a.mode, "bnb": a.bnb, "stage": "start",
             "env": {k: os.environ.get(k) for k in ("AMD_SERIALIZE_KERNEL", "UNSLOTH_ROCM_SDPA_CHECK",
                                                     "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "HIP_LAUNCH_BLOCKING")}}
SEGS = (1300, 748)
WORDS = ("river stone light paper green window table cloud music garden silver morning letter forest "
         "orange simple yellow bridge castle winter summer number little people market doctor travel "
         "kitchen engine planet circle animal yellow mirror pocket dinner button ticket island").split()


def dump():
    a.out.write_text(json.dumps(res, indent = 1, default = str), encoding = "utf-8")


def build():
    import torch
    import transformers
    from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(a.seed)
    cfg = Qwen3Config(hidden_size = 1024, intermediate_size = 2048, num_hidden_layers = 2, num_attention_heads = 8,
                      num_key_value_heads = 2, head_dim = 128, vocab_size = 151936, tie_word_embeddings = True,
                      rms_norm_eps = 1e-6, max_position_embeddings = 32768, rope_theta = 1000000,
                      eos_token_id = 151645, bos_token_id = 151643)
    model = Qwen3ForCausalLM(cfg).to(torch.bfloat16)
    model.save_pretrained(a.model_dir, safe_serialization = True)
    tok = AutoTokenizer.from_pretrained("unsloth/Qwen3-0.6B")
    tok.save_pretrained(a.model_dir)
    res.update({"stage": "built", "transformers": transformers.__version__,
                "params": sum(p.numel() for p in model.parameters())})


def common_import():
    import unsloth  # noqa: F401  (before transformers / trl)
    import torch
    import transformers
    import trl
    import peft
    import unsloth.utils.attention_dispatch as ad

    res["unsloth_file"] = unsloth.__file__
    res["ad_file"] = ad.__file__
    props = torch.cuda.get_device_properties(0)
    res["versions"] = {"torch": torch.__version__, "hip": torch.version.hip, "transformers": transformers.__version__,
                       "trl": trl.__version__, "peft": peft.__version__, "platform": sys.platform,
                       "arch": getattr(props, "gcnArchName", None)}
    try:
        import unsloth_zoo
        res["versions"]["unsloth_zoo"] = unsloth_zoo.__version__
    except Exception as e:  # noqa: BLE001
        res["versions"]["unsloth_zoo"] = f"ERR {e!r}"
    verdict = getattr(ad, "rocm_sdpa_verdict", None)
    res["has_verdict_api"] = verdict is not None
    if os.environ.get("I18403_FORCE_REFUSE") == "1" and verdict is not None:
        # Stand-in for gfx1200: the self-check child dies without a result, as a faulting kernel would.
        import unsloth.utils.rocm_sdpa_check as check
        check._CHILD = "import sys; sys.exit(3)"
        check._VERDICTS.clear()
        res["forced_refusal"] = True
    if verdict is not None:
        import time
        t0 = time.time()
        try:
            verdict(0)
        except Exception as e:  # noqa: BLE001
            res["verdict_error"] = repr(e)
        res["verdict_seconds"] = round(time.time() - t0, 2)
    res["sdpa_flags"] = {"flash": torch.backends.cuda.flash_sdp_enabled(),
                         "efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
                         "math": torch.backends.cuda.math_sdp_enabled()}
    res["stage"] = "imported"
    dump()
    return ad, verdict


def load_model(fp16_lora_dropout = 0.0, gc = True):
    import torch
    from unsloth import FastLanguageModel

    model, tok = FastLanguageModel.from_pretrained(str(a.model_dir), max_seq_length = 2048,
                                                   load_in_4bit = bool(a.bnb), dtype = torch.float16)
    model = FastLanguageModel.get_peft_model(
        model, r = 16, lora_alpha = 32, lora_dropout = fp16_lora_dropout, bias = "none",
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing = gc, random_state = a.seed)
    return model, tok


def route():
    import torch
    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel

    ad, verdict = common_import()
    if verdict is not None:
        try:
            res["verdict"] = verdict(0)
        except Exception as e:  # noqa: BLE001
            res["verdict"] = {"error": repr(e)}
        dump()
    model, tok = load_model()
    res["stage"] = "loaded"
    dump()
    calls: dict = {}
    original = F.scaled_dot_product_attention

    @torch._dynamo.disable
    def logged(query, key, value, attn_mask = None, dropout_p = 0.0, is_causal = False, scale = None, enable_gqa = False,
               **kw):
        try:
            choice = int(torch._fused_sdp_choice(query, key, value, attn_mask = attn_mask, dropout_p = dropout_p,
                                                 is_causal = is_causal, scale = scale, enable_gqa = enable_gqa))
            name = SDPBackend(choice).name
        except Exception as e:  # noqa: BLE001
            name = f"ERR {str(e)[:120]}"
        key_ = (f"{name}|gqa={bool(enable_gqa)}|mask={attn_mask is not None}|causal={bool(is_causal)}|"
                f"qcontig={query.is_contiguous()}|q={tuple(query.shape)}|k={tuple(key.shape)}|"
                f"grad={torch.is_grad_enabled() and query.requires_grad}|{query.dtype}")
        new = key_ not in calls
        calls[key_] = calls.get(key_, 0) + 1
        if new:
            res["route_calls"] = dict(calls)
            dump()  # a fault in the call below still leaves which backend it went to
        return original(query, key, value, attn_mask = attn_mask, dropout_p = dropout_p, is_causal = is_causal,
                        scale = scale, enable_gqa = enable_gqa, **kw)

    g = torch.Generator().manual_seed(a.seed)
    T = sum(SEGS)
    ids = torch.randint(1000, 100000, (1, T), generator = g).cuda()
    pos = torch.cat([torch.arange(L) for L in SEGS]).unsqueeze(0).cuda()
    psl = torch.tensor(SEGS, dtype = torch.int32).cuda()  # the Trainer moves the batch to the device
    model.train()

    def forward(tag):
        out = model(input_ids = ids, position_ids = pos, packed_seq_lengths = psl, labels = ids)
        logits = getattr(out, "logits", None)
        loss = out.loss
        loss.backward()
        torch.cuda.synchronize()
        res[f"{tag}_loss"] = float(loss)
        return None if logits is None or logits.numel() == 0 else logits.detach().float()

    F.scaled_dot_product_attention = logged
    try:
        res["stage"] = "default_forward"
        dump()
        got = forward("default")
    finally:
        F.scaled_dot_product_attention = original
    res["route_calls"] = dict(calls)
    res["stage"] = "math_forward"
    dump()
    model.zero_grad(set_to_none = True)
    with sdpa_kernel([SDPBackend.MATH]):
        ref = forward("math")
    if got is not None and ref is not None:
        fin = bool(torch.isfinite(got).all())
        mx = float((got - ref).abs().max()) if fin else float("inf")
        res["parity"] = {"finite": fin, "max_abs": mx, "max_ref": float(ref.abs().max()),
                         "rel": mx / max(float(ref.abs().max()), 1e-6)}
    else:
        res["parity"] = {"note": "logits not returned (UNSLOTH_RETURN_LOGITS off); loss compared only"}
    res["parity"]["loss_default"] = res.get("default_loss")
    res["parity"]["loss_math"] = res.get("math_loss")
    res["stage"] = "done"


def rows(tok, n, rng):
    out = []
    for _ in range(n):
        nu, na = rng.randint(600, 1050), rng.randint(450, 850)
        user = " ".join(rng.choice(WORDS) for _ in range(nu))
        asst = " ".join(rng.choice(WORDS) for _ in range(na))
        out.append("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                   f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n{asst}<|im_end|>\n")
    lens = [len(tok(t)["input_ids"]) for t in out[:8]]
    res["row_token_lengths_first8"] = lens
    return out


def train():
    import torch
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer
    from unsloth.chat_templates import train_on_responses_only

    ad, verdict = common_import()
    if verdict is not None:
        try:
            res["verdict"] = verdict(0)
        except Exception as e:  # noqa: BLE001
            res["verdict"] = {"error": repr(e)}
    model, tok = load_model(fp16_lora_dropout = 0.1, gc = True)
    res["stage"] = "loaded"
    dump()
    rng = random.Random(a.seed)
    ds = Dataset.from_dict({"text": rows(tok, 2 * (a.steps + 2), rng)})
    cfg = SFTConfig(output_dir = str(a.work / "out"), dataset_text_field = "text", max_length = 2048, packing = False,
                    per_device_train_batch_size = 2, gradient_accumulation_steps = 1, max_steps = a.steps,
                    learning_rate = 2e-4, logging_steps = 1, seed = a.seed, data_seed = a.seed, report_to = "none",
                    fp16 = True, bf16 = False, optim = "adamw_8bit" if a.bnb else "adamw_torch",
                    lr_scheduler_type = "linear", warmup_steps = 0, save_strategy = "no", dataloader_num_workers = 0,
                    dataset_num_proc = 1)
    trainer = SFTTrainer(model = model, processing_class = tok, train_dataset = ds, args = cfg)
    trainer = train_on_responses_only(trainer, instruction_part = "<|im_start|>user\n",
                                      response_part = "<|im_start|>assistant\n")
    res["collator"] = type(trainer.data_collator).__name__
    res["padding_free"] = getattr(trainer.args, "padding_free", None)
    res["stage"] = "training"
    dump()
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        trainer.train()
    torch.cuda.synchronize()
    hist = trainer.state.log_history
    res["losses"] = {int(h["step"]): float(h["loss"]) for h in hist if "loss" in h}
    res["grad_norms"] = {int(h["step"]): float(h["grad_norm"]) for h in hist if "grad_norm" in h}
    res["finite"] = bool(res["losses"]) and all(math.isfinite(v) for v in res["losses"].values())
    res["max_memory_allocated"] = torch.cuda.max_memory_allocated()
    res["train_stdout_tail"] = log.getvalue()[-1500:]
    res["stage"] = "done"


try:
    if a.build:
        build()
    elif a.mode == "route":
        route()
    else:
        train()
except BaseException as e:  # noqa: BLE001
    res["error"] = f"{type(e).__name__}: {str(e)[:1500]}"
    res["traceback"] = traceback.format_exc()[-4000:]
dump()
