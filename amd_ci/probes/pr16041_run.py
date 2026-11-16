#!/usr/bin/env python3
"""One process of the PR 16041 probe (Gemma 3 generate off flex_attention). Observes only.

PYTHONPATH puts the state's unsloth checkout first. Writes JSON to --out after every stage:
  * which unsloth was imported, versions, device / arch
  * every decoder attention config's _attn_implementation right after from_pretrained
  * greedy generate of 128 new tokens: --warmup untimed calls, then --repeats timed calls (ms/token)
  * the attn impl each attention module read DURING generate (forward pre-hook on module.config) and after
  * one short untimed generate with counters on the attention dispatch (flex vs sdpa functions)
  * whether _NonFlexGenerate engaged (head only; absent at the base)
  * get_peft_model LoRA r8 + --steps SFT steps on a fixed tiny dataset: losses, attn impl read during
    training, configs after training
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import random
import statistics
import sys
import time
import traceback
from pathlib import Path

MODEL = "unsloth/gemma-3-270m-it"
NEW_TOKENS = 128
WORDS = ("the of and to in is was for on as with by he at from his an were are which this be "
         "or has had not but first one their its new after who they have two also been other more all "
         "she time year can city world river music house water light green small large north south").split()


def make_docs(n = 32, seed = 0):
    rng = random.Random(seed)
    return [" ".join(rng.choice(WORDS) for _ in range(rng.choice([8, 15, 25, 40]))) + "." for _ in range(n)]


def ident(f):
    f = getattr(f, "_torchdynamo_orig_callable", f)
    f = getattr(f, "__func__", f)
    while hasattr(f, "__wrapped__"):
        f = f.__wrapped__
    c = getattr(f, "__code__", None)
    return f"{getattr(f, '__module__', '?')}.{getattr(f, '__qualname__', '?')}@{c.co_firstlineno if c else '?'}"


def config_snapshot(model):
    """_attn_implementation of the top config and every sub-config, keyed by path."""
    out = {}

    def walk(cfg, path, depth = 0):
        if cfg is None or depth > 4:
            return
        out[path] = {"model_type": getattr(cfg, "model_type", None),
                     "_attn_implementation": getattr(cfg, "_attn_implementation", None)}
        for k in ("text_config", "vision_config"):
            sub = getattr(cfg, k, None)
            if sub is not None and sub is not cfg:
                walk(sub, f"{path}.{k}", depth + 1)

    walk(getattr(model, "config", None), "config")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--work", required = True, type = Path)
    ap.add_argument("--warmup", type = int, default = 2)
    ap.add_argument("--repeats", type = int, default = 3)
    ap.add_argument("--steps", type = int, default = 3)
    args = ap.parse_args()
    res: dict = {"stage": "start", "model": MODEL}

    def save():
        args.out.write_text(json.dumps(res, indent = 2, default = str), encoding = "utf-8")

    try:
        res["env"] = {k: os.environ.get(k) for k in ("UNSLOTH_FLEX_ATTENTION_GENERATE", "UNSLOTH_COMPILE_DISABLE",
                                                     "UNSLOTH_COMPILE_LOCATION", "PYTHONPATH")}
        res["stage"] = "import"
        save()
        import unsloth
        from unsloth import FastLanguageModel
        import unsloth_zoo
        import torch
        import transformers
        res["unsloth_file"], res["zoo_file"] = unsloth.__file__, unsloth_zoo.__file__
        p = torch.cuda.get_device_properties(0)
        v = {"torch": torch.__version__, "hip": torch.version.hip, "cuda": torch.version.cuda,
             "transformers": transformers.__version__, "device": p.name, "arch": getattr(p, "gcnArchName", None)}
        for m in ("trl", "peft", "triton", "unsloth", "unsloth_zoo"):
            try:
                v[m] = __import__(m).__version__
            except Exception as e:  # noqa: BLE001
                v[m] = f"ERR {type(e).__name__}"
        res["versions"] = v

        # Engagement counter on the PR's context manager (absent at the base).
        import unsloth.models.vision as vision
        nfg = getattr(vision, "_NonFlexGenerate", None)
        res["has_NonFlexGenerate"] = nfg is not None
        engaged = collections.Counter()
        if nfg is not None:
            orig_enter = nfg.__enter__

            def counting_enter(self):
                r = orig_enter(self)
                engaged["active" if self.active else "inactive"] += 1
                return r
            nfg.__enter__ = counting_enter

        res["stage"] = "load"
        save()
        model, tok = FastLanguageModel.from_pretrained(MODEL, max_seq_length = 1024, load_in_4bit = False,
                                                       dtype = None)
        tt = getattr(tok, "tokenizer", tok)
        res["model_class"] = type(model).__name__
        res["dtype"] = str(getattr(model, "dtype", None))
        res["generate_fn"] = ident(model.generate)
        res["configs_after_load"] = config_snapshot(model)
        res["attn_impl_after_load"] = getattr(model.config, "_attn_implementation", None)

        attn_mods = [m for m in model.modules() if type(m).__name__.endswith("Attention") and hasattr(m, "config")]
        res["n_attention_modules"] = len(attn_mods)
        seen = collections.Counter()
        hook_on = {"v": False}

        def pre_hook(mod, _args):
            if hook_on["v"]:
                seen[str(getattr(mod.config, "_attn_implementation", None))] += 1
        handles = [m.register_forward_pre_hook(pre_hook) for m in attn_mods]

        msgs = [{"role": "user", "content": "Write a short story about a lighthouse keeper and a robot."}]
        prompt = tt.apply_chat_template(msgs, tokenize = False, add_generation_prompt = True)
        enc = tt(prompt, return_tensors = "pt", add_special_tokens = False).to("cuda")
        n_prompt = enc.input_ids.shape[1]
        res["prompt_tokens"] = n_prompt
        FastLanguageModel.for_inference(model)
        res["configs_after_for_inference"] = config_snapshot(model)

        def gen(n = NEW_TOKENS):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens = n, min_new_tokens = n, do_sample = False,
                                     use_cache = True)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            ids = out[0, n_prompt:].tolist()
            return ids, dt

        res["stage"] = "warmup"
        save()
        hook_on["v"] = True
        warm = []
        for _ in range(args.warmup):
            ids, dt = gen()
            warm.append(round(dt * 1000 / max(1, len(ids)), 4))
        res["warmup_ms_per_token"] = warm
        res["stage"] = "timed"
        save()
        timed, all_ids = [], []
        for _ in range(args.repeats):
            ids, dt = gen()
            timed.append(round(dt * 1000 / max(1, len(ids)), 4))
            all_ids.append(ids)
        hook_on["v"] = False
        res["ms_per_token"] = timed
        res["ms_per_token_median"] = statistics.median(timed)
        res["new_tokens"] = len(all_ids[0])
        res["token_ids"] = all_ids[0]
        res["tokens_stable_across_repeats"] = all(x == all_ids[0] for x in all_ids)
        res["text"] = tt.decode(all_ids[0], skip_special_tokens = True)[:600]
        res["attn_impl_seen_during_generate"] = dict(seen)
        res["configs_after_generate"] = config_snapshot(model)
        res["nonflex_generate_engaged"] = dict(engaged) if nfg is not None else "absent"
        save()

        # Untimed: which attention function actually ran during a short generate.
        res["stage"] = "dispatch"
        save()
        disp = collections.Counter()
        try:
            from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
            saved_fns = {}
            for key in ("flex_attention", "sdpa", "eager"):
                fn = ALL_ATTENTION_FUNCTIONS.get(key) if hasattr(ALL_ATTENTION_FUNCTIONS, "get") else None
                if fn is None:
                    continue
                saved_fns[key] = fn

                def wrap(*a, _fn = fn, _key = key, **k):
                    disp[f"ALL_ATTENTION_FUNCTIONS[{_key}]"] += 1
                    return _fn(*a, **k)
                ALL_ATTENTION_FUNCTIONS[key] = wrap
            import torch.nn.functional as F
            orig_sdpa = F.scaled_dot_product_attention

            def sdpa_wrap(*a, **k):
                disp["F.scaled_dot_product_attention"] += 1
                return orig_sdpa(*a, **k)
            F.scaled_dot_product_attention = sdpa_wrap
            try:
                gen(8)
            finally:
                F.scaled_dot_product_attention = orig_sdpa
                for key, fn in saved_fns.items():
                    ALL_ATTENTION_FUNCTIONS[key] = fn
            res["dispatch_during_generate"] = dict(disp)
        except Exception as e:  # noqa: BLE001
            res["dispatch_error"] = f"{type(e).__name__}: {str(e)[:400]}"
        res["configs_after_dispatch_generate"] = config_snapshot(model)

        # Training after generate.
        res["stage"] = "peft"
        save()
        from datasets import Dataset
        from trl import SFTConfig, SFTTrainer
        model = FastLanguageModel.get_peft_model(model, r = 8, lora_alpha = 16, lora_dropout = 0.0, random_state = 0,
                                                 target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                                                                   "gate_proj", "up_proj", "down_proj"],
                                                 use_gradient_checkpointing = "unsloth")
        FastLanguageModel.for_training(model)
        res["configs_after_peft"] = config_snapshot(model)
        if tt.pad_token is None:
            tt.pad_token = tt.eos_token
        torch.manual_seed(0)
        ds = Dataset.from_dict({"text": [d + (tt.eos_token or "") for d in make_docs()]})
        bf16 = model.dtype == torch.bfloat16 if hasattr(model, "dtype") else True
        cfg = SFTConfig(output_dir = str(args.work / "trainer"), per_device_train_batch_size = 2,
                        gradient_accumulation_steps = 1, max_steps = args.steps, learning_rate = 2e-4,
                        logging_steps = 1, bf16 = bf16, fp16 = not bf16, report_to = "none", seed = 0,
                        dataset_text_field = "text", max_length = 64, save_strategy = "no", dataset_num_proc = 1,
                        warmup_steps = 0, packing = False)
        res["stage"] = "train"
        save()
        trainer = SFTTrainer(model = model, train_dataset = ds, processing_class = tt, args = cfg)
        seen.clear()
        hook_on["v"] = True
        trainer.train()
        hook_on["v"] = False
        logs = [h for h in trainer.state.log_history if "loss" in h]
        res["losses"] = [float(h["loss"]) for h in logs]
        res["grad_norms"] = [float(h["grad_norm"]) if h.get("grad_norm") is not None else None for h in logs]
        res["attn_impl_seen_during_train"] = dict(seen)
        res["configs_after_train"] = config_snapshot(model)
        for h in handles:
            h.remove()
        res["stage"] = "done"
    except Exception as e:  # noqa: BLE001
        res["error"] = f"{type(e).__name__}: {e}"[:1500]
        res["traceback"] = traceback.format_exc()[-5000:]
    save()
    return 0


if __name__ == "__main__":
    sys.exit(main())
