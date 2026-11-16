#!/usr/bin/env python3
"""One process of the PR 16197 / zoo 2296 probe (reset the inference stance counter at every generate). Observes only.

PYTHONPATH puts the arm's unsloth checkout, then its unsloth-zoo checkout, first. Writes JSON to --out after
every generate:
  * which unsloth / unsloth_zoo were imported, whether zoo has unsloth_reset_inference_runs, whether
    FastBaseModel.for_inference calls it, versions, device / arch, attn impl after load
  * per generate of --phases (<prompt_len>x<new_tokens>) and then --same repeats of --same-shape:
      - the Dynamo stance at the ENTRY of every top-level forward (forward pre-hook on the model):
        index 0 = prefill, index 1 = decode step 1 (the body runs under the stance read at entry; the
        generated block that switches it runs at the logits site, after the body)
      - flex eager fallback count: calls of torch._higher_order_ops.flex_attention.math_attention
      - new Dynamo graphs (counters["stats"]["unique_graphs"] delta), UNSLOTH_INFERENCE_RUNS before / after
      - ms/token over the whole generate (prefill included), greedy ids
  * optionally (--sft N) LoRA r8 + N SFT steps after the generates: losses
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import random
import sys
import time
import traceback
from pathlib import Path

MODELS = {"270m": "unsloth/gemma-3-270m-it", "1b": "unsloth/gemma-3-1b-it"}
PARA = ("The history of navigation spans thousands of years, from reading stars and currents to magnetic "
        "compasses, chronometers and satellite positioning. ")
WORDS = ("the of and to in is was for on as with by he at from his an were are which this be "
         "or has had not but first one their its new after who they have two also been other more all "
         "she time year can city world river music house water light green small large north south").split()


def make_docs(n = 32, seed = 0):
    rng = random.Random(seed)
    return [" ".join(rng.choice(WORDS) for _ in range(rng.choice([8, 15, 25, 40]))) + "." for _ in range(n)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--work", required = True, type = Path)
    ap.add_argument("--model", required = True, choices = sorted(MODELS))
    ap.add_argument("--phases", default = "2048x1,2048x32,2048x1,2048x1,2048x32")
    ap.add_argument("--same", type = int, default = 0)
    ap.add_argument("--same-shape", default = "2048x32")
    ap.add_argument("--sft", type = int, default = 0)
    args = ap.parse_args()
    res: dict = {"stage": "start", "model": MODELS[args.model], "generates": []}

    def save():
        args.out.write_text(json.dumps(res, indent = 2, default = str), encoding = "utf-8")

    try:
        res["stage"] = "import"
        save()
        import unsloth
        from unsloth import FastModel
        import unsloth_zoo
        import torch
        import torch._dynamo.eval_frame as EF
        import transformers
        from torch._dynamo.utils import counters
        import unsloth_zoo.temporary_patches.utils as ZU
        import unsloth.models.vision as UV

        res["unsloth_file"], res["zoo_file"] = unsloth.__file__, unsloth_zoo.__file__
        res["zoo_has_reset"] = hasattr(ZU, "unsloth_reset_inference_runs")
        res["for_inference_calls_reset"] = "unsloth_reset_inference_runs" in UV.FastBaseModel.for_inference.__code__.co_names
        p = torch.cuda.get_device_properties(0)
        v = {"torch": torch.__version__, "hip": torch.version.hip, "cuda": torch.version.cuda,
             "transformers": transformers.__version__, "device": p.name, "arch": getattr(p, "gcnArchName", None)}
        for m in ("trl", "peft", "triton", "unsloth", "unsloth_zoo"):
            try:
                v[m] = __import__(m).__version__
            except Exception as e:  # noqa: BLE001
                v[m] = f"ERR {type(e).__name__}"
        res["versions"] = v

        import importlib
        hop = importlib.import_module("torch._higher_order_ops.flex_attention")
        math_n = collections.Counter()
        _math = hop.math_attention

        def _counted(*a, **k):
            math_n["n"] += 1
            return _math(*a, **k)

        hop.math_attention = _counted
        res["math_hook"] = "torch._higher_order_ops.flex_attention.math_attention"

        def stance():
            try:
                return EF._stance.stance
            except Exception as e:  # noqa: BLE001
                return repr(e)

        def zoo_runs():
            shared = getattr(ZU, "UNSLOTH_INFERENCE_RUNS", None)
            mods = [getattr(m, "INFERENCE_RUNS", None) for n, m in list(sys.modules.items())
                    if n.startswith("unsloth_compiled_module_") and hasattr(m, "INFERENCE_RUNS")]
            return {"shared": None if shared is None else shared[0], "modules": mods}

        res["stage"] = "load"
        save()
        model, tok = FastModel.from_pretrained(MODELS[args.model], max_seq_length = 4096, load_in_4bit = False,
                                               dtype = None, full_finetuning = False)
        tt = getattr(tok, "tokenizer", tok)
        tc = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
        res["model_class"] = type(model).__name__
        res["dtype"] = str(getattr(model, "dtype", None))
        res["attn_impl"] = getattr(tc, "_attn_implementation", None)
        res["attn_impl_top"] = getattr(model.config, "_attn_implementation", None)
        res["compiled_modules"] = sorted(n for n in sys.modules if n.startswith("unsloth_compiled_module_"))

        stances: list = []
        rec_on = {"v": False}

        def pre_hook(_mod, _a, _k):
            if rec_on["v"]:
                stances.append(stance())

        handle = model.register_forward_pre_hook(pre_hook, with_kwargs = True)

        def chat(text):
            return tt.apply_chat_template([{"role": "user", "content": text}], tokenize = False,
                                          add_generation_prompt = True)

        def long_ids(n):
            body = tt(" ".join("Paragraph %d: %s" % (i, PARA) for i in range(n // 20 + 10)),
                      add_special_tokens = False)["input_ids"]
            tail = tt(chat("Summarise the text above in two sentences."), add_special_tokens = False)["input_ids"]
            ids = [tt.bos_token_id] + body[: max(0, n - 1 - len(tail))] + tail
            return torch.tensor([ids[:max(n, len(tail) + 1)]], device = "cuda")

        def generate(tag, phase):
            a, b = phase.split("x")
            new = int(b)
            ids = long_ids(int(a))
            stances.clear()
            m0, g0 = math_n["n"], counters["stats"]["unique_graphs"]
            r = {"tag": tag, "phase": phase, "prompt": int(ids.shape[1]), "new": new,
                 "runs_before": zoo_runs(), "stance_before": stance()}
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            rec_on["v"] = True
            try:
                with torch.no_grad():
                    out = model.generate(input_ids = ids, attention_mask = torch.ones_like(ids), max_new_tokens = new,
                                         min_new_tokens = new, do_sample = False, use_cache = True)
                r["ids"] = out[0, ids.shape[1]:].tolist()
            except Exception as e:  # noqa: BLE001
                r["error"] = "".join(traceback.format_exception(e))[-4000:]
            finally:
                rec_on["v"] = False
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            r.update(seconds = round(dt, 3), ms_per_token = round(1000 * dt / max(1, len(r.get("ids") or [])), 3),
                     math = math_n["n"] - m0, new_graphs = counters["stats"]["unique_graphs"] - g0,
                     n_forwards = len(stances), stances = stances[:8],
                     stance_counts = dict(collections.Counter(stances)),
                     decode_step1_stance = stances[1] if len(stances) > 1 else None,
                     prefill_stance = stances[0] if stances else None,
                     runs_after = zoo_runs(), stance_after = stance())
            res["generates"].append(r)
            save()
            print("GEN", tag, phase, "math", r["math"], "graphs", r["new_graphs"], "ms/tok", r["ms_per_token"],
                  "stances", r["stances"][:4], "err", (r.get("error") or "")[-300:], flush = True)

        res["stage"] = "generate"
        save()
        for i, ph in enumerate(args.phases.split(",")):
            generate(f"seq{i}", ph)
        for i in range(args.same):
            generate(f"same{i}", args.same_shape)
        handle.remove()
        res["stage_generate_done"] = True

        if args.sft:
            res["stage"] = "peft"
            save()
            from datasets import Dataset
            from trl import SFTConfig, SFTTrainer
            model = FastModel.get_peft_model(model, r = 8, lora_alpha = 16, lora_dropout = 0.0, random_state = 0,
                                             finetune_vision_layers = False, finetune_language_layers = True,
                                             finetune_attention_modules = True, finetune_mlp_modules = True,
                                             use_gradient_checkpointing = "unsloth")
            FastModel.for_training(model)
            if tt.pad_token is None:
                tt.pad_token = tt.eos_token
            torch.manual_seed(0)
            ds = Dataset.from_dict({"text": [d + (tt.eos_token or "") for d in make_docs()]})
            bf16 = model.dtype == torch.bfloat16 if hasattr(model, "dtype") else True
            cfg = SFTConfig(output_dir = str(args.work / "trainer"), per_device_train_batch_size = 2,
                            gradient_accumulation_steps = 1, max_steps = args.sft, learning_rate = 2e-4,
                            logging_steps = 1, bf16 = bf16, fp16 = not bf16, report_to = "none", seed = 0,
                            dataset_text_field = "text", max_length = 64, save_strategy = "no", dataset_num_proc = 1,
                            warmup_steps = 0, packing = False)
            res["stage"] = "train"
            save()
            res["stance_before_train"] = stance()
            trainer = SFTTrainer(model = model, train_dataset = ds, processing_class = tt, args = cfg)
            trainer.train()
            logs = [h for h in trainer.state.log_history if "loss" in h]
            res["losses"] = [float(h["loss"]) for h in logs]
            res["grad_norms"] = [float(h["grad_norm"]) if h.get("grad_norm") is not None else None for h in logs]
            res["stance_after_train"] = stance()
            res["runs_after_train"] = zoo_runs()
        res["stage"] = "done"
    except Exception as e:  # noqa: BLE001
        res["error"] = f"{type(e).__name__}: {e}"[:1500]
        res["traceback"] = traceback.format_exc()[-5000:]
    save()
    return 0


if __name__ == "__main__":
    sys.exit(main())
