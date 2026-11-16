#!/usr/bin/env python3
"""One tiny MoE LoRA / QLoRA training arm (Unsloth gradient checkpointing) at the zoo state on PYTHONPATH.
Writes JSON: stage, losses, step-0 LoRA grad norm, finite flag, unsloth_zoo file, MoE counters, versions.
usage: zoo_moe_train.py --case qwen3moe:16bit --out X.json [--steps 6]"""
import argparse, json, os, sys, time, traceback

ap = argparse.ArgumentParser()
ap.add_argument("--case", required = True)
ap.add_argument("--out", required = True)
ap.add_argument("--steps", type = int, default = 6)
args = ap.parse_args()
rec = {"case": args.case, "stage": "start"}


def dump():
    with open(args.out, "w", encoding = "utf-8") as f:
        json.dump(rec, f, indent = 1, default = str)


MODELS = {"qwen3moe": "trl-internal-testing/tiny-Qwen3MoeForCausalLM", "gptoss": "hf-internal-testing/tiny-random-GptOssForCausalLM"}


def counters():
    out = {}
    for mod, attr, key in (("unsloth_zoo.temporary_patches.moe_grouped_fp16", "GENERIC_CALLS", "triton_gemm"),
                           ("unsloth_zoo.gradient_checkpointing", "MOE_GC_ROUTING_STATS", "routing"),
                           ("unsloth_zoo.temporary_patches.moe_expert_cache", "EXPERT_CACHE_STATS", "expert_cache"),
                           ("unsloth_zoo.temporary_patches.gpt_oss_fused_glue", "FUSED_GLUE_CALLS", "fused_glue")):
        try:
            m = __import__(mod, fromlist = [attr])
            out[key] = dict(getattr(m, attr))
        except Exception:
            pass
    return out


try:
    name, load = args.case.split(":")
    import unsloth  # noqa: F401
    from unsloth import FastLanguageModel
    import torch, unsloth_zoo, transformers
    rec["unsloth_zoo_file"] = unsloth_zoo.__file__
    rec["versions"] = {"torch": torch.__version__, "hip": getattr(torch.version, "hip", None),
                       "transformers": transformers.__version__, "unsloth": getattr(unsloth, "__version__", "?")}
    rec["device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    rec["stage"] = "imported"; dump()
    torch.manual_seed(0)
    t0 = time.time()
    model, tok = FastLanguageModel.from_pretrained(MODELS[name], max_seq_length = 256, load_in_4bit = load == "4bit",
                                                   dtype = torch.bfloat16)
    model = FastLanguageModel.get_peft_model(
        model, r = 16, lora_alpha = 16, lora_dropout = 0, bias = "none",
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing = "unsloth", random_state = 0)
    rec["stage"] = "loaded"; dump()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr = 1e-3)
    g = torch.Generator().manual_seed(1)
    vocab = getattr(model.config, "vocab_size", 1000) or 1000
    dev = model.get_input_embeddings().weight.device
    losses, gnorm = [], None
    model.train()
    for step in range(args.steps):
        ids = torch.randint(3, min(vocab, 30000), (2, 128), generator = g).to(dev)
        out = model(input_ids = ids, labels = ids)
        out.loss.backward()
        if step == 0:
            gnorm = float(torch.sqrt(sum((p.grad.float() ** 2).sum().cpu() for p in model.parameters()
                                         if p.grad is not None)))
        opt.step(); opt.zero_grad(set_to_none = True)
        losses.append(float(out.loss))
    rec.update(losses = losses, grad_norm_step0 = gnorm, secs = round(time.time() - t0, 1), counters = counters(),
               finite = all(x == x and abs(x) < 1e6 for x in losses) and gnorm == gnorm,
               ok = True, stage = "done")
except Exception as e:  # noqa: BLE001
    rec.update(ok = False, error = f"{type(e).__name__}: {e}"[:800], where = traceback.format_exc()[-3000:])
dump()
sys.exit(0 if rec.get("ok") else 1)
