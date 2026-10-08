#!/usr/bin/env python3
"""Training worker for qlora_hang_probe.py (issue unslothai/unsloth#11498).

Runs ONE arm and writes progress as JSON lines to --heartbeat. It judges nothing:
the probe watches the heartbeat for stalls and reads the final record.

Arms share data, schedule and the loop, so the only difference is what runs under
model(...):
  unsloth  FastModel 4-bit + get_peft_model, Unsloth gradient checkpointing
  peft     Transformers + bitsandbytes NF4 + PEFT, never imports unsloth
Env differences (UNSLOTH_DISABLE_VENDORED_FLA, PYTHONPATH for a pinned release)
are set by the probe before this process starts.

The loop is a plain loop, not SFTTrainer, so both arms see byte-identical batches.
Shapes follow the failing Studio config: bs 1, GA 16, max_seq 1024, lr 2e-4
cosine with 5 warmup steps, adamw_8bit, bf16.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


class Beat:
    def __init__(self, path):
        self.fh = open(path, "a", encoding = "utf-8")

    def __call__(self, **kw):
        kw["t"] = time.time()
        self.fh.write(json.dumps(kw) + "\n")
        self.fh.flush()
        os.fsync(self.fh.fileno())


def compiled_triton_kernels(prefixes):
    """Count Triton kernels compiled at least once in modules under `prefixes`.
    Engagement evidence: a kill switch that silently did nothing shows up here."""
    counts = {}
    seen = set()

    def n_compiled(obj, depth = 0):
        if depth > 4 or id(obj) in seen:
            return 0
        seen.add(id(obj))
        total = 0
        caches = getattr(obj, "device_caches", None)
        if isinstance(caches, dict):
            for v in caches.values():
                kc = v[0] if isinstance(v, tuple) else v
                if isinstance(kc, dict):
                    total += len(kc)
        cache = getattr(obj, "cache", None)
        if isinstance(cache, dict) and caches is None:
            for v in cache.values():
                total += len(v) if isinstance(v, dict) else 0
        inner = getattr(obj, "fn", None)
        if inner is not None and inner is not obj:
            total += n_compiled(inner, depth + 1)
        return total

    for name, mod in list(sys.modules.items()):
        if mod is None or not any(name == p or name.startswith(p + ".") for p in prefixes):
            continue
        for attr, obj in list(vars(mod).items()):
            if type(obj).__name__ in ("JITFunction", "Autotuner", "Heuristics", "CachedAutotuner"):
                c = n_compiled(obj)
                if c:
                    counts[f"{name}.{attr}"] = c
    return counts


def build_batches(tokenizer, n, max_len, seed):
    """n sequences with lengths spread over 64..max_len, from real instruction data.
    Consecutive alpaca records are concatenated to reach each target length, so
    long and short shapes both occur, as with a real Studio dataset."""
    from datasets import load_dataset

    ds = load_dataset("yahma/alpaca-cleaned", split = "train")
    rng = random.Random(seed)
    eos = tokenizer.eos_token or ""
    out, buf, i = [], [], 0
    while len(out) < n:
        target = rng.randint(64, max_len)
        while len(buf) < target:
            r = ds[i % len(ds)]
            i += 1
            text = (f"### Instruction:\n{r['instruction']}\n\n### Input:\n{r['input']}\n\n"
                    f"### Response:\n{r['output']}{eos}")
            buf += tokenizer(text, add_special_tokens = False)["input_ids"]
        out.append(buf[:target])
        buf = []
    return out


def load_unsloth(model_name, seq):
    import torch
    from unsloth import FastModel

    model, _ = FastModel.from_pretrained(
        model_name, max_seq_length = seq, load_in_4bit = True, dtype = torch.bfloat16,
    )
    model = FastModel.get_peft_model(
        model, r = 16, lora_alpha = 16, lora_dropout = 0, bias = "none",
        target_modules = TARGETS, use_gradient_checkpointing = "unsloth", random_state = 3407,
    )
    return model


def load_peft(model_name):
    import torch
    import transformers
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    qc = transformers.BitsAndBytesConfig(
        load_in_4bit = True, bnb_4bit_quant_type = "nf4", bnb_4bit_use_double_quant = True,
        bnb_4bit_compute_dtype = torch.bfloat16,
    )
    cls = transformers.AutoModelForCausalLM
    cfg = transformers.AutoConfig.from_pretrained(model_name)
    if "ConditionalGeneration" in (cfg.architectures or [""])[0]:
        cls = transformers.AutoModelForImageTextToText
    model = cls.from_pretrained(
        model_name, quantization_config = qc, dtype = torch.bfloat16, device_map = {"": 0},
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing = True)
    model = get_peft_model(model, LoraConfig(
        r = 16, lora_alpha = 16, lora_dropout = 0.0, bias = "none", target_modules = TARGETS,
        task_type = "CAUSAL_LM",
    ))
    return model


def versions():
    out = {"python": sys.version.split()[0]}
    for mod in ("torch", "triton", "bitsandbytes", "transformers", "peft", "trl"):
        try:
            m = __import__(mod)
            out[mod] = getattr(m, "__version__", "?")
        except Exception as e:  # noqa: BLE001
            out[mod] = f"unavailable: {type(e).__name__}"
    for mod in ("unsloth", "unsloth_zoo"):
        m = sys.modules.get(mod)
        out[mod] = None if m is None else {"version": getattr(m, "__version__", "?"),
                                            "file": getattr(m, "__file__", "?")}
    try:
        import torch
        out["hip"] = torch.version.hip
        p = torch.cuda.get_device_properties(0)
        out["device"] = {"name": p.name, "arch": getattr(p, "gcnArchName", None),
                         "total_gib": round(p.total_memory / 2**30, 2)}
    except Exception as e:  # noqa: BLE001
        out["device"] = f"unavailable: {type(e).__name__}: {e}"
    try:
        import bitsandbytes.cextension as ce
        out["bnb_lib"] = str(getattr(getattr(ce, "lib", None), "_name", None)
                             or getattr(ce, "BNB_BACKEND", None))
    except Exception:  # noqa: BLE001
        pass
    return out


def engagement():
    eng = {"unsloth_imported": "unsloth" in sys.modules}
    fla = sys.modules.get("fla")
    eng["fla_module"] = None if fla is None else getattr(fla, "__file__", "?")
    eng["fla_vendored"] = bool(fla is not None and getattr(fla, "_UNSLOTH_VENDORED_FLA", False))
    eng["fla_kernels"] = compiled_triton_kernels(["fla"])
    eng["unsloth_kernels"] = compiled_triton_kernels(["unsloth", "unsloth_zoo"])
    utils = sys.modules.get("unsloth.kernels.utils")
    if utils is not None:
        eng["nf4_triton"] = getattr(utils, "_USE_NF4_KERNELS", "absent (pre-#12113)")
    return eng


def train(args, beat):
    import torch

    torch.manual_seed(args.seed)
    if args.arm == "unsloth":
        model = load_unsloth(args.model, args.seq)
    else:
        model = load_peft(args.model)
    import transformers
    tok = transformers.AutoTokenizer.from_pretrained(args.model)
    beat(phase = "loaded", versions = versions())

    import bitsandbytes as bnb
    params = [p for p in model.parameters() if p.requires_grad]
    opt = bnb.optim.AdamW8bit(params, lr = 2e-4, weight_decay = 0.001)
    sched = transformers.get_cosine_schedule_with_warmup(opt, 5, args.max_steps)
    batches = build_batches(tok, args.max_steps * args.ga, args.seq, args.seed)
    beat(phase = "data", n = len(batches))

    model.train()
    deadline = time.time() + args.minutes * 60
    micro = 0
    for step in range(args.max_steps):
        total = 0.0
        for _ in range(args.ga):
            ids = torch.tensor([batches[micro]], device = "cuda")
            out = model(input_ids = ids, attention_mask = torch.ones_like(ids), labels = ids)
            (out.loss / args.ga).backward()
            loss = float(out.loss.detach())  # syncs: the heartbeat means the GPU finished
            total += loss / args.ga
            micro += 1
            beat(phase = "micro", micro = micro, len = ids.shape[1], loss = loss)
        gn = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none = True)
        torch.cuda.synchronize()
        beat(phase = "step", step = step + 1, micro = micro, loss = total, grad_norm = gn,
             peak_gib = round(torch.cuda.max_memory_allocated() / 2**30, 2))
        if step == 0:
            beat(phase = "engagement", engagement = engagement())
        if time.time() > deadline:
            beat(phase = "budget", step = step + 1)
            break
    beat(phase = "engagement", engagement = engagement())


def fla_stress(args, beat):
    """Vendored chunk_gated_delta_rule fwd+bwd at Qwen3.5-9B linear-attention shapes
    (32 value heads after the k-head repeat, head dim 128), T over 16..1024 with
    tails off the 64 chunk. Looks for hangs, faults and non-finite outputs."""
    import torch
    import unsloth  # noqa: F401  (injects the vendored fla)
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    beat(phase = "loaded", versions = versions())
    rng = random.Random(args.seed)
    deadline = time.time() + args.minutes * 60
    H, D = 32, 128
    it = 0
    while time.time() < deadline:
        T = rng.choice([16, 63, 64, 65, 127, 200, 511, 513, 777, 1000, 1023, 1024, rng.randint(16, 1024)])
        kw = dict(device = "cuda", dtype = torch.bfloat16)
        q = torch.randn(1, T, H, D, **kw).requires_grad_()
        k = torch.randn(1, T, H, D, **kw).requires_grad_()
        v = torch.randn(1, T, H, D, **kw).requires_grad_()
        g = torch.nn.functional.logsigmoid(torch.randn(1, T, H, device = "cuda", dtype = torch.float32)).requires_grad_()
        b = torch.rand(1, T, H, **kw).sigmoid().requires_grad_()
        o, _ = chunk_gated_delta_rule(q, k, v, g = g, beta = b, use_qk_l2norm_in_kernel = True)
        o.float().square().mean().backward()
        finite = all(bool(torch.isfinite(t).all()) for t in (o, q.grad, k.grad, v.grad, g.grad, b.grad))
        it += 1
        beat(phase = "micro", micro = it, len = T, finite = finite)
        if not finite:
            beat(phase = "nonfinite", micro = it, len = T)
    beat(phase = "engagement", engagement = engagement())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices = ["unsloth", "peft", "fla_stress"], required = True)
    ap.add_argument("--model", default = "unsloth/Qwen3.5-2B")
    ap.add_argument("--max-steps", type = int, default = 120)
    ap.add_argument("--ga", type = int, default = 16)
    ap.add_argument("--seq", type = int, default = 1024)
    ap.add_argument("--minutes", type = float, default = 35)
    ap.add_argument("--seed", type = int, default = 3407)
    ap.add_argument("--heartbeat", required = True)
    args = ap.parse_args()

    beat = Beat(args.heartbeat)
    beat(phase = "start", arm = args.arm, argv = sys.argv[1:],
         env = {k: v for k, v in os.environ.items() if k.startswith(("UNSLOTH_", "HIP_", "AMD_", "HSA_", "PYTHONPATH"))})
    if args.arm == "fla_stress":
        fla_stress(args, beat)
    else:
        train(args, beat)
    beat(phase = "done")


if __name__ == "__main__":
    main()
