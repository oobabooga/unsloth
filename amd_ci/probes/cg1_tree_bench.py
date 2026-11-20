#!/usr/bin/env python3
"""CG1: AsyncGRPO prefix-trie rows on Unsloth fast backbones, FlexAttention tree mask vs unroll.

Usage (remote GPU): python cg1_tree_bench.py <patch.diff> <base_sha> [--models a,b] [--groups G]
Installs unsloth at <base_sha> + the patch, unsloth_zoo main, trl 1.18.0, then for each model:
  correctness: per-token logps + LoRA grads, tree vs unroll vs per-sample reference (bf16)
  perf: fwd+bwd step time and peak memory, interleaved A1 B A2 rounds (A = unroll, B = tree), A/A spread
"""

import json
import os
import subprocess
import sys
import time


def sh(cmd, **kw):
    print("+", cmd if isinstance(cmd, str) else " ".join(cmd), flush = True)
    subprocess.run(cmd, check = True, shell = isinstance(cmd, str), **kw)


def setup(diff, sha):
    work = os.path.abspath("cg1_src")
    if not os.path.isdir(work):
        sh(["git", "clone", "-q", "--filter=blob:none", "https://github.com/unslothai/unsloth", work])
    sh(["git", "-C", work, "checkout", "-q", sha])
    sh(["git", "-C", work, "apply", "--include=unsloth/*", os.path.abspath(diff)])
    pip = [sys.executable, "-m", "pip", "install", "-q"]
    try:
        subprocess.run(["uv", "--version"], check = True, capture_output = True)
        pip = ["uv", "pip", "install", "-q", "--python", sys.executable]
    except Exception:
        pass
    sh(pip + ["git+https://github.com/unslothai/unsloth_zoo", work])
    sh(pip + ["trl==1.18.0", "transformers==5.23.0", "bitsandbytes"])
    sh(pip + ["--no-deps", "--reinstall", "git+https://github.com/unslothai/unsloth_zoo", work])


def make_samples(module, groups, prompt_len, n_completions, completion_len, fork_at, vocab, seed):
    import numpy as np
    import torch

    g = torch.Generator().manual_seed(seed)
    samples = []
    system = torch.randint(100, vocab, (min(64, prompt_len // 4),), generator = g).tolist()
    for group in range(groups):
        prompt = system + torch.randint(100, vocab, (prompt_len - len(system),), generator = g).tolist()
        shared = torch.randint(100, vocab, (fork_at,), generator = g).tolist()
        for c in range(n_completions):
            # Half the completions share their first fork_at tokens (a multi-turn style fork).
            head = shared if c % 2 == 0 else []
            tail = torch.randint(100, vocab, (completion_len - len(head),), generator = g).tolist()
            ids = prompt + head + tail
            n = len(ids)
            samples.append(
                {
                    "input_ids": torch.tensor(ids),
                    "completion_mask": torch.tensor([0] * len(prompt) + [1] * (n - len(prompt))),
                    "old_log_probs": torch.zeros(n),
                    "advantage": 1.0,
                    "group_id": group,
                    "sampling_mask": [[] for _ in range(n)],
                    "sampler_ids": np.zeros((n, 0), dtype = "int64"),
                    "sampler_log_probs": np.zeros((n, 0), dtype = "float32"),
                }
            )
    row = module.TreeRow()
    for s in samples:
        row.add(s)
    packed = row.pack()
    return samples, packed


def run_bench(models, groups, dtype_name = "bf16", perf = True):
    import torch

    os.environ.setdefault("UNSLOTH_COMPILE_LOCATION", os.path.abspath("unsloth_compiled_cache"))
    import unsloth  # noqa: F401
    from unsloth import FastLanguageModel
    import unsloth.models.rl as rl
    import trl
    import transformers
    import trl.trainer.async_grpo_trainer as M

    if not getattr(M.add_fused_lm_head, "_unsloth_async_grpo_patched", False):
        rl.patch_trl_async_grpo()
    assert M.add_fused_lm_head._unsloth_async_grpo_patched
    info = {
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "trl": trl.__version__,
        "transformers": transformers.__version__,
        "unsloth_file": rl.__file__,
    }
    print("ENV", json.dumps(info), flush = True)
    results = {"env": info, "models": {}}

    for name in models:
        model, tok = FastLanguageModel.from_pretrained(
            name, max_seq_length = 32768, load_in_4bit = False, dtype = {"fp32": torch.float32, "fp16": torch.float16}.get(dtype_name, torch.bfloat16)
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r = 16,
            lora_alpha = 16,
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            use_gradient_checkpointing = "unsloth",
        )
        model.train()
        base = model.get_base_model()
        M.add_fused_lm_head(model, temperature = 1.0, outputs = ("log_probs", "entropy"))
        lora = [p for n, p in model.named_parameters() if p.requires_grad]
        vocab = min(int(base.config.vocab_size), 30000)
        res = {}

        def forward(packed, mode):
            os.environ["UNSLOTH_ASYNC_GRPO_TREE_ATTENTION"] = "1" if mode == "tree" else "0"
            inputs = {k: v[None].cuda() for k, v in packed.items()}
            out = base(
                input_ids = inputs["input_ids"],
                position_ids = inputs["position_ids"],
                shift_labels = inputs["shift_labels"],
                fused_lm_head = True,
                **M.unpad_tree(inputs),
            )
            return out.log_probs.flatten()[inputs["token_index"][0]]

        def grads_of(loss):
            # .backward(), not autograd.grad: Unsloth's gradient checkpointing is reentrant.
            for p in lora:
                p.grad = None
            loss.backward()
            grads = [p.grad.detach().clone() for p in lora]
            for p in lora:
                p.grad = None
            return grads

        # Correctness: small row, tree vs unroll vs each sample alone.
        samples, packed = make_samples(M, 2, 96, 4, 48, 16, vocab, 0)
        ref, ref_grads = [], None
        loss = 0
        for s in samples:
            ids = s["input_ids"][None].cuda()
            p = {
                "input_ids": s["input_ids"],
                "position_ids": torch.arange(len(s["input_ids"])),
                "shift_labels": torch.cat([s["input_ids"][1:], torch.tensor([-100])]),
            }
            mask = s["completion_mask"][1:].bool()
            p["shift_labels"][:-1][~mask] = -100
            inputs = {k: v[None].cuda() for k, v in p.items()}
            out = base(**inputs, fused_lm_head = True)
            lp = out.log_probs[0][inputs["shift_labels"][0] != -100]
            ref.append(lp.detach().float())
            loss = loss + lp.float().sum()
        ref_grads = grads_of(loss)
        ref = torch.cat(ref)
        corr = {}
        for mode in ("unroll", "tree"):
            before = dict(rl._ASYNC_GRPO_TREE_COUNTS)
            lp = forward(packed, mode)
            grads = grads_of(lp.float().sum())
            after = dict(rl._ASYNC_GRPO_TREE_COUNTS)
            num = sum(float((a.float() - b.float()).pow(2).sum()) for a, b in zip(grads, ref_grads))
            den = sum(float(b.float().pow(2).sum()) for b in ref_grads)
            corr[mode] = {
                "logp_maxabs_vs_ref": float((lp.float() - ref).abs().max()),
                "logp_meanabs_vs_ref": float((lp.float() - ref).abs().mean()),
                "grad_rel_l2_vs_ref": (num / den) ** 0.5,
                "engaged": {k: after[k] - before[k] for k in after},
                "_lp": lp.detach().float(),
                "_g": [x.detach() for x in grads],
            }
        num = sum(
            float((a.float() - b.float()).pow(2).sum()) for a, b in zip(corr["tree"]["_g"], corr["unroll"]["_g"])
        )
        den = sum(float(b.float().pow(2).sum()) for b in corr["unroll"]["_g"])
        res["tree_vs_unroll"] = {
            "logp_maxabs": float((corr["tree"]["_lp"] - corr["unroll"]["_lp"]).abs().max()),
            "grad_rel_l2": (num / den) ** 0.5,
        }
        for mode in corr:
            corr[mode].pop("_lp")
            corr[mode].pop("_g")
        res["correctness"] = corr
        res["correctness_unique_tokens"] = int(packed["input_ids"].numel())
        res["correctness_raw_tokens"] = int(sum(len(s["input_ids"]) for s in samples))
        print("CORRECTNESS", name, json.dumps(res), flush = True)

        if not perf:
            results["models"][name] = res
            del model, base
            torch.cuda.empty_cache()
            continue
        # Perf: a GRPO-shaped row (shared prompt, 8 completions, half forked).
        samples, packed = make_samples(M, groups, 1536, 8, 384, 128, vocab, 1)
        unique = int(packed["input_ids"].numel())
        raw = int(sum(len(s["input_ids"]) for s in samples))
        res["perf_row"] = {"unique_tokens": unique, "raw_tokens": raw, "packing_ratio": raw / unique}

        def step(mode):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            t = time.perf_counter()
            lp = forward(packed, mode)
            lp.float().sum().backward()
            torch.cuda.synchronize()
            dt = time.perf_counter() - t
            for p in lora:
                p.grad = None
            return dt, torch.cuda.max_memory_allocated() / 2**30

        for mode in ("unroll", "tree", "unroll", "tree"):  # warmup / compile
            step(mode)
        series = {"A1_unroll": [], "B_tree": [], "A2_unroll": []}
        mems = {"A1_unroll": [], "B_tree": [], "A2_unroll": []}
        for _ in range(5):
            for arm, mode in (("A1_unroll", "unroll"), ("B_tree", "tree"), ("A2_unroll", "unroll")):
                dt, mem = step(mode)
                series[arm].append(dt)
                mems[arm].append(mem)
        res["perf"] = {
            arm: {
                "median_s": sorted(v)[len(v) // 2],
                "min_s": min(v),
                "max_s": max(v),
                "peak_gib": max(mems[arm]),
                "times": v,
            }
            for arm, v in series.items()
        }
        res["counts_total"] = dict(rl._ASYNC_GRPO_TREE_COUNTS)
        print("PERF", name, json.dumps(res["perf_row"]), json.dumps(res["perf"]), flush = True)
        results["models"][name] = res
        del model, base
        torch.cuda.empty_cache()
    out = os.environ.get("CG1_OUT", "cg1_tree_bench.json")
    with open(out, "w") as f:
        json.dump(results, f, indent = 1)
    print("WROTE", out, flush = True)


if __name__ == "__main__":
    args = sys.argv[1:]
    models = ["unsloth/Qwen3-0.6B", "unsloth/Llama-3.2-1B-Instruct"]
    groups = 1
    if "--models" in args:
        models = args[args.index("--models") + 1].split(",")
    if "--groups" in args:
        groups = int(args[args.index("--groups") + 1])
    if "--bench-only" not in args:
        setup(args[0], args[1])
        cmd = [sys.executable, os.path.abspath(__file__), *args, "--bench-only"]
        sys.exit(subprocess.run(cmd).returncode)
    run_bench(
        models,
        groups,
        "fp32" if "--fp32" in args else ("fp16" if "--fp16" in args else "bf16"),
        perf = "--no-perf" not in args,
    )
