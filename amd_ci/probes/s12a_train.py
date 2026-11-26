#!/usr/bin/env python3
"""A12-2 cell: one MoE LoRA training arm at the zoo state on PYTHONPATH (timed), or --build the shared model dir.

Model: Qwen3-MoE with the Qwen3-30B-A3B layer shape (hidden 2048, 128 experts, top-8, moe_intermediate 768,
32 q / 4 kv heads, head_dim 128), 2 layers, random weights, built once (--build) and loaded from disk by every arm.
LoRA r16 on q/k/v/o + experts (gate/up/down), Unsloth gradient checkpointing, AdamW lr 1e-3, fixed random tokens.

Writes JSON (--out): per-step seconds (synchronised), losses, step-0 grad norm, MoE counters (GENERIC_CALLS etc.),
the Triton policy per GEMM class, the self-check disable reason, a CPU-profiler op count of one extra step, and
grad parity vs --grads-ref (max abs, max rel to the ref tensor's max, global rel L2, cosine) at steps 0 and 1.
--grads-save stores this run's step-0 / step-1 LoRA grads (fp32, CPU) for later arms.
"""
import argparse, json, os, sys, time, traceback

ap = argparse.ArgumentParser()
ap.add_argument("--out", required = True)
ap.add_argument("--model-dir", required = True)
ap.add_argument("--build", action = "store_true")
ap.add_argument("--arm", default = "?")
ap.add_argument("--dtype", default = "bf16")
ap.add_argument("--steps", type = int, default = 16)
ap.add_argument("--warmup", type = int, default = 4)
ap.add_argument("--batch", type = int, default = 2)
ap.add_argument("--seqlen", type = int, default = 1024)
ap.add_argument("--grads-save", default = None)
ap.add_argument("--grads-ref", default = None)
ap.add_argument("--profile", action = "store_true")
args = ap.parse_args()
rec = {"arm": args.arm, "dtype": args.dtype, "stage": "start", "pid": os.getpid(),
       "env_triton": os.environ.get("UNSLOTH_MOE_GROUPED_TRITON")}


def dump():
    with open(args.out, "w", encoding = "utf-8") as f:
        json.dump(rec, f, indent = 1, default = str)


TOK_REPO = "trl-internal-testing/tiny-Qwen3MoeForCausalLM"


def build():
    import torch
    from transformers import AutoTokenizer, Qwen3MoeConfig, Qwen3MoeForCausalLM
    tok = AutoTokenizer.from_pretrained(TOK_REPO)
    cfg = Qwen3MoeConfig(
        vocab_size = 151936, hidden_size = 2048, intermediate_size = 6144, num_hidden_layers = 2,
        num_attention_heads = 32, num_key_value_heads = 4, head_dim = 128, num_experts = 128,
        num_experts_per_tok = 8, moe_intermediate_size = 768, decoder_sparse_step = 1, mlp_only_layers = [],
        norm_topk_prob = True, max_position_embeddings = 4096, tie_word_embeddings = False,
        bos_token_id = tok.bos_token_id, eos_token_id = tok.eos_token_id, pad_token_id = tok.pad_token_id,
        torch_dtype = "bfloat16")
    torch.manual_seed(0)
    model = Qwen3MoeForCausalLM(cfg).to(torch.bfloat16)
    model.save_pretrained(args.model_dir)
    tok.save_pretrained(args.model_dir)
    rec.update(ok = True, stage = "done", params = sum(p.numel() for p in model.parameters()))


def counters():
    out = {}
    for mod, attr, key in (("unsloth_zoo.temporary_patches.moe_grouped_fp16", "GENERIC_CALLS", "triton_gemm"),
                           ("unsloth_zoo.temporary_patches.moe_grouped_fp16", "CALLS", "fp16_calls"),
                           ("unsloth_zoo.temporary_patches.moe_lora_fold", "LORA_FOLD_STATS", "lora_fold"),
                           ("unsloth_zoo.temporary_patches.moe_lora_fold", "DECLINE_REASONS", "fold_declines")):
        try:
            m = __import__(mod, fromlist = [attr])
            out[key] = dict(getattr(m, attr))
        except Exception:
            pass
    return out


def lora_grads(model):
    import torch
    return {n: p.grad.detach().float().cpu().clone() for n, p in model.named_parameters()
            if p.requires_grad and p.grad is not None}


def compare(got, ref):
    import torch
    worst_abs = worst_rel = 0.0
    dd = rr = dot = gg = 0.0
    worst_name, missing = None, sorted(set(ref) ^ set(got))
    for n, r in ref.items():
        g = got.get(n)
        if g is None or g.shape != r.shape:
            continue
        d = (g - r).abs()
        a = float(d.max()) if d.numel() else 0.0
        rel = a / max(float(r.abs().max()), 1e-30)
        if rel > worst_rel:
            worst_rel, worst_name = rel, n
        worst_abs = max(worst_abs, a)
        dd += float((d.double() ** 2).sum()); rr += float((r.double() ** 2).sum())
        gg += float((g.double() ** 2).sum()); dot += float((g.double() * r.double()).sum())
    return {"max_abs": worst_abs, "max_rel_to_tensor_max": worst_rel, "worst_tensor": worst_name,
            "rel_l2": (dd ** 0.5) / max(rr ** 0.5, 1e-30), "cosine": dot / max((gg * rr) ** 0.5, 1e-30),
            "n_tensors": len(ref), "name_mismatch": missing[:10]}


def train():
    import unsloth  # noqa: F401
    from unsloth import FastLanguageModel
    import torch, unsloth_zoo, transformers
    from unsloth_zoo.temporary_patches import moe_utils as mu, moe_grouped_fp16 as mg
    rec["unsloth_zoo_file"] = unsloth_zoo.__file__
    rec["versions"] = {"torch": torch.__version__, "hip": getattr(torch.version, "hip", None),
                       "transformers": transformers.__version__, "unsloth": getattr(unsloth, "__version__", "?")}
    try:
        import triton
        rec["versions"]["triton"] = triton.__version__
    except Exception as e:  # noqa: BLE001
        rec["versions"]["triton"] = "ERR " + repr(e)[:200]
    rec["device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    props = torch.cuda.get_device_properties(0)
    rec["arch"] = getattr(props, "gcnArchName", None)
    rec["capability"] = list(torch.cuda.get_device_capability(0))
    rec["stage"] = "imported"; dump()
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    model, tok = FastLanguageModel.from_pretrained(args.model_dir, max_seq_length = args.seqlen, load_in_4bit = False,
                                                   dtype = dtype)
    model = FastLanguageModel.get_peft_model(
        model, r = 16, lora_alpha = 16, lora_dropout = 0, bias = "none",
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing = "unsloth", random_state = 0)
    rec["stage"] = "loaded"; dump()
    try:
        rec["moe_backend"] = mu.select_moe_backend()
    except Exception as e:  # noqa: BLE001
        rec["moe_backend"] = "ERR " + repr(e)[:200]
    pol = {}
    for kind in ("lora", "many", "few", "dw", "dw_few"):
        try:
            pol[kind] = mu._triton_grouped_mm_max_rows(0, kind)
        except Exception as e:  # noqa: BLE001
            pol[kind] = "ERR " + repr(e)[:120]
    rec["triton_policy"] = pol
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr = 1e-3)
    g = torch.Generator().manual_seed(1)
    batches = [torch.randint(3, 30000, (args.batch, args.seqlen), generator = g) for _ in range(args.steps + 1)]
    dev = model.get_input_embeddings().weight.device
    ref = torch.load(args.grads_ref, map_location = "cpu") if args.grads_ref and os.path.isfile(args.grads_ref) else None
    save, parity = {}, {}
    losses, secs, gnorm = [], [], None
    model.train()
    for step in range(args.steps):
        ids = batches[step].to(dev)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model(input_ids = ids, labels = ids)
        out.loss.backward()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        if step in (0, 1):
            gr = lora_grads(model)
            if step == 0:
                gnorm = float(torch.sqrt(sum((v.double() ** 2).sum() for v in gr.values())))
            if args.grads_save:
                save[step] = gr
            if ref is not None and step in ref:
                parity[f"step{step}"] = compare(gr, ref[step])
            del gr
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        opt.step(); opt.zero_grad(set_to_none = True)
        torch.cuda.synchronize()
        t3 = time.perf_counter()
        secs.append((t1 - t0) + (t3 - t2))  # excludes the grad copy-out
        losses.append(float(out.loss))
        if step == args.warmup - 1:
            rec["counters_after_warmup"] = counters()
        rec.update(losses = losses, step_secs = secs, stage = f"step {step}"); dump()
    if args.grads_save:
        torch.save(save, args.grads_save)
    rec["counters"] = counters()
    rec["generic_disabled_reason"] = getattr(mg, "_GENERIC_DISABLED_REASON", None)
    if args.profile:
        from torch.profiler import profile, ProfilerActivity
        before = counters()
        ids = batches[args.steps].to(dev)
        with profile(activities = [ProfilerActivity.CPU]) as prof:
            out = model(input_ids = ids, labels = ids)
            out.loss.backward()
            torch.cuda.synchronize()
        opt.zero_grad(set_to_none = True)
        after = counters()
        ev = {}
        for e in prof.key_averages():
            ev[e.key] = e.count
        keep = {k: v for k, v in ev.items() if any(s in k for s in ("mm", "grouped", "item", "local_scalar", "Memcpy",
                                                                     "memcpy", "nonzero", "unsloth", "triton", "_to_copy"))}
        rec["profile_step"] = {"op_counts": dict(sorted(keep.items(), key = lambda kv: -kv[1])[:40]),
                               "triton_gemm_delta": {k: after.get("triton_gemm", {}).get(k, 0) - before.get("triton_gemm", {}).get(k, 0)
                                                     for k in after.get("triton_gemm", {})}}
    timed = secs[args.warmup:]
    s = sorted(timed)
    rec.update(grad_norm_step0 = gnorm, parity = parity, median_step_s = s[len(s) // 2] if s else None,
               finite = all(x == x and abs(x) < 1e6 for x in losses) and gnorm == gnorm,
               peak_mem_gb = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2), ok = True, stage = "done")


try:
    build() if args.build else train()
except Exception as e:  # noqa: BLE001
    rec.update(ok = False, error = f"{type(e).__name__}: {e}"[:1500], where = traceback.format_exc()[-4000:])
dump()
sys.exit(0 if rec.get("ok") else 1)
