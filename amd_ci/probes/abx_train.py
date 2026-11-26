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
ap.add_argument("--save-init", default = None, help = "save the trainable tensors before step 0 + the LoRA config (oracle input)")
ap.add_argument("--oracle", default = None, help = "oracle mode: CPU float64 plain transformers + PEFT from --init; write grads / acts here")
ap.add_argument("--init", default = None)
ap.add_argument("--oracle-ref", default = None, help = "compare step-0 grads / activations with this oracle file")
args = ap.parse_args()
rec = {"arm": args.arm, "dtype": args.dtype, "stage": "start", "pid": os.getpid(),
       "env_triton": os.environ.get("UNSLOTH_MOE_GROUPED_TRITON"), "env_backend": os.environ.get("UNSLOTH_MOE_BACKEND")}


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


ACT_SUFFIXES = ("self_attn", "mlp", "mlp.gate", "mlp.experts")


def _act_key(name):
    i = name.find("layers.")
    if i < 0:
        return None
    rest = name[i + len("layers."):]
    idx, _, sub = rest.partition(".")
    return f"L{idx}.{sub}" if idx.isdigit() and sub in ACT_SUFFIXES else None


def act_hooks(model, store, active):
    """Step-0 forward outputs of attention / MoE block / router / experts, first call only (the checkpoint
    recompute in backward is skipped: active[0] is False then)."""
    import torch
    hs = []
    for name, m in model.named_modules():
        k = _act_key(name)
        if k is None:
            continue
        def hook(mod, inp, out, k = k):
            if not active[0] or k in store:
                return
            outs = out if isinstance(out, (tuple, list)) else (out,)
            store[k] = [o.detach().to("cpu").double() if o.is_floating_point() else o.detach().to("cpu")
                        for o in outs if torch.is_tensor(o) and o.dim() <= 3]  # no 4-D attention probs
        hs.append(m.register_forward_hook(hook))
    return hs


def family(name, wrapper_params):
    """Param family of a trainable LoRA tensor name."""
    ab = "A" if ".lora_A." in name else ("B" if ".lora_B." in name else "?")
    if ".self_attn." in name:
        return f"attention LoRA {ab}"
    if ".experts" in name:
        owner = name.split(".lora_")[0]
        return f"expert {wrapper_params.get(owner, '?')} LoRA {ab}"
    if ".gate." in name or ".router" in name:
        return "router"
    return f"other {ab}"


def compare_oracle(grads, acts, loss0, oracle_path):
    import torch
    o = torch.load(oracle_path, map_location = "cpu", weights_only = False)
    og, fam = o["grads"], o["families"]
    agg = {}
    for n, r in og.items():
        g = grads.get(n)
        f = fam.get(n, "?")
        a = agg.setdefault(f, {"dd": 0.0, "rr": 0.0, "gg": 0.0, "dot": 0.0, "n": 0, "missing": 0})
        if g is None or g.shape != r.shape:
            a["missing"] += 1
            continue
        g = g.double()
        a["dd"] += float(((g - r) ** 2).sum()); a["rr"] += float((r ** 2).sum())
        a["gg"] += float((g ** 2).sum()); a["dot"] += float((g * r).sum()); a["n"] += 1
    tot = {k: sum(v[k] for v in agg.values()) for k in ("dd", "rr", "gg", "dot")}
    def fin(v):
        if v["rr"] == 0.0:  # LoRA A at step 0 (B starts at zero): the exact grad is 0
            return {"rel_l2": None, "ref_zero": True, "arm_norm": v["gg"] ** 0.5, "n": v.get("n"), "missing": v.get("missing")}
        return {"rel_l2": (v["dd"] ** 0.5) / v["rr"] ** 0.5,
                "cosine": v["dot"] / max((v["gg"] * v["rr"]) ** 0.5, 1e-300), "n": v.get("n"), "missing": v.get("missing")}
    res = {"all": fin(tot), "families": {f: fin(v) for f, v in sorted(agg.items())},
           "loss_rel": abs(loss0 - o["loss"]) / abs(o["loss"]), "oracle_loss": o["loss"]}
    av = {}
    for k, ref in (o.get("acts") or {}).items():
        got = acts.get(k)
        if got is None:
            av[k] = "not called"
            continue
        row = []
        for x, y in zip(got, ref):
            if x.shape != y.shape:
                row.append(f"shape {tuple(x.shape)} vs {tuple(y.shape)}")
            elif y.is_floating_point():
                row.append(float((x.double() - y).norm() / max(float(y.norm()), 1e-300)))
            else:  # top-k expert ids: set agreement per token
                xs, ys = x.sort(-1).values, y.sort(-1).values
                row.append({"topk_rows_equal": float((xs == ys).all(-1).double().mean())})
        av[k] = row
    res["acts"] = av
    return res


def oracle():
    """CPU float64 reference: plain transformers (eager attention, eager experts) + PEFT built from the arm's own
    LoRA config and its pre-step-0 LoRA tensors; step-0 loss, LoRA grads and activations. No unsloth import."""
    import torch, transformers, peft
    from transformers import AutoModelForCausalLM
    torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
    init = torch.load(args.init, map_location = "cpu", weights_only = False)
    rec["versions"] = {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__}
    kw = dict(dtype = torch.float64, attn_implementation = "eager")
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model_dir, experts_implementation = "eager", **kw)
        rec["experts_implementation"] = "eager"
    except Exception as e:  # noqa: BLE001
        rec["experts_implementation_error"] = repr(e)[:300]
        model = AutoModelForCausalLM.from_pretrained(args.model_dir, **kw)
    cfgd = dict(init["peft_config"])
    fields = peft.LoraConfig.__dataclass_fields__
    cfg = peft.LoraConfig(**{k: v for k, v in cfgd.items() if k in fields and k not in ("peft_type", "task_type", "base_model_name_or_path", "revision", "auto_mapping", "runtime_config")})
    rec["lora_config_used"] = {k: str(v)[:200] for k, v in cfg.to_dict().items() if v not in (None, False, [], {})}
    model = peft.get_peft_model(model, cfg)
    names = {n for n, p in model.named_parameters() if p.requires_grad}
    want = set(init["tensors"])
    rec["names_only_oracle"], rec["names_only_arm"] = sorted(names - want)[:20], sorted(want - names)[:20]
    if names != want:
        raise RuntimeError(f"oracle LoRA names differ from the arm's: {len(names - want)} / {len(want - names)}")
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in init["tensors"]:
                t = init["tensors"][n]
                if t.shape != p.shape:
                    raise RuntimeError(f"{n}: shape {tuple(t.shape)} vs oracle {tuple(p.shape)}")
                p.copy_(t.double())
    wrapper_params = {}
    for n, m in model.named_modules():
        pn = getattr(m, "parameter_name", None)
        if pn:
            wrapper_params[n] = pn
    rec["wrapper_params"] = wrapper_params
    fams = {n: family(n, wrapper_params) for n in names}
    g = torch.Generator().manual_seed(1)
    ids = torch.randint(3, 30000, (args.batch, args.seqlen), generator = g)
    acts, active = {}, [True]
    act_hooks(model, acts, active)
    model.train()
    out = model(input_ids = ids, labels = ids)
    active[0] = False
    out.loss.backward()
    grads = {n: p.grad.detach().double().clone() for n, p in model.named_parameters() if p.requires_grad and p.grad is not None}
    torch.save({"grads": grads, "families": fams, "loss": float(out.loss), "acts": acts}, args.oracle)
    rec.update(loss = float(out.loss), n_grads = len(grads), families = sorted(set(fams.values())), acts_keys = sorted(acts),
               ok = True, finite = True, stage = "done")


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
    if args.save_init:
        cfg = model.peft_config["default"]
        d = cfg.to_dict() if hasattr(cfg, "to_dict") else dict(vars(cfg))
        d = {k: (sorted(v) if isinstance(v, set) else v) for k, v in d.items()}
        torch.save({"tensors": {n: p.detach().float().cpu().clone() for n, p in model.named_parameters() if p.requires_grad},
                    "peft_config": d}, args.save_init)
        rec["saved_init"] = args.save_init
    acts, active = {}, [False]
    if args.oracle_ref:
        act_hooks(model, acts, active)
    try:
        rec["moe_backend"] = mu.select_moe_backend()
    except Exception as e:  # noqa: BLE001
        rec["moe_backend"] = "ERR " + repr(e)[:200]
    rec["moe_backend_env"] = os.environ.get("UNSLOTH_MOE_BACKEND")
    rec["probe_failure"] = getattr(mu, "_GROUPED_MM_TRAINING_PROBE_FAILURE", "absent")
    rec["grouped_mm_supported"] = getattr(mu, "_TORCH_GROUPED_MM_SUPPORTED", "absent")
    rec["triton_standin"] = getattr(mu, "_GROUPED_MM_TRITON_STANDIN", "absent")
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
        active[0] = step == 0
        out = model(input_ids = ids, labels = ids)
        active[0] = False
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
            if step == 0 and args.oracle_ref and os.path.isfile(args.oracle_ref):
                try:
                    rec["oracle"] = compare_oracle(gr, acts, float(out.loss), args.oracle_ref)
                except Exception as e:  # noqa: BLE001
                    rec["oracle"] = {"error": f"{type(e).__name__}: {e}"[:400]}
                acts.clear()
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
    oracle() if args.oracle else (build() if args.build else train())
except Exception as e:  # noqa: BLE001
    rec.update(ok = False, error = f"{type(e).__name__}: {e}"[:1500], where = traceback.format_exc()[-4000:])
dump()
sys.exit(0 if rec.get("ok") else 1)
