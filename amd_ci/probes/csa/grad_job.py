"""One-step LoRA gradients through Studio's saved-attention checkpoint around regionally compiled DiT blocks.

Per family (real transformer config, fewer layers, random bf16 weights, PEFT LoRA r16, fp32 LoRA params, bf16
autocast, compile_repeated_blocks(fullgraph=True, dynamic=True), cuDNN SDPA off as Studio's trainer does), per LoRA
target set (the family preset, and "post": to_out + MLP only, which leaves q/k/v without grad in the first block),
per seed: grads of every LoRA param after one forward + backward in these arms:
  ref         eager (force_eager), no checkpoint
  eager_sac   eager, Studio main saved-attention checkpoint
  comp_nock   compiled, no checkpoint
  comp_off    compiled, plain checkpoint (UNSLOTH_DIFFUSION_GC_SAVE_ATTENTION=0 path)
  comp_sac    compiled, Studio main saved-attention checkpoint (x2: A/A)
  comp_fix    compiled, fixed saved-attention checkpoint (gcb_head.py)
Then a timing loop (steady s/step) for comp_off / comp_sac / comp_fix, ABBA-interleaved, same process.
Writes artifacts/<family>.json.
"""

import argparse
import importlib.util
import json
import os
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--families", default = "zimage,qwen21,flux3")
ap.add_argument("--res", type = int, default = 512)
ap.add_argument("--seeds", type = int, default = 3)
ap.add_argument("--tsets", default = "preset,post,mlp")
ap.add_argument("--time-steps", type = int, default = 12)
ap.add_argument("--time-res", type = int, default = 1024)
ap.add_argument("--time-rounds", type = int, default = 2)
ap.add_argument("--dump-grads", default = None, help = "dir: torch.save per family/tset {seed|arm: fp32 grads}")
ap.add_argument("--gcb-tree", default = None, help = "diffusion_gc_budget.py of the checkout under test: arm comp_tree")
ap.add_argument("--out-dir", default = None)
args = ap.parse_args()
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = args.out_dir or os.path.join(os.getcwd(), "artifacts")
os.makedirs(OUT, exist_ok = True)
SMOKE = os.environ.get("SMOKE") == "1"

import torch  # noqa: E402


def load(name, path = None):
    spec = importlib.util.spec_from_file_location(name, path or os.path.join(HERE, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gmain, ghead = load("gcb_main"), load("gcb_head")
gtree = load("gcb_tree", args.gcb_tree) if args.gcb_tree else None
dev = "cpu" if SMOKE else "cuda"
if not SMOKE:
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.matmul.allow_tf32 = False
dc = torch._dynamo.config
for attr in ("recompile_limit", "cache_size_limit"):
    if hasattr(dc, attr):
        setattr(dc, attr, max(getattr(dc, attr) or 0, 64))
dc.suppress_errors = False

from peft import LoraConfig  # noqa: E402

CFG = json.load(open(os.path.join(HERE, "configs.json")))
TARGETS = {
    "zimage": {"preset": ["to_q", "to_k", "to_v", "to_out.0", "w1", "w2", "w3"], "post": ["to_out.0", "w1", "w2", "w3"], "mlp": ["w1", "w2", "w3"]},
    "qwen21": {"preset": ["to_q", "to_k", "to_v", "to_out.0", "img_mlp.proj", "img_mlp.gate_layer", "img_mlp.out"],
               "post": ["to_out.0", "img_mlp.proj", "img_mlp.gate_layer", "img_mlp.out"],
               "mlp": ["img_mlp.proj", "img_mlp.gate_layer", "img_mlp.out"]},
    "flux3": {"preset": ["to_q", "to_k", "to_v", "to_out.0", "ff.linear_in", "ff.linear_out"],
              "post": ["to_out.0", "ff.linear_in", "ff.linear_out"], "mlp": ["ff.linear_in", "ff.linear_out"]},
}


def build(family, targets):
    import diffusers

    cfg = {k: v for k, v in CFG[family].items() if not k.startswith("_")}
    if family == "zimage":
        cls = diffusers.ZImageTransformer2DModel
        cfg.update(n_layers = 2, n_refiner_layers = 1)
        if SMOKE:
            cfg.update(dim = 64, n_heads = 2, n_kv_heads = 2, axes_dims = [8, 12, 12], cap_feat_dim = 32)
    elif family == "qwen21":
        cls = diffusers.QwenImage21Transformer2DModel
        cfg.update(num_layers = 2)
        if SMOKE:
            cfg.update(num_attention_heads = 2, attention_head_dim = 32, axes_dims_rope = [8, 12, 12], context_in_dim = 32)
    else:
        cls = diffusers.Flux3Transformer2DModel
        cfg.update(num_layers = 1, num_single_layers = 2)
        if SMOKE:
            cfg.update(num_attention_heads = 2, attention_head_dim = 32, axes_dims_rope = [8, 8, 8, 8], joint_attention_dim = 32)
    torch.manual_seed(0)
    torch.set_default_dtype(torch.bfloat16)
    with torch.device(dev):
        model = cls(**cfg)
    torch.set_default_dtype(torch.float32)
    for name in getattr(model, "_keep_in_fp32_modules", None) or []:
        for mn, mod in model.named_modules():
            if mn.split(".")[-1] == name:
                mod.float()
    model.requires_grad_(False)
    model.add_adapter(LoraConfig(r = 16, lora_alpha = 16, init_lora_weights = "gaussian", target_modules = targets))
    lora = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    for _, p in lora:
        p.data = p.data.float()
    model.enable_gradient_checkpointing(gradient_checkpointing_func = Switch())
    model.compile_repeated_blocks(fullgraph = True, dynamic = True)
    model.train()
    return model, lora, cfg


class Switch:
    """gradient_checkpointing_func whose implementation the arm picks (no recompile: it sits outside the blocks)."""

    impl = None

    def __call__(self, fn, *a, **k):
        return Switch.impl(fn, *a, **k)


def none_ckpt(fn, *a, **k):
    return fn(*a, **k)


def make_inputs(family, cfg, seed, res):
    g = torch.Generator().manual_seed(1000 + seed)
    side = res // 16
    text = 77 if not SMOKE else 16
    if family == "zimage":
        lat = torch.randn(1, cfg["in_channels"], res // 8, res // 8, generator = g)
        cap = torch.randn(text, cfg["cap_feat_dim"], generator = g) * 0.5
        extra = {"cap": cap}
    elif family == "qwen21":
        lat = torch.randn(1, cfg["in_channels"], 1, side, side, generator = g)
        extra = {"pe": torch.randn(1, text, cfg["context_in_dim"], generator = g) * 0.5}
    else:
        lat = torch.randn(1, cfg["in_channels"], side, side, generator = g)
        extra = {"pe": torch.randn(1, 80, cfg["joint_attention_dim"], generator = g) * 0.5}
    noise = torch.randn(lat.shape, generator = g)
    sig = torch.rand(1, generator = g) * 0.8 + 0.1
    return lat.to(dev), noise.to(dev), sig.to(dev), {k: v.to(dev, torch.bfloat16) for k, v in extra.items()}


def flux3_ids(rows, cols, l_coord, scale):
    ids = torch.cartesian_prod(torch.arange(1), torch.arange(rows), torch.arange(cols), l_coord).float()
    ids[:, 1:3] = ids[:, 1:3] * scale
    return ids.to(dev)


def forward(model, family, lat, noise, sig, extra):
    """Studio's family forwards (diffusion_dit_trainer _zimage_forward / _qwen21_call / _flux3_forward)."""
    noisy = ((1 - sig.reshape([-1] + [1] * (lat.ndim - 1))) * lat + sig.reshape([-1] + [1] * (lat.ndim - 1)) * noise)
    noisy = noisy.to(torch.bfloat16)
    t = sig * 1000
    with torch.autocast(dev, dtype = torch.bfloat16):
        if family == "zimage":
            x_list = list(noisy.unsqueeze(2).unbind(dim = 0))
            out = model(x_list, (1000 - t) / 1000, [extra["cap"]], return_dict = False)[0]
            pred = -torch.stack(out, dim = 0).squeeze(2)
        elif family == "qwen21":
            b, c, _f, h, w = noisy.shape
            packed = noisy.reshape(b, c, h * w).transpose(1, 2) if False else None
            from diffusers import QwenImage21Pipeline
            packed = QwenImage21Pipeline._pack_latents(noisy, b, c, h, w)
            pe = extra["pe"]
            img_mask = torch.cat([torch.zeros((b, pe.shape[1]), dtype = torch.bool, device = dev),
                                  torch.ones((b, (h * w) // 4), dtype = torch.bool, device = dev)], dim = 1)
            out = model(hidden_states = packed, timestep = t / 1000, encoder_hidden_states = pe,
                        encoder_hidden_states_mask = None, img_shapes = [[(1, h, w)]] * b, img_mask = img_mask,
                        return_dict = False)[0]
            pred = out[:, -(h * w):]
            pred = pred.transpose(1, 2).reshape(b, pred.shape[-1], 1, h, w)
        else:
            b, c, h, w = noisy.shape
            packed = noisy.flatten(2).transpose(1, 2)
            pe = extra["pe"]
            out = model(hidden_states = packed, encoder_hidden_states = pe, timestep = t / 1000,
                        img_ids = flux3_ids(h, w, torch.tensor([0]), 0.5),
                        txt_ids = flux3_ids(1, 1, torch.arange(pe.shape[1]), 1.0), return_dict = False)[0]
            pred = out.reshape(b, h, w, out.shape[-1]).permute(0, 3, 1, 2)
    target = noise - lat
    return torch.nn.functional.mse_loss(pred.float(), target.float())


def reset_lora(lora, seed):
    g = torch.Generator().manual_seed(seed)
    for _, p in lora:
        # Both factors nonzero (a mid-run step): with B = 0 the A grads would be zero and prove nothing.
        p.data.copy_((torch.randn(p.shape, generator = g) * 0.02).to(p.device))


def grads_for(model, lora, family, inputs, impl, eager):
    Switch.impl = impl
    for _, p in lora:
        p.grad = None
    ctx = torch.compiler.set_stance("force_eager") if eager else torch.compiler.set_stance("default")
    with ctx:
        loss = forward(model, family, *inputs)
        loss.backward()
    return float(loss), {n: p.grad.detach().double().cpu().clone() for n, p in lora if p.grad is not None}


def rel(a, b):
    num = sum(float((a[k] - b[k]).norm() ** 2) for k in b) ** 0.5
    den = sum(float(b[k].norm() ** 2) for k in b) ** 0.5
    return num / max(den, 1e-30)


def per_param_max(a, b):
    worst, name = 0.0, ""
    for k in b:
        d = float((a[k] - b[k]).norm() / b[k].norm().clamp_min(1e-30))
        if d > worst:
            worst, name = d, k
    return worst, name


def main():
    res = {"torch": torch.__version__, "device": torch.cuda.get_device_name() if not SMOKE else "cpu",
           "hip": getattr(torch.version, "hip", None), "cuda": torch.version.cuda,
           "arch": getattr(torch.cuda.get_device_properties(0), "gcnArchName", None) if not SMOKE else None,
           "gcb_tree": args.gcb_tree}
    try:
        import peft
        res["peft"] = peft.__version__
    except Exception:  # noqa: BLE001
        pass
    dumps = {}
    try:
        import diffusers
        res["diffusers"] = diffusers.__version__
    except Exception:  # noqa: BLE001
        pass
    os.environ.pop("UNSLOTH_DIFFUSION_GC_SAVE_ATTENTION", None)
    sac_main = gmain._Checkpoint(save_attention = True)
    sac_head = ghead._Checkpoint(save_attention = True)
    off = gmain._Checkpoint(save_attention = False)
    assert sac_main.context_fn is not None and sac_head.context_fn is not None, "SAC gate closed on this torch"
    arms = {
        "eager_sac": (sac_main, True),
        "comp_nock": (none_ckpt, False),
        "comp_off": (off, False),
        "comp_sac": (sac_main, False),
        "comp_sac_aa": (sac_main, False),
        "comp_fix": (sac_head, False),
    }
    if gtree is not None:
        sac_tree = gtree._Checkpoint(save_attention = True)
        assert sac_tree.context_fn is not None, "SAC gate closed in the checkout's module"
        arms["comp_tree"] = (sac_tree, False)
    for family in args.families.split(","):
        for tset in args.tsets.split(","):
            torch._dynamo.reset()
            from torch._dynamo.utils import counters
            counters.clear()
            t0 = time.time()
            model, lora, cfg = build(family, TARGETS[family][tset])
            entry = {"family": family, "targets": tset, "n_lora": len(lora), "seeds": []}
            for seed in range(args.seeds):
                reset_lora(lora, seed)
                inputs = make_inputs(family, cfg, seed, args.res)
                ref_loss, ref = grads_for(model, lora, family, inputs, none_ckpt, True)
                s = {"seed": seed, "ref_loss": ref_loss}
                got = {}
                for name, (impl, eager) in arms.items():
                    before = getattr(impl, "saved_attention", None)
                    loss, g = grads_for(model, lora, family, inputs, impl, eager)
                    got[name] = g
                    if args.dump_grads:
                        dumps[f"{family}|{tset}|{seed}|{name}"] = {k: v.float() for k, v in g.items()}
                    s[name] = {"loss": loss, "rel_vs_ref": rel(g, ref), "worst_param": per_param_max(g, ref),
                               "saved_calls": (getattr(impl, "saved_attention", 0) - before) if before is not None else None}
                if args.dump_grads:
                    dumps[f"{family}|{tset}|{seed}|ref"] = {k: v.float() for k, v in ref.items()}
                pairs = [("comp_tree", "comp_off"), ("comp_tree", "comp_sac"), ("comp_tree", "comp_fix")] if "comp_tree" in got else []
                for a_, b_ in pairs + [("comp_sac", "comp_off"), ("comp_fix", "comp_off"), ("comp_sac", "comp_sac_aa"),
                               ("comp_off", "comp_nock"), ("eager_sac", "comp_off"), ("comp_fix", "comp_nock")]:
                    s[f"{a_}__vs__{b_}"] = {"rel": rel(got[a_], got[b_]), "worst": per_param_max(got[a_], got[b_])}
                entry["seeds"].append(s)
                print("RESULT", family, tset, seed, json.dumps({k: (v["rel"] if isinstance(v, dict) and "rel" in v else
                      (v["rel_vs_ref"] if isinstance(v, dict) and "rel_vs_ref" in v else v)) for k, v in s.items()}),
                      flush = True)
            entry["setup_and_grads_s"] = time.time() - t0
            entry["dynamo"] = {"unique_graphs": int(counters["stats"].get("unique_graphs", 0)),
                               "calls_captured": int(counters["stats"].get("calls_captured", 0)),
                               "graph_breaks": int(sum(counters["graph_break"].values())),
                               "graph_break_reasons": list(counters["graph_break"])[:10],
                               "inductor": {k: int(v) for k, v in counters["inductor"].items()
                                            if k in ("fxgraph_cache_miss", "fxgraph_cache_hit", "pattern_matcher_count")},
                               "aot_autograd": {k: int(v) for k, v in counters["aot_autograd"].items()}}
            print("DYNAMO", family, tset, json.dumps(entry["dynamo"]), flush = True)
            if args.dump_grads:
                os.makedirs(args.dump_grads, exist_ok = True)
                torch.save(dumps, os.path.join(args.dump_grads, f"{family}__{tset}.pt"))
                dumps.clear()
            if tset == "preset" and args.time_steps > 0 and not SMOKE:
                entry["timing"] = timing(model, lora, family, cfg, {"off": off, "sac": sac_main, "fix": sac_head})
                print("TIMING", family, json.dumps(entry["timing"]["median"]), flush = True)
            res.setdefault("runs", []).append(entry)
            del model, lora
            if not SMOKE:
                torch.cuda.empty_cache()
            json.dump(res, open(os.path.join(OUT, "grad_results.json"), "w"), indent = 1)
    print("ALL DONE", flush = True)


def timing(model, lora, family, cfg, impls):
    """Steady s/step, ABBA-style interleaving of the arms, same process / card / inputs."""
    import statistics

    opt = torch.optim.AdamW([p for _, p in lora], lr = 1e-5)
    inputs = make_inputs(family, cfg, 7, args.time_res)
    order = []
    names = list(impls)
    for r in range(args.time_rounds):
        order += names if r % 2 == 0 else names[::-1]
    out = {n: [] for n in names}
    for name in order:
        Switch.impl = impls[name]
        times = []
        for step in range(args.time_steps):
            torch.cuda.synchronize()
            ts = time.perf_counter()
            loss = forward(model, family, *inputs)
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none = True)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - ts)
        out[name].append(statistics.median(times[3:]))
    return {"per_round": out, "median": {n: statistics.median(v) for n, v in out.items()},
            "peak_gib": torch.cuda.max_memory_allocated() / 2**30, "res": args.time_res}


main()
