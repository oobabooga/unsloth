"""REAP end to end on the GPU through Unsloth, for one torch install (run once per torch version).

    python reap_gpu_check.py --tree <unsloth head checkout> --out result.json

Per tiny MoE checkpoint: FastLanguageModel.from_pretrained (16-bit), reap_prune(sparsity 0.5,
save_directory), reload the saved folder with plain transformers and compare logits, then LoRA
(get_peft_model) for 3 SFT steps on the pruned model and save_pretrained_merged + reload. Also checks
that an in-memory prune refuses get_peft_model, and runs the REAP unit tests on CPU.
Every step records ok / detail; the file is written even when a step fails.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import traceback

MODELS = [
    "trl-internal-testing/tiny-Qwen3MoeForCausalLM",
    "hf-internal-testing/tiny-random-OlmoeForCausalLM",
    "trl-internal-testing/tiny-GptOssForCausalLM",
]


def rows(n = 32, vocab = 200, seed = 0):
    import torch

    g = torch.Generator().manual_seed(seed)
    lengths = torch.randint(16, 96, (n,), generator = g)
    return [torch.randint(3, vocab, (int(k),), generator = g).tolist() for k in lengths]


def one_model(name, work, result):
    import torch
    from unsloth import FastLanguageModel
    from transformers import AutoModelForCausalLM

    r = result.setdefault(name, {})
    model, tok = FastLanguageModel.from_pretrained(name, load_in_4bit = False, max_seq_length = 256)
    dtype = next(model.parameters()).dtype
    r["dtype"] = str(dtype)
    vocab = model.get_input_embeddings().weight.shape[0]
    data = rows(vocab = min(vocab, 1000))
    cfg = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
    before = next(getattr(cfg, k) for k in ("num_experts", "num_local_experts", "n_routed_experts") if getattr(cfg, k, None))
    save = os.path.join(work, name.split("/")[-1] + "_reap")
    t0 = time.time()
    FastLanguageModel.reap_prune(model, tok, sparsity = 0.5, dataset = data, device = "cuda", save_directory = save)
    torch.cuda.synchronize()
    r["reap_s"] = round(time.time() - t0, 2)
    after = next(getattr(cfg, k) for k in ("num_experts", "num_local_experts", "n_routed_experts") if getattr(cfg, k, None))
    r["experts"] = [before, after]
    stale = [n for n, p in model.named_parameters() if p.dim() >= 1 and p.shape[0] == before and "expert" in n]
    r["stale_tensors"] = stale[:5]
    ids = torch.tensor([data[0][:48]], device = "cuda")
    with torch.no_grad():
        logits = model(ids).logits.float()
    reloaded = AutoModelForCausalLM.from_pretrained(save, dtype = dtype).to("cuda").eval()
    with torch.no_grad():
        logits_r = reloaded(ids).logits.float()
    r["reload_max_abs_diff"] = float((logits - logits_r).abs().max())
    r["finite"] = bool(torch.isfinite(logits).all())
    del reloaded

    # LoRA on the pruned base, then a merged save that must reload with the pruned expert count.
    model = FastLanguageModel.get_peft_model(model, r = 8, lora_alpha = 8, target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"])
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr = 1e-3)
    model.train()
    losses = []
    for step in range(3):
        x = torch.tensor([data[step + 1][:64]], device = "cuda")
        loss = model(input_ids = x, labels = x).loss
        loss.backward()
        opt.step()
        opt.zero_grad()
        losses.append(round(float(loss), 4))
    r["lora_losses"] = losses
    merged = os.path.join(work, name.split("/")[-1] + "_merged")
    model.save_pretrained_merged(merged, tok, save_method = "merged_16bit")
    m2 = AutoModelForCausalLM.from_pretrained(merged, dtype = dtype)
    c2 = m2.config.get_text_config() if hasattr(m2.config, "get_text_config") else m2.config
    r["merged_experts"] = next(getattr(c2, k) for k in ("num_experts", "num_local_experts", "n_routed_experts") if getattr(c2, k, None))
    del m2, model
    torch.cuda.empty_cache()

    # In-memory prune (no save_directory) must refuse LoRA before wrapping anything.
    model, tok = FastLanguageModel.from_pretrained(name, load_in_4bit = False, max_seq_length = 256)
    FastLanguageModel.reap_prune(model, tok, sparsity = 0.5, dataset = data, device = "cuda")
    try:
        FastLanguageModel.get_peft_model(model, r = 8, target_modules = ["q_proj"])
        r["in_memory_lora_refused"] = False
    except RuntimeError as e:
        r["in_memory_lora_refused"] = "pruned in memory" in str(e)
    del model
    torch.cuda.empty_cache()
    r["ok"] = (
        after == before // 2 and not stale and r["finite"] and r["reload_max_abs_diff"] < 1e-2
        and r["merged_experts"] == after and r["in_memory_lora_refused"]
        and all(l == l and l < 1e4 for l in losses)
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required = True)
    ap.add_argument("--out", required = True)
    a = ap.parse_args()
    sys.path.insert(0, a.tree)
    os.environ.setdefault("UNSLOTH_COMPILE_LOCATION", tempfile.mkdtemp(prefix = "uc_"))
    result = {"tree": a.tree}
    try:
        import unsloth, torch, transformers

        result["unsloth_file"] = unsloth.__file__
        result["versions"] = {"torch": torch.__version__, "transformers": transformers.__version__}
        result["gpu"] = torch.cuda.get_device_name(0)
        result["capability"] = list(torch.cuda.get_device_capability(0))
    except Exception:
        result["import_error"] = traceback.format_exc()[-2000:]
        json.dump(result, open(a.out, "w"), indent = 1)
        return 1
    work = tempfile.mkdtemp(prefix = "reap_gpu_")
    models = result.setdefault("models", {})
    for name in MODELS:
        try:
            one_model(name, work, models)
        except Exception:
            models.setdefault(name, {})["error"] = traceback.format_exc()[-2500:]
            models[name]["ok"] = False
        json.dump(result, open(a.out, "w"), indent = 1)
    # REAP unit tests on CPU (needs pytest in this env).
    p = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_reap_prune.py", "-q", "-p", "no:cacheprovider"],
        cwd = a.tree, capture_output = True, text = True, env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""},
    )
    result["pytest"] = {"rc": p.returncode, "tail": (p.stdout + p.stderr).strip().splitlines()[-1:]}
    result["ok"] = all(m.get("ok") for m in models.values()) and p.returncode == 0
    json.dump(result, open(a.out, "w"), indent = 1)
    print("REAP_GPU_CHECK", "OK" if result["ok"] else "FAIL", json.dumps({k: v.get("ok") for k, v in models.items()}), result["pytest"]["tail"])
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
