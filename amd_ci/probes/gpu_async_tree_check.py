# SPDX-License-Identifier: AGPL-3.0-or-later
# Tiny correctness check: TRL >= 1.17 prefix-trie rows through a real Unsloth fast backbone + TRL's fused head.
# argv[1] = unsloth tree to import. Prints max |packed - unpacked| for the head arm and the base (no-unroll) arm.
import os
import sys
import warnings

tree = sys.argv[1]
sys.path.insert(0, tree)
os.environ.setdefault("UNSLOTH_COMPILE_LOCATION", os.path.join(tree, "..", "uc_async_tree"))
import unsloth  # noqa: E402
from unsloth import FastLanguageModel  # noqa: E402

assert os.path.realpath(unsloth.__file__).startswith(os.path.realpath(tree)), unsloth.__file__
import numpy as np  # noqa: E402
import torch  # noqa: E402
import trl  # noqa: E402

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    try:
        import trl.trainer.async_grpo_trainer as module
    except ImportError:
        import trl.experimental.async_grpo.async_grpo_trainer as module
    if not hasattr(module, "TreeRow"):
        from trl.experimental.async_grpo import packing as _p

        module.TreeRow, module.unpad_tree = _p.TreeRow, _p.unpad_tree
from unsloth.models import rl as unsloth_rl  # noqa: E402

print("trl", trl.__version__, "module", module.__name__, "unsloth", unsloth.__file__)
print("sentinel", getattr(module.add_fused_lm_head, "_unsloth_async_grpo_patched", False))

model, tok = FastLanguageModel.from_pretrained(
    "trl-internal-testing/tiny-Qwen3ForCausalLM",
    max_seq_length = 128,
    dtype = torch.float32,
    load_in_4bit = False,
)
model = FastLanguageModel.get_peft_model(model, r = 8, lora_alpha = 16, target_modules = ["q_proj", "v_proj"])
model.eval()
base = model.get_base_model()
print("fast backbone", unsloth_rl._is_unsloth_fast_backbone(base.base_model))
if hasattr(module, "register_tree_attention"):
    module.register_tree_attention()
    model.set_attn_implementation(module.TREE_ATTENTION)  # as AsyncGRPOTrainer.__init__ does
module.add_fused_lm_head(base, temperature = 1.0, outputs = ("log_probs", "entropy"))

rng = np.random.default_rng(0)
system = list(rng.integers(100, 5000, 5))
prompts = [system + list(rng.integers(100, 5000, 6)), system + list(rng.integers(100, 5000, 4))]
comps = [[list(rng.integers(100, 5000, n)) for n in (6, 5, 7)], [list(rng.integers(100, 5000, n)) for n in (4, 8)]]
comps[0][1][:2] = comps[0][0][:2]  # two completions share their first two tokens
samples = []
for g, (p, cs) in enumerate(zip(prompts, comps)):
    for c in cs:
        ids = [int(x) for x in p + c]
        n = len(ids)
        samples.append(
            dict(
                input_ids = torch.tensor(ids),
                completion_mask = torch.tensor([0] * len(p) + [1] * len(c)),
                old_log_probs = torch.zeros(n),
                advantage = 1.0,
                group_id = g,
                sampling_mask = [[] for _ in range(n)],
                sampler_ids = np.zeros((n, 0), dtype = "int64"),
                sampler_log_probs = np.zeros((n, 0), dtype = "float32"),
            )
        )
row = module.TreeRow()
for s in samples:
    row.add(s)
packed = row.pack()
dev = "cuda"
inputs = {k: v[None].to(dev) for k, v in packed.items()}
print("unique tokens", packed["input_ids"].numel(), "raw", sum(len(s["input_ids"]) for s in samples))


def fused(fn, **kw):
    with torch.no_grad():
        return fn(fused_lm_head = True, support_ids = None, score_ids = None, **kw)


# Reference: each sample alone.
ref = []
for s in samples:
    ids = s["input_ids"][None].to(dev)
    labels = torch.cat([s["input_ids"][1:], torch.tensor([-100])])[None].to(dev)
    out = fused(model, input_ids = ids, position_ids = torch.arange(ids.shape[1], device = dev)[None], shift_labels = labels)
    lp = out.log_probs[0]
    ref += [lp[p - 1] for p in range(1, len(s["input_ids"])) if s["completion_mask"][p]]
ref = torch.stack(ref).float()
kw = dict(input_ids = inputs["input_ids"], position_ids = inputs["position_ids"], shift_labels = inputs["shift_labels"])
tree_kw = module.unpad_tree(inputs)
idx = inputs["token_index"][0]

head_out = fused(model, **kw, **tree_kw)
head = head_out.log_probs.flatten()[idx].float()
print("HEAD arm max|packed-unpacked|", (head - ref).abs().max().item())
# Base arm (main on TRL >= 1.18: no Unsloth wrapper, TRL's fused head calls Unsloth's backbone with the tree kwargs).
trl_head = base.forward.__wrapped__ if hasattr(base.forward, "__wrapped__") else base.forward
base_out = fused(trl_head, **kw, **tree_kw)
base_lp = base_out.log_probs.flatten()[idx].float()
print("BASE arm (TRL fused head, no unroll) max|packed-unpacked|", (base_lp - ref).abs().max().item())
ok = (head - ref).abs().max().item() < 2e-3 and (base_lp - ref).abs().max().item() > 1e-2
print("VERDICT", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
