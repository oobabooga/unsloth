# Copied from scripts/w20a/dpo_cg_repro.py for AMD CI: output_dir / RESULT file via env (W20A_OUT_DIR, W20A_RESULT).
"""Tiny DPO (TRL, PEFT ref adapter) with Unsloth defaults; block CUDA graphs auto or forced.

argv: SRC_U SRC_Z KIND(dpo|kto|orpo|sft) [STEPS]
Prints RESULT json: ok / error, auto-mode state, which params got pool-clone hooks.
"""
import json
import os
import sys

SRC_U, SRC_Z, KIND = sys.argv[1:4]
STEPS = int(sys.argv[4]) if len(sys.argv) > 4 else 8
os.environ.setdefault("UNSLOTH_DISABLE_AUTO_UPDATES", "1")
sys.path[:0] = [SRC_U, SRC_Z]

import unsloth  # noqa: E402
from unsloth import FastLanguageModel  # noqa: E402

assert os.path.realpath(unsloth.__file__).startswith(os.path.realpath(SRC_U)), unsloth.__file__
import unsloth_zoo  # noqa: E402

assert os.path.realpath(unsloth_zoo.__file__).startswith(os.path.realpath(SRC_Z)), unsloth_zoo.__file__
import torch  # noqa: E402
import transformers  # noqa: E402
import trl  # noqa: E402
import peft  # noqa: E402
from datasets import Dataset  # noqa: E402
from unsloth.models import _block_compile as bc  # noqa: E402

REPO = os.environ.get("REPO", "trl-internal-testing/tiny-Qwen3ForCausalLM")
torch.manual_seed(0)
model, tok = FastLanguageModel.from_pretrained(REPO, max_seq_length = 256, load_in_4bit = False, dtype = None)
model = FastLanguageModel.get_peft_model(
    model, r = 8, lora_alpha = 16, random_state = 3407,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing = os.environ.get("GC", "unsloth"),
)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
words = [" the", " cat", " sat", " on", " a", " mat", " and", " it", " was", " good"]


def text(n, k):
    return "".join(words[(i + k) % len(words)] for i in range(n))


B = 2
n = B * STEPS
common = dict(
    output_dir = os.environ.get("W20A_OUT_DIR", "w20a_out") + "_" + KIND, per_device_train_batch_size = B, gradient_accumulation_steps = 1,
    max_steps = STEPS, learning_rate = 1e-4, logging_steps = 1, report_to = "none",
    bf16 = torch.cuda.is_bf16_supported(), fp16 = not torch.cuda.is_bf16_supported(),
    seed = 3407, save_strategy = "no", dataloader_num_workers = 0, max_grad_norm = 1.0,
)
if KIND in ("dpo", "orpo"):
    ds = Dataset.from_list([{"prompt": text(30, i), "chosen": text(30, i + 1), "rejected": text(30, i + 3)} for i in range(n)])
elif KIND == "kto":
    ds = Dataset.from_list([{"prompt": text(30, i), "completion": text(30, i + 1), "label": bool(i % 2)} for i in range(n)])
else:
    ds = Dataset.from_list([{"text": text(60, i)} for i in range(n)])
if KIND == "dpo":
    from trl import DPOConfig as C, DPOTrainer as T
    extra = {"ref_model": None}
    cfg = C(beta = 0.1, max_length = 128, **common)
elif KIND == "kto":
    try:
        from trl import KTOConfig as C, KTOTrainer as T
    except ImportError:
        from trl.experimental.kto import KTOConfig as C, KTOTrainer as T
    extra = {"ref_model": None}
    cfg = C(beta = 0.1, max_length = 128, **common)
elif KIND == "orpo":
    try:
        from trl.experimental.orpo import ORPOConfig as C, ORPOTrainer as T
    except ImportError:
        from trl import ORPOConfig as C, ORPOTrainer as T
    extra = {}
    cfg = C(beta = 0.1, max_length = 128, **common)
else:
    from trl import SFTConfig as C, SFTTrainer as T
    extra = {}
    cfg = C(max_length = 128, dataset_text_field = "text", padding_free = os.environ.get("SFT_PADDING_FREE", "0") == "1", **common)
trainer = T(model = model, args = cfg, train_dataset = ds, processing_class = tok, **extra)
hooked = []
orig_hooks = bc._param_grads_out_of_pool
res = {"kind": KIND, "torch": torch.__version__, "transformers": transformers.__version__, "trl": trl.__version__,
       "peft": peft.__version__, "env": os.environ.get("UNSLOTH_BLOCK_CUDAGRAPHS"), "gpu": torch.cuda.get_device_name(0),
       "adapters": list(getattr(model, "peft_config", {}).keys())}
try:
    trainer.train()
    res["ok"] = True
    res["losses"] = [h["loss"] for h in trainer.state.log_history if "loss" in h]
    res["grad_norms"] = [h.get("grad_norm") for h in trainer.state.log_history if "loss" in h]
except Exception as e:  # noqa: BLE001
    res["ok"] = False
    res["error"] = f"{type(e).__name__}: {str(e)[:300]}"
hk = {"default_hooked": 0, "default_total": 0, "ref_hooked": 0, "ref_total": 0}
for name, prm in model.named_parameters():
    for tag in ("default", "ref"):
        if f".{tag}." in name:
            hk[f"{tag}_total"] += 1
            hk[f"{tag}_hooked"] += int(bool(getattr(prm, "_backward_hooks", None)))
res["hooks"] = hk
st = bc._CUDAGRAPH_STATE
res["auto"] = {k: st["auto"].get(k) for k in ("phase", "decision", "why", "passes")}
res["graphs_off"] = st["off"]
res["graph_ids"] = len(st["ours"])
print("RESULT " + json.dumps(res, default = str), flush = True)
if os.environ.get("W20A_RESULT"):
    with open(os.environ["W20A_RESULT"], "w", encoding = "utf-8") as fh:
        json.dump(res, fh, default = str)
sys.exit(0 if res["ok"] else 1)
