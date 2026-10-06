#!/usr/bin/env python
# coding: utf-8
# Converted from: AMD-Gemma4_(26B_A4B)-Vision.ipynb

import subprocess
import os
import sys
import re


# Working directory (replaces Colab's /content/)
_WORKING_DIR = os.getcwd()

# ---- AMD CI instrumentation (observes only) ----
import json as _amd_json, os as _amd_os, time as _amd_time, traceback as _amd_tb, atexit as _amd_atexit
_AMD_OUT = _amd_os.environ.get("AMD_NB_OUT", "amd_nb_obs.json")
_AMD = {"cells_reached": [], "last_cell": None, "completed": False, "generates": [], "steps": [],
        "census": {"module_present": None, "wrapped": False, "import_error": None,
                   "generate": {"calls": 0, "returned": 0, "declined": 0, "compiling_calls": 0,
                                "decline_reasons": {}, "nf4_routed": 0, "bf16_moe": 0},
                   "other": {"calls": 0, "returned": 0, "declined": 0, "compiling_calls": 0,
                             "decline_reasons": {}, "nf4_routed": 0, "bf16_moe": 0}},
        "versions": {}, "train": {}}
_AMD_PHASE = ["other"]

def _amd_dump():
    try:
        with open(_AMD_OUT, "w", encoding = "utf-8") as _f:
            _amd_json.dump(_AMD, _f, indent = 2, default = str)
    except Exception:
        pass

def _amd_proxy_patch():
    # Harness, both arms (run 37418984180: TRL SFTTrainer / transformers align_special_tokens write
    # pad/eos_token_id through get_text_config(), the read-only _Gemma4KVSharedSafeProxy raises).
    # Same two methods as the open unsloth-zoo PR #1583.
    import sys as _s
    if _AMD.get("proxy_patched"):
        return
    g4 = _s.modules.get("unsloth_zoo.temporary_patches.gemma4")
    P = getattr(g4, "_Gemma4KVSharedSafeProxy", None)
    if P is None:
        return
    if "__setattr__" not in P.__dict__:
        def __setattr__(self, name, value):
            if name == "_real":
                object.__setattr__(self, name, value)
            else:
                setattr(self._real, name, value)
        def __delattr__(self, name):
            delattr(self._real, name)
        P.__setattr__ = __setattr__
        P.__delattr__ = __delattr__
        _AMD["proxy_patched"] = "applied"
    else:
        _AMD["proxy_patched"] = "native"

def _amd_cell(k):
    _amd_proxy_patch()
    _AMD["last_cell"] = k
    _AMD["cells_reached"].append(k)
    _amd_dump()

def _amd_versions():
    v = {}
    try:
        import torch as _t
        v["torch"] = _t.__version__; v["hip"] = getattr(_t.version, "hip", None); v["cuda"] = _t.version.cuda
        if _t.cuda.is_available():
            p = _t.cuda.get_device_properties(0)
            v["device"] = p.name; v["arch"] = getattr(p, "gcnArchName", None)
            v["total_mem_gb"] = round(p.total_memory / 2**30, 1)
    except Exception as e:
        v["torch_error"] = repr(e)
    import importlib
    for m in ("unsloth", "unsloth_zoo", "transformers", "trl", "peft", "bitsandbytes", "triton", "accelerate", "datasets"):
        try:
            mod = importlib.import_module(m)
            v[m] = getattr(mod, "__version__", "?"); v[m + "_file"] = getattr(mod, "__file__", None)
        except Exception as e:
            v[m] = "ERR " + repr(e)[:200]
    try:
        import bitsandbytes.cextension as _ce
        lib = getattr(_ce, "lib", None)
        v["bnb_lib"] = str(getattr(getattr(lib, "_lib", None), "_name", None) or lib)
        v["bnb_backend"] = str(getattr(_ce, "BNB_BACKEND", None))
    except Exception as e:
        v["bnb_lib"] = "ERR " + repr(e)[:200]
    try:
        v["rocm_info"] = open("/opt/rocm/.info/version", encoding = "utf-8").read().strip()
    except Exception:
        pass
    _AMD["versions"] = v

def _amd_wrap_routed():
    c = _AMD["census"]
    if c["wrapped"] or c["module_present"] is False:
        return
    try:
        import unsloth_zoo.temporary_patches.moe_routed as MR
    except Exception as e:  # absent in the before arm
        c["module_present"] = False; c["import_error"] = repr(e)[:300]
        return
    import torch as _t
    c["module_present"] = True
    real = MR.routed_moe_forward
    def routed_moe_forward(experts, hidden_states, top_k_index, top_k_weights):
        d = c[_AMD_PHASE[0]]
        d["calls"] += 1
        if _t.compiler.is_compiling():
            d["compiling_calls"] += 1
        out = real(experts, hidden_states, top_k_index, top_k_weights)
        if out is not None:
            d["returned"] += 1
        else:
            d["declined"] += 1
            if _t.is_grad_enabled():
                r = "grad_enabled"
            elif getattr(experts, "training", False):
                r = "experts.training"
            else:
                r = "other(numel=%d,dtype=%s,%s)" % (top_k_index.numel(), hidden_states.dtype, type(getattr(experts, "gate_up_proj", None)).__name__)
            d["decline_reasons"][r] = d["decline_reasons"].get(r, 0) + 1
        return out
    MR.routed_moe_forward = routed_moe_forward
    for name, key in (("_nf4_routed", "nf4_routed"), ("routed_bf16_moe", "bf16_moe")):
        f = getattr(MR, name, None)
        if f is None:
            continue
        def make(f = f, key = key):
            def w(*a, **k):
                c[_AMD_PHASE[0]][key] += 1
                return f(*a, **k)
            return w
        setattr(MR, name, make())
    c["wrapped"] = True

def _amd_cg(fn):
    def wrapper(*args, **kwargs):
        import torch as _t
        _amd_wrap_routed()
        g = _AMD["census"]["generate"]
        before = {k: g[k] for k in ("calls", "returned", "declined", "nf4_routed", "bf16_moe")}
        ids = kwargs.get("input_ids", args[0] if args else None)
        in_len = int(ids.shape[-1]) if ids is not None and hasattr(ids, "shape") else None
        _t.cuda.synchronize(); t0 = _amd_time.perf_counter()
        _AMD_PHASE[0] = "generate"
        rec = {"cell": _AMD["last_cell"], "in_len": in_len, "max_new_tokens": kwargs.get("max_new_tokens")}
        try:
            out = fn(*args, **kwargs)
        except BaseException as e:
            rec["error"] = repr(e)[:500]; _AMD["generates"].append(rec); _amd_dump()
            raise
        finally:
            _AMD_PHASE[0] = "other"
        _t.cuda.synchronize(); dt = _amd_time.perf_counter() - t0
        seq = out if isinstance(out, _t.Tensor) else getattr(out, "sequences", None)
        n_new = (int(seq.shape[-1]) - in_len) if (seq is not None and in_len is not None) else None
        rec.update(seconds = round(dt, 3), new_tokens = n_new,
                   tok_per_s = round(n_new / dt, 3) if n_new else None,
                   routed = {k: g[k] - before[k] for k in before})
        try:
            tok = globals().get("tokenizer") or globals().get("processor")
            rec["text"] = tok.decode(seq[0, in_len:].tolist(), skip_special_tokens = False)[:400]
        except Exception as e:
            rec["text_error"] = repr(e)[:200]
        _AMD["generates"].append(rec); _amd_dump()
        return out
    return wrapper

def _AmdStepTimer():
    # Lazy: importing transformers at script start would pin the pre-upgrade transformers in this
    # process (cell 2 upgrades it in a subprocess), unlike the notebook.
    from transformers import TrainerCallback
    class _T(TrainerCallback):
        def on_step_begin(self, args, state, control, **kw):
            import torch as _t
            _t.cuda.synchronize(); self._t0 = _amd_time.perf_counter()
        def on_step_end(self, args, state, control, **kw):
            import torch as _t
            _t.cuda.synchronize()
            _AMD["steps"].append(round(_amd_time.perf_counter() - self._t0, 3)); _amd_dump()
        def on_log(self, args, state, control, logs = None, **kw):
            if logs and "loss" in logs:
                _AMD["train"].setdefault("losses", []).append(logs["loss"])
            _amd_dump()
    return _T()

def _amd_exit():
    try:
        _amd_versions()
    except Exception as e:
        _AMD["versions_error"] = repr(e)
    _amd_dump()
_amd_atexit.register(_amd_exit)
# ---- end instrumentation ----

# To run this, press "*Run*" and press "*Run All*" on **AMD Dev Cloud**!
# <div class="align-center">
# <a href="https://unsloth.ai/"><img src="https://github.com/unslothai/unsloth/raw/main/images/unsloth%20new%20logo.png" width="115"></a>
# <a href="https://discord.gg/unsloth"><img src="https://github.com/unslothai/unsloth/raw/main/images/Discord button.png" width="145"></a>
# <a href="https://unsloth.ai/docs/"><img src="https://github.com/unslothai/unsloth/blob/main/images/documentation%20green%20button.png?raw=true" width="125"></a> Join Discord if you need help + ⭐ <i>Star us on <a href="https://github.com/unslothai/unsloth">Github</a> </i> ⭐
# </div>
# 
# To install Unsloth on your local device, follow [our guide](https://unsloth.ai/docs/get-started/install). This notebook is licensed [LGPL-3.0](https://github.com/unslothai/notebooks?tab=LGPL-3.0-1-ov-file#readme).
# 
# You will learn how to do [data prep](#Data), how to [train](#Train), how to [run the model](#Inference), & how to save it

# ### News

# Introducing **[Unsloth Desktop](https://unsloth.ai/docs/desktop)**, the first desktop app to run and train models. Free and open-source for macOS, Windows and Linux. [GitHub](https://github.com/unslothai/unsloth) • [Download](https://unsloth.ai/download)
# 
# <p>
# <a href="https://unsloth.ai/docs/desktop"><img src="https://raw.githubusercontent.com/unslothai/notebooks/refs/heads/main/assets/unsloth-qwen3-8.png" width="350" alt="Introducing Unsloth Desktop"></a>
# </p>
# 
# Train MoEs - DeepSeek, GLM, Qwen and gpt-oss 12x faster with 35% less VRAM. [Blog](https://unsloth.ai/docs/new/faster-moe)
# 
# Ultra Long-Context Reinforcement Learning is here with 7x more context windows! [Blog](https://unsloth.ai/docs/new/grpo-long-context)
# 
# New in Reinforcement Learning: [FP8 RL](https://unsloth.ai/docs/new/fp8-reinforcement-learning) • [Vision RL](https://unsloth.ai/docs/new/vision-reinforcement-learning-vlm-rl) • [Standby](https://unsloth.ai/docs/basics/memory-efficient-rl) • [gpt-oss RL](https://unsloth.ai/docs/new/gpt-oss-reinforcement-learning)
# 
# Visit our docs for all our [model uploads](https://unsloth.ai/docs/get-started/unsloth-model-catalog) and [notebooks](https://unsloth.ai/docs/get-started/unsloth-notebooks).

# ### Installation

# Gemma 4 requires transformers >= 5.5.0 / trl >= 0.28.0

_amd_cell(2)
import torch; torch._dynamo.config.recompile_limit = 64;
subprocess.run('uv pip install -qqq --upgrade --no-deps "transformers>=5.5.0" "huggingface_hub>=1.5.0,<2.0" "datasets==4.3.0" accelerate peft sentencepiece protobuf hf_transfer "trl>=0.28.0" timm', shell=True)
pass  # AMD-CI: torchcodec install skipped (CUDA build, OSError at `import unsloth` on ROCm)

# ### Unsloth

_amd_cell(3)
from unsloth import FastVisionModel # FastLanguageModel for LLMs
import torch

gemma4_models = [
    # Gemma-4 instruct models:
    "unsloth/gemma-4-E2B-it",
    "unsloth/gemma-4-E4B-it",
    "unsloth/gemma-4-31B-it",
    "unsloth/gemma-4-26B-A4B-it",
    # Gemma-4 base models:
    "unsloth/gemma-4-E2B",
    "unsloth/gemma-4-E4B",
    "unsloth/gemma-4-31B",
    "unsloth/gemma-4-26B-A4B",
] # More models at https://huggingface.co/unsloth

model, processor = FastVisionModel.from_pretrained(
    "unsloth/gemma-4-26B-A4B-it",
    load_in_4bit = True,
    attn_implementation = "eager",  # AMD-CI: SDPA NaN on gfx1151 # Use 4bit to reduce memory use. False for 16bit LoRA.
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for long context
)

# We now add LoRA adapters for parameter efficient fine-tuning, allowing us to train only 1% of all model parameters efficiently.
# 
# **[NEW]** We also support fine-tuning only the vision component, only the language component, or both. Additionally, you can choose to fine-tune the attention modules, the MLP layers, or both!

_amd_cell(4)
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers     = True, # False if not finetuning vision layers
    finetune_language_layers   = True, # False if not finetuning language layers
    finetune_attention_modules = True, # False if not finetuning attention layers
    finetune_mlp_modules       = True, # False if not finetuning MLP layers

    r = 32,                           # The larger, the higher the accuracy, but might overfit
    lora_alpha = 32,                  # Recommended alpha == r at least
    lora_dropout = 0,
    bias = "none",
    random_state = 3407,
    use_rslora = False,               # We support rank stabilized LoRA
    loftq_config = None,               # And LoftQ
    target_modules = "all-linear",    # Optional now! Can specify a list if needed
)

# <a name="Data"></a>
# ### Data Prep
# We'll use a sampled dataset of handwritten math formulas. The objective is to convert these images into a computer-readable format—specifically LaTeX—so they can be rendered. This is particularly useful for complex expressions.
# 
# You can access the dataset [here](https://huggingface.co/datasets/unsloth/LaTeX_OCR). The full dataset is [here](https://huggingface.co/datasets/linxy/LaTeX_OCR).

_amd_cell(5)
from datasets import load_dataset
dataset = load_dataset("unsloth/LaTeX_OCR", split = "train")

# Let's take an overview of the dataset. We'll examine the second image and its corresponding caption.

_amd_cell(6)
dataset

_amd_cell(7)
dataset[2]["image"]

_amd_cell(8)
dataset[2]["text"]

# We can also render LaTeX directly in the browser!

_amd_cell(9)
from IPython.display import display, Math, Latex

latex = dataset[3]["text"]
display(Math(latex))

# To format the dataset, all vision fine-tuning tasks should follow this format:
# 
# ```python
# [
#     {
#         "role": "user",
#         "content": [
#             {"type": "text", "text": instruction},
#             {"type": "image", "image": sample["image"]},
#         ],
#     },
#     {
#         "role": "user",
#         "content": [
#             {"type": "text", "text": instruction},
#             {"type": "image", "image": sample["image"]},
#         ],
#     },
# ]
# ```

_amd_cell(10)
instruction = "Write the LaTeX representation for this image."

def convert_to_conversation(sample):
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": instruction},
                {"type": "image", "image": sample["image"]},
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": sample["text"]}]},
    ]
    return {"messages": conversation}
pass

# Let's convert the dataset into the "correct" format for finetuning:

_amd_cell(11)
converted_dataset = [convert_to_conversation(sample) for sample in dataset]

# The first example is now structured like below:

_amd_cell(12)
converted_dataset[0]

# Lets take the Gemma 4 instruction chat template and use it in our base model

_amd_cell(13)
from unsloth import get_chat_template

processor = get_chat_template(
    processor,
    "gemma-4-thinking"
)

# Before fine-tuning, let us evaluate the base model's performance. We do not expect strong results, as it has not encountered this chat template before.

_amd_cell(14)
image = dataset[2]["image"]
instruction = "Write the LaTeX representation for this image."

messages = [
    {
        "role": "user",
        "content": [{"type": "image"}, {"type": "text", "text": instruction}],
    }
]
input_text = processor.apply_chat_template(messages, add_generation_prompt = True)
inputs = processor(
    image,
    input_text,
    add_special_tokens = False,
    return_tensors = "pt",
).to("cuda")

from transformers import TextStreamer

text_streamer = TextStreamer(processor, skip_prompt = True)
result = _amd_cg(model.generate)(**inputs, streamer = text_streamer, max_new_tokens = 128,
                        use_cache = True, temperature = 1.0, top_p = 0.95, top_k = 64)

# You can see it's absolutely terrible! It doesn't follow instructions at all

# <a name="Train"></a>
# ### Train the model
# Now let's train our model. We do 60 steps to speed things up, but you can set `num_train_epochs=1` for a full run, and turn off `max_steps=None`. We also support `DPOTrainer` and `GRPOTrainer` for reinforcement learning!
# 
# We use our new `UnslothVisionDataCollator` which will help in our vision finetuning setup.

_amd_cell(15)
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig

trainer = SFTTrainer(
    model = model,
    train_dataset = converted_dataset,
    processing_class = processor.tokenizer,
    data_collator = UnslothVisionDataCollator(model, processor),
    args = SFTConfig(
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 4,
        max_grad_norm = 0.3,
        warmup_ratio = 0.03,
        max_steps = 10,
        # num_train_epochs = 2, # Set this instead of max_steps for full training runs
        learning_rate = 2e-4,
        logging_steps = 1,
        save_strategy = "steps",
        optim = "adamw_8bit",
        weight_decay = 0.001,
        lr_scheduler_type = "cosine",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none", # For Weights and Biases or others

        # You MUST put the below items for vision finetuning:
        remove_unused_columns = False,
        dataset_text_field = "",
        dataset_kwargs = {"skip_prepare_dataset": True},
        max_length = 2048,
    )
)

# @title Show current memory stats
_amd_cell(16)
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

_amd_cell(17)
trainer.add_callback(_AmdStepTimer())
trainer_stats = trainer.train()

# @title Show final memory and time stats
_amd_cell(18)
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(
    f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training."
)
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

# <a name="Inference"></a>
# ### Inference
# Let's run the model! You can modify the instruction and input—just leave the output blank.
# 
# We'll use the best hyperparameters for inference on Gemma: `top_p=0.95`, `top_k=64`, and `temperature=1.0`.

_amd_cell(19)
image = dataset[10]["image"]
instruction = "Write the LaTeX representation for this image."

messages = [
    {
        "role": "user",
        "content": [{"type": "image"}, {"type": "text", "text": instruction}],
    }
]

input_text = processor.apply_chat_template(messages, add_generation_prompt = True)

inputs = processor(
    image,
    input_text,
    add_special_tokens = False,
    return_tensors = "pt",
).to("cuda")

from transformers import TextStreamer

text_streamer = TextStreamer(processor, skip_prompt = True)
result = _amd_cg(model.generate)(**inputs, streamer = text_streamer, max_new_tokens = 128,
                        use_cache = True, temperature = 1.0, top_p = 0.95, top_k = 64)

# <a name="Save"></a>
# ### Saving, loading finetuned models
# To save the final model as LoRA adapters, use Hugging Face’s `push_to_hub` for online saving, or `save_pretrained` for local storage.
# 
# **[NOTE]** This ONLY saves the LoRA adapters, and not the full model. To save to 16bit or GGUF, scroll down!

_amd_cell(20)
model.save_pretrained("gemma_4_lora")  # Local saving
processor.save_pretrained("gemma_4_lora")
# model.push_to_hub("your_name/gemma_4_lora", token = "YOUR_HF_TOKEN") # Online saving
# processor.push_to_hub("your_name/gemma_4_lora", token = "YOUR_HF_TOKEN") # Online saving

# Now if you want to load the LoRA adapters we just saved for inference, set `False` to `True`:

_amd_cell(21)
if False:
    from unsloth import FastVisionModel

    model, processor = FastVisionModel.from_pretrained(
        model_name = "gemma_4_lora",  # YOUR MODEL YOU USED FOR TRAINING
        load_in_4bit = True,
        attn_implementation = "eager",  # AMD-CI: SDPA NaN on gfx1151  # Set to False for 16bit LoRA
    )

sample = dataset[1]
image = sample["image"].convert("RGB")
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": sample["text"],
            },
            {
                "type": "image",
            },
        ],
    },
]
input_text = processor.apply_chat_template(messages, add_generation_prompt = True)
inputs = processor(
    image,
    input_text,
    add_special_tokens = False,
    return_tensors = "pt",
).to("cuda")

from transformers import TextStreamer

text_streamer = TextStreamer(processor.tokenizer, skip_prompt = True)
_ = _amd_cg(model.generate)(**inputs, streamer = text_streamer, max_new_tokens = 128,
                   use_cache = True, temperature = 1.0, top_p = 0.95, top_k = 64)

# ### Saving to float16 for VLLM
# 
# We also support saving to `float16` directly. Select `merged_16bit` for float16. Use `push_to_hub_merged` to upload to your Hugging Face account! You can go to https://huggingface.co/settings/tokens for your personal tokens. See [our docs](https://unsloth.ai/docs/basics/inference-and-deployment) for more deployment options.

# Select ONLY 1 to save! (Both not needed!)

# Save locally to 16bit
_amd_cell(22)
if False: model.save_pretrained_merged("unsloth_finetune", processor,)

# To export and save to your Hugging Face account
if False: model.push_to_hub_merged("YOUR_USERNAME/unsloth_finetune", processor, token = "YOUR_HF_TOKEN")

# And we're done! If you have any questions on Unsloth, we have a [Discord](https://discord.gg/unsloth) channel! If you find any bugs or want to keep updated with the latest LLM stuff, or need help, join projects etc, feel free to join our Discord!
# 
# Some other resources:
# 1. Train your own reasoning model - Llama GRPO notebook [Free Colab](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_(8B)-GRPO.ipynb)
# 2. Saving finetunes to Ollama. [Free notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3_(8B)-Ollama.ipynb)
# 3. Llama 3.2 Vision finetuning - Radiography use case. [Free Colab](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.2_(11B)-Vision.ipynb)
# 4. See notebooks for DPO, ORPO, Continued pretraining, conversational finetuning and more on our [documentation](https://unsloth.ai/docs/get-started/unsloth-notebooks)!
# 
# <div class="align-center">
#   <a href="https://unsloth.ai"><img src="https://github.com/unslothai/unsloth/raw/main/images/unsloth%20new%20logo.png" width="115"></a>
#   <a href="https://discord.gg/unsloth"><img src="https://github.com/unslothai/unsloth/raw/main/images/Discord.png" width="145"></a>
#   <a href="https://unsloth.ai/docs/"><img src="https://github.com/unslothai/unsloth/blob/main/images/documentation%20green%20button.png?raw=true" width="125"></a>
# 
#   Join Discord if you need help + ⭐️ <i>Star us on <a href="https://github.com/unslothai/unsloth">Github</a> </i> ⭐️
# </div>
# 
#   This notebook and all Unsloth notebooks are licensed [LGPL-3.0](https://github.com/unslothai/notebooks?tab=LGPL-3.0-1-ov-file#readme).


_AMD['completed'] = True
_amd_dump()
