#!/usr/bin/env python
# coding: utf-8
# Converted from: AMD-Gemma4_(26B_A4B)-Text.ipynb

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

def _amd_cell(k):
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

try:
    from transformers import TrainerCallback as _AmdTCB
except Exception:
    _AmdTCB = object
class _AmdStepTimer(_AmdTCB):
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
subprocess.run('uv pip install -qqq torchcodec', shell=True)

# ### Unsloth
# 
# `FastModel` supports loading nearly any model now! This includes Vision and Text models!

_amd_cell(3)
from unsloth import FastModel
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

model, tokenizer = FastModel.from_pretrained(
    model_name = "unsloth/gemma-4-26B-A4B-it",
    dtype = None, # None for auto detection
    max_seq_length = 8192, # Choose any for long context!
    load_in_4bit = True,  # 4 bit quantization to reduce memory
    full_finetuning = False, # [NEW!] We have full finetuning now!
    # token = "YOUR_HF_TOKEN", # HF Token for gated models
)

# # Gemma 4 can process Text, Vision and Audio!
# 
# Let's first experience how Gemma 4 can handle multimodal inputs. We use Gemma 4's recommended settings of `temperature = 1.0, top_p = 0.95, top_k = 64`

_amd_cell(4)
from transformers import TextStreamer
# Helper function for inference
def do_gemma_4_inference(messages, max_new_tokens = 128):
    _ = _amd_cg(model.generate)(
        **tokenizer.apply_chat_template(
            messages,
            add_generation_prompt = True, # Must add for generation
            tokenize = True,
            return_dict = True,
            return_tensors = "pt",
        ).to("cuda"),
        max_new_tokens = max_new_tokens,
        use_cache = True,
        temperature = 1.0, top_p = 0.95, top_k = 64,
        streamer = TextStreamer(tokenizer, skip_prompt = True),
    )

# # Gemma 4 can see images!
# 
# <img src="https://files.worldwildlife.org/wwfcmsprod/images/Sloth_Sitting_iStock_3_12_2014/story_full_width/8l7pbjmj29_iStock_000011145477Large_mini__1_.jpg" alt="Alt text" height="256">

_amd_cell(5)
sloth_link = "https://files.worldwildlife.org/wwfcmsprod/images/Sloth_Sitting_iStock_3_12_2014/story_full_width/8l7pbjmj29_iStock_000011145477Large_mini__1_.jpg"

_amd_cell(6)
messages = [{
    "role" : "user",
    "content": [
        { "type": "image", "image" : sloth_link },
        { "type": "text",  "text" : "Which films does this animal feature in?" }
    ]
}]
# You might have to wait 1 minute for Unsloth's auto compiler
do_gemma_4_inference(messages, max_new_tokens = 256)

# Let's make a poem about sloths!

messages = [{
    "role": "user",
    "content": [{ "type" : "text",
                  "text" : "Write a poem about sloths." }]
}]
do_gemma_4_inference(messages)

# # Let's finetune Gemma 4!
# 
# You can finetune the vision and text parts for now through selection - the audio part can also be finetuned - we're working to make it selectable as well!

# We now add LoRA adapters so we only need to update a small amount of parameters!

_amd_cell(7)
model = FastModel.get_peft_model(
    model,
    finetune_vision_layers     = False, # Turn off for just text!
    finetune_language_layers   = True,  # Should leave on!
    finetune_attention_modules = True,  # Attention good for GRPO
    finetune_mlp_modules       = True,  # Should leave on always!

    r = 8,           # Larger = higher accuracy, but might overfit
    lora_alpha = 8,  # Recommended alpha == r at least
    lora_dropout = 0,
    bias = "none",
    random_state = 3407,
)

# <a name="Data"></a>
# ### Data Prep
# We now use the `Gemma-4` format for conversation style finetunes. We use [Maxime Labonne's FineTome-100k](https://huggingface.co/datasets/mlabonne/FineTome-100k) dataset in ShareGPT style. Gemma-4 renders multi turn conversations like below:
# 
# ```
# <bos><|turn>user
# Hello<turn|>
# <|turn>model
# Hey there!<turn|>
# ```
# We use our `get_chat_template` function to get the correct chat template. We support `zephyr, chatml, mistral, llama, alpaca, vicuna, vicuna_old, phi3, llama3, phi4, qwen2.5, gemma3, gemma-4` and more.

_amd_cell(8)
from unsloth.chat_templates import get_chat_template
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "gemma-4-thinking",
)

# We get the first 3000 rows of the dataset

_amd_cell(9)
from datasets import load_dataset
dataset = load_dataset("mlabonne/FineTome-100k", split = "train[:3000]")

# We now use `standardize_data_formats` to try converting datasets to the correct format for finetuning purposes!

_amd_cell(10)
from unsloth.chat_templates import standardize_data_formats
dataset = standardize_data_formats(dataset)

# Let's see how row 100 looks like!

_amd_cell(11)
dataset[100]

# We now have to apply the chat template for `Gemma-3` onto the conversations, and save it to `text`. We remove the `<bos>` token using removeprefix(`'<bos>'`) since we're finetuning. The Processor will add this token before training and the model expects only one.

_amd_cell(12)
def formatting_prompts_func(examples):
   convos = examples["conversations"]
   texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False).removeprefix('<bos>') for convo in convos]
   return { "text" : texts, }

dataset = dataset.map(formatting_prompts_func, batched = True)

# Let's see how the chat template did! Notice there is no `<bos>` token as the processor tokenizer will be adding one.

_amd_cell(13)
dataset[100]["text"]

# <a name="Train"></a>
# ### Train the model
# Now let's train our model. We do 60 steps to speed things up, but you can set `num_train_epochs=1` for a full run, and turn off `max_steps=None`.

_amd_cell(14)
from trl import SFTTrainer, SFTConfig
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    eval_dataset = None, # Can set up evaluation!
    args = SFTConfig(
        dataset_text_field = "text",
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 4, # Use GA to mimic batch size!
        warmup_steps = 5,
        # num_train_epochs = 1, # Set this for 1 full training run.
        max_steps = 10,
        learning_rate = 2e-4, # Reduce to 2e-5 for long training runs
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.001,
        lr_scheduler_type = "linear",
        seed = 3407,
        report_to = "none", # Use TrackIO/WandB etc
    ),
)

# We also use Unsloth's `train_on_completions` method to only train on the assistant outputs and ignore the loss on the user's inputs. This helps increase accuracy of finetunes!

_amd_cell(15)
from unsloth.chat_templates import train_on_responses_only
trainer = train_on_responses_only(
    trainer,
    instruction_part = "<|turn>user\n",
    response_part = "<|turn>model\n",
)

# Let's verify masking the instruction part is done! Let's print the 100th row again.  Notice how the sample only has a single `<bos>` as expected!

_amd_cell(16)
tokenizer.decode(trainer.train_dataset[100]["input_ids"])

# Now let's print the masked out example - you should see only the answer is present:

_amd_cell(17)
tokenizer.decode([tokenizer.pad_token_id if x == -100 else x for x in trainer.train_dataset[100]["labels"]]).replace(tokenizer.pad_token, " ")

# @title Show current memory stats
_amd_cell(18)
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

# # Let's train the model!
# 
# To resume a training run, set `trainer.train(resume_from_checkpoint = True)`

_amd_cell(19)
trainer.add_callback(_AmdStepTimer())
trainer_stats = trainer.train()

# @title Show final memory and time stats
_amd_cell(20)
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
# Let's run the model via Unsloth native inference! According to the `Gemma-3` team, the recommended settings for inference are `temperature = 1.0, top_p = 0.95, top_k = 64`

_amd_cell(21)
from unsloth.chat_templates import get_chat_template
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "gemma-4-thinking",
)
_amd_cell(22)
messages = [{
    "role": "user",
    "content": [{
        "type" : "text",
        "text" : "Continue the sequence: 1, 1, 2, 3, 5, 8,",
    }]
}]
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt = True, # Must add for generation
    return_tensors = "pt",
    tokenize = True,
    return_dict = True,
).to("cuda")
outputs = _amd_cg(model.generate)(
    **inputs,
    max_new_tokens = 64, # Increase for longer outputs!
    use_cache = True,
    # Recommended Gemma-3 settings!
    temperature = 1.0, top_p = 0.95, top_k = 64,
)
tokenizer.batch_decode(outputs)

# You can also use a `TextStreamer` for continuous inference - so you can see the generation token by token, instead of waiting the whole time!

messages = [{
    "role": "user",
    "content": [{"type" : "text", "text" : "Why is the sky blue?",}]
}]
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt = True, # Must add for generation
    return_tensors = "pt",
    tokenize = True,
    return_dict = True,
).to("cuda")

from transformers import TextStreamer
_ = _amd_cg(model.generate)(
    **inputs,
    max_new_tokens = 64, # Increase for longer outputs!
    use_cache = True,
    # Recommended Gemma-3 settings!
    temperature = 1.0, top_p = 0.95, top_k = 64,
    streamer = TextStreamer(tokenizer, skip_prompt = True),
)

# <a name="Save"></a>
# ### Saving, loading finetuned models
# To save the final model as LoRA adapters, either use Hugging Face's `push_to_hub` for an online save or `save_pretrained` for a local save.
# 
# **[NOTE]** This ONLY saves the LoRA adapters, and not the full model. To save to 16bit or GGUF, scroll down!

_amd_cell(23)
model.save_pretrained("gemma_4_lora")  # Local saving
tokenizer.save_pretrained("gemma_4_lora")
# model.push_to_hub("HF_ACCOUNT/gemma_4_lora", token = "YOUR_HF_TOKEN") # Online saving
# tokenizer.push_to_hub("HF_ACCOUNT/gemma_4_lora", token = "YOUR_HF_TOKEN") # Online saving

# Now if you want to load the LoRA adapters we just saved for inference, set `False` to `True`:

_amd_cell(24)
if False:
    from unsloth import FastModel
    model, tokenizer = FastModel.from_pretrained(
        model_name = "gemma_4_lora", # YOUR MODEL YOU USED FOR TRAINING
        max_seq_length = 2048,
        load_in_4bit = True,
    )

messages = [{
    "role": "user",
    "content": [{"type" : "text", "text" : "What is Gemma-4?",}]
}]
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt = True, # Must add for generation
    return_tensors = "pt",
    tokenize = True,
    return_dict = True,
).to("cuda")

from transformers import TextStreamer
_ = _amd_cg(model.generate)(
    **inputs,
    max_new_tokens = 128, # Increase for longer outputs!
    # Recommended Gemma-3 settings!
    temperature = 1.0, top_p = 0.95, top_k = 64,
    streamer = TextStreamer(tokenizer, skip_prompt = True),
)

# ### Saving to float16 for VLLM
# 
# We also support saving to `float16` directly for deployment! We save it in the folder `gemma-4-finetune`. Set `if False` to `if True` to let it run!

_amd_cell(25)
if False: # Change to True to save finetune!
    model.save_pretrained_merged("gemma-4-finetune", tokenizer)

# If you want to upload / push to your Hugging Face account, set `if False` to `if True` and add your Hugging Face token and upload location!

_amd_cell(26)
if False: # Change to True to upload finetune
    model.push_to_hub_merged(
        "HF_ACCOUNT/gemma-4-finetune", tokenizer,
        token = "YOUR_HF_TOKEN"
    )

# ### GGUF / llama.cpp Conversion
# To save to `GGUF` / `llama.cpp`, we support it natively now for all models! For now, you can convert easily to `Q8_0, F16 or BF16` precision. `Q4_K_M` for 4bit will come later!

_amd_cell(27)
if False: # Change to True to save to GGUF
    model.save_pretrained_gguf(
        "gemma_4_finetune",
        tokenizer,
        quantization_method = "Q8_0", # For now only Q8_0, BF16, F16 supported
    )

# Likewise, if you want to instead push to GGUF to your Hugging Face account, set `if False` to `if True` and add your Hugging Face token and upload location!

_amd_cell(28)
if False: # Change to True to upload GGUF
    model.push_to_hub_gguf(
        "HF_ACCOUNT/gemma_4_finetune",
        tokenizer,
        quantization_method = "Q8_0", # Only Q8_0, BF16, F16 supported
        token = "YOUR_HF_TOKEN",
    )

# Now, use the `gemma-4-finetune.gguf` file or `gemma-4-finetune-Q4_K_M.gguf` file in llama.cpp.
# 
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
