#!/usr/bin/env python3
"""One arm of the unsloth #17492 probe: 3 LoRA SFT steps on a tiny Llama with block compile on (the default),
then record each decoder layer's block-compile mode and every 'block compile failed' line. Observes only."""
import argparse, contextlib, io, json, os, sys, traceback

ap = argparse.ArgumentParser()
ap.add_argument("--out", required = True)
ap.add_argument("--model", default = "trl-internal-testing/tiny-Qwen3ForCausalLM")
a = ap.parse_args()
res = {"stage": "start"}
buf = io.StringIO()


class Tee(io.TextIOBase):
    def write(self, s):
        buf.write(s); sys.__stdout__.write(s); return len(s)


try:
    with contextlib.redirect_stdout(Tee()):
        import unsloth
        import unsloth.models._block_compile as BC
        import torch
        from unsloth import FastLanguageModel
        res.update(unsloth_file = unsloth.__file__, torch = torch.__version__, hip = torch.version.hip,
                   has_suppress_errors = hasattr(torch._dynamo.config, "suppress_errors"),
                   has_strict_helper = hasattr(BC, "_strict_dynamo"), device = torch.cuda.get_device_name(0))
        res["stage"] = "load"
        model, tok = FastLanguageModel.from_pretrained(a.model, max_seq_length = 128, load_in_4bit = False,
                                                       dtype = torch.bfloat16)
        model = FastLanguageModel.get_peft_model(model, r = 8, lora_alpha = 16, target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], use_gradient_checkpointing = "unsloth")
        res["stage"] = "train"
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr = 1e-3)
        torch.manual_seed(0)
        ids = torch.randint(10, 1000, (2, 64), device = "cuda")
        losses = []
        model.train()
        for _ in range(3):
            out = model(input_ids = ids, labels = ids)
            out.loss.backward(); opt.step(); opt.zero_grad()
            losses.append(float(out.loss))
        res["losses"] = losses
        modes = []
        for m in model.modules():
            f = type(m).__dict__.get("forward")
            if getattr(f, "_unsloth_block_state", None) is not None:
                modes.append(str(m.__dict__.get("_unsloth_block_mode")))
        res["layer_modes"] = modes
        res["installed"] = bool(getattr(model, "_unsloth_block_compile_installed", False)) or bool(modes)
        res["stage"] = "done"
except Exception as e:
    res["error"] = f"{type(e).__name__}: {e}"[:800]
    res["traceback"] = traceback.format_exc()[-3000:]
text = buf.getvalue()
res["failed_lines"] = [l for l in text.splitlines() if "block compile failed" in l][:20]
res["stdout_tail"] = text[-3000:]
json.dump(res, open(a.out, "w"), indent = 1)
