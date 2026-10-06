#!/usr/bin/env python3
"""Build a tiny gpt-oss checkpoint in Unsloth's bnb-4bit layout (split experts), deterministically.

trl-internal-testing/tiny-GptOssForCausalLM stores stacked experts (experts.gate_up_proj [E, H, 2I],
experts.down_proj [E, I, H], router.weight). Unsloth's 4-bit gpt-oss path (GptOssExpertsBnb4bit, the class
unsloth/gpt-oss-*-unsloth-bnb-4bit loads into) expects per-expert Linear keys experts.gate_up_projs.{e}.weight
[2I, H], experts.down_projs.{e}.weight [H, I] and router.linear.{weight,bias}. Loading the stacked checkpoint
with load_in_4bit=True leaves those keys MISSING (random, unquantized experts), so this writes the same weights
in the split layout; FastLanguageModel(load_in_4bit=True) then quantizes every expert to NF4 on load.
Run without unsloth imported (plain transformers). Prints nothing load-bearing; --out is the directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

SRC = "trl-internal-testing/tiny-GptOssForCausalLM"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required = True, type = Path)
    args = ap.parse_args()
    import torch
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file, save_file

    src = Path(snapshot_download(SRC))
    sd = {}
    for f in sorted(src.glob("*.safetensors")):
        sd.update(load_file(str(f)))
    out = {}
    for k, v in sd.items():
        if k.endswith("mlp.experts.gate_up_proj"):          # [E, H, 2I] -> per expert Linear(H, 2I).weight [2I, H]
            for e in range(v.shape[0]):
                out[k[:-len("gate_up_proj")] + f"gate_up_projs.{e}.weight"] = v[e].t().contiguous()
        elif k.endswith("mlp.experts.gate_up_proj_bias"):
            for e in range(v.shape[0]):
                out[k[:-len("gate_up_proj_bias")] + f"gate_up_projs.{e}.bias"] = v[e].contiguous()
        elif k.endswith("mlp.experts.down_proj"):           # [E, I, H] -> Linear(I, H).weight [H, I]
            for e in range(v.shape[0]):
                out[k[:-len("down_proj")] + f"down_projs.{e}.weight"] = v[e].t().contiguous()
        elif k.endswith("mlp.experts.down_proj_bias"):
            for e in range(v.shape[0]):
                out[k[:-len("down_proj_bias")] + f"down_projs.{e}.bias"] = v[e].contiguous()
        elif k.endswith("mlp.router.weight"):
            out[k[:-len("weight")] + "linear.weight"] = v.contiguous()
        elif k.endswith("mlp.router.bias"):
            out[k[:-len("bias")] + "linear.bias"] = v.contiguous()
        else:
            out[k] = v.contiguous()
    out = {k: (t.to(torch.bfloat16) if t.is_floating_point() else t) for k, t in out.items()}
    args.out.mkdir(parents = True, exist_ok = True)
    save_file(out, str(args.out / "model.safetensors"), metadata = {"format": "pt"})
    for f in src.iterdir():
        if f.is_file() and not f.name.endswith((".safetensors", ".bin")) and not f.name.endswith(".index.json"):
            (args.out / f.name).write_bytes(f.read_bytes())
    cfg = json.loads((args.out / "config.json").read_text(encoding = "utf-8"))
    cfg.pop("quantization_config", None)
    cfg["torch_dtype"] = cfg["dtype"] = "bfloat16"
    (args.out / "config.json").write_text(json.dumps(cfg, indent = 2), encoding = "utf-8")
    h = hashlib.sha256()
    for k in sorted(out):
        h.update(k.encode())
        h.update(out[k].float().numpy().tobytes())
    (args.out / "zoo1591_ckpt.json").write_text(json.dumps({"source": SRC, "n_tensors": len(out),
        "digest": h.hexdigest()[:16], "keys_sample": sorted(out)[:12]}, indent = 2), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
