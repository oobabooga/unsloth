#!/usr/bin/env python3
"""Probe: can THIS checkout's Studio train GRPO and DPO on the GPU?

Uses the checkout's own backend path (format_rl_dataset -> build_rl_trainer -> train) on a tiny
public model with a LoRA adapter, two optimizer steps per objective. Observes only: whether the RL
module exists, the device, per-step losses and grad norms, and any exception. Judging is left to
criteria/rl_trains_on_gpu.py.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from pathlib import Path

MODEL = "trl-internal-testing/tiny-Qwen3ForCausalLM"


def _history(trainer) -> list[dict]:
    keep = ("loss", "grad_norm", "reward", "reward_std")
    return [
        {k: v for k, v in row.items() if k in keep}
        for row in trainer.state.log_history
        if "loss" in row
    ]


def _model(FastLanguageModel):
    model, tok = FastLanguageModel.from_pretrained(MODEL, max_seq_length = 256, load_in_4bit = False)
    model = FastLanguageModel.get_peft_model(
        model, r = 8, lora_alpha = 16,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    return model, tok


def run(objective: str, rl, rewards, FastLanguageModel, Dataset, work: Path) -> dict:
    model, tok = _model(FastLanguageModel)
    if objective == "grpo":
        rows = [{"question": f"What is {i} + {i}?", "answer": str(2 * i)} for i in range(8)]
        settings = {"num_generations": 4, "max_completion_length": 32}
        # Rules that split a random tiny model's completions, so group advantages are non-zero.
        def rule(name, body):
            return {"name": name, "kind": "rule", "weight": 1.0, "rule": rewards.validate_rule(body)}
        specs = [
            {**rewards.get_reward("exact-answer"), "weight": 1.0},
            rule("letter_q", {"type": "regex", "pattern": "[qQ]", "mode": "search",
                           "score": {"match": 1.0, "miss": -1.0}}),
            rule("short", {"type": "length", "max_chars": 100, "score": {"over": -1.0, "under": 1.0}}),
        ]
    else:
        rows = [{"prompt": f"Say {i}.", "chosen": str(i), "rejected": "no"} for i in range(8)]
        settings, specs = {}, None
    ds, _ = rl.format_rl_dataset(Dataset.from_list(rows), objective)
    trainer = rl.build_rl_trainer(
        objective, model = model, tokenizer = tok, train_dataset = ds, eval_dataset = None,
        config_args = {
            "output_dir": str(work / objective), "max_steps": 2, "per_device_train_batch_size": 4,
            "learning_rate": 5e-6, "logging_steps": 1, "report_to": "none", "save_strategy": "no",
            "max_seq_length": 256,
        },
        settings = settings, reward_specs = specs,
    )
    trainer.train()
    hist = _history(trainer)
    losses = [h.get("loss") for h in hist]
    return {
        "steps": len(hist),
        "losses": losses,
        "grad_norms": [h.get("grad_norm") for h in hist],
        "grad_positive": any((h.get("grad_norm") or 0) > 0 for h in hist),
        "finite": bool(losses) and all(
            isinstance(x, (int, float)) and math.isfinite(x) for x in losses
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True, type = Path)
    ap.add_argument("--out", required = True, type = Path)
    args = ap.parse_args()
    obs: dict = {"state": args.state}
    backend = args.checkout / "studio" / "backend"
    obs["rl_available"] = (backend / "core" / "training" / "rl.py").is_file()
    if obs["rl_available"]:
        sys.path.insert(0, str(backend))
        try:
            import torch
            obs["device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            obs["hip"] = getattr(torch.version, "hip", None)
            obs["cuda"] = getattr(torch.version, "cuda", None)
            from unsloth import FastLanguageModel  # noqa: F401  (patches TRL first)
            from datasets import Dataset
            from core.training import rewards, rl
            work = args.out.parent / f"rl_work_{args.state}"
            for objective in ("grpo", "dpo"):
                try:
                    obs[objective] = run(objective, rl, rewards, FastLanguageModel, Dataset, work)
                except Exception as exc:  # noqa: BLE001
                    obs[objective] = {"error": f"{type(exc).__name__}: {exc}",
                                      "trace": traceback.format_exc()[-1500:]}
        except Exception as exc:  # noqa: BLE001
            obs["error"] = f"{type(exc).__name__}: {exc}"
            obs["trace"] = traceback.format_exc()[-1500:]
    args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
