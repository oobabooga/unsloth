#!/usr/bin/env python3
"""Probe: Gemma / Gemma2 generation through Unsloth vs plain transformers eager, per state.

Observes only. For each state it builds tiny random Gemma and Gemma2 checkpoints
(head_dim 32, Gemma2 sliding_window 8 so decode crosses the window), generates
greedily through FastLanguageModel for a left padded batch and for each row alone,
then scores the same token sequences with transformers eager in a process that
never imports Unsloth. Records the max |logit| difference per case, whether the
batched rows reproduce the single-row tokens, the accelerator / flash gates, and
the result of the PR's Gemma test files where they exist at this state.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ARCHS = {
    "gemma": "trl-internal-testing/tiny-GemmaForCausalLM",
    "gemma2": "trl-internal-testing/tiny-Gemma2ForCausalLM",
}
PROMPT_LENS = [12, 7, 3]
NEW_TOKENS = 8
TEST_FILES = [
    "tests/test_gemma2_flash_decode.py",
    "tests/test_gemma2_attention_masks.py",
    "tests/test_gemma_embed_scale.py",
]


def _prompts(vocab):
    import torch

    g = torch.Generator().manual_seed(0)
    return [torch.randint(10, min(vocab, 2000), (n,), generator = g).tolist() for n in PROMPT_LENS]


def role_build(work: Path):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    for arch, repo in ARCHS.items():
        path = work / f"tiny_{arch}"
        if (path / "config.json").exists():
            continue
        config = AutoConfig.from_pretrained(repo)
        config.update(dict(hidden_size = 64, intermediate_size = 128, num_attention_heads = 2,
                           num_key_value_heads = 1, head_dim = 32, num_hidden_layers = 2))
        if arch == "gemma2":
            config.update(dict(sliding_window = int(os.environ.get("GEMMA2_PROBE_WINDOW", "8"))))
        torch.manual_seed(0)
        AutoModelForCausalLM.from_config(config).save_pretrained(path)
        AutoTokenizer.from_pretrained(repo).save_pretrained(path)


def role_unsloth(work: Path, checkout: Path, out: Path):
    import unsloth
    import torch
    import transformers
    from unsloth import FastLanguageModel
    import unsloth.models.gemma2 as g2
    from unsloth.models import _utils

    info = {
        "unsloth_file": unsloth.__file__,
        "unsloth_from_checkout": str(Path(unsloth.__file__).resolve()).startswith(str(checkout.resolve())),
        "torch": torch.__version__,
        "hip": getattr(torch.version, "hip", None),
        "transformers": transformers.__version__,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "has_flash_softcapping": bool(getattr(g2, "HAS_FLASH_ATTENTION_SOFTCAPPING", False)),
        "flash_decode_gate": getattr(g2, "_FLASH_DECODE", None),
        "has_embed_scale_helper": hasattr(_utils, "embedding_applies_scale"),
    }
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    info["dtype"] = str(dtype)
    results = {}
    for arch in ARCHS:
        model, tok = FastLanguageModel.from_pretrained(
            str(work / f"tiny_{arch}"), max_seq_length = 64, dtype = dtype, load_in_4bit = False
        )
        FastLanguageModel.for_inference(model)
        prompts = _prompts(model.config.vocab_size)
        pad = tok.pad_token_id if tok.pad_token_id is not None else 0
        width = max(len(p) for p in prompts)
        ids = torch.tensor([[pad] * (width - len(p)) + p for p in prompts], device = "cuda")
        mask = torch.tensor([[0] * (width - len(p)) + [1] * len(p) for p in prompts], device = "cuda")

        def gen(i, m):
            try:
                return _gen(i, m)
            except Exception as e:  # recorded per case; the base may crash where the head is fixed
                return repr(e)[:300]

        def _gen(i, m):
            o = model.generate(input_ids = i, attention_mask = m, max_new_tokens = NEW_TOKENS,
                               do_sample = False, output_logits = True, return_dict_in_generate = True,
                               pad_token_id = pad, eos_token_id = None)
            return o.sequences.cpu(), torch.stack(o.logits, 1).float().cpu()

        batched = gen(ids, mask)
        singles = []
        for p in prompts:
            t = torch.tensor([p], device = "cuda")
            singles.append(gen(t, torch.ones_like(t)))
        results[arch] = dict(ids = ids.cpu(), mask = mask.cpu(), batched = batched, singles = singles, pad = pad)
        del model
        torch.cuda.empty_cache()
    torch.save(dict(info = info, results = results), out)


def role_hf(work: Path, inp: Path, out: Path):
    import torch
    from transformers import AutoModelForCausalLM

    data = torch.load(inp, weights_only = False)
    dtype = getattr(torch, data["info"]["dtype"].split(".")[-1])
    report = {}
    for arch, r in data["results"].items():
        model = AutoModelForCausalLM.from_pretrained(
            str(work / f"tiny_{arch}"), torch_dtype = dtype, attn_implementation = "eager"
        ).cuda().eval()

        def ref_logits(seqs, prompt_mask):
            full_mask = torch.cat([prompt_mask, torch.ones(seqs.shape[0], NEW_TOKENS, dtype = prompt_mask.dtype)], 1)
            pos = (full_mask.cumsum(-1) - 1).clamp(min = 0)
            with torch.no_grad():
                lg = model(input_ids = seqs.cuda(), attention_mask = full_mask.cuda(),
                           position_ids = pos.cuda()).logits.float().cpu()
            start = prompt_mask.shape[1] - 1
            return lg[:, start:start + NEW_TOKENS]

        ids, mask, width = r["ids"], r["mask"], r["ids"].shape[1]
        rec = dict(errors = [])
        if isinstance(r["batched"], str):
            rec["errors"].append("batched: " + r["batched"])
            rec["batched_max_diff_per_row"] = None
            seqs = None
        else:
            seqs, logits = r["batched"]
            rec["batched_max_diff_per_row"] = (ref_logits(seqs, mask) - logits).abs().amax(dim = (1, 2)).tolist()
        single_diff, agree = [], []
        for row, single in enumerate(r["singles"]):
            if isinstance(single, str):
                rec["errors"].append(f"single row {row}: " + single)
                single_diff.append(None)
                agree.append(None)
                continue
            s_seq, s_logits = single
            s_mask = torch.ones(1, s_seq.shape[1] - NEW_TOKENS, dtype = mask.dtype)
            single_diff.append((ref_logits(s_seq, s_mask) - s_logits).abs().max().item())
            agree.append(None if seqs is None else seqs[row, width:].tolist() == s_seq[0, -NEW_TOKENS:].tolist())
        rec.update(single_max_diff = single_diff, batched_tokens_match_single = agree)
        report[arch] = rec
        del model
    out.write_text(json.dumps(report, indent = 2), encoding = "utf-8")


def _run(cmd, env, cwd, log):
    with open(log, "a", encoding = "utf-8") as fh:
        return subprocess.run(cmd, env = env, cwd = cwd, stdout = fh, stderr = subprocess.STDOUT).returncode


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state")
    ap.add_argument("--checkout", type = Path)
    ap.add_argument("--out", type = Path)
    ap.add_argument("--python", default = sys.executable)
    ap.add_argument("--transformers", default = None, help = "overlay this transformers version for both processes")
    ap.add_argument("--role", default = None)
    ap.add_argument("--work", type = Path)
    ap.add_argument("--inp", type = Path)
    ap.add_argument("--subdir", default = None)  # accepted for workflow compatibility
    a = ap.parse_args()

    if a.role == "build":
        role_build(a.work); return 0
    if a.role == "unsloth":
        role_unsloth(a.work, a.checkout, a.out); return 0
    if a.role == "hf":
        role_hf(a.work, a.inp, a.out); return 0

    obs = {"state": a.state}
    work = a.out.parent / f"gemma_probe_{a.state}"
    work.mkdir(parents = True, exist_ok = True)
    log = work / "probe.log"
    env = dict(os.environ)
    env["UNSLOTH_IS_PRESENT"] = "1"
    env["UNSLOTH_COMPILE_LOCATION"] = str(work / "compiled")
    # The runner's shared Hugging Face cache is not writable by every job.
    env["HF_HOME"] = str(a.out.parent / "hf_home")
    overlay = ""
    if a.transformers:
        overlay_dir = a.out.parent / f"tf_overlay_{a.transformers}"
        if not overlay_dir.exists():
            rc = _run([a.python, "-m", "pip", "install", "-q", "--target", str(overlay_dir),
                       f"transformers=={a.transformers}"], env, None, log)
            obs["overlay_install_rc"] = rc
        overlay = str(overlay_dir)
    me = str(Path(__file__).resolve())
    base_pp = os.pathsep.join(x for x in [overlay, env.get("PYTHONPATH", "")] if x)
    hf_env = dict(env, PYTHONPATH = base_pp)
    us_env = dict(env, PYTHONPATH = os.pathsep.join(x for x in [str(a.checkout), base_pp] if x))
    try:
        rc = _run([a.python, me, "--role", "build", "--work", str(work)], hf_env, None, log)
        if rc:
            raise RuntimeError(f"build rc={rc}")
        raw = work / "unsloth_out.pt"
        rc = _run([a.python, me, "--role", "unsloth", "--work", str(work), "--checkout", str(a.checkout),
                   "--out", str(raw)], us_env, str(a.checkout), log)
        if rc:
            raise RuntimeError(f"unsloth generate rc={rc}")
        ref = work / "hf_report.json"
        rc = _run([a.python, me, "--role", "hf", "--work", str(work), "--inp", str(raw), "--out", str(ref)],
                  hf_env, None, log)
        if rc:
            raise RuntimeError(f"hf reference rc={rc}")
        import torch

        obs["info"] = torch.load(raw, weights_only = False)["info"]
        obs["cases"] = json.loads(ref.read_text(encoding = "utf-8"))
    except Exception as e:  # observed, not judged
        obs["error"] = repr(e)

    present = [t for t in TEST_FILES if (a.checkout / t).exists()]
    obs["tests_present"] = present
    if present:
        junit = work / "junit.xml"
        rc = _run([a.python, "-m", "pytest", *present, "-q", "-p", "no:cacheprovider", f"--junitxml={junit}"],
                  us_env, str(a.checkout), log)
        obs["pytest_rc"] = rc
        passed, failed, skipped = [], [], []
        if junit.exists():
            import xml.etree.ElementTree as ET

            for case in ET.parse(junit).getroot().iter("testcase"):
                cid = f"{case.get('classname')}::{case.get('name')}"
                if case.find("failure") is not None or case.find("error") is not None:
                    failed.append(cid)
                elif case.find("skipped") is not None:
                    skipped.append(cid)
                else:
                    passed.append(cid)
        obs["pytest"] = dict(passed = passed, failed = failed, skipped = skipped)
    try:
        obs["log_tail"] = log.read_text(encoding = "utf-8", errors = "replace")[-3000:]
    except OSError:
        pass
    a.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
