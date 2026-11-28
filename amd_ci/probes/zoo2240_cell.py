#!/usr/bin/env python3
"""One cell for the unsloth-zoo PR 2240 probe (GLM-5.3-Flash / glm5_next sparse flex on ROCm). Observes only;
writes JSON via --out. Run from a checkout that HAS tests/test_glm5_next_sparse.py (the head): the tiny model,
mask layouts and the upstream-reference context come from that file, so the cell uses the PR's own fixtures.

--mode (one process each, tiny random 4-layer Glm5NextTextModel, forward + backward on the GPU above the
index budget, index_topk 64 + index_kpool 16 = 80 keys):
  on        bf16, patch installed, flex allowed (what a user gets)
  off       bf16, patch installed, UNSLOTH_SPARSE_ATTN_FLEX=0 (chunked indexer, dense masked attention)
  upstream  bf16, transformers' own indexer + attention (patch's _REFERENCE forwards)
  fp32      fp32, transformers' own path: the reference for all three
Outputs and every parameter grad are saved to --pt. Engagement from SPARSE_ATTN_STATS["*"]["glm5_next"] and the
sparse_blocks._FAILED latch. --compare DIR writes error metrics (no thresholds: the criteria judge).
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import os
import sys
import time
import traceback

CASES = {"s300_left": (300, "left"), "s600_plain": (600, "plain")}
MODES = ("on", "off", "upstream", "fp32")
TEST_FILE = "tests/test_glm5_next_sparse.py"


def _load_test_module():
    spec = importlib.util.spec_from_file_location("amd_ci_glm5_next_fixtures", os.path.join(os.getcwd(), TEST_FILE))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_cell(args) -> dict:
    rec: dict = {"mode": args.mode, "stage": "import", "cases": {}}
    if args.mode == "off":
        os.environ["UNSLOTH_SPARSE_ATTN_FLEX"] = "0"
    import torch
    import transformers
    rec["versions"] = {"torch": torch.__version__, "hip": torch.version.hip, "transformers": transformers.__version__}
    rec["cuda_available"] = torch.cuda.is_available()
    if rec["cuda_available"]:
        p = torch.cuda.get_device_properties(0)
        rec["device"] = {"name": p.name, "arch": getattr(p, "gcnArchName", None)}
    import unsloth_zoo
    rec["unsloth_zoo_file"] = unsloth_zoo.__file__
    t = _load_test_module()
    gates, sb = t.gates, t.sb
    g = t._patched()
    rec["patch_installed"] = "indexer" in g._REFERENCE
    q_probe = torch.zeros(1, 4, 32, 16, device = args.device, dtype = torch.bfloat16)
    rec["gates"] = {"sparse_flex_enabled": gates.sparse_flex_enabled(), "flex_available": sb._flex_available(),
                    "device_supported": sb._device_supported(0) if rec["cuda_available"] else None, "sparse_flex_device_ok": sb.sparse_flex_device_ok(q_probe)}
    rec["stage"] = "build"
    dtype = torch.float32 if args.mode == "fp32" else torch.bfloat16
    model = t._model().to(args.device, dtype)
    ctx = t._upstream if args.mode in ("upstream", "fp32") else contextlib.nullcontext
    saved = {}
    for case, (seq, layout) in CASES.items():
        c: dict = {"stage": "forward", "seq": seq, "layout": layout}
        rec["cases"][case] = c
        ids = torch.randint(0, 256, (2, seq), generator = torch.Generator().manual_seed(3)).to(args.device)
        mask = t._mask(2, seq, layout).to(args.device)
        gates.reset_sparse_attn_stats()
        failed_before = dict(sb._FAILED)
        model.zero_grad(set_to_none = True)
        t0 = time.time()
        try:
            with ctx():
                out = model(input_ids = ids, attention_mask = mask).last_hidden_state
                loss = (out * mask[..., None]).float().pow(2).mean()
                c["stage"] = "backward"
                loss.backward()
            if args.device == "cuda":
                torch.cuda.synchronize()
            c["stage"] = "done"
            c["loss"] = float(loss.item())
            grads = {n: p.grad.detach().float().cpu() for n, p in model.named_parameters() if p.grad is not None}
            saved[case] = {"out": out.detach().float().cpu(), "loss": c["loss"], "grads": grads, "valid_rows": mask.cpu()}
            c["finite"] = bool(torch.isfinite(out).all().item()) and all(bool(torch.isfinite(v).all()) for v in grads.values())
            c["n_grads"] = len(grads)
        except Exception as e:  # noqa: BLE001
            c["error"] = repr(e)[:1500]
            c["traceback"] = traceback.format_exc()[-4000:]
        c["seconds"] = round(time.time() - t0, 2)
        c["stats"] = {k: v.get("glm5_next", 0) for k, v in gates.SPARSE_ATTN_STATS.items() if v.get("glm5_next")}
        c["failed_new"] = {repr(k): v for k, v in sb._FAILED.items() if k not in failed_before}
    rec["failed_latch"] = {repr(k): v for k, v in sb._FAILED.items()}
    if saved:
        torch.save(saved, args.pt)
    rec["stage"] = "done"
    return rec


def _rel(a, b):
    return float(((a.float() - b.float()).abs().max() / b.float().abs().max().clamp_min(1e-6)).item())


def compare(directory) -> dict:
    import torch
    runs = {}
    for mode in MODES:
        path = os.path.join(directory, f"glm5next_{mode}.pt")
        runs[mode] = torch.load(path) if os.path.isfile(path) else None
    res: dict = {"present": {m: r is not None for m, r in runs.items()}, "cases": {}}
    for case in CASES:
        have = {m: (r or {}).get(case) for m, r in runs.items()}
        if not all(have.values()):
            res["cases"][case] = {"missing": [m for m, v in have.items() if not v]}
            continue
        ref = have["fp32"]
        rows = ref["valid_rows"]
        names = [n for n in ref["grads"] if ref["grads"][n].numel() and ref["grads"][n].abs().max() > 0]
        m: dict = {"n_grad_tensors": len(names)}
        for a, b in (("on", "fp32"), ("off", "fp32"), ("upstream", "fp32"), ("on", "upstream")):
            x, y = have[a], have[b]
            gr = {n: _rel(x["grads"][n], y["grads"][n]) for n in names if n in x["grads"] and n in y["grads"]}
            worst = max(gr, key = gr.get) if gr else None
            m[f"{a}_vs_{b}"] = {
                "loss_abs": abs(x["loss"] - y["loss"]),
                "out_rel": _rel(x["out"][rows], y["out"][rows]),
                "grad_rel_max": gr[worst] if worst else None, "grad_rel_worst_param": worst,
                "grads_missing": [n for n in names if n not in x["grads"] or n not in y["grads"]],
            }
        m["loss"] = {k: have[k]["loss"] for k in have}
        res["cases"][case] = m
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices = MODES)
    ap.add_argument("--pt")
    ap.add_argument("--compare", metavar = "DIR")
    ap.add_argument("--out", required = True)
    ap.add_argument("--device", default = "cuda", help = "cpu = local harness smoke only")
    args = ap.parse_args()
    if args.compare:
        rec = compare(args.compare)
    else:
        try:
            rec = run_cell(args)
        except Exception as e:  # noqa: BLE001
            rec = {"mode": args.mode, "fatal": repr(e)[:1500], "traceback": traceback.format_exc()[-4000:]}
    with open(args.out, "w", encoding = "utf-8") as fh:
        json.dump(rec, fh, indent = 2, default = str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
