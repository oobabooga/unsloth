#!/usr/bin/env python3
"""Probe: DPO-style head + loss + backward through unsloth_zoo's log-prob path on this GPU.

base state -> unsloth-zoo main (`chunked_hidden_states_selective_log_softmax`, autograd keeps every chunk's logits);
head state -> unsloth-zoo#1803 (`unsloth_chunked_logprobs`, backward recomputes per chunk). Observes extra peak
memory, time and the error against an fp64 oracle; judging is criteria/chunked_logprob_memory.py's job.
"""

from __future__ import annotations

import argparse, json, os, subprocess, sys, time
from pathlib import Path

ZOO = "https://github.com/unslothai/unsloth-zoo"
REF = {"base": "main", "head": "chunked-logprob-recompute"}


def zoo_checkout(state: str) -> Path:
    root = Path(os.environ.get("RUNNER_TEMP", "/tmp")) / f"zoo_{state}"
    if not (root / "unsloth_zoo").is_dir():
        subprocess.run(["git", "clone", "-q", "--depth", "1", "-b", REF[state], ZOO, str(root)], check = True)
    return root


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True, type = Path)
    ap.add_argument("--out", required = True, type = Path)
    args = ap.parse_args()
    state = "head" if "head" in args.state else "base"
    obs: dict = {"state": args.state, "zoo_ref": REF[state]}
    try:
        sys.path.insert(0, str(zoo_checkout(state)))
        os.environ["UNSLOTH_IS_PRESENT"] = "1"
        import torch
        import torch.nn.functional as F
        obs["torch"] = torch.__version__
        obs["hip"] = getattr(torch.version, "hip", None)
        obs["arch"] = getattr(torch.cuda.get_device_properties(0), "gcnArchName", None)
        if state == "head":
            from unsloth_zoo.fused_losses.chunked_logprob import unsloth_masked_logprobs as fn
            logps = lambda h, W, t: fn(h, W, t)
        else:
            from unsloth_zoo.rl_replacements import chunked_hidden_states_selective_log_softmax as fn
            logps = lambda h, W, t: fn(h, W, t.clamp(min = 0), 32) * (t != -100)
        g = torch.Generator().manual_seed(3407)
        H, V = 2048, 151936
        W = (torch.randn(V, H, generator = g) / H**0.5).cuda().bfloat16()
        results = {}
        for S in (1024, 4096):
            h0 = torch.randn(2, S, H, generator = g).cuda().bfloat16()
            ref_h = (h0.float() + 0.05 * torch.randn(2, S, H, generator = g).cuda()).bfloat16()
            t = torch.randint(0, V, (2, S), generator = g).cuda()
            t[:, : S // 10] = -100

            def dpo(pol, ref):
                return -F.logsigmoid(0.1 * ((pol[:1] - ref[:1]) - (pol[1:] - ref[1:]))).mean()

            times = []
            for i in range(3):
                h = h0.clone().requires_grad_(True)
                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                base_mem = torch.cuda.memory_allocated(); t0 = time.perf_counter()
                with torch.no_grad():
                    r = logps(ref_h, W, t).sum(-1)
                loss = dpo(logps(h, W, t).sum(-1), r)
                loss.backward(); torch.cuda.synchronize()
                times.append((time.perf_counter() - t0) * 1e3)
                peak = (torch.cuda.max_memory_allocated() - base_mem) / 2**30
            row = {"peak_extra_gib": round(peak, 3), "ms": round(sorted(times)[1], 2), "loss": float(loss),
                   "finite": bool(torch.isfinite(h.grad).all())}
            if S == 1024:
                Wd = W.double(); sel = t.clamp(min = 0)[..., None]; m = t != -100
                lp64 = lambda x: (torch.log_softmax(x @ Wd.t(), -1).gather(-1, sel).squeeze(-1) * m).sum(-1)
                with torch.no_grad():
                    r64 = lp64(ref_h.double())
                hd = h0.double().requires_grad_(True)
                l64 = dpo(lp64(hd), r64); l64.backward()
                row["loss_err"] = abs(float(loss) - float(l64))
                row["grad_rel_err"] = float((h.grad.double() - hd.grad).norm() / hd.grad.norm())
            results[str(S)] = row
            del h0, ref_h, h
            torch.cuda.empty_cache()
        obs["results"] = results
    except Exception as e:  # noqa: BLE001
        obs["error"] = f"{type(e).__name__}: {e}"
    args.out.write_text(json.dumps(obs, indent = 1), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
