#!/usr/bin/env python3
"""Fused linear + loss benchmark: Unsloth zoo (old CE, #1782, #1803, new fused_linear) vs TRL 1.15, Liger, CCE.

Each (workload, shape, head mode, arm) runs in a FRESH subprocess; arms are interleaved ABBA across
--rounds (round 0 in order, round 1 reversed, ...). Per process: one cold call (compile / JIT, reported
apart), warmup, then --reps timed steps with CUDA events (forward to loss, backward, total), plus one
memory step: peak allocated / reserved over the inputs (inputs = hidden, reference hidden, head, labels),
returned gradients included. --accuracy adds an fp64 oracle column (loss abs error, hidden-grad relative
error) on the smallest token count, with a peaked logit distribution (top-1 ~0.6, runner-up ~0.3).

Workloads (all heads frozen unless --head trainable / both):
  ce        SFT token CE over the labelled rows (25% prompt rows ignored), mean over labelled rows
  dpo       sigmoid DPO: policy log-probs with grad + reference log-probs under no_grad, pairs of S-token
            sequences (chosen = first half), 25% prompt rows ignored
  grpo      per-token log-probs with grad, clipped-ratio-free surrogate -(A * exp(logp - old)).mean()
  grpo_ent  grpo minus 0.01 * mean entropy (entropy gradient used)

Local (worktrees):
  python scripts/fused_linear_bench.py --zoo-main temp/zoo_main_wt --zoo-pr1803 impl5291_unsloth-zoo \
      --zoo-new wt_fused_linear --extra-path temp/trl115_site --extra-path temp/bench_site --tiny
Remote (clones; ships the uncommitted new tree as an overlay):
  python scripts/fused_linear_bench.py --make-overlay wt_fused_linear --overlay-base origin/main --out temp/new_overlay.tgz
  python fused_linear_bench.py --setup --new-overlay new_overlay.tgz --preset core --accuracy
The last stdout line is  FUSED_LINEAR_BENCH_RESULT <json summary>; --out holds results.json + table.md.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import statistics
import subprocess
import sys
import tarfile
import time
from pathlib import Path

RESULT_TAG = "FUSED_LINEAR_BENCH_RESULT"
ZOO_REPO = "https://github.com/unslothai/unsloth-zoo"
TRL_CHUNK = 32768  # trl.trainer.utils._CHUNKED_LOGPROB_CHUNK_SIZE in 1.15.0

# arm -> (zoo tree it needs or None, workloads it implements)
ARMS = {
    "old_zoo":        ("main",   ("ce",)),
    "main1782":       ("main",   ("dpo",)),
    "main_old":       ("main",   ("grpo",)),
    "main_recompute": ("main",   ("grpo", "grpo_ent")),
    "pr1803":         ("pr1803", ("ce", "dpo", "grpo")),
    "new":            ("new",    ("ce", "dpo", "grpo", "grpo_ent")),
    "new_routed":     ("new",    ("ce",)),
    "trl":            (None,     ("ce", "dpo", "grpo", "grpo_ent")),
    "liger":          (None,     ("ce", "dpo")),
    "cce":            (None,     ("ce",)),
    "dense":          (None,     ("ce", "dpo", "grpo", "grpo_ent")),  # stock HF / TRL: full logits, fp32 log_softmax
}  # fmt: skip
WORKLOADS = ("ce", "dpo", "grpo", "grpo_ent")
SEQ = 1024  # tokens per sequence (DPO pairs, prompt masking)
PROMPT_FRAC = 0.25
BETA = 0.1

PRESETS = {
    # token counts x (H, V); "extra" shapes run at the middle token count only
    "core": dict(tokens = [2048, 8192, 32768], shapes = [(2560, 151936)],
                 extra = [(4096, 151936), (2560, 262144), (2560, 32001)]),
    "small": dict(tokens = [2048, 8192], shapes = [(2560, 151936)], extra = [(2560, 32001)]),
    "tiny": dict(tokens = [256], shapes = [(64, 1003)], extra = [(64, 512)]),
}


# ---------------------------------------------------------------- inputs


def make_inputs(torch, spec, dev, dtype):
    """Deterministic peaked inputs: unit-norm head rows, h = 13.7 * (w_a + 0.95 w_b) + noise, so p(a) ~ 0.6,
    p(b) ~ 0.3 at V ~ 150k; targets drawn a / b / uniform with 0.6 / 0.3 / 0.1."""
    n, H, V = spec["tokens"], spec["H"], spec["V"]
    g = torch.Generator(device = "cpu").manual_seed(spec.get("seed", 3407))
    W = torch.randn(V, H, generator = g)
    W = W / W.norm(dim = 1, keepdim = True)
    a = torch.randint(0, V, (n,), generator = g)
    b = (a + 1 + torch.randint(0, V - 1, (n,), generator = g)) % V
    scale = float(spec.get("logit_scale_h", 13.7))
    h = scale * (W[a] + 0.95 * W[b]) + 0.5 * torch.randn(n, H, generator = g) / H**0.5
    u = torch.rand(n, generator = g)
    t = torch.where(u < 0.6, a, torch.where(u < 0.9, b, torch.randint(0, V, (n,), generator = g)))
    ref_h = h + 0.3 * torch.randn(n, H, generator = g) / H**0.5 * scale
    seq = min(SEQ, n)
    pos = torch.arange(n) % seq
    valid = pos >= int(PROMPT_FRAC * seq) if spec["workload"] in ("ce", "dpo") else torch.ones(n, dtype = torch.bool)
    labels = torch.where(valid, t, torch.full_like(t, -100))
    seq_id = torch.arange(n) // seq
    n_seq = int(seq_id.max()) + 1
    adv = torch.randn(n_seq, generator = g).sign()
    old = -2.0 + 0.1 * torch.randn(n, generator = g)
    out = dict(W = W.to(dev, dtype), h = h.to(dev, dtype), ref_h = ref_h.to(dev, dtype), labels = labels.to(dev),
               valid = valid.to(dev), seq_id = seq_id.to(dev), n_seq = n_seq, seq = seq, adv = adv.to(dev),
               old = old.to(dev))  # fmt: skip
    return out


def loss_from_logps(torch, wl, inp, logp, ent = None, ref_logp = None):
    """Shared loss head (arms that return per-token log-probs, and the fp64 oracle). logp / ent are [N], 0 off-mask."""
    valid = inp["valid"]
    if wl == "ce":
        return -(logp * valid).sum() / valid.sum().clamp(min = 1)
    if wl == "dpo":
        n_seq = inp["n_seq"]
        pol = torch.zeros(n_seq, dtype = logp.dtype, device = logp.device).index_add(0, inp["seq_id"], logp * valid)
        ref = torch.zeros(n_seq, dtype = logp.dtype, device = logp.device).index_add(0, inp["seq_id"], ref_logp * valid)
        half = n_seq // 2
        if half == 0:  # one sequence: compare it against itself shifted, still a nonlinear function of logps
            margin = pol[0] - ref[0]
        else:
            margin = (pol[:half] - ref[:half]) - (pol[half:2 * half] - ref[half:2 * half])
        return -torch.nn.functional.logsigmoid(BETA * margin).mean()
    adv = inp["adv"][inp["seq_id"]].to(logp.dtype)
    loss = -(adv * torch.exp(logp - inp["old"].to(logp.dtype))).mean()
    if wl == "grpo_ent":
        loss = loss - 0.01 * ent.mean()
    return loss


# ---------------------------------------------------------------- arms


class Skip(Exception):
    pass


def _flat_scatter(torch, mask, x):
    return x.new_zeros(mask.shape).masked_scatter(mask, x)


def build_arm(torch, spec, inp):
    """Return step() -> scalar loss for this arm (the caller runs backward)."""
    arm, wl = spec["arm"], spec["workload"]
    W, labels, valid = inp["W"], inp["labels"], inp["valid"]
    tclamp = labels.clamp(min = 0)
    rows = spec.get("rows")
    dev_cuda = W.is_cuda

    def need_cuda():
        if not dev_cuda:
            raise Skip("needs CUDA")

    def logps_loss(per_token_fn, ent_fn = False):
        """per_token_fn(hidden, grad: bool) -> logp [N] (or (logp, ent)); loss via the shared head."""
        def step():
            h = inp["h_leaf"]
            ref = None
            if wl == "dpo":
                with torch.no_grad():
                    ref = per_token_fn(inp["ref_h"], False)
                    ref = ref[0] if isinstance(ref, tuple) else ref
            out = per_token_fn(h, True)
            lp, ent = (out if isinstance(out, tuple) else (out, None))
            return loss_from_logps(torch, wl, inp, lp.float(), None if ent is None else ent.float(),
                                   None if ref is None else ref.float())
        return step

    want_ent = wl == "grpo_ent"

    if arm == "trl":
        need_cuda()
        try:
            from trl.kernels import ChunkedLogProbFunction as F_
        except Exception as e:  # noqa: BLE001
            raise Skip(f"trl.kernels.ChunkedLogProbFunction unavailable: {type(e).__name__}: {e}")
        outputs = ("entropy",) if want_ent else ()

        def per_token(h, grad):
            m = valid
            r = F_.apply(h[m], W, None, labels[m], 1.0, TRL_CHUNK, None, 1.0, outputs)
            lp = _flat_scatter(torch, m, r[0])
            return (lp, _flat_scatter(torch, m, r[1])) if want_ent else lp
        return logps_loss(per_token)

    if arm == "liger":
        need_cuda()
        if wl == "ce":
            try:
                from liger_kernel.transformers.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyLoss
            except Exception as e:  # noqa: BLE001
                raise Skip(f"liger-kernel unavailable: {type(e).__name__}: {e}")
            fn = LigerFusedLinearCrossEntropyLoss(ignore_index = -100, reduction = "mean")
            return lambda: fn(W, inp["h_leaf"], labels)
        try:
            from liger_kernel.chunked_loss import LigerFusedLinearDPOLoss
        except Exception as e:  # noqa: BLE001
            raise Skip(f"liger DPO unavailable: {type(e).__name__}: {e}")
        n_seq, seq = inp["n_seq"], inp["seq"]
        if n_seq < 2:
            raise Skip("liger DPO needs >= 2 sequences")
        fn = LigerFusedLinearDPOLoss(ignore_index = -100, beta = BETA, use_ref_model = True)
        H = W.shape[1]

        def step():
            out = fn(W, inp["h_leaf"].view(n_seq, seq, H), labels.view(n_seq, seq),
                     ref_input = inp["ref_h"].view(n_seq, seq, H), ref_weight = W)
            return out[0] if isinstance(out, tuple) else out
        return step

    if arm == "cce":
        need_cuda()
        try:
            from cut_cross_entropy import linear_cross_entropy
        except Exception as e:  # noqa: BLE001
            raise Skip(f"cut-cross-entropy unavailable: {type(e).__name__}: {e}")
        return lambda: linear_cross_entropy(inp["h_leaf"], W, labels, ignore_index = -100, reduction = "mean")

    if arm == "dense":
        # What a run without any fused head does: full [N, V] logits in the compute dtype, upcast, log_softmax.
        if wl == "ce":
            return lambda: torch.nn.functional.cross_entropy((inp["h_leaf"] @ W.t()).float(), labels, ignore_index = -100)

        def per_token(h, grad):
            lsm = (h @ W.t()).float().log_softmax(-1)
            lp = lsm.gather(-1, tclamp[:, None])[:, 0]
            if want_ent:
                return lp, -(lsm.exp() * lsm).sum(-1)
            return lp
        return logps_loss(per_token)

    # zoo arms
    if arm in ("old_zoo", "pr1803", "new_routed") and wl == "ce":
        from unsloth_zoo.fused_losses.cross_entropy_loss import unsloth_fused_ce_loss

        H = W.shape[1]
        return lambda: unsloth_fused_ce_loss(None, inp["h_leaf"].view(1, -1, H), W, None, labels.view(1, -1),
                                             shift_labels = False)
    if arm == "new":
        try:
            from unsloth_zoo.fused_losses.fused_linear import fused_linear_cross_entropy, fused_linear_logprobs
        except Exception as e:  # noqa: BLE001
            raise Skip(f"new tree has no fused_linear API: {type(e).__name__}: {e}")
        if wl == "ce":
            n_valid = valid.sum().float()
            return lambda: fused_linear_cross_entropy(inp["h_leaf"], W, None, labels, divisor = n_valid,
                                                      rows_per_chunk = rows)
        mask = valid if wl == "dpo" else None

        def per_token(h, grad):
            r = fused_linear_logprobs(h, W, tclamp, mask = mask, return_entropy = want_ent, rows_per_chunk = rows)
            return r if want_ent else (r[0] if isinstance(r, tuple) else r)
        return logps_loss(per_token)
    if arm == "pr1803":
        from unsloth_zoo.fused_losses.chunked_logprob import unsloth_chunked_logprobs, unsloth_masked_logprobs

        if wl == "dpo":
            return logps_loss(lambda h, grad: unsloth_masked_logprobs(h, W, labels))
        return logps_loss(lambda h, grad: unsloth_chunked_logprobs(h, W, tclamp))
    if arm == "main1782":
        from unsloth_zoo.fused_losses.preference_loss import fused_linear_token_logps

        return logps_loss(lambda h, grad: fused_linear_token_logps(h, W, None, tclamp, valid)["per_token_logps"])
    if arm in ("main_old", "main_recompute"):
        import unsloth_zoo.rl_replacements as rr

        H = W.shape[1]
        if arm == "main_old":
            f = rr.chunked_hidden_states_selective_log_softmax
            return logps_loss(lambda h, grad: f(h.view(1, -1, H), W, tclamp.view(1, -1)).view(-1))
        f = rr.chunked_hidden_states_selective_log_softmax_recompute

        def per_token(h, grad):
            r = f(h.view(1, -1, H), W, tclamp.view(1, -1), return_entropy = want_ent)
            if want_ent:
                return r[0].view(-1), r[1].view(-1)
            return (r[0] if isinstance(r, tuple) else r).view(-1)
        return logps_loss(per_token)
    raise Skip(f"arm {arm} does not implement {wl}")


# ---------------------------------------------------------------- fp64 oracle


def oracle(torch, wl, inp, rows = 256):
    """fp64 loss and hidden gradient of the policy, from the same (already rounded) inputs."""
    W = inp["W"].double()
    valid = inp["valid"]
    t = inp["labels"].clamp(min = 0)

    def stats(h):
        n = h.shape[0]
        lp, ent = torch.zeros(n, dtype = torch.float64, device = h.device), torch.zeros(n, dtype = torch.float64, device = h.device)
        for a in range(0, n, rows):
            z = h[a:a + rows].double() @ W.t()
            lse = torch.logsumexp(z, -1)
            lp[a:a + rows] = z.gather(1, t[a:a + rows, None]).squeeze(1) - lse
            p = torch.exp(z - lse[:, None])
            ent[a:a + rows] = lse - (p * z).sum(-1)
        return lp * valid, ent * valid

    lp, ent = stats(inp["h"])
    ref_lp = stats(inp["ref_h"])[0] if wl == "dpo" else None
    lp_l, ent_l = lp.clone().requires_grad_(True), ent.clone().requires_grad_(True)
    loss = loss_from_logps(torch, wl, inp, lp_l, ent_l, ref_lp)
    g_lp, g_ent = torch.autograd.grad(loss, [lp_l, ent_l], allow_unused = True)
    g_lp = torch.zeros_like(lp) if g_lp is None else g_lp * valid
    g_ent = torch.zeros_like(lp) if g_ent is None else g_ent * valid
    h = inp["h"]
    grad = torch.zeros(h.shape, dtype = torch.float64, device = h.device)
    for a in range(0, h.shape[0], rows):
        z = h[a:a + rows].double() @ W.t()
        lse = torch.logsumexp(z, -1, keepdim = True)
        p = torch.exp(z - lse)
        dz = -p * g_lp[a:a + rows, None]
        dz.scatter_add_(1, t[a:a + rows, None], g_lp[a:a + rows, None])
        H_ = (lse.squeeze(1) - (p * z).sum(-1))[:, None]
        dz -= g_ent[a:a + rows, None] * p * (z - lse + H_)
        grad[a:a + rows] = dz @ W
    return float(loss), grad


# ---------------------------------------------------------------- worker


def worker(spec):
    for p in reversed(spec.get("sys_path", [])):
        sys.path.insert(0, p)
    os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")
    for k, v in (spec.get("env") or {}).items():
        os.environ[k] = str(v)
    import torch

    res = dict(arm = spec["arm"], workload = spec["workload"])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cpu" and not spec.get("allow_cpu"):
        return {**res, "skip": "no CUDA device"}
    if dev == "cpu":
        os.environ.setdefault("UNSLOTH_ALLOW_CPU", "1")  # zoo's own CPU test mode
    if dev == "cuda":
        cap = torch.cuda.get_device_capability()
        dtype = torch.bfloat16 if (cap[0] >= 8 or torch.version.hip) else torch.float16
    else:
        dtype = torch.float32
    res["dtype"] = str(dtype).replace("torch.", "")
    inp = make_inputs(torch, spec, dev, dtype)
    trainable = spec.get("head") == "trainable"
    if trainable:
        inp["W"].requires_grad_(True)
    inp["h_leaf"] = inp["h"].clone().requires_grad_(True)
    tree = spec.get("tree")
    try:
        step = build_arm(torch, spec, inp)
    except Skip as e:
        return {**res, "skip": str(e)}
    except Exception as e:  # noqa: BLE001
        return {**res, "error": f"build: {type(e).__name__}: {str(e)[:400]}"}
    if tree:
        import unsloth_zoo

        res["zoo_file"] = unsloth_zoo.__file__
        if not os.path.realpath(unsloth_zoo.__file__).startswith(os.path.realpath(tree)):
            return {**res, "error": f"VOID: imported unsloth_zoo from {unsloth_zoo.__file__}, not {tree}"}
    sync = torch.cuda.synchronize if dev == "cuda" else (lambda: None)

    def clear():
        inp["h_leaf"].grad = None
        if trainable:
            inp["W"].grad = None

    def run_once():
        loss = step()
        loss.backward()
        return loss

    try:
        sync(); t0 = time.perf_counter()
        loss = run_once()
        sync(); res["cold_ms"] = round((time.perf_counter() - t0) * 1e3, 3)
        res["loss"] = float(loss)
        if not torch.isfinite(loss.detach().float()).item():
            res["nonfinite_loss"] = True
        gnorm = inp["h_leaf"].grad
        res["grad_hidden_norm"] = None if gnorm is None else float(gnorm.float().norm())
        if gnorm is None or float(gnorm.float().norm()) == 0.0:
            res["error"] = "VOID: no hidden gradient reached the policy hidden states"
            return res
        if spec.get("accuracy"):
            g_arm = inp["h_leaf"].grad.double()
            cache = spec.get("oracle_cache")
            ref = None
            if cache and os.path.exists(cache):
                ref = torch.load(cache, map_location = dev)
            if ref is None:
                l64, g64 = oracle(torch, spec["workload"], inp)
                ref = {"loss": l64, "grad": g64}
                if cache:
                    torch.save(ref, cache)
            res["loss_abs_err"] = abs(float(loss) - ref["loss"])
            res["grad_rel_err"] = float((g_arm - ref["grad"]).norm() / ref["grad"].norm().clamp_min(1e-300))
            res["oracle_loss"] = ref["loss"]
        clear()
        for _ in range(spec.get("warmup", 3)):
            run_once(); clear()
        sync()
        if dev == "cuda":
            torch.cuda.empty_cache()
            base_a, base_r = torch.cuda.memory_allocated(), torch.cuda.memory_reserved()
            torch.cuda.reset_peak_memory_stats()
            run_once()
            sync()
            res["peak_extra_alloc_gib"] = round((torch.cuda.max_memory_allocated() - base_a) / 2**30, 4)
            res["peak_extra_reserved_gib"] = round((torch.cuda.max_memory_reserved() - base_r) / 2**30, 4)
            res["free_mib"] = round(torch.cuda.mem_get_info()[0] / 2**20)
            clear()
        fwd, bwd, tot = [], [], []
        for _ in range(spec.get("reps", 10)):
            if dev == "cuda":
                e = [torch.cuda.Event(enable_timing = True) for _ in range(3)]
                e[0].record(); loss = step(); e[1].record(); loss.backward(); e[2].record()
                sync()
                fwd.append(e[0].elapsed_time(e[1])); bwd.append(e[1].elapsed_time(e[2])); tot.append(e[0].elapsed_time(e[2]))
            else:
                t0 = time.perf_counter(); loss = step(); t1 = time.perf_counter(); loss.backward(); t2 = time.perf_counter()
                fwd.append((t1 - t0) * 1e3); bwd.append((t2 - t1) * 1e3); tot.append((t2 - t0) * 1e3)
            clear()
        res.update(fwd_ms = fwd, bwd_ms = bwd, total_ms = tot)
    except Skip as e:
        return {**res, "skip": str(e)}
    except torch.OutOfMemoryError as e:
        return {**res, "error": f"OOM: {str(e)[:300]}"}
    except Exception as e:  # noqa: BLE001
        import traceback

        return {**res, "error": f"{type(e).__name__}: {str(e)[:400]}", "tb": traceback.format_exc()[-1500:]}
    return res


# ---------------------------------------------------------------- setup / overlay


def sh(*cmd, check = True, **kw):
    print("+", " ".join(map(str, cmd)), flush = True)
    return subprocess.run([str(c) for c in cmd], check = check, **kw)


def make_overlay(wt, base, out):
    """Tar every file that differs from `base` in worktree `wt` (tracked edits + untracked files)."""
    wt = Path(wt)
    base_sha = subprocess.check_output(["git", "-C", wt, "rev-parse", base], text = True).strip()
    changed = subprocess.check_output(["git", "-C", wt, "diff", "--name-only", "--no-renames", base], text = True).split()
    untracked = subprocess.check_output(["git", "-C", wt, "ls-files", "--others", "--exclude-standard"], text = True).split()
    files = sorted(set(changed + untracked))
    deleted = [f for f in files if not (wt / f).exists()]
    keep = [f for f in files if (wt / f).is_file()]
    head = subprocess.check_output(["git", "-C", wt, "rev-parse", "HEAD"], text = True).strip()
    meta = dict(base = base_sha, head = head, dirty = bool(untracked) or bool(
        subprocess.check_output(["git", "-C", wt, "status", "--porcelain"], text = True).strip()), deleted = deleted)
    with tarfile.open(out, "w:gz") as tf:
        for f in keep:
            tf.add(wt / f, arcname = f)
        data = json.dumps(meta).encode()
        info = tarfile.TarInfo("OVERLAY_META.json")
        info.size = len(data)
        import io

        tf.addfile(info, io.BytesIO(data))
    print(json.dumps({**meta, "files": len(keep), "out": str(out)}))


def clone_tree(work, name, ref, repo = ZOO_REPO):
    d = Path(work) / f"zoo_{name}"
    if not (d / "unsloth_zoo").is_dir():
        sh("git", "clone", "-q", "--filter=blob:none", repo, d)
    sh("git", "-C", d, "fetch", "-q", "origin", ref)
    sh("git", "-C", d, "checkout", "-q", "--force", "FETCH_HEAD")
    sh("git", "-C", d, "clean", "-qfdx")
    return d


def apply_overlay(work, tgz):
    with tarfile.open(tgz) as tf:
        meta = json.loads(tf.extractfile("OVERLAY_META.json").read())
        d = clone_tree(work, "new", meta["base"])
        tf.extractall(d, filter = "data") if sys.version_info >= (3, 12) else tf.extractall(d)
    for f in meta.get("deleted", []):
        p = d / f
        if p.exists():
            p.unlink()
    (d / "OVERLAY_META.json").unlink(missing_ok = True)
    print("overlay", json.dumps(meta), flush = True)
    return d, meta


def setup(args):
    """Remote: install what is missing, clone the trees. Never touches torch."""
    pip = [sys.executable, "-m", "pip", "install", "-q"]

    def have(mod):
        return subprocess.run([sys.executable, "-c", f"import {mod}"], capture_output = True).returncode == 0

    if not have("unsloth_zoo"):
        sh(*pip, "unsloth_zoo")  # its dependencies; the trees below shadow the package itself
    if not have("unsloth"):
        sh(*pip, "--no-deps", "git+https://github.com/unslothai/unsloth")
    for mod, req in (("trl", "trl==1.15.0"), ("liger_kernel", "liger-kernel"), ("cut_cross_entropy", "cut-cross-entropy")):
        if not have(mod):
            sh(*pip, "--no-deps", req, check = False)
    work = Path(args.work)
    work.mkdir(parents = True, exist_ok = True)
    trees = {}
    if not args.zoo_main:
        trees["main"] = str(clone_tree(work, "main", args.main_ref))
    if not args.zoo_pr1803:
        trees["pr1803"] = str(clone_tree(work, "pr1803", args.pr1803_ref))
    if not args.zoo_new:
        if args.new_overlay:
            d, meta = apply_overlay(work, args.new_overlay)
            trees["new"] = str(d)
            trees["new_meta"] = meta
        elif args.new_ref:
            trees["new"] = str(clone_tree(work, "new", args.new_ref, args.new_repo))
    return trees


# ---------------------------------------------------------------- orchestrator


def configs(args):
    p = PRESETS[args.preset]
    tokens = [int(x) for x in args.tokens.split(",")] if args.tokens else p["tokens"]
    shapes = [tuple(int(y) for y in x.split("x")) for x in args.shapes.split(",")] if args.shapes else p["shapes"]
    extra = [] if args.no_extra else p["extra"]
    mid = tokens[len(tokens) // 2]
    out = []
    wls = args.workloads.split(",")
    heads = ["frozen", "trainable"] if args.head == "both" else [args.head]
    for wl in wls:
        for head in heads:
            for H, V in shapes:
                for n in tokens:
                    out.append(dict(workload = wl, head = head, tokens = n, H = H, V = V))
            for H, V in extra:
                out.append(dict(workload = wl, head = head, tokens = mid, H = H, V = V))
    return out


def arm_variants(args, wl):
    arms = [a for a in (args.arms.split(",") if args.arms else ARMS) if wl in ARMS[a][1]]
    out = []
    for a in arms:
        out.append((a, None))
        if a == "new" and args.sweep_rows:
            out += [(a, int(r)) for r in args.sweep_rows.split(",")]
        if a == "new" and args.new_env:
            out += [(a, f"E:{tag}") for tag in _env_variants(args)]
    return out


def _env_variants(args):
    # --new-env "half:UNSLOTH_LOGPROB_FP32_LOGITS=0,exact:UNSLOTH_LOGPROB_FP32_LOGITS=2" -> {tag: {VAR: value}}
    out = {}
    for item in (args.new_env or "").split(","):
        if item.strip():
            tag, kv = item.split(":", 1)
            k, v = kv.split("=", 1)
            out.setdefault(tag.strip(), {})[k.strip()] = v.strip()
    return out


def _label(arm, rows):
    if rows is None:
        return arm
    return f"{arm}@{rows[2:]}" if isinstance(rows, str) else f"{arm}@R{rows}"


def summarize(runs):
    tot = [x for r in runs for x in r.get("total_ms", [])]
    if not tot:
        return {}
    meds = [statistics.median(r["total_ms"]) for r in runs if r.get("total_ms")]
    med = statistics.median(tot)
    s = dict(total_ms = round(med, 3),
             fwd_ms = round(statistics.median([x for r in runs for x in r.get("fwd_ms", [])]), 3),
             bwd_ms = round(statistics.median([x for r in runs for x in r.get("bwd_ms", [])]), 3),
             spread_pct = round(100 * (max(meds) - min(meds)) / med, 2) if len(meds) > 1 else None,
             rep_spread_pct = round(100 * (max(tot) - min(tot)) / med, 2),
             cold_ms = max(r.get("cold_ms", 0) for r in runs),
             processes = len(runs))  # fmt: skip
    for k in ("peak_extra_alloc_gib", "peak_extra_reserved_gib"):
        v = [r[k] for r in runs if k in r]
        if v:
            s[k] = max(v)
    for k in ("loss_abs_err", "grad_rel_err"):
        v = [r[k] for r in runs if k in r]
        if v:
            s[k] = max(v)
    return s


def table(summary):
    lines = ["| workload | head | tokens | H | V | arm | total ms | fwd ms | bwd ms | spread % (A/A) | cold ms | "
             "peak alloc GiB | peak reserved GiB | loss err | grad rel err | note |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
    f = lambda v, d = 3: "-" if v is None else (f"{v:.{d}g}" if isinstance(v, float) else str(v))  # noqa: E731
    for row in summary:
        s = row.get("stats") or {}
        note = row.get("skip") or row.get("error") or ""
        lines.append("| " + " | ".join(str(x) for x in [
            row["workload"], row["head"], row["tokens"], row["H"], row["V"], row["arm_label"],
            f(s.get("total_ms"), 5), f(s.get("fwd_ms"), 4), f(s.get("bwd_ms"), 4), f(s.get("spread_pct")),
            f(s.get("cold_ms"), 4), f(s.get("peak_extra_alloc_gib")), f(s.get("peak_extra_reserved_gib")),
            f(s.get("loss_abs_err")), f(s.get("grad_rel_err")), note[:120].replace("|", "/")]) + " |")
    return "\n".join(lines)


def env_info():
    info = dict(python = sys.version.split()[0], host = os.uname().nodename, loadavg = os.getloadavg(),
                nproc = os.cpu_count())
    try:
        import torch

        info["torch"] = torch.__version__
        info["hip"] = torch.version.hip
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            info.update(gpu = p.name, capability = f"{p.major}.{p.minor}", vram_gib = round(p.total_memory / 2**30, 1))
        import triton

        info["triton"] = triton.__version__
    except Exception:  # noqa: BLE001
        pass
    return info


def main():
    ap = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--worker", help = argparse.SUPPRESS)
    ap.add_argument("--worker-out", help = argparse.SUPPRESS)
    ap.add_argument("--make-overlay", metavar = "WT")
    ap.add_argument("--overlay-base", default = "origin/main")
    ap.add_argument("--setup", action = "store_true", help = "install missing packages, clone trees (remote)")
    ap.add_argument("--work", default = "fl_bench_work")
    ap.add_argument("--zoo-main"); ap.add_argument("--zoo-pr1803"); ap.add_argument("--zoo-new")
    ap.add_argument("--main-ref", default = "main")
    ap.add_argument("--pr1803-ref", default = "chunked-logprob-recompute")
    ap.add_argument("--new-ref"); ap.add_argument("--new-repo", default = ZOO_REPO)
    ap.add_argument("--new-overlay", help = "tar.gz from --make-overlay (applied onto a clone of its base)")
    ap.add_argument("--extra-path", action = "append", default = [], help = "prepended to every worker's sys.path")
    ap.add_argument("--preset", default = "core", choices = sorted(PRESETS))
    ap.add_argument("--tiny", action = "store_true", help = "preset tiny, 3 reps, 1 round (smoke only)")
    ap.add_argument("--dry", action = "store_true", help = "tiny + CPU allowed (arms that need CUDA skip)")
    ap.add_argument("--tokens"); ap.add_argument("--shapes", help = "HxV,HxV")
    ap.add_argument("--no-extra", action = "store_true")
    ap.add_argument("--workloads", default = ",".join(WORKLOADS))
    ap.add_argument("--arms", help = "comma list; default every arm that implements the workload")
    ap.add_argument("--head", default = "frozen", choices = ("frozen", "trainable", "both"))
    ap.add_argument("--rounds", type = int, default = 2)
    ap.add_argument("--reps", type = int, default = 10)
    ap.add_argument("--warmup", type = int, default = 3)
    ap.add_argument("--accuracy", action = "store_true", help = "fp64 oracle errors at the smallest token count")
    ap.add_argument("--new-env", help = "extra new-arm variants with env vars, 'tag:VAR=value,tag2:VAR=value'")
    ap.add_argument("--sweep-rows", help = "extra new-arm variants with rows_per_chunk=R (and UNSLOTH_FUSED_LINEAR_MIN_ROWS=R)")
    ap.add_argument("--timeout", type = int, default = 1800)
    ap.add_argument("--out", default = "outputs/fused_linear_bench")
    args = ap.parse_args()

    if args.worker:
        spec = json.loads(Path(args.worker).read_text())
        res = worker(spec)
        Path(args.worker_out).write_text(json.dumps(res))
        return 0
    if args.make_overlay:
        make_overlay(args.make_overlay, args.overlay_base, args.out if args.out.endswith(".tgz") else "new_overlay.tgz")
        return 0
    if args.tiny or args.dry:
        args.preset, args.reps, args.rounds, args.warmup = "tiny", 3, 1, 1

    trees = {"main": args.zoo_main, "pr1803": args.zoo_pr1803, "new": args.zoo_new}
    meta = {}
    if args.setup:
        got = setup(args)
        meta["new_overlay"] = got.pop("new_meta", None)
        trees.update({k: v for k, v in got.items() if v})
    trees = {k: (os.path.abspath(v) if v else None) for k, v in trees.items()}
    extra = [os.path.abspath(p) for p in args.extra_path]
    out = Path(args.out)
    (out / "runs").mkdir(parents = True, exist_ok = True)
    meta.update(env = env_info(), trees = trees, args = vars(args), started = time.strftime("%Y-%m-%d %H:%M:%S"))
    for k, v in trees.items():
        if v and Path(v, ".git").exists():
            meta.setdefault("tree_heads", {})[k] = subprocess.run(
                ["git", "-C", v, "rev-parse", "HEAD"], capture_output = True, text = True).stdout.strip()

    cfgs = configs(args)
    min_tokens = min(c["tokens"] for c in cfgs)
    summary, k = [], 0
    for cfg in cfgs:
        variants = arm_variants(args, cfg["workload"])
        runs = {v: [] for v in variants}
        notes = {}
        cache = str((out / "runs" / f"oracle_{cfg['workload']}_{cfg['head']}_{cfg['tokens']}_{cfg['H']}x{cfg['V']}.pt").resolve())
        for rnd in range(args.rounds):
            order = variants if rnd % 2 == 0 else list(reversed(variants))
            for arm, rows in order:
                if arm in notes:
                    continue
                tree_key = ARMS[arm][0]
                tree = trees.get(tree_key) if tree_key else None
                label = _label(arm, rows)
                if tree_key and not tree:
                    notes[arm] = {"skip": f"no {tree_key} zoo tree given"}
                    continue
                env_variant = _env_variants(args).get(rows[2:], {}) if isinstance(rows, str) else {}
                spec = dict(cfg, arm = arm, rows = None if isinstance(rows, str) else rows, tree = tree,
                            reps = args.reps, warmup = args.warmup,
                            sys_path = ([tree] if tree else []) + extra, allow_cpu = args.dry,
                            accuracy = args.accuracy and cfg["tokens"] == min_tokens and rnd == 0, oracle_cache = cache,
                            env = env_variant | ({"UNSLOTH_FUSED_LINEAR_MIN_ROWS": rows} if isinstance(rows, int) else {}) | (
                                {"UNSLOTH_CHUNKED_CE": "0"} if arm == "old_zoo" else {}))  # fmt: skip
                k += 1
                sp, rp = out / "runs" / f"spec_{k}.json", out / "runs" / f"res_{k}.json"
                sp.write_text(json.dumps(spec))
                env = dict(os.environ, TORCHINDUCTOR_CACHE_DIR = str((out / "runs" / f"inductor_{k}").resolve()),
                           UNSLOTH_COMPILE_LOCATION = str((out / "runs" / f"ucc_{k}").resolve()),
                           PYTHONUNBUFFERED = "1")  # fmt: skip
                t0 = time.time()
                try:
                    p = subprocess.run([sys.executable, os.path.abspath(__file__), "--worker", str(sp), "--worker-out", str(rp)],
                                       env = env, capture_output = True, text = True, timeout = args.timeout)
                    res = json.loads(rp.read_text()) if rp.exists() else {"error": f"rc={p.returncode}: {p.stderr[-1200:]}"}
                except subprocess.TimeoutExpired:
                    res = {"error": f"timeout {args.timeout}s"}
                for d in (f"inductor_{k}", f"ucc_{k}"):
                    shutil.rmtree(out / "runs" / d, ignore_errors = True)
                res["wall_s"] = round(time.time() - t0, 1)
                print(f"[{label} {cfg}] " + json.dumps({kk: res.get(kk) for kk in (
                    "skip", "error", "cold_ms", "peak_extra_alloc_gib", "loss_abs_err", "grad_rel_err")}
                    | ({"median_ms": round(statistics.median(res["total_ms"]), 3)} if res.get("total_ms") else {})),
                    flush = True)  # fmt: skip
                if res.get("skip") or res.get("error"):
                    notes[arm if rows is None else label] = res
                    if res.get("skip"):
                        notes[arm] = res
                runs[(arm, rows)].append(res)
        for (arm, rows), rs in runs.items():
            label = _label(arm, rows)
            ok = [r for r in rs if r.get("total_ms")]
            row = dict(cfg, arm = arm, rows = rows, arm_label = label, stats = summarize(ok))
            bad = [r for r in rs if r.get("skip") or r.get("error")] or ([notes[arm]] if arm in notes and not rs else [])
            if bad:
                row["skip" if bad[0].get("skip") else "error"] = bad[0].get("skip") or bad[0].get("error")
                if bad[0].get("tb"):
                    row["tb"] = bad[0]["tb"]
            if not ok and not bad:
                row["skip"] = "not run"
            summary.append(row)
    meta["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    md = f"# fused_linear_bench\n\n`{json.dumps(meta['env'])}`\n\nTrees: `{json.dumps(meta.get('tree_heads', trees))}`\n\n" + table(summary)
    (out / "results.json").write_text(json.dumps(dict(meta = meta, rows = summary), indent = 1, default = str))
    (out / "table.md").write_text(md + "\n")
    print(md)
    n_ok = sum(1 for r in summary if r.get("stats"))
    print(RESULT_TAG + " " + json.dumps(dict(out = str(out), rows = len(summary), measured = n_ok,
                                              gpu = meta["env"].get("gpu"), b64_results = base64.b64encode(
                                                  json.dumps(dict(meta = meta, rows = summary), default = str).encode()).decode()
                                              if args.setup else None)), flush = True)  # fmt: skip
    return 0 if n_ok else 1


if __name__ == "__main__":
    sys.exit(main())
