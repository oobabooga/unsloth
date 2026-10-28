#!/usr/bin/env python3
"""Fused linear losses (unsloth_zoo.fused_losses.fused_linear) in fp16: accuracy vs fp64 + TRL / HF math, speed and
memory vs plain torch, and split-device (hidden / head on different GPUs) correctness. One JSON line to --out.

  --mode acc    : CE / log-probs / JSD, frozen and trainable head, fused vs reference (HF CE upcast, TRL
                  selective_log_softmax and generalized_jsd_loss in the input dtype) vs fp64 oracle
  --mode perf   : one arm (--arm fused|ref) fwd+bwd median ms + peak MiB per workload, frozen head
  --mode mgpu   : needs 2 devices; every op with hidden / weight (and the JSD teacher) split across devices vs the same
                  op on one device, grad devices, and the hidden device's extra memory (a weight copy would show)
"""
import argparse, json, math, os, statistics, sys, time

import torch
import torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("--mode", required = True, choices = ["acc", "perf", "mgpu"])
ap.add_argument("--arm", default = "fused", choices = ["fused", "ref"])
ap.add_argument("--out", required = True)
ap.add_argument("--n", type = int, default = 1024)
ap.add_argument("--h", type = int, default = 1024)
ap.add_argument("--vocabs", default = "151936,32001")
ap.add_argument("--reps", type = int, default = 10)
ap.add_argument("--scale", type = float, default = 1024.0, help = "GradScaler-like grad_output (fp16 training)")
args = ap.parse_args()

import unsloth_zoo.fused_losses.fused_linear as fl  # noqa: E402

DT = torch.float16
CPU = os.environ.get("FLCHECK_CPU") == "1"  # harness smoke test only
dev = torch.device("cpu") if CPU else torch.device("cuda", 0)
res = dict(mode = args.mode, arm = args.arm, torch = torch.__version__, hip = torch.version.hip,
           gpu = "cpu" if CPU else torch.cuda.get_device_name(0), n_gpu = torch.cuda.device_count(), dtype = str(DT),
           zoo = os.path.dirname(fl.__file__), triton_on = getattr(fl, "_TRITON", None))


# ---- TRL 0.25.1 reference math, copied verbatim (selective_log_softmax fp16 branch, generalized_jsd_loss) ----
def trl_selective_log_softmax(logits, index):
    if logits.dtype in [torch.float32, torch.float64]:
        sel = torch.gather(logits, dim = -1, index = index.unsqueeze(-1)).squeeze(-1)
        lse = torch.stack([torch.logsumexp(lg, dim = -1) for lg in logits])
        return sel - lse
    out = []
    for row_logits, row_labels in zip(logits, index):
        row_logps = F.log_softmax(row_logits, dim = -1)
        out.append(row_logps.gather(dim = -1, index = row_labels.unsqueeze(-1)).squeeze(-1))
    return torch.stack(out)


def trl_generalized_jsd_loss(student_logits, teacher_logits, labels = None, beta = 0.5, temperature = 1.0):
    student_logits = student_logits / temperature
    teacher_logits = teacher_logits / temperature
    s = F.log_softmax(student_logits, dim = -1)
    t = F.log_softmax(teacher_logits, dim = -1)
    if beta == 0:
        jsd = F.kl_div(s, t, reduction = "none", log_target = True)
    elif beta == 1:
        jsd = F.kl_div(t, s, reduction = "none", log_target = True)
    else:
        b = torch.tensor(beta, dtype = s.dtype)
        m = torch.logsumexp(torch.stack([s + torch.log(1 - b), t + torch.log(b)]), dim = 0)
        jsd = b * F.kl_div(m, t, reduction = "none", log_target = True) + (1 - b) * F.kl_div(m, s, reduction = "none", log_target = True)
    mask = labels != -100
    return jsd[mask].sum() / mask.sum()


def make(n, h, v, device, seed = 0):
    g = torch.Generator().manual_seed(seed)
    hid = torch.randn(n, h, generator = g)
    w = torch.randn(v, h, generator = g) * (2.5 / h ** 0.5)
    hid[: n // 8] *= 3.0  # some peaked rows
    t = torch.randint(0, v, (n,), generator = g)
    t[torch.rand(n, generator = g) < 0.2] = -100
    th = torch.randn(n, h, generator = g)
    tw = torch.randn(v, h, generator = g) * (2.5 / h ** 0.5)
    up = torch.randn(n, generator = g)  # upstream grad for per-token log-probs
    return dict(h = hid.to(device, DT), w = w.to(device, DT), t = t.to(device), th = th.to(device, DT),
                tw = tw.to(device, DT), up = up.to(device))


# ---- one loss per op; impl in {fused, ref, oracle}; returns (loss, grad_h, grad_w or None) -----------------------
def run(op, impl, d, head, scale, hdev = None, wdev = None, tdev = None):
    hdev = hdev or d["h"].device
    wdev = wdev or d["w"].device
    tdev = tdev or d["th"].device
    big = impl == "oracle"
    cast = (lambda x: x.double()) if big else (lambda x: x)
    h = cast(d["h"]).to(hdev).detach().requires_grad_()
    w = cast(d["w"]).to(wdev).detach().requires_grad_(head)
    t = d["t"].to(hdev)
    keep = t != -100
    n_keep = int(keep.sum())
    if op == "ce":
        if impl == "fused":
            loss = fl.fused_linear_cross_entropy(h, w, None, t, divisor = float(n_keep), scaling = scale)
        else:
            logits = h.to(w.device) @ w.t()
            loss = F.cross_entropy(logits.float() if not big else logits, t.to(w.device), reduction = "sum") / n_keep
    elif op == "logprobs":
        up = d["up"].to(hdev)
        tt = t.clamp(min = 0)
        if impl == "fused":
            lp = fl.fused_linear_logprobs(h, w, tt, mask = keep)
        else:
            logits = h.to(w.device) @ w.t()
            lp = trl_selective_log_softmax(logits, tt.to(w.device)).to(hdev)
            lp = torch.where(keep, lp, 0)
        loss = (lp.double() * up.double() * keep).sum() / n_keep if big else (lp.float() * up * keep).sum() / n_keep
    elif op == "jsd":
        th = cast(d["th"]).to(tdev)
        tw = cast(d["tw"]).to(tdev)
        if impl == "fused":
            loss = fl.fused_linear_jsd(h, w, th, tw, labels = t, beta = 0.5, scaling = scale)
        else:
            sl = (h.to(w.device) @ w.t())
            tl = (th @ tw.t()).to(w.device)
            loss = trl_generalized_jsd_loss(sl[None], tl[None], labels = t.to(w.device)[None], beta = 0.5)
    (loss * scale).backward()
    gh = h.grad / scale
    gw = (w.grad / scale) if head else None
    return loss.detach(), gh, gw


def rel(x, o):
    x, o = x.double().to(o.device), o.double()
    return float((x - o).abs().max() / o.abs().max().clamp_min(1e-30))


def finite(*xs):
    return all(x is None or bool(torch.isfinite(x.float()).all()) for x in xs)


if args.mode == "acc":
    res["cases"] = []
    for v in [int(x) for x in args.vocabs.split(",")]:
        d = make(args.n, args.h, v, dev)
        for op in ("ce", "logprobs", "jsd"):
            for head in (False, True):
                o = run(op, "oracle", d, head, 1.0)
                case = dict(v = v, op = op, trainable_head = head)
                for impl in ("fused", "ref"):
                    try:
                        c0 = dict(fl.CALLS)
                        l, gh, gw = run(op, impl, d, head, args.scale)
                        case[impl] = dict(loss_abs = abs(float(l) - float(o[0])), grad_h_rel = rel(gh, o[1]),
                                          grad_w_rel = rel(gw, o[2]) if head else None, finite = finite(l, gh, gw),
                                          calls = {k: fl.CALLS[k] - c0.get(k, 0) for k in fl.CALLS
                                                   if fl.CALLS[k] != c0.get(k, 0)} if impl == "fused" else None)
                    except Exception as e:  # recorded, judged as a failure
                        case[impl] = dict(error = f"{type(e).__name__}: {e}"[:400])
                    torch.cuda.empty_cache()
                res["cases"].append(case)
                print(json.dumps(case), flush = True)
        del d
        torch.cuda.empty_cache()

elif args.mode == "perf":
    v = int(args.vocabs.split(",")[0])
    d = make(args.n, args.h, v, dev)
    res["shape"] = dict(n = args.n, h = args.h, v = v)
    for op in ("ce", "logprobs", "jsd"):
        try:
            for _ in range(2):
                run(op, args.arm, d, False, args.scale)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
            ms = []
            for _ in range(args.reps):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                run(op, args.arm, d, False, args.scale)
                torch.cuda.synchronize()
                ms.append((time.perf_counter() - t0) * 1e3)
            res[op] = dict(ms = round(statistics.median(ms), 3), spread = round(max(ms) - min(ms), 3),
                           peak_mib = round((torch.cuda.max_memory_allocated() - base) / 2**20, 1))
        except Exception as e:
            res[op] = dict(error = f"{type(e).__name__}: {e}"[:400])
        print(op, res[op], flush = True)

else:  # mgpu
    assert torch.cuda.device_count() >= 2, "mgpu needs 2 devices"
    d0, d1 = torch.device("cuda", 0), torch.device("cuda", 1)
    v = int(args.vocabs.split(",")[0])
    d = make(args.n, args.h, v, torch.device("cpu"))
    res["weight_mib"] = round(v * args.h * 2 / 2**20, 1)
    res["cases"] = []
    for op in ("ce", "logprobs", "jsd"):
        for head in (False, True):
            o = run(op, "oracle", {k: x.to(d0) for k, x in d.items()}, head, 1.0)
            for wd in (d0, d1):  # single-device reference on the weight's device
                single = run(op, "fused", {k: x.to(wd) for k, x in d.items()}, head, args.scale)
                hd = d1 if wd == d0 else d0
                case = dict(op = op, trainable_head = head, hidden_dev = str(hd), weight_dev = str(wd))
                try:
                    # Inputs placed first, so the measured window holds only what the op allocates. JSD: the teacher
                    # sits with the hidden states (the other device from the student head).
                    placed = {k: x.to(wd if k == "w" else hd) for k, x in d.items()}
                    torch.cuda.synchronize(hd)
                    torch.cuda.reset_peak_memory_stats(hd)
                    before = torch.cuda.memory_allocated(hd)
                    l, gh, gw = run(op, "fused", placed, head, args.scale)
                    torch.cuda.synchronize(hd)
                    extra = (torch.cuda.max_memory_allocated(hd) - before) / 2**20
                    case.update(ok = True, grad_h_dev = str(gh.device), grad_w_dev = str(gw.device) if head else None,
                                loss_vs_single = abs(float(l) - float(single[0])),
                                grad_h_vs_single = float((gh.to(wd).float() - single[1].float()).abs().max()),
                                grad_w_vs_single = float((gw.float() - single[2].float()).abs().max()) if head else None,
                                grad_h_rel_oracle = rel(gh, o[1]), single_grad_h_rel_oracle = rel(single[1], o[1]),
                                finite = finite(l, gh, gw), hidden_dev_extra_mib = round(extra, 1))
                except Exception as e:
                    import traceback
                    case.update(ok = False, error = f"{type(e).__name__}: {e}"[:400], tb = traceback.format_exc()[-1500:])
                res["cases"].append(case)
                print(json.dumps(case), flush = True)
                torch.cuda.empty_cache()

with open(args.out, "w", encoding = "utf-8") as f:
    json.dump(res, f, indent = 1)
print("FLCHECK_DONE", flush = True)
