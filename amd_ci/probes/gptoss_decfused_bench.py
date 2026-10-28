"""GPT-OSS 20b eager decode: ms/token at 1k context (batch 1 and 4), greedy tokens, kernels + launches / token.

    python -u bench.py OUT.json MODEL [--no-4bit]

unsloth_zoo comes from PYTHONPATH (base or head tree), so arms differ only in zoo.
"""
import json, os, statistics, sys, time

out_path, model_name = sys.argv[1], sys.argv[2]
no4 = "--no-4bit" in sys.argv
CTX, NEW = 1024, 96
os.environ.setdefault("UNSLOTH_DISABLE_AUTO_UPDATES", "1")
from unsloth import FastLanguageModel  # noqa: E402
import torch  # noqa: E402
import unsloth_zoo  # noqa: E402

res = {"model": model_name, "zoo": unsloth_zoo.__file__, "gpu": torch.cuda.get_device_name(0),
       "arch": getattr(torch.cuda.get_device_properties(0), "gcnArchName", None), "hip": torch.version.hip}
try:
    from unsloth_zoo import gpt_oss_decode_fused as FD
    res["fused_enabled_at_import"] = bool(FD._ENABLED)
    res["libdevice"] = FD._libdevice is not None
except Exception as e:
    FD = None
    res["fused_import_error"] = repr(e)
model, tok = FastLanguageModel.from_pretrained(model_name, max_seq_length = CTX + 160, load_in_4bit = not no4,
                                               dtype = torch.bfloat16)
FastLanguageModel.for_inference(model)
dev = torch.device("cuda")
pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
text = ("The history of numerical linear algebra begins with Gauss, who described elimination for "
        "solving the normal equations of least squares. ") * 60


def inputs(b):
    g = torch.Generator().manual_seed(0)
    ids = torch.randint(1000, 150000, (b, CTX), generator = g)
    real = tok(text, return_tensors = "pt").input_ids[0, :CTX]
    ids[0, -real.shape[0]:] = real  # row 0: real text, so greedy decoding is not degenerate
    return ids.to(dev), torch.ones(b, CTX, dtype = torch.long, device = dev)


def gen(ids, am, n, scores = False):
    torch.cuda.synchronize()
    t = time.perf_counter()
    with torch.no_grad():
        out = model.generate(input_ids = ids, attention_mask = am, max_new_tokens = n, min_new_tokens = n,
                             do_sample = False, pad_token_id = pad, output_scores = scores,
                             return_dict_in_generate = scores)
    torch.cuda.synchronize()
    return time.perf_counter() - t, out


from torch.profiler import ProfilerActivity, profile  # noqa: E402

LAUNCH = ("cudaLaunchKernel", "cuLaunchKernel", "cudaLaunchKernelExC", "cuLaunchKernelEx", "hipLaunchKernel",
          "hipExtLaunchKernel", "hipModuleLaunchKernel", "hipExtModuleLaunchKernel", "hipLaunchKernelExC")


def counts(ids, am, n):
    with profile(activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
        gen(ids, am, n)
    k = l = 0
    for e in p.events():
        if str(getattr(e, "device_type", "")).endswith("CUDA"):
            k += 1
        elif e.name in LAUNCH:
            l += 1
    return k, l


for b in (1, 4):
    ids, am = inputs(b)
    gen(ids, am, 1); gen(ids, am, 8)
    if FD is not None:
        before = dict(FD.FUSED_DECODE_CALLS)
    p1 = [gen(ids, am, 1)[0] for _ in range(5)]
    pn = [gen(ids, am, 65)[0] for _ in range(5)]
    r = {"decode_ms_per_token": round((statistics.median(pn) - statistics.median(p1)) / 64 * 1e3, 3),
         "decode_all_s": [round(x, 4) for x in pn], "prefill_all_s": [round(x, 4) for x in p1]}
    if FD is not None:
        r["fused_calls_per_token"] = {k: (FD.FUSED_DECODE_CALLS[k] - before[k]) / (5 * 64 + 0.0) for k in before}
    try:  # the profiler is diagnostic only: a ROCm kineto failure must not lose the timing
        k1, l1 = counts(ids, am, 1)
        k9, l9 = counts(ids, am, 9)
        r["kernels_per_token"] = (k9 - k1) / 8
        r["launches_per_token"] = (l9 - l1) / 8
    except Exception as e:
        r["profile_error"] = repr(e)[:300]
    _, out = gen(ids, am, NEW, scores = True)
    r["tokens"] = out.sequences[:, CTX:].tolist()
    sc = torch.stack(out.scores, 1).float()  # [b, NEW, V]
    top2 = sc.topk(2, dim = -1).values
    r["top_margin_min"] = float((top2[..., 0] - top2[..., 1]).min())
    r["scores_first8_top"] = sc[:, :8].max(-1).values.tolist()
    res[f"b{b}"] = r
    print("TIMING", b, json.dumps({k: v for k, v in r.items() if k not in ("tokens", "decode_all_s", "prefill_all_s")}),
          flush = True)
    torch.save(sc.to(torch.bfloat16).cpu(), out_path + f".scores_b{b}.pt")
with open(out_path, "w") as f:
    json.dump(res, f)
print("DONE", out_path, flush = True)
