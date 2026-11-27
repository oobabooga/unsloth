#!/usr/bin/env python3
"""Round-9d gfx1151 Linux items (observes only; judged by criteria/r9d_judge.py). Every arm is a fresh process
running probes/r9d/block_arm.py: a torch.compile'd diffusers WanTransformerBlock at Wan 2.2 5B dims, shapes
832x480x49 (S 5070) and 1280x704x49 (S 11440), CUDA-event timing + torch.profiler kernel table per arm.

  smoke  gfx1151 device + one bf16 GEMM, then a tiny compiled block (diffusers, compile, profiler all work).
  c62    C6-2: weight layouts contig (main) / pitch (P 64, the weight-pitch branch) / trans (transposed problem, NN
         GEMM) / trans_pitch; processes A P T X A X T P A (A three times = A/A).
  c66    C6-6: glue share from the c62 contig arms; ONLY if glue >= 10% (either shape): Inductor flags cdt
         (coordinate_descent_tuning) / mag (max_autotune_gemm) / maxat (max_autotune + cdt), processes
         cdt mag maxat A maxat mag cdt.
  bufops AMDGCN_USE_BUFFER_OPS render gate: Qwen-Image-2.1 at Studio defaults (INT8 on gfx1151) 1024^2, default
         steps, =0 (A) vs =1 (B), processes ABBA x3, each 1 warm-up + 1 measured render; fresh TRITON_CACHE_DIR per
         process (Inductor's cache ignores the variable); engagement = buffer_load / buffer_store counts in that
         process's compiled .amdgcn; Triton's own default recorded first.
"""
from __future__ import annotations

from pathlib import Path

import json
import shutil

from r9d_common import R9, e2e, prefetch, run_code, run_py

GLUE_GATE = 0.10

SMOKE = r"""
import json, time, torch
p = torch.cuda.get_device_properties(0)
out = dict(torch = torch.__version__, hip = getattr(torch.version, "hip", None), cuda = torch.cuda.is_available(),
           arch = getattr(p, "gcnArchName", None), total_gib = round(p.total_memory / 2**30, 3))
x = torch.randn(4096, 4096, device = "cuda", dtype = torch.bfloat16)
for _ in range(3): x @ x
torch.cuda.synchronize(); t = time.perf_counter()
for _ in range(20): x @ x
torch.cuda.synchronize(); out["gemm_tflops"] = round(20 * 2 * 4096**3 / (time.perf_counter() - t) / 1e12, 2)
print("R9JSON " + json.dumps(out), flush = True)
"""


def _arm(ctx, tag, args, timeout = 2700):
    od = Path(ctx["work"]) / "block"
    od.mkdir(parents = True, exist_ok = True)
    ref = Path(ctx["scratch"]) / "block_ref"
    out = od / f"{tag}.json"
    r = run_py(ctx, R9 / "block_arm.py", [*args, "--ref-dir", ref, "--out", out], out, timeout = timeout, tag = tag)
    return r


def item_smoke(ctx, obs):
    r = run_code(ctx, SMOKE, [], timeout = 600, tag = "smoke")
    obs["smoke"] = r
    obs["smoke_ok"] = bool(r.get("cuda")) and str(r.get("arch") or "").startswith("gfx1151")
    ctx["save"]()
    if not obs["smoke_ok"]:
        raise RuntimeError("smoke failed: no gfx1151 device")
    t = _arm(ctx, "smoke_tiny", ["--dims", "tiny", "--shapes", "tiny", "--iters", "5", "--reps", "2",
                                 "--layout", "trans_pitch"], timeout = 1200)
    obs["smoke_block"] = {k: t.get(k) for k in ("rc", "verdict", "error", "env", "shapes", "wall_s")}
    ctx["save"]()
    if t.get("verdict") != "ok":
        raise RuntimeError(f"smoke block failed: {str(t.get('error'))[-600:]}")


def item_c62(ctx, obs):
    rows = []
    obs["c62"] = rows
    order = ["contig", "pitch", "trans", "trans_pitch", "contig", "trans_pitch", "trans", "pitch", "contig"]
    for i, lay in enumerate(order):
        r = _arm(ctx, f"c62_{i}_{lay}", ["--layout", lay])
        r["layout"], r["flags"], r["pos"] = lay, "none", i
        rows.append(r)
        ctx["save"]()


def item_c66(ctx, obs):
    res: dict = {"arms": []}
    obs["c66"] = res
    base = [r for r in obs.get("c62") or [] if r.get("layout") == "contig" and r.get("verdict") == "ok"]
    if not base:
        r = _arm(ctx, "c66_base", ["--layout", "contig"])
        r["layout"], r["flags"] = "contig", "none"
        res["arms"].append(r)
        base = [r] if r.get("verdict") == "ok" else []
    glue = {}
    for r in base:
        for sh, v in (r.get("shapes") or {}).items():
            if v.get("glue_share") is not None:
                glue.setdefault(sh, []).append(v["glue_share"])
    res["glue_by_shape"] = glue
    gmax = max((sorted(v)[len(v) // 2] for v in glue.values()), default = None)
    res["glue_median_max"] = gmax
    ctx["save"]()
    if gmax is None:
        res["not_run"] = "no contig arm produced a profile"
        return
    if gmax < GLUE_GATE:
        res["not_run"] = f"glue share {gmax:.4f} under {GLUE_GATE}: flag arms not run (kill line)"
        return
    for i, fl in enumerate(["cdt", "mag", "maxat", "none", "maxat", "mag", "cdt"]):
        r = _arm(ctx, f"c66_{i}_{fl}", ["--layout", "contig", "--flags", fl], timeout = 3600)
        r["layout"], r["flags"], r["pos"] = "contig", fl, i
        r.pop("weights", None)
        res["arms"].append(r)
        ctx["save"]()


QWEN = "unsloth/Qwen-Image-2.1"
BUFOPS_DEFAULT = r"""
import json, os, triton
out = {"triton": triton.__version__, "env": os.environ.get("AMDGCN_USE_BUFFER_OPS")}
try:
    out["knob_default"] = bool(triton.knobs.amd.use_buffer_ops)
except Exception as exc:
    out["knob_error"] = repr(exc)
print("R9JSON " + json.dumps(out), flush = True)
"""


def _isa_counts(d: Path) -> dict:
    c = {"amdgcn_files": 0, "buffer_load": 0, "buffer_store": 0, "global_load": 0, "global_store": 0}
    for p in d.rglob("*.amdgcn"):
        c["amdgcn_files"] += 1
        t = p.read_text(encoding = "utf-8", errors = "replace")
        for k in ("buffer_load", "buffer_store", "global_load", "global_store"):
            c[k] += t.count(k)
    return c


def item_bufops(ctx, obs):
    res: dict = {"arms": []}
    obs["bufops"] = res
    res["default"] = run_code(ctx, BUFOPS_DEFAULT, [], timeout = 300, tag = "bufops_default",
                              extra_env = {"AMDGCN_USE_BUFFER_OPS": None})
    res["prefetch"] = prefetch(ctx, QWEN)
    ctx["save"]()
    for i, arm in enumerate("ABBAABBAABBA"):
        tag = f"bufops_{i}_{arm}"
        tdir = Path(ctx["scratch"]) / f"tri_{tag}"
        shutil.rmtree(tdir, ignore_errors = True)
        r = e2e(ctx, ctx["main_backend"], QWEN, tag, ["--renders", "2", "--gen", json.dumps({"width": 1024, "height": 1024})],
                extra_env = {"AMDGCN_USE_BUFFER_OPS": "1" if arm == "B" else "0", "TRITON_CACHE_DIR": str(tdir)})
        keep = {k: r.get(k) for k in ("verdict", "error", "load_s", "rc", "wall_s")}
        keep["renders"] = [{k: x.get(k) for k in ("r", "wall_s", "s_per_step", "md5", "rdna_mm", "transformer_quant",
                                                  "speed_optims")} for x in r.get("renders") or []]
        st_ = r.get("status") or {}
        keep["status"] = {k: st_.get(k) for k in ("transformer_quant", "transformer_quant_backend", "speed_mode",
                                                  "attention_backend")}
        keep.update(arm = arm, i = i, isa = _isa_counts(tdir))
        shutil.rmtree(tdir, ignore_errors = True)
        res["arms"].append(keep)
        ctx["save"]()


ITEMS = {"smoke": item_smoke, "c62": item_c62, "c66": item_c66, "bufops": item_bufops}
