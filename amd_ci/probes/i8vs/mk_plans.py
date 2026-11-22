#!/usr/bin/env python3
"""Part 2 plans, one per card: python mk_plans.py. Per family: D (Studio default) / G (attention_backend="sage", the
kernels-hub build) / U (attention_backend="unsloth_int8", forced) as 3 interleaved fresh processes each (D G U, U G D,
D G U), then GP (pip SageAttention 2.2.0 visible to Studio through PYTHONPATH: diffusers' sage backend), N (INT8 off,
default backend: the bf16 baseline) and the bf16-swap noise floor K (INT8 off, unpinned SDPA; K2 = cuDNN when K kept
N's backend). LPIPS against D1 and against N. Each first process of a letter also profiles a 2-step render."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
I8 = "UNSLOTH_DIFFUSION_INT8_ATTENTION_KERNEL"
FAMS = {
    "klein": {"kind": "image", "model": "black-forest-labs/flux-3-image-klein", "load": {}},
    "q21": {"kind": "image", "model": "unsloth/Qwen-Image-2.1", "load": {"model_kind": "pipeline"}},
    "zimage": {"kind": "image", "model": "unsloth/Z-Image-Turbo", "load": {"model_kind": "pipeline"}},
    "wan5b": {"kind": "video", "model": "Wan-AI/Wan2.2-TI2V-5B-Diffusers", "load": {"model_kind": "pipeline"},
              "gen": {"width": 1280, "height": 704, "num_frames": 49, "steps": 10}},
}
SAGE_PIP = {"PYTHONPATH": "$ROOT/ext/sage2"}


def arms(sage = "S", sage_procs = 3, studio_sage = True, prompts = 2):
    """Interleaved D / <sage> / U, 3 fresh processes each (D S U, U S D, D S U). S = attention_backend="sage" with the
    INT8 kernel switched off (UNSLOTH_DIFFUSION_INT8_ATTENTION_KERNEL=0), i.e. Sage alone: with the switch unset, a
    "sage" load of these families still races Studio's INT8 routes against Sage and keeps INT8 where it wins. SP = the
    same through pip Sage 2.2.0 (PYTHONPATH). G = the plain attention_backend="sage" request, once, to record what
    Studio actually runs for it."""
    off = {I8: "0"}
    L = {"D": {}, "U": {"load": {"attention_backend": "unsloth_int8"}},
         "S": {"env": off, "load": {"attention_backend": "sage"}},
         "SP": {"env": {**SAGE_PIP, **off}, "load": {"attention_backend": "sage"}}}
    order = ["D", sage, "U", "U", sage, "D", "D", sage, "U"]
    seen: dict = {}
    out = []
    for x in order:
        if x is None:
            continue
        seen[x] = seen.get(x, 0) + 1
        if x == sage and seen[x] > sage_procs:
            continue
        a = {"tag": f"{x}{seen[x]}", "prompts": prompts, **L[x]}
        if seen[x] == 1:
            a["profile_steps"] = 2
        out.append(a)
    if studio_sage:
        g = {"load": {"attention_backend": "sage"}}
        if sage == "SP":
            g["env"] = SAGE_PIP
        out.append({"tag": "G1", "prompts": prompts, "profile_steps": 2, **g})
    out.append({"tag": "N", "prompts": prompts, "env": {I8: "0"}, "profile_steps": 2})
    out.append({"tag": "K", "prompts": prompts, "env": {I8: "0"}, "load": {"attention_backend": "native"}, "profile_steps": 2})
    out.append({"tag": "K2", "prompts": prompts, "env": {I8: "0"}, "load": {"attention_backend": "cudnn"}, "profile_steps": 2,
                "only_if_same_backend": ["K", "N"]})
    return out


def fam(name, gen = None, timeout = 5400, **kw):
    f = dict(FAMS[name])
    a = arms(**kw)
    tags = [x["tag"] for x in a]
    pairs = [f"{t}:D1" for t in tags if t != "D1"] + [f"{t}:N" for t in tags if t != "N"]
    f.update(name = name, prompts = kw.get("prompts", 2), arms = a, pairs = pairs, score_ref = "D1", arm_timeout = timeout)
    if gen:
        f["gen"] = {**f.get("gen", {}), **gen}
    return f


PLANS = {
    "m_l4": {"families": [fam("klein"), fam("q21"), fam("wan5b", timeout = 7200)]},
    "m_l4q": {"families": [fam("q21")]},
    "m_l4w": {"families": [fam("wan5b", timeout = 7200)]},
    "m2_a100": {"families": [fam("klein"), fam("q21"), fam("wan5b", timeout = 7200)]},
    # G4: the hub build has no sm120 variant; pip Sage 2.2.0 is the Sage arm (SP), G1 = the plain sage request on pip
    "m2_g4": {"families": [fam("klein", sage = "SP"), fam("q21", sage = "SP"), fam("wan5b", sage = "SP", timeout = 7200)]},
    # T4: no Sage build runs on sm75: D / U only, G1 records the declined request; one timed render per process
    "m2_t4": {"families": [fam("klein", sage = None, prompts = 1),
                           fam("wan5b", gen = {"width": 832, "height": 480}, sage = None, prompts = 1, timeout = 9000)]},
}

for rid, plan in (PLANS.items() if __name__ == "__main__" else ()):
    plan["run_id"] = rid
    (HERE / "plans").mkdir(exist_ok = True)
    (HERE / "plans" / f"{rid}.json").write_text(json.dumps(plan, indent = 1))
    print(rid, [(f["name"], [x["tag"] for x in f["arms"]]) for f in plan["families"]])
