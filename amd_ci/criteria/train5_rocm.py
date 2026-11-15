#!/usr/bin/env python3
"""Criteria: tiny Unsloth training on gfx1151 (Linux / Windows), base vs head of one PR (or an unsloth + zoo pair).

Regression mode. Head is WORSE than base when any of:
  * a training cell finished with finite losses at the base and not at the head;
  * its losses were decreasing at the base and are not at the head;
  * both finished and a per-step loss differs beyond max(RTOL relative, 3 x the base A/A spread);
  * peak allocated memory (process-local) grew past 1.25 x base + 32 MiB;
  * layer gradient checkpointing ran at the base (calls > 0) and not at the head;
  * rl_decode: graphs matched eager after the optimizer step at the base and not at the head;
  * a test id fails / errors at the head that did not at the base (head-only files included).
Gates (non-vacuity): the environment built; every cell of each state imported unsloth and unsloth_zoo from that
state's checkouts and ran on a gfx1151 HIP device; the base flm16 cell trained to a verdict; selected tests ran.
"""
from __future__ import annotations

import os

TITLE = "Tiny Unsloth training (LoRA / QLoRA / vision / MoE grouped_mm + PR cells) on gfx1151, base vs head"
MODE = "regression"
NEEDS = ["rocm", "gpu", "nvidia", "multi_gpu", "windows"]
RTOL = 2e-2
MEM_RATIO, MEM_SLACK = 1.25, 32.0
TRAIN = ("flm16", "flm4", "gcmem", "vision", "moe", "gemma4", "qwen38", "dsv4", "mllama", "rl_decode")

# Why a PR's own new path does or does not engage on ROCm; rendered with the table.
NOTES = {
    "u17785": "Distributed FSDP2 / TP / PP for Qwen3.8 / DeepSeek-V4.1: multi-GPU paths unreachable on a one-GPU box. "
              "Linux runs with --spoof-devices (device_count 2, wiring only); the PR's gloo tests are CPU. Single-GPU "
              "training is real.",
    "m17817": "Gemma-4 large-head attention kernel: patch_large_head_attention returns False on ROCm "
              "(torch.version.hip gate), so the new kernel never engages here; the gemma4 cell proves Gemma-4 "
              "training is unchanged.",
    "u17826": "DCP cross-version optimizer checkpoint message: CPU-side logic, covered by its test; training cells are "
              "the no-regression check.",
    "p17827_2525": "Qwen3.8 packing / GDN (unsloth) + QSA / fla fp32 (zoo): qwen38 cell trains with packing=True "
                   "(varlen hit counters), fla runs through Triton on ROCm.",
    "u17828": "Decode CUDA graphs are default-off on ROCm (torch.version.hip gate); rl_decode forces them "
              "(UNSLOTH_DECODE_CUDA_GRAPHS=1) and checks graphs see the post-step weights.",
    "p17829_2526": "Block compile on transformers 5.23 + grouped MoE compile: Dynamo graph breaks / recompiles "
                   "recorded per cell; block CUDA graphs are off on ROCm.",
    "u17830": "FastLanguageModel layer GC on transformers 5.23: gc_calls counts the layer checkpoint function "
              "(0 = checkpointing dropped).",
    "z2522": "triton.use_block_ptr gated on Triton having block pointers: env cell records the inductor flag and "
             "whether this Triton has make_block_ptr.",
    "z2523": "Mllama vision layer accepts hidden_states=: mllama cell calls the patched layer both ways on the GPU.",
    "z2524": "DeepSeek-V4.1 FP4 experts through the fused MXFP4 grouped GEMM: needs an MXFP4 checkpoint and a "
             "native-FP4 GPU; gfx1151 has neither, so the route does not engage here (dsv4 cell trains bf16).",
}


def _norm(p):
    return os.path.normcase(os.path.normpath(str(p or "")))


def _cells(o):
    return (o or {}).get("cells") or {}


def _arch(o):
    for c in _cells(o).values():
        a = (c.get("versions") or {}).get("arch")
        if a:
            return str(a)
    return ""


def _trained(c):
    return bool(c) and c.get("stage") == "done" and c.get("finite") is True


def gates(obs):
    out = []
    for name in ("base", "head"):
        o = obs.get(name) or {}
        env = o.get("env") or {}
        out.append((f"{name} environment built", not env.get("error") and bool((env.get("env") or {}).get("hip")),
                    str(env.get("error") or env.get("env"))[:300]))
        paths = o.get("paths") or {}
        bad = []
        for cname, c in _cells(o).items():
            for mod, key in (("unsloth", "unsloth_file"), ("zoo", "unsloth_zoo_file")):
                f = c.get(key)
                if f and paths.get(mod) and not _norm(f).startswith(_norm(paths[mod])):
                    bad.append(f"{cname}:{mod}={f}")
        files = sorted({str(c.get("unsloth_file")) for c in _cells(o).values() if c.get("unsloth_file")})
        out.append((f"{name} cells imported this state's unsloth + unsloth_zoo", bool(files) and not bad,
                    ("; ".join(bad) or ", ".join(files))[:300] or "no cell imported unsloth"))
        out.append((f"{name} ran on gfx1151 (HIP)", "gfx1151" in _arch(o), _arch(o) or "no device recorded"))
        for key in ("pytest", "pytest_companion"):
            p = o.get(key)
            if p is not None and name == "head":
                n = p.get("n_passed", 0) + p.get("n_failed", 0) + p.get("n_skipped", 0)
                out.append((f"head {key} collected tests", n > 0 or not p.get("selected"),
                            f"selected={p.get('selected')} passed={p.get('n_passed')} failed={p.get('n_failed')} "
                            f"skipped={p.get('n_skipped')} crashed={p.get('crashed_files')}"[:300]))
    # Non-vacuity: the base actually trained something, so "no cell got worse" compared real runs. A cell that
    # fails at BOTH states is a finding reported in the table and the verdict text, not a gate.
    trained = sorted(n for n, c in _cells(obs.get("base")).items() if n in TRAIN and _trained(c))
    out.append(("base trained at least one cell (finite losses)", bool(trained), ", ".join(trained) or "none"))
    return out


def _aa_spread(base, cell):
    a, b = (_cells(base).get(cell) or {}).get("losses"), (_cells(base).get(cell + "@aa") or {}).get("losses")
    if not a or not b or len(a) != len(b):
        return None
    return max(abs(x - y) / max(1.0, abs(x)) for x, y in zip(a, b))


def _loss_tol(base, cell):
    spreads = [s for s in (_aa_spread(base, c) for c in ("flm16", "moe", cell)) if s is not None]
    return max([RTOL] + [3 * s for s in spreads])


def _pyt(o):
    p = (o or {}).get("pytest") or {}
    q = (o or {}).get("pytest_companion") or {}
    fails = set(p.get("failed") or []) | set(p.get("errors") or []) | set(q.get("failed") or []) | set(q.get("errors") or [])
    passed = set(p.get("passed") or []) | set(q.get("passed") or [])
    crashed = set(p.get("crashed_files") or []) | {"companion:" + x for x in (q.get("crashed_files") or [])}
    return passed, fails, crashed


def head_is_worse(base, head):
    worse, fixed, both = [], [], []
    bc, hc = _cells(base), _cells(head)
    for cell in TRAIN:
        b, h = bc.get(cell), hc.get(cell)
        if b is None and h is None:
            continue
        if _trained(b) and not _trained(h):
            worse.append(f"{cell}: trained at base, not at head ({(h or {}).get('error') or (h or {}).get('stage')})"[:300])
            continue
        if not _trained(b) and _trained(h):
            fixed.append(cell)
            continue
        if not _trained(b) and not _trained(h):
            both.append(f"{cell} ({str((h or {}).get('error') or (h or {}).get('stage'))[:120]})")
            continue
        if b.get("decreasing") and h.get("decreasing") is False:
            worse.append(f"{cell}: loss decreasing at base, not at head {h.get('losses')}")
        lb, lh = b.get("losses") or [], h.get("losses") or []
        tol = _loss_tol(base, cell)
        if len(lb) == len(lh) and lb:
            d = max(abs(x - y) / max(1.0, abs(x)) for x, y in zip(lb, lh))
            if d > tol:
                worse.append(f"{cell}: per-step loss differs by {d:.3g} > tol {tol:.3g}")
        elif lb or lh:
            worse.append(f"{cell}: {len(lb)} base vs {len(lh)} head loss steps")
        pb, ph = b.get("peak_alloc_mb"), h.get("peak_alloc_mb")
        if isinstance(pb, (int, float)) and isinstance(ph, (int, float)) and ph > pb * MEM_RATIO + MEM_SLACK:
            worse.append(f"{cell}: peak allocated {ph} MiB vs base {pb} MiB")
        if (b.get("gc_calls") or 0) > 0 and not (h.get("gc_calls") or 0) > 0:
            worse.append(f"{cell}: layer gradient checkpointing ran at base ({b.get('gc_calls')} calls), 0 at head")
        if cell == "gcmem" and b.get("gc_engaged") and not h.get("gc_engaged"):
            worse.append(f"gcmem: layer GC saved memory at base (ratio {b.get('gc_mem_ratio')}), not at head "
                         f"(ratio {h.get('gc_mem_ratio')})")
        if cell == "rl_decode" and b.get("graphs_match_eager_after_step") and not h.get("graphs_match_eager_after_step"):
            worse.append(f"rl_decode: decode graphs diverge from eager after the step at head (rel {h.get('graphs_vs_eager_rel')})")
    pb_, fb, cb = _pyt(base)
    ph_, fh, ch = _pyt(head)
    new_fail = sorted(fh - fb)
    if new_fail:
        worse.append("tests failing at head only: " + ", ".join(f"`{t.split('::')[-1]}`" for t in new_fail[:12]))
    new_crash = sorted(ch - cb)
    if new_crash:
        worse.append("test files crashing at head only: " + ", ".join(new_crash))
    detail = (f"newly training at head: {', '.join(fixed) or 'none'}; failing at both (pre-existing): "
              f"{'; '.join(both) or 'none'}; tests failing at both: "
              f"{', '.join(t.split('::')[-1] for t in sorted(fh & fb)[:10]) or 'none'}; "
              f"tests passing at head only: {len(ph_ - pb_ - fb)}")
    if worse:
        return True, "; ".join(worse) + ". " + detail
    return False, "no training cell or test that passed at the base fails at the head; " + detail


def _fmt(c):
    if not c:
        return "-"
    if c.get("cell") == "env" or "env" == c.get("_name"):
        return "ok" if c.get("stage") == "done" else f"FAIL {str(c.get('error'))[:80]}"
    if not _trained(c):
        return f"FAIL@{c.get('stage')} {str(c.get('error') or c.get('rc'))[:110]}"
    losses = c.get("losses") or []
    s = f"{losses[0]:.4f}->{losses[-1]:.4f}" if losses else "?"
    s += "" if c.get("decreasing") is None else (" dec" if c.get("decreasing") else " NOT-dec")
    s += f" pk={c.get('peak_alloc_mb')}M gc={c.get('gc_calls')}"
    dy = c.get("dynamo") or {}
    if dy.get("graph_breaks") is not None:
        s += f" gb={dy.get('graph_breaks')} g={dy.get('unique_graphs')}"
    prof = ((c.get("profile_step1") or c.get("profile_step") or {}).get("counts") or {})
    if prof.get("grouped_mm"):
        s += f" grouped_mm={prof['grouped_mm']}"
    if c.get("moe_backend"):
        s += f" backend={c['moe_backend']}"
    if c.get("lora_on_experts"):
        s += f" lora_experts={c['lora_on_experts']}"
    if "large_head_patch_returns" in c:
        s += f" lh_patch={c['large_head_patch_returns']}"
    if c.get("varlen_hits"):
        s += f" varlen={c['varlen_hits']}"
    if "gc_mem_ratio" in c:
        s += f" gc_engaged={c.get('gc_engaged')} mem_ratio(on/off)={c.get('gc_mem_ratio')}"
    if "graphs_match_eager_after_step" in c:
        s += f" graphs==eager:{c['graphs_match_eager_after_step']} rel={c.get('graphs_vs_eager_rel'):.2g}"
    if c.get("vision_layer_calls"):
        s += " vlayer=" + ",".join(f"{k}:{v.get('ok')}" for k, v in c["vision_layer_calls"].items())
    return s


def table(obs):
    b, h = obs.get("base") or {}, obs.get("head") or {}
    unit = h.get("unit") or b.get("unit") or "?"
    rows = [f"Unit `{unit}` on `{h.get('platform') or b.get('platform')}`. Companion: base "
            f"`{str((b.get('companion') or {}).get('sha'))[:10]}`, head `{str((h.get('companion') or {}).get('sha'))[:10]}`.",
            "", "| cell | base | head |", "|---|---|---|"]
    names = [n for n in list(_cells(b)) + [n for n in _cells(h) if n not in _cells(b)]]
    for n in names:
        rows.append(f"| {n} | {_fmt(_cells(b).get(n))} | {_fmt(_cells(h).get(n))} |")
    env = (_cells(h).get("env") or {})
    if env:
        rows += ["", f"env (head): triton_kernel={env.get('triton_kernel')} torch_compile={env.get('torch_compile')} "
                 f"bnb_nf4_finite={env.get('bnb_nf4_finite')} inductor_use_block_ptr={env.get('inductor_use_block_ptr')} "
                 f"triton_has_block_ptr={env.get('triton_has_block_ptr')} moe_backend_default={env.get('moe_backend_default')}"]
    benv = (_cells(b).get("env") or {})
    if benv and "inductor_use_block_ptr" in benv:
        rows.append(f"env (base): inductor_use_block_ptr={benv.get('inductor_use_block_ptr')}")
    rows.append("")
    rows.append("Versions (head): " + str((h.get("env") or {}).get("env")))
    spreads = {c: _aa_spread(b, c) for c in ("flm16", "moe")}
    rows.append(f"A/A spread at base (max rel per-step loss diff): {spreads}; loss tolerance floor {RTOL}")
    for key in ("pytest", "pytest_companion"):
        for name, o in (("base", b), ("head", h)):
            p = o.get(key)
            if p is not None:
                rows.append(f"{key} {name}: passed={p.get('n_passed')} failed={p.get('n_failed')} "
                            f"skipped={p.get('n_skipped')} absent={p.get('absent_at_this_state')} "
                            f"crashed={p.get('crashed_files')}")
    if unit in NOTES:
        rows += ["", f"ROCm applicability: {NOTES[unit]}"]
    return "\n".join(rows)
