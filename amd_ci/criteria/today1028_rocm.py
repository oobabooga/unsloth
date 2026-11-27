#!/usr/bin/env python3
"""Criteria: unsloth main 2026-10-28 (25fd237aad -> 987731ec56) on gfx1151 ROCm, base vs head.

Regression mode. Head is WORSE than base when any of:
  * a head cell crashes where the base cell finished (sft16 / sft4 / packed / infer / decisions)
  * sft16 / sft4 / packed per-step losses differ head vs base beyond RTOL; infer generated ids differ
  * block CUDA graphs auto (#15768) at head is not off for "ROCm", turns triton.cudagraphs on, or the head
    model carries more forward pre-hooks than the base (or any _unsloth_watch_hidden_states flag)
  * the varlen-over-flex switch (#15514) returned True / moved attention off flex in any head cell, or
    torch varlen is offered on ROCm
  * FP8 rowwise (#15584 / #15791): head takes the direct route, a head case fails where base ran, or a
    head output digest differs from the base's
  * CE chunk config / MoE heuristic configs / SDPA decode outputs differ head vs base
  * a test id fails / errors at the head that did not at the base (new head-only files included)
Gates (non-vacuity): each cell imported its own checkout's unsloth on HIP; base sft16, infer, decisions finished;
block compile wrapped a class in head sft16 and its auto state was consulted (phase recorded); pytest collected.
"""

from __future__ import annotations

TITLE = "unsloth main 2026-10-28 (25fd237aad -> 987731ec56) on gfx1151 ROCm: LoRA SFT, inference, routing, tests"
MODE = "regression"
NEEDS = ["rocm", "gpu", "nvidia", "windows", "multi_gpu"]
RTOL = 2e-2
TRAIN = ("sft16", "sft4", "packed", "packed_r2")


def _cell(o, k):
    return ((o or {}).get("cells") or {}).get(k) or {}


def _done(c):
    return c.get("stage") == "done" and c.get("ok") is True and not c.get("error")


def _close(a, b):
    if not a or len(a) != len(b):
        return False
    return all(abs(x - y) <= RTOL * max(1.0, abs(x)) for x, y in zip(a, b))


def gates(obs):
    out = []
    for name in ("base", "head"):
        o = obs.get(name) or {}
        ck = str(o.get("checkout") or "<none>")
        cells = o.get("cells") or {}
        files = {k: str(c.get("unsloth_file") or "") for k, c in cells.items() if c.get("unsloth_file")}
        out.append((f"{name} cells imported the state's unsloth", bool(files) and all(f.startswith(ck) for f in files.values()),
                    ", ".join(sorted(set(files.values()))) or "no cells"))
        hips = {str((c.get("versions") or {}).get("hip")) for c in cells.values() if c.get("versions")}
        out.append((f"{name} cells ran on HIP", bool(hips) and "None" not in hips, f"hip {hips}"))
        p = o.get("pytest") or {}
        n = p.get("n_passed", 0) + p.get("n_failed", 0) + p.get("n_skipped", 0)
        out.append((f"{name} pytest collected", n > 0, f"rc={p.get('rc')} passed {p.get('n_passed')}, failed "
                    f"{p.get('n_failed')}, skipped {p.get('n_skipped')} {str(p.get('stderr_tail') or '')[-200:] if n == 0 else ''}"))
    b = obs.get("base") or {}
    for k in ("decisions", "sft16", "infer"):
        c = _cell(b, k)
        out.append((f"base {k} finished", _done(c), str(c.get("error") or c.get("stage"))[:400]))
    h16 = _cell(obs.get("head"), "sft16")
    hd = _cell(obs.get("head"), "decisions").get("block_graphs") or {}
    out.append(("head: block-graph auto decision observable (in training, or the platform gate read directly)",
                (h16.get("block_graphs") or {}).get("auto_phase") in ("off", "on") or hd.get("has_auto") is True,
                f"sft16 wrapped {h16.get('block_compile_wrapped')} state {h16.get('block_graphs')}; decisions {hd}"))
    return out


def head_is_worse(base, head):
    problems, notes = [], []
    for name, o in (("base", base), ("head", head)):
        mc = _cell(o, "moe_control")
        notes.append(f"moe_control (tiny Qwen3.5 MoE fp32 on GPU, no multi-LoRA) at {name}: rc {mc.get('rc')} steps "
                     f"{mc.get('steps')} died in {mc.get('current_step')!r}")
    for k in ("decisions",) + TRAIN + ("infer",):
        bc, hc = _cell(base, k), _cell(head, k)
        if _done(bc) and not _done(hc):
            problems.append(f"head {k} crashed (base finished): {str(hc.get('error') or hc.get('stage'))[:400]}")
        elif not _done(bc) and not _done(hc):
            notes.append(f"{k} did not finish at either state (pre-existing / unsupported here): base "
                         f"{str(bc.get('error'))[:200]}; head {str(hc.get('error'))[:200]}")
        elif not _done(bc) and _done(hc):
            notes.append(f"{k} failed at base, finished at head: base {str(bc.get('error'))[:200]}")
    for k in TRAIN:
        bc, hc = _cell(base, k), _cell(head, k)
        if _done(bc) and _done(hc):
            same = _close(bc.get("losses"), hc.get("losses"))
            line = f"{k} losses base {bc.get('losses')} head {hc.get('losses')}"
            (notes if same else problems).append(("" if same else "LOSS DRIFT ") + line)
    bi, hi = _cell(base, "infer"), _cell(head, "infer")
    if _done(bi) and _done(hi):
        same = bi.get("generated_ids") == hi.get("generated_ids")
        line = (f"infer greedy ids equal={same} (n={hi.get('n_new')}), logits digest base {bi.get('logits_digest')} "
                f"head {hi.get('logits_digest')}")
        (notes if same else problems).append(line)

    # block CUDA graphs auto (#15768)
    h16, b16 = _cell(head, "sft16"), _cell(base, "sft16")
    bg = h16.get("block_graphs") or {}
    hd = (_cell(head, "decisions").get("block_graphs") or {})
    exercised = any(x for x in (h16.get("block_compile_wrapped") or []))
    ok = (not bg.get("triton_cudagraphs") and hd.get("auto_platform_reason") == "ROCm"
          and (not exercised or (bg.get("auto_phase") == "off" and bg.get("auto_why") == "ROCm")))
    notes.append(f"block compile exercised in head sft16 training: {exercised}")
    line = (f"block graphs at head: mode {bg.get('mode')} phase {bg.get('auto_phase')} why {bg.get('auto_why')!r} "
            f"triton.cudagraphs {bg.get('triton_cudagraphs')}; platform reason {hd.get('auto_platform_reason')!r}; "
            f"static reason on a CUDA tensor {hd.get('auto_static_reason_cuda_tensor')!r}")
    (notes if ok else problems).append(line)
    for k in TRAIN:
        bc, hc = _cell(base, k), _cell(head, k)
        if _done(bc) and _done(hc):
            bh, hh = bc.get("hooks_after_train") or {}, hc.get("hooks_after_train") or {}
            extra = (hh.get("forward_pre_hooks") or 0) - (bh.get("forward_pre_hooks") or 0)
            line = (f"{k} forward pre-hooks after train base {bh.get('forward_pre_hooks')} head {hh.get('forward_pre_hooks')}, "
                    f"watch flags head {hh.get('watch_hidden_states_flags')}; wrapped head {hc.get('block_compile_wrapped')}; "
                    f"head graph state {(hc.get('block_graphs') or {}).get('auto_why')!r}")
            (problems if extra > 0 or hh.get("watch_hidden_states_flags") else notes).append(line)

    # varlen over flex (#15514)
    hv = _cell(head, "decisions").get("varlen") or {}
    sw = [c for k in TRAIN for c in (_cell(head, k).get("varlen_switch_calls") or [])]
    engaged = any(c.get("returned") is True for c in sw) or hv.get("forced_switch_returned") is True
    tv_none = hv.get("torch_varlen") == "None"
    line = (f"varlen-over-flex at head: torch varlen {hv.get('torch_varlen')} (torch has varlen_attn: "
            f"{hv.get('torch_has_varlen_attn')}); forced-reach call returned {hv.get('forced_switch_returned')!r}, attn after "
            f"{hv.get('forced_switch_attn_after')!r}, gate calls {hv.get('forced_switch_calls')}; training switch calls {sw}; "
            f"packed attn loaded {_cell(head, 'packed').get('attn_impl_loaded')!r} after train "
            f"{_cell(head, 'packed').get('attn_impl_after_train')!r} (base {_cell(base, 'packed').get('attn_impl_after_train')!r})")
    (problems if engaged or not tv_none else notes).append(line)
    bp, hp = _cell(base, "packed"), _cell(head, "packed")
    if _done(bp) and _done(hp) and bp.get("attn_impl_after_train") != hp.get("attn_impl_after_train"):
        problems.append(f"packed attention implementation changed: base {bp.get('attn_impl_after_train')} head {hp.get('attn_impl_after_train')}")

    # FP8 (#15584 / #15791)
    bf, hf = _cell(base, "decisions").get("fp8") or {}, _cell(head, "decisions").get("fp8") or {}
    diffs, direct = [], []
    for rows, hcse in (hf.get("cases") or {}).items():
        bcse = (bf.get("cases") or {}).get(rows) or {}
        if hcse.get("direct_route") not in (None, "None") or (hcse.get("spies") or {}).get("_fp8_rowwise_direct"):
            direct.append(rows)
        if bcse.get("ok") and not hcse.get("ok"):
            diffs.append(f"rows {rows}: head failed {hcse.get('error')}")
        elif bcse.get("ok") and hcse.get("ok") and bcse.get("digest") != hcse.get("digest"):
            diffs.append(f"rows {rows}: digest base {bcse.get('digest')} head {hcse.get('digest')}")
    line = (f"FP8 rowwise: backend base {bf.get('backend')!r} head {hf.get('backend')!r}; direct route taken at rows {direct}; "
            f"head dequant max rows (large weight) {hf.get('dequant_max_rows_large')}; ok base "
            f"{sum(1 for c in (bf.get('cases') or {}).values() if c.get('ok'))}/{len(bf.get('cases') or {})} head "
            f"{sum(1 for c in (hf.get('cases') or {}).values() if c.get('ok'))}/{len(hf.get('cases') or {})}; "
            f"head spies {[c.get('spies') for c in (hf.get('cases') or {}).values()][:2]}")
    if direct or diffs or bf.get("backend") != hf.get("backend"):
        problems.append("FP8 ROUTING CHANGED " + line + "; " + "; ".join(diffs))
    else:
        notes.append(line)
        errs = {c.get("error") for c in (hf.get("cases") or {}).values() if not c.get("ok")}
        if errs:
            notes.append(f"FP8 cases failing at BOTH states (pre-existing): {sorted(map(str, errs))[:3]}")

    # CE / MoE / SDPA decode
    bd, hd2 = _cell(base, "decisions"), _cell(head, "decisions")
    for key in ("ce", "moe"):
        b_, h_ = dict(bd.get(key) or {}), dict(hd2.get(key) or {})
        b_.pop("wide_dw_device_ok", None)
        wd = h_.pop("wide_dw_device_ok", None)
        line = f"{key} base {b_} head {h_}" + (f" (head _wide_dw_device_ok {wd})" if key == "moe" else "")
        (notes if b_ == h_ else problems).append(line)
    bs, hs = bd.get("sdpa_decode") or {}, hd2.get("sdpa_decode") or {}
    for k in ("default", "math_only"):
        b_, h_ = bs.get(k) or {}, hs.get(k) or {}
        same = b_.get("ok") == h_.get("ok") and b_.get("digest") == h_.get("digest")
        line = (f"sdpa decode {k}: base {b_.get('digest') or b_.get('error')} head {h_.get('digest') or h_.get('error')} "
                f"(head cudnn_free_index {hs.get('cudnn_free_index')})")
        (notes if same else problems).append(line)

    # packed A/A (same state twice): noise floor for the packed head-vs-base loss comparison
    for name, o in (("base", base), ("head", head)):
        a, b2 = _cell(o, "packed"), _cell(o, "packed_r2")
        if _done(a) and _done(b2):
            d = max((abs(x - y) for x, y in zip(a.get("losses") or [], b2.get("losses") or [])), default = None)
            notes.append(f"packed A/A at {name}: max |dloss| {d}")
    # pytest: whole-file native crashes first
    bcr = set(((base or {}).get("pytest") or {}).get("crashed_files") or [])
    hcr = set(((head or {}).get("pytest") or {}).get("crashed_files") or [])
    if hcr - bcr:
        problems.append(f"test files whose pytest process crashed at head, not at base: {sorted(hcr - bcr)} "
                        f"{[((head.get('pytest') or {}).get('files') or {}).get(t, {}).get('stderr_tail', '')[-600:] for t in sorted(hcr - bcr)]}")
    if hcr & bcr:
        notes.append(f"test files crashing at both states (pre-existing): {sorted(hcr & bcr)}")
    # pytest by id
    bp_, hp_ = (base or {}).get("pytest") or {}, (head or {}).get("pytest") or {}
    bpass = set(bp_.get("passed") or [])
    hbad = set(hp_.get("failed") or []) | set(hp_.get("errors") or [])
    bbad = set(bp_.get("failed") or []) | set(bp_.get("errors") or [])
    lost = sorted(t for t in bpass if t not in set(hp_.get("passed") or []) and t not in (hp_.get("skipped") or {}))
    if lost:
        problems.append(f"passed at base, not at head: {lost}")
    newf = sorted(hbad - bbad)
    if newf:
        problems.append(f"failing at head, not at base: {newf}")
    pre = sorted(hbad & bbad)
    if pre:
        notes.append(f"failing at both (pre-existing): {pre}")
    notes.append(f"pytest base {bp_.get('n_passed')}/{bp_.get('n_failed')}/{bp_.get('n_skipped')} head "
                 f"{hp_.get('n_passed')}/{hp_.get('n_failed')}/{hp_.get('n_skipped')} (passed/failed/skipped)")
    if problems:
        return True, "; ".join(problems)
    return False, "; ".join(notes)


def table(obs):
    rows = ["| state | cell | stage | losses | attn loaded -> after | wrapped | graph state | hooks (pre, watch) |",
            "|" + "---|" * 8]
    for name in ("base", "head"):
        for k, c in ((obs.get(name) or {}).get("cells") or {}).items():
            g = c.get("block_graphs") or {}
            h = c.get("hooks_after_train") or {}
            rows.append(f"| {name} | {k} | {c.get('stage')}{' ERR ' + str(c.get('error'))[:200] if c.get('error') else ''} | "
                        f"{c.get('losses') or c.get('generated_ids') or ''} | {c.get('attn_impl_loaded')} -> {c.get('attn_impl_after_train')} | "
                        f"{c.get('block_compile_wrapped')} | {g.get('auto_phase')}/{g.get('auto_why')} | "
                        f"{h.get('forward_pre_hooks')}, {h.get('watch_hidden_states_flags')} |")
    rows.append("")
    for name in ("base", "head"):
        v = (_cell(obs.get(name), "decisions").get("versions")) or {}
        rows.append(f"{name} versions: {v}")
    rows += ["", "| state | pytest passed | failed | skipped | failing tests |", "|---|---|---|---|---|"]
    for name in ("base", "head"):
        p = (obs.get(name) or {}).get("pytest") or {}
        bad = sorted(set(p.get("failed") or []) | set(p.get("errors") or []))
        rows.append(f"| {name} | {p.get('n_passed')} | {p.get('n_failed')} | {p.get('n_skipped')} | {', '.join(bad) or 'none'} |")
    rows.append("")
    for name in ("base", "head"):
        p = (obs.get(name) or {}).get("pytest") or {}
        for t, m in sorted((p.get("fail_msgs") or {}).items()):
            rows.append(f"{name} FAIL `{t}`: {m[:500]!r}")
        rows.append("")
    return "\n".join(rows)
