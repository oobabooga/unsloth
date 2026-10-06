#!/usr/bin/env python3
"""Criteria (unsloth-zoo PR 1582): fused BF16 / FP16 routed epilogues and Triton routed LoRA, gfx1151 ROCm.

Regression mode. Base (67299f48) already has the routed NF4 / BF16 decode path (PR 1580); the
PR adds fused gate_up+act / down+top-k-sum kernels for routed BF16 (UNSLOTH_MOE_ROUTED_FUSED,
default on, "0" = torch epilogues) and Triton routed_lora_a / routed_lora_b replacing the torch
LoRA glue (BF16 unfused path and the NF4 path, no switch). Head is WORSE than base when any of:
  * a head decode cell crashes or produces non-finite logits (any arm)
  * the head's UNSLOTH_MOE_ROUTED_KERNEL=0 arm is not bit-identical to the base's (same cell)
  * a cell whose routed path declined at the head is not bit-identical to the base default
  * a head cell that routes with neither BF16 LoRA nor NF4 LoRA (nothing the PR changes runs)
    is not bit-identical to the base default, in the default arm or the FUSED=0 arm
  * a BF16 routed LoRA cell does not run the fused ops in the default arm
    (routed_bf16_gate_up_act / routed_bf16_down_sum calls == 0), or still runs them with
    UNSLOTH_MOE_ROUTED_FUSED=0
  * an NF4 routed LoRA cell does not call routed_lora_a, or a BF16 LoRA FUSED=0 arm does not
    call routed_lora_a / routed_lora_b
  * a prefill-sized call engaged the routed path
  * the head default or FUSED=0 arm is less accurate vs the fp32 twin than the base default
    (max > 1.5x or mean > 1.25x), or flips a teacher-forced greedy argmax at a wide margin
  * a test that ran at the base fails at the head, or a test the PR adds fails at the head

Gates (non-vacuity): each state imported its own checkout; every base cell finished, finite;
both states have moe_routed, only the head has routed_bf16_gate_up_act; base FUSED=0 arm
== base default (the switch is inert at the base); the base routed path engaged in the
LoRA cells (else the comparison never touches the changed code); pytest ran and collected.
"""

from __future__ import annotations

TITLE = "unsloth-zoo PR 1582: fused routed BF16 epilogues + Triton routed LoRA on gfx1151 ROCm, base vs head"
MODE = "regression"
# nvidia: the PR's kernels were tuned / validated on CUDA (B200); NVIDIA behaviour unreachable here.
NEEDS = ["rocm", "gpu", "nvidia"]

_RAN = (0, 1)
_ARMS = ("arm0", "default_forced", "unfused_forced")


def _cells(o: dict) -> dict:
    return (o or {}).get("decode") or {}


def _done(c: dict) -> bool:
    return c.get("stage") == "done" and c.get("ok") is True and not c.get("error") and \
        all((c.get(a) or {}).get("finite") is True for a in _ARMS)


def _dec(c: dict, arm: str) -> dict:
    return ((c.get(arm) or {}).get("counts") or {}).get("decode") or {}


def _pre(c: dict, arm: str) -> dict:
    return ((c.get(arm) or {}).get("counts") or {}).get("prefill") or {}


def _sha(c: dict, arm: str):
    return (c.get(arm) or {}).get("logits_sha")


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out = []
    for name in ("base", "head"):
        o = obs.get(name) or {}
        ck = str(o.get("checkout") or "<none>")
        cells = _cells(o)
        files = {str(c.get("unsloth_zoo_file") or "") for c in cells.values()}
        out.append((f"{name} cells imported the state's unsloth_zoo", bool(cells) and all(f.startswith(ck) and f for f in files),
                    ", ".join(sorted(files)) or "no cells"))
        p = o.get("pytest") or {}
        out.append((f"{name} pytest ran", p.get("rc") in _RAN,
                    f"rc={p.get('rc')} " + (str(p.get("stderr_tail") or "")[-200:] if p.get("rc") not in _RAN else "")))
        out.append((f"{name} pytest collected", (p.get("n_passed", 0) + p.get("n_failed", 0) + p.get("n_skipped", 0)) > 0,
                    f"passed {p.get('n_passed', 0)}, failed {p.get('n_failed', 0)}, skipped {p.get('n_skipped', 0)}"))
    b, h = _cells(obs.get("base")), _cells(obs.get("head"))
    bad = [k for k, c in b.items() if not _done(c)]
    out.append(("every base decode cell finished, finite", bool(b) and not bad,
                "; ".join(f"{k}: {b[k].get('error') or b[k].get('stage')} rc={b[k].get('rc')}" for k in bad) or f"{len(b)} cells"))
    hstarted = [c for c in h.values() if c.get("stage") != "import"]
    out.append(("both states have moe_routed; only the head has routed_bf16_gate_up_act",
                bool(b) and bool(hstarted) and all(c.get("has_moe_routed") for c in b.values())
                and all(c.get("has_moe_routed") for c in hstarted)
                and all("routed_bf16_gate_up_act" in (c.get("wrapped_absent") or []) for c in b.values())
                and all("routed_bf16_gate_up_act" not in (c.get("wrapped_absent") or []) for c in hstarted),
                f"base absent {sorted({a for c in b.values() for a in c.get('wrapped_absent') or []})}, "
                f"head absent {sorted({a for c in hstarted for a in c.get('wrapped_absent') or []})}"))
    same = [k for k, c in b.items() if _done(c) and _sha(c, "unfused_forced") == _sha(c, "default_forced")]
    out.append(("base FUSED=0 arm == base default (switch inert at the base)", len(same) == len(b) and bool(b),
                f"{len(same)}/{len(b)} cells identical"))
    lora = [k for k in b if k.endswith("/lora")]
    eng = [k for k in lora if _done(b[k]) and _dec(b[k], "default_forced").get("routed_moe_forward_returned")]
    out.append(("base routed path engaged in every LoRA cell", bool(lora) and len(eng) == len(lora),
                f"{len(eng)}/{len(lora)}: {eng}"))
    return out


def _f(v, p = 4):
    return f"{v:.{p}g}" if isinstance(v, float) else str(v)


def table(obs: dict) -> str:
    rows = ["| state | cell | routed calls / returned (default) | gate_up_act / down_sum (default) | lora_a / lora_b (default) "
            "| gate_up_act / down_sum (FUSED=0) | lora_a / lora_b (FUSED=0) | torch _lora_delta / _lora_h (default) "
            "| prefill returned | '0' vs fp32 max / mean | default vs fp32 max / mean | FUSED=0 vs fp32 max / mean "
            "| max abs default-vs-0 | max abs default-vs-FUSED=0 | forced flips (unexplained) | '0' sha | default sha | FUSED=0 sha |",
            "|" + "---|" * 18]
    vers = None
    for name in ("base", "head", "merge"):
        for key, c in _cells(obs.get(name)).items():
            vers = vers or c.get("versions")
            if not _done(c):
                rows.append(f"| {name} | {key} | CRASH: {c.get('error') or c.get('stage')} rc={c.get('rc')} |" + " |" * 16)
                continue
            d, u, cmp_ = _dec(c, "default_forced"), _dec(c, "unfused_forced"), c.get("compare") or {}
            rows.append(
                f"| {name} | {key} | {d.get('routed_moe_forward_calls')} / {d.get('routed_moe_forward_returned')} "
                f"| {d.get('bf16_gate_up_act', '-')} / {d.get('bf16_down_sum', '-')} | {d.get('lora_a', '-')} / {d.get('lora_b', '-')} "
                f"| {u.get('bf16_gate_up_act', '-')} / {u.get('bf16_down_sum', '-')} | {u.get('lora_a', '-')} / {u.get('lora_b', '-')} "
                f"| {d.get('lora_delta_torch', '-')} / {d.get('lora_h_torch', '-')} "
                f"| {_pre(c, 'default_forced').get('routed_moe_forward_returned')} "
                f"| {_f(cmp_.get('max_abs_0_vs_fp32'))} / {_f(cmp_.get('mean_abs_0_vs_fp32'))} "
                f"| {_f(cmp_.get('max_abs_default_vs_fp32'))} / {_f(cmp_.get('mean_abs_default_vs_fp32'))} "
                f"| {_f(cmp_.get('max_abs_unfused_vs_fp32'))} / {_f(cmp_.get('mean_abs_unfused_vs_fp32'))} "
                f"| {_f(cmp_.get('max_abs_default_vs_0'))} | {_f(cmp_.get('max_abs_default_vs_unfused'))} "
                f"| {cmp_.get('n_forced_flips')} ({cmp_.get('n_unexplained_flips')}) "
                f"| {_sha(c, 'arm0')} | {_sha(c, 'default_forced')} | {_sha(c, 'unfused_forced')} |")
    rows += ["", "| state | pytest passed | failed | skipped | failing tests |", "|---|---|---|---|---|"]
    for name in ("base", "head", "merge"):
        p = (obs.get(name) or {}).get("pytest")
        if not p:
            continue
        bad = ", ".join(f"`{t.split('::')[-1]}`" for t in p.get("failed", []) + p.get("errors", [])) or "none"
        rows.append(f"| {name} | {p.get('n_passed')} | {p.get('n_failed')} | {p.get('n_skipped')} | {bad} |")
    for name in ("base", "head"):
        p = (obs.get(name) or {}).get("pytest") or {}
        per: dict = {}
        for kind, ids in (("passed", p.get("passed") or []), ("failed", (p.get("failed") or []) + (p.get("errors") or [])),
                          ("skipped", list((p.get("skipped") or {}).keys()))):
            for tid in ids:
                parts = tid.split("::")[0].split(".")
                f = next((x for x in parts if x.startswith("test_")), parts[-1])
                per.setdefault(f, {"passed": 0, "failed": 0, "skipped": 0})[kind] += 1
        if per:
            rows += ["", f"{name} per file: " + "; ".join(f"`{f}` {v['passed']}p / {v['failed']}f / {v['skipped']}s"
                                                           for f, v in sorted(per.items()))]
    hp = (obs.get("head") or {}).get("pytest") or {}
    reasons: dict = {}
    for tid, msg in (hp.get("skipped") or {}).items():
        reasons.setdefault(msg, []).append(tid.split("::")[-1])
    if reasons:
        rows += ["", "Head skip reasons:"]
        for msg, tids in sorted(reasons.items(), key = lambda kv: -len(kv[1])):
            rows.append(f"- {len(tids)}x `{msg}` (e.g. `{tids[0]}`)")
    if vers:
        rows += ["", "Versions: " + ", ".join(f"{k}={v}" for k, v in vers.items())]
    return "\n".join(rows)


def head_is_worse(base: dict, head: dict) -> tuple[bool, str]:
    problems, notes = [], []
    bc, hc = _cells(base), _cells(head)
    fused_eng, lora_eng, identical, changed = [], [], [], []
    for key, c in hc.items():
        if not _done(c):
            problems.append(f"head {key} did not finish: {c.get('error') or ('rc=' + str(c.get('rc')) + ' stage=' + str(c.get('stage')))}")
            continue
        b = bc.get(key) or {}
        if not _done(b):
            continue
        h0, hd, hu = _sha(c, "arm0"), _sha(c, "default_forced"), _sha(c, "unfused_forced")
        b0, bd = _sha(b, "arm0"), _sha(b, "default_forced")
        if h0 != b0:
            problems.append(f"{key}: head UNSLOTH_MOE_ROUTED_KERNEL=0 logits ({h0}) not bit-identical to base ({b0})")
        for arm in ("default_forced", "unfused_forced"):
            if _pre(c, arm).get("routed_moe_forward_returned"):
                problems.append(f"{key}: a prefill-sized call took the routed path in {arm} ({_pre(c, arm)})")
        d, u = _dec(c, "default_forced"), _dec(c, "unfused_forced")
        routed = d.get("routed_moe_forward_returned") or 0
        is_lora, is_bf16 = key.endswith("/lora"), "/bf16/" in key
        if not routed or not is_lora:
            # Nothing the PR changes runs here: default AND FUSED=0 must be exactly the base.
            for arm, s in (("default", hd), ("FUSED=0", hu)):
                if s != bd:
                    problems.append(f"{key}: {arm} logits ({s}) differ from base default ({bd}) though "
                                    f"{'the routed path declined' if not routed else 'no LoRA routes'}")
            if d.get("bf16_gate_up_act") or u.get("bf16_gate_up_act"):
                notes.append(f"{key}: fused ops ran in a no-LoRA cell ({d}, {u})")
            identical.append(key)
            continue
        if is_bf16:
            if not (d.get("bf16_gate_up_act") and d.get("bf16_down_sum")):
                problems.append(f"{key}: fused routed_bf16_gate_up_act / down_sum not engaged in the default arm ({d})")
            else:
                fused_eng.append(f"{key} gate_up_act {d['bf16_gate_up_act']} / down_sum {d['bf16_down_sum']} "
                                 f"(launches {d.get('launch_gate_up_act')} / {d.get('launch_down_sum')})")
            if u.get("bf16_gate_up_act") or u.get("bf16_down_sum"):
                problems.append(f"{key}: UNSLOTH_MOE_ROUTED_FUSED=0 still ran the fused ops ({u})")
            if not (u.get("lora_a") and u.get("lora_b")):
                problems.append(f"{key}: FUSED=0 arm did not call routed_lora_a / routed_lora_b ({u})")
            else:
                lora_eng.append(f"{key} FUSED=0 lora_a {u['lora_a']} / lora_b {u['lora_b']}")
        else:
            if not d.get("lora_a"):
                problems.append(f"{key}: NF4 LoRA cell did not call routed_lora_a ({d})")
            else:
                lora_eng.append(f"{key} lora_a {d['lora_a']} / lora_b {d.get('lora_b')}")
        changed.append(key)
        bcmp, cmp_ = b.get("compare") or {}, c.get("compare") or {}
        bx, bm = bcmp.get("max_abs_default_vs_fp32"), bcmp.get("mean_abs_default_vs_fp32")
        for arm, x, m in (("default", cmp_.get("max_abs_default_vs_fp32"), cmp_.get("mean_abs_default_vs_fp32")),
                          ("FUSED=0", cmp_.get("max_abs_unfused_vs_fp32"), cmp_.get("mean_abs_unfused_vs_fp32"))):
            if None in (x, m, bx, bm):
                problems.append(f"{key}: comparison missing ({arm})")
                continue
            if x > 1.5 * bx or m > 1.25 * bm:
                problems.append(f"{key}: head {arm} less accurate vs fp32 than base routed (max {x:.4g} vs {bx:.4g}, "
                                f"mean {m:.4g} vs {bm:.4g})")
            else:
                notes.append(f"{key}: vs fp32 max/mean base {bx:.4g}/{bm:.4g}, head {arm} {x:.4g}/{m:.4g}")
        if cmp_.get("n_unexplained_flips"):
            problems.append(f"{key}: greedy argmax flipped at a wide margin: {c.get('forced_flips')}")
        elif cmp_.get("n_forced_flips"):
            notes.append(f"{key}: {cmp_['n_forced_flips']} forced argmax flip(s) vs '0', all near ties")
        if hu == _sha(b, "default_forced"):
            notes.append(f"{key}: FUSED=0 bit-identical to base routed")
        else:
            notes.append(f"{key}: FUSED=0 not bit-identical to base routed (expected: the torch LoRA glue was "
                         f"replaced by routed_lora_a / routed_lora_b, no switch)")
    notes.insert(0, "fused BF16 ops ENGAGED in " + (", ".join(fused_eng) or "no cell")
                 + "; Triton routed LoRA ENGAGED in " + (", ".join(lora_eng) or "no cell")
                 + "; bit-identical-to-base required (nothing changed runs) in " + (", ".join(identical) or "no cell"))
    bp, hp = base.get("pytest") or {}, head.get("pytest") or {}
    b = set(bp.get("failed", [])) | set(bp.get("errors", []))
    h = set(hp.get("failed", [])) | set(hp.get("errors", []))
    new = sorted(h - b)
    if new:
        problems.append("failing at the head (not failing at the base): " + ", ".join(f"`{n.split('::')[-1]}`" for n in new))
    if b & h:
        notes.append("failing at BOTH states (pre-existing): " + ", ".join(f"`{n.split('::')[-1]}`" for n in sorted(b & h)))
    if problems:
        return True, "; ".join(problems) + ". Also: " + "; ".join(notes)
    return False, "; ".join(notes)
