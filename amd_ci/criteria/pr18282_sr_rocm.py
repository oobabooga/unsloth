#!/usr/bin/env python3
"""Criteria (unsloth #18282): one-pass Triton AdamW + stochastic rounding for bf16 full fine-tuning, gfx1151.

Regression mode (training-path rule: the head must not be worse than main on this platform). Head is WORSE when any:
  * a test that ran at the base (passed) does not pass at the head; or a head-only test
    (tests/test_bf16_stochastic_rounding_fused_gpu.py) fails or errors at the head
  * an SFT cell finished at the base but not at the head (crash, non-finite weights)
  * SR on with torch AdamW / bitsandbytes: the head neither engaged its one-pass path (counter > 0) nor fell back
    cleanly (per-device Triton probe False, counter 0, the run still finished): a silent non-engagement with a
    passing probe is worse
  * the loss drop (step-1 loss minus the mean of the last 10 steps) differs from the base cell's by more than
    max(10% of the base drop, 3 x the base's own spread between its adamw_torch and adamw_torch_fused SR cells).
    Calibrated on B200 (tiny Qwen3, lr 1e-4, 30 steps): SR drop 0.00764 base / 0.00766 head, SR off 0.00575 (-25%),
    so an SR path that stopped keeping small updates is caught; a 1%-of-loss band would not be (the drop is 0.06%).
Gates (non-vacuity): each state loaded its own bf16_stochastic_rounding module; head has the kernel, base does not;
pytest ran and collected in both; the base SR adamw_torch cell finished on the gfx1151 GPU with SR wrapping > 0
weights; the SR-off cell finished at the base (scale reference).
"""
from __future__ import annotations

TITLE = "unsloth #18282: one-pass bf16 SR AdamW kernel on gfx1151, base vs head (tests + tiny bf16 full-FT SFT)"
MODE = "regression"
NEEDS = ["rocm", "gpu", "nvidia", "multi_gpu", "windows"]

NEW_TEST = "test_bf16_stochastic_rounding_fused_gpu"
SR_CELLS = ("adamw_torch_sr1", "adamw_torch_fused_sr1", "adamw_8bit_sr1")
ALL_CELLS = SR_CELLS + ("adamw_torch_sr0",)


def _cases(o):
    return ((o or {}).get("test") or {}).get("cases") or {}


def _sft(o, k):
    return (((o or {}).get("sft") or {}).get(k)) or {}


def _tail_mean(c, n = 10):
    ls = c.get("losses") or {}
    steps = sorted(int(s) for s in ls)
    if len(steps) < n:
        return None
    return sum(ls[str(s)] if str(s) in ls else ls[s] for s in steps[-n:]) / n


def _drop(c):
    ls = c.get("losses") or {}
    steps = sorted(int(s) for s in ls)
    tail = _tail_mean(c)
    if not steps or tail is None:
        return None
    first = ls[str(steps[0])] if str(steps[0]) in ls else ls[steps[0]]
    return first - tail


def _ok(c):
    return c.get("stage") == "done" and c.get("finite_weights") is True


def _tol(base):
    a, b = _drop(_sft(base, "adamw_torch_sr1")), _drop(_sft(base, "adamw_torch_fused_sr1"))
    if a is None:
        return None
    spread = abs(a - b) if b is not None else 0.0
    return max(0.10 * abs(a), 3 * spread)


def gates(obs):
    out = []
    for name in ("base", "head"):
        o = obs.get(name) or {}
        imp = o.get("imports") or {}
        f = str(imp.get("sr_file") or "").replace("\\", "/")
        ck = str(o.get("checkout") or "").replace("\\", "/")
        out.append((f"{name} loaded its own bf16_stochastic_rounding", bool(f) and bool(ck) and f.startswith(ck),
                    f or str(imp.get("sr_error"))[:300]))
        t = o.get("test") or {}
        out.append((f"{name} pytest ran and collected", t.get("rc") in (0, 1) and len(_cases(o)) > 0,
                    f"rc={t.get('rc')} cases={len(_cases(o))} {str(t.get('stderr_tail', ''))[-200:] if t.get('rc') not in (0, 1) else ''}"))
    h, b = obs.get("head") or {}, obs.get("base") or {}
    out.append(("head has the one-pass kernel", (h.get("imports") or {}).get("has_adamw_kernel") is True, ""))
    out.append(("base lacks the one-pass kernel", (b.get("imports") or {}).get("has_adamw_kernel") is False, ""))
    c = _sft(b, "adamw_torch_sr1")
    v = c.get("versions") or {}
    out.append(("base SR adamw_torch cell finished on the gfx1151 GPU with SR on",
                _ok(c) and str(c.get("model_device", "")).startswith("cuda") and "gfx1151" in str(v.get("arch"))
                and (c.get("sr_wrapped") or 0) > 0,
                f"stage={c.get('stage')} {c.get('error') or ''} device={c.get('model_device')} arch={v.get('arch')} "
                f"wrapped={c.get('sr_wrapped')}"[:400]))
    c0 = _sft(b, "adamw_torch_sr0")
    out.append(("base SR-off cell finished (scale reference)", _ok(c0), f"stage={c0.get('stage')} {c0.get('error') or ''}"[:300]))
    return out


def head_is_worse(base, head):
    bad = []
    bc, hc = _cases(base), _cases(head)
    for tid, c in hc.items():
        short = tid.split("::")[-1]
        if tid in bc:
            if bc[tid].get("outcome") == "passed" and c.get("outcome") != "passed":
                bad.append(f"`{short}` passed at base, {c.get('outcome')} at head")
        elif NEW_TEST in tid and c.get("outcome") in ("failed", "error"):
            bad.append(f"head-only `{short}` {c.get('outcome')}: {' '.join(c.get('message', '').split())[:200]}")
    tol = _tol(base)
    for k in ALL_CELLS:
        bcell, hcell = _sft(base, k), _sft(head, k)
        if _ok(bcell) and not _ok(hcell):
            bad.append(f"SFT {k}: base finished, head {hcell.get('stage')} {hcell.get('error') or ''}"[:300])
            continue
        if not (_ok(bcell) and _ok(hcell)):
            continue
        if k in SR_CELLS:
            counter = hcell.get("sr_triton_rounds") if "8bit" in k else hcell.get("sr_fused_steps")
            probe = list((hcell.get("triton_ok") or {}).values())
            engaged = (counter or 0) > 0
            no_triton = bool((head.get("imports") or {}).get("triton_error"))
            clean_fallback = (counter or 0) == 0 and ((probe and not any(probe)) or (no_triton and not probe))
            if not (engaged or clean_fallback):
                bad.append(f"SFT {k}: one-pass path neither engaged (counter {counter}) nor cleanly refused (probe {probe})")
        mb, mh = _drop(bcell), _drop(hcell)
        if tol is not None and mb is not None and mh is not None and abs(mh - mb) > tol:
            bad.append(f"SFT {k}: loss drop head {mh:.5f} vs base {mb:.5f} (|d| {abs(mh - mb):.5f} > tol {tol:.5f})")
    if bad:
        return True, "; ".join(bad)
    return False, (f"no base-passing test fails at the head, head-only tests pass, every SFT cell that finished at the "
                   f"base finished at the head with loss within tol {tol}")


def table(obs):
    rows = ["| state | commit | torch / triton / arch | probe | " + " | ".join(ALL_CELLS) + " |",
            "|---|---|---|---|" + "---|" * len(ALL_CELLS)]
    for name in ("base", "head", "merge"):
        o = obs.get(name)
        if not o:
            continue
        imp = o.get("imports") or {}
        cells = []
        for k in ALL_CELLS:
            c = _sft(o, k)
            if not _ok(c):
                cells.append(f"{c.get('stage')}: {' '.join(str(c.get('error') or '').split())[:120]}")
                continue
            m, d = _tail_mean(c), _drop(c)
            cells.append(f"last10 {m if m is None else round(m, 5)}, drop {d if d is None else round(d, 5)}; fused {c.get('sr_fused_steps')}, rounds {c.get('sr_triton_rounds')}, "
                         f"wrapped {c.get('sr_wrapped')}, probe {c.get('triton_ok')}"
                         + (f", msg {c.get('fallback_messages')[:1]}" if c.get("fallback_messages") else ""))
        rows.append(f"| {name} | {str(o.get('commit'))[:10]} | {imp.get('torch')} / {imp.get('triton') or imp.get('triton_error')} / "
                    f"{imp.get('arch')} | {imp.get('triton_ok_probe')} | " + " | ".join(x.replace("|", "/") for x in cells) + " |")
    lines = ["\n".join(rows), "", f"Loss tolerance from the base: {_tol(obs.get('base') or {})}", "", "Per-test ids:"]
    for name in ("base", "head", "merge"):
        o = obs.get(name)
        if o:
            lines.append(f"- {name}: " + ", ".join(f"`{t.split('::')[-1]}` {c.get('outcome')}"
                                                   for t, c in sorted(_cases(o).items())))
            fails = [(t, c) for t, c in sorted(_cases(o).items()) if c.get("outcome") in ("failed", "error")]
            for t, c in fails[:12]:
                lines.append(f"  - {name} {c.get('outcome')} `{t.split('::')[-1]}`: {' '.join(c.get('message', '').split())[:300]}")
    return "\n".join(lines)
