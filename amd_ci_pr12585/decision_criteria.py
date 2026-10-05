#!/usr/bin/env python3
"""Criteria for PR 12585 decision_probe: judges, never observes.

Regression mode. The control (plain tiny Qwen3 LoRA SFT) exists at base and head and must
not get worse. The decision scenarios exist only at the head (the PR adds them); a head
scenario that crashes or trains non-finite counts as worse. 4-bit is allowed to refuse
with an Unsloth message naming 4-bit / bitsandbytes (a clear refusal), not to crash.
"""

from __future__ import annotations

import math

TITLE = "PR 12585 decision models on gfx1151: control SFT, tiny Clef bf16/fp16/4-bit, Laya 20 steps"
MODE = "regression"
NEEDS = ["gpu", "nvidia", "multi_gpu"]
DECISION = ("clef_bf16", "clef_fp16", "clef_4bit", "laya")


def _sc(state: dict, name: str) -> dict:
    return (state.get("scenarios") or {}).get(name) or {}


def _refused_clearly(rec: dict) -> bool:
    err = (rec.get("error") or "")
    head = err.split(":", 1)[0]
    return (head in ("ValueError", "RuntimeError", "NotImplementedError")
            and "Unsloth" in err and any(k in err.lower() for k in ("4-bit", "4bit", "bitsandbytes", "qlora")))


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    base, head = obs.get("base") or {}, obs.get("head") or {}
    out = []
    for n, st in (("base", base), ("head", head)):
        c = _sc(st, "control")
        out.append((f"control SFT trained at {n}", bool(c.get("ok")),
                    c.get("error", "") or f"{len(c.get('losses') or [])} steps"))
    missing = [n for n in DECISION if not _sc(head, n) or _sc(head, n).get("no_result")]
    out.append(("every decision scenario produced a result at head", not missing,
                ", ".join(missing) or "all"))
    absent_base = [n for n in DECISION if _sc(base, n).get("absent")]
    out.append(("base has no FastDecisionModel (feature is new)", len(absent_base) == len(DECISION),
                f"{len(absent_base)}/{len(DECISION)} absent"))
    hd = _sc(head, "control").get("unsloth_file") or ""
    out.append(("head imported its own checkout", bool(hd) and str(head.get("checkout", "")) in hd, hd))
    return out


def _fmt_logs(rec: dict) -> str:
    ls = rec.get("logs") or rec.get("losses") or []
    if not ls:
        return "-"
    pick = ls if len(ls) <= 5 else ls[:2] + ls[-2:]
    return ", ".join(f"{l:.3f}/{g:.2f}" for l, g in pick)


def table(obs: dict) -> str:
    rows = ["| state | scenario | ok | loss/grad (first, last) | detail |", "|---|---|---|---|---|"]
    for sn in ("base", "head"):
        st = obs.get(sn) or {}
        for name, rec in (st.get("scenarios") or {}).items():
            if rec.get("absent"):
                rows.append(f"| {sn} | {name} | absent | - | not in this tree |")
                continue
            det = []
            for k in ("forced_float32", "fast_backbone", "clef_compile_supported", "linear4bit_modules",
                      "trainer_fp16", "trainer_bf16", "eval_loss", "s_per_step_median", "peak_gb",
                      "encoder_reference_compile", "train_s"):
                if k in rec:
                    v = rec[k]
                    det.append(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}")
            for k in ("acc_before", "acc_after"):
                if isinstance(rec.get(k), dict):
                    det.append(f"{k}=" + ",".join(f"{a}={b:.3f}" for a, b in rec[k].items()
                                                   if isinstance(b, (int, float))))
            if rec.get("gated_delta_kernels"):
                det.append("kernels=" + str(rec["gated_delta_kernels"]))
            if rec.get("dynamo"):
                det.append("dynamo=" + str(rec["dynamo"]))
            if rec.get("error"):
                det.append("ERROR " + rec["error"][:400].replace("|", "/").replace("\n", " "))
            ok = "yes" if rec.get("ok") else ("refused" if _refused_clearly(rec) else "NO")
            rows.append(f"| {sn} | {name} | {ok} | {_fmt_logs(rec)} | {'; '.join(det)} |")
    lv = (_sc(obs.get("head") or {}, "control").get("levers")) or {}
    rows += ["", "Head environment: " + ", ".join(f"{k}={v}" for k, v in lv.items())]
    return "\n".join(rows)


def head_is_worse(base: dict, head: dict) -> tuple[bool, str]:
    bad = []
    bc, hc = _sc(base, "control").get("losses") or [], _sc(head, "control").get("losses") or []
    if len(bc) != len(hc):
        bad.append(f"control step count {len(bc)} vs {len(hc)}")
    else:
        for (lb, _), (lh, _) in zip(bc, hc):
            if not math.isfinite(lh) or abs(lh - lb) > 1e-3 + 0.05 * abs(lb):
                bad.append(f"control loss {lb:.4f} -> {lh:.4f}")
                break
    for n in DECISION:
        rec = _sc(head, n)
        if rec.get("ok"):
            continue
        if n == "clef_4bit" and _refused_clearly(rec):
            continue
        bad.append(f"{n}: {(rec.get('error') or 'not ok')[:300]}")
    if bad:
        return True, "; ".join(bad)
    return False, "control SFT unchanged within 1e-3 + 5%, every head decision scenario trained (or 4-bit refused clearly)"
